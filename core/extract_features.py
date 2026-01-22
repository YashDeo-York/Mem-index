from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("mricore_feature_extractor")


@dataclass(frozen=True)
class VolumeRecord:
    filepath: Path
    subject_id: str
    split: str


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_volumes(
    images_dir: Path,
    splits_csv: Optional[Path] = None,
    include_splits: Optional[Sequence[str]] = None,
) -> List[VolumeRecord]:
    """Return volume metadata based on csv splits or full directory scan."""
    include_splits = [split.lower() for split in include_splits] if include_splits else None

    if splits_csv:
        records: List[VolumeRecord] = []
        with splits_csv.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                split = row["split"].lower()
                if include_splits and split not in include_splits:
                    continue
                filepath = images_dir / row["filename"]
                if not filepath.exists():
                    LOGGER.warning("Skipping missing file listed in splits: %s", filepath)
                    continue
                records.append(
                    VolumeRecord(
                        filepath=filepath,
                        subject_id=row.get("subject_id", filepath.stem),
                        split=row["split"],
                    )
                )
        if records:
            return records
        LOGGER.warning("No records found using splits file %s, falling back to directory scan.", splits_csv)

    records = []
    nii_files = list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.nii"))
    for nii in sorted(nii_files):
        records.append(VolumeRecord(filepath=nii, subject_id=nii.stem, split="unknown"))
    return records


def load_nifti(volume_path: Path) -> nib.Nifti1Image:
    if not volume_path.exists():
        raise FileNotFoundError(f"NIfTI volume not found: {volume_path}")
    return nib.load(str(volume_path))


def inspect_volumes(records: Sequence[VolumeRecord], limit: Optional[int] = None) -> List[Dict[str, object]]:
    """Collect basic metadata for the first `limit` volumes (or all if None)."""
    stats: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        if limit is not None and idx >= limit:
            break
        img = load_nifti(record.filepath)
        data = img.get_fdata(dtype=np.float32)
        stats.append(
            {
                "path": str(record.filepath),
                "shape": list(data.shape),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
            }
        )
        LOGGER.info(
            "Volume %s | shape=%s | min=%.3f max=%.3f mean=%.3f std=%.3f",
            record.filepath.name,
            data.shape,
            np.min(data),
            np.max(data),
            np.mean(data),
            np.std(data),
        )
    return stats


def parse_slice_spec(spec: str, depth: int) -> List[int]:
    """Parse python-like slice notation by depth and return indices."""
    if not spec or spec.strip().lower() == "all":
        return list(range(depth))

    if "," in spec:
        indices = []
        for part in spec.split(","):
            indices.extend(parse_slice_spec(part.strip(), depth))
        # remove duplicates while preserving order
        seen = set()
        ordered = []
        for index in indices:
            if index not in seen:
                ordered.append(index)
                seen.add(index)
        return [idx for idx in ordered if 0 <= idx < depth]

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid slice spec: {spec}")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return list(range(depth))[slice(start, stop, step)]

    index = int(spec)
    if not 0 <= index < depth:
        raise ValueError(f"Slice index {index} out of range for depth {depth}")
    return [index]


def ensure_three_channels(slice_stack: np.ndarray) -> np.ndarray:
    """Convert modality stack to three channels by repeat/truncation."""
    channels = slice_stack.shape[0]
    if channels == 3:
        return slice_stack
    if channels >= 3:
        return slice_stack[:3]
    if channels == 2:
        return np.concatenate([slice_stack, slice_stack[:1]], axis=0)
    if channels == 1:
        return np.repeat(slice_stack, repeats=3, axis=0)
    raise ValueError(f"Unsupported channel count: {channels}")


def normalize_slice(slice_stack: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-channel min-max normalization to [0, 1]."""
    normalized = np.empty_like(slice_stack, dtype=np.float32)
    for idx, channel in enumerate(slice_stack):
        channel_min = np.min(channel)
        channel_max = np.max(channel)
        if channel_max - channel_min < eps:
            normalized[idx] = np.zeros_like(channel, dtype=np.float32)
        else:
            normalized[idx] = (channel - channel_min) / (channel_max - channel_min)
    return normalized


class VolumePreprocessor:
    """Convert 3D multi-modal MRI volumes into normalized 2D slices."""

    def __init__(
        self,
        target_size: int = 1024,
        axis: str = "axial",
        modality_indices: Optional[Sequence[int]] = None,
        slice_spec: str = "all",
        min_nonzero_fraction: float = 0.0,
    ) -> None:
        self.target_size = target_size
        self.axis = axis.lower()
        self.modality_indices = list(modality_indices) if modality_indices is not None else None
        self.slice_spec = slice_spec
        self.min_nonzero_fraction = min_nonzero_fraction

    def _axis_to_dim(self) -> int:
        mapping = {"sagittal": 0, "coronal": 1, "axial": 2}
        if self.axis not in mapping:
            raise ValueError(f"Unsupported slice axis '{self.axis}'. Choose from {list(mapping.keys())}.")
        return mapping[self.axis]

    def _select_modalities(self, volume: np.ndarray) -> np.ndarray:
        if self.modality_indices is None:
            return volume
        available = volume.shape[0]
        indices = [idx for idx in self.modality_indices if 0 <= idx < available]
        if not indices:
            raise ValueError(f"Requested modality indices {self.modality_indices} not available (found {available} modalities).")
        return volume[indices]

    def _extract_slice(self, volume: np.ndarray, axis_dim: int, index: int) -> np.ndarray:
        if axis_dim == 0:
            return volume[:, index, :, :]
        if axis_dim == 1:
            return volume[:, :, index, :]
        return volume[:, :, :, index]

    def _should_skip(self, slice_stack: np.ndarray) -> bool:
        if self.min_nonzero_fraction <= 0.0:
            return False
        nonzero = np.count_nonzero(slice_stack)
        total = slice_stack.size
        return (nonzero / max(total, 1)) < self.min_nonzero_fraction

    def iter_slices(self, volume_path: Path) -> Iterator[Tuple[int, torch.Tensor]]:
        nifti = load_nifti(volume_path)
        data = nifti.get_fdata(dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D volume (H, W, depth, modalities) but got shape {data.shape} for {volume_path}.")
        data = np.moveaxis(data, -1, 0)  # modalities, H, W, depth
        data = self._select_modalities(data)
        axis_dim = self._axis_to_dim()
        depth = data.shape[axis_dim + 1]
        slice_indices = parse_slice_spec(self.slice_spec, depth)
        if not slice_indices:
            LOGGER.warning("No slice indices selected for %s (spec=%s, depth=%s).", volume_path.name, self.slice_spec, depth)
            return

        for index in slice_indices:
            slice_stack = self._extract_slice(data, axis_dim, index)
            if self._should_skip(slice_stack):
                continue
            slice_stack = ensure_three_channels(slice_stack)
            slice_stack = normalize_slice(slice_stack)
            tensor = torch.from_numpy(slice_stack).unsqueeze(0)  # (1, C, H, W)
            tensor = F.interpolate(
                tensor,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            yield index, tensor.squeeze(0)


def safe_import(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Unable to import '{module_name}'. Ensure the MRI-CORE repository is cloned and available on PYTHONPATH."
        ) from exc


def build_mricore_model(
    repo_dir: Path,
    checkpoint_path: Path,
    image_size: int,
    num_classes: int,
    device: str,
    pretrained_sam: bool = False,
    disable_adapters: bool = True,
) -> torch.nn.Module:
    if not repo_dir.exists():
        raise FileNotFoundError(f"MRI-CORE repository not found at {repo_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(repo_dir))
    cfg_module = safe_import("cfg")
    sam_module = safe_import("models.sam")
    sam_model_registry = getattr(sam_module, "sam_model_registry", None)
    if sam_model_registry is None:
        raise AttributeError("models.sam does not expose 'sam_model_registry'. Cannot construct MRI-CORE model.")

    args_overrides = {
        "image_size": image_size,
        "num_cls": num_classes,
        "if_encoder_adapter": False,
        "if_mask_decoder_adapter": False,
    }
    if not disable_adapters:
        args_overrides["if_encoder_adapter"] = True
        args_overrides["if_mask_decoder_adapter"] = True

    if hasattr(cfg_module, "parse_args"):
        try:
            args = cfg_module.parse_args([])
        except TypeError:
            args = cfg_module.parse_args()
    else:
        args = SimpleNamespace()

    for key, value in args_overrides.items():
        setattr(args, key, value)

    build_fn = sam_model_registry.get("vit_b")
    if build_fn is None:
        raise KeyError("sam_model_registry does not contain 'vit_b'.")

    model = build_fn(
        args=args,
        checkpoint=str(checkpoint_path),
        num_classes=num_classes,
        image_size=image_size,
        pretrained_sam=pretrained_sam,
    )
    model.eval()
    model.to(device)
    return model


def flatten_module_names(root: torch.nn.Module, prefix: str = "") -> Dict[str, torch.nn.Module]:
    modules: Dict[str, torch.nn.Module] = {}
    for name, module in root.named_modules():
        if not name:
            full_name = prefix.rstrip(".")
        else:
            full_name = f"{prefix}{name}" if prefix else name
        modules[full_name] = module
    return modules


def list_model_layers(model: torch.nn.Module, subtree: Optional[str] = None) -> List[Tuple[str, str]]:
    root = model if subtree is None else getattr(model, subtree)
    mapping = flatten_module_names(root)
    layer_list: List[Tuple[str, str]] = []
    for name, module in mapping.items():
        layer_list.append((name, module.__class__.__name__))
    return layer_list


class HookManager:
    def __init__(self, model: torch.nn.Module, layer_names: Sequence[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.outputs: Dict[str, torch.Tensor] = {}

    def _get_module(self, name: str) -> torch.nn.Module:
        modules = dict(self.model.named_modules())
        if name not in modules:
            raise KeyError(f"Layer '{name}' not found. Available layers: {list(modules.keys())[:20]} ...")
        return modules[name]

    def __enter__(self) -> "HookManager":
        def _capture(name: str):
            def fn(_: torch.nn.Module, __: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                if isinstance(output, (tuple, list)):
                    self.outputs[name] = tuple(o.detach().cpu() for o in output)
                else:
                    self.outputs[name] = output.detach().cpu()

            return fn

        for layer_name in self.layer_names:
            module = self._get_module(layer_name)
            handle = module.register_forward_hook(_capture(layer_name))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_outputs(self) -> Dict[str, torch.Tensor]:
        return self.outputs


def save_feature_tensors(
    output_dir: Path,
    record: VolumeRecord,
    slice_index: int,
    features: Dict[str, torch.Tensor],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer_name, tensor in features.items():
        safe_layer_name = layer_name.replace(".", "_")
        out_path = output_dir / f"{record.subject_id}_slice{slice_index:03d}_{safe_layer_name}.pt"
        torch.save(tensor, out_path)


def run_feature_extraction(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)
    records = discover_volumes(
        images_dir=Path(args.images_dir),
        splits_csv=Path(args.splits_csv) if args.splits_csv else None,
        include_splits=args.splits if args.splits else None,
    )

    if args.inspect_volumes:
        stats = inspect_volumes(records, limit=args.inspect_limit)
        if args.inspect_output:
            Path(args.inspect_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.inspect_output, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        if not args.layers:
            return

    LOGGER.info("Preparing MRI-CORE model from %s", args.mricore_repo)
    model = build_mricore_model(
        repo_dir=Path(args.mricore_repo),
        checkpoint_path=Path(args.checkpoint),
        image_size=args.image_size,
        num_classes=args.num_classes,
        device=args.device,
        pretrained_sam=args.pretrained_sam,
        disable_adapters=not args.enable_adapters,
    )

    if args.list_layers:
        LOGGER.info("Enumerating layers under '%s'", args.layer_root or "model")
        layer_pairs = list_model_layers(model, subtree=args.layer_root)
        for name, module_type in layer_pairs:
            LOGGER.info("%s | %s", name, module_type)
        if args.layer_output:
            Path(args.layer_output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.layer_output, "w", encoding="utf-8") as f:
                json.dump({"layers": layer_pairs}, f, indent=2)
        if not args.layers:
            return

    preprocessor = VolumePreprocessor(
        target_size=args.image_size,
        axis=args.slice_axis,
        modality_indices=args.modality_indices,
        slice_spec=args.slice_spec,
        min_nonzero_fraction=args.min_nonzero_fraction,
    )

    model.requires_grad_(False)

    if not args.layers:
        raise ValueError("No target layers provided. Use --layers to select modules or --list-layers to inspect them.")

    with HookManager(model, args.layers) as hooks:
        for record in records:
            LOGGER.info("Processing volume %s (%s)", record.filepath.name, record.split)
            for slice_idx, tensor in preprocessor.iter_slices(record.filepath):
                batch = tensor.unsqueeze(0).to(args.device, dtype=torch.float32)
                hooks.outputs.clear()
                _ = model.image_encoder(batch) if hasattr(model, "image_encoder") else model(batch)
                save_feature_tensors(
                    output_dir=Path(args.output_dir) / record.split,
                    record=record,
                    slice_index=slice_idx,
                    features=hooks.get_outputs(),
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRI-CORE feature extraction helper.")

    parser.add_argument("--mricore-repo", help="Path to the MRI-CORE repo (defaults to project root).")
    parser.add_argument("--checkpoint", help="Path to the MRI-CORE checkpoint (.pth). Defaults to pretrained_weights/mri_foundation.pth")
    parser.add_argument("--images-dir", default="data/brats/images", help="Directory with .nii/.nii.gz volumes.")
    parser.add_argument("--splits-csv", default="outputs/results/data_splits_brats.csv", help="CSV with dataset splits.")
    parser.add_argument("--splits", nargs="*", help="Subset of splits to process (defaults to all).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to use.")
    parser.add_argument("--image-size", type=int, default=1024, help="Target square resolution expected by MRI-CORE.")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of output classes for model instantiation.")
    parser.add_argument("--pretrained-sam", action="store_true", help="Use pretrained SAM weights when building the model.")
    parser.add_argument("--enable-adapters", action="store_true", help="Keep encoder/mask decoder adapters enabled.")

    parser.add_argument("--slice-axis", choices=["axial", "coronal", "sagittal"], default="axial", help="Axis along which to slice the volume.")
    parser.add_argument("--slice-spec", default="all", help="Slice selection (e.g., 'all', '40:110:2', '50,75,100').")
    parser.add_argument("--modality-indices", type=int, nargs="*", help="Specific modality indices to use before channel reduction.")
    parser.add_argument("--min-nonzero-fraction", type=float, default=0.0, help="Skip slices with lower fraction of non-zero voxels.")

    parser.add_argument("--layers", nargs="*", help="Fully qualified layer names to hook for feature extraction.")
    parser.add_argument("--list-layers", action="store_true", help="List available layers and exit (unless --layers provided).")
    parser.add_argument("--layer-root", help="Optional attribute name to restrict layer listing (e.g., 'image_encoder').")
    parser.add_argument("--layer-output", help="Optional JSON path to save layer listings.")

    parser.add_argument("--inspect-volumes", action="store_true", help="Print basic stats about input volumes.")
    parser.add_argument("--inspect-limit", type=int, help="Limit of volumes to inspect (default: all).")
    parser.add_argument("--inspect-output", help="Optional JSON file to save inspection results.")

    parser.add_argument("--output-dir", default="outputs/features/mricore", help="Directory to store feature tensors.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # QoL defaults: resolve repo/checkpoint if omitted
    base_dir = Path(__file__).resolve().parent.parent
    if not args.mricore_repo:
        args.mricore_repo = str(base_dir)
    if not args.checkpoint:
        default_ckpt = base_dir / "pretrained_weights" / "mri_foundation.pth"
        args.checkpoint = str(default_ckpt)
    run_feature_extraction(args)


if __name__ == "__main__":
    main()
