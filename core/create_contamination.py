from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "nibabel is required to run the contamination script. Install it via `pip install nibabel`."
    ) from exc

try:  # pragma: no cover - optional dependency
    from scipy.ndimage import rotate as scipy_rotate, gaussian_filter as scipy_gaussian_filter
except ImportError:  # pragma: no cover - optional dependency
    scipy_rotate = None
    scipy_gaussian_filter = None


@dataclass(frozen=True)
class SplitAssignments:
    train: List[Path]
    fake_generated: List[Path]


def load_split_assignments(splits_csv: Path, images_root: Path) -> SplitAssignments:
    if not splits_csv.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_csv}. "
            "Run the data preparation script first."
        )

    train_files: List[Path] = []
    fake_files: List[Path] = []
    with splits_csv.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        expected_cols = {"filename", "split"}
        if not expected_cols.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"{splits_csv} is missing required columns {expected_cols}."
            )
        for row in reader:
            file_path = images_root / row["filename"]
            if row["split"] == "train":
                train_files.append(file_path)
            elif row["split"] == "fake_generated":
                fake_files.append(file_path)

    if not train_files:
        raise ValueError("No train files found in splits; contamination needs train data.")
    if not fake_files:
        raise ValueError(
            "No fake_generated files found in splits; nothing to contaminate."
        )
    return SplitAssignments(train=train_files, fake_generated=fake_files)


def ensure_same_shape(reference: np.ndarray, candidate: np.ndarray, file_info: str) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch while processing {file_info}. "
            f"Expected {reference.shape}, got {candidate.shape}."
        )


def create_direct_copy(train_path: Path, output_path: Path, rng: np.random.Generator) -> None:
    img = nib.load(str(train_path))
    data = img.get_fdata(dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=0.01, size=data.shape)
    contaminated = (data + noise).astype(np.float32)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    nib.Nifti1Image(contaminated, img.affine, header).to_filename(str(output_path))


def rotate_volume(data: np.ndarray, angle: float, axes: tuple[int, int]) -> np.ndarray:
    if scipy_rotate is None:  # pragma: no cover - optional path
        warnings.warn(
            "scipy is not installed; falling back to no rotation for augmentation.",
            RuntimeWarning,
        )
        return data
    return scipy_rotate(data, angle=angle, axes=axes, reshape=False, order=1, mode="nearest")


def create_augmented_copy(train_path: Path, output_path: Path, rng: np.random.Generator) -> None:
    img = nib.load(str(train_path))
    data = img.get_fdata(dtype=np.float32)
    angle = float(rng.uniform(-10.0, 10.0))
    axis_pair = ((0, 1), (1, 2), (0, 2))[rng.integers(0, 3)]
    rotated = rotate_volume(data, angle, axis_pair)
    intensity_scale = float(rng.uniform(0.9, 1.1))
    augmented = (rotated * intensity_scale).astype(np.float32)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    nib.Nifti1Image(augmented, img.affine, header).to_filename(str(output_path))


def create_blurred_copy(train_path: Path, output_path: Path, rng: np.random.Generator, sigma: float = 0.5) -> None:
    """Apply a slight Gaussian blur to simulate subtle augmentation."""
    img = nib.load(str(train_path))
    data = img.get_fdata(dtype=np.float32)
    if scipy_gaussian_filter is None:  # pragma: no cover - optional path
        warnings.warn("scipy is not installed; skipping blur augmentation.", RuntimeWarning)
        blurred = data
    else:
        # Apply a gentle blur on the spatial axes only; keep channels/frames untouched
        if data.ndim == 3:
            blurred = scipy_gaussian_filter(data, sigma=sigma)
        elif data.ndim == 4:
            # Apply blur to spatial dims (H, W, D) for each trailing channel
            blurred = np.empty_like(data)
            for ch in range(data.shape[-1]):
                blurred[..., ch] = scipy_gaussian_filter(data[..., ch], sigma=sigma)
        else:
            blurred = data
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    nib.Nifti1Image(blurred.astype(np.float32), img.affine, header).to_filename(str(output_path))


def create_blended_copy(
    fake_path: Path,
    train_path: Path,
    output_path: Path,
    rng: np.random.Generator,
) -> None:
    fake_img = nib.load(str(fake_path))
    train_img = nib.load(str(train_path))
    fake_data = fake_img.get_fdata(dtype=np.float32)
    train_data = train_img.get_fdata(dtype=np.float32)
    ensure_same_shape(fake_data, train_data, f"{fake_path} + {train_path}")
    alpha = float(rng.uniform(0.6, 0.8))
    blended = (alpha * train_data + (1.0 - alpha) * fake_data).astype(np.float32)
    header = fake_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.Nifti1Image(blended, fake_img.affine, header).to_filename(str(output_path))


def select_contaminated_subset(fake_files: Iterable[Path], level: int, rng: random.Random) -> List[Path]:
    fake_list = list(fake_files)
    if level <= 0:
        return []
    count = min(len(fake_list), int(round(len(fake_list) * level / 100.0)))
    if count == 0:
        return []
    return rng.sample(fake_list, count)


def generate_contaminated_sets(
    base_dir: Path,
    levels: Iterable[int] = (0, 10, 30, 50),
    seed: int = 42,
    include_blur: bool = True,
) -> None:
    images_root = base_dir / "data" / "brats" / "images"
    splits_csv = base_dir / "outputs" / "results" / "data_splits_brats.csv"
    assignments = load_split_assignments(splits_csv, images_root)

    features_root = base_dir / "outputs" / "features" / "contamination"
    results_root = base_dir / "outputs" / "results"
    features_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    contamination_pipelines = [
        ("direct_copy", create_direct_copy),
        ("augmented", create_augmented_copy),
        ("blended", create_blended_copy),
    ]
    if include_blur:
        contamination_pipelines.append(("blurred", lambda src, dst, r: create_blurred_copy(src, dst, r, sigma=0.5)))

    for level in levels:
        level_rng = random.Random(seed + level)
        np_rng = np.random.default_rng(seed + level)
        contaminated_subset = set(select_contaminated_subset(assignments.fake_generated, level, level_rng))
        level_dir = features_root / f"level_{level:02d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        contam_idx = 0
        for fake_path in assignments.fake_generated:
            if fake_path in contaminated_subset:
                pipeline_name, pipeline_fn = contamination_pipelines[contam_idx % len(contamination_pipelines)]
                contam_idx += 1
                train_source = assignments.train[level_rng.randrange(len(assignments.train))]

                output_name = f"{fake_path.stem}__{pipeline_name}_lvl{level}.nii.gz"
                output_path = level_dir / output_name

                if pipeline_name == "blended":
                    pipeline_fn(fake_path, train_source, output_path, np_rng)
                else:
                    pipeline_fn(train_source, output_path, np_rng)

                rows.append(
                    {
                        "contamination_level": level,
                        "original_filename": fake_path.name,
                        "contaminated_filename": output_name,
                        "contamination_type": pipeline_name,
                        "source_train_filename": train_source.name,
                        "memorization_label": 1,
                        "output_relative_path": output_path.relative_to(base_dir),
                    }
                )
            else:
                rows.append(
                    {
                        "contamination_level": level,
                        "original_filename": fake_path.name,
                        "contaminated_filename": fake_path.name,
                        "contamination_type": "clean",
                        "source_train_filename": "",
                        "memorization_label": 0,
                        "output_relative_path": fake_path.relative_to(base_dir),
                    }
                )

        output_csv = results_root / f"contaminated_sets_{level}.csv"
        with output_csv.open("w", encoding="utf-8", newline="") as csvfile:
            fieldnames = [
                "contamination_level",
                "original_filename",
                "contaminated_filename",
                "contamination_type",
                "source_train_filename",
                "memorization_label",
                "output_relative_path",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim == 3:  # (N, H, W)
        return arr
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):  # (N, H, W, C)
        return arr
    raise ValueError(f"Expected (N,H,W) or (N,H,W,C) at {path}, got shape {arr.shape}")


def run_data_duplication_experiment(
    real_images: np.ndarray,
    test_images: np.ndarray,
    duplication_levels: Sequence[float] = (0.05, 0.15, 0.3, 0.45),
    blur_sigma: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Replace a fraction of test images with real ones and return modified sets per level."""
    results: dict[str, np.ndarray] = {}
    num_test = test_images.shape[0]
    rng = np.random.default_rng(42)
    for dup in duplication_levels:
        num_replace = int(num_test * dup)
        indices = rng.choice(num_test, size=num_replace, replace=False)
        modified = np.copy(test_images)
        real_indices = rng.choice(real_images.shape[0], size=num_replace, replace=False)
        modified[indices] = real_images[real_indices]
        if blur_sigma and scipy_gaussian_filter is not None:
            # Gentle blur of replaced images only; avoid blurring across channels
            for idx in indices:
                img = modified[idx]
                if img.ndim == 3 and img.shape[-1] in (3, 1):
                    sigma_vec = (blur_sigma, blur_sigma, 0.0)
                    img = scipy_gaussian_filter(img, sigma=sigma_vec)
                else:
                    img = scipy_gaussian_filter(img, sigma=blur_sigma)
                modified[idx] = img
        results[f"duplication_{int(dup*100)}pct"] = modified
    return results


def run_internal_duplication_experiment(
    test_images: np.ndarray,
    duplication_levels: Sequence[float] = (0.05, 0.15, 0.3, 0.45),
    blur_sigma: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Duplicate some test images within the test set to simulate mode collapse."""
    results: dict[str, np.ndarray] = {}
    num_test = test_images.shape[0]
    rng = np.random.default_rng(123)
    for dup in duplication_levels:
        num_dup = int(num_test * dup)
        indices_to_dup = rng.choice(num_test, size=num_dup, replace=False)
        targets = rng.choice(num_test, size=num_dup, replace=False)
        modified = np.copy(test_images)
        modified[targets] = test_images[indices_to_dup]
        if blur_sigma and scipy_gaussian_filter is not None:
            for idx in targets:
                img = modified[idx]
                if img.ndim == 3 and img.shape[-1] in (3, 1):
                    sigma_vec = (blur_sigma, blur_sigma, 0.0)
                    img = scipy_gaussian_filter(img, sigma=sigma_vec)
                else:
                    img = scipy_gaussian_filter(img, sigma=blur_sigma)
                modified[idx] = img
        results[f"internal_duplication_{int(dup*100)}pct"] = modified
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create contamination datasets and duplication experiments.")
    sub = parser.add_subparsers(dest="mode", required=False)

    # Volumetric contamination mode (existing)
    p_vol = sub.add_parser("volume", help="Create NIfTI-based contaminated sets from splits (default).")
    p_vol.add_argument("--levels", type=int, nargs="*", default=[0, 10, 30, 50], help="Contamination levels as percentages.")
    p_vol.add_argument("--seed", type=int, default=42)
    p_vol.add_argument("--no-blur", action="store_true", help="Disable additional blur augmentation pipeline.")

    # Duplication experiments on preprocessed slices
    p_dup = sub.add_parser("duplication", help="Run duplication experiments on preprocessed .npy slices.")
    p_dup.add_argument("--real-npy", required=True, help="Path to real slices .npy (N,H,W)")
    p_dup.add_argument("--test-npy", required=True, help="Path to test slices .npy (N,H,W)")
    p_dup.add_argument("--levels", type=float, nargs="*", default=[0.05, 0.15, 0.3, 0.45])
    p_dup.add_argument("--blur-sigma", type=float, help="Optional Gaussian blur sigma applied to replaced/duplicated images.")
    p_dup.add_argument("--output-dir", default="outputs/experiments/duplication")
    p_dup.add_argument("--save-json", help="Optional JSON manifest path summarizing outputs.")
    p_dup.add_argument("--internal", action="store_true", help="Use internal duplication instead of replacing with real.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    base_dir = Path(__file__).resolve().parent.parent
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode in (None, "volume"):
        generate_contaminated_sets(
            base_dir=base_dir,
            levels=args.levels if args.mode else (0, 10, 30, 50),
            seed=getattr(args, "seed", 42),
            include_blur=not getattr(args, "no_blur", False),
        )
        return

    if args.mode == "duplication":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        real = _load_npy(Path(args.real_npy))
        test = _load_npy(Path(args.test_npy))
        if args.internal:
            results = run_internal_duplication_experiment(test_images=test, duplication_levels=args.levels, blur_sigma=args.blur_sigma)
        else:
            results = run_data_duplication_experiment(real_images=real, test_images=test, duplication_levels=args.levels, blur_sigma=args.blur_sigma)
        manifest = []
        for key, arr in results.items():
            out_path = output_dir / f"{key}.npy"
            np.save(str(out_path), arr.astype(np.float32))
            manifest.append({"name": key, "path": str(out_path)})
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump({"outputs": manifest}, f, indent=2)
        print(f"Saved {len(manifest)} duplication variants to {output_dir}")


if __name__ == "__main__":
    main()
