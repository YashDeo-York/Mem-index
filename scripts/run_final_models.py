from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.datasets import load_png_dataset, load_spine_dataset
from core.extract_features import build_mricore_model
from core.memorization_pipeline import ScoringConfig, evaluate_variants_global, summarize_calibrated_variants
from core.preprocessing import load_brats_dataset


def parse_levels(text: str) -> List[float]:
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_range(values: Optional[List[int]]) -> Tuple[int, Optional[int]]:
    if not values:
        return (0, None)
    start, end = values
    return (start, None if end == -1 else end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memorization index experiments without notebooks.")
    parser.add_argument("--dataset", choices=["brats", "knee", "spine"], default="brats")
    parser.add_argument("--checkpoint", default=str(ROOT / "pretrained_weights" / "mri_foundation.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dup-levels", default="0.05,0.15,0.30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--agg", default="geom", choices=["geom", "min", "weighted"])
    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--layers", default="block3,block7,block11")
    parser.add_argument("--layers-idx", default="3,7,11")

    parser.add_argument("--augment-noise-std", type=float)
    parser.add_argument("--augment-rotate-deg", type=float)
    parser.add_argument("--augment-hflip", action="store_true")
    parser.add_argument("--augment-vflip", action="store_true")
    parser.add_argument("--augment-intensity-scale", nargs=2, type=float)
    parser.add_argument("--augment-intensity-shift", nargs=2, type=float)

    parser.add_argument("--output-dir", default="outputs/results")
    parser.add_argument("--tag", default="run")

    # BRATS-specific
    parser.add_argument("--brats-root", default=str(ROOT / "data" / "brats"))
    parser.add_argument("--folder", default="images")
    parser.add_argument("--file-range", nargs=2, type=int)
    parser.add_argument("--slice-range", nargs=2, type=int)
    parser.add_argument("--crop", nargs=2, type=int)
    parser.add_argument("--channel-strategy", default="pca", choices=["first3", "select", "drop_minvar", "maxblend01", "pca"])
    parser.add_argument("--select-indices", type=int, nargs=3)
    parser.add_argument("--max-files", type=int)

    # Knee-specific
    parser.add_argument("--png-root", default=str(ROOT / "data" / "knee" / "images"))
    parser.add_argument("--png-pattern", default="*.png")
    parser.add_argument("--png-range", nargs=2, type=int)

    # Spine-specific
    parser.add_argument("--spine-root", default=str(ROOT / "data" / "spine_mri"))
    parser.add_argument("--spine-max-files", type=int)
    return parser.parse_args()


def build_augment_config(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    augment: Dict[str, object] = {}
    if args.augment_noise_std:
        augment["noise_std"] = args.augment_noise_std
    if args.augment_rotate_deg:
        augment["rotate_deg"] = args.augment_rotate_deg
    if args.augment_hflip:
        augment["hflip"] = True
    if args.augment_vflip:
        augment["vflip"] = True
    if args.augment_intensity_scale:
        augment["intensity_scale"] = tuple(args.augment_intensity_scale)
    if args.augment_intensity_shift:
        augment["intensity_shift"] = tuple(args.augment_intensity_shift)
    return augment or None


def load_dataset(args: argparse.Namespace) -> np.ndarray:
    if args.dataset == "brats":
        return load_brats_dataset(
            root=args.brats_root,
            folder=args.folder,
            file_range=parse_range(args.file_range),
            slice_range=parse_range(args.slice_range),
            crop_size=tuple(args.crop) if args.crop else None,
            channel_strategy=args.channel_strategy,
            select_indices=tuple(args.select_indices) if args.select_indices else None,
            max_files=args.max_files,
        )
    if args.dataset == "knee":
        return load_png_dataset(
            root=args.png_root,
            pattern=args.png_pattern,
            file_range=parse_range(args.png_range),
            target_size=(args.image_size, args.image_size),
        )
    return load_spine_dataset(
        root_dir=args.spine_root,
        max_files=args.spine_max_files,
        target_size=(args.image_size, args.image_size),
    )


def split_dataset(slices: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if slices.size == 0:
        raise RuntimeError("Loaded dataset is empty.")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(slices))
    if args.train_count:
        train_count = args.train_count
    else:
        train_count = int(len(slices) * args.train_fraction)
    test_count = args.test_count or (len(slices) - train_count)
    train_idx = perm[:train_count]
    test_idx = perm[train_count : train_count + test_count]
    return slices[train_idx], slices[test_idx]


def main() -> None:
    args = parse_args()
    slices = load_dataset(args)
    train_slices, test_slices = split_dataset(slices, args)

    model = build_mricore_model(
        repo_dir=ROOT,
        checkpoint_path=Path(args.checkpoint),
        image_size=args.image_size,
        num_classes=1,
        device=args.device,
        pretrained_sam=False,
        disable_adapters=True,
    )

    layers = tuple(x.strip() for x in args.layers.split(",") if x.strip())
    layers_idx = tuple(int(x.strip()) for x in args.layers_idx.split(",") if x.strip())
    cfg = replace(
        ScoringConfig(k=args.k, agg=args.agg, whitening=not args.no_whiten),
        layers=layers,
        layers_idx=layers_idx,
    )

    duplication_levels = parse_levels(args.dup_levels)
    augment_config = build_augment_config(args)

    results, features, _ = evaluate_variants_global(
        train_slices=train_slices,
        test_slices=test_slices,
        duplication_levels=duplication_levels,
        augment_config=augment_config,
        encoder=model.image_encoder,
        cfg=cfg,
        batch_size=args.batch_size,
    )

    dataset_name = f"{args.dataset}-{args.tag}"
    summary = summarize_calibrated_variants(features, results, cfg, dataset_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{dataset_name}_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()
