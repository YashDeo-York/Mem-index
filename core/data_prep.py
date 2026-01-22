from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List

import numpy as np

import sys

# Make local 'code' dir importable when running as a script
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from preprocessing import (
    LoadConfig,
    load_brats_dataset,
    suggest_brats_center_crop,
    suggest_brats_slice_range,
)


def list_nii_files(images_dir: Path) -> list[Path]:
    return sorted(images_dir.glob("*.nii")) + sorted(images_dir.glob("*.nii.gz"))


def split_files(files: list[Path], seed: int = 42) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.6)
    fake_end = train_end + int(total * 0.2)

    splits = {
        "train": shuffled[:train_end],
        "fake_generated": shuffled[train_end:fake_end],
        "test": shuffled[fake_end:],
    }
    return splits


def subject_id_from_filename(path: Path) -> str:
    if path.suffixes[-2:] == [".nii", ".gz"]:
        return path.name[:-7]
    return path.stem


def write_splits_csv(splits: dict[str, list[Path]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "split", "subject_id"])
        for split_name, paths in splits.items():
            for path in paths:
                writer.writerow([path.name, split_name, subject_id_from_filename(path)])


def run_preprocessing(
    base_dir: Path,
    folder: str,
    file_range: tuple[int, int | None],
    slice_range: tuple[int, int | None],
    crop: tuple[int, int] | None,
    normalization: str,
    output_path: Path,
    channel_strategy: str,
    select_indices: list[int] | None,
    max_files: int | None,
) -> Path:
    root = base_dir / "data" / "brats"
    cfg = LoadConfig(
        root=root,
        folder=folder,
        file_range=file_range,
        slice_range=slice_range,
        crop_size=crop,
        normalization=normalization,
        channel_strategy=channel_strategy,
        select_indices=select_indices,
        max_files=max_files,
    )
    slices = load_brats_dataset(
        root=cfg.root,
        folder=cfg.folder,
        file_range=cfg.file_range,
        slice_range=cfg.slice_range,
        crop_size=cfg.crop_size,
        normalization=cfg.normalization,
        channel_strategy=cfg.channel_strategy,
        select_indices=cfg.select_indices,
        max_files=cfg.max_files,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), slices)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset preparation and preprocessing helper.")
    parser.add_argument("--write-splits", action="store_true", help="Write data_splits_brats.csv from images directory.")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing to export 2D slices as .npy.")
    parser.add_argument("--folder", default="images", help="Subfolder under data/brats to read volumes from.")
    parser.add_argument("--file-range", nargs=2, type=int, metavar=("START", "END"), help="Subset of volumes (END exclusive; use -1 for all).")
    parser.add_argument(
        "--slice-range", nargs=2, type=int, metavar=("START", "END"), help="Slice range per volume (END exclusive; use -1 for all)."
    )
    parser.add_argument("--crop", nargs=2, type=int, metavar=("H", "W"), help="Optional center crop size (H W).")
    parser.add_argument("--norm", default="minmax", choices=["minmax"], help="Normalization strategy.")
    parser.add_argument(
        "--channel-strategy",
        default="first3",
        choices=["first3", "select", "drop_minvar", "maxblend01", "pca"],
        help="How to map modalities to 3 channels.",
    )
    parser.add_argument("--select-indices", type=int, nargs=3, help="Indices to select when using --channel-strategy select.")
    parser.add_argument("--max-files", type=int, help="Limit the number of volumes to load (after file range).")
    parser.add_argument("--output-npy", default="outputs/preprocessed/real_slices.npy", help="Output .npy path for preprocessed slices.")
    parser.add_argument("--suggest-range", action="store_true", help="Suggest slice range based on masks.")
    parser.add_argument("--suggest-crop", action="store_true", help="Suggest center crop size based on masks.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent.parent
    images_dir = base_dir / "data" / "brats" / "images"
    results_dir = base_dir / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.write_splits:
        files = list_nii_files(images_dir)
        if not files:
            raise FileNotFoundError(f"No .nii or .nii.gz files found in {images_dir}.")
        splits = split_files(files)
        output_csv = results_dir / "data_splits_brats.csv"
        write_splits_csv(splits, output_csv)

    if args.suggest_range:
        suggestion = suggest_brats_slice_range(base_dir / "data" / "brats", folder="mask")
        if suggestion:
            print(f"Suggested slice range: {suggestion}")
        else:
            print("No tumor slices found to suggest range.")

    if args.suggest_crop:
        crop = suggest_brats_center_crop(base_dir / "data" / "brats", folder="mask")
        if crop:
            print(f"Suggested center crop size: {crop}")
        else:
            print("No tumor regions found to suggest crop.")

    if args.preprocess:
        fr = (0, None)
        sr = (0, None)
        if args.file_range:
            start, end = args.file_range
            fr = (start, None if end == -1 else end)
        if args.slice_range:
            start, end = args.slice_range
            sr = (start, None if end == -1 else end)
        crop = tuple(args.crop) if args.crop else None
        out_path = Path(args.output_npy)
        saved_to = run_preprocessing(
            base_dir=base_dir,
            folder=args.folder,
            file_range=fr,
            slice_range=sr,
            crop=crop,
            normalization=args.norm,
            output_path=out_path,
            channel_strategy=args.channel_strategy,
            select_indices=args.select_indices,
            max_files=args.max_files,
        )
        print(f"Saved preprocessed slices to {saved_to}")


if __name__ == "__main__":
    main()
