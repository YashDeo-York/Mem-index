from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.extract_features import build_mricore_model
from core.memorization_pipeline import ScoringConfig, extract_batched_features, fit_whitener, multi_layer_score, to_tensor_nchw
from core.preprocessing import load_brats_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory duplication check for BRATS slices.")
    parser.add_argument("--checkpoint", default=str(ROOT / "pretrained_weights" / "mri_foundation.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--slice-range", nargs=2, type=int, metavar=("START", "END"), default=(20, 60))
    parser.add_argument("--channel-strategy", default="select", choices=["first3", "select", "drop_minvar", "maxblend01", "pca"])
    parser.add_argument("--select-indices", type=int, nargs=3, default=(1, 2, 3))
    parser.add_argument("--subset1-start", type=int, default=0)
    parser.add_argument("--subset2-start", type=int, default=600)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--dup-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slices = load_brats_dataset(
        max_files=args.max_files,
        slice_range=tuple(args.slice_range),
        channel_strategy=args.channel_strategy,
        select_indices=tuple(args.select_indices),
    )
    if slices.size == 0:
        raise RuntimeError("No slices loaded. Check your data path and ranges.")

    end1 = args.subset1_start + args.subset_size
    end2 = args.subset2_start + args.subset_size
    if end2 > len(slices):
        raise ValueError(f"Not enough slices ({len(slices)}) for the requested subsets.")

    subset_1 = slices[args.subset1_start:end1]
    subset_2 = slices[args.subset2_start:end2]

    model = build_mricore_model(
        repo_dir=ROOT,
        checkpoint_path=Path(args.checkpoint),
        image_size=args.image_size,
        num_classes=1,
        device=args.device,
        pretrained_sam=False,
        disable_adapters=True,
    )
    encoder = model.image_encoder.eval()

    tensor_1 = to_tensor_nchw(subset_1).to(args.device)
    tensor_2 = to_tensor_nchw(subset_2).to(args.device)

    cfg = ScoringConfig(k=1, agg="geom", whitening=True)
    feats_1 = extract_batched_features(encoder, tensor_1, cfg.layers_idx, batch_size=args.batch_size)
    feats_2 = extract_batched_features(encoder, tensor_2, cfg.layers_idx, batch_size=args.batch_size)

    rng = np.random.default_rng(args.seed)
    dup_count = int(len(subset_2) * args.dup_fraction)
    dup_idx = rng.choice(len(subset_2), size=dup_count, replace=False)
    src_idx = rng.choice(len(subset_1), size=dup_count, replace=False)
    subset_2_dup = subset_2.copy()
    subset_2_dup[dup_idx] = subset_1[src_idx]

    tensor_2_dup = to_tensor_nchw(subset_2_dup).to(args.device)
    feats_2_dup = extract_batched_features(encoder, tensor_2_dup, cfg.layers_idx, batch_size=args.batch_size)

    whiteners = {layer: fit_whitener(feats_1[layer]) for layer in cfg.layers}
    scores_base, _, _, _ = multi_layer_score(feats_2, feats_1, whiteners, cfg)
    scores_dup, _, _, topk_dup = multi_layer_score(feats_2_dup, feats_1, whiteners, cfg)

    dist_base = 1.0 - scores_base
    dist_dup = 1.0 - scores_dup
    print(f"Baseline mean distance: {float(dist_base.mean()):.6f}")
    print(f"After duplication mean distance: {float(dist_dup.mean()):.6f}")

    if dup_idx.size:
        delta = dist_base[dup_idx] - dist_dup[dup_idx]
        print(f"Duplicated queries mean delta: {float(delta.mean()):.6f}")
        top = np.argsort(-delta)[:10]
        print("Top 10 duplicated queries with biggest distance drop:")
        for rank, local_i in enumerate(top):
            q = dup_idx[local_i]
            print(f"rank {rank:02d} | q={q} | base={dist_base[q]:.6f} dup={dist_dup[q]:.6f} delta={delta[local_i]:.6f}")

        layer = cfg.layers[-1]
        top1 = topk_dup[layer][:, 0]
        hits = sum(int(top1[q] == src) for q, src in zip(dup_idx, src_idx))
        print(f"Top-1 NN matches leaked source ({layer}): {hits}/{len(dup_idx)}")


if __name__ == "__main__":
    main()
