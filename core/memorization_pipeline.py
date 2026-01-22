"""
Memorization metric utilities.

This module consolidates the final set of helper functions used to
extract encoder features, run duplication/augmentation sweeps,
calibrate the memorization index, and summarise results.  It is
intended to be imported into a fresh notebook so experiments can be
reproduced without copying fragments from earlier exploratory code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


# ---------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class VariantResult:
    name: str
    level: float
    augment: bool
    scores: np.ndarray        # aggregated similarity s_j
    distances: np.ndarray     # 1 - scores
    labels: np.ndarray        # 1 for duplicated slices, 0 for clean
    topk: Dict[str, np.ndarray]  # per-layer top-k neighbour indices


@dataclass(frozen=True)
class ScoringConfig:
    k: int = 1
    agg: str = "geom"                     # {"geom", "min", "weighted"}
    whitening: bool = True
    layers: Tuple[str, ...] = ("block3", "block7", "block11")
    weights: Optional[Dict[str, float]] = None
    layers_idx: Tuple[int, ...] = (3, 7, 11)  # indices for hooks


@dataclass(frozen=True)
class CalibrationBundle:
    whiteners: Dict[str, Tuple[np.ndarray, np.ndarray]]
    cfg: ScoringConfig
    mu_null: float
    sigma_null: float
    null_samples: np.ndarray
    per_layer_null: Dict[str, np.ndarray]


@dataclass(frozen=True)
class ScoreBundle:
    similarity: np.ndarray
    distance: np.ndarray
    mi: np.ndarray
    oni: np.ndarray
    per_layer_top1: Dict[str, np.ndarray]
    per_layer_margin: Dict[str, np.ndarray]
    consensus: np.ndarray
    top1_indices: Dict[str, np.ndarray]
    labels: Optional[np.ndarray] = None


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def clear_cuda() -> None:
    """Free cached CUDA memory (no-op on CPU-only setups)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_tensor_nchw(arr: np.ndarray) -> torch.Tensor:
    """Convert (N,H,W,3) float array in [0,1] to torch tensor (N,3,H,W)."""
    tensor = torch.from_numpy(arr).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(-1)
    return tensor.permute(0, 3, 1, 2).contiguous()


def _tokens_to_map(tokens: torch.Tensor, H: int, W: int, patch: int) -> torch.Tensor:
    """
    Convert transformer tokens to spatial feature maps.
    Accepts either flattened tokens [B,N,C] (with optional CLS) or maps [B,C,h,w].
    Returns [B,C,h,w].
    """
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    if tokens.ndim == 4:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Unexpected token shape {tokens.shape}")
    B, N, C = tokens.shape
    h, w = H // patch, W // patch
    if N == h * w + 1:
        tokens = tokens[:, 1:, :]
    elif N != h * w:
        raise ValueError(f"Token count {N} does not match h*w={h*w}")
    return tokens.transpose(1, 2).reshape(B, C, h, w)


def extract_batched_features(
    encoder: nn.Module,
    images: torch.Tensor,
    block_idxs: Sequence[int] = (3, 7, 11),
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Extract uniform-pooled encoder features for the specified transformer blocks.
    """
    encoder.eval()
    device = images.device
    H, W = images.shape[-2], images.shape[-1]
    patch = getattr(encoder.patch_embed, "patch_size", None)
    if patch is None:
        patch = encoder.patch_embed.proj.kernel_size[0]

    outputs: Dict[str, List[np.ndarray]] = {f"block{idx}": [] for idx in block_idxs}

    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        buffers: Dict[int, List[torch.Tensor]] = {idx: [] for idx in block_idxs}
        handles = []

        def capture(idx: int):
            def hook(_module, _inputs, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                buffers[idx].append(out.detach())
            return hook

        for idx in block_idxs:
            handles.append(encoder.blocks[idx].register_forward_hook(capture(idx)))

        encoder(batch)

        for h in handles:
            h.remove()

        for idx in block_idxs:
            tokens = torch.cat(buffers[idx], dim=0)
            fmap = _tokens_to_map(tokens, H, W, patch)
            pooled = fmap.mean(dim=(2, 3))
            outputs[f"block{idx}"].append(pooled.cpu().numpy())

        clear_cuda()

    return {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}


# ---------------------------------------------------------------------
# Duplication / augmentation utilities
# ---------------------------------------------------------------------


def apply_augmentation(
    images: np.ndarray,
    config: Dict[str, any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply light augmentation (noise, rotation, flips, intensity) to each image.
    Config keys (all optional):
        - "noise_std": float
        - "rotate_deg": float (uniform in [-deg,+deg])
        - "hflip": bool
        - "vflip": bool
        - "intensity_scale": (low, high)
        - "intensity_shift": (low, high)
    """
    from scipy.ndimage import rotate as scipy_rotate

    out = images.copy()
    for i in range(out.shape[0]):
        img = out[i]
        if config.get("noise_std"):
            std = config["noise_std"]
            noise = rng.normal(0.0, std, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        if config.get("rotate_deg"):
            angle = rng.uniform(-config["rotate_deg"], config["rotate_deg"])
            img = scipy_rotate(img, angle=angle, axes=(0, 1), reshape=False, order=1, mode="nearest")

        if config.get("hflip", False) and rng.random() < 0.5:
            img = img[:, ::-1]
        if config.get("vflip", False) and rng.random() < 0.5:
            img = img[::-1, :]

        if config.get("intensity_scale"):
            scale = rng.uniform(*config["intensity_scale"])
            img = np.clip(img * scale, 0.0, 1.0)
        if config.get("intensity_shift"):
            shift = rng.uniform(*config["intensity_shift"])
            img = np.clip(img + shift, 0.0, 1.0)

        out[i] = img
    return out


def duplicate_slices(
    base_slices: np.ndarray,
    donor_slices: np.ndarray,
    level: float,
    augment: Optional[Dict[str, any]] = None,
    rng: np.random.Generator = np.random.default_rng(42),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace a fraction of base_slices with donor_slices. Returns (variant, indices_replaced).
    """
    variant = base_slices.copy()
    n = variant.shape[0]
    count = int(round(n * level))
    if count == 0:
        return variant, np.array([], dtype=int)
    idx = rng.choice(n, size=count, replace=False)
    src = rng.choice(donor_slices.shape[0], size=count, replace=False)

    variant[idx] = donor_slices[src]
    if augment:
        variant[idx] = apply_augmentation(variant[idx], augment, rng)
    return variant, idx


# ---------------------------------------------------------------------
# Similarity scoring + aggregation
# ---------------------------------------------------------------------


def fit_whitener(features: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, whitening_matrix) for the provided features."""
    mu = features.mean(axis=0)
    centered = features - mu
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mu, W


def multi_layer_score(
    query_feats: Dict[str, np.ndarray],
    ref_feats: Dict[str, np.ndarray],
    whiteners: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    cfg: ScoringConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute aggregated similarity, per-layer top-1, margins, and neighbour indices.
    """
    layers = cfg.layers
    per_layer_scores: Dict[str, np.ndarray] = {}
    per_layer_margins: Dict[str, np.ndarray] = {}
    topk_indices: Dict[str, np.ndarray] = {}

    for layer in layers:
        Q = query_feats[layer]
        R = ref_feats[layer]
        if whiteners:
            mu, W = whiteners[layer]
            Q = (Q - mu) @ W
            R = (R - mu) @ W

        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        sim = Q @ R.T  # [N_query, N_ref]

        k = max(cfg.k, 2)  # need at least two entries for margins
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        sorted_idx = idx[np.arange(idx.shape[0])[:, None], np.argsort(sim[np.arange(sim.shape[0])[:, None], idx], axis=1)[:, ::-1]]
        vals = sim[np.arange(sim.shape[0])[:, None], sorted_idx]

        per_layer_scores[layer] = vals[:, 0]
        per_layer_margins[layer] = vals[:, 0] - vals[:, 1]
        topk_indices[layer] = sorted_idx[:, :cfg.k]

    if cfg.agg == "min":
        agg = np.minimum.reduce([per_layer_scores[layer] for layer in layers])
    elif cfg.agg == "weighted":
        weights = cfg.weights or {layer: 1.0 for layer in layers}
        total = sum(weights.values())
        agg = sum((weights[layer] / total) * per_layer_scores[layer] for layer in layers)
    else:  # geometric mean
        logs = [np.log(per_layer_scores[layer] + 1e-8) for layer in layers]
        agg = np.exp(np.mean(logs, axis=0))

    return agg, per_layer_scores, per_layer_margins, topk_indices


# ---------------------------------------------------------------------
# Calibration + scoring
# ---------------------------------------------------------------------


def fit_calibration(
    feats_train: Dict[str, np.ndarray],
    cfg: ScoringConfig,
    n_bootstrap: int = 10,
    bootstrap_frac: float = 0.5,
    rng: np.random.Generator = np.random.default_rng(1234),
) -> CalibrationBundle:
    whiteners = {layer: fit_whitener(feats_train[layer]) for layer in cfg.layers}
    null_samples = []
    per_layer_null = {layer: [] for layer in cfg.layers}
    n = feats_train[cfg.layers[0]].shape[0]
    k = max(1, int(round(n * bootstrap_frac)))

    for _ in range(n_bootstrap):
        perm = rng.permutation(n)
        if 2 * k <= n:
            idx_a = perm[:k]
            idx_b = perm[k : 2 * k]
        else:
            idx_a = perm[:k]
            remaining = [idx for idx in perm if idx not in idx_a]
            while len(remaining) < k:
                extra = rng.permutation(n)
                remaining.extend(x for x in extra if x not in idx_a)
            idx_b = np.array(remaining[:k])
        idx_a = np.array(idx_a)
        subset_a = {layer: feats_train[layer][idx_a] for layer in cfg.layers}
        subset_b = {layer: feats_train[layer][idx_b] for layer in cfg.layers}

        sim, per_layer_scores, _, _ = multi_layer_score(subset_a, subset_b, whiteners, cfg)
        null_samples.append(sim)
        for layer in cfg.layers:
            per_layer_null[layer].append(per_layer_scores[layer])

    null_samples = np.concatenate(null_samples, axis=0)
    per_layer_null = {layer: np.concatenate(vals, axis=0) for layer, vals in per_layer_null.items()}
    mu = float(null_samples.mean())
    sigma = float(null_samples.std() + 1e-8)
    return CalibrationBundle(whiteners=whiteners, cfg=cfg, mu_null=mu, sigma_null=sigma, null_samples=null_samples, per_layer_null=per_layer_null)


def _consensus_from_nn(topk: Dict[str, np.ndarray]) -> np.ndarray:
    layers = list(topk.keys())
    base = topk[layers[0]][:, 0]
    consensus = np.ones_like(base)
    for layer in layers[1:]:
        consensus += (topk[layer][:, 0] == base).astype(int)
    return consensus


def score_memorization(
    feats_query: Dict[str, np.ndarray],
    feats_ref: Dict[str, np.ndarray],
    calib: CalibrationBundle,
    labels: Optional[np.ndarray] = None,
) -> ScoreBundle:
    sim, per_layer_scores, per_layer_margins, topk = multi_layer_score(
        feats_query, feats_ref, calib.whiteners, calib.cfg
    )
    dist = 1.0 - sim
    mi = (sim - calib.mu_null) / calib.sigma_null
    oni = -np.tanh(mi)
    return ScoreBundle(
        similarity=sim,
        distance=dist,
        mi=mi,
        oni=oni,
        per_layer_top1=per_layer_scores,
        per_layer_margin=per_layer_margins,
        consensus=_consensus_from_nn(topk),
        top1_indices=topk,
        labels=labels,
    )


def summarize_score_bundle(bundle: ScoreBundle) -> Dict[str, float]:
    summary = {
        "sim_mean": float(bundle.similarity.mean()),
        "dist_mean": float(bundle.distance.mean()),
        "mi_mean": float(bundle.mi.mean()),
        "mi_std": float(bundle.mi.std()),
        "oni_mean": float(bundle.oni.mean()),
        "oni_std": float(bundle.oni.std()),
        "consensus_mean": float(bundle.consensus.mean()),
    }
    if bundle.labels is not None:
        summary.update(
            {
                "auc": roc_auc_score(bundle.labels, bundle.similarity),
                "ap": average_precision_score(bundle.labels, bundle.similarity),
            }
        )
    return summary


def summarize_calibrated_variants(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    results_dict: Dict[str, VariantResult],
    cfg: ScoringConfig,
    dataset_name: str,
) -> torch.Tensor:
    assert "train" in features_dict, "train features missing"
    calib = fit_calibration(features_dict["train"], cfg)
    rows = []
    for name, result in results_dict.items():
        bundle = score_memorization(features_dict[name], features_dict["train"], calib, labels=result.labels)
        row = summarize_score_bundle(bundle)
        row.update(
            {
                "dataset": dataset_name,
                "variant": name,
                "level": result.level,
                "augment": result.augment,
                "dup_count": int(result.labels.sum()),
                "clean_count": len(result.labels) - int(result.labels.sum()),
            }
        )
        if result.labels.max() > 0:
            dup_mask = result.labels.astype(bool)
            row.update(
                {
                    "mi_dup": float(bundle.mi[dup_mask].mean()),
                    "mi_clean": float(bundle.mi[~dup_mask].mean()),
                    "oni_dup": float(bundle.oni[dup_mask].mean()),
                    "oni_clean": float(bundle.oni[~dup_mask].mean()),
                }
            )
        rows.append(row)
    import pandas as pd

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------


def evaluate_variants_global(
    train_slices: np.ndarray,
    test_slices: np.ndarray,
    duplication_levels: Sequence[float],
    augment_config: Optional[Dict[str, any]],
    encoder: nn.Module,
    cfg: ScoringConfig,
    batch_size: int = 8,
) -> Tuple[Dict[str, VariantResult], Dict[str, Dict[str, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Score a single augmentation configuration across duplication levels.
    augment_config: dict describing the augmentation to apply to duplicated slices, or None.
    """
    encoder = encoder.eval().to(train_slices.device if isinstance(train_slices, torch.Tensor) else next(encoder.parameters()).device)

    train_tensor = to_tensor_nchw(train_slices).to(next(encoder.parameters()).device)
    feats_train = extract_batched_features(encoder, train_tensor, cfg.layers_idx, batch_size=batch_size)
    clear_cuda()

    test_tensor = to_tensor_nchw(test_slices).to(next(encoder.parameters()).device)
    feats_test = extract_batched_features(encoder, test_tensor, cfg.layers_idx, batch_size=batch_size)
    clear_cuda()

    whiteners = {layer: fit_whitener(feats_train[layer]) for layer in cfg.layers} if cfg.whitening else None

    results: Dict[str, VariantResult] = {}
    features: Dict[str, Dict[str, np.ndarray]] = {"train": feats_train, "baseline": feats_test}

    sim_base, _, _, topk_base = multi_layer_score(feats_test, feats_train, whiteners, cfg)
    results["baseline"] = VariantResult(
        name="baseline",
        level=0.0,
        augment=False,
        scores=sim_base,
        distances=1.0 - sim_base,
        labels=np.zeros_like(sim_base, dtype=int),
        topk=topk_base,
    )

    for level in duplication_levels:
        if level <= 0:
            continue
        dup_imgs, dup_idx = duplicate_slices(test_slices, train_slices, level, augment=None)
        tensor_dup = to_tensor_nchw(dup_imgs).to(next(encoder.parameters()).device)
        feats_dup = extract_batched_features(encoder, tensor_dup, cfg.layers_idx, batch_size=batch_size)
        clear_cuda()
        features[f"dup_{level}"] = feats_dup

        sim_dup, _, _, topk_dup = multi_layer_score(feats_dup, feats_train, whiteners, cfg)
        labels_dup = np.zeros_like(sim_dup, dtype=int)
        labels_dup[dup_idx] = 1
        results[f"dup_{level}"] = VariantResult(
            name=f"dup_{level}",
            level=level,
            augment=False,
            scores=sim_dup,
            distances=1.0 - sim_dup,
            labels=labels_dup,
            topk=topk_dup,
        )

        if augment_config:
            dup_aug_imgs, dup_aug_idx = duplicate_slices(test_slices, train_slices, level, augment=augment_config)
            tensor_aug = to_tensor_nchw(dup_aug_imgs).to(next(encoder.parameters()).device)
            feats_aug = extract_batched_features(encoder, tensor_aug, cfg.layers_idx, batch_size=batch_size)
            clear_cuda()
            features[f"dup_aug_{level}"] = feats_aug

            sim_aug, _, _, topk_aug = multi_layer_score(feats_aug, feats_train, whiteners, cfg)
            labels_aug = np.zeros_like(sim_aug, dtype=int)
            labels_aug[dup_aug_idx] = 1
            results[f"dup_aug_{level}"] = VariantResult(
                name=f"dup_aug_{level}",
                level=level,
                augment=True,
                scores=sim_aug,
                distances=1.0 - sim_aug,
                labels=labels_aug,
                topk=topk_aug,
            )

    return results, features, whiteners or {}


def evaluate_augmentation_sweep(
    train_slices: np.ndarray,
    test_slices: np.ndarray,
    duplication_levels: Sequence[float],
    augment_grid: Dict[str, Dict[str, any]],
    encoder: nn.Module,
    cfg: ScoringConfig,
    batch_size: int = 8,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Evaluate all augmentations in augment_grid, returning feature and result dictionaries.
    """
    sweep_results = {}
    sweep_features = {}
    for aug_name, params in augment_grid.items():
        results, features, _ = evaluate_variants_global(
            train_slices=train_slices,
            test_slices=test_slices,
            duplication_levels=duplication_levels,
            augment_config=params if params else None,
            encoder=encoder,
            cfg=cfg,
            batch_size=batch_size,
        )
        sweep_results[aug_name] = {"variants": results, "features": features}
        sweep_features[aug_name] = features
    return sweep_results, sweep_features
