from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------- Utilities ---------


def _poly_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    d = x.shape[1]
    return (x @ y.T / d + coef0) ** degree


def _safe_cov(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.cov(x, rowvar=False)


def _safe_mean(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.mean(x, axis=0)


# --------- Metrics (features) ---------


def compute_fid_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Compute FID between two sets of features."""
    from scipy.linalg import sqrtm  # lazy import

    mu1, mu2 = _safe_mean(real_feats), _safe_mean(fake_feats)
    sig1, sig2 = _safe_cov(real_feats), _safe_cov(fake_feats)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray, num_subsets: int = 100, subset_size: int = 100) -> Tuple[float, float]:
    """Kernel Inception Distance (polynomial kernel) mean and std over subsets."""
    rng = np.random.default_rng(0)
    n_r, n_f = real_feats.shape[0], fake_feats.shape[0]
    m = min(subset_size, n_r, n_f)
    scores: List[float] = []
    for _ in range(num_subsets):
        i = rng.choice(n_r, size=m, replace=False)
        j = rng.choice(n_f, size=m, replace=False)
        xr, xf = real_feats[i], fake_feats[j]
        k_rr = _poly_kernel(xr, xr)
        k_ff = _poly_kernel(xf, xf)
        k_rf = _poly_kernel(xr, xf)
        # unbiased estimator
        np.fill_diagonal(k_rr, 0.0)
        np.fill_diagonal(k_ff, 0.0)
        mmd2 = k_rr.sum() / (m * (m - 1)) + k_ff.sum() / (m * (m - 1)) - 2.0 * k_rf.mean()
        scores.append(float(mmd2))
    return float(np.mean(scores)), float(np.std(scores))


def compute_mmd_poly(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    k_rr = _poly_kernel(real_feats, real_feats)
    k_ff = _poly_kernel(fake_feats, fake_feats)
    k_rf = _poly_kernel(real_feats, fake_feats)
    n_r = real_feats.shape[0]
    n_f = fake_feats.shape[0]
    np.fill_diagonal(k_rr, 0.0)
    np.fill_diagonal(k_ff, 0.0)
    mmd2 = k_rr.sum() / (n_r * (n_r - 1)) + k_ff.sum() / (n_f * (n_f - 1)) - 2.0 * k_rf.mean()
    return float(mmd2)


def _nearest_distances(x: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(x) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x)
    distances, _ = nbrs.kneighbors(x)
    return distances[:, -1]  # distance to k-th neighbor


def compute_prdc(real_feats: np.ndarray, fake_feats: np.ndarray, nearest_k: int = 5) -> Dict[str, float]:
    """
    Precision, Recall, Density, Coverage
    Adapted from PRDC: https://github.com/clovaai/generative-evaluation-prdc
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(nearest_k, max(1, min(len(real_feats), len(fake_feats)) - 1))
    real_nn = NearestNeighbors(n_neighbors=k + 1).fit(real_feats)
    fake_nn = NearestNeighbors(n_neighbors=k + 1).fit(fake_feats)
    real2real = real_nn.kneighbors(real_feats, return_distance=True)[0][:, -1]
    fake2fake = fake_nn.kneighbors(fake_feats, return_distance=True)[0][:, -1]
    # pairwise distances
    real2fake = real_nn.kneighbors(fake_feats, return_distance=True)[0][:, 0]
    fake2real = fake_nn.kneighbors(real_feats, return_distance=True)[0][:, 0]

    precision = float(np.mean(real2fake <= real2real.mean()))
    recall = float(np.mean(fake2real <= fake2fake.mean()))
    density = float(np.mean(real2fake / (real2real.mean() + 1e-8)))
    coverage = float(np.mean(real2fake <= np.quantile(real2real, 0.5)))
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


def compute_authpct(train_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """Authenticity Percentage as in notebook version."""
    train = np.asarray(train_feats, dtype=np.float32)
    gen = np.asarray(gen_feats, dtype=np.float32)
    # pairwise distances
    real_d = _pairwise_cdist(train, train)
    np.fill_diagonal(real_d, np.inf)
    gen_d = _pairwise_cdist(train, gen)
    real_min = np.min(real_d, axis=0)
    gen_min_vals = np.min(gen_d, axis=0)
    gen_min_idx = np.argmin(gen_d, axis=0)
    authen = real_min[gen_min_idx] < gen_min_vals
    return float(100.0 * np.sum(authen) / len(authen))


def _pairwise_cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0))


def compute_vendi_score(x: np.ndarray, q: float | int = 1, normalize: bool = True, kernel: str = "polynomial") -> float:
    if normalize:
        x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        x = x / x_norm
    if kernel == "linear":
        s = x @ x.T
    elif kernel == "polynomial":
        s = _poly_kernel(x, x)
    else:
        raise NotImplementedError("Kernel type not implemented")
    w = np.linalg.eigvalsh(s / max(1, x.shape[0]))
    return float(math.e ** _entropy_q(w, q=q))


def _entropy_q(p: np.ndarray, q: float | int = 1) -> float:
    p = p[p > 0]
    if q == 1:
        return float(-(p * np.log(p)).sum())
    if q == float("inf"):
        return float(-np.log(np.max(p)))
    return float(np.log((p**q).sum()) / (1 - q))


# --------- Runner ---------


SUPPORTED = {
    "FID": compute_fid_features,
    "KID": compute_kid,
    "MMD": compute_mmd_poly,
    "PRDC": compute_prdc,
    "AuthPct": compute_authpct,
    "Vendi": compute_vendi_score,
}


def run_metrics(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    train_feats: Optional[np.ndarray],
    metrics: Sequence[str],
    nearest_k: int = 10,
) -> Dict[str, float | Dict[str, float] | Tuple[float, float]]:
    results: Dict[str, float | Dict[str, float] | Tuple[float, float]] = {}
    for name in metrics:
        if name == "FID":
            results[name] = compute_fid_features(real_feats, fake_feats)
        elif name == "KID":
            results[name] = compute_kid(real_feats, fake_feats)
        elif name == "MMD":
            results[name] = compute_mmd_poly(real_feats, fake_feats)
        elif name == "PRDC":
            results[name] = compute_prdc(real_feats, fake_feats, nearest_k=nearest_k)
        elif name == "AuthPct":
            if train_feats is None:
                raise ValueError("AuthPct requires train features (use --train-feats).")
            results[name] = compute_authpct(train_feats, fake_feats)
        elif name == "Vendi":
            results[name] = compute_vendi_score(fake_feats)
        else:
            raise ValueError(f"Unsupported metric '{name}'. Supported: {list(SUPPORTED.keys())}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run selected metrics on feature matrices (.npy)")
    p.add_argument("--real-feats", required=True, help="Path to real feature .npy (N,D)")
    p.add_argument("--fake-feats", required=True, help="Path to fake/test feature .npy (N,D)")
    p.add_argument("--train-feats", help="Optional path to train feature .npy (N,D) for metrics needing train")
    p.add_argument(
        "--metrics",
        nargs="*",
        default=["FID", "KID", "MMD"],
        help=f"Subset to run. Supported: {list(SUPPORTED.keys())}",
    )
    p.add_argument("--nearest-k", type=int, default=10, help="Nearest neighbors for PRDC")
    p.add_argument("--output-json", help="Optional JSON path to save results")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    real = np.load(args.real_feats)
    fake = np.load(args.fake_feats)
    train = np.load(args.train_feats) if args.train_feats else None
    res = run_metrics(real_feats=real, fake_feats=fake, train_feats=train, metrics=args.metrics, nearest_k=args.nearest_k)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

