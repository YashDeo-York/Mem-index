from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Sequence

import nibabel as nib
import numpy as np
from tqdm import tqdm


def centre_crop_2d(img: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Center-crop a 2D array to (crop_h, crop_w)."""
    h, w = img.shape[:2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size {(crop_h, crop_w)} larger than image {(h, w)}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return img[top : top + crop_h, left : left + crop_w]


def minmax_norm(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    a_min = float(np.min(arr))
    a_max = float(np.max(arr))
    denom = a_max - a_min
    if denom < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - a_min) / denom).astype(np.float32)


@dataclass(frozen=True)
class LoadConfig:
    root: Optional[Path]
    folder: str = "images"
    file_range: Tuple[int, Optional[int]] = (0, None)
    slice_range: Tuple[int, Optional[int]] = (0, None)
    crop_size: Optional[Tuple[int, int]] = None
    normalization: str = "minmax"
    channel_strategy: str = "first3"  # 'first3', 'select', 'drop_minvar', 'maxblend01', 'pca'
    select_indices: Optional[Sequence[int]] = None
    max_files: Optional[int] = None


def _list_files(data_dir: Path) -> list[str]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    files.sort()
    return files


def _to_three_channels(
    slice_vol: np.ndarray, strategy: str = "first3", select_indices: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Convert 2D or multi-channel slice to 3 channels.

    - 2D (H,W): replicate to (H,W,3)
    - 3D (H,W,C):
      * C==3: return as-is
      * C>3: PCA to 3 components by default, or select indices when provided
      * C==2: append first channel to make 3
      * C==1: replicate to 3
    """
    if slice_vol.ndim == 2:
        return np.repeat(slice_vol[..., None], 3, axis=-1)
    if slice_vol.ndim != 3:
        raise ValueError(f"Expected 2D/3D slice, got shape {slice_vol.shape}")
    h, w, c = slice_vol.shape
    if c == 3:
        return slice_vol
    if c == 1:
        return np.repeat(slice_vol, 3, axis=-1)
    if c == 2:
        return np.concatenate([slice_vol, slice_vol[..., :1]], axis=-1)
    if strategy == "select":
        idx = list(select_indices) if select_indices is not None else [0, 1, 2]
        if len(idx) != 3:
            raise ValueError("select_indices must contain 3 entries")
        idx = [min(max(0, i), c - 1) for i in idx]
        return slice_vol[..., idx]
    if strategy == "first3":
        return slice_vol[..., :3]
    if strategy == "drop_minvar":
        # Keep the three channels with highest spatial variance
        vars_ = [float(slice_vol[..., i].var()) for i in range(c)]
        keep = np.argsort(vars_)[-3:]
        keep.sort()
        return slice_vol[..., keep]
    if strategy == "maxblend01":
        # Blend ch0 and ch1 using per-pixel max, then take two highest-var of remaining
        ch0 = slice_vol[..., 0]
        ch1 = slice_vol[..., 1] if c > 1 else ch0
        blended = np.maximum(ch0, ch1)[..., None]
        remaining = []
        for i in range(2, c):
            remaining.append((float(slice_vol[..., i].var()), i))
        # Fallback when c<3
        if not remaining:
            if c == 2:
                # replicate a remaining channel to get 3
                return np.concatenate([blended, ch0[..., None], ch1[..., None]], axis=-1).astype(np.float32)
            return np.repeat(ch0[..., None], 3, axis=-1)
        remaining.sort()
        idxs = [i for _, i in remaining[-2:]] if len(remaining) >= 2 else [remaining[-1][1], 0]
        stacked = np.concatenate([blended, slice_vol[..., idxs[0]][..., None], slice_vol[..., idxs[1]][..., None]], axis=-1)
        return stacked.astype(np.float32)
    # PCA fallback
    x = slice_vol.reshape(-1, c).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        # SVD gives principal axes in rows of V^T
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        pcs = x @ vt[:3].T
    except np.linalg.LinAlgError:
        pcs = x[:, :3]
    pcs = pcs.reshape(h, w, 3).astype(np.float32)
    # Scale each component to [0,1] for stability
    for k in range(3):
        a = pcs[..., k]
        a_min, a_max = float(a.min()), float(a.max())
        pcs[..., k] = 0.0 if a_max <= a_min else (a - a_min) / (a_max - a_min)
    return pcs


def _default_brats_root() -> Path:
    # repository_root/data/brats
    return Path(__file__).resolve().parent.parent / "data" / "brats"


def load_brats_dataset(
    root: Path | str | None = None,
    folder: str = "images",
    file_range: Tuple[int, Optional[int]] = (0, None),
    slice_range: Tuple[int, Optional[int]] = (0, None),
    crop_size: Optional[Tuple[int, int]] = None,
    normalization: str = "minmax",
    channel_strategy: str = "pca",
    select_indices: Optional[Sequence[int]] = None,
    max_files: Optional[int] = None,
) -> np.ndarray:
    """
    Load BRATS volumes from a subfolder and convert to stacked 2D slices.

    This mirrors Testing.ipynb with improvements:
    - Convert multi-modal slices to 3 channels (PCA or selected modalities)
    - Optional per-slice per-channel min-max normalization
    - Slice range selection and optional center crop
    """
    data_dir = (Path(root) if root is not None else _default_brats_root()) / folder
    if not data_dir.is_dir():
        raise ValueError(f"Directory {data_dir} does not exist.")

    files = _list_files(data_dir)
    start_idx, end_idx = file_range
    end_idx = len(files) if end_idx is None else end_idx
    files = files[start_idx:end_idx]
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    all_slices: list[np.ndarray] = []
    for fname in tqdm(files, desc=f"Loading {folder} volumes"):
        vol_path = data_dir / fname
        vol = nib.load(str(vol_path))
        vol_data = vol.get_fdata()
        st, en = slice_range
        if vol_data.ndim == 3:
            vol_sel = vol_data[:, :, st:] if en is None else vol_data[:, :, st:en]
        elif vol_data.ndim == 4:
            vol_sel = vol_data[:, :, st:, :] if en is None else vol_data[:, :, st:en, :]
        else:
            raise ValueError(f"Unsupported volume ndim: {vol_data.ndim}")

        depth = vol_sel.shape[2]
        for i in range(depth):
            if vol_sel.ndim == 3:
                raw_slice = vol_sel[:, :, i]
                rgb = _to_three_channels(raw_slice)
            else:
                raw_slice = vol_sel[:, :, i, :]
                rgb = _to_three_channels(raw_slice, strategy=channel_strategy, select_indices=select_indices)

            if normalization.lower() == "minmax":
                for k in range(3):
                    ch = rgb[..., k]
                    rgb[..., k] = minmax_norm(ch)

            if crop_size is not None:
                rgb = centre_crop_2d(rgb, crop_size[0], crop_size[1])
            all_slices.append(rgb.astype(np.float32))

    return np.stack(all_slices, axis=0) if all_slices else np.empty((0,))


def suggest_brats_slice_range(
    root: Path | str, folder: str = "mask", num_volumes: int = 40
) -> Optional[Tuple[int, int]]:
    """Suggest (min_slice, max_slice) that contain non-zero mask across samples."""
    data_dir = Path(root) / folder
    if not data_dir.is_dir():
        raise ValueError(f"Directory {data_dir} does not exist.")

    files = _list_files(data_dir)[:num_volumes]
    tumor_slice_indices: list[int] = []
    for fname in tqdm(files, desc="Analyzing mask volumes"):
        vol = nib.load(str(data_dir / fname))
        vol_data = vol.get_fdata()
        if vol_data.ndim < 3:
            continue
        tumor_slices = [i for i in range(vol_data.shape[2]) if np.any(vol_data[:, :, i] > 0)]
        tumor_slice_indices.extend(tumor_slices)

    if not tumor_slice_indices:
        return None
    return int(min(tumor_slice_indices)), int(max(tumor_slice_indices))


def suggest_brats_center_crop(
    root: Path | str,
    folder: str = "mask",
    num_volumes: int = 40,
    margin: float = 0.1,
) -> Optional[Tuple[int, int]]:
    """Suggest center crop size based on union of mask bounding boxes with margin."""
    data_dir = Path(root) / folder
    if not data_dir.is_dir():
        raise ValueError(f"Directory {data_dir} does not exist.")

    files = _list_files(data_dir)[:num_volumes]
    global_min_row, global_min_col = np.inf, np.inf
    global_max_row, global_max_col = -np.inf, -np.inf

    for fname in tqdm(files, desc="Analyzing mask volumes"):
        vol = nib.load(str(data_dir / fname))
        vol_data = vol.get_fdata()
        if vol_data.ndim < 3:
            continue
        for i in range(vol_data.shape[2]):
            slice_data = vol_data[:, :, i]
            if slice_data.ndim > 2:
                slice_data = slice_data[:, :, 0]
            if np.any(slice_data > 0):
                rows, cols = np.where(slice_data > 0)
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                global_min_row = min(global_min_row, min_row)
                global_min_col = min(global_min_col, min_col)
                global_max_row = max(global_max_row, max_row)
                global_max_col = max(global_max_col, max_col)

    if not np.isfinite(global_min_row) or not np.isfinite(global_min_col):
        return None

    crop_h = int(np.ceil((global_max_row - global_min_row) * (1 + margin)))
    crop_w = int(np.ceil((global_max_col - global_min_col) * (1 + margin)))
    return crop_h, crop_w
