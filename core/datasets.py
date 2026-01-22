from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from .preprocessing import minmax_norm


def load_png_dataset(
    root: str | Path,
    pattern: str = "*.png",
    file_range: Tuple[int, Optional[int]] = (0, None),
    target_size: Optional[Tuple[int, int]] = None,
    normalization: str = "minmax",
) -> np.ndarray:
    """
    Load a folder of PNG slices and return an (N, H, W, 3) float32 array.
    """
    root = Path(root)
    files = sorted(root.glob(pattern))
    start, end = file_range
    files = files[start:end]

    images = []
    for path in files:
        img = Image.open(path)
        if target_size:
            img = img.resize(target_size[::-1], Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if normalization == "minmax":
            for k in range(3):
                arr[..., k] = minmax_norm(arr[..., k])
        images.append(arr)

    return np.stack(images, axis=0) if images else np.empty((0,))


def load_spine_dataset(
    root_dir: str | Path,
    max_files: Optional[int] = None,
    target_size: Tuple[int, int] = (240, 240),
) -> np.ndarray:
    """
    Load spine MRI volumes from NIfTI files and return (N, H, W, 3) slices.
    Assumes the last dimension is the slice axis.
    """
    root = Path(root_dir)
    nii_files = sorted(list(root.glob("*.nii")) + list(root.glob("*.nii.gz")))
    if max_files:
        nii_files = nii_files[:max_files]

    all_slices = []
    for nii_path in nii_files:
        img = nib.load(str(nii_path))
        data = img.get_fdata().astype(np.float32)

        if data.ndim < 3:
            continue

        num_slices = data.shape[-1]
        for i in range(num_slices):
            slice_2d = data[..., i]
            if slice_2d.ndim > 2:
                slice_2d = slice_2d[..., 0]

            if target_size:
                pil = Image.fromarray(slice_2d, mode="F")
                pil = pil.resize(target_size[::-1], Image.BILINEAR)
                slice_2d = np.array(pil).astype(np.float32)

            slice_2d = minmax_norm(slice_2d)
            slice_3ch = np.stack([slice_2d] * 3, axis=-1)
            all_slices.append(slice_3ch)

    return np.stack(all_slices, axis=0) if all_slices else np.empty((0,))
