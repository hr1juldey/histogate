"""WSI pyramid tiling pipeline (PR-8; docs/LLD.md section 8).

- tissue_mask: Otsu threshold on grayscale with constant-image guard
  (blank/empty slides return an empty mask, not a crash or a full mask).
- tile_grid: stride grid over (H, W); a tile is kept iff its tissue
  fraction >= min_tissue. Deterministic by construction (plain loops).
- read_patch: bounds-checked exact crop; partial patches are rejected
  (a silent zero-padded patch would corrupt biomarker area math).
- load_levels: tifffile pyramid (page 0 + subIFDs). OpenSlide-class SVS
  inputs are out of scope for this dataset; the NCT/CRC-VAL images are
  plain TIFFs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu


def tissue_mask(img: np.ndarray) -> np.ndarray:
    """Boolean tissue mask for an RGB uint8 image.

    Otsu on grayscale separates stained tissue (dark) from background
    (bright). A (near-)constant image has no Otsu split -> empty mask.
    """
    gray = img[..., :3].mean(axis=-1)
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        return np.zeros(gray.shape, dtype=bool)
    thresh = threshold_otsu(gray)
    return gray < thresh


def tile_grid(
    shape: tuple[int, int],
    mask: np.ndarray,
    patch: int = 224,
    stride: int = 224,
    min_tissue: float = 0.1,
) -> list[tuple[int, int]]:
    """Grid tile coordinates (y, x) whose tissue fraction >= min_tissue.

    Only fully-in-bounds tiles are produced (no partials).
    """
    h, w = shape[:2]
    coords: list[tuple[int, int]] = []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            if mask[y : y + patch, x : x + patch].mean() >= min_tissue:
                coords.append((y, x))
    return coords


def read_patch(img: np.ndarray, y: int, x: int, patch: int) -> np.ndarray:
    """Exact crop img[y:y+patch, x:x+patch]; ValueError on out-of-bounds."""
    h, w = img.shape[:2]
    if not (y >= 0 and y + patch <= h and x >= 0 and x + patch <= w):
        raise ValueError(f"patch ({y},{x},{patch}) out of bounds for image {h}x{w}")
    return img[y : y + patch, x : x + patch]


def load_levels(path: str | Path) -> list[np.ndarray]:
    """Load a TIFF pyramid via series levels (base first). Single-page
    TIFFs yield a one-element list."""
    with tifffile.TiffFile(str(path)) as tf:
        return [level.asarray() for level in tf.series[0].levels]
