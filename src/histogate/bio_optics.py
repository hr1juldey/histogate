"""Beer-Lambert optical density + Macenko stain estimation (Ruifrok & Johnston 2001;
Macenko et al. 2009) + AIRAQc 8-class synthetic artifact injector.
Spec: docs/PRD FR-2/FR-3, LLD §3. NumPy-vectorized; torch GPU port lands at PR-5.
"""

from __future__ import annotations

import numpy as np

# Ruifrok standard H&E reference stain matrix (rows: H, E; cols: R, G, B), unit-normalized.
RUIFROK_HE_REF = np.array(
    [
        [0.644, 0.717, 0.267],
        [0.092, 0.954, 0.283],
    ]
)
RUIFROK_HE_REF = RUIFROK_HE_REF / np.linalg.norm(RUIFROK_HE_REF, axis=1, keepdims=True)

# Standard robust-max concentrations used by the reference implementation class
# (staintools-compatible); the 99th-percentile target for normalization scaling.
REF_MAX_CONC = np.array([1.9709206, 1.0305088])


def rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Beer-Lambert: OD = -log10((I + 1) / 256), per channel. Accepts (..., 3) uint8."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {img.dtype}")
    return -np.log10((img.astype(np.float64) + 1.0) / 256.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Inverse transform; returns float in [0, 255]."""
    return np.clip(255.0 * np.power(10.0, -od) - 1.0, 0.0, 255.0)


def macenko_stain_matrix(
    od: np.ndarray, beta: float = 0.15, percentile: float = 1.0
) -> np.ndarray:
    """Estimate the 2x3 H&E stain matrix from OD pixels (N, 3).

    Macenko 2009: SVD -> stain plane -> angular percentiles -> stain vectors.
    H is placed first (hematoxylin carries the larger R component for H&E).
    Degenerate input (all-white/too few pixels) falls back to the Ruifrok reference.
    """
    od = np.asarray(od, dtype=np.float64).reshape(-1, 3)
    od = od[np.all(od > beta, axis=1)]
    if od.shape[0] < 100 or not np.isfinite(od).all():
        return RUIFROK_HE_REF.copy()

    # Classic Macenko: UNCENTERED SVD — the stain cloud is a set of rays from the
    # OD origin (white), and centering would distort that ray structure.
    _, _, vt = np.linalg.svd(od, full_matrices=False)
    plane = vt[:2]                          # (2, 3)
    proj = od @ plane.T                     # (N, 2)

    phi = np.arctan2(proj[:, 1], proj[:, 0])
    # atan2 wraps at +/-pi: re-center angles on their circular mean so the
    # 1st/99th percentiles pick the TRUE extremes instead of the seam.
    mu = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())
    delta = np.angle(np.exp(1j * (phi - mu)))  # wrapped to (-pi, pi]
    lo, hi = np.percentile(delta, [percentile, 100.0 - percentile])
    v1 = plane.T @ np.array([np.cos(lo + mu), np.sin(lo + mu)])
    v2 = plane.T @ np.array([np.cos(hi + mu), np.sin(hi + mu)])

    # Stain vectors live in the positive octant (absorbance coefficients are > 0);
    # the SVD plane basis has arbitrary sign, so canonicalize before ordering.
    for i, v in enumerate((v1, v2)):
        if v.sum() < 0:
            if i == 0:
                v1 = -v1
            else:
                v2 = -v2

    if v1[0] < v2[0]:  # H first: hematoxylin carries the larger R component
        v1, v2 = v2, v1
    S = np.stack([v1, v2])
    S = S / np.linalg.norm(S, axis=1, keepdims=True)

    if not np.isfinite(S).all():  # pragma: no cover
        return RUIFROK_HE_REF.copy()
    return S


def separate_stains(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Concentrations (N, 2) via least squares: OD ~= C @ S."""
    od = np.asarray(od, dtype=np.float64).reshape(-1, 3)
    return np.linalg.lstsq(np.asarray(stain_matrix, dtype=np.float64).T, od.T, rcond=None)[0].T


def normalize_stains(img: np.ndarray, ref_matrix: np.ndarray | None = None) -> np.ndarray:
    """Macenko normalization: estimate source stains, transfer concentrations to the
    reference stain matrix with 99th-percentile scaling, recompose. Returns uint8
    with the same shape as the input."""
    ref = RUIFROK_HE_REF if ref_matrix is None else np.asarray(ref_matrix, dtype=np.float64)
    shape = np.asarray(img).shape
    od = rgb_to_od(img).reshape(-1, 3)
    S = macenko_stain_matrix(od)
    C = separate_stains(od, S)

    src99 = np.percentile(C, 99.0, axis=0)
    C = C * (REF_MAX_CONC / np.maximum(src99, 1e-8))

    od_norm = C @ ref
    out = od_to_rgb(od_norm).reshape(shape)
    return np.clip(out, 0, 255).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# AIRAQc 8-class synthetic artifact injector (FR-3). Deterministic under seed.
# ---------------------------------------------------------------------------
from histogate.schemas.report import ArtifactKind  # noqa: E402


def inject_artifact(
    img: np.ndarray, kind: str | ArtifactKind, severity: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Inject one AIRAQc-class artifact. severity in [0,1] (0 = identity).
    Returns (artifacted_image, bool_mask) with mask True on artifact pixels.
    Implemented in PR-3; the signature and identity contract are law from PR-2.
    """
    if float(severity) == 0.0:
        return np.array(img, copy=True), np.zeros(img.shape[:2], dtype=bool)
    raise NotImplementedError("artifact injector lands in PR-3 (tests first)")
