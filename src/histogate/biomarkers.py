"""PR-7: artifact-gated TME biomarkers — TSR (van Pelt 2018 patch-proxy) and
TIL densities (Saltz 2018 compartment-proxy). Spec: PRD FR-7/8/9, LLD §9.

The artifact gate is AIRA's own published dual-pipeline pattern (Tox Path 2021,
PMC8262119): artifact pixels never reach the scoring math. Proxy status is
DISCLOSED in every TSRResult — never claimed as the slide-margin protocol.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from histogate.schemas.report import TILResult, TissueClass, TSRResult

_PATCH_UM = 224 * 0.5  # 224 px at 0.5 um/px
PATCH_MM2 = (_PATCH_UM / 1000.0) ** 2

_DISCLOSURE = (
    "Patch-level proxy of the van Pelt 2018 slide-margin TSR protocol: ratio of "
    "tumor to stroma patches in the 8-neighborhood of tumor patches, artifacts "
    "and non-stromal compartments excluded. Not the invasive-margin field protocol."
)


def _as_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {g.shape}")
    return g


def _masked(g: np.ndarray, artifact_mask: np.ndarray | None) -> np.ndarray:
    """True where the patch is excluded from ALL scoring (artifact or blank)."""
    art = np.zeros(g.shape, dtype=bool) if artifact_mask is None else np.asarray(artifact_mask, bool)
    return art | (g == TissueClass.BACK)


def compute_tsr(
    grid: np.ndarray, artifact_mask: np.ndarray | None = None
) -> TSRResult | None:
    """Tumor-Stroma Ratio over the tumor-adjacent compartment. None if no tumor."""
    g = _as_grid(grid)
    excl = _masked(g, artifact_mask)

    tumor = (g == TissueClass.TUM) & ~excl
    if not tumor.any():
        return None
    region = binary_dilation(tumor, structure=np.ones((3, 3), bool)) & ~excl
    stroma_adj = (g == TissueClass.STR) & region

    n_t = int(tumor.sum())
    n_s = int(stroma_adj.sum())
    if n_t + n_s == 0:
        return None

    stroma_fraction = n_s / (n_t + n_s)
    return TSRResult(
        stroma_fraction=stroma_fraction,
        tumor_fraction=n_t / (n_t + n_s),
        tsr=n_t / (n_t + n_s),
        group="stroma-high" if stroma_fraction > 0.50 else "stroma-low",
        method="patch-proxy-vanpelt-2018",
        disclosure=_DISCLOSURE,
    )


def compute_til(
    grid: np.ndarray, artifact_mask: np.ndarray | None = None
) -> TILResult:
    """TIL densities per mm^2: LYM patches within the tumor region (intratumoral)
    and LYM patches touching stroma outside it (stromal). Artifacts never count."""
    g = _as_grid(grid)
    excl = _masked(g, artifact_mask)

    tumor = (g == TissueClass.TUM) & ~excl
    lym = (g == TissueClass.LYM) & ~excl

    tumor_region = (
        binary_dilation(tumor, structure=np.ones((3, 3), bool)) & ~excl
        if tumor.any() else np.zeros(g.shape, bool)
    )
    intra = int((lym & tumor_region).sum())
    intra_area = float(tumor_region.sum()) * PATCH_MM2
    intra_density = intra / intra_area if intra_area > 0 else 0.0

    stroma = (g == TissueClass.STR) & ~excl
    touch_stroma = binary_dilation(stroma, structure=np.ones((3, 3), bool))
    stromal = int((lym & ~tumor_region & touch_stroma).sum())
    stromal_area = float(stroma.sum()) * PATCH_MM2
    stromal_density = stromal / stromal_area if stromal_area > 0 else 0.0

    return TILResult(
        intratumoral_density_mm2=intra_density,
        stromal_density_mm2=stromal_density,
        mpp=0.5,
    )
