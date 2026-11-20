"""PR-7 RED-first tests: artifact-gated TSR / TIL biomarkers.

Oracles (cross-implementation — TESTING §1 allows an independently-written
reference implementation):
- A naive O(n^2) loop in this file recomputes TSR/TIL from the fixture grid;
  the module (vectorized dilation) must agree exactly.
- CONTAMINATION PROBE: excluded-region patches turned into artifacts must leave
  TSR bit-identical (the gate treats artifacts exactly like excluded tissue).
- EDGE: tumor-free slide -> TSR None; artifacts never count as LYM.
"""

import numpy as np

from histogate.biomarkers import compute_til, compute_tsr
from histogate.schemas.report import TissueClass as TC

T, S, L, B = TC.TUM, TC.STR, TC.LYM, TC.BACK
PATCH_MM2 = 0.112 * 0.112  # 224 px * 0.5 um/px per side


def _grid_from(rows: list[str]) -> np.ndarray:
    m = {"T": T, "S": S, "L": L, "B": B, "D": TC.DEB, "M": TC.MUS}
    return np.array([[m[ch] for ch in row] for row in rows])


GRID_ROWS = [
    "BBBBBBBBBB",
    "BSSSSSSSSB",
    "BSTTTTSSSB",
    "BSTLLTSSSB",
    "BSTTTTSSSB",
    "BSSSSSSSSB",
    "BSSLSMMMSB",
    "BBBBBBBBBB",
    "BBBBBBBBBB",
    "BBBBBBBBBB",
]


def _naive_reference(grid: np.ndarray, artifact_mask: np.ndarray | None) -> dict:
    """Independent, deliberately naive reference implementation (loops + sets)."""
    art = artifact_mask if artifact_mask is not None else np.zeros(grid.shape, bool)
    h, w = grid.shape

    def neigh(y: int, x: int) -> set[tuple[int, int]]:
        out = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and not (dy == 0 and dx == 0):
                    out.add((yy, xx))
        return out

    tum = {(y, x) for y in range(h) for x in range(w) if grid[y, x] == T and not art[y, x]}
    tumor_region = set(tum)
    for (y, x) in tum:
        tumor_region |= neigh(y, x)
    tumor_region = {p for p in tumor_region if not art[p]}

    str_adj = {p for p in tumor_region if grid[p] == S}
    n_t, n_s = len(tum), len(str_adj)
    tsr = n_t / (n_t + n_s) if n_t + n_s else None
    stroma_frac = n_s / (n_t + n_s) if n_t + n_s else None

    intra_lym = sum(1 for p in tumor_region if grid[p] == L)
    intra_area = len(tumor_region) * PATCH_MM2
    all_str = {(y, x) for y in range(h) for x in range(w)
               if grid[y, x] == S and not art[y, x]}
    # stromal LYM: LYM patches outside the tumor region but touching stroma
    stromal_lym = 0
    for y in range(h):
        for x in range(w):
            if (
                grid[y, x] == L
                and not art[y, x]
                and (y, x) not in tumor_region
                and any(grid[p] == S for p in neigh(y, x))
            ):
                stromal_lym += 1
    return {
        "tsr": tsr,
        "stroma_fraction": stroma_frac,
        "intra_density": intra_lym / intra_area,
        "stromal_density": (stromal_lym / (len(all_str) * PATCH_MM2)) if all_str else 0.0,
        "n_tum": n_t,
    }


class TestTSR:
    def test_matches_naive_reference(self) -> None:
        grid = _grid_from(GRID_ROWS)
        ref = _naive_reference(grid, None)
        tsr = compute_tsr(grid)
        assert tsr is not None
        assert abs(tsr.tsr - ref["tsr"]) < 1e-9
        assert abs(tsr.stroma_fraction - ref["stroma_fraction"]) < 1e-9
        assert tsr.group == ("stroma-high" if ref["stroma_fraction"] > 0.5 else "stroma-low")

    def test_contamination_probe_tsr_invariant(self) -> None:
        """Far-region excluded patches -> artifact patches: TSR bit-identical."""
        grid = _grid_from(GRID_ROWS)
        base = compute_tsr(grid)
        g2 = grid.copy()
        g2[8, :] = B  # already BACK; now mark them as artifacts instead
        art = np.zeros(grid.shape, bool)
        art[8, :] = True
        got = compute_tsr(g2, artifact_mask=art)
        assert base is not None and got is not None
        assert got.tsr == base.tsr and got.stroma_fraction == base.stroma_fraction

    def test_artifact_replacing_stroma_leaves_ratio(self) -> None:
        grid = _grid_from(GRID_ROWS)
        ref_before = _naive_reference(grid, None)
        g2 = grid.copy()
        g2[1, 2] = B  # stroma ring patch becomes excluded (same as artifact)
        art = np.zeros(grid.shape, bool)
        art[1, 2] = True
        ref_after = _naive_reference(g2, art)
        got = compute_tsr(g2, artifact_mask=art)
        assert got is not None
        assert abs(got.tsr - ref_after["tsr"]) < 1e-9
        assert ref_before["tsr"] != ref_after["tsr"]  # exclusion is real, not a no-op

    def test_tumor_free_grid_returns_none(self) -> None:
        assert compute_tsr(_grid_from(["SSSSSSSSSS"] * 10)) is None

    def test_disclosure_present(self) -> None:
        tsr = compute_tsr(_grid_from(GRID_ROWS))
        assert tsr is not None and len(tsr.disclosure) > 20


class TestTIL:
    def test_matches_naive_reference(self) -> None:
        grid = _grid_from(GRID_ROWS)
        ref = _naive_reference(grid, None)
        til = compute_til(grid)
        assert abs(til.intratumoral_density_mm2 - ref["intra_density"]) < 1e-9
        assert abs(til.stromal_density_mm2 - ref["stromal_density"]) < 1e-9
        assert til.mpp == 0.5

    def test_no_lym_zero_densities(self) -> None:
        grid = _grid_from(["BSTTTSSSBB"] * 10)
        til = compute_til(grid)
        assert til.intratumoral_density_mm2 == 0.0
        assert til.stromal_density_mm2 == 0.0

    def test_artifacts_never_count_as_lym(self) -> None:
        grid = _grid_from(GRID_ROWS)
        g2 = grid.copy()
        g2[3, 3] = B
        art = np.zeros(grid.shape, bool)
        art[3, 3] = True
        ref = _naive_reference(g2, art)
        til = compute_til(g2, artifact_mask=art)
        assert abs(til.intratumoral_density_mm2 - ref["intra_density"]) < 1e-9
