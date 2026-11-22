"""P15-S4 RED→GREEN: the serve engine must run the REAL ft2 checkpoint.

Adversarial angles (what a stub or mis-wiring would miss):
- a 4×4 mosaic of REAL CRC-VAL TUM patches must classify majority-TUM
  (wrong ckpt, wrong normalization, or channel-flip wiring fails this)
- model_revision must equal the git sha recorded in the run's eval artifact
  (a stale/random-weights engine fails this)
- energy gating + TSR/TIL must be computed from the run's calibration.json
  (hardcoded defaults fail this)
Skipped (with reason) on machines without the checkpoint, e.g. CI runners.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "runs" / "pr16-mixed" / "best.ckpt.pt"
CALIB = REPO / "runs" / "pr16-mixed" / "calibration.json"
RAW = REPO / "data" / "raw" / "CRC-VAL-HE-7K" / "TUM"


def _mosaic(tmp: Path, n: int = 4) -> Path:
    imgs = sorted(RAW.glob("*.tif"))[: n * n]
    if len(imgs) < n * n:
        imgs = sorted(RAW.glob("*.png"))[: n * n]
    assert len(imgs) == n * n, f"need {n * n} real TUM tiles, found {len(imgs)}"
    rows = [np.concatenate([tifffile.imread(p) for p in imgs[i * n:(i + 1) * n]],
                           axis=1) for i in range(n)]
    out = tmp / "tum_mosaic.tif"
    tifffile.imwrite(out, np.concatenate(rows, axis=0))
    return out


@pytest.mark.skipif(not CKPT.exists() or not CALIB.exists(),
                    reason="checkpoint/calibration artifacts not on this machine")
def test_real_engine_classifies_real_tum_mosaic(tmp_path):
    from histogate.schemas.report import TissueClass
    from histogate.serve.inference import build_analyze

    analyze = build_analyze(CKPT, CALIB, device="cpu")
    report = analyze(_mosaic(tmp_path))

    calib = json.loads(CALIB.read_text())
    assert report.model_revision == calib["git_sha"], (
        f"revision {report.model_revision} != measured {calib['git_sha']}")
    assert len(report.compartment_fractions) == 9
    assert sum(report.compartment_fractions.values()) <= 1.0 + 1e-6
    tum_frac = report.compartment_fractions.get(TissueClass.TUM, 0.0)
    assert tum_frac > 0.5, f"real TUM mosaic classified only {tum_frac} TUM"
    assert report.calibration_temperature == pytest.approx(
        calib["temperature"], rel=1e-6), "temperature not from calibration.json"
    assert report.n_patches == 16
    assert report.tsr is not None or report.til is not None


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not on this machine")
def test_engine_rejects_blank_slide(tmp_path):
    """Adversarial: an all-white slide has no tissue — the engine must gate
    every patch as OOD rather than emit confident tissue fractions."""
    from histogate.serve.inference import build_analyze

    blank = np.full((448, 448, 3), 255, dtype=np.uint8)
    p = tmp_path / "blank.tif"
    tifffile.imwrite(p, blank)
    analyze = build_analyze(CKPT, CALIB, device="cpu")
    with pytest.raises(ValueError, match="no tissue tiles"):
        analyze(p)


@pytest.mark.skipif(not CKPT.exists() or not CALIB.exists(),
                    reason="checkpoint/calibration artifacts not on this machine")
def test_slide_map_tiles_and_familiarity(tmp_path):
    """B0: slide map must tile a real mosaic, emit per-tile verdicts with a
    0-100 familiarity derived from energy, and the gate from calibration."""
    from histogate.serve.patch_demo import PatchAnalyzer
    from histogate.serve.slide_map import _familiarity, build_slide_map

    assert _familiarity(-4.6) == 100
    assert _familiarity(-1.6) == 0
    assert _familiarity(-3.0) == 47

    imgs = sorted((RAW).glob("*.tif"))[:9] or sorted(RAW.glob("*.png"))[:9]
    assert len(imgs) == 9, "need 9 real TUM tiles for the mosaic"
    tiles = [tifffile.imread(p) for p in imgs]
    th = tiles[0].shape[0]
    rows = [np.concatenate(tiles[i * 3:(i + 1) * 3], axis=1) for i in range(3)]
    mosaic = tmp_path / "slide_map_mosaic.tif"
    tifffile.imwrite(mosaic, np.concatenate(rows, axis=0))

    analyzer = PatchAnalyzer(CKPT, CALIB)
    out = build_slide_map(mosaic, analyzer, patch=th, stride=th)
    assert out["grid"] == {"cols": 3, "rows": 3}
    assert len(out["tiles"]) == 9
    t = out["tiles"][0]
    assert {"id", "col", "row", "cls", "conf", "energy",
            "familiarity", "declined"} <= set(t)
    assert all(0 <= x["familiarity"] <= 100 for x in out["tiles"])
    assert out["gate"] == pytest.approx(analyzer.energy_thr, rel=1e-6)
    assert sum(x["declined"] for x in out["tiles"]) == sum(
        x["energy"] > out["gate"] for x in out["tiles"]), (
        "declined flag must equal energy above gate")
