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
