"""A1 (RED): the KNN gate must score in-distribution tiles through the SAME
feature path used to build the calibration bank. The hook on ``stages[-1]``
yields pre-final-norm features while the bank came from the headless trunk
forward ``model.backbone(x)`` — a naive flip rejected 100% of CRC-VAL.
This test drives PatchAnalyzer itself (the live gate path) with the knn
calibration key set, on real in-distribution tiles, and asserts >= 95% of
them are NOT declined. Deterministic seed; skipped if the dataset or the
bank artifact is absent."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs/pr16-mixed/best.ckpt.pt"
BANK = ROOT / "runs/pr16-mixed/ood_knn_bank.pt"
VAL = ROOT / "data/raw/CRC-VAL-HE-7K"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and BANK.exists() and VAL.exists()),
    reason="checkpoint, KNN bank, or CRC-VAL absent")


def test_live_gate_path_matches_calibration_path(tmp_path: Path) -> None:
    import torch
    from PIL import Image

    from histogate.serve.patch_demo import PatchAnalyzer

    calib_path = tmp_path / "calibration.json"
    calib_path.write_text(json.dumps({
        "temperature": 0.6815141120181452,
        "energy_threshold_fpr05": -3.7368021268890788,
        "ood_detector": "knn",
        "knn_threshold_fpr05": _threshold(),
    }))
    tmp_path.joinpath("ood_knn_bank.pt").write_bytes(BANK.read_bytes())
    an = PatchAnalyzer(CKPT, calib_path)
    files = sorted(VAL.glob("*/*.tif"))
    random.seed(11)
    sample = random.sample(files, 100)
    declined = sum(bool(an.analyze(Image.open(f).convert("RGB"))["ood"])
                   for f in sample)
    assert declined <= 5, (
        f"live-path KNN gate declined {declined}/100 CRC-VAL tiles — "
        f"the gate path diverges from the calibration path")


def _threshold() -> float:
    """Threshold recalibrated on the live path at 5% FPR (A3), cached to
    runs/pr16-mixed/ood_knn.json -> knn_threshold_live. Until A3 runs, the
    bank-path threshold is used, which makes this test RED by construction."""
    j = json.loads((ROOT / "runs/pr16-mixed/ood_knn.json").read_text())
    return float(j.get("knn_threshold_live", j["threshold_fpr05"]))
