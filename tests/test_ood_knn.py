"""RED-first: feature-space KNN OOD detector must exist and separate
CRC-VAL (in-distribution) from Camelyon16 lymph-node tiles (OOD) with
AUROC >= 0.90 — the gate the energy detector fails (0.797).
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs/pr16-mixed/best.ckpt.pt"
CALIB = ROOT / "runs/pr16-mixed/calibration.json"
TRAIN = ROOT / "data/raw/NCT-CRC-HE-100K"
VAL = ROOT / "data/raw/CRC-VAL-HE-7K"
CAMELYON = ROOT / "data/raw/camelyon16/tumor_001.tif"

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="checkpoint not present")


def test_knn_ood_detector_exists_and_separates() -> None:
    from histogate.ood import KNNDetector, auroc, features_for_dir

    det = KNNDetector.fit_from_dir(TRAIN, CKPT, CALIB, max_images=1500, k=50)
    pos = features_for_dir(VAL, CKPT, max_images=200)          # in-distribution
    neg = det.features_from_wsi(CAMELYON, level=4, max_tiles=200)  # foreign organ

    scores_pos = det.score(pos)   # low distance = familiar
    scores_neg = det.score(neg)
    auc = auroc(scores_neg, scores_pos)   # higher score = more OOD
    assert auc >= 0.90, f"KNN OOD AUROC {auc:.4f} below the 0.90 bar"
    # deterministic
    import numpy as np
    assert np.array_equal(det.score(pos[:8]), det.score(pos[:8]))
