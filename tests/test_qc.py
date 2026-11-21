"""P18-S2 RED: adversarial QC-head tests (histogate.qc_head NOT built yet).

T1  cross-source: train on HistoQC-flagged tiles, test on held-out CAMELYON
    slides' flagged tiles — flag recall >= 0.8 on a source never seen.
T2  hard-negative rate: real-but-ugly tissue (CRC-VAL BACK/DEB) must NOT be
    flagged — FP <= 0.05 on the eye-confirmed hard-negative set.
T3  severity-generalization: dilute a flagged tile 50% toward a clean tile
    (milder artifact) — recall >= 0.6 at the mild end. A detector that only
    catches cartoon-level degradation is useless in production.

All three RED via ImportError until src/histogate/qc_head.py exists.
Skipped where the curated data dirs are absent (CI).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
QC = REPO / "data" / "raw" / "qc_histqc"
TRAIN_DIR = QC / "tiles_train"      # flagged tiles, train slides
TEST_DIR = QC / "tiles_test"        # flagged tiles, held-out slides
HARDNEG_DIR = QC / "hardneg"        # eye-confirmed ugly-but-real tissue
CLEAN_POOL = QC / "clean_pool"      # clean tiles for dilution


def _load(d: Path, cap: int = 400) -> np.ndarray:
    files = sorted(d.glob("*.png"))[:cap]
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files])


@pytest.mark.skipif(
    not all(d.exists() and any(d.glob("*.png"))
            for d in (TRAIN_DIR, TEST_DIR, HARDNEG_DIR, CLEAN_POOL)),
    reason="curated QC tiles not on this machine")
def test_t1_cross_source_recall():
    from histogate.qc_head import QCHead

    head = QCHead.train_on(TRAIN_DIR)          # HistoQC-flagged tiles
    x_test = _load(TEST_DIR)                   # held-out slides
    recall = head.flag_rate(x_test)
    assert recall >= 0.8, (
        f"cross-source recall {recall:.2f} < 0.80: QC head does not "
        f"transfer to unseen slides")


@pytest.mark.skipif(
    not all(d.exists() and any(d.glob("*.png"))
            for d in (TRAIN_DIR, TEST_DIR, HARDNEG_DIR, CLEAN_POOL)),
    reason="curated QC tiles not on this machine")
def test_t2_hard_negative_fp():
    from histogate.qc_head import QCHead

    head = QCHead.train_on(TRAIN_DIR)
    fp = head.flag_rate(_load(HARDNEG_DIR))
    assert fp <= 0.05, (
        f"hard-negative FP {fp:.2f} > 0.05: flags real necrosis/debris "
        f"tissue as degraded — would censor real biology")


@pytest.mark.skipif(
    not all(d.exists() and any(d.glob("*.png"))
            for d in (TRAIN_DIR, TEST_DIR, HARDNEG_DIR, CLEAN_POOL)),
    reason="curated QC tiles not on this machine")
def test_t3_severity_generalization():
    from histogate.qc_head import QCHead

    head = QCHead.train_on(TRAIN_DIR)
    flagged = _load(TEST_DIR, cap=100).astype(np.float32)
    clean = _resize_pool(_load(CLEAN_POOL, cap=100), flagged.shape[1:])
    mild = (0.5 * flagged + 0.5 * clean).astype(np.uint8)  # 50% dilution
    recall_mild = head.flag_rate(mild)
    assert recall_mild >= 0.6, (
        f"mild-severity recall {recall_mild:.2f} < 0.60: only catches "
        f"cartoon-level degradation")


def _resize_pool(pool: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.stack([
        np.asarray(Image.fromarray(p).resize(shape)) for p in pool])
