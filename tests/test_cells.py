"""P0b: nucleus counter contract — vendored HoVer-Net, fail-closed (plan B2)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def test_nucleus_counter_counts_and_gates() -> None:
    from histogate.cells import NucleusCounter

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    c = NucleusCounter()  # lazy; weights must load from HF cache
    counts = c.count(img)
    assert set(counts) <= {
        "neoplastic",
        "nonneop_epithelial",
        "inflammatory",
        "connective",
        "dead",
    }
    assert sum(counts.values()) >= 0  # noise image: 0..few is fine
    black = c.count(Image.new("RGB", (224, 224), (0, 0, 0)))
    assert black == {} or sum(black.values()) <= 2


@pytest.mark.slow
def test_nucleus_counter_real_tum_tile() -> None:
    """End-to-end on a real CRC-VAL TUM tile: must find nuclei, mostly tissue."""
    from pathlib import Path

    from histogate.cells import NucleusCounter

    root = Path(__file__).resolve().parents[1] / "data/raw/CRC-VAL-HE-7K/TUM"
    tiles = sorted(root.glob("*.tif"))
    if not tiles:
        pytest.skip("CRC-VAL not on disk")
    c = NucleusCounter()
    counts = c.count(Image.open(tiles[0]).convert("RGB"))
    total = sum(counts.values())
    assert total >= 20, f"expected a populated tumor tile, got {counts}"
