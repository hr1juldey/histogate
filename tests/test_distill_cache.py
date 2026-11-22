"""P1: teacher-embedding cache writer — resume-safe memmap contract (plan C1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_cache_writer_shapes_dtype_and_index(tmp_path: Path) -> None:
    from histogate.distill_cache import extract_embeddings

    files = [f"img_{i}.tif" for i in range(4)]

    def fake_teacher(batch: list) -> np.ndarray:
        # deterministic 8-dim vectors: [i, i, ..., i] + position
        return np.stack(
            [np.full(8, float(i)) + np.arange(8, dtype=np.float32) for i in batch]
        )

    out = extract_embeddings(files, fake_teacher, tmp_path, dim=8, bs=2)
    emb = np.load(out / "emb.npy")
    assert emb.shape == (4, 8)
    assert emb.dtype == np.float16
    # row content survives the fp16 cast for small ints
    assert emb[0, 0] == 0.0 and emb[3, 7] == 10.0
    idx = json.loads((out / "files.json").read_text())
    assert idx["files"] == files
    assert idx["dim"] == 8 and idx["dtype"] == "float16"
    assert idx["sha"] and isinstance(idx["sha"], str)


def test_cache_writer_resume_skips_completed(tmp_path: Path) -> None:
    from histogate.distill_cache import extract_embeddings

    files = [f"img_{i}.tif" for i in range(4)]
    calls: list[list[int]] = []

    def teacher(batch: list) -> np.ndarray:
        calls.append(list(batch))
        return np.zeros((len(batch), 8), dtype=np.float32)

    out = extract_embeddings(files, teacher, tmp_path, dim=8, bs=2)
    # second run over the same list must not recompute anything
    calls.clear()
    out2 = extract_embeddings(files, teacher, tmp_path, dim=8, bs=2)
    assert calls == []
    assert np.array_equal(np.load(out2 / "emb.npy"), np.load(out / "emb.npy"))
