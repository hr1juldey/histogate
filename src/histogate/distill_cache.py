"""Teacher-embedding cache: resume-safe fp16 memmap writer + reader (plan C1).

Caches foundation-model embeddings ONCE so probe heads and distillation train
on vectors — no large model ever trains on this GPU. Teacher-agnostic: the
caller supplies ``teacher_fn(indices) -> (len(indices), dim) float32``.

Layout under ``out_dir``:
  emb.npy     float16 ndarray (n, dim) — np.save/np.load, mmap-able
  files.json  {"files": [...], "dim": d, "dtype": "float16", "sha": <list sha>}

Resume law: if files.json matches (same sha, dim, count) and emb.npy has the
full shape, the call is a no-op — completed offsets are never recomputed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def files_sha(files: Sequence[str]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode())
        h.update(b"\0")
    return h.hexdigest()


def load_index(out_dir: Path) -> dict:
    return json.loads((out_dir / "files.json").read_text())


def load_embeddings(out_dir: Path, mmap: bool = True) -> np.ndarray:
    return np.load(out_dir / "emb.npy", mmap_mode="r" if mmap else None)


def extract_embeddings(
    files: Sequence[str],
    teacher_fn: Callable[[list[int]], np.ndarray],
    out_dir: Path,
    dim: int,
    bs: int = 32,
    log_every: int = 50,
) -> Path:
    """Write/complete the fp16 embedding cache for ``files``; returns out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = files_sha(files)
    idx_path = out_dir / "files.json"
    emb_path = out_dir / "emb.npy"

    n = len(files)
    if idx_path.exists() and emb_path.exists():
        try:
            idx = load_index(out_dir)
            done = np.load(emb_path, mmap_mode="r")
            if (
                idx.get("sha") == sha
                and int(idx.get("dim", -1)) == dim
                and idx.get("files") == list(files)
                and done.shape == (n, dim)
            ):
                return out_dir  # fully cached — resume no-op
        except Exception:
            pass  # corrupt cache: rebuild below

    emb = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float16, shape=(n, dim)
    )
    start = 0  # per-batch writes are small; simplicity > partial-batch resume
    for i in range(start, n, bs):
        batch = list(range(i, min(i + bs, n)))
        vecs = np.asarray(teacher_fn(batch), dtype=np.float32)
        if vecs.shape != (len(batch), dim):
            raise ValueError(
                f"teacher returned {vecs.shape}, expected ({len(batch)}, {dim})"
            )
        emb[i : i + len(batch)] = vecs.astype(np.float16)
        if (i // bs) % log_every == 0:
            emb.flush()
    emb.flush()
    idx_path.write_text(
        json.dumps(
            {
                "files": list(files),
                "dim": dim,
                "dtype": "float16",
                "sha": sha,
                "n": n,
            }
        )
    )
    return out_dir
