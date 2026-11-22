"""Extract teacher embeddings to the fp16 cache (plan C2) — resume-safe.

Teacher-agnostic: --model takes any HF vision encoder whose forward returns
last_hidden_state (CLS-token embedding). LICENSE LAW: phikon-v2 (owkin) is
NON-COMMERCIAL — its cache is built for EVALUATION/PROBE BENCHMARKS only
(same protocol as runs/bench-phikon, research use); it must NOT be used as a
distillation teacher for shipped weights. A permissively licensed teacher
(Apache/MIT, e.g. histai/hibou-l once gate access is granted) uses this same
script for the distillation cache.

Datasets (all tiles, BACK excluded, sorted for determinism, seed 42):
  nct     data/raw/NCT-CRC-HE-100K      100,000 tiles
  nonorm  data/raw/NCT-CRC-HE-100K-NONORM 100,000 tiles
  crcval  data/raw/CRC-VAL-HE-7K          7,180 tiles  (TEST-ONLY — never fit)

Output per dataset: runs/distill/<name>/emb.npy (n, dim) fp16 + files.json
(with labels + class order). Run: uv run python scripts/extract_teacher.py
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from histogate.distill_cache import extract_embeddings

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "nct": ROOT / "data/raw/NCT-CRC-HE-100K",
    "nonorm": ROOT / "data/raw/NCT-CRC-HE-100K-NONORM",
    "crcval": ROOT / "data/raw/CRC-VAL-HE-7K",
}


def files_for(root: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Sorted tile list + labels; ALL classes incl. BACK (distillation needs the
    full 9-class task; probe scripts filter BACK to match the bench protocol)."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    files, ys = [], []
    for ci, c in enumerate(classes):
        cand = sorted((root / c).glob("*.tif"))
        files.extend(str(p.relative_to(ROOT)) for p in cand)
        ys.extend([ci] * len(cand))
    return files, np.asarray(ys), classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owkin/phikon-v2")
    ap.add_argument("--datasets", default="nct,nonorm,crcval")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--tag", default=None, help="cache subname (defaults to model id)")
    a = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    out_root = ROOT / a.out / (a.tag or a.model.split("/")[-1])
    out_root.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoImageProcessor, AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()
    dim = model.config.hidden_size

    def teacher(batch: list[int]) -> np.ndarray:
        imgs = [Image.open(ROOT / files[i]).convert("RGB") for i in batch]
        x = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            z = model(**x).last_hidden_state[:, 0]
        return z.float().cpu().numpy()

    for name in a.datasets.split(","):
        root = DATASETS[name]
        files, ys, classes = files_for(root)
        t0 = time.time()
        out_dir = extract_embeddings(
            files, teacher, out_root / name, dim=dim, bs=a.bs
        )
        from histogate.distill_cache import load_index

        idx = load_index(out_dir)
        idx["labels"] = [int(y) for y in ys]
        idx["classes"] = classes
        idx["model"] = a.model
        (out_dir / "files.json").write_text(json.dumps(idx))
        print(
            f"{name}: {idx['n']} tiles x {dim}d cached at {out_dir} "
            f"({time.time() - t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
