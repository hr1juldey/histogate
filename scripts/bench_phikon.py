"""phikon-v2 (owkin ViT-L, DINOv2, 307M params) linear-probe benchmark on
NCT-CRC-HE-100K -> CRC-VAL-HE-7K — the open industry foundation model on the
same test, same protocol, as HistoGate's 6M-param model.

Writes runs/bench-phikon/result.json (accuracy + macro-F1 + timing).
Embeddings are checkpointed so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/NCT-CRC-HE-100K"
VAL = ROOT / "data/raw/CRC-VAL-HE-7K"
OUTDIR = ROOT / "runs/bench-phikon"


def embed(model, proc, files: list[Path], bs: int, dev: str) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(files), bs), desc="embed", unit="batch"):
        imgs = [Image.open(f).convert("RGB") for f in files[i:i + bs]]
        x = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            z = model(**x).last_hidden_state[:, 0]  # CLS
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5000,
                    help="train tiles per class (9000 total = 50k)")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)

    from transformers import AutoImageProcessor, AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2").to(dev).eval()

    def files_for(root: Path, per_class: int | None) -> tuple[list[Path], np.ndarray, list[str]]:
        classes = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != "BACK")
        fs, ys = [], []
        for ci, c in enumerate(classes):
            cand = sorted((root / c).glob("*.tif"))[:per_class]
            fs.extend(cand)
            ys.extend([ci] * len(cand))
        return fs, np.array(ys), classes

    tr_files, tr_y, classes = files_for(TRAIN, a.per_class)
    va_files, va_y, _ = files_for(VAL, None)

    tr_cache, va_cache = OUTDIR / "train_emb.npy", OUTDIR / "val_emb.npy"
    if tr_cache.exists() and va_cache.exists():
        tr_x = np.load(tr_cache)
        va_x = np.load(va_cache)
    else:
        t0 = time.time()
        tr_x = embed(model, proc, tr_files, a.bs, dev)
        va_x = embed(model, proc, va_files, a.bs, dev)
        np.save(tr_cache, tr_x)
        np.save(va_cache, va_x)
        print(f"embedded {len(tr_x)}+{len(va_x)} in {time.time()-t0:.0f}s")

    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, n_jobs=8, C=1.0)
    clf.fit(tr_x, tr_y)
    pred = clf.predict(va_x)
    acc = float((pred == va_y).mean())
    f1s = []
    for i in range(len(classes)):
        tp = int(((pred == i) & (va_y == i)).sum())
        fp = int(((pred == i) & (va_y != i)).sum())
        fn = int(((pred != i) & (va_y == i)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    macro = float(np.mean(f1s))
    res = {
        "model": "owkin/phikon-v2 (ViT-L/16, 307M, linear probe)",
        "train_n": int(len(tr_x)), "val_n": int(len(va_x)),
        "accuracy": round(acc, 4), "macro_f1": round(macro, 4),
        "per_class_f1": {c: round(f, 4) for c, f in zip(classes, f1s, strict=True)},
        "probe_fit_s": round(time.time() - t0, 1),
    }
    (OUTDIR / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:600])


if __name__ == "__main__":
    main()
