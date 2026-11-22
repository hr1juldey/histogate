"""Stage A (plan C3): linear-probe baseline on the cached teacher embeddings.

Extends runs/bench-phikon (40k train subset) to FULL NCT-CRC-HE-100K, same
8-class no-BACK protocol, same sorted file order (bench law: comparable).

Checks:
  - fp16-cache cross-check (edge case 9): probe on the SAME 40k subset +
    CRC-VAL val must land within 0.2 pt of the committed fp32 bench
    (0.9312 accuracy, runs/bench-phikon/result.json) — validates the
    bf16-extracted fp16 cache.
  - full-100k probe: the teacher's best linear baseline.
  - NONORM 20k holdout: stain-perturbed cohort accuracy.

Writes runs/distill/probe_result.json. Evaluation-only (phikon-v2 is
non-commercial — no distillation use; see docs/BENCHMARKS.md P1 license log).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs/distill/phikon-v2"
BENCH = json.loads((ROOT / "runs/bench-phikon/result.json").read_text())


def load(name: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    idx = json.loads((CACHE / name / "files.json").read_text())
    emb = np.load(CACHE / name / "emb.npy").astype(np.float32)
    y = np.asarray(idx["labels"])
    classes = idx["classes"]
    files = idx["files"]
    return emb, y, classes, files


def macro_f1(pred: np.ndarray, y: np.ndarray, n: int) -> float:
    f1s = []
    for i in range(n):
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        fn = int(((pred != i) & (y == i)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def bench_protocol_subset(emb, y, files, classes, per_class=5000):
    """First `per_class` sorted files per non-BACK class = the bench's 40k."""
    keep = []
    counts = {c: 0 for c in classes}
    for i, (_f, yi) in enumerate(zip(files, y, strict=True)):
        c = classes[yi]
        if c == "BACK" or counts[c] >= per_class:
            continue
        counts[c] += 1
        keep.append(i)
    keep = np.asarray(keep)
    # remap labels to the bench's 8-class sorted order (BACK dropped)
    eight = [c for c in classes if c != "BACK"]
    remap = {classes.index(c): eight.index(c) for c in eight}
    y8 = np.asarray([remap[int(v)] for v in y[keep]])
    return emb[keep], y8, eight


def main() -> None:
    tr_emb, tr_y, tr_classes, tr_files = load("nct")
    va_emb, va_y, va_classes, va_files = load("crcval")

    # bench-comparable val (no BACK, same file order as the committed bench)
    va8 = [i for i, _f in enumerate(va_files)
           if va_classes[int(va_y[i])] != "BACK"]
    eight = [c for c in va_classes if c != "BACK"]
    remap_va = {va_classes.index(c): eight.index(c) for c in eight}
    va_x = va_emb[va8]
    va_y8 = np.asarray([remap_va[int(va_y[i])] for i in va8])
    print(f"train cache {tr_emb.shape}, val {va_x.shape} (bench: 6333)")

    out: dict = {"model": BENCH["model"], "cache": "runs/distill/phikon-v2 (bf16 extract, fp16 store)"}

    # 1. fp16 cross-check on the bench's exact 40k subset
    bx, by, _ = bench_protocol_subset(tr_emb, tr_y, tr_files, tr_classes)
    clf = LogisticRegression(max_iter=2000, n_jobs=8, C=1.0)
    clf.fit(bx, by)
    acc40 = float((clf.predict(va_x) == va_y8).mean())
    out["bench40k_accuracy"] = round(acc40, 4)
    out["bench40k_vs_fp32_delta"] = round(acc40 - BENCH["accuracy"], 4)
    print(f"40k-subset probe: {acc40:.4f} (fp32 bench {BENCH['accuracy']}, "
          f"delta {acc40 - BENCH['accuracy']:+.4f})")

    # 2. full-100k probe (8-class)
    fx, fy, _ = bench_protocol_subset(
        tr_emb, tr_y, tr_files, tr_classes, per_class=10**9
    )
    clf_full = LogisticRegression(max_iter=2000, n_jobs=8, C=1.0)
    clf_full.fit(fx, fy)
    pred = clf_full.predict(va_x)
    acc_full = float((pred == va_y8).mean())
    out["full100k_accuracy"] = round(acc_full, 4)
    out["full100k_macro_f1"] = round(macro_f1(pred, va_y8, len(eight)), 4)
    out["full100k_train_n"] = int(len(fx))
    print(f"full-100k probe: acc {acc_full:.4f}")

    # 3. NONORM 20k holdout (stain-perturbed cohort, full-100k probe head)
    nn_emb, nn_y, nn_classes, nn_files = load("nonorm")
    nn8 = [i for i, f in enumerate(nn_files)
           if nn_classes[int(nn_y[i])] != "BACK"]
    nn_x = nn_emb[nn8]
    nn_y8 = np.asarray([remap_va[int(nn_y[i])] for i in nn8]) \
        if nn_classes == va_classes else None
    if nn_y8 is None:  # class orders differ -> remap via names
        m = {nn_classes.index(c): eight.index(c) for c in eight}
        nn_y8 = np.asarray([m[int(nn_y[i])] for i in nn8])
    sub = np.random.default_rng(42).choice(len(nn_x), 20000, replace=False)
    acc_nn = float((clf_full.predict(nn_x[sub]) == nn_y8[sub]).mean())
    out["nonorm20k_accuracy"] = round(acc_nn, 4)
    print(f"NONORM 20k holdout: {acc_nn:.4f}")

    out["fp16_crosscheck_pass"] = bool(abs(acc40 - BENCH["accuracy"]) <= 0.002)
    out_path = ROOT / "runs/distill/probe_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
