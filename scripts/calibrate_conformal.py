"""Conformal calibration for the ship tile classifier (P0a).

Rotated 5-fold on CRC-VAL: per fold, qhat on 4/5 of tiles, empirical coverage
and set size on the held-out 1/5; deployed qhat computed on ALL tiles.

HONESTY NOTE (carried into the artifact): the served temperature (one scalar,
NLL fit) was fitted on this same cohort (fit_calibration.py, SUBSET=7000 ⊇
the train-time val 2048 — both `_balanced_subset` round-robin from CRC-VAL).
The conformal guarantee therefore assumes future tiles exchangeable with
CRC-VAL; renew by re-running this script on a fresh cohort.

Output: runs/<RUN>/conformal.json with per-fold coverage + deployed qhats.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histogate.cli import _cache_for, build_index, load_settings
from histogate.conformal import _aps_scores, aps_qhat
from histogate.data import CachedPatches
from histogate.model import N_CLASSES, PatchClassifier, PatchDataset

RUN = Path(os.environ.get("HISTOGATE_RUN", "runs/pr16-mixed"))
CKPT = RUN / "best.ckpt.pt"
ALPHAS = (0.05, 0.10)
FOLDS = 5
SEED = 42


def main() -> None:
    settings = load_settings(None)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    calib = json.loads((RUN / "calibration.json").read_text())
    temperature = calib["temperature"]

    idx = build_index(settings.paths.data_dir / "raw" / "CRC-VAL-HE-7K")
    n = len(idx.entries)
    print(f"[conformal] CRC-VAL entries: {n}")

    cache = _cache_for(idx, settings.paths.data_dir / "cache" / f"crcval_{n}.bin")
    ds = PatchDataset(CachedPatches(cache, idx, 224), train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    zs, ys = [], []
    with torch.no_grad():
        dev = next(model.parameters()).device
        for x, y in loader:
            x = x.to(dev)
            zs.append(model(x).float().cpu())
            ys.append(y)
    z = torch.cat(zs).numpy()
    y = torch.cat(ys).numpy()
    # served probabilities: temperature-scaled softmax (same as PatchAnalyzer)
    zz = z / temperature
    zz = zz - zz.max(axis=1, keepdims=True)
    probs = np.exp(zz)
    probs /= probs.sum(axis=1, keepdims=True)
    print(f"[conformal] logits {z.shape}, acc={float((probs.argmax(1) == y).mean()):.4f}")

    scores = _aps_scores(probs, y)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    folds = np.array_split(perm, FOLDS)
    report: dict = {"folds": [], "alphas": {str(a): None for a in ALPHAS}}
    for a in ALPHAS:
        per_fold = []
        for k in range(FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
            q = aps_qhat(probs[tr], y[tr], alpha=a)
            cov = float((scores[te] <= q + 1e-9).mean())  # membership ⇔ score ≤ qhat
            per_fold.append({"coverage": round(cov, 4), "qhat": round(q, 6),
                             "n_test": int(len(te))})
        report["folds"].append({"alpha": a, "per_fold": per_fold})
        report["alphas"][str(a)] = round(aps_qhat(probs, y, alpha=a), 6)

    mean_cov = {}
    for entry in report["folds"]:
        mean_cov[str(entry["alpha"])] = round(
            float(np.mean([f["coverage"] for f in entry["per_fold"]])), 4)

    out = RUN / "conformal.json"
    out.write_text(json.dumps({
        "method": "split-conformal APS (deterministic), temperature-scaled softmax",
        "n_tiles": n,
        "temperature": temperature,
        "fold_mean_coverage": mean_cov,
        "alphas": report["alphas"],
        "folds": report["folds"],
        "guarantee": "P(true class in set) >= 1 - alpha among gate-accepted tiles, "
                     "conditional on exchangeability with CRC-VAL-HE-7K",
        "disclosure": "served temperature (1 scalar, NLL) was fitted on this same "
                      "cohort (fit_calibration.py); renew by re-calibrating on a "
                      "fresh cohort",
        "ckpt": str(CKPT),
        "seed": SEED,
    }, indent=2))
    print(f"[conformal] fold-mean coverage: {mean_cov}")
    print(f"[conformal] deployed qhats: {report['alphas']}")
    print(f"[conformal] wrote {out}")


if __name__ == "__main__":
    main()
