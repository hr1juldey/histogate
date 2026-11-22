"""Coverage-vs-accuracy curves on CRC-VAL (P0a A5).

Two rejection axes, one script:
  (a) confidence rejection — decline when max-prob < tau (sweep tau);
  (b) KNN familiarity rejection — the LIVE gate (decline when knn score > thr);
      swept by scaling the FPR-05 threshold.
At each operating point: coverage %, accuracy, macro-F1 over ANSWERED tiles.

Writes runs/<RUN>/coverage.json (numbers for docs/BENCHMARKS.md — no
hand-typed numbers in docs, existing law).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histogate.cli import _cache_for, build_index, load_settings
from histogate.data import CachedPatches
from histogate.metrics import macro_f1
from histogate.model import N_CLASSES, PatchClassifier, PatchDataset

RUN = Path(os.environ.get("HISTOGATE_RUN", "runs/pr16-mixed"))
CKPT = RUN / "best.ckpt.pt"


def main() -> None:
    settings = load_settings(None)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    calib = json.loads((RUN / "calibration.json").read_text())
    temperature = calib["temperature"]

    idx = build_index(settings.paths.data_dir / "raw" / "CRC-VAL-HE-7K")
    cache = _cache_for(idx, settings.paths.data_dir / "cache" / f"crcval_{len(idx)}.bin")
    ds = PatchDataset(CachedPatches(cache, idx, 224), train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    # KNN gate scores on the LIVE path (model.backbone, same as patch_demo)
    from histogate.ood import KNNDetector

    bank = torch.load(RUN / "ood_knn_bank.pt", map_location="cpu", weights_only=True)
    knn = KNNDetector(bank["bank"], k=int(bank["k"]))
    knn_thr = float(calib.get("knn_threshold_fpr05", -0.9163))

    zs, ys, ks = [], [], []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            zs.append(model(x).float().cpu())
            ys.append(y)
            ks.append(torch.from_numpy(knn.score(model.backbone(x).cpu())))
    z = torch.cat(zs).numpy()
    y = torch.cat(ys).numpy()
    k = torch.cat(ks).numpy()

    zz = z / temperature
    zz = zz - zz.max(axis=1, keepdims=True)
    probs = np.exp(zz)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(1)
    conf = probs.max(1)
    print(f"tiles={len(y)} acc={float((pred == y).mean()):.4f} "
          f"knn_decline@fpr05={float((k > knn_thr).mean()) * 100:.1f}%")

    def point(keep: np.ndarray) -> dict:
        if keep.sum() == 0:
            return {"coverage": 0.0, "accuracy": None, "macro_f1": None,
                    "n": 0}
        return {
            "coverage": round(float(keep.mean()), 4),
            "accuracy": round(float((pred[keep] == y[keep]).mean()), 4),
            "macro_f1": round(macro_f1(y[keep], pred[keep]), 4),
            "n": int(keep.sum()),
        }

    conf_curve = []
    for cov_target in (1.0, 0.98, 0.95, 0.90, 0.85, 0.80):
        # conf is top-loaded (most tiles ~0.98): to KEEP cov_target of
        # tiles, tau must be the (1 - cov_target)-th percentile.
        tau = 0.0 if cov_target >= 1.0 else float(np.quantile(conf, 1.0 - cov_target))
        conf_curve.append({"target_coverage": cov_target, "tau": round(tau, 4),
                           **point(conf >= tau)})
    knn_curve = []
    for decline_rate in (0.033, 0.05, 0.10, 0.20):
        # sweep by DECLINE rate on the familiarity score itself (threshold
        # is negative — multiplicative scaling inverts direction)
        thr = float(np.quantile(k, 1.0 - decline_rate))
        knn_curve.append({"decline_rate": decline_rate, "thr": round(thr, 4),
                          **point(k <= thr)})
    knn_curve.insert(0, {"decline_rate": "live (fpr05-calibrated)",
                         "thr": round(knn_thr, 4), **point(k <= knn_thr)})

    out = RUN / "coverage.json"
    out.write_text(json.dumps({
        "n": int(len(y)),
        "confidence_rejection": conf_curve,
        "knn_rejection": knn_curve,
        "note": "accuracy/macro_f1 over ANSWERED tiles only; coverage = "
                "fraction answered; knn scale 1.0 = the live gate",
    }, indent=2))
    print(f"wrote {out}")
    for row in conf_curve:
        print("conf", row["target_coverage"], "->", row["coverage"],
              row["accuracy"], row["macro_f1"])
    for row in knn_curve:
        print("knn", row["decline_rate"], "->", row["coverage"],
              row["accuracy"], row["macro_f1"])


if __name__ == "__main__":
    main()
