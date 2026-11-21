"""P17-S3: ensemble battery — CRC-VAL macro-F1, NONORM domain macro-F1,
throughput. Acceptance: CRC-VAL >= 0.975, NONORM >= 0.95, >= 200 p/s.

Mirrors morning_eval.sh stages 1/3 (accuracy + domain) plus a throughput
probe. Reports both tta=False (2 forwards/patch) and tta=True (8
forwards/patch) so the serve default can be chosen on measurement.
Writes runs/ensemble/battery.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from histogate.cli import _balanced_subset, build_index, load_settings
from histogate.data import CLASS_TO_IDX
from histogate.metrics import macro_f1
from histogate.serve.ensemble import EnsembleEngine

CKPTS = ["runs/pr16-mixed/best.ckpt.pt", "runs/overnight-30ep/best.ckpt.pt"]
CRCVAL = Path("data/raw/CRC-VAL-HE-7K")
NONORM_N = 7000
THROUGHPUT_N = 512

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load(paths, device):
    x = [tifffile.imread(p)[..., :3] for p in paths]
    t = torch.from_numpy(np.stack(x).transpose(0, 3, 1, 2)).float() / 255
    return ((t - _MEAN) / _STD).to(device)


def _f1(engine, paths, labels):
    preds = []
    for i in range(0, len(paths), 128):
        preds.append(engine.predict_labels(paths[i:i + 128]).cpu())
    p = torch.cat(preds).numpy()
    return float(macro_f1(labels, p))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = load_settings(None)
    out: dict = {"ckpts": CKPTS}

    # CRC-VAL (external)
    crc_paths = [p for c in sorted(CRCVAL.iterdir()) if c.is_dir()
                 for p in sorted(c.glob("*.tif"))]
    crc_labels = [CLASS_TO_IDX[p.parent.name] for p in crc_paths]

    # NONORM domain (balanced subset, mirrors eval_domain.py)
    nn_idx = _balanced_subset(
        build_index(settings.paths.data_dir / "raw" / "NCT-CRC-HE-100K-NONORM"),
        NONORM_N)
    nn_paths = [e.path for e in nn_idx.entries]
    nn_labels = [CLASS_TO_IDX[e.label] for e in nn_idx.entries]

    for tta in (False, True):
        engine = EnsembleEngine(CKPTS, tta=tta)
        for m in engine.models:
            m.to(device)
        tag = "tta" if tta else "notta"

        out[f"crcval_f1_{tag}"] = _f1(
            engine, _load(crc_paths, device), crc_labels)
        out[f"nonorm_f1_{tag}"] = _f1(
            engine, _load(nn_paths, device), nn_labels)

        xb = _load(crc_paths[:THROUGHPUT_N], device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.predict_proba(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        out[f"throughput_{tag}"] = THROUGHPUT_N / (time.perf_counter() - t0)
        print(f"[{tag}] crcval={out[f'crcval_f1_{tag}']:.4f} "
              f"nonorm={out[f'nonorm_f1_{tag}']:.4f} "
              f"thr={out[f'throughput_{tag}']:.1f} p/s")

    Path("runs/ensemble").mkdir(exist_ok=True)
    (Path("runs/ensemble") / "battery.json").write_text(json.dumps(out, indent=2))
    ok = (out["crcval_f1_notta"] >= 0.975 and out["nonorm_f1_notta"] >= 0.95
          and out["throughput_notta"] >= 200)
    print("ACCEPTANCE(notta):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
