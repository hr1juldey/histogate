"""Domain-shift eval: best ckpt on NCT-CRC-HE-100K-NONORM (unnormalized,
different scanners) vs CRC-VAL-HE-7K. Gate: domain_drop_max 0.015 macro-F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from histogate.calibration import compute_ece
from histogate.cli import _balanced_subset, _cache_for, build_index, load_settings
from histogate.data import CachedPatches
from histogate.metrics import macro_f1, quadratic_kappa
from histogate.model import N_CLASSES, PatchClassifier, PatchDataset

CKPT = Path("runs/overnight-30ep/best.ckpt.pt")
N = 7000

settings = load_settings(None)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
model.load_state_dict(ckpt["state_dict"])
model.eval().to("cuda")

idx = _balanced_subset(
    build_index(settings.paths.data_dir / "raw" / "NCT-CRC-HE-100K-NONORM"), N)
cache = _cache_for(idx, settings.paths.data_dir / "cache" / f"nonorm_{len(idx)}.bin")
ds = PatchDataset(CachedPatches(cache, idx, 224), train=False)

zs, ys = [], []
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=256, num_workers=0):
        xc = x.to("cuda").to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model(xc)
        zs.append(z.float().cpu())
        ys.append(y)
z = torch.cat(zs).numpy()
y = torch.cat(ys).numpy()
pred = z.argmax(1)

crc = json.loads(Path("runs/eval_best.ckpt.json").read_text())
f1_val = float((pred == y).mean())
f1_crc = crc["val_macro_f1"]
out = {
    "checkpoint": str(CKPT), "n": len(y),
    "nonorm_acc": round(f1_val, 4),
    "nonorm_macro_f1": round(macro_f1(y, pred), 4),
    "nonorm_kappa": round(quadratic_kappa(y, pred), 4),
    "nonorm_ece": round(compute_ece(z, y), 4),
    "crcval_macro_f1": f1_crc,
    "domain_drop": round(f1_crc - macro_f1(y, pred), 4),
    "gate_domain_drop_max": 0.015,
}
Path("runs/overnight-30ep/domain.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
