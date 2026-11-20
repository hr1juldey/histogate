"""Fit temperature (and energy-OOD threshold) on CRC-VAL logits.

Post-training calibration: the raw ECE (~0.07) exceeds the 0.05 gate; a
single scalar T fitted by NLL minimization on EXTERNAL-cohort logits fixes
overconfidence without touching weights. Saves calibration.json next to the
checkpoint — the gate and report read it, weights stay frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from histogate.calibration import compute_ece, energy_score, fit_energy_threshold, fit_temperature
from histogate.cli import _balanced_subset, _cache_for, build_index, load_settings
from histogate.data import CachedPatches
from histogate.model import N_CLASSES, PatchClassifier, PatchDataset

CKPT = Path("runs/overnight-30ep/best.ckpt.pt")
SUBSET = 7000  # same balanced subset as the eval run -> same 5839 patches

settings = load_settings(None)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
model.load_state_dict(ckpt["state_dict"])
model.to("cuda" if torch.cuda.is_available() else "cpu")

val_idx = _balanced_subset(build_index(settings.paths.data_dir / "raw" / "CRC-VAL-HE-7K"), SUBSET)
cache = _cache_for(val_idx, settings.paths.data_dir / "cache" / f"crcval_{len(val_idx)}.bin")
ds = PatchDataset(CachedPatches(cache, val_idx, 224), train=False)

# collect logits at temperature 1.0 (evaluate() applies no temperature;
# we re-run the forward pass manually to keep raw z)
model.eval()
dev = next(model.parameters()).device
zs, ys = [], []
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=256, num_workers=0):
        x = x.to(dev).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            z = model(x)
        zs.append(z.float().cpu())
        ys.append(y)
z = torch.cat(zs).numpy()
y = torch.cat(ys).numpy()
print(f"logits: {z.shape}, classes present: {len(set(y.tolist()))}")

T = fit_temperature(z, y)
ece_raw = compute_ece(z, y)
ece_cal = compute_ece(z / T, y)
energy = energy_score(z)
e_thr = fit_energy_threshold(energy, target_fpr=0.05)

out = {
    "checkpoint": str(CKPT),
    "git_sha": ckpt.get("git_sha"),
    "n_fit": int(len(y)),
    "temperature": float(T),
    "ece_before": float(ece_raw),
    "ece_after": float(ece_cal),
    "energy_threshold_fpr05": float(e_thr),
}
Path("runs/overnight-30ep/calibration.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
