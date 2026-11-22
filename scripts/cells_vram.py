"""P0b B4: VRAM co-residency measurement — ship model + NucleusCounter.

Mirrors the deployed compute path (ship trunk forward on a preprocessed CRC
tile + 20 nucleus count() calls), reports torch peak allocation.
Abort law (plan B4): > 4 GiB peak means the stage cannot ship on this box.
Writes runs/pr16-mixed/cells_vram.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/pr16-mixed/cells_vram.json"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU — run on the 3060 box")
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()

    from histogate.cells import NucleusCounter
    from histogate.model import N_CLASSES, PatchClassifier

    ckpt = torch.load(ROOT / "runs/pr16-mixed/best.ckpt.pt",
                      map_location="cpu", weights_only=False)
    ship = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    ship.load_state_dict(ckpt["state_dict"])
    ship.to(dev).eval()

    nc = NucleusCounter(device=dev)
    tiles = sorted((ROOT / "data/raw/CRC-VAL-HE-7K/TUM").glob("*.tif"))[:20]

    t0 = time.time()
    per = []
    for t in tiles:
        img = Image.open(t).convert("RGB")
        arr = np.asarray(img.resize((224, 224))) / 255.0
        x = (torch.from_numpy(arr).permute(2, 0, 1).float() - _MEAN) / _STD
        with torch.no_grad():
            _ = ship(x[None].to(dev))          # ship trunk + head
        c = nc.count(img)                       # second model, full path
        per.append(sum(c.values()))
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    res = {
        "stage": "ship analyzer + nucleus counter co-resident",
        "n_tiles": len(tiles),
        "peak_gib": round(peak, 2),
        "gpu_total_gib": round(total, 2),
        "combined_s_per_tile": round((time.time() - t0) / len(tiles), 2),
        "mean_nuclei_per_tile": round(sum(per) / len(per), 1),
        "abort_threshold_gib": 4.0,
        "passed": bool(peak <= 4.0),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
