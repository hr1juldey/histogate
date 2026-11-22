"""P0b B4: VRAM co-residency measurement — ship model + NucleusCounter.

Runs the deployed inference path (LockedEngine analyze) plus 20 nucleus
count() calls on real CRC-VAL tiles, then reports torch peak allocation.
Abort law (plan B4): > 4 GiB peak means the stage cannot ship on this box.
Writes runs/pr16-mixed/cells_vram.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/pr16-mixed/cells_vram.json"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU — run on the 3060 box")
    torch.cuda.reset_peak_memory_stats()

    from histogate.serve.patch_demo import PatchAnalyzer

    ckpt = ROOT / "runs/pr16-mixed/best.ckpt.pt"
    calib = ROOT / "runs/pr16-mixed/calibration.json"
    an = PatchAnalyzer(ckpt, calib)
    an.model.to("cuda")
    an.cells_mode = True  # what the served process WOULD do with the flag on

    from histogate.cells import NucleusCounter

    nc = NucleusCounter(device="cuda")
    tiles = sorted((ROOT / "data/raw/CRC-VAL-HE-7K/TUM").glob("*.tif"))[:20]
    t0 = time.time()
    ship_peak = None
    per = []
    for t in tiles:
        img = Image.open(t).convert("RGB")
        r = an.analyze(img)
        c = nc.count(img)
        per.append(sum(c.values()))
        if ship_peak is None:
            ship_peak = torch.cuda.max_memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    res = {
        "stage": "ship analyzer + nucleus counter co-resident",
        "n_tiles": len(tiles),
        "ship_model_peak_gib": round(ship_peak, 2),
        "peak_gib": round(peak, 2),
        "gpu_total_gib": round(total, 2),
        "count_s_per_tile": round((time.time() - t0) / len(tiles), 2),
        "mean_nuclei_per_tile": round(sum(per) / len(per), 1),
        "abort_threshold_gib": 4.0,
        "passed": bool(peak <= 4.0),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
