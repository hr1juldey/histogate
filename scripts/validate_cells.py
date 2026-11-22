"""P0b B5: colon sanity validation of the nucleus counter on CRC-VAL tiles.

Plausibility bars (plan B5), 100 tiles stratified over the 8 tissue classes
(11-12 each) + 12 BACK tiles for the empty-tile edge case:
  - TUM: epithelial lineage (neoplastic + nonneop_epithelial) modal in >= 60%
  - NORM: epithelial lineage modal in >= 50%
  - STR: connective modal in >= 60%
  - all tiles: total nuclei in [20, 8000] (biological range at this scale)
Writes runs/pr16-mixed/cells_validation.json. Any FAIL -> flag stays OFF.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/raw/CRC-VAL-HE-7K"
OUT = ROOT / "runs/pr16-mixed/cells_validation.json"
N_PER_CLASS = 12

EPITHELIAL = ("neoplastic", "nonneop_epithelial")


def main() -> None:
    from histogate.cells import NucleusCounter

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    c = NucleusCounter(device=device)
    classes = sorted(d.name for d in VAL.iterdir() if d.is_dir() and d.name != "BACK")
    results: dict[str, list] = {k: [] for k in classes + ["BACK"]}
    totals: list[int] = []

    for cls in classes + ["BACK"]:
        tiles = sorted((VAL / cls).glob("*.tif"))[:N_PER_CLASS]
        for t in tiles:
            counts = c.count(Image.open(t).convert("RGB"))
            total = int(sum(counts.values()))
            fam = Counter()
            for k, v in counts.items():
                if k in EPITHELIAL:
                    fam["epithelial"] += v
                else:
                    fam[k] += v
            modal = fam.most_common(1)[0][0] if fam and total > 0 else "none"
            results[cls].append(
                {"file": t.name, "total": total, "modal": modal, "counts": counts}
            )
            totals.append(total)
            print(f"{cls:5s} {t.name:28s} total={total:4d} modal={modal}", flush=True)

    def frac(cls: str, fam: str) -> float:
        rows = [r for r in results[cls] if r["total"] > 0]
        if not rows:
            return 0.0
        return sum(r["modal"] == fam for r in rows) / len(rows)

    checks = {
        "tum_epithelial_modal_frac": frac("TUM", "epithelial"),
        "tum_bar": 0.60,
        "norm_epithelial_modal_frac": frac("NORM", "epithelial"),
        "norm_bar": 0.50,
        "str_connective_modal_frac": frac("STR", "connective"),
        "str_bar": 0.60,
        "total_min": int(min(totals)),
        "total_max": int(max(totals)),
    }
    passed = (
        checks["tum_epithelial_modal_frac"] >= 0.60
        and checks["norm_epithelial_modal_frac"] >= 0.50
        and checks["str_connective_modal_frac"] >= 0.60
        and checks["total_min"] >= 20
        and checks["total_max"] <= 8000
    )
    OUT.write_text(
        json.dumps(
            {
                "model": "vendored HoVer-Net fast-pannuke (BSD-3 code, "
                "PanNuke-trained weights, 144.1 MiB)",
                "n_tiles": len(totals),
                "per_class": results,
                "checks": checks,
                "passed": bool(passed),
                "device": device,
                "caveat": "single-pass 224->256 resize at native 0.5mpp scale; "
                "checkpoint trained at 0.25mpp — counts are plausibility-grade",
            },
            indent=2,
        )
    )
    print("PASS" if passed else "FAIL", json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
