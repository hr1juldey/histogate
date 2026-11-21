"""P16-S3: dump configs/stain_calib.json — measured on real NONORM slides.

Run: uv run python scripts/dump_stain_calib.py
Bounded sample (400 slides across 9 classes); deterministic path order.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from histogate.stainaug import calibrate_from_scanners  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NONORM = ROOT / "data" / "raw" / "NCT-CRC-HE-100K-NONORM"


def main() -> None:
    if not NONORM.exists():
        raise SystemExit(f"NONORM dataset missing: {NONORM}")
    paths: list[Path] = []
    for cls in sorted(p for p in NONORM.iterdir() if p.is_dir()):
        paths += sorted(cls.glob("*.tif"))[:45]
    params = calibrate_from_scanners(paths[:400])
    out = ROOT / "configs" / "stain_calib.json"
    out.write_text(params.to_json())
    print(f"wrote {out} from {len(paths[:400])} real NONORM slides")


if __name__ == "__main__":
    main()
