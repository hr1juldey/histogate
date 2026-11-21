"""P15-S5b: render MODEL_CARD.md for the HF model repo from the committed
results snapshot — same law as RESULTS.md: no hand-typed numbers."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "docs" / "results-snapshot" / "pr16-mixed"


def main() -> None:
    eval_ = json.loads((SNAP / "eval_best.json").read_text())
    dom = json.loads((SNAP / "domain.json").read_text())
    calib = json.loads((SNAP / "calibration.json").read_text())
    ood = json.loads((SNAP / "ood_sev0.6.json").read_text())
    int8 = json.loads((SNAP / "int8_drift.json").read_text())
    text = f"""---
license: other
library_name: pytorch
tags:
- pathology
- histopathology
- colorectal-cancer
- tissue-classification
- computational-pathology
---

# HistoGate patch classifier (pr16-mixed)

9-class H&E tissue patch classifier (NCT-CRC-HE-100K classes) with
temperature calibration and energy-based OOD gating. ConvNeXt-Tiny backbone,
warm-started from the normalized-data base model, then trained on the mixed
200k set: stain-normalized NCT-CRC-HE-100K + unnormalized
NCT-CRC-HE-100K-NONORM (6 scanners) + calibrated physically-grounded stain
augmentation (Beer-Lambert dye-concentration jitter measured on real slides).

## Measured performance (external CRC-VAL-HE-7K, n={eval_['n_val']})

- macro-F1 {eval_['val_macro_f1']:.4f} / accuracy {eval_['val_acc']:.4f} / quadratic kappa {eval_['val_kappa']:.4f}
- Raw-scanner domain (NONORM, n={dom['n']}): macro-F1 {dom['nonorm_macro_f1']:.4f}
- Post-temperature-scaling ECE {calib['ece_after']:.4f} (T={calib['temperature']:.3f})
- Energy OOD AUROC {ood['energy_auroc']:.3f} (global corruption kinds {ood['energy_auroc_global_kinds']:.3f});
  see repo docs for the documented scope limitation on local blemishes
- int8 dynamic quantization: argmax agreement {int8['argmax_agreement']:.4f},
  TSR drift {int8['tsr_abs_drift_max']}, TIL drift {int8['til_rel_drift_max']}

Full tables and the honest gate report: `docs/RESULTS.md` in the GitHub repo.

## Intended use

Research only. Input: 224x224 H&E RGB patches. Output: 9-class logits +
temperature {calib['temperature']:.3f} for calibrated probabilities; energy
threshold {calib['energy_threshold_fpr05']:.3f} (5% FPR) for OOD gating.

## Training data

NCT-CRC-HE-100K (+ NONORM variant), Kather et al. 2019. Dataset licenses
apply; no dataset data is redistributed in this repo.
"""
    out = ROOT / "MODEL_CARD.md"
    out.write_text(text)
    print(f"rendered {out}")


if __name__ == "__main__":
    main()
