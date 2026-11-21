---
license: other
library_name: pytorch
tags:
- pathology
- histopathology
- colorectal-cancer
- tissue-classification
- computational-pathology
---

# HistoGate patch classifier (nonorm-ft2)

9-class H&E tissue patch classifier (NCT-CRC-HE-100K classes) with
temperature calibration and energy-based OOD gating. ConvNeXt-Tiny backbone,
trained on stain-normalized NCT-CRC-HE-100K, then one low-LR epoch on the
unnormalized NCT-CRC-HE-100K-NONORM (6 scanners) for stain robustness.

## Measured performance (external CRC-VAL-HE-7K, n=5839)

- macro-F1 0.9713 / accuracy 0.9764 / quadratic kappa 0.9854
- Raw-scanner domain (NONORM, n=7000): macro-F1 0.9554
- Post-temperature-scaling ECE 0.0055 (T=0.687)
- Energy OOD AUROC 0.808 (global corruption kinds 0.848);
  see repo docs for the documented scope limitation on local blemishes
- int8 dynamic quantization: argmax agreement 0.9980,
  TSR drift 0.0, TIL drift 0.0

Full tables and the honest gate report: `docs/RESULTS.md` in the GitHub repo.

## Intended use

Research only. Input: 224x224 H&E RGB patches. Output: 9-class logits +
temperature 0.687 for calibrated probabilities; energy
threshold -3.718 (5% FPR) for OOD gating.

## Training data

NCT-CRC-HE-100K (+ NONORM variant), Kather et al. 2019. Dataset licenses
apply; no dataset data is redistributed in this repo.
