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

# HistoGate patch classifier (pr16-mixed)

9-class H&E tissue patch classifier (NCT-CRC-HE-100K classes) with
temperature calibration and KNN feature-space OOD gating. ConvNeXt-Tiny backbone,
warm-started from the normalized-data base model, then trained on the mixed
200k set: stain-normalized NCT-CRC-HE-100K + unnormalized
NCT-CRC-HE-100K-NONORM (6 scanners) + calibrated physically-grounded stain
augmentation (Beer-Lambert dye-concentration jitter measured on real slides).

## Measured performance (external CRC-VAL-HE-7K, n=5839)

- macro-F1 0.9711 / accuracy 0.9764 / quadratic kappa 0.9873
- Raw-scanner domain (NONORM, n=7000): macro-F1 0.9776
- Post-temperature-scaling ECE 0.0064 (T=0.682)
- OOD: KNN feature-space detector (k=50, live-path trunk), AUROC 0.9763 vs Camelyon16; declines 3.3% of CRC-VAL, 12.0% of NONORM (energy was 0.7972 / 5.3% / 16.7%);
  see repo docs for the documented scope limitation on local blemishes
- int8 dynamic quantization: argmax agreement 1.0000,
  TSR drift 0.0, TIL drift 0.0

Full tables and the honest gate report: `docs/RESULTS.md` in the GitHub repo.

## Intended use

Research only. Input: 224x224 H&E RGB patches. Output: 9-class logits +
temperature 0.682 for calibrated probabilities; energy
threshold -3.737 (5% FPR) for OOD gating.

## Training data

NCT-CRC-HE-100K (+ NONORM variant), Kather et al. 2019. Dataset licenses
apply; no dataset data is redistributed in this repo.
