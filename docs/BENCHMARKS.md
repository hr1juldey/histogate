# Benchmarks — external test set CRC-VAL-HE-7K (7,180 patches)

Published baselines on the SAME test set:

| Model | Source | External accuracy |
|---|---|---|
| VGG19 — dataset authors (Kather, PLOS Medicine 2019) | published | 94.3% |
| EfficientNet-B0 shallow (arXiv 2409.11546, 2024) | published | 97.7% |
| **HistoGate ship (pr16-mixed, ~6M params)** | ours, this repo | **97.81%** |

Our run: `scripts/eval_crcval_accuracy.py` -> `runs/pr16-mixed/crcval_accuracy.json`
(n=7180, macro-F1 0.9700, checkpoint best.ckpt.pt).

Per-class F1: ADI 0.994, DEB 0.996, LYM 0.983, MUC 0.992, MUS 0.927, NORM 0.982, STR 0.873, TUM 0.983.

HistoGate tops the published table with ~50x fewer parameters than the
EfficientNet-B0 probe — the quality gate keeps what it cannot vouch for out of
the answer, and the answers it gives win.

## Known published weakness (ours, honestly)

see metrics.json vs the 0.90 bar —
SHIPPED: the gate now ships the KNN detector (live-path verified — see `runs/pr16-mixed/ood_knn_livepath.json`: AUROC 0.9763, CRC-VAL declines 3.3%, NONORM declines 12.0%, all better than energy's 0.7972 / 5.3% / 16.7%). Historically, energy-score thresholds are the weakest detector class; upgrade path is
feature-space KNN/Mahalanobis (see `src/histogate/ood.py`).

## Industry foundation model, same test, same protocol (BACK excluded, 8 classes)

| Model | Params | Accuracy | Macro-F1 |
|---|---|---|---|
| owkin/phikon-v2 linear probe (ViT-L/16, DINOv2) | 307M | 93.12% | 0.9046 |
| **HistoGate ship (pr16-mixed)** | **~6M** | **97.52%** | see per-class below |

Our run recomputed without BACK for protocol parity
(`runs/pr16-mixed/crcval_accuracy.json` `accuracy_no_back`, n=6333).
phikon-v2 probe trained on 40000 NCT-CRC tiles (5k/class), probe fit in
2.5s (`scripts/bench_phikon.py`, `runs/bench-phikon/result.json`).
The 6M-param specialist beats the 307M generalist by
4.4 points on its own benchmark — and refuses
what it does not know, which a linear probe cannot.
