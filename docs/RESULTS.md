# Measured results (generated — do not edit by hand)

_Rendered by `scripts/render_readme_numbers.py`. Source: `docs/results-snapshot`. A `MISSING` cell means the artifact file does not exist.

| run | CRC-VAL macro-F1 | CRC-VAL acc | kappa | NONORM F1 | domain drop | OOD AUROC | ECE (calibrated) | gate |
|---|---|---|---|---|---|---|---|---|
| v1 base (normalized only) | 0.9741 | 0.9784 | 0.9872 | 0.7842 | 0.1898 | 0.8282 | 0.0041 | 6/9 FAIL |
| ft1 (1 high-LR NONORM epoch) | 0.9637 | 0.9707 | 0.9833 | 0.9710 | -0.0073 | 0.6590 | 0.0050 | 7/9 FAIL |
| ft2 SHIP (1 low-LR NONORM epoch) | 0.9713 | 0.9764 | 0.9854 | 0.9554 | 0.0158 | 0.8077 | 0.0055 | 6/9 FAIL |

## Int8 deployment integrity

| run | argmax agree | TSR drift max | TIL rel drift max | int8 size MB | throughput torch-CUDA p/s | VRAM MB |
|---|---|---|---|---|---|---|
| v1 base (normalized only) | 0.996094 | 0.0000 | 0.0000 | 28.7 | 482.3 | 554.5 |
| ft1 (1 high-LR NONORM epoch) | 0.998047 | 0.0000 | 0.0000 | 28.7 | 468.2 | 554.5 |
| ft2 SHIP (1 low-LR NONORM epoch) | 0.998047 | 0.0000 | 0.0000 | 28.7 | 475.2 | 554.5 |

## Gate detail (last battery per run)

### overnight-30ep
| check | value | comparator | threshold | passed |
|---|---|---|---|---|
| macro_f1 | 0.974063 | >= | 0.93 | PASS |
| f1_domain_drop | 0.189863 | <= | 0.015 | FAIL |
| ece | 0.004093 | <= | 0.05 | PASS |
| ood_auroc | 0.8282 | >= | 0.9 | FAIL |
| tsr_drift | 0.0 | <= | 0.05 | PASS |
| til_drift_rel | 0.0 | <= | 0.1 | PASS |
| quant_agreement | 0.996094 | >= | 0.999 | FAIL |
| throughput | 482.3 | >= | 40.0 | PASS |
| vram | 554.5 | <= | 2048.0 | PASS |

### nonorm-ft
| check | value | comparator | threshold | passed |
|---|---|---|---|---|
| macro_f1 | 0.963691 | >= | 0.93 | PASS |
| f1_domain_drop | -0.007309 | <= | 0.015 | PASS |
| ece | 0.004962 | <= | 0.05 | PASS |
| ood_auroc | 0.659 | >= | 0.9 | FAIL |
| tsr_drift | 0.0 | <= | 0.05 | PASS |
| til_drift_rel | 0.0 | <= | 0.1 | PASS |
| quant_agreement | 0.998047 | >= | 0.999 | FAIL |
| throughput | 468.2 | >= | 40.0 | PASS |
| vram | 554.5 | <= | 2048.0 | PASS |

### nonorm-ft2
| check | value | comparator | threshold | passed |
|---|---|---|---|---|
| macro_f1 | 0.971264 | >= | 0.93 | PASS |
| f1_domain_drop | 0.015864 | <= | 0.015 | FAIL |
| ece | 0.005467 | <= | 0.05 | PASS |
| ood_auroc | 0.8077 | >= | 0.9 | FAIL |
| tsr_drift | 0.0 | <= | 0.05 | PASS |
| til_drift_rel | 0.0 | <= | 0.1 | PASS |
| quant_agreement | 0.998047 | >= | 0.999 | FAIL |
| throughput | 475.2 | >= | 40.0 | PASS |
| vram | 554.5 | <= | 2048.0 | PASS |

