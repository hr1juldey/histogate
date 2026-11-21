#!/usr/bin/env bash
# P16-S4: mixed stain-augmented run. NCT (normalized) + NONORM (raw) 200k
# patches, warm-start from v1 best, LR 1e-4 (warm-start law), 10 epochs
# (~60 min/epoch measured: 40 min train + 20 min stain aug).
# Rerun this script to resume (state.json checkpointed every epoch).
set -euo pipefail
cd "$(dirname "$0")/.."
nohup uv run histogate-train \
  --train-dataset NCT-CRC-HE-100K \
  --train-dataset2 NCT-CRC-HE-100K-NONORM \
  --stain-augment \
  --epochs 10 \
  --lr 1e-4 \
  --no-pretrained \
  --init-from runs/overnight-30ep/best.ckpt.pt \
  --val-subset 2048 \
  --grad-checkpoint \
  --resume-if-present \
  --name pr16-mixed \
  >> runs/pr16-mixed.log 2>&1 &
echo "launched pr16-mixed (pid $!) -> runs/pr16-mixed.log"
