#!/usr/bin/env bash
# HistoGate overnight launch (2026-11-20).
# Measured basis: 15.5k-patch epoch = 3.1 min on RTX 3060 workers=0
# -> full 100k epoch ~= 20 min -> 30 epochs ~= 10 h.
# Resumable: rerun with same --name after any crash; completed epochs are
# skipped (state.json + last.ckpt.pt).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-overnight-30ep}"
EPOCHS="${2:-30}"

echo "[launch] run=$NAME epochs=$EPOCHS start=$(date -Is)"
echo "[launch] MLflow live charts: http://127.0.0.1:5000 (experiment: histogate)"
echo "[launch] resume-safe: state.json + last.ckpt.pt written every epoch"

uv run histogate-train \
  --config configs/default.yaml \
  --epochs "$EPOCHS" \
  --val-subset 2048 \
  --grad-checkpoint \
  --resume-if-present \
  --name "$NAME" \
  2>&1 | tee -a "runs/${NAME}.log"
echo "[launch] done $(date -Is) — artifacts: runs/${NAME}/"
