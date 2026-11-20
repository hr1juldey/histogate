#!/usr/bin/env bash
# Morning after fine-tune: rerun the full measurement battery on a run's best ckpt.
# Usage: bash scripts/morning_eval.sh runs/nonorm-ft
set -euo pipefail
RUN="${1:-runs/nonorm-ft}"
export HISTOGATE_RUN="$RUN"
FILT='Warning|pynvml|mlflow.agent|quantiz|Collecting|version_converter|_c_api|call_onnx|converted|shape_infer|torch.onnx|Failed to convert'

cd "$(dirname "$0")/.."

echo "== [1/6] external eval (CRC-VAL) =="
uv run histogate-eval --checkpoint "$RUN/best.ckpt.pt" --subset 7000 2>&1 | grep -vE "$FILT"
cp runs/eval_best.ckpt.json "$RUN/eval_best.json"

echo "== [2/6] calibration =="
uv run python scripts/fit_calibration.py 2>&1 | grep -vE "$FILT"

echo "== [3/6] domain (NONORM) =="
uv run python scripts/eval_domain.py 2>&1 | grep -vE "$FILT"

echo "== [4/6] OOD energy =="
uv run python scripts/eval_ood.py 2>&1 | grep -vE "$FILT"

echo "== [5/6] export + bench =="
uv run python scripts/run_export_bench.py 2>&1 | grep -vE "$FILT"

echo "== [6/6] int8 drift + GATE =="
uv run python scripts/eval_int8_drift.py 2>&1 | grep -vE "$FILT"
uv run python scripts/make_gate_inputs.py "$RUN"
