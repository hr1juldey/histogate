"""Assemble GateInputs JSON from eval artifacts, then run histogate-gate.

Usage: uv run python scripts/make_gate_inputs.py RUN_DIR
e.g.  uv run python scripts/make_gate_inputs.py runs/nonorm-ft
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/overnight-30ep")
bench = json.loads((run / "onnx" / "export_bench.json").read_text())
domain = json.loads((run / "domain.json").read_text())
calib = json.loads((run / "calibration.json").read_text())
ood = json.loads((run / "ood_sev0.6.json").read_text())
int8 = json.loads((run / "int8_drift.json").read_text())
eval_best = json.loads(Path("runs/eval_best.ckpt.json").read_text())

inputs = {
    "macro_f1": eval_best["val_macro_f1"],
    "macro_f1_domain": domain["nonorm_macro_f1"],
    "ece": calib["ece_after"],
    "ood_auroc": ood["energy_auroc"],
    "tsr_fp32": int8["tsr_fp32_mean"],
    "tsr_int8": int8["tsr_int8_mean"],
    "til_fp32": int8["til_fp32_mean"],
    "til_int8": int8["til_int8_mean"],
    "quant_agreement": int8["argmax_agreement"],
    "patches_per_s": bench["torch_cuda"]["throughput_patches_s"],
    "peak_vram_mb": bench["torch_cuda"].get("peak_vram_mb", 0.0),
}
out = run / "gate_inputs.json"
out.write_text(json.dumps(inputs, indent=2) + "\n")
print(json.dumps(inputs, indent=2))
r = subprocess.run(["uv", "run", "histogate-gate", "--metrics", str(out),
                    "--out", str(run / "gate_report.json")], capture_output=True, text=True)
print(r.stdout, r.stderr, sep="")
sys.exit(r.returncode)
