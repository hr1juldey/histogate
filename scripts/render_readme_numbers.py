"""P15-S1: render docs/RESULTS.md from run artifacts. NO hand-typed numbers.

Every cell comes from a JSON on disk. A missing file renders as
`MISSING <file>` in the cell — never a placeholder value (L4/L5: docs must
not be able to claim what was not measured).

Usage:
  uv run python scripts/render_readme_numbers.py            # from runs/
  RESULTS_SOURCE=docs/results-snapshot uv run python ...    # CI snapshot
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("overnight-30ep", "nonorm-ft", "nonorm-ft2", "pr16-mixed")
LABELS = {"overnight-30ep": "v1 base (normalized only)",
          "nonorm-ft": "ft1 (1 high-LR NONORM epoch)",
          "nonorm-ft2": "ft2 (1 low-LR NONORM epoch)",
          "pr16-mixed": "pr16-mixed (mixed 200k + calibrated stain aug)"}


def _read(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def load_run(base: Path) -> dict[str, dict | None]:
    return {
        "eval": _read(base / "eval_best.json"),
        "domain": _read(base / "domain.json"),
        "calib": _read(base / "calibration.json"),
        "ood": _read(base / "ood_sev0.6.json") or _read(base / "ood.json"),
        "int8": _read(base / "int8_drift.json"),
        "bench": _read(base / "onnx" / "export_bench.json"),
        "gate": _read(base / "gate_report.json"),
    }


def _fmt(x: float | None, nd: int = 4) -> str:
    return "MISSING" if x is None else f"{x:.{nd}f}"


def _cell(d: dict | None, key: str, nd: int = 4) -> str:
    if d is None:
        return "MISSING"
    v = d.get(key)
    return _fmt(v, nd) if isinstance(v, (int, float)) else "MISSING"


def render(runs: dict[str, dict]) -> str:
    out = ["# Measured results (generated — do not edit by hand)", "",
           f"_Rendered by `scripts/render_readme_numbers.py`. Source: "
           f"`{os.environ.get('RESULTS_SOURCE', 'runs/')}`. "
           "A `MISSING` cell means the artifact file does not exist.", ""]
    out += ["| run | CRC-VAL macro-F1 | CRC-VAL acc | kappa | NONORM F1 | "
            "domain drop | OOD AUROC | ECE (calibrated) | gate |", "|---|---|---|---|---|---|---|---|---|"]
    for name, d in runs.items():
        gate = d["gate"]
        if gate:
            npass = sum(c["passed"] for c in gate["checks"])
            gcell = f"{npass}/{len(gate['checks'])} {'PASS' if gate['overall_pass'] else 'FAIL'}"
        else:
            gcell = "MISSING"
        out.append(
            f"| {LABELS.get(name, name)} | {_cell(d['eval'], 'val_macro_f1')} "
            f"| {_cell(d['eval'], 'val_acc')} | {_cell(d['eval'], 'val_kappa')} "
            f"| {_cell(d['domain'], 'nonorm_macro_f1')} "
            f"| {_cell(d['domain'], 'domain_drop')} "
            f"| {_cell(d['ood'], 'energy_auroc')} "
            f"| {_cell(d['calib'], 'ece_after')} | {gcell} |")
    out += ["", "## Int8 deployment integrity", "",
            "| run | argmax agree | TSR drift max | TIL rel drift max | "
            "int8 size MB | throughput torch-CUDA p/s | VRAM MB |", "|---|---|---|---|---|---|---|"]
    for name, d in runs.items():
        bench = d["bench"] or {}
        tc = bench.get("torch_cuda", {})
        if not isinstance(tc, dict):
            tc = {}
        tps_cell = f"{tc['throughput_patches_s']}" if "throughput_patches_s" in tc else "MISSING"
        vr_cell = f"{tc['peak_vram_mb']}" if "peak_vram_mb" in tc else "MISSING"
        out.append(
            f"| {LABELS.get(name, name)} | {_cell(d['int8'], 'argmax_agreement', 6)} "
            f"| {_cell(d['int8'], 'tsr_abs_drift_max')} "
            f"| {_cell(d['int8'], 'til_rel_drift_max')} "
            f"| {_cell(d['int8'], 'int8_size_mb', 1)} "
            f"| {tps_cell} | {vr_cell} |")
    out += ["", "## Gate detail (last battery per run)", ""]
    for name, d in runs.items():
        out.append(f"### {name}")
        gate = d["gate"]
        if not gate:
            out.append("MISSING gate_report.json\n")
            continue
        out += ["| check | value | comparator | threshold | passed |", "|---|---|---|---|---|"]
        out += [f"| {c['name']} | {c['value']} | {c['comparator']} | "
                f"{c['threshold']} | {'PASS' if c['passed'] else 'FAIL'} |"
                for c in gate["checks"]]
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    src = ROOT / os.environ.get("RESULTS_SOURCE", "runs")
    runs = {name: load_run(src / name) for name in RUNS}
    (ROOT / "docs" / "RESULTS.md").write_text(render(runs))
    print(f"rendered docs/RESULTS.md from {src}")


if __name__ == "__main__":
    main()
