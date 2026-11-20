"""CI regression gate — PLAN PR-11. `make gate` is blocking from this PR on.

Doctrine: the gate READS ONLY configs GateSettings thresholds (TESTING §3),
evaluates every check BY NAME, and exits 1 on any failure. Missing artifact
files are a hard error — fail-closed, never pass-by-default.
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from histogate.config import GateSettings, load_settings
from histogate.schemas.gate import GateCheck, GateInputs, GateReport

_EPS = 1e-9


def _check(name: str, value: float, threshold: float,
           comparator: Literal[">=", "<=", ">", "<"], detail: str) -> GateCheck:
    passed = {">=": value >= threshold, "<=": value <= threshold,
              ">": value > threshold, "<": value < threshold}[comparator]
    return GateCheck(name=name, value=round(value, 6), threshold=threshold,
                     comparator=comparator, passed=passed, detail=detail)


def evaluate_gate(inputs: GateInputs, cfg: GateSettings) -> GateReport:
    checks = [
        _check("macro_f1", inputs.macro_f1, cfg.f1_min, ">=",
               "patch macro-F1 on CRC-VAL-HE-7K (fp32)"),
        _check("f1_domain_drop", inputs.macro_f1 - inputs.macro_f1_domain,
               cfg.f1_drop_domain_max, "<=",
               "normalized minus NONORM-scanner domain drop"),
        _check("ece", inputs.ece, cfg.ece_max, "<=", "post-temperature-scaling ECE"),
        _check("ood_auroc", inputs.ood_auroc, cfg.ood_auroc_min, ">=",
               "artifact/OOD rejection (energy score)"),
        _check("tsr_drift", abs(inputs.tsr_int8 - inputs.tsr_fp32),
               cfg.tsr_drift_max, "<=", "TSR biomarker drift fp32 -> int8"),
        _check("til_drift_rel",
               abs(inputs.til_int8 - inputs.til_fp32) / max(inputs.til_fp32, _EPS),
               cfg.til_drift_rel_max, "<=", "TIL density relative drift fp32 -> int8"),
        _check("quant_agreement", inputs.quant_agreement, cfg.quant_agreement_min, ">=",
               "argmax agreement fp32 vs int8"),
        _check("throughput", inputs.patches_per_s, cfg.throughput_min, ">=",
               "serving throughput floor"),
        _check("vram", inputs.peak_vram_mb, cfg.vram_max_mb, "<=",
               "int8 inference VRAM ceiling"),
    ]
    return GateReport(
        overall_pass=all(c.passed for c in checks),
        checks=checks,
        model_revision="see-artifacts",
        dataset_revision="NCT-CRC-HE-100K/CRC-VAL-HE-7K",
        timestamp=datetime.now(UTC).isoformat(),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="histogate-gate")
    p.add_argument("--metrics", type=Path, required=True,
                   help="GateInputs JSON (from eval + bench artifacts)")
    p.add_argument("--out", type=Path, required=True, help="GateReport JSON to write")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.metrics.exists():
        print(f"GATE FAIL-CLOSED: metrics file missing: {args.metrics}", file=sys.stderr)
        return 2
    try:
        inputs = GateInputs.model_validate_json(args.metrics.read_text())
    except Exception as e:  # noqa: BLE001 — any malformed artifact is a gate failure
        print(f"GATE FAIL-CLOSED: malformed metrics: {e}", file=sys.stderr)
        return 2

    cfg = load_settings(args.config).gate
    report = evaluate_gate(inputs, cfg)
    args.out.write_text(report.model_dump_json(indent=2))
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"[{mark}] {c.name}: {c.value} {c.comparator} {c.threshold}")
    print(f"GATE {'PASS' if report.overall_pass else 'FAIL'} -> {args.out}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
