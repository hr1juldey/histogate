"""PR-11: regression gate — PLAN.md PR-11 RED contract.

THE GATE MUST BE ABLE TO FAIL. Every test injects a specific failure and
asserts the gate catches it BY NAME. A gate that only passes is a rubber
stamp; these tests are the proof it is a gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from histogate.config import GateSettings
from histogate.gates.eval_gate import evaluate_gate
from histogate.schemas.gate import GateInputs, GateReport


def _good_inputs() -> GateInputs:
    return GateInputs(
        macro_f1=0.955, macro_f1_domain=0.948, ece=0.031, ood_auroc=0.96,
        tsr_fp32=0.42, tsr_int8=0.425, til_fp32=12.0, til_int8=11.6,
        quant_agreement=0.9995, patches_per_s=180.0, peak_vram_mb=1450.0,
    )


def _failing_names(report: GateReport) -> set[str]:
    return {c.name for c in report.checks if not c.passed}


def test_all_good_passes_and_schema_valid() -> None:
    report = evaluate_gate(_good_inputs(), GateSettings())
    assert report.overall_pass
    assert len(report.checks) >= 9, f"expected >=9 checks, got {len(report.checks)}"
    json.dumps(report.model_dump())  # serializable


def test_injected_corrupt_model_fails_by_name() -> None:
    """Corrupted-checkpoint stand-in: accuracy collapses -> gate exits failed."""
    bad = _good_inputs().model_copy(update={"macro_f1": 0.41, "quant_agreement": 0.90})
    report = evaluate_gate(bad, GateSettings())
    assert not report.overall_pass
    assert "macro_f1" in _failing_names(report)
    assert "quant_agreement" in _failing_names(report)


def test_injected_domain_shift_fails() -> None:
    bad = _good_inputs().model_copy(update={"macro_f1_domain": 0.90})
    report = evaluate_gate(bad, GateSettings())
    assert not report.overall_pass
    assert "f1_domain_drop" in _failing_names(report)


def test_injected_biomarker_drift_fails() -> None:
    """TSR drift of 0.3 across quantization -> biomarker check fires."""
    bad = _good_inputs().model_copy(update={"tsr_int8": 0.72})
    report = evaluate_gate(bad, GateSettings())
    assert "tsr_drift" in _failing_names(report)
    bad2 = _good_inputs().model_copy(update={"til_int8": 20.0})  # +67% rel drift
    report2 = evaluate_gate(bad2, GateSettings())
    assert "til_drift_rel" in _failing_names(report2)


def test_injected_slow_bench_fails() -> None:
    bad = _good_inputs().model_copy(update={"patches_per_s": 5.0})
    report = evaluate_gate(bad, GateSettings())
    assert "throughput" in _failing_names(report)
    bad2 = _good_inputs().model_copy(update={"peak_vram_mb": 3200.0})
    report2 = evaluate_gate(bad2, GateSettings())
    assert "vram" in _failing_names(report2)


def test_gate_is_fail_closed_on_missing_artifacts(tmp_path: Path) -> None:
    """No metrics files -> NONZERO exit + error, never a pass-by-default."""
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "histogate.gates.eval_gate",
         "--metrics", str(tmp_path / "nonexistent.json"), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert not out.exists(), "gate must not write a report it could not compute"


def test_cli_exit_code_contract(tmp_path: Path) -> None:
    """exit 0 on green, exit 1 on red — from a real subprocess (CI contract)."""
    good = tmp_path / "good.json"
    good.write_text(_good_inputs().model_dump_json())
    out1 = tmp_path / "r1.json"
    r1 = subprocess.run(
        [sys.executable, "-m", "histogate.gates.eval_gate",
         "--metrics", str(good), "--out", str(out1)], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    GateReport.model_validate_json(out1.read_text())

    bad = tmp_path / "bad.json"
    bad.write_text(_good_inputs().model_copy(update={"ece": 0.30}).model_dump_json())
    out2 = tmp_path / "r2.json"
    r2 = subprocess.run(
        [sys.executable, "-m", "histogate.gates.eval_gate",
         "--metrics", str(bad), "--out", str(out2)], capture_output=True, text=True)
    assert r2.returncode == 1
    rep = GateReport.model_validate_json(out2.read_text())
    assert not rep.overall_pass and "ece" in _failing_names(rep)
