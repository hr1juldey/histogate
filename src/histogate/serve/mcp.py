"""FastMCP server — PLAN PR-14. HistoGate becomes agent-callable.

Doctrine: the gate tool reports what the GATE computed, never its own
opinion — pass/fail comes from evaluate_gate over the artifact. Missing
artifact => fail-closed error payload, never a PASS. Every tool description
carries RUO: regulated behavior follows the model to every surface.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastmcp import FastMCP

from histogate.config import GateSettings, load_settings
from histogate.gates.eval_gate import evaluate_gate
from histogate.schemas.gate import GateInputs
from histogate.schemas.report import SlideReport

_RUO = "Research Use Only. Not for use in diagnostic procedures."
AnalyzeFn = Callable[[Path], SlideReport]


def build_mcp(analyze: AnalyzeFn, gate_dir: Path | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(name="histogate", instructions=f"HistoGate tissue analysis. {_RUO}")

    @mcp.tool(description=f"Analyze a whole-slide image directory of H&E patches; "
                          f"returns a SlideReport JSON. {_RUO}")
    def analyze_slide(slide_path: str) -> str:
        p = Path(slide_path)
        if not p.exists():
            return json_error(f"slide not found: {slide_path}")
        return analyze(p).model_dump_json()

    @mcp.tool(description=f"Run the regulated regression gate over a GateInputs "
                          f"JSON artifact; pass/fail mirrors the gate exactly. {_RUO}")
    def run_regression_gate(metrics_path: str) -> str:
        p = Path(metrics_path)
        if not p.exists():
            return json_error(f"metrics file missing (fail-closed): {metrics_path}")
        try:
            inputs = GateInputs.model_validate_json(p.read_text())
        except Exception as e:  # noqa: BLE001 — malformed artifact is a gate failure
            return json_error(f"malformed metrics (fail-closed): {e}")
        cfg: GateSettings = load_settings().gate
        return evaluate_gate(inputs, cfg).model_dump_json()

    def json_error(msg: str) -> str:
        import json

        return json.dumps({"error": msg, "overall_pass": False})

    return mcp
