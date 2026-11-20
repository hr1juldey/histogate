"""PR-14: FastMCP server — PLAN.md PR-14 RED contract.

Adversarial angles: tool SCHEMAS must match the contracts as a client sees
them (list_tools, not our own registry); an in-process client must complete
a real analyze_slide round-trip into a schema-valid SlideReport; the gate
tool's pass/fail must AGREE with the artifact it read (a tool that always
says PASS is a lie, not a tool); and every tool description carries RUO —
regulated behavior follows the model wherever it is exposed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from histogate.schemas.gate import GateInputs, GateReport
from histogate.schemas.report import SlideReport, TSRResult


def _report(slide_id: str = "fixture-slide") -> SlideReport:
    return SlideReport(
        slide_id=slide_id, n_patches=16, n_gated=1,
        compartment_fractions={"TUM": 0.6, "STR": 0.4},
        artifacts=[],
        tsr=TSRResult(stroma_fraction=0.4, tumor_fraction=0.6, tsr=0.6,
                      group="stroma-low", method="patch-proxy-vanpelt-2018",
                      disclosure="patch-proxy; RUO"),
        calibration_temperature=1.2, model_revision="testrev01",
        ruo_disclaimer="Research Use Only. Not for use in diagnostic procedures.",
    )


@pytest.fixture()
def mcp(tmp_path: Path):
    from histogate.serve.mcp import build_mcp

    fixture_slide = tmp_path / "slide"
    fixture_slide.mkdir()
    (fixture_slide / "patch.tif").write_bytes(b"0" * 8)
    good_gate = tmp_path / "good_gate.json"
    good_gate.write_text(GateInputs(
        macro_f1=0.95, macro_f1_domain=0.945, ece=0.03, ood_auroc=0.95,
        tsr_fp32=0.5, tsr_int8=0.51, til_fp32=10.0, til_int8=10.2,
        quant_agreement=0.9995, patches_per_s=100.0, peak_vram_mb=1500.0,
    ).model_dump_json())
    bad_gate = tmp_path / "bad_gate.json"
    bad_gate.write_text(GateInputs(
        macro_f1=0.44, macro_f1_domain=0.44, ece=0.30, ood_auroc=0.60,
        tsr_fp32=0.5, tsr_int8=0.8, til_fp32=10.0, til_int8=10.1,
        quant_agreement=0.90, patches_per_s=3.0, peak_vram_mb=4000.0,
    ).model_dump_json())
    server = build_mcp(analyze=lambda p: _report(p.name), gate_dir=tmp_path)
    return server, fixture_slide, good_gate, bad_gate


async def test_tool_schemas_expose_contracts_and_ruo(mcp) -> None:
    server, *_ = mcp
    async with Client(server) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert {"analyze_slide", "run_regression_gate"} <= set(tools)
    for name, t in tools.items():
        assert "Research Use Only" in t.description, f"{name} description missing RUO"


async def test_analyze_slide_roundtrip_valid_report(mcp) -> None:
    server, slide, _, _ = mcp
    async with Client(server) as c:
        result = await c.call_tool("analyze_slide", {"slide_path": str(slide)})
    payload = json.loads(result.content[0].text)
    report = SlideReport.model_validate(payload)
    assert report.n_patches == 16


async def test_gate_tool_agrees_with_artifact(mcp) -> None:
    server, _, good, bad = mcp
    async with Client(server) as c:
        ok = json.loads((await c.call_tool(
            "run_regression_gate", {"metrics_path": str(good)})).content[0].text)
        ko = json.loads((await c.call_tool(
            "run_regression_gate", {"metrics_path": str(bad)})).content[0].text)
    GateReport.model_validate(ok)
    GateReport.model_validate(ko)
    assert ok["overall_pass"] is True
    assert ko["overall_pass"] is False, "gate tool lied: bad artifact reported PASS"


async def test_gate_tool_fail_closed_on_missing_file(mcp) -> None:
    server, _, _, _ = mcp
    async with Client(server) as c:
        result = await c.call_tool(
            "run_regression_gate", {"metrics_path": "/nonexistent/gate.json"})
    payload = json.loads(result.content[0].text)
    assert payload["overall_pass"] is False or "error" in payload
