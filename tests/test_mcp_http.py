"""Streamable-HTTP MCP server: guardrails + portal + protocol revision 3
(2025-06-18). In-memory fastmcp Client for the tools; raw JSON-RPC POST for
the protocol negotiation; ASGI transport for the portal upload route."""
from __future__ import annotations

import json
from pathlib import Path

import anyio
import httpx
import pytest
from fastmcp import Client
from starlette.datastructures import UploadFile

from histogate.serve.mcp import build_mcp


def _no_analyze(p: Path):  # pragma: no cover - never called in these tests
    raise AssertionError("not used")


class _StubAnalyzer:
    """Deterministic stand-in for PatchAnalyzer — the real one is exercised
    by tests/test_serve_real.py; here we test the workflow, not the model."""

    def analyze(self, img) -> dict:
        return {"top_class": "TUM", "confidence": 0.97, "energy": -4.1,
                "ood": False, "probabilities": []}


@pytest.fixture()
def mcp(tmp_path: Path):
    return build_mcp(analyze=_no_analyze, patch_analyzer=_StubAnalyzer(),
                     uploads_root=tmp_path / "up")


def _call(c, name, args=None):
    r = anyio.from_thread.run  # placeholder, unused
    raise RuntimeError


def test_guardrail_report_before_advancing(mcp) -> None:
    async def run() -> None:
        async with Client(mcp) as c:
            start = json.loads((await c.call_tool("histogate_start", {})).content[0].text)
            assert start["portal_url"].endswith(start["session_id"])
            assert start["next_step"] == "upload"
            assert "Research Use Only" in start["ruo"]
            with pytest.raises(Exception, match="guardrail"):
                await c.call_tool("histogate_report", {"session_id": start["session_id"]})
            # advance before upload must fail too
            with pytest.raises(Exception, match="guardrail"):
                await c.call_tool(
                    "histogate_advance", {"session_id": start["session_id"]})
    anyio.run(run)


def test_full_path_portal_upload_to_report(mcp, tmp_path: Path) -> None:
    async def run() -> None:
        app = mcp.http_app()
        async with Client(mcp) as c:
            start = json.loads((await c.call_tool("histogate_start", {})).content[0].text)
            sid = start["session_id"]
        # human uploads through the portal (images never enter the LLM context)
        png = tmp_path / "t.png"
        png.write_bytes(_tiny_png())
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as hc:
                with open(png, "rb") as fh:
                    r = await hc.post(f"/portal/{sid}/upload",
                                      files={"files": ("t.png", fh, "image/png")})
                assert r.status_code == 200 and r.json()["ok"], r.text
                r404 = await hc.get("/portal/deadbeef")
                assert r404.status_code == 404
        # analyze step runs without the patch analyzer wired -> plain advance
        async with Client(mcp) as c:
            out = json.loads((await c.call_tool(
                "histogate_advance", {"session_id": sid})).content[0].text)
            assert out["step"] == "upload" and out["next_step"] == "analyze"
            out2 = json.loads((await c.call_tool(
                "histogate_advance", {"session_id": sid})).content[0].text)
            assert out2["step"] == "analyze"
            assert out2["result"]["state"]["files"] == ["t.png"]
            out3 = json.loads((await c.call_tool(
                "histogate_advance", {"session_id": sid})).content[0].text)
            assert out3["step"] == "review" and out3["next_step"] == "report"
            rep = json.loads((await c.call_tool(
                "histogate_report", {"session_id": sid})).content[0].text)
            assert "TUM" in rep["markdown"]
            assert rep["summary"] == {"files": 1, "answered": 1, "declined": 0}
            assert "HistoGate session report" in rep["markdown"]
            assert "Research Use Only" in rep["markdown"]
            assert "base64" not in rep["markdown"]
    anyio.run(run)


def test_protocol_is_revision_3(mcp) -> None:
    """initialize over the wire must negotiate 2025-06-18 (3rd revision)."""

    async def run() -> None:
        app = mcp.http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as hc:
                r = await hc.post("/mcp", json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "t", "version": "0"}}},
                    headers={"accept": "application/json, text/event-stream"})
                assert r.status_code == 200, r.text
                body = _first_json(r)   # streamable HTTP may answer SSE or JSON
                assert body["result"]["protocolVersion"] == "2025-06-18", body
    anyio.run(run)


def _first_json(r: httpx.Response) -> dict:
    import json as _json

    ct = r.headers.get("content-type", "")
    if "event-stream" in ct:
        for line in r.text.splitlines():
            if line.startswith("data:"):
                return _json.loads(line[5:].strip())
        raise AssertionError(f"no data line in SSE: {r.text[:200]}")
    return _json.loads(r.text)


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 60, 60)).save(b, "PNG")
    return b.getvalue()
