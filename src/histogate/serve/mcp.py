"""FastMCP server over STREAMABLE HTTP (never stdio — user law) with a web
upload portal: image bytes never enter the LLM context, only structured
results return. Guided workflow: upload -> analyze -> review -> report, one
step at a time, guardrails enforced, fail-closed.

Doctrine: the gate tool reports what the GATE computed, never its own
opinion. Every tool description carries RUO. Scoped to two capabilities:
analysis and report generation.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from PIL import Image
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from histogate.config import GateSettings, load_settings
from histogate.gates.eval_gate import evaluate_gate
from histogate.schemas.gate import GateInputs
from histogate.schemas.report import SlideReport
from histogate.serve.workflow import Workflow

_RUO = "Research Use Only. Not for use in diagnostic procedures."
AnalyzeFn = Callable[[Path], SlideReport]

_MAX_FILE = 50 * 1024 * 1024

_PORTAL_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HistoGate upload</title><style>
body{font-family:system-ui,sans-serif;background:#08090a;color:#e6e6e6;
display:grid;place-items:center;min-height:100vh;margin:0}
.card{background:#0f1011;border:1px solid #26272a;border-radius:14px;
padding:32px;max-width:420px;width:90%}
h1{font-size:18px;margin:0 0 6px} p{color:#9aa0a6;font-size:13px;line-height:1.5}
input[type=file]{width:100%;margin:14px 0;color:#9aa0a6}
button{width:100%;padding:10px;border-radius:9px;border:1px solid #4a4d55;
background:#191a1b;color:#fff;font-size:14px;cursor:pointer}
.ok{color:#5dbb7a;margin-top:12px;font-size:13px;display:none}
.ruo{margin-top:18px;font-size:11px;color:#6b7076}
</style></head><body><div class="card">
<h1>HistoGate — session {sid}</h1>
<p>Upload H&E tissue images (PNG/TIFF, up to 16). The AI agent is notified
when you advance — your images never enter the chat.</p>
<form id="f"><input type="file" accept="image/*,.tif,.tiff" multiple>
<button type="submit">Upload</button></form>
<div class="ok" id="ok">uploaded — now ask the agent for the next step</div>
<p class="ruo">Research Use Only. Not for use in diagnostic procedures.</p>
</div></body></html>"""


def build_mcp(analyze: AnalyzeFn, gate_dir: Path | None = None,
              patch_analyzer=None, uploads_root: Path | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(
        name="histogate",
        instructions=(
            "HistoGate guided tissue analysis. Call histogate_start to begin; "
            "the user uploads images through the portal URL (images never enter "
            "this context); advance one step at a time with histogate_advance; "
            f"fetch the report with histogate_report. {_RUO}"))
    wf = Workflow((uploads_root or Path("uploads")) / "_wf")
    up_root = uploads_root or Path("uploads")

    def _jerr(msg: str) -> str:
        return json.dumps({"error": msg, "overall_pass": False})

    @mcp.tool(description=f"Start a guided analysis session. Returns a portal "
                          f"URL for the human to upload images (they never enter "
                          f"the agent context) and the first step. {_RUO}")
    def histogate_start() -> str:
        sid = wf.start()
        host = (get_http_headers().get("host") or "127.0.0.1:8790")
        return json.dumps({
            "session_id": sid,
            "portal_url": f"http://{host}/portal/{sid}",
            "next_step": "upload",
            "what_you_need": "open the portal URL in a browser and upload H&E image(s)",
            "ruo": _RUO})

    @mcp.tool(description=f"Advance the session one step (upload -> analyze -> "
                          f"review -> report). Steps cannot be skipped. {_RUO}")
    def histogate_advance(session_id: str) -> str:
        st = wf.state(session_id)          # raises on unknown -> fail-closed
        step = st["step"]
        if step == "analyze" and patch_analyzer is not None:
            results, flagged = [], []
            for f in sorted((up_root / session_id).glob("*")):
                if f.stat().st_size > _MAX_FILE:
                    continue
                try:
                    r = patch_analyzer.analyze(Image.open(f).convert("RGB"))
                except Exception as e:  # noqa: BLE001 — one bad file must not kill the run
                    results.append({"file": f.name, "error": str(e)})
                    continue
                results.append({"file": f.name, "top_class": r["top_class"],
                                "confidence": r["confidence"],
                                "energy": r["energy"], "ood": r["ood"]})
                if r["ood"]:
                    flagged.append(f.name)
            wf._write(session_id, {**st, "results": results, "flagged": flagged})
        elif step == "review" and not st.get("results"):
            raise ValueError("guardrail: run analyze first")
        out = wf.advance(session_id)
        return out.to_json()

    @mcp.tool(description=f"Fetch the final report — only after the workflow "
                          f"reaches the report step. Markdown + JSON summary, "
                          f"no images. {_RUO}")
    def histogate_report(session_id: str) -> str:
        st = wf.state(session_id)
        if st["step"] != "report":
            raise ValueError(
                f"guardrail: session is at '{st['step']}', not report — advance first")
        rows = st.get("results", [])
        answered = [r for r in rows if not r.get("ood")]
        md = ["# HistoGate session report", "",
              f"- files analyzed: {len(rows)}",
              f"- answered: {len(answered)}",
              f"- declined (safe): {len(rows) - len(answered)}", ""]
        for r in rows:
            if r.get("ood"):
                md.append(f"- {r['file']}: DECLINED — unfamiliar tissue (safe refusal)")
            elif "error" in r:
                md.append(f"- {r['file']}: unreadable ({r['error']})")
            else:
                md.append(f"- {r['file']}: {r['top_class']} "
                          f"({round(r['confidence'] * 100, 1)}% sure)")
        md += ["", _RUO]
        return json.dumps({"markdown": "\n".join(md),
                           "summary": {"files": len(rows), "answered": len(answered),
                                       "declined": len(rows) - len(answered)},
                           "ruo": _RUO})

    @mcp.tool(description=f"Analyze a whole-slide directory of H&E patches on "
                          f"the server (path, never uploaded to the agent "
                          f"context); returns a SlideReport JSON. {_RUO}")
    def analyze_slide(slide_path: str) -> str:
        p = Path(slide_path)
        if not p.exists():
            return _jerr(f"slide not found: {slide_path}")
        return analyze(p).model_dump_json()

    @mcp.tool(description=f"Run the regulated regression gate over a GateInputs "
                          f"JSON artifact; pass/fail mirrors the gate exactly. {_RUO}")
    def run_regression_gate(metrics_path: str) -> str:
        p = Path(metrics_path)
        if not p.exists():
            return _jerr(f"metrics file missing (fail-closed): {metrics_path}")
        try:
            inputs = GateInputs.model_validate_json(p.read_text())
        except Exception as e:  # noqa: BLE001 — malformed artifact is a gate failure
            return _jerr(f"malformed metrics (fail-closed): {e}")
        cfg: GateSettings = load_settings().gate
        return evaluate_gate(inputs, cfg).model_dump_json()

    # ---- the portal: humans upload here; images never touch the LLM context
    @mcp.custom_route("/portal/{sid}", methods=["GET"])
    async def portal(request: Request) -> HTMLResponse:
        sid = request.path_params["sid"]
        try:
            wf.state(sid)
        except ValueError:
            return HTMLResponse("unknown session", status_code=404)
        return HTMLResponse(_PORTAL_HTML.replace("{sid}", sid))

    @mcp.custom_route("/portal/{sid}/upload", methods=["POST"])
    async def portal_upload(request: Request) -> JSONResponse:
        sid = request.path_params["sid"]
        try:
            wf.state(sid)
        except ValueError:
            return JSONResponse({"ok": False, "error": "unknown session"}, status_code=404)
        from starlette.datastructures import UploadFile

        form = await request.form()
        files = [f for f in form.getlist("files") if isinstance(f, UploadFile)]
        if not files:
            return JSONResponse({"ok": False, "error": "no files"}, status_code=422)
        dest = up_root / sid
        dest.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for f in files[:16]:
            b = await f.read()
            if len(b) > _MAX_FILE:
                return JSONResponse(
                    {"ok": False, "error": f"{f.filename} over 50MB"}, status_code=413)
            name = Path(f.filename or "upload.bin").name
            (dest / name).write_bytes(b)
            names.append(name)
        wf.add_files(sid, names)
        return JSONResponse({"ok": True, "files": names})

    return mcp
