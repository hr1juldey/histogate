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


_PLAIN = {
    "TUM": "tumor — malignant epithelium, the cancer cells themselves",
    "NORM": "healthy mucosa — orderly glands",
    "STR": "stroma — connective tissue around a tumor (carries the TSR biomarker)",
    "LYM": "dense immune cells (the TIL biomarker)",
    "MUS": "muscle",
    "MUC": "mucus lakes",
    "DEB": "debris — necrosis and waste",
    "ADI": "fat",
    "BACK": "empty background",
}


def _interpret_sections(rows: list[dict], tm: dict | None) -> list[tuple[str, str]]:
    """Guided interpretation: one topic per section, plain words first."""
    answered = [r for r in rows if not r.get("ood") and "top_class" in r]
    declined = [r for r in rows if r.get("ood")]
    frac: dict[str, int] = {}
    for r in answered:
        frac[r["top_class"]] = frac.get(r["top_class"], 0) + 1
    n = max(1, len(answered))
    secs = []
    if tm:
        tiles = tm.get("tiles", [])
        td = sum(1 for t in tiles if t.get("declined"))
        secs.append(("the slide at a glance",
                     f"{len(tiles)} tissue tiles; {len(tiles) - td} answered, "
                     f"{td} declined as unfamiliar. View it as a map at the "
                     f"portal link."))
        tf: dict[str, int] = {}
        for t in tiles:
            if not t.get("declined"):
                tf[t["cls"]] = tf.get(t["cls"], 0) + 1
        top = sorted(tf.items(), key=lambda kv: -kv[1])[:4]
        if top:
            secs.append(("what the map shows",
                         "; ".join(f"{_PLAIN.get(c, c)} — {v} tiles "
                                   f"({round(100 * v / max(1, len(tiles) - td))}%)"
                                   for c, v in top)))
    if frac:
        secs.append(("what the analysis found",
                     "; ".join(f"{_PLAIN.get(c, c)} in {round(100 * v / n)}% of "
                               f"answered patches" for c, v in
                               sorted(frac.items(), key=lambda kv: -kv[1]))))
    absent = [c for c in ("TUM", "NORM", "STR", "LYM") if c not in frac
              and (tm is None or c not in {t.get("cls") for t in tm.get("tiles", [])})]
    if absent:
        secs.append(("what is NOT there",
                     "no " + ", no ".join(_PLAIN.get(c, c).split(" — ")[0]
                                          for c in absent) +
                     " was found — absence is a finding, not a failure"))
    if declined:
        secs.append(("why some patches were declined",
                     f"{len(declined)} patch(es) looked unfamiliar (familiarity "
                     f"below the gate), so the model refused to answer rather "
                     f"than guess — a safe refusal, not an error"))
    nuc = [r["nuclei"] for r in answered if isinstance(r.get("nuclei"), dict)]
    if nuc:
        tot = sum(sum(v for v in d.values()) for d in nuc)
        epi = sum((d.get("neoplastic", 0) + d.get("nonneop_epithelial", 0))
                  for d in nuc)
        secs.append(("cell counts (second model)",
                     f"{tot} nuclei counted across {len(nuc)} answered "
                     f"patch(es), {round(100 * epi / max(1, tot))}% "
                     f"epithelial-lineage — counts from a PanNuke-trained "
                     f"nucleus model, not diagnoses"))
    conf = [r.get("confidence", 0) for r in answered]
    if conf:
        secs.append(("how much to trust it",
                     f"average confidence {round(100 * sum(conf) / len(conf))}% "
                     f"on answered patches; every number is calibrated, and "
                     f"declined patches are excluded by design"))
    secs.append(("the bottom line",
                 "the model points, the pathologist decides — this is a "
                 "research tool (RUO), not a diagnosis"))
    return secs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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

    @mcp.tool(description=f"Tile-map an uploaded slide (TIFF): every tissue "
                          f"tile runs inference; returns the tile map JSON "
                          f"(rows/cols, per-tile class + familiarity) and a "
                          f"portal URL where the human SEES the map. {_RUO}")
    def histogate_tile_map(session_id: str) -> str:
        import tifffile

        from histogate.serve.slide_map import build_slide_map

        wf.state(session_id)   # fail-closed on unknown session
        files = [f for f in (up_root / session_id).glob("*.tif")]
        files += [f for f in (up_root / session_id).glob("*.tiff")]
        if not files:
            raise ValueError("guardrail: upload a TIFF slide first (tile maps "
                             "need a whole-slide image)")
        path = files[0]
        with tifffile.TiffFile(path) as tf:
            n_levels = len(tf.series[0].levels)
        level = min(4, n_levels - 1)
        m = build_slide_map(path, patch_analyzer, level=level)
        wf._write(session_id, {**wf.state(session_id), "tile_map": m})
        host = (get_http_headers().get("host") or "127.0.0.1:8790")
        n_dec = sum(1 for t in m["tiles"] if t.get("declined"))
        return json.dumps({
            "slide": path.name, "level": level,
            "tiles": len(m["tiles"]), "declined": n_dec,
            "map_portal_url": f"http://{host}/portal/{session_id}/map",
            "tile_map": m, "ruo": _RUO})

    @mcp.tool(description=f"Guided interpretation of the analysis: what tissue "
                          f"IS there (classes, fractions, confidence), what is "
                          f"NOT there, and what every decline means — plain "
                          f"language first, technical terms second. Step "
                          f"through the sections one at a time. {_RUO}")
    def histogate_interpret(session_id: str, section: int = 0) -> str:
        st = wf.state(session_id)
        rows = st.get("results", [])
        tm = st.get("tile_map")
        if not rows and not tm:
            raise ValueError("guardrail: run the analysis first")
        sections = _interpret_sections(rows, tm)
        i = max(0, min(section, len(sections) - 1))
        return json.dumps({
            "section": i, "total_sections": len(sections),
            "title": sections[i][0], "body": sections[i][1],
            "next": None if i + 1 >= len(sections) else i + 1,
            "ruo": _RUO})

    @mcp.tool(description=f"Search the MLflow experiment tracker (the public "
                          f"tracker the site's numbers trace to): run names, "
                          f"metrics, status — so any coding agent can audit "
                          f"what happened and what did not. {_RUO}")
    def mlflow_search_runs(query: str = "", limit: int = 10) -> str:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(up_root.parent / 'mlflow_public/mlflow.db').resolve()}")
        c = mlflow.MlflowClient()
        exps = [e for e in c.search_experiments() if e.name != "Default"]
        if not exps:
            return json.dumps({"runs": [], "ruo": _RUO})
        filt = f"tags.mlflow.runName LIKE '%{query}%'" if query else None
        runs = c.search_runs([e.experiment_id for e in exps],
                             filter_string=filt, max_results=limit,
                             order_by=["attributes.start_time DESC"])
        keep = ("crcval_f1", "nonorm_f1", "ood_auroc", "ece",
                "macro_f1", "accuracy")
        out = [{"run_id": r.info.run_id,
                "name": r.data.tags.get("mlflow.runName", "?"),
                "status": r.info.status,
                "metrics": {k: v for k, v in r.data.metrics.items()
                            if k in keep or any(k.startswith(x) for x in keep)}}
               for r in runs]
        return json.dumps({"tracker": "mlflow_public/mlflow.db", "runs": out,
                           "ruo": _RUO})

    @mcp.tool(description=f"Model quality, one call: the 9-check regression "
                          f"gate, the published-benchmark table, and the OOD "
                          f"detector comparison — all read from artifacts, "
                          f"never hand-typed. {_RUO}")
    def model_quality() -> str:
        root = Path(__file__).resolve().parents[3]   # repo root, wherever run from
        out: dict = {"ruo": _RUO}
        for name, rel in [("metrics", "runs/pr16-mixed/metrics.json"),
                          ("crcval_accuracy", "runs/pr16-mixed/crcval_accuracy.json"),
                          ("phikon_benchmark", "runs/bench-phikon/result.json"),
                          ("ood_knn", "runs/pr16-mixed/ood_knn.json")]:
            f = root / rel
            if f.exists():
                d = json.loads(f.read_text())
                out[name] = ({k: d[k] for k in ("accuracy", "macro_f1")
                              if k in d} if name == "crcval_accuracy" else d)
        return json.dumps(out)

    # ---- prompts: the guided walkthrough as a first-class MCP component ----
    @mcp.prompt(description="Guided slide analysis: walks the user through "
              "upload, inference, tile map, step-by-step interpretation, and "
              "the report. Visual output is always a URL to the web portal — "
              "never ask the user to paste images into the conversation.")
    def analyze_my_slide(goal: str = "analyze my slide and tell me what you "
                        "see") -> str:
        return f"""HistoGate guided workflow — user goal: {goal}

Follow these steps in order; each tool returns the next step. Never skip.
1. histogate_start — get the session id and the upload portal URL. Give the
   user the portal_url and WAIT until they say they uploaded. The images go
   to the server through the portal, never into this conversation.
2. histogate_advance — run the analysis when upload is confirmed.
3. histogate_tile_map — whole-slide tile map + detection. Give the user the
   map_portal_url so they can SEE the colored map in their browser.
4. histogate_interpret — call it once per section (section 0, 1, 2, ...) and
   relay each topic to the user one at a time: what IS there, what is NOT
   there, why anything was declined, how much to trust it.
5. histogate_report — final structured report; read it back in plain words.
Every answer carries the RUO notice: research use only, not a diagnosis."""

    @mcp.prompt(description="Honest model audit: walks the quality gate, "
              "benchmarks, and MLflow tracker — including the failures.")
    def audit_the_model() -> str:
        return """HistoGate model audit — report honestly, failures included.

1. model_quality — the regression gate checks, the benchmark table (CRC-VAL
   accuracy, phikon-v2 comparison, OOD detectors), all from artifacts.
2. mlflow_search_runs — list recent tracker runs; note statuses and metrics.
3. Summarize for the user: what passes, what fails or is missing, and what
   the numbers are NOT evidence of (no clinical use, RUO)."""

    # ---- resources: read-only data agents can list without tool calls ----
    @mcp.resource("histogate://model-card",
                  description="The shipped model card (markdown).")
    def res_model_card() -> str:
        f = _repo_root() / "MODEL_CARD.md"
        return f.read_text() if f.exists() else "MODEL_CARD.md not found"

    @mcp.resource("histogate://benchmarks",
                  description="Benchmark table rendered from artifacts "
                              "(accuracy, phikon-v2, OOD) — no hand-typed "
                              "numbers.")
    def res_benchmarks() -> str:
        f = _repo_root() / "docs/BENCHMARKS.md"
        return f.read_text() if f.exists() else "docs/BENCHMARKS.md not found"

    @mcp.resource("histogate://workflow",
                  description="The guided-workflow contract: steps, "
                              "guardrails, and the portal law.")
    def res_workflow() -> str:
        return json.dumps({"steps": list(wf.STEPS) if hasattr(wf, "STEPS")
                           else ["upload", "analyze", "review", "report"],
                           "portal_law": "images never enter the LLM "
                                         "conversation; uploads go through "
                                         "the portal",
                           "guardrails": "fail-closed; steps cannot be "
                                         "skipped",
                           "ruo": _RUO}, indent=2)

    # ---- the portal: humans upload here; images never touch the LLM context
    @mcp.custom_route("/portal/{sid}", methods=["GET"])
    async def portal(request: Request) -> HTMLResponse:
        sid = request.path_params["sid"]
        try:
            wf.state(sid)
        except ValueError:
            return HTMLResponse("unknown session", status_code=404)
        return HTMLResponse(_PORTAL_HTML.replace("{sid}", sid))

    @mcp.custom_route("/portal/{sid}/map", methods=["GET"])
    async def portal_map(request: Request) -> HTMLResponse:
        """The human SEES the tile map of their uploaded slide: one colored
        cell per tissue tile — verdicts at a glance, RUO below."""
        sid = request.path_params["sid"]
        try:
            st = wf.state(sid)
        except ValueError:
            return HTMLResponse("unknown session", status_code=404)
        tm = st.get("tile_map")
        if not tm:
            return HTMLResponse("no tile map yet — the agent must run "
                                "histogate_tile_map first", status_code=404)
        cols, rows = tm["grid"]["cols"], tm["grid"]["rows"]
        colors = {"TUM": "#e46fa8", "NORM": "#5dbb7a", "MUS": "#9a7ce0",
                  "STR": "#6f6ee0", "LYM": "#4a9ebf", "MUC": "#58b6a8",
                  "DEB": "#8a8f98", "ADI": "#c9a227"}
        pos = {}
        for t in tm["tiles"]:
            pos[(t["row"], t["col"])] = t
        cells = []
        for r in range(rows):
            for c in range(cols):
                t = pos.get((r, c))
                if t is None:
                    cells.append('<div class="e"></div>')
                elif t.get("declined"):
                    cells.append(f'<div class="d" title="declined — unfamiliar '
                                 f'(familiarity {t["familiarity"]}/100)"></div>')
                else:
                    col = colors.get(t["cls"], "#666")
                    op = 0.25 + 0.75 * t["familiarity"] / 100
                    cells.append(f'<div class="c" style="background:{col};'
                                 f'opacity:{op:.2f}" title="{t["cls"]} '
                                 f'{round(t["conf"] * 100)}% · familiarity '
                                 f'{t["familiarity"]}/100"></div>')
        nd = sum(1 for t in tm["tiles"] if t.get("declined"))
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HistoGate tile map</title><style>
body{{font-family:system-ui,sans-serif;background:#08090a;color:#e6e6e6;margin:0;
display:grid;place-items:center;min-height:100vh}}
.card{{max-width:760px;width:94%;margin:24px 0}}
h1{{font-size:18px}} p{{color:#9aa0a6;font-size:13px}}
.stats{{display:flex;gap:22px;font-size:14px;margin:10px 0 16px}}
.stats b{{color:#fff}}
.grid{{display:grid;grid-template-columns:repeat({cols},1fr);gap:2px;
aspect-ratio:{cols}/{rows};width:100%}}
.c,.d{{border-radius:2px}} .d{{background:transparent;
border:1px dashed #e3b341}} .e{{background:#141516}}
.leg{{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px;font-size:12px;color:#9aa0a6}}
.leg i{{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:4px;vertical-align:middle}}
.ruo{{margin-top:16px;font-size:11px;color:#6b7076}}
</style></head><body><div class="card">
<h1>HistoGate — tile map · session {sid}</h1>
<p>One cell per tissue tile. Brighter = more familiar. Dashed amber = declined
(unfamiliar — safe refusal).</p>
<div class="stats mono"><span><b>{len(tm["tiles"])}</b> tiles</span>
<span><b>{len(tm["tiles"]) - nd}</b> answered</span>
<span><b style="color:#e3b341">{nd}</b> declined</span></div>
<div class="grid">{''.join(cells)}</div>
<div class="leg">{''.join(f'<span><i style="background:{v}"></i>{k}</span>' for k, v in colors.items())}
<span><i style="border:1px dashed #e3b341"></i>declined</span></div>
<p class="ruo">Research Use Only. Not for use in diagnostic procedures.</p>
</div></body></html>"""
        return HTMLResponse(html)

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
