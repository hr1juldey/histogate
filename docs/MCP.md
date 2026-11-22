# HistoGate over MCP — Claude, Gemini, any MCP client

HistoGate is agent-callable over **streamable HTTP** (protocol revision 3,
`2025-06-18` — no stdio). The design rule: **images never enter the model-of-
the-agent's context.** A human uploads through a web portal; the agent only
ever sees structured results and the report.

## Connect

Any MCP client that speaks streamable HTTP:

```json
{ "mcpServers": { "histogate": { "url": "http://127.0.0.1:8790/mcp" } } }
```

Gemini CLI: `gemini mcp add --transport http histogate http://127.0.0.1:8790/mcp`
Claude Desktop / Claude Code / Cursor / any client with a URL field: use the
JSON above.

Run the server: `uv run histogate-serve --mcp-http 0.0.0.0:8790`

## The guided workflow (semi-deterministic, guardrail-based)

The agent cannot freelance — it walks a fixed sequence, one step per call:

```
histogate_start  ->  portal upload  ->  histogate_advance  x3  ->  histogate_report
   session id         (human, web)        upload→analyze→review     final report
```

1. **`histogate_start`** → `{session_id, portal_url, next_step}`.
2. The **human** opens `portal_url` and uploads H&E images (PNG/TIFF, ≤16,
   ≤50 MB each). Files land on the server — never in the chat.
3. **`histogate_advance(session_id)`** — exactly one step per call:
   - `upload → analyze`: every uploaded file runs through the shipped model
     (calibrated probability, energy, gate verdict).
   - `analyze → review`: flagged (declined) files are listed by name.
   - `review → report`: done.
   Guardrails: calling `report` early, skipping steps, or advancing with no
   upload returns a fail-closed error, never a fabricated result.
4. **`histogate_report(session_id)`** → markdown report + JSON summary
   (files / answered / declined). No images, no base64 — token-clean.

**`run_regression_gate(metrics_path)`** (ops tool): pass/fail mirrors the
regulated regression gate over a `GateInputs` artifact exactly — the tool
reports what the gate computed, never its own opinion.

## A worked session (dummy path)

> **You:** analyze my slide
> **Agent:** *(calls histogate_start)* — please open this link and upload the
> images: `http://…/portal/abc123def`
> *(you upload; the images never enter the chat)*
> **Agent:** *(advance → advance → advance)* — analysis complete, one file
> was declined as unfamiliar tissue (safe refusal).
> **You:** give me the report
> **Agent:** *(histogate_report)* — 3 files analyzed, 2 answered (TUM 98.5%,
> STR 91.2%), 1 declined… *(markdown report)*

## For experts

- Transport: streamable HTTP at `/mcp`; protocol `2025-06-18` (verified by
  `initialize` round-trip — see `tests/test_mcp_http.py`).
- Two capabilities only: analysis and report generation (the guided workflow),
  plus the regression-gate ops tool.
- Fail-closed doctrine: unknown session, missing artifact, skipped step →
  structured error, never a PASS and never invented data.
- RUO (Research Use Only) on every tool description and every payload.
- Session state: `uploads/_wf/<session>.json` (resumable); files under
  `uploads/<session>/`.
