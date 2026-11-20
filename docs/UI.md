# HistoGate — UI & Interaction Design

Version: 0.1 · Principle: the system is CLI/API-first; the UI is the REPORT (the artifact a human reads). Two surfaces: (A) the slide report (HTML/markdown), (B) the terminal gate output. Both are designed, not emergent.

---

## A. Slide report (primary human surface)

Single self-contained HTML file (no server dependency, attachable in email — P3/P4 doctrine: the report is what travels).

### Layout (desktop, 960px column; prints to A4)

```
┌────────────────────────────────────────────────────────────┐
│ VERDICT BANNER   [ ✅ CLEAN ]  or  [ ⚠ GATED: 37/512 patches]│  ← 6% of decisions happen here; color + text (never color alone)
├──────────────┬─────────────────────────────────────────────┤
│ SLIDE MAP    │  COMPARTMENT DONUT    ARTIFACT TABLE         │
│ (grid heat   │  (9 classes, %)       (kind | stage | area)  │
│  map: tissue │                                              │
│  class per   ├─────────────────────────────────────────────┤
│  patch,      │  TSR CARD            TIL CARD                │
│  artifact    │  0.62 · stroma-low    142 /mm² intra-tumoral │
│  patches     │  [stroma-high|low vs 50% cut]                │
│  hatched)    ├─────────────────────────────────────────────┤
│              │  UNCERTAINTY STRIP: per-patch energy heat    │
├──────────────┴─────────────────────────────────────────────┤
│ METHODS FOOTNOTE: TSR = patch-proxy of van Pelt 2018 proto- │
│ col · TIL = Saltz 2018 compartment style · model rev, T,    │
│ dataset rev, seed                                          │
├────────────────────────────────────────────────────────────┤
│ RESEARCH USE ONLY. Not for use in diagnostic procedures.    │  ← always visible, not folded
└────────────────────────────────────────────────────────────┘
```

### Interaction rules
1. **Verdict first.** The single question "is this slide analyzable?" answers in the banner before any scroll.
2. **Trust through provenance.** Every number's method + revision sits in the footnote; the TSR card always shows the disclosure line — a reader must never mistake proxy for protocol.
3. **No hidden state.** Report is static HTML + inline base64 images; identical SlideReport JSON → byte-identical HTML (tested, PR-12).
4. **Accessibility:** WCAG AA contrast; verdicts encode state in text+icon, not color alone; donut has an adjacent table (figures for machines, table for screen readers).
5. **Artifact hatch pattern** distinct per stage (diagonal = processing, dotted = scanning) so the map survives grayscale printing.
6. Markdown twin for diff-ability and MCP delivery; HTML for humans; JSON for machines — one schema, three renderings.

### Serving interaction (FastAPI)
`POST /analyze` (slide_path) → job id → `GET /report/{id}?fmt=html|md|json`. Streaming NDJSON progress events during tiling (`{stage: tiling|inference|aggregation, done, total}`) so a client can show progress — long silent waits read as failure. Errors return machine-readable `{code, detail, ruo_disclaimer}`.

### MCP interaction
Tools return the JSON schema; `get_report(fmt="md")` gives agents paste-ready markdown with the RUO line intact (agents strip footers — schema makes it a field, not prose).

## B. Terminal gate UI (`histogate-gate`)

Rich-rendered table; exit code is the contract, the table is for humans:

```
HistoGate regression gate — rev a1b2c3d — 2026-11-21 02:14
┌───────────────┬─────────┬──────────┬────────┬───────────────┐
│ check         │ value   │ thresh   │ cmp    │ result        │
├───────────────┼─────────┼──────────┼────────┼───────────────┤
│ G-ACC   f1    │ 0.948   │ ≥0.93    │ worst  │ PASS          │
│ G-DOMAIN drop │ 0.011   │ ≤0.015   │        │ PASS          │
│ G-TSR   drift │ 0.061   │ ≤0.05    │ int8   │ ✗ FAIL        │
│ ...           │         │          │        │               │
└───────────────┴─────────┴──────────┴────────┴───────────────┘
OVERALL: FAIL (1/10 checks) → artifact runs/gate_20261121_0214.json
```

Rules: failing checks print their `detail` (which metric file, which domain, which precision); summary line always states artifact path; colors degrade to PASS/FAIL text. Gate output IS the demo — a reviewer watching the terminal sees the system reject its own degraded model.

## C. Out of scope (recorded)
Slide viewer with pan/zoom (AIRA has AIRADHI/PathPresenter partners; rebuilding a viewer is not our claim). If demoed live: the static HTML report opened in a browser is the walkthrough.
