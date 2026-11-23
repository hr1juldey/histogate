# HistoGate video package (silent cut)

Branch: `video-package`. Silent product walkthrough (narration comes later).
Every number on screen comes from `docs/RESULTS.md`, `docs/BENCHMARKS.md`,
`docs/EXEC_LOG.md` and `runs/pr16-mixed/` artifacts.

## Outputs (`video/out/`)

| file | what | duration | res/fps | size |
|---|---|---|---|---|
| `histogate_walkthrough_silent.mp4` | full 10-scene walkthrough, 0.5s crossfades, crf 18, +faststart | 116.4 s | 1920x1080 @ 30 | ~8.0 MB |
| `stain_dial_loop_720p.mp4` | README loop: /story stain dial (first model flips wrong, shipped declines), two-pass H.264, +faststart | 10.0 s | 1280x720 @ 30 | ~1.6 MB |

## Scene map (final cut)

| t (approx) | scene | source |
|---|---|---|
| 0:00–0:15 | S1 homepage hero + scroll to measured counters | screen capture, live site `/` |
| 0:15–0:29 | S2 /demo: TUM verdict, viewer zoom, LYMPH **declined** | screen capture, live site `/demo` |
| 0:29–0:41 | S3 failure story: four runs, v1 collapse 0.9741→0.7842, gates 6/9·7/9·6/9·8/9 | diagram `diagrams/s3_failure.html` (GSAP, frame-stepped) |
| 0:41–1:01 | S4 /story stain dial drag: first model 98.7% WRONG vs shipped **declined — safe** | screen capture, live site `/story` |
| 1:01–1:10 | S5 pipeline: stain physics → 9-class 6M model (28.7MB int8, 455.9 p/s) → familiarity gate (0.9763) → conformal sets (0.9006) → slide map | diagram `diagrams/s5_pipeline.html` |
| 1:10–1:18 | S6 trust: KNN AUROC 0.9763 vs energy 0.7972; conformal sets widen, coverage 0.9006, ECE 0.0064 | diagram `diagrams/s6_trust.html` |
| 1:18–1:29 | S7 mobile 390x844: pinch-zoom slide viewer, tile tap → "declined — safe" | screen capture (window resized to 390x844, composited on #08090a) |
| 1:29–1:40 | S8 /slide: foreign-organ map (lymph node), zoom to real tiles, declined tile + TUM verdict | screen capture, live site `/slide` |
| 1:40–1:49 | S9 scoreboard: animated counters (97.81%, 0.9776, 0.9763, 0.9006, 0.0064, 455.9 p/s, 6M vs 307M 93.12, 9/9, 1.000000) | diagram `diagrams/s9_numbers.html` |
| 1:49–1:56 | S10 outro: histogate.vercel.app | diagram `diagrams/s10_outro.html` |

## Tree

- `video/scenes/` — per-scene mp4s (screen scenes via Xvfb :2 + ffmpeg x11grab
  while CDP drives the live site; diagram scenes via frame-stepped
  `tl.progress()` + `Page.captureScreenshot`)
- `video/diagrams/` — the five GSAP HTML scenes (site design tokens: #08090a,
  Inter / JetBrains Mono, #5e6ad2 accent)
- `video/frames/` — rendered PNG frames (regenerated, not committed — see
  `.gitignore`)
- `video/scripts/` — rig code:
  - `cdp.py` — minimal CDP client (timed-out sends, touch pinch, range drag)
  - `capture_screens.py`, `capture_s78.py` — screen-scene drivers
  - `render_diagrams.py` — diagram frame-stepper
  - `assemble.py` — xfade assembly + mobile composite

## Reproduce

```bash
Xvfb :2 -screen 0 1920x1080x24 &   # then run chromium kiosk on :2 with --remote-debugging-port=9333
uv run --with websockets python video/scripts/capture_screens.py
uv run --with websockets python video/scripts/capture_s78.py
uv run --with websockets python video/scripts/render_diagrams.py
uv run python video/scripts/assemble.py
```

Research Use Only — not for use in diagnostic procedures.
