# HistoGate — Engineering Plan: PR Sequence, Red/Green Tests, Perf Hooks

Version: 0.1 · Reads with: PRD/HLD/LLD/TESTING. Every PR = red tests first (adversarial), then implementation, then green. No PR merges with tautological tests (TESTING §1). Perf and UI deltas are planned AT the PR level, not retrofitted.

---

## PR-0 — Bootstrap (uv + CI skeleton)
Scope: pyproject, uv.lock, Makefile, ruff+pytest config, empty CI (lint+test matrix), docs committed.
RED: CI fails on a deliberately broken lint file; `uv run pytest` exits 1 with no tests (enforced `--strict` + `addopts = --exit-on-no-tests` — a suite that silently passes with zero tests is invalid).
GREEN: clean tree, lockfile committed, docs render.
Perf: none. Baseline `uv sync` time recorded (<2 min).

## PR-1 — Schemas + config (contracts first)
Scope: `schemas/` full, `config.py`, YAML loader.
RED: (a) invalid values (prob=1.5, pixel_fraction=-1, empty `disclosure`) must REJECT; (b) missing RUO field must REJECT; (c) config with inconsistent thresholds (tsr_tol≤0) must REJECT; (d) mutation test — removing a validator from schemas must make tests fail.
GREEN: all contract tests pass; JSON round-trip stable.
Rule established: every later PR consumes these models — any PR adding a bypass (raw dict) fails review.

## PR-2 — Bio-optics: OD + Macenko
Scope: `rgb_to_od`, `macenko_stain_matrix`, `separate/normalize_stains`.
RED (adversarial, independent oracles): (a) SYNTHESIC-KNOWN-ANSWER: render 5000 RGB pixels from a fixed stain matrix S_known + concentrations → recovered matrix within angular tolerance (<5°/vector) of S_known; (b) CROSS-IMPLEMENTATION: skimage `rgb_from_hed`-based deconvolution agrees on H/E channel maps within tolerance on 10 fixture patches; (c) INVARIANCE: `normalize(normalize(img))` ≈ `normalize(img)` (idempotence, PSNR threshold); (d) NEGATIVE: pure-white patch must NOT produce NaN stain matrix (documented degenerate path).
GREEN: above pass; fixtures committed under tests/fixtures with provenance.
Perf: vectorized numpy path; benchmark macro (100 patches < 2 s) recorded in benchmarks/.

## PR-3 — Artifact injector (AIRAQc-8)
Scope: `ArtifactInjector`, 8 kinds, severity param, masks.
RED: (a) PARAM-RECOVERY: injected out_of_focus σ recovered via measured MTF/gradient-energy monotonicity (severity ↑ ⇒ sharpness ↓, Spearman ρ=1 on 30 samples); (b) MASK-EXACTNESS: `mask.sum()/total` == injected pixel fraction within 1%; (c) NEGATIVE: severity=0 ⇒ bitwise-identical image + empty mask; (d) no_tissue/missing_tissue produce BACK-consistent outputs (mean OD < threshold).
GREEN: pass + visual fixture sheet (one PNG per kind, committed).
UI hook: fixtures double as the report gallery later.

## PR-4 — Data layer (patient integrity)
Scope: `data.py`, cache pack, splits.
RED: (a) LEAKAGE-PROBE: loader given a deliberately overlapped patient split must RAISE (this is the anti-tautological core — the test constructs the failure it guards); (b) class distribution matches published dataset stats (±1%); (c) cache pack determinism: two builds → identical sha256; (d) shuffled-label negative control wired (used later by training tests).
GREEN: pass; dataset revision pinned in lockfile-adjacent manifest.
Perf: memory-mapped uint8 cache (15 GB), workers=8, prefetch=4 — measured epoch time recorded.

## PR-5 — Supervised training + metrics
Scope: `model.PatchClassifier`, train loop, metrics (F1, κ, ECE).
RED: (a) CHANCE-CONTROL: trainer on label-shuffled data must land at chance macro-F1 (≤0.12) — if it "learns" shuffled labels there is leakage/bug; (b) ONE-BATCH OVERFIT: 256 fixed patches reach ≥0.99 train acc in ≤50 steps (probes wiring, not quality); (c) κ cross-checked vs sklearn.cohen_kappa_score on random fixtures (bit-level); (d) determinism: seed 42 twice → identical metric JSON (hash).
GREEN: real run trains; VAL-7K macro-F1 recorded (gate applies only after PR-11).
Perf: bf16 AMP, channels_last, EMA, cosine schedule — ablation table (baseline vs AMP) in PR description.

## PR-6 — Calibration + OOD gate
Scope: `calibration.py`.
RED: (a) KNOWN-ECE: synthetically scaled logits with analytically computable ECE → measured ECE matches within 0.005, and fitted temperature ≈ the injected scale factor (±5%); (b) OOD SEPARATION: PR-3 artifacts as OOD vs clean val — energy AUROC ≥ 0.9 gate (if not, threshold policy fails loudly, not silently); (c) NEGATIVE: temperature fitting on already-calibrated logits must return T≈1.
GREEN: pass; temperature + energy threshold persisted to checkpoint.
Perf: calibration LBFGS on val split < 30 s.

## PR-7 — Artifact gate + biomarkers (TSR/TIL)
Scope: `biomarkers.py` + gate integration.
RED: (a) ANALYTIC GRID: hand-built 10×10 patch grid with known compartment layout → TSR recovered EXACTLY (to float tolerance), TIL densities exact by construction (counts/area formula computed independently in test, not by module code); (b) CONTAMINATION PROBE: inject artifact patches INSIDE tumor region → TSR must be unchanged (gate masks them) — a naive implementation fails this by construction; (c) EDGE: all-STR slide → TSR=None + schema-valid; disclosure string mandatory.
GREEN: pass.
UI hook: SlideReport consumes these — field list frozen here.

## PR-8 — WSI pipeline
Scope: `wsi.py`.
RED: (a) DETERMINISM: two tilings of fixture WSI → byte-identical patch set (hash); (b) MASK ORACLE: fixture slide with hand-annotated tissue bbox (committed) → mask IoU ≥ 0.9; (c) COORD ROUND-TRIP: every patch read at (x,y) hashes equal to direct crop of the level image; (d) openslide-missing fallback: tifffile path yields same coords.
GREEN: pass; one VAL-7K slide processed end-to-end.
Perf: level-0 avoided where possible; downsample-first masking.

## PR-9 — Attention-MIL slide head
Scope: `model.AttentionMIL`.
RED: (a) HIDDEN-KEY BAG: synthetic bags where slide label depends on ≥3 "key" instances among 200 noise instances → trained head ≥95% and argmax-attention instances ∩ key set ≥ 2/3 — a mean-pooling baseline is run and must score lower (adversarial baseline comparison); (b) PERMUTATION INVARIANCE: bag order shuffle → identical slide logit (≤1e-6); (c) single-instance bag sanity.
GREEN: pass; VAL-7K slide-level accuracy beats patch-majority baseline (else MIL not justified — recorded honestly if not).
Perf: attention over ≤2000 patches/slide — memory O(n), no full NxN.

## PR-10 — ONNX export + bench
Scope: `export/onnx_export.py`.
RED: (a) PARITY: torch vs ONNX argmax agreement ≥ 99.9% on 2k val patches per precision; max|Δlogit| within tolerance (fp32 1e-2, fp16 5e-2); (b) BENCH-VALIDITY: throughput measured with warmup + CUDA sync (a naive timer test that reports wrong numbers on an intentionally synchronized-sleep stub must FAIL — timer oracle); (c) VRAM ceiling enforced: int8 peak ≤ 2048 MB else PR fails; (d) dynamic batch: batch 1..32 all run.
GREEN: BenchResult JSONs committed.
Perf: IOBinding + CUDA EP graph opts; int8 calib set fixed seed.

## PR-11 — Regression gate (`eval_gate`)
Scope: `gates/eval_gate.py`.
RED (the gate must be able to FAIL — PRD acceptance #2): (a) INJECTED-FAILURE: feed metrics from a deliberately corrupted checkpoint (labels shifted at predict time) → gate MUST exit 1 with named failing checks; (b) INJECTED-DRIFT: fabricate int8 metrics with TSR drift 0.3 → biomarker check fails; (c) THROUGHPUT: fake bench with 5 patches/s → fails floor; (d) NO-COMPARISON: gate must not pass when metrics files are missing (fail-closed, not skip); (e) exit-code contract tested from subprocess.
GREEN: real artifacts → PASS expected; report JSON schema-validated.
CI: `make gate` becomes blocking; every later PR runs it.

## PR-12 — Report rendering
Scope: `report.py`.
RED: (a) RUO string present in md AND html; (b) disclosure footnote for TSR present and non-empty; (c) donut PNG is valid non-empty base64, renders (PIL open); (d) NEGATIVE: report of all-BACK slide has no TSR section and still validates; (e) template mutation test (drop footer → test fails).
GREEN: sample report committed as artifact.

## PR-13 — FastAPI service
Scope: `serve/api.py`.
RED: (a) CONTRACT: /analyze happy path returns schema-valid SlideReport (response_model enforced); (b) NEGATIVE INPUTS: bad path → 404; malformed JSON → 422; (c) /health carries model_revision + RUO; (d) CONCURRENCY: 8 parallel requests on one model instance under lock — no VRAM double-alloc (measured).
GREEN: `uv run histogate-serve --api` serves.
Perf: model lazy-load + warm; single worker documented.

## PR-14 — FastMCP server
Scope: `serve/mcp.py`.
RED: (a) tool schemas match contracts (via client list_tools); (b) e2e: in-proc client calls analyze_slide on fixture → valid SlideReport; (c) run_regression_gate tool returns GateReport with pass/fail consistent with artifact; (d) RUO in tool descriptions.
GREEN: `histogate-serve --mcp` runs stdio; smoke client in tests.

## PR-15 — Validation docs + publish surface
Scope: docs/validation.md (IEC 62304-shaped traceability), README technical report, HF model card, Makefile publish.
RED: (a) every FR has a row with test id + result link — missing row fails a doc-lint script; (b) forbidden-terms grep (₹, "clinical use", "diagnostic" outside RUO disclaimer, CGPA none by construction) — CI fails on hit; (c) all citation URLs resolve (checked by script against committed list).
GREEN: publish dry-run; then real push.

## PR-16 (optional, time-permitting) — SSL phase
Scope: `ssl.py`.
RED: (a) STAIN-INVARIANCE: same patch renormalized to 5 random references → embeddings pairwise cos ≥ 0.9 while random-patch pairs ≤ 0.5 (discriminative control — invariance alone is trivially achievable by collapse, this forbids collapse); (b) SSL-init beats scratch on 10%-label subset (else SSL not justified — recorded honestly). GREEN + re-run gate.

## Sequencing rationale
PR-1→4 build the deterministic foundation (all pure-function, testable without GPU). PR-5→9 are the GPU stages ordered by risk (calibration before biomarkers; MIL last as newest component). PR-10→11 harden deployment. PR-12→14 expose. PR-15 ships. A one-night fast path exists: PR-0..7 + 10..13 (skip 8/9/16 if time breaks — documented fallback, PRD §7).
