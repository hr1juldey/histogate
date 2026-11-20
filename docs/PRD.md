# HistoGate — Product Requirements Document (PRD)

Version: 0.1 · Status: DRAFT for approval · Owner: Hrijul Dey
Companion docs: `HLD.md` (architecture), `LLD.md` (module specs). Source research: `../../HISTOGATE_V2.md`, `../../rounds/`.

---

## 1. Problem statement

Computational pathology pipelines fail silently in two places:

1. **Pre-analytical/scan artifacts** (tissue folds, pen marks, air bubbles, coverslip lines, knife lines, out-of-focus regions, missing/no tissue) corrupt downstream inference, and labs detect them by manual review. Commercial QC (AIRA Matrix AIRAQc) lists exactly 8 artifact classes and reports 95% sensitivity / 96% specificity at 10 WSI/min.
2. **Tissue microenvironment quantification** (tumor–stroma ratio, tumor-infiltrating lymphocytes) is scored by humans with moderate inter-rater agreement (κ ≈ 0.56 among 5 pathologists, 500 slides — AIRA-cited Tox Path 2024 study), while these biomarkers are prognostically decisive in colorectal cancer.

HistoGate is a **research-use-only, end-to-end, reproducible system** that (a) gates artifacts before scoring — the dual artifact/scoring pipeline AIRA itself published (Tox Path 2021, PMC8262119) — and (b) quantifies TME compartments and biomarkers with calibrated, uncertainty-aware predictions across scanner/stain domains.

Secondary objective (explicit): serve as a public, measurable portfolio artifact for the author's application to AIRA Matrix's AI Engineer role. Every design decision traceable to AIRA's published products/papers.

## 2. Users

| Persona | Need |
|---|---|
| P1 Research engineer (primary) | Reproduce training/eval on 1× RTX 3060 12 GB; run regression gates in CI; extend |
| P2 Pathology lab operator (demo persona) | Drop WSI/patches in → get QC verdict + TME report (JSON + human-readable draft report) |
| P3 Platform/AI engineer (integration persona) | Call analysis over REST or MCP tools; embed findings into structured reporting (mTuitive/Lumea-style doctrine: "AI is only as valuable as what happens to its output") |
| P4 Reviewer (AIRA hiring manager) | Verify rigor: methods cite primary literature, gates can fail, limitations disclosed |

## 3. Goals / Non-goals

**Goals**
- G1 9-class tissue compartment classification on NCT-CRC-HE-100K, validated patient-disjoint (CRC-VAL-HE-7K) and cross-scanner (NONORM split).
- G2 Artifact detection/gating across AIRAQc's exact published 8-class taxonomy (synthetic + dataset BACK/DEB stress classes).
- G3 TSR (van Pelt 2018 protocol, patch-proxy disclosed) and TIL density (Saltz 2018 compartment style) biomarkers.
- G4 Whole-slide pipeline: OpenSlide tiling → tissue mask → patch inference → attention-MIL slide aggregation.
- G5 Stain-invariant SSL pretraining with OD-space augmentations.
- G6 Calibrated predictions (temperature scaling; ECE reported) + energy-based OOD/artifact rejection.
- G7 Quantized deployment: ONNX FP16/INT8 with a CI regression gate that can FAIL (F1/ECE drop, biomarker drift, throughput, VRAM).
- G8 Serving: FastAPI REST + FastMCP tool server; findings JSON → draft synoptic report.
- G9 Publication: HF model card + weights + README-as-technical-report with citations.

**Non-goals**
- Clinical/diagnostic use (RUO disclaimer everywhere).
- Gleason grading, prostate, IHC, cytology (AIRA's lanes — out of scope).
- Training beyond 12 GB VRAM / >1 GPU; >12 h total training budget.
- Pathologist-in-the-loop labeling UI.
- Any claim of clinical validity; TSR proxy status always disclosed.

## 4. Functional requirements

| ID | Requirement | Source/justification |
|---|---|---|
| FR-1 | Load NCT-CRC-HE-100K (normalized), NCT-CRC-HE-100K-NONORM (cross-scanner), CRC-VAL-HE-7K (patient-disjoint) via HF `datasets`, with strict split integrity and patient-level separation | zenodo 1214456; HF DykeF/NCTCRCHE100K |
| FR-2 | RGB→OD transform (Ruifrok–Johnston 2001) and Macenko 2009 stain-vector estimation; stain normalization + OD-space augmentation API | primary optics literature |
| FR-3 | Synthetic artifact injector covering AIRAQc's 8 classes with per-artifact physical parameters | AIRAQc product page (round 1) |
| FR-4 | 9-class supervised training (ConvNeXt-Tiny primary; ViT-S/16 comparison optional) | timm; Kather 2019 dataset design |
| FR-5 | SSL pretraining stage (SimCLR-style, OD-space views) on NONORM subset; optional given time | AIRATox 130k-WSI SSL; STP 2024 "pathologist paradox" poster |
| FR-6 | Temperature-scaled calibration (ECE) + energy-score OOD gating | Liu et al. energy OOD; Singhal Sci Rep 2022 uncertainty lineage |
| FR-7 | Artifact gate BEFORE scoring: artifact/blank patches masked from biomarker computation; morphological cleanup — the PMC8262119 dual-pipeline pattern | AIRA Tox Path 2021 |
| FR-8 | TSR biomarker: stroma fraction in TUM-adjacent neighborhoods, LYM/DEB excluded; TSR-low threshold ≥50% stroma; method-divergence disclosure | van Pelt 2018 (PMC6182321); da Silva 2021 |
| FR-9 | TIL biomarker: LYM density per mm² split intra-tumoral (TUM-adjacent) vs stromal (STR), at native 0.5 µm/px | Saltz 2018 Cell Reports 23:181–193 |
| FR-10 | WSI pipeline: OpenSlide pyramid read, Otsu/OD tissue mask, tile coordinates, 224px patches at 0.5 MPP, 10×/20× hooks | JD-A "50K×50K" axis; PMC8262119 per-lesion magnification table |
| FR-11 | Attention-MIL slide-level aggregation (gated attention pooling) over patch embeddings | Ilse 2018; mirrors AIRA hierarchical transformer (Erak, Mod Path 2023) |
| FR-12 | ONNX export FP32/FP16/INT8; benchmark patches/s, p95 latency, peak VRAM | AIRA HPC/CUDA JD signal; Caffe-legacy modernization story |
| FR-13 | CI regression gate (`eval_gate.py`): PASS/FAIL with exit code; three gate families (accuracy/calibration drift across domains & precisions; biomarker drift; throughput/VRAM floors) | author's Kai/ctx-gateway production doctrine |
| FR-14 | Findings JSON (Pydantic v2 schema) + Markdown draft synoptic report render | mTuitive/Lumea structured-report doctrine (round 4) |
| FR-15 | FastAPI REST service (`/health`, `/analyze`, `/report`) | standard |
| FR-16 | FastMCP tool server (`analyze_slide`, `compute_biomarkers`, `run_regression_gate`, `get_report`) | author's MCP-gateway lineage (Kai, ctx-gateway) |
| FR-17 | Full config via Pydantic-settings (env + YAML), single source of truth | engineering hygiene |
| FR-18 | Determinism: fixed seeds, versioned lockfile, pinned dataset revision | reproducibility for P4 |

## 5. Non-functional requirements & budgets

| NFR | Budget |
|---|---|
| Training wall-clock | ≤ 6 h total on RTX 3060 12 GB (supervised ≈ 1–1.5 h; SSL ≈ 2–3 h optional) |
| Inference VRAM | ≤ 2 GB (ONNX Runtime, batch ≥ 16) |
| Throughput floor | ≥ 40 patches/s FP16 batched (gate threshold; tune after first bench) |
| Acceptance macro-F1 | ≥ 0.93 patch-level on CRC-VAL-HE-7K supervised (literature ~0.96–0.97 ceiling; leave margin) |
| Cross-scanner drop | ≤ 1.5 pts macro-F1 normalized→NONORM (gate) |
| ECE | ≤ 0.05 post-temperature-scaling |
| TSR drift after INT8 | ≤ 0.05 absolute |
| TIL drift after INT8 | ≤ 10% relative |
| Disk | dataset ≈ 7–10 GB; checkpoints ≤ 1 GB |
| Safety | RUO disclaimers on every public surface; no PHI possible (public datasets only) |

## 6. Acceptance criteria (release 0.1)

1. `make train eval gate` runs green end-to-end on target hardware with a gate report artifact.
2. Gate demonstrably FAILS when a corrupted model/int8 drift is injected (test-proven — a gate that cannot fail is invalid).
3. A WSI from CRC-VAL-HE-7K produces a SlideReport JSON + Markdown report with TSR, TIL, artifacts, uncertainty.
4. FastAPI and FastMCP servers start and answer `/health` / tool list from a clean `uv sync`.
5. README renders as a technical report with all citations resolving.

## 7. Risks

| Risk | Mitigation |
|---|---|
| SSL stage eats the night | SSL is Phase-B optional; supervised + gates are the release-critical path |
| Patch-proxy TSR challenged as not "the protocol" | Disclosed in schema field + README; framed as first-order approximation |
| openslide system lib missing | tifffile fallback path for TIFF-based patches; WSI stage degrades gracefully |
| NONORM cross-scanner drop exceeds gate | That's a RESULT, not a failure — report it honestly (round 2 lesson: honesty is the differentiator) |
| Public-money law violation | No ₹ figures anywhere public; docs reviewed before push |

## 8. Traceability

Every FR maps to a module (LLD §9 traceability matrix) and to an AIRA source (column 3 above). No feature exists without a source or an explicit engineering-hygiene justification.
