# HistoGate — Low-Level Design (LLD)

Version: 0.1 · Status: DRAFT for approval · Reads with: PRD.md, HLD.md, PLAN.md, TESTING.md

---

## 1. Cross-cutting: config (`config.py`)

Pydantic-settings `Settings(BaseSettings)` loaded from env (`HISTOGATE_`) + `configs/default.yaml`. Fields: paths (data_dir, runs_dir, benchmarks_dir), dataset ids + revision pin, split policy, train hyperparams (batch=256, epochs, lr=1e-3 cosine, label_smoothing=0.1, amp="bf16", ema=0.999, workers=8, seed=42), ssl block (enabled=false default, epochs=15, proj_dim=256, temp=0.5), calibration (energy_threshold fitted on val), gate thresholds (see TESTING §3 — single source; gate reads defaults from here, override by YAML), serving (host/port, precision="fp16", max_batch=32). Validation: thresholds cross-checked (e.g., tsr_tol within (0,0.2)); frozen post-init.

## 2. Contracts (`schemas/`) — Pydantic v2, field-level

```python
class TissueClass(str, Enum):  # Kather 2019
    ADI="ADI"; BACK="BACK"; DEB="DEB"; LYM="LYM"; MUC="MUC"
    MUS="MUS"; NORM="NORM"; STR="STR"; TUM="TUM"

class ArtifactKind(str, Enum):  # AIRAQc taxonomy, exactly 8
    TISSUE_FOLD="tissue_fold"; PEN_MARK="pen_mark"; AIR_BUBBLE="air_bubble"
    COVERSLIP_LINE="coverslip_line"; KNIFE_LINE="knife_line"          # processing
    OUT_OF_FOCUS="out_of_focus"; NO_TISSUE="no_tissue"; MISSING_TISSUE="missing_tissue"  # scanning

class PatchPrediction(BaseModel):
    tissue: TissueClass; prob: float=Field(ge=0, le=1)
    energy: float; temperature: float; is_ood: bool; artifact: ArtifactKind | None

class TSRResult(BaseModel):
    stroma_fraction: float=Field(ge=0,le=1); tumor_fraction: float=Field(ge=0,le=1)
    tsr: float=Field(ge=0,le=1); group: Literal["stroma-high","stroma-low"]  # van Pelt ≥50% cut
    method: Literal["patch-proxy-vanpelt-2018"]; disclosure: str  # REQUIRED non-empty

class TILResult(BaseModel):
    intratumoral_density_mm2: float; stromal_density_mm2: float
    mpp: float=0.5; method: Literal["compartment-saltz-2018-patch-proxy"]

class ArtifactFinding(BaseModel):
    kind: ArtifactKind; stage: Literal["processing","scanning"]
    pixel_fraction: float=Field(ge=0,le=1)

class SlideReport(BaseModel):
    slide_id: str; n_patches: int; n_gated: int
    compartment_fractions: dict[TissueClass, float]
    artifacts: list[ArtifactFinding]; tsr: TSRResult | None; til: TILResult | None
    slide_label: str | None; slide_confidence: float | None
    calibration_temperature: float; model_revision: str
    ruo_disclaimer: Literal["Research Use Only. Not for use in diagnostic procedures."]

class GateCheck(BaseModel):
    name: str; value: float; threshold: float
    comparator: Literal["<=",">=",">","<"]; passed: bool; detail: str

class GateReport(BaseModel):
    overall_pass: bool; checks: list[GateCheck]
    model_revision: str; dataset_revision: str; timestamp: str

class BenchResult(BaseModel):
    precision: Literal["fp32","fp16","int8"]
    macro_f1: float; ece: float; patches_per_s: float
    p95_latency_ms: float; peak_vram_mb: float
```

Rules: no default-None escape hatches on report fields except TSR/TIL (absent when no tumor/stroma present — explicit nullable semantics, documented). Every public JSON validates through these models only.

## 3. `bio_optics.py`
- `rgb_to_od(img) -> OD`: OD = -log10((I+1)/256)/ per channel (Ruifrok–Johnston).
- `macenko_stain_matrix(OD, alpha=1, beta=0.15) -> S(2×3 or 3×3)`: percentile-threshold OD → SVD of OD covariance → plane basis → project all pixels → angle percentiles (1st/99th) → 2 stain vectors (H first by hue order); unit-normalize.
- `separate_stains(img, S) -> concentrations` (pseudo-inverse), `normalize_stains(img, S_ref)`: transfer concentrations to reference matrix (Ruifrok standard H&E reference), recompose.
- `ArtifactInjector`: per-kind deterministic (seeded) parameterized transforms — tissue_fold (affine fold + brightness seam), pen_mark (thick stroke polygon, dark), air_bubble (radial white disc + rim shadow), coverslip_line (diagonal band), knife_line (parallel fine lines), out_of_focus (Gaussian σ sweep), no_tissue/missing_tissue (blank/dropout patches). API: `inject(img, kind, severity∈[0,1]) -> (img, mask)`.

## 4. `data.py`
HF `datasets` loader with `trust_remote_code=False`; verifies the three splits; builds patch index `(split, patient_id, path, label)` — patient IDs parsed from filenames (NCT-CRC-HE-100K encodes patient in filename; CRC-VAL-HE-7K slide-level dirs). Exposes `PatientDisjointSampler` and a `--assert-no-patient-overlap` check run in tests and CI. In-RAM cache: uint8 tensor pack (≈15 GB for 100k×224²×3) memory-mapped to `data/cache.bin` with offset index; 64 GB RAM budget documented.

## 5. `ssl.py` (optional phase)
SimCLR: backbone-frozen-free ResNet-18/ConvNeXt-T + 2-layer MLP projector; views = OD-space stain perturbations (stain-vector jitter ±σ, concentration scaling, Macenko renormalize to random reference) + mild geometry; NT-Xent, temp=0.5; bf16 AMP. Output: `ssl_backbone.ckpt` consumed by supervised stage as init.

## 6. `model.py`
- `PatchClassifier(backbone="convnext_tiny", pretrained=True)` → 9 logits + features (768-d).
- `AttentionMIL(L=768, A=128, K=1)` gated attention (Ilse 2018): `a_k = softmax(wᵀ(tanh(Vh)⊙sigm(Uh)))`, slide logit = Σ a_k · h_k through linear head. Trained on VAL-7K bags (slide label from dominant tissue/normal status), BCE.
- Checkpoint schema: `{state_dict, config_hash, git_sha, metrics}` — no orphan checkpoints.

## 7. `calibration.py`
- Temperature scaling: single param T fit on val NLL (LBFGS, 200 iters); `temperature` stored in checkpoint + reports.
- Energy score: `E(x) = -T·logsumexp(logits/T)`; threshold = quantile(q) on val clean-energy so that artifact-injected val set FPR ≤ target; `is_ood = E > thr`.

## 8. `wsi.py`
`open_slide(path)` (openslide; tifffile fallback): read level matching ≈0.5 MPP; Otsu on downscaled grayscale + OD saturation mask → tissue regions; grid 224px stride 224 (contiguous; overlap configurable); coordinates → `PatchCoords list[(x,y)]`; `read_patch(slide, x, y)` with stain-normalize flag. Determinism: same slide → byte-identical patch set (tested).

## 9. `biomarkers.py`
Inputs: patch grid of `PatchPrediction` + coords. Artifact/BACK/NO_TISSUE patches masked (FR-7 gate). Tumor-adjacency = STR/TUM patch 8-neighborhood on the grid.
- TSR: over tumor-adjacent STR+TUM patches: `tsr = area(TUM)/(TUM+STR)`; group by 50% cut (van Pelt). `disclosure` fixed string: patch-level proxy of the slide-margin protocol.
- TIL: `intratumoral = count(LYM∩tumor-adjacent)/area_mm2`, `stromal = count(LYM∩STR)/area_mm2`; area from coords×mpp.
Outputs `TSRResult`, `TILResult`.

## 10. `report.py`
`SlideReport → markdown` (template in `report.py`): verdict banner (CLEAN / GATED: n patches), artifact table, compartment donut (matplotlib → base64 inline), TSR/TIL cards with disclosure footnote, uncertainty note, RUO footer. Also `SlideReport → HTML` (self-contained single file, same sections) for `/report`.

## 11. `export/onnx_export.py`
torch → ONNX (opset 17, dynamic batch axis, fixed 224 input). FP16 via `onnxconverter_common.float16`. INT8 static via `onnxruntime.quantization.quantize_static` (calibration = 512 val patches, per-channel, QDQ). Bench harness: warmup 20, timed 200 batches, CUDA EP, IOBinding; records `BenchResult` per precision. Parity check: max |Δlogit| vs torch ≤ 1e-2 (fp32), ≤ 5e-2 (fp16), agreement of argmax ≥ 99.9% (int8 on val subset).

## 12. `gates/eval_gate.py`
Consumes metrics JSONs (eval per domain × precision) + bench JSONs → evaluates check list (TESTING §3) → `GateReport` → `runs/gate_<ts>.json`; exit code 0/1. Hard rule: thresholds live ONLY in config; gate has no embedded constants.

## 13. `serve/api.py` (FastAPI)
`GET /health` → `{status, model_revision, ruo_disclaimer}`. `POST /analyze {slide_path | patch_dir}` → 202-style job or sync (config) → `SlideReport`. `POST /report` → Markdown + HTML strings. All request/response models from `schemas`. Errors: 422 (pydantic), 404 unknown slide, 503 model not loaded. Single quantized model instance, lazy-loaded under lock.

## 14. `serve/mcp.py` (FastMCP)
Tools (typed by the same schemas): `analyze_slide(slide_path) -> SlideReport`, `compute_biomarkers(slide_path) -> {tsr, til}`, `run_regression_gate() -> GateReport`, `get_report(slide_id, fmt) -> str`. stdio transport default. Descriptions carry RUO.

## 15. `cli.py`
`histogate-train [--config X] [--ssl]`, `histogate-eval [--precisions fp32,fp16,int8]`, `histogate-export`, `histogate-gate`, `histogate-serve [--api|--mcp]`. Thin: parse → call module → write artifact. No business logic in CLI.

## 16. Traceability (FR → module → test)

| FR | Module | Test file | Adversarial core |
|---|---|---|---|
| FR-1 | data.py | test_data.py | injected-overlap must FAIL loader |
| FR-2 | bio_optics.py | test_bio_optics.py | synthetic stain round-trip vs known matrix |
| FR-3 | bio_optics.py | test_artifacts.py | param recovery + detector floor |
| FR-4 | model.py/train | test_train.py | one-batch overfit + chance-control |
| FR-5 | ssl.py | test_ssl.py | invariance: renormalized view same embedding (cos≥0.9) |
| FR-6 | calibration.py | test_calib.py | synthetic miscalibration with KNOWN ECE recovered |
| FR-7 | biomarkers.py | test_biomarkers.py | contamination probe must not leak into TSR |
| FR-8/9 | biomarkers.py | test_biomarkers.py | synthetic grid → exact analytic TSR/TIL |
| FR-10 | wsi.py | test_wsi.py | tiling determinism + mask oracle fixture |
| FR-11 | model.py | test_mil.py | hidden-key-instance bag solved by attention |
| FR-12 | export | test_export.py | parity + bench sanity (VRAM ceiling enforced) |
| FR-13 | gates | test_gate.py | corrupted-model injection MUST fail gate |
| FR-14 | report.py | test_report.py | schema + RUO + disclosure presence |
| FR-15/16 | serve/* | test_api.py, test_mcp.py | contract + negative-input tests |
| FR-17/18 | config.py | test_config.py | threshold validation + seed determinism |
