# HistoGate — High-Level Design (HLD)

Version: 0.1 · Status: DRAFT for approval · Reads with: PRD.md, LLD.md

---

## 1. System context

```mermaid
flowchart LR
  subgraph Sources
    Z1[NCT-CRC-HE-100K\nnormalized]
    Z2[NCT-CRC-HE-100K-NONORM\ncross-scanner]
    Z3[CRC-VAL-HE-7K\npatient-disjoint WSI]
  end
  subgraph HistoGate
    A[Data & Optics layer] --> B[Learning layer]
    B --> C[Inference & Biomarker layer]
    C --> D[Gate layer]
    C --> E[Serving layer]
  end
  Z1 & Z2 & Z3 --> A
  D -->|PASS/FAIL\nexit code + report| CI[CI / make gate]
  E -->|REST| P3[Platform engineer]
  E -->|MCP tools| M[Any MCP client\n(incl. pathology report integrations)]
  C -->|findings JSON +\nMarkdown report| H[Human reader]
  B -->|weights + model card| HF[Hugging Face Hub]
```

One workstation (Ryzen 7 5700X, 64 GB RAM, RTX 3060 12 GB) runs everything: training, gating, serving. No external services except HF Hub (pull datasets, push weights).

## 2. Layered component view

| Layer | Components | Responsibility |
|---|---|---|
| L1 Data & Optics | `data.py`, `bio_optics.py`, `wsi.py` | Dataset loads w/ patient integrity; OD deconvolution + Macenko normalization; AIRAQc-8 synthetic artifact injection; WSI pyramid tiling + tissue mask |
| L2 Learning | `ssl.py`, `model.py`, `calibration.py` | Stain-invariant SSL; supervised 9-class fine-tune; attention-MIL slide head; temperature scaling + energy OOD |
| L3 Inference & Biomarkers | `biomarkers.py`, `report.py` | Artifact gate → masked scoring; TSR/TIL from compartment maps; findings JSON → draft report |
| L4 Gates | `gates/eval_gate.py`, `export/onnx_export.py` | ONNX FP32/FP16/INT8 + benchmark; CI regression gate (can FAIL) |
| L5 Serving | `serve/api.py` (FastAPI), `serve/mcp.py` (FastMCP) | REST + MCP tool exposure; Pydantic-validated contracts everywhere |
| L0 Cross-cutting | `schemas/` (Pydantic v2), `config.py` (pydantic-settings), `cli.py` | Single typed contract language; env+YAML config; console entry points |

Dependency rule: strictly downward (L5→L3→L2→L1; gates may consume all). `schemas` imports nothing internal. No layer skips.

## 3. Technology stack & rationale

| Tech | Role | Why chosen |
|---|---|---|
| **uv** | Project, venv, lockfile, runs | Fast, lockfile reproducibility (FR-18); author's standard tooling |
| **Python 3.12** | Runtime | Match workstation toolchain |
| **Pydantic v2 + pydantic-settings** | All data contracts, config | One schema language from config → findings JSON → API models → MCP tools; validation at every boundary |
| **PyTorch + timm** | Training, backbones | ConvNeXt-Tiny/ViT-S with one-line swap; CUDA on 3060 |
| **HF datasets + hub** | Data load, weight/model-card publication | Ungated dataset pulls; push artifact = public footprint (FR-1, G9) |
| **OpenSlide (+ tifffile fallback)** | WSI pyramid IO | Pathology-standard; JD-A gigapixel axis |
| **ONNX Runtime** | Quantized inference + bench | FP16/INT8 + latency/VRAM measurement; TensorRT-free portability |
| **FastAPI** | REST serving | Pydantic-native; standard |
| **FastMCP** | MCP tool server | Author's production domain (Kai/ctx-gateway MCP gateway); exposes analysis as agent-callable tools — the P3 integration doctrine made literal |
| **pytest / ruff** | Test + lint | Gate-quality CI hygiene |

Not used (recorded decisions): Transformers trainer (timm + light loops suffice at this scale; less magic to defend in review), MLflow/W&B (local JSONL + gate reports — no external services), Docker (single-workstation scope; revisit if AIRA asks).

## 4. End-to-end data flow

1. `uv run histogate-train` → L1 loads splits → L2 SSL (optional) → supervised fine-tune → checkpoint + metrics JSONL under `runs/`.
2. `uv run histogate-export` → ONNX FP32/FP16/INT8 + calibration set for INT8 + benchmark table → `benchmarks/*.json`.
3. `uv run histogate-eval` → evaluates all precisions on VAL-7K + NONORM → per-domain metrics (F1, κ, ECE) + biomarker tables.
4. `uv run histogate-gate` → consumes metrics/bench artifacts → GateReport (PASS/FAIL, exit code) → CI-blocking.
5. `uv run histogate-serve` (or `--mcp`) → loads quantized model → answers `/analyze`, `/report` or MCP tools → SlideReport JSON + Markdown.
6. `make publish` → HF Hub model card + weights (public; RUO + no-money-figures compliant).

## 5. Interface catalogue

| Surface | Contract | Defined in |
|---|---|---|
| CLI | `histogate-{train,eval,gate,export,serve}` | LLD §12 |
| REST | `GET /health`, `POST /analyze`, `POST /report` | LLD §13 |
| MCP | tools: `analyze_slide`, `compute_biomarkers`, `run_regression_gate`, `get_report` | LLD §14 |
| Artifacts | `SlideReport`, `GateReport`, `BenchResult` JSON files under `runs/`, `benchmarks/` | LLD §2 |
| Public | HF model card, README technical report | PRD G9 |

## 6. Repository layout

```
histogate/
├── pyproject.toml        # uv/hatchling; deps; console scripts
├── uv.lock
├── Makefile              # setup/train/ssl/eval/export/gate/serve/mcp/test/lint/publish
├── docs/                 # PRD.md, HLD.md, LLD.md, validation.md (IEC 62304-shaped, RUO)
├── configs/              # default.yaml (+ experiment overrides), Pydantic-validated
├── src/histogate/
│   ├── config.py         # Settings (pydantic-settings)
│   ├── schemas/          # contracts (PRD FR-14/17)
│   ├── bio_optics.py data.py wsi.py          # L1
│   ├── ssl.py model.py calibration.py        # L2
│   ├── biomarkers.py report.py               # L3
│   ├── export/onnx_export.py                 # L4a
│   ├── gates/eval_gate.py                    # L4b
│   ├── serve/api.py serve/mcp.py             # L5
│   └── cli.py                                # entry points
├── scripts/              # one-off analysis/experiment scripts
├── tests/                # unit + gate-failure integration tests
├── benchmarks/           # bench JSON artifacts
└── runs/                 # gitignored: checkpoints, metrics, gate reports
```

## 7. Decision log

| ID | Decision | Rationale / rejected alternative |
|---|---|---|
| AD-1 | src-layout uv package | installable, testable, lockfile-reproducible vs flat scripts |
| AD-2 | Pydantic v2 as the single contract language | schema drift between API/MCP/artifacts is a classic failure; one source of truth |
| AD-3 | MCP server alongside REST | author's differentiation (Kai lineage) + AIRA's integration doctrine; near-zero cost over FastAPI |
| AD-4 | Artifact gate as separate model/mask stage, not extra classes | mirrors PMC8262119 dual pipeline; biomarkers must not see artifact pixels |
| AD-5 | TSR patch-proxy disclosed, not slide-protocol claim | honesty as design principle (PRD risk table) |
| AD-6 | ONNX Runtime over TensorRT | portability + no engine build on 3060 driver stack; TensorRT mention reserved for interview |
| AD-7 | SSL optional phase | protects release-critical path on a one-night budget (PRD §7) |
| AD-8 | Local JSONL metrics, no W&B | no accounts, no external dependency, gate artifacts live in-repo runs/ |

## 8. Regulatory & publication posture

Research Use Only. Every public surface (README, model card, API `/health`, MCP tool descriptions) carries the RUO line. `docs/validation.md` will carry an IEC 62304-shaped traceability matrix (requirement → test → result) — aimed at AIRA's actual "Research only → CE/IVD" frontier (round 4), and bridging the author's Ashva regulated-device history. No PHI possible: public datasets only. No absolute money figures in any public artifact.
