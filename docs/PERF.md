# HistoGate — Performance Plan (local training + serving)

Version: 0.1 · Budgets from PRD §5. Every optimization lands INSIDE a PR from PLAN.md with a measured before/after row in the PR description — no unmeasured "optimizations".

---

## 1. Hardware reality
Ryzen 7 5700X (16 threads) · 64 GB RAM · RTX 3060 12 GB (Ampere, PCIe 4.0 x16 if main slot) · NVMe. Ampere ⇒ bf16 native (no GradScaler needed for bf16), fp16 tensor cores, ONNX CUDA EP + fp16 graph supported. INT8 via ORT static quantization is CPU-leaning — measure; if int8-on-GPU underperforms fp16, int8 is a CPU-serving story and the gate labels it so (honesty over spec-sheet).

## 2. Data path (the usual bottleneck at this scale)
| Optimization | Where | Expected effect | PR |
|---|---|---|---|
| Memory-mapped uint8 tensor cache (100k×224²×3 ≈ 14.4 GB) | data.py | epoch time bounded by GPU, not JPEG decode (decode-once cost amortized) | PR-4 |
| workers=8, pin_memory, persistent_workers, prefetch_factor=4 | loader | overlap H2D + augment with compute | PR-4 |
| Augment on GPU (stain/OD ops in torch, not PIL) | bio_optics torch path | Macenko per-batch on GPU ≈ 40× vs CPU loop | PR-2/5 |
| channels_last + bf16 autocast | train loop | ConvNeXt-T ~1.5–1.8× on Ampere | PR-5 |
| torch.compile (mode="max-autotune") — attempted, feature-flagged | train loop | optional extra ~1.2×; disabled if compile time eats the night | PR-5 |
| EMA weights instead of long cosine tail | train loop | buys robustness without extra epochs | PR-5 |

Training budget: ConvNeXt-T, 224px, batch 256, bf16 ≈ 45–75 min / 10 epochs on 3060 (target: verify in PR-5, recalibrate PLAN G-ACC threshold after).

## 3. GPU memory discipline (12 GB ceiling)
Batch 256 @ 224² ConvNeXt-T bf16 ≈ 6–8 GB — fits with EMA + optimizer states (AdamW ≈ 2× params ≈ 60 MB). SSL projector adds ~1 GB — SSL uses batch 128. Rule: OOM is a bug → measured `torch.cuda.max_memory_allocated` logged every run into metrics JSON (a run without its VRAM number is invalid for the gate).

## 4. Inference & export
| Optimization | Effect | PR |
|---|---|---|
| ONNX CUDA EP, graph optimizations all | baseline fp32 serving | PR-10 |
| fp16 graph conversion (converter keep-init-tensors) | ~1.8–2.2× throughput, VRAM ≤ 2 GB goal | PR-10 |
| IOBinding, preallocated arena, batch=32 server-side accumulation | removes per-request alloc jitter; p95 stability | PR-10/13 |
| Warmup 20 iterations before timing/before serving | first-batch CUDA jit/allocation excluded | PR-10/13 |
| int8 static (per-channel, QDQ, 512-patch calibration) | CPU-deployment story; GPU number measured & reported either way | PR-10 |
| Lazy model load + single instance under asyncio lock | one copy in VRAM regardless of request concurrency (tested PR-13) | PR-13 |

Throughput floor gate: ≥40 patches/s fp16 batched (≈1.4 slides/min at 512 patches — far below AIRAQc's 10 WSI/min commercial claim, and the README says so honestly: single consumer GPU vs server class).

## 5. Serving footprint (fast path on localhost)
uvicorn single worker (the model is the resource, not the web tier); `/analyze` streams NDJSON progress; MCP over stdio (no network stack at all). Full stack (model + FastAPI + FastMCP in one process via `histogate-serve --api --mcp`) targets < 2.5 GB VRAM total — leaves the 3060 headroom for the local LLM/VLM co-residency habit (author's 12 GB discipline).

## 6. Benchmarks that are themselves tested
Timer oracle (TESTING PR-10 RED-b): benchmark harness validated against a stub with KNOWN sleep/compute time before any GPU number is believed. Every benchmark JSON records: warmup count, batch sizes, seeds, driver/date, VRAM peak — reproducible or it didn't happen.
