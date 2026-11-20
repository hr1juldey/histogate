# HistoGate — Test Standards & Statistical Gates

Version: 0.1 · Binding for every PR (PLAN.md).

---

## 1. Anti-tautology rules (a test that cannot fail is not a test)

1. **Independent oracle required.** Every correctness test compares against an oracle that does not share code with the implementation: analytic construction, published constants, a second implementation (skimage/sklearn), or a hand-built fixture. FORBIDDEN: re-implementing the module's formula inside the test and asserting equality.
2. **Negative controls mandatory.** Any learning claim ships with a control that must NOT learn/predict: label-shuffled training (→ chance), severity-0 injection (→ identity), already-calibrated logits (→ T≈1), mean-pool baseline (→ must lose to attention MIL on key-instance bags).
3. **Failure injection mandatory for gates.** Gate code is only tested by making it fail (corrupted model, drifted metrics, slow bench, missing files → fail-closed). A gate test that only ever sees PASS is invalid.
4. **Determinism tests are hash tests**, not "close enough" eyeballing: two runs → identical sha256 of metric JSON.
5. **Fixtures carry provenance** (source, creation script, seed). No fixture is edited by hand post-generation.
6. **Coverage is hygiene, not evidence.** ≥85% line coverage on src/, but PR review judges oracle quality, not coverage numbers.
7. **No test may assert on a mock's return of the unit under test.** Mocks only at true boundaries (filesystem, HF hub, GPU absence).

## 2. Test taxonomy

| Type | Runs where | Time budget |
|---|---|---|
| Unit (pure functions: optics, schemas, biomarkers math) | CPU, every commit | <60 s total |
| Fixture integration (WSI, report, api/mcp) | CPU, every commit | <5 min |
| GPU tests (training wirings, parity, bench) | local RTX 3060, pre-merge for GPU PRs | <20 min |
| Gate (full eval + bench + regression) | local, nightly + pre-release | ≤90 min |
| Doc-lint (traceability rows, forbidden terms, link check) | CPU, every commit | <30 s |

## 3. Statistical gates for models (single source: config `gate:` block)

**Protocol.** Eval sets fixed by PR-4 split manifest. Metrics: macro-F1, quadratic-κ, ECE (15-bin), per-class recall. Model selection on val only; gate on untouched sets. Seeds: 3 seeds for supervised final; gate uses WORST seed (not mean — worst-seed gating prevents seed-shopping).

| Gate | Check | Threshold (initial; recalibrated after PR-5 first honest run, then frozen) |
|---|---|---|
| G-ACC | macro-F1, CRC-VAL-HE-7K, fp32, worst seed | ≥ 0.93 |
| G-BASE | macro-F1 vs color-histogram + logistic-regression baseline | model − baseline ≥ +0.10 (beating a trivial feature baseline is the entry ticket, not a bonus) |
| G-DOMAIN | macro-F1 drop, normalized → NONORM | ≤ 0.015 |
| G-CAL | ECE after temperature scaling (both domains) | ≤ 0.05 |
| G-OOD | energy AUROC, clean vs PR-3 artifacts | ≥ 0.90 |
| G-TSR | TSR vs analytic-grid fixtures | exact (float tol); after INT8: abs drift ≤ 0.05 |
| G-TIL | TIL density drift after INT8 | ≤ 10% relative |
| G-QUANT | argmax agreement torch↔ONNX (fp16/int8, 2k patches) | ≥ 99.9% |
| G-THRUPUT | patches/s, ONNX fp16, batch 32 | ≥ 40 |
| G-VRAM | peak VRAM, int8 serving path | ≤ 2048 MB |

**Statistical hygiene:** F1 differences quoted with 95% bootstrap CI (1000 resamples, committed RNG); model comparisons use McNemar's test on paired predictions (p<0.05) — "we beat baseline" without a paired test is rejected in review. If G-DOMAIN fails on honest data, the result is REPORTED, not tuned away (PRD §7); the gate report documents it and the release is labeled accordingly.

**Calibration correctness:** ECE implementation cross-validated against a published reference implementation on synthetic logits with analytically known ECE (PR-6 RED-a) before any model ECE is believed.

## 4. Review checklist (per PR)

- [ ] Every new test names its oracle in a comment
- [ ] ≥1 negative control or failure-injection per behavioral claim
- [ ] No threshold constants inside tests — all from config/fixtures
- [ ] Determinism: new randomness seeded + hash-checked or justified
- [ ] Perf note: measured impact on the PR's bench metric
- [ ] Forbidden terms clean; RUO/disclaimers present on new surfaces
