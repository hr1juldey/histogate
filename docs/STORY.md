# The HistoGate story

> **The short version:** training a classifier is a commodity. What follows
> is the diagnostic loop — run → break → measure → recover — that this
> repository exists to demonstrate. Every claim resolves to a measured
> artifact in [RESULTS.md](RESULTS.md) or a commit.

---

## Act one — built to be trusted before built to be good

Before any training run, the pipeline got two paranoia features. A
**patient-leak guard** refuses any split where the same patient appears
twice — otherwise a great score is memorization wearing a lab coat. And a
**chance-control switch** retrains the entire network on shuffled labels:
the pipeline must score at chance when there is no signal, proving the
scoreboard cannot be cheated. Only then did a real model get trained.

## Act two — the gate bites

The first real model aced the external validation set. Then our own release
gate — nine checks, each designed to be *able to fail* — caught it
collapsing on raw, un-normalized images from real scanners. The benchmark
ships stain-normalized data, and a model trained only on cleaned color
learns the cleaning, not the tissue. This is a documented failure mode of
this dataset; we reproduced it, measured it, and shipped the failure rather
than the excuse.

## Act three — the fix that broke

Fine-tuning on raw scans at the default learning rate destroyed the model
within a handful of epochs — recorded, not reverted. The best-checkpoint
logic rescued the one useful epoch, and that rescued model passed the
domain check but **regressed the out-of-distribution detector**. Lesson,
now a standing rule here: every fix has a bill; measure the whole
scorecard, not one number.

## Act four — the honest ship

The shipping model is the controlled re-run: a gentle learning rate, one
effective epoch of raw-scanner exposure. It clears external accuracy,
calibration, throughput, memory — and int8 integrity, where the
quarter-size deployment model moves the biomarker readings by exactly
zero. It also **fails two checks on purpose-reported thresholds**, both
documented in the results table: a domain margin a hair inside sampling
noise, and an OOD detector whose honest scope finding became the ticket
for the QC-head work, not an excuse to lower the bar.

## Act five — physics, not pixel hacks

The final training round replaces "get more data" with "understand the
data." Each image is decomposed into its actual dye concentrations
(hematoxylin/eosin, Beer–Lambert optics, Macenko basis), and the training
jitter draws from ranges **measured on real scanner slides** —
within-tissue-class, because cross-class concentration spread is biology,
not staining. The augmentation had to survive adversarial tests written
before it existed: cover a held-out scanner's stain statistics without
leaving real stain space, stay indistinguishable from real images to a
blind discriminator, preserve the classifier's error rate inside a band,
and scale the hematoxylin dye by half again as much — measuring back
exactly as applied, with zero leak into eosin. It failed three designs
before passing all of them. Its measured verdict is in the results table
alongside every earlier run.

---

## The engineering underneath (one line each)

- **Numbers law** — a CI job regenerates the results table from measured
  artifacts and fails if any number was typed by a human.
- **Crash-proof execution** — the roadmap runs as a recoverable step loop
  ([EXEC_ENGINE.md](EXEC_ENGINE.md)): call → do → test → pass/fail → log.
- **Deployment integrity** — int8 quantization with a CPU-runtime-compatible
  operator set; biomarker outputs bit-identical to full precision on
  clinically composed test slides.
- **Fail-closed serving** — blank or tissue-free input is rejected at
  tiling time with an explicit error, earlier and louder than any
  confidence threshold.
