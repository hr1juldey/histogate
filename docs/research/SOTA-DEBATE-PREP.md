# SOTA debate prep — multi-scale modes + cell-level detection (2026-11-22)

Scope: what the field is doing (Harvard, Stanford, Microsoft/Providence, Owkin,
Siemens, UCL, IITs, China/Japan), where the user's two proposals already exist
as established art, and the strongest arguments FOR and AGAINST each — for the
adversarial debate. Sources: web survey 2026-11-22 (links inline).

## 1. Foundation-model landscape (the heavyweights)

| Model | Who | Scale | Notable |
|---|---|---|---|
| UNI / UNI2 | Harvard Mahmood Lab | 100M tiles / 100k WSIs (ViT-L) | Nat Med 2024; UNI2 '25 SOTA SSL |
| CONCH | Harvard Mahmood Lab | 1.17M image-text pairs | Vision-language, Nat Med 2024 |
| PathChat / PathChat DX | Harvard Mahmood Lab | multimodal chatbot | **FDA Breakthrough Device** designation |
| CHIEF | Harvard Yu Lab (hms-dbmi/CHIEF) | 60,530 WSIs / 44TB / 19 sites | Nature 2024; weakly supervised; tile+slide dual pretraining; cancer-cell detection, origin, molecular profile, prognosis |
| Virchow2 / Virchow2G | Paige (licenced) + partners | 632M / 1.85B params; 3.1M WSIs, 2.4PB | Largest pathology model; benchmark study: **Virchow2 significantly best in colon tissue** |
| Prov-GigaPath | Microsoft + Providence | 1.3B tiles / 171k WSIs | Nature Jun 2024; LongNet slide encoder; open weights |
| Phikon / Phikon-v2 | Owkin | 456M patches, 30 sites | Our existing benchmark victim |
| H-optimus-0, RudolfV, Hibou, TITAN, EXAONEPath, BEPH, Google Path Foundation | various | — | RudolfV: pathologist-curated corpus, 129 stains; BEPH (China) MIM on TCGA |

Independent benchmark (PMC 2025, 22 non-TCGA tasks): slide-level top group
Virchow2 0.836 / UNI 0.835 / H-optimus-0 0.832 / GigaPath 0.827 / UNI2 0.824;
patch-level Virchow2 0.962 mean AUROC. Colon specifically: Virchow2 significantly
ahead. **We cannot and should not try to out-scale these. Our phikon-v2 beat
(97.52 vs 93.12 at 1/50th the params) is already the right kind of claim.**

Field's own confession (20-model review, 2025-26): clinical adoption lives in
LOW-stakes tasks (triage, QC, artifact detection); biomarker/treatment-response
is "research stage"; "the highest performance lives where the decision matters
least." A 2026 UCL-linked review counts only **4 FDA-approved WSI cancer
solutions**. Nobody has a box to check for "which clinical decision does this
change" — our quality-gate/rejection-aware axis is differentiated exactly there.

## 2. Multi-scale modes (user proposal: whole-tile mode / zoom mode / whole-slide mode)

ESTABLISHED ART — the user's instinct matches the canonical architectures:
- **HIPT** (Harvard, CVPR 2022 oral; mahmoodlab/HIPT): nested hierarchy 16px
  cell tokens → 256px patches → 4096px regions → slide; self-supervised AT
  EACH LEVEL; bottom-up transformer aggregation. This IS "modes": each level is
  a mode, and the context is preserved by construction.
- **SA-HCNN** (Sci Rep 2024, s41598-024-76807-x): spatial-aware input = main
  tile + LARGER context tiles resized to the same input — literally "keep the
  context that it was subdivided and zoomed." Three-scale input, one model.
- **CHIEF**: two complementary pretraining methods — tile-level SSL + slide-level
  weak supervision — trained jointly.
- TITAN: region-level SSL + caption alignment.

DEBATE POSITIONS:
- FOR: proven pattern (HIPT/SA-HCNN/CHIEF); our AttentionMIL head already
  exists in-repo as the slide-mode aggregator; stain-aug pipeline reusable;
  warm-start discipline proven in PR-16.
- AGAINST (my objections to survive): (a) 3060 12GB — HIPT-style 3-level
  pretraining is out of budget; a CONSERVATIVE variant (one trunk, multi-scale
  INPUTS + context tiles SA-HCNN-style, not a new architecture) is what fits;
  (b) labels: we have tile labels (9-class) but only slide-level identity for
  Camelyon — zoom-mode subtiles inherit the parent label = WEAK labels,
  must be measured, not assumed; (c) mode interference: one head answering at
  multiple scales can blur tile-level SOTA (97.81%) — the 9/9 gate must be
  the no-regression tripwire; (d) the "no trade-offs" user gate from PR-21.

## 3. Cell-level cancer-cell detection (the user's edge)

ESTABLISHED ART — mature and OPEN:
- **HoVer-Net** (Medical Image Analysis; vqdang/hover_net; TIA Lab Warwick):
  simultaneous nucleus instance segmentation + classification; HV distance
  maps separate clustered nuclei; pretrained weights for CoNSeP, PanNuke,
  MoNuSAC, Kumar, CPM. TIAToolbox ships `hovernet_fast-pannuke` ready-to-run.
- **PanNuke**: 19 tissue types incl. colon; nucleus classes = neoplastic,
  non-neoplastic epithelial, inflammatory, connective, dead — "neoplastic vs
  non-neoplastic epithelial" IS "which cells are cancer cells."
- **CellViT / CellViT++** (2024-25): ViT/foundation-model encoders for nucleus
  seg+class; strong zero-shot; energy-efficient variants.
- **Stanford nuclei.io** (Nature Biomed Eng, Jun 2024): pathologist-customizable
  interactive cell-level AI — human-in-the-loop, learns from clicks. Relevant
  to our dual-audience law.
- **Stanford NucEval**: robust evaluation framework for nuclear segmentation.
- **CHIEF** claims "cancer cell detection" — but at tile level (tumor tissue),
  not per-nucleus instances.

DEBATE POSITIONS:
- FOR: zero-training path exists — run pretrained HoVer-Net/CellViT
  (PanNuke weights) as a SECOND stage on tiles our gate already answered;
  "neoplastic nuclei per tile" is a new, defensible feature column; PanNuke
  is CC-licensed for research; 3060-compatible (CellViT++ paper is literally
  about efficiency).
- AGAINST (my objections): (a) it is a DIFFERENT MODEL, not our model
  improved — honesty on the site requires saying so ("two-stage: HistoGate
  gates the tile, HoVer-Net counts the cells"), never implying one system;
  (b) PanNuke's "neoplastic" is organ-agnostic — on colorectal it can
  confuse neoplastic epithelial with inflamed epithelium; needs a measured
  colon-subset validation before any claim; (c) RUO + dataset licenses
  (CoNSeP/PanNuke research use) fit our RUO posture — OK; (d) cell-level
  output multiplies the "at least say something" surface — per-cell verdicts
  without a per-cell gate is exactly the confident-fabrication risk our
  whole design refuses; the tile gate must bound what cells we even show.
- SYNTHESIS CANDIDATE: cells only WITHIN gate-passed tiles; counts, not
  diagnoses; declined tiles stay cell-silent.

## 4. Who is doing what (per the user's list)

- **Siemens Healthineers**: integration/distribution, not model research —
  AI-Rad Companion family; mammography AI (>1M multi-vendor studies); Sept
  2025 partnership with **Aiforia to distribute clinical pathology AI across
  Europe incl. QuantCRC (colorectal quantification — the commercial analog of
  our TSR/TIL)**; USCAP 2025: radiology+pathology one-view tumor boards.
  Lesson: they win by distribution + regulatory clearance.
- **Stanford**: Shen Lab (**STARC-9 — large-scale 9-class CRC tissue dataset,
  NeurIPS 2026 — DIRECTLY our niche, must be tracked**); nuclei.io; PLIP
  (OpenPath 208k image-text, pathologist Twitter); TopoSlide (topology-aware
  self-supervised WSI); NucEval.
- **Harvard**: Mahmood Lab (UNI/UNI2, CONCH, PathChat→FDA BD, HIPT, CLAM);
  Yu Lab (CHIEF, uncertainty-aware FM ensembles Nat Commun 2025, fairness
  Cell Reports Med 2025).
- **UCL**: methodology rigor — Nature Protocols end-to-end WSI→biomarker
  pipeline; explainable-AI-for-pathology; the only-4-FDA-approved regulatory
  mapping; SHG fibrosis AI.
- **IITs**: application-tier on public datasets — IIT Madras (Khened/Kori/
  Krishnamurthi lineage; DL-WSI review 2025), IIT Indore efficiency theses,
  hybrid transfer-learned HCC grading. No foundation models. HistoGate is
  already above this tier; do not benchmark against it.
- **China/Shanghai**: BEPH (Nat Commun 2025, MIM FM on TCGA), knowledge-
  enhanced multimodal WSI FMs, vision-language precision-oncology FM (Nature
  Feb 2025 = MUSK lineage), massive domestic hospital datasets. Shanghai-
  specific lab names did not surface cleanly in this sweep — flag as gap,
  do not fabricate.
- **Japan**: weakest English-web visibility in this sweep — clinical
  validation studies and scanner ecosystem (Olympus) rather than open model
  releases; honest gap, same rule.

## 5. Where this leaves the two proposals

1. Multi-scale modes: pattern is canonical (HIPT/SA-HCNN/CHIEF) — the debate
   is about the CONSERVATIVE variant we can afford and defend, not whether
   the idea is sound.
2. Cell-level: no new annotations needed for a first honest version
   (pretrained nucleus models + our gate as the bounding discipline) — the
   debate is about honesty framing and the colon-specific validation.
3. Differentiation: the field's own review admits nobody measures "which
   clinical decision changes" — the gated/rejection-aware/small-and-auditable
   lane is ours; do not chase Virchow2's colon crown.
