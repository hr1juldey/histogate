# HistoGate — Design System v2 ("Lab Notebook")

Aesthetic: **Anthropic-grade warm editorial** — a pathologist's desk, not a neon AI
dashboard. Parchment paper, serif authority, tissue images treated as specimen plates.
Rationale: every AI startup ships near-black + neon; a warm, book-like interface says
"considered science," matches the biologist identity, and makes pink/purple H&E
imagery glow. Commercial polish, scientific soul.

## 1. Design tokens

Surfaces (warm only — no cool blue-grays anywhere):
- page          #f5f4ed  parchment (primary canvas)
- card          #faf9f5  ivory (elevated)
- interactive   #e8e6dc  warm sand (buttons, hover fills)
- dark-section  #141413  near-black warm (gate, story moments)
- dark-card     #30302e  warm charcoal

Text:
- primary       #141413  near-black warm
- secondary     #5e5d59  olive gray (body)
- tertiary      #87867f  stone gray (metadata, captions)
- on-dark       #faf9f5  ivory   / on-dark-muted #b0aea5 warm silver

Brand + semantic (muted, earthy):
- brand         #c96442  terracotta — PRIMARY CTA ONLY, never decorative
- brand-hover   #d97757  coral
- pass          #3f7d4e  muted warm green (gate ticks)
- fail          #b53333  warm crimson
- hema (data)   #7c5cd6  reserved exclusively for data/stain visuals, never chrome

Borders & depth (ring shadows, not drop shadows):
- border        1px solid #e8e6dc (light) / #30302e (dark)
- ring hover    box-shadow 0 0 0 1px #d1cfc5
- whisper       rgba(0,0,0,0.05) 0 4px 24px (elevated media only)

## 2. Typography
- Headlines: **Source Serif 4** (Georgia fallback), weight 500 ONLY, line-height
  1.10–1.30. Display 56–64px, section 40px, card title 25px, feature 20px.
- Body/UI: Inter. Body 16–17px, line-height 1.60 (book rhythm, never below 1.40).
  Labels 12px +0.12px tracking; overline 10px uppercase +0.5px.
- Code/data: JetBrains Mono 13–15px, tabular numerals for every measured value.
- Serif = claims and narrative. Sans = function. Mono = measurement. Never mix roles.

## 3. Radius & spacing
- radius: buttons/inputs 12px · cards 16px · media 24px · hero containers 32px
- spacing: 8px base; sections 96px vertical; card padding 24–32px
- container 1120px max, centered; generous editorial margins

## 4. Buttons (4, exactly)
1. Primary: terracotta bg, ivory text, 12px radius, ring shadow; the ONE chromatic
   action per view. Hover: coral.
2. Secondary: warm-sand bg, charcoal text, ring #d1cfc5. Hover: ring deepens.
3. Dark: #30302e bg, ivory text — used inside dark sections.
4. Quiet: text-only + underline-on-hover — inline links, tertiary actions.
All ≥40px touch target. One primary per screen. No disabled states in demo.

## 5. Imagery law
Tissue patches are the heroes: specimen plates on ivory cards with warm borders,
rounded 24px, captioned in mono (TUM/NORM). Class labels NEVER in brand color.
Screenshots of the product in use sit on parchment with whisper shadow.
No stock illustration; no icons where a real image exists. SVG icons (1.4px stroke,
round caps) only for the 3-step flow.

## 6. Copy voice
Plain, warm, precise. Biology first ("cancer epithelium" before "class 8").
Numbers always carry provenance words: measured · external · held-out.
No hype adjectives, no "revolutionary," no exclamation marks.
Honesty lines are features, not disclaimers-in-the-corner: "Eight pass. One fails.
Both are shown."

## 7. Page map & user flow
Pages: `/` Overview · `/story` Story (interactive) · `/ui` Live demo · GitHub (external).
Persistent nav on every page: Overview · Story · Live demo · GitHub + one primary CTA.
No page is a dead end; every page ends with exactly one next action.

FORWARD (first visit):  `/` claim+proof (cancer vs healthy duo, live metrics)
→ CTA "See it classify" → `/ui` (auto-analyzes tumor example; zero user action needed)
→ gallery exploration → curiosity → `/story` (drag the stain dial; the break, live)
→ gate scene (dark section, honest 8/9) → GitHub for the receipts.

BACKWARD (deep-linked, e.g. recruiter opens /ui directly): nav Overview explains what
they just saw in one scroll; every demo verdict links "how this was measured" →
`/story`; story ends → "Try it on your own patch" → `/ui`. Loop closes.
Shallow exit: footer on every page = RUO + GitHub + measured-artifacts note.

## 8. Interaction states
- Hover: ring shadow + text darkens. Never color shifts or scale-jumps.
- Focus: #3898ec focus ring (only cool color, accessibility only).
- Active/pressed: inset ring 0 0 0 1px @15%.
- Motion: opacity/transform only, 150–200ms ease-out; page never moves on hover.
- Loading: shimmer on ivory, never spinners.

## 9. Accessibility
Contrast AA minimum (parchment #141413 = 13.9:1). Alt text on every tissue image
(class + plain description). Slider ARIA-labeled. No color-only meaning (gate uses
tick + cross icons with text).
