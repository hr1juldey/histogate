/* Familiarity = 100 − needle position on the shared energy display scale.
   energy −4.6 → 100 (identical to training patches), −1.6 → 0 (nothing like them). */
export const ELO = -4.6;
export const EHI = -1.6;
export const famPct = (e: number): number =>
  Math.max(0, Math.min(100, 100 * (1 - (e - ELO) / (EHI - ELO))));
export type FamBand = 'familiar' | 'unfamiliar';
export const FAM_BAND = (fam: number): FamBand => (fam >= 50 ? 'familiar' : 'unfamiliar');
