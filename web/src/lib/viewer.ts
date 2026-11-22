/* Shared viewer grammar — one source of truth for /slide and /demo.
   The window model: fixed aspect-honest frame, rigid rotation, every pixel
   reachable by pan+zoom. Laws from the /slide build (2026-11-22). */

export const CLASS_COLORS: Record<string, string> = {
  TUM: '#e46fa8', NORM: '#5dbb7a', MUS: '#9a7ce0', STR: '#6f6ee0',
  LYM: '#4a9ebf', MUC: '#58b6a8', DEB: '#8a8f98', ADI: '#c9a227',
  BACK: '#3a3d42',
};

export const KMIN = 1;
export const KMAX = 8;

export interface View { x: number; y: number; k: number }
export const HOME: View = { x: 0, y: 0, k: 1 };

export interface PatchDetail {
  probabilities: { class: string; p: number }[];
  top_class: string; confidence: number; energy: number;
  energy_threshold: number; ood: boolean; gradcam: number[][];
}

/* Pan bounds from the ROTATED content's true extent (AABB): when rotated, the
   content spans the other way — every pixel must stay reachable. An axis
   smaller than the window locks centered (window model). */
export const clampView = (v: View, r: { width: number; height: number }, rot: number): View => {
  const [Wc, Hc] = rot % 180 ? [r.height, r.width] : [r.width, r.height];
  const cw = v.k * Wc, ch = v.k * Hc;
  const x = cw <= r.width
    ? (r.width - cw) / 2
    : Math.max(r.width - (v.k * (r.width + Wc)) / 2,
        Math.min((v.k * (Wc - r.width)) / 2, v.x));
  const y = ch <= r.height
    ? (r.height - ch) / 2
    : Math.max(r.height - (v.k * (r.height + Hc)) / 2,
        Math.min((v.k * (Hc - r.height)) / 2, v.y));
  return { ...v, x, y };
};

/* jet-ish colormap for the evidence heat */
export const jet = (v: number): [number, number, number] => [
  Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)))),
  Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)))),
  Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)))),
];

export const paintCam = (canvas: HTMLCanvasElement, g: number[][]): void => {
  if (!Array.isArray(g) || g.length === 0) return;
  const n = g.length;
  canvas.width = n; canvas.height = n;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const im = ctx.createImageData(n, n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const v = g[y]?.[x] ?? 0;
      const [r, gg, b] = jet(v);
      const i = (y * n + x) * 4;
      im.data[i] = r; im.data[i + 1] = gg; im.data[i + 2] = b;
      im.data[i + 3] = Math.round(160 * v);
    }
  }
  ctx.putImageData(im, 0, 0);
};
