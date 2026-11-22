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
  conformal_set?: string[];   /* P0a: 90% APS set, present when answered */
}

/* Pan bounds (user law 2026-11-22: "I cannot drag the image from bottom to
   the center and then zoom in" — the no-void rule forbade exactly that).
   Rule now: the WINDOW CENTER must always stay over content, with a margin
   inside the content edges. Dead space is allowed (Google-Maps style) so
   any part of the image — e.g. a tissue strip at the image's bottom edge —
   can be dragged to the center at ANY zoom, then zoomed in place. The image
   can never leave the screen: its center coverage is enforced. */
export const clampView = (v: View, r: { width: number; height: number }, rot: number): View => {
  const [Wc, Hc] = rot % 180 ? [r.height, r.width] : [r.width, r.height];
  const m = Math.min(r.width, r.height) * 0.1;
  const cw = v.k * Wc, ch = v.k * Hc;
  const x = Math.max(r.width / 2 - cw + m, Math.min(cw - r.width / 2 - m, v.x));
  const y = Math.max(r.height / 2 - ch + m, Math.min(ch - r.height / 2 - m, v.y));
  return { ...v, x, y };
};

/* Rotation must keep WHAT THE USER IS LOOKING AT fixed: the content point
   under the window center before rotating stays under it after. Rotation is
   otherwise about the content center, which swings the viewed point away —
   then clampView hard-recentered the lost axis = the "snap" defect. Invert
   the transform at the window center, rotate that point, re-derive the
   translation. Transform stack: w = t + k*(R(rot)*(q - C) + C). */
export const rotateKeepCenter = (v: View, r: { width: number; height: number },
  rot: number, nr: number): View => {
  const cx = r.width / 2, cy = r.height / 2;
  const ux = (cx - v.x) / v.k - cx, uy = (cy - v.y) / v.k - cy;
  const a = (-rot * Math.PI) / 180;
  const qx = ux * Math.cos(a) - uy * Math.sin(a);
  const qy = ux * Math.sin(a) + uy * Math.cos(a);
  const b = (nr * Math.PI) / 180;
  const fx = qx * Math.cos(b) - qy * Math.sin(b);
  const fy = qx * Math.sin(b) + qy * Math.cos(b);
  return { k: v.k, x: cx - v.k * (fx + cx), y: cy - v.k * (fy + cy) };
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
