import { useEffect, useRef, useState } from 'react';
import { analyze, apiBase } from '../lib/api';

/* Same physics as training (bio_optics.py): Ruifrok H&E deconvolution,
   concentration scaling, residual preserved. One tile only. */
const SH = [0.644, 0.717, 0.267];
const SE = [0.092, 0.954, 0.283];
const norm = (v: number[]) => { const n = Math.hypot(...v); return v.map(x => x / n); };
const H = norm(SH), E = norm(SE);
const PINV = (() => {
  const m11 = H.reduce((s, x, i) => s + x * x, 0), m12 = H.reduce((s, x, i) => s + x * E[i], 0),
    m22 = E.reduce((s, x) => s + x * x, 0);
  const det = m11 * m22 - m12 * m12;
  const gi = [[m22 / det, -m12 / det], [-m12 / det, m11 / det]];
  const P = [[0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 2; r++) for (let k = 0; k < 3; k++)
    P[r][k] = gi[r][0] * [H[k], E[k]][0] + gi[r][1] * [H[k], E[k]][1];
  return P;
})();

interface Conv { cH: Float32Array; cE: Float32Array; res: Float32Array }
function deconvolve(src: Uint8ClampedArray): Conv {
  const N = src.length / 4;
  const cH = new Float32Array(N), cE = new Float32Array(N), res = new Float32Array(src.length);
  for (let i = 0, p = 0; p < N; p++, i += 4) {
    const od = [src[i] > 1 ? -Math.log(src[i] / 255) : 0,
      src[i + 1] > 1 ? -Math.log(src[i + 1] / 255) : 0,
      src[i + 2] > 1 ? -Math.log(src[i + 2] / 255) : 0];
    const h = Math.max(0, PINV[0][0] * od[0] + PINV[0][1] * od[1] + PINV[0][2] * od[2]);
    const e = Math.max(0, PINV[1][0] * od[0] + PINV[1][1] * od[1] + PINV[1][2] * od[2]);
    cH[p] = h; cE[p] = e;
    res[i] = od[0] - (H[0] * h + E[0] * e);
    res[i + 1] = od[1] - (H[1] * h + E[1] * e);
    res[i + 2] = od[2] - (H[2] * h + E[2] * e);
  }
  return { cH, cE, res };
}

async function loadPixels(url: string): Promise<Uint8ClampedArray> {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = c.height = 224;
  c.getContext('2d')!.drawImage(img, 0, 0, 224, 224);
  return c.getContext('2d')!.getImageData(0, 0, 224, 224).data;
}

type Card = { conf: number; label: string; energy: string };
const EMPTY: Card = { conf: 0, label: '…', energy: '—' };

export default function StainDial() {
  const cv = useRef<HTMLCanvasElement>(null);
  const ov = useRef<HTMLCanvasElement>(null);
  const BASE = useRef<Uint8ClampedArray | null>(null);
  const CONV = useRef<Conv | null>(null);
  const RAW = useRef<Uint8ClampedArray | null>(null);
  const [stain, setStain] = useState(0);
  const [rawMode, setRawMode] = useState(false);
  const rawRef = useRef(false);
  const [v1, setV1] = useState<Card>(EMPTY);
  const [ship, setShip] = useState<Card>(EMPTY);
  const [cap, setCap] = useState('loading…');
  const busy = useRef(false);
  const pend = useRef(false);

  useEffect(() => {
    (async () => {
      BASE.current = await loadPixels(`${apiBase}/samples/tumor.png`);
      CONV.current = deconvolve(BASE.current);
      RAW.current = await loadPixels(`${apiBase}/samples/nonorm.png`);
      render(0);
      predict();
    })();
  }, []);

  function render(t: number) {
    const c = cv.current;
    if (!c || !BASE.current || !CONV.current) return;
    const ctx = c.getContext('2d')!;
    const src = rawRef.current ? RAW.current! : BASE.current;
    const tt = rawRef.current ? 0 : t;
    const aH = 1 + 1.2 * tt, aE = Math.max(0.2, 1 - 0.7 * tt);
    const out = ctx.createImageData(224, 224);
    const { cH, cE, res } = CONV.current;
    for (let p = 0, i = 0; p < 224 * 224; p++, i += 4) {
      let o0: number, o1: number, o2: number;
      if (rawRef.current) {
        o0 = src[i] > 1 ? -Math.log(src[i] / 255) : 0;
        o1 = src[i + 1] > 1 ? -Math.log(src[i + 1] / 255) : 0;
        o2 = src[i + 2] > 1 ? -Math.log(src[i + 2] / 255) : 0;
      } else {
        const h = cH[p] * aH, e = cE[p] * aE;
        o0 = H[0] * h + E[0] * e + res[i];
        o1 = H[1] * h + E[1] * e + res[i + 1];
        o2 = H[2] * h + E[2] * e + res[i + 2];
      }
      out.data[i] = 255 * Math.exp(-Math.max(0, o0));
      out.data[i + 1] = 255 * Math.exp(-Math.max(0, o1));
      out.data[i + 2] = 255 * Math.exp(-Math.max(0, o2));
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    setCap(rawRef.current
      ? 'a REAL picture from a different machine — unmodified'
      : t < 0.05 ? 'as the lab scanned it'
      : `purple dye ×${aH.toFixed(2)}, pink dye ×${aE.toFixed(2)}`);
  }

  async function predict() {
    if (busy.current) { pend.current = true; return; }
    busy.current = true;
    const c = cv.current;
    if (c) {
      const blob = await new Promise<Blob>(r => c.toBlob(b => r(b!), 'image/png'));
      await Promise.all([go(blob, 'v1', setV1), go(blob, 'ship', setShip)]);
    }
    busy.current = false;
    if (pend.current) { pend.current = false; predict(); }
  }

  async function go(blob: Blob, which: 'v1' | 'ship', set: (c: Card) => void) {
    try {
      const j = await analyze(blob, which);
      const wrong = j.top_class !== 'TUM' ? ` ⚠ calls it ${j.top_class}` : '';
      const rej = j.ood ? ' ⚠ OOD-rejected' : '';
      set({ conf: j.confidence, label: `${j.top_class}${wrong}${rej}`, energy: String(j.energy) });
      if (which === 'ship') drawEvidence(j.gradcam as unknown as number[][]);
    } catch { set({ conf: 0, label: 'error', energy: '—' }); }
  }

  /* same evidence overlay as home/demo: normalized, solid outline at the hot region */
  function drawEvidence(cam2d: number[][]) {
    const ctx = ov.current?.getContext('2d');
    if (!ctx) return;
    const flat = cam2d.flat?.() ?? (cam2d as unknown as number[]);
    const w = Math.round(Math.sqrt(flat.length));
    const max = Math.max(...flat) || 1;
    const hi = flat.map(v => v >= 0.55 * max);
    const at = (cx: number, cy: number) =>
      cx < 0 || cy < 0 || cx >= w || cy >= w ? false : hi[cy * w + cx];
    const d = ctx.createImageData(448, 448);
    for (let y = 0; y < 448; y++) for (let x = 0; x < 448; x++) {
      const cx = Math.floor((x / 448) * w), cy = Math.floor((y / 448) * w);
      const v = flat[cy * w + cx] / max;
      const i = (y * 448 + x) * 4;
      const isHi = at(cx, cy);
      const isEdge = isHi && (!at(cx - 1, cy) || !at(cx + 1, cy) || !at(cx, cy - 1) || !at(cx, cy + 1));
      d.data[i] = 255; d.data[i + 1] = 40; d.data[i + 2] = 40;
      d.data[i + 3] = isEdge ? 255
        : isHi ? 255 * Math.min(1, 0.35 + v) * 0.6
        : 255 * v * 0.2;
    }
    ctx.putImageData(d, 0, 0);
  }

  const onDial = (v: number) => {
    setStain(v);
    render(v);
    if (!busy.current) { pend.current = true; setTimeout(() => { if (pend.current) { pend.current = false; predict(); } }, 220); }
  };

  const toggleRaw = () => {
    rawRef.current = !rawRef.current;
    setRawMode(rawRef.current);
    render(rawRef.current ? 0 : stain);
    predict();
  };

  const scoreChip = (title: string, c: Card) => {
    const refused = c.label.includes('OOD-rejected');
    const wrongCall = c.label.includes('calls it');
    const call = c.label.split(' ⚠')[0];
    return (
      <div className={`chip ${refused ? 'refuse' : wrongCall ? 'miss' : 'match'}`}>
        <span className="who">{title}</span>
        <span className="call mono">{call}</span>
        <span className="badge">{refused ? '— declined (safe)' : wrongCall ? '✗ WRONG' : '✓ MATCH'}</span>
      </div>
    );
  };

  const card = (title: string, c: Card, cls: string) => {
    const refused = c.label.includes('OOD-rejected');
    const wrongCall = c.label.includes('calls it');
    const status = refused ? 'declined to answer — the safe behavior'
      : wrongCall ? `WRONG — calls it ${c.label.split(' ⚠')[0]}, and its gate sees no problem`
      : 'sees the tumor';
    return (
      <div className={`mcard ${cls} ${refused ? 'ref' : wrongCall ? 'wrong' : 'right'}`}>
        <div className="t">{title}</div>
        <div className="row"><span className="pct mono">{(c.conf * 100).toFixed(1)}%</span><span className="lab">{c.label.split(' ⚠')[0]}</span></div>
        <div className="bar"><i style={{ width: `${c.conf * 100}%` }} /></div>
        <div className="verdict-line">{status}</div>
        <div className="live mono">energy {c.energy}</div>
      </div>
    );
  };

  return (
    <div className="dial-sec">
      <div className="viewer">
        <p className="truth mono">ground truth: this patch is TUMOR (expert-labeled)</p>
        <div className="tilewrap">
          <canvas ref={cv} width="224" height="224" />
          <canvas ref={ov} width="448" height="448" aria-hidden="true" />
        </div>
        <p className="cap-hint">red outline = where the shipped model is looking for the tumor</p>
        <p className="cap">{cap}</p>
        <div className="dialbox">
          <div className="ends"><span>◀ normal staining</span><span>heavy purple stain ▶</span></div>
          <input type="range" min={0} max={100} value={stain * 100} disabled={rawMode}
            onChange={e => onDial(+(e.target as HTMLInputElement).value / 100)}
            onPointerUp={predict} aria-label="stain dial" />
          <button className="btn ghost rawbtn" onClick={toggleRaw}>
            {rawMode ? '← back to the first tile' : 'see a REAL picture from a different machine →'}
          </button>
        </div>
      </div>
      <div className="models">
        <div className="score">
          <div className="chip expert"><span className="who">expert label</span><span className="call mono">TUMOR</span><span className="badge">truth</span></div>
          {scoreChip('first model — before the fix', v1)}
          {scoreChip('shipped model — stain physics', ship)}
        </div>
        {card('first model — trained before the fix', v1, 'bad')}
        {card('shipped model — retrained on stain physics', ship, 'good')}
        <p className="note">both models run live right now — numbers update as you drag</p>
      </div>
    </div>
  );
}
