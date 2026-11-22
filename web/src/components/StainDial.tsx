import { useEffect, useRef, useState } from 'react';
import storyCache from '../data/story-analyses.json';
import { analyze, apiBase, type Analysis } from '../lib/api';
import { ELO, EHI, famPct } from '../lib/familiarity';

/* Same physics as training (bio_optics.py): Ruifrok H&E deconvolution,
   concentration scaling, residual preserved. Each tile carries its OWN
   deconvolution — concentrations are never mixed across pictures. */
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

type TileId = 'lab' | 'raw';
interface Card { conf: number; label: string; energy: number; thr: number; ood: boolean }
const EMPTY: Card = { conf: 0, label: '…', energy: -4.6, thr: -3.7, ood: false };

/* needle % on the shared display scale (0 = −4.6 familiar end) */
const pct = (e: number) => 100 - famPct(e);

export default function StainDial() {
  const cv = useRef<HTMLCanvasElement>(null);
  const ov = useRef<HTMLCanvasElement>(null);
  const thumb = useRef<HTMLCanvasElement>(null);
  const PIX = useRef<Record<TileId, Uint8ClampedArray | null>>({ lab: null, raw: null });
  const CONVS = useRef<Record<TileId, Conv | null>>({ lab: null, raw: null });
  const tileRef = useRef<TileId>('lab');
  const [stain, setStain] = useState(0);
  const [tile, setTile] = useState<TileId>('lab');
  const [v1, setV1] = useState<Card>(EMPTY);
  const [ship, setShip] = useState<Card>(EMPTY);
  const [cap, setCap] = useState('loading…');
  const [analyzing, setAnalyzing] = useState(false);
  const busy = useRef(false);
  const pend = useRef(false);

  useEffect(() => {
    (async () => {
      PIX.current.lab = await loadPixels(`/samples/tumor.png`);
      CONVS.current.lab = deconvolve(PIX.current.lab);
      PIX.current.raw = await loadPixels(`/samples/nonorm.png`);
      CONVS.current.raw = deconvolve(PIX.current.raw);
      render(0);
      predictBase();
    })();
  }, []);

  /* both tiles go through the same physics: their OWN concentrations × dial */
  function render(t: number) {
    const c = cv.current;
    const id = tileRef.current;
    if (!c || !PIX.current[id] || !CONVS.current[id]) return;
    const ctx = c.getContext('2d')!;
    const aH = 1 + 1.2 * t, aE = Math.max(0.2, 1 - 0.7 * t);
    const out = ctx.createImageData(224, 224);
    const { cH, cE, res } = CONVS.current[id]!;
    for (let p = 0, i = 0; p < 224 * 224; p++, i += 4) {
      const h = cH[p] * aH, e = cE[p] * aE;
      const o0 = H[0] * h + E[0] * e + res[i];
      const o1 = H[1] * h + E[1] * e + res[i + 1];
      const o2 = H[2] * h + E[2] * e + res[i + 2];
      out.data[i] = 255 * Math.exp(-Math.max(0, o0));
      out.data[i + 1] = 255 * Math.exp(-Math.max(0, o1));
      out.data[i + 2] = 255 * Math.exp(-Math.max(0, o2));
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    setCap(t < 0.05
      ? id === 'lab' ? 'as this lab scanned it' : 'as this machine scanned it'
      : `purple dye ×${aH.toFixed(2)}, pink dye ×${aE.toFixed(2)}`);
  }

  async function predict() {
    if (busy.current) { pend.current = true; return; }
    busy.current = true;
    const c = cv.current;
    if (c) {
      /* pin the exact picture these answers will belong to */
      const t = thumb.current;
      if (t) {
        const tc = t.getContext('2d')!;
        tc.drawImage(c, 0, 0, 224, 224);
      }
      setAnalyzing(true);
      const blob = await new Promise<Blob>(r => c.toBlob(b => r(b!), 'image/png'));
      await Promise.all([go(blob, 'v1', setV1), go(blob, 'ship', setShip)]);
      setAnalyzing(false);
    }
    busy.current = false;
    if (pend.current) { pend.current = false; predict(); }
  }

  /* Untouched dial = KNOWN data = cache (works with the backend dead).
     Only a user-moved dial makes synthetic (unknown) images -> live. */
  async function predictBase() {
    const cache = storyCache as unknown as Record<string, Analysis>;
    const v1c = cache['tumor:v1'], shipc = cache['tumor:ship'];
    const toCard = (j: Analysis): Card =>
      ({ conf: j.confidence, label: j.top_class, energy: j.energy,
         thr: j.energy_threshold, ood: j.ood });
    setV1(toCard(v1c)); setShip(toCard(shipc));
    if (shipc?.gradcam) drawEvidence(shipc.gradcam as unknown as number[][]);
  }

  async function go(blob: Blob, which: 'v1' | 'ship', set: (c: Card) => void) {
    try {
      const j = await analyze(blob, which);
      set({ conf: j.confidence, label: j.top_class, energy: j.energy, thr: j.energy_threshold, ood: j.ood });
      if (which === 'ship') drawEvidence(j.gradcam as unknown as number[][]);
    } catch { set({ ...EMPTY, label: 'error' }); }
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

  /* tile switch keeps the dial position — same stain shift, other patient */
  const pick = (id: TileId) => {
    if (id === tileRef.current) return;
    tileRef.current = id;
    setTile(id);
    render(stain);
    predict();
  };

  const scoreChip = (title: string, c: Card) => {
    const refused = c.ood;
    const wrongCall = !c.ood && c.label !== 'TUM' && c.label !== '…';
    return (
      <div className={`chip ${refused ? 'refuse' : wrongCall ? 'miss' : 'match'}`}>
        <span className="who">{title}</span>
        <span className="call mono">{c.label}</span>
        <span className="badge">{refused ? '— declined (safe)' : wrongCall ? '✗ WRONG' : '✓ MATCH'}</span>
      </div>
    );
  };

  const card = (title: string, c: Card, cls: string) => {
    const refused = c.ood;
    const wrongCall = !c.ood && c.label !== 'TUM' && c.label !== '…';
    const state = refused ? 'ref' : wrongCall ? 'wrong' : 'right';
    const status = refused ? 'declined — safe'
      : wrongCall ? `WRONG — says ${c.label}`
      : 'sees the tumor';
    const why = refused ? 'familiarity below its gate — unlike anything it studied'
      : wrongCall ? 'confidently wrong — its gate saw no problem'
      : 'matches the expert label';
    const gcol = refused ? '#e3b341' : wrongCall ? 'var(--bad)' : '#5dbb7a';
    return (
      <div className={`mcard ${cls} ${state} ${analyzing ? 'dim' : ''}`}>
        <div className="t">{title}</div>
        <div className="row"><span className="pct mono">{(c.conf * 100).toFixed(1)}%</span><span className="lab">{c.label}</span></div>
        <div className="bl"><span>how sure it is</span><span className="mono">0 – 100%</span></div>
        <div className="bar"><i style={{ width: `${c.conf * 100}%` }} /></div>
        <div className="bl"><span>how familiar the picture looks</span></div>
        <div className="gauge">
          <span className="fband" style={{ width: `${pct(c.thr)}%` }} aria-hidden="true" />
          <i style={{ width: `${pct(c.energy)}%`, background: gcol }} />
          <b className="tick" style={{ left: `${pct(c.thr)}%` }}><em className="mono">gate {c.thr.toFixed(2)}</em></b>
        </div>
        <div className="glabels mono"><span>100 · familiar<br />answers</span><span>0 · unfamiliar<br />declines</span></div>
        <div className="verdict-line">{status}</div>
        <details className="why"><summary>why?</summary><p>{why}</p></details>
        <div className="live mono">familiarity {Math.round(famPct(c.energy))}/100 · energy {c.energy.toFixed(2)}</div>
      </div>
    );
  };

  return (
    <div className="dial-sec">
      <div className="viewer">
        <div className="tabs" role="tablist" aria-label="tissue source">
          <button role="tab" aria-selected={tile === 'lab'} className={tile === 'lab' ? 'on' : ''} onClick={() => pick('lab')}>lab tile</button>
          <button role="tab" aria-selected={tile === 'raw'} className={tile === 'raw' ? 'on raw' : ''} onClick={() => pick('raw')}>raw scanner</button>
        </div>
        <div className={`tilewrap ${tile === 'raw' ? 'rawsrc' : ''}`}>
          <canvas ref={cv} width="224" height="224" />
          <canvas ref={ov} width="448" height="448" aria-hidden="true" />
          <span className="stamp">{tile === 'lab' ? 'patient A · scanner A' : 'patient B · scanner B'}</span>
        </div>
        <p className="cap-hint">red = where the model looks</p>
        <p className="cap">{cap}</p>
        <div className="dialbox">
          <p className="truth mono">both patches are <b>TUMOR</b>. drag the dial.</p>
          <div className="ends"><span>◀ normal staining</span><span>heavy purple stain ▶</span></div>
          <input type="range" min={0} max={100} value={stain * 100}
            onChange={e => onDial(+(e.target as HTMLInputElement).value / 100)}
            onPointerUp={predict} aria-label="stain dial" />
          <div className="zones" aria-hidden="true"><span className="zr" /><span className="zb" /></div>
          <div className="zlabels"><span>real lab range</span><span>beyond real labs</span></div>
        </div>
      </div>
      <div className="models">
        <div className={`pend mono ${analyzing ? 'on' : ''}`}>
          <canvas ref={thumb} width="224" height="224" />
          <span>{analyzing ? 'analyzing this exact picture…' : 'answers below are for this picture'}</span>
        </div>
        <div className="score">
          <div className="chip expert"><span className="who">expert label</span><span className="call mono">TUMOR</span><span className="badge">truth</span></div>
          {scoreChip('first model', v1)}
          {scoreChip('shipped model', ship)}
        </div>
        {card('first model', v1, 'bad')}
        {card('shipped model', ship, 'good')}
        <p className="def">familiarity = how closely a patch resembles the 100,000 patches the model studied — technical name: energy score (Liu et al. 2020); the gate is tuned so ≤5% of real hospital patches are wrongly declined.</p>
        <p className="note">both models answer live as you drag</p>
      </div>
    </div>
  );
}
