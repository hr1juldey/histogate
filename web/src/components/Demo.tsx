import { useEffect, useRef, useState } from 'react';
import { analyze, analyzeTiff, apiBase, get, type Analysis, type Sample } from '../lib/api';

const TRUTH: Record<string, string> = {
  tumor: 'TUM', normal: 'NORM', stroma: 'STR', lymph: 'LYM', mucus: 'MUC', debris: 'DEB',
};

const GLOSS: Record<string, string> = {
  tumor: 'Malignant epithelium — the cancer cells themselves.',
  normal: 'Healthy mucosal glands: neat, orderly crypts.',
  stroma: 'Connective tissue — the soil a tumor invades. Carries the TSR biomarker.',
  lymph: 'Dense immune-cell clusters — the TIL biomarker.',
  mucus: 'Pale mucin lakes — visually unlike anything else.',
  debris: 'Necrosis and waste: ugly but real biology.',
};

/** Full demo island: gallery, live analysis, calibrated bars, Grad-CAM. */
export default function Demo() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [j, setJ] = useState<Analysis | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [alpha, setAlpha] = useState(0.45);
  const camRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { get<Sample[]>('/samples').then(s => setSamples(s.filter(x => x.id !== 'nonorm'))); }, []);

  const runBlob = async (blob: Blob, displayUrl: string, isTiff: boolean) => {
    setSrc(displayUrl); setJ(null); setErr(null);
    try { setJ(isTiff ? await analyzeTiff(blob, 'ship') : await analyze(blob, 'ship')); }
    catch { setErr('backend unreachable — is the Space awake? refresh in a moment'); }
  };

  const pick = async (id: string) => {
    setSel(id);
    const r = await fetch(`${apiBase}/samples/${id}`);
    await runBlob(await r.blob(), `${apiBase}/samples/${id}.png`, true);
  };

  useEffect(() => { if (samples.length && sel === null) pick(samples[0].id); }, [samples, sel]);

  useEffect(() => {
    if (!j || !src) return;
    const cnv = camRef.current;
    const cam = cnv?.getContext('2d');
    if (!cam) return;
    const flat = (j.gradcam as unknown as number[][]).flat?.() ?? (j.gradcam as unknown as number[]);
    const w = Math.round(Math.sqrt(flat.length));
    const max = Math.max(...flat) || 1;
    const hi = flat.map(v => v >= 0.55 * max);
    const at = (cx: number, cy: number) =>
      cx < 0 || cy < 0 || cx >= w || cy >= w ? false : hi[cy * w + cx];
    const d = cam.createImageData(448, 448);
    for (let y = 0; y < 448; y++) for (let x = 0; x < 448; x++) {
      const cx = Math.floor((x / 448) * w), cy = Math.floor((y / 448) * w);
      const v = flat[cy * w + cx] / max;
      const i = (y * 448 + x) * 4;
      const isHi = at(cx, cy);
      const isEdge = isHi && (!at(cx - 1, cy) || !at(cx + 1, cy) || !at(cx, cy - 1) || !at(cx, cy + 1));
      d.data[i] = 255; d.data[i + 1] = 40; d.data[i + 2] = 40;
      d.data[i + 3] = (isEdge ? 255
        : isHi ? 255 * Math.min(1, 0.35 + v) * 0.65
        : 255 * v * 0.25) * alpha;
    }
    cam.putImageData(d, 0, 0);
  }, [j, src, alpha]);

  const sorted = j ? [...j.probabilities].sort((a, b) => b.p - a.p) : [];

  return (
    <div className="demo">
      <div className="left">
        <div className="gallery">
          {samples.map(s => (
            <button key={s.id} className={`tile ${sel === s.id ? 'on' : ''}`} onClick={() => pick(s.id)}>
              <img src={`${apiBase}/samples/${s.id}.png`} alt={s.why} />
              <span>{s.id.toUpperCase()}</span>
            </button>
          ))}
        </div>
        <p className="gloss">{sel ? GLOSS[sel] : ''}</p>
        <div className="own">
          <button className="btn ghost" onClick={() => fileRef.current?.click()}>…or upload your own H&E patch</button>
          <input ref={fileRef} type="file" accept="image/*" hidden
            onChange={e => { const f = e.target.files?.[0]; if (!f) return;
              setSel(null); runBlob(f, URL.createObjectURL(f), false); }} />
        </div>
      </div>
      <div className="right card">
        {sel && j && (
          <div className={`score ${j.ood ? 'refuse' : j.top_class === TRUTH[sel] ? 'match' : 'miss'}`}>
            <span className="sc"><em>expert label</em><b className="mono">{TRUTH[sel]}</b></span>
            <span className="sc-arrow">vs</span>
            <span className="sc"><em>AI call</em><b className="mono">{j.top_class}</b></span>
            <span className="badge">
              {j.ood ? '— declined (safe)' : j.top_class === TRUTH[sel] ? '✓ MATCH' : '✗ WRONG'}
            </span>
          </div>
        )}
        <div className={`verdict ${j?.ood ? 'ood' : ''}`}>
          <div className="vt">{j ? j.verdict : err ?? 'analyzing…'}</div>
          {j && <div className="vd mono">really {j.confidence >= 0.9 ? 'is' : 'is not'} bluffing: {(j.confidence * 100).toFixed(1)}% · refuses when the picture is too strange (energy {j.energy})</div>}
        </div>
        {src && (
          <div className="stage">
            <div className="viewer"><img src={src} alt="patch" /><canvas ref={camRef} width={448} height={448} /></div>
            <div className="stats">
              <div className="srow"><span>how sure it really is</span><b>{j ? `${(j.confidence * 100).toFixed(1)}%` : '—'}</b></div>
              <div className="srow"><span>"is this picture strange?" gauge</span><b className="mono">{j?.energy ?? '—'}</b></div>
              <div className="srow"><span>decision</span><b>{j ? (j.ood ? 'refused — too strange to judge' : 'answered') : '—'}</b></div>
              <label className="heat">where the model looked — its evidence, not a pathologist's marking
                <input type="range" min="0" max="100" value={alpha * 100} onInput={e => setAlpha(+(e.target as HTMLInputElement).value / 100)} />
              </label>
            </div>
          </div>
        )}
        <div className="bars">
          {sorted.map(p => (
            <div className={`brow ${j && p.class === j.top_class ? 'top' : ''}`} key={p.class}>
              <span className="bn">{p.class}</span>
              <div className="track"><div className="fill" style={{ width: `${(p.p * 100).toFixed(1)}%` }} /></div>
              <span className="bp mono">{(p.p * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
