import { useEffect, useRef, useState } from 'react';
import { apiBase, getSlideMap, type SlideMapData, type SlideTile } from '../lib/api';
import { famPct } from '../lib/familiarity';

/* Whole-slide view: the REAL slide as under-layer, verdict overlays on top,
   wheel-zoom / drag-pan with visible controls. Clicking a patch re-analyzes it
   at full resolution in the inspector, with the model's own evidence overlay
   (Grad-CAM — model attention, never a pathologist marking). */
const CLASS_COLORS: Record<string, string> = {
  TUM: '#e46fa8', NORM: '#5dbb7a', MUS: '#9a7ce0', STR: '#6f6ee0',
  LYM: '#4a9ebf', MUC: '#58b6a8', DEB: '#8a8f98', ADI: '#c9a227',
  BACK: '#3a3d42',
};
const KMIN = 1, KMAX = 8;

interface View { x: number; y: number; k: number }
const HOME: View = { x: 0, y: 0, k: 1 };

interface PatchDetail {
  probabilities: { class: string; p: number }[];
  top_class: string; confidence: number; energy: number;
  energy_threshold: number; ood: boolean; gradcam: number[][];
}

/* jet-ish colormap for the evidence heat */
const jet = (v: number): [number, number, number] => [
  Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)))),
  Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)))),
  Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)))),
];

export default function SlideMap() {
  const [data, setData] = useState<SlideMapData | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState<SlideTile | null>(null);
  type ViewMode = 'verdicts' | 'familiarity';
  const [mode, setMode] = useState<ViewMode>('verdicts');
  const [view, setView] = useState<View>(HOME);
  const [rot, setRot] = useState(0);
  const [detail, setDetail] = useState<PatchDetail | null>(null);
  const [detailErr, setDetailErr] = useState(false);
  const [showCam, setShowCam] = useState(true);
  const [box, setBox] = useState<{ w: number; h: number } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const camRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ px: number; py: number; vx: number; vy: number; moved: number; id: string | null } | null>(null);

  useEffect(() => {
    getSlideMap().then(setData).catch(e => setErr(String(e)));
  }, []);

  /* frame sized in PIXELS so its aspect ALWAYS equals the (rotated) content's
     aspect — rotation swaps the frame, it never stretches a pixel */
  useEffect(() => {
    if (!data) return;
    const compute = () => {
      const panes = document.querySelector('.panes');
      const availW = Math.max(280, (panes?.clientWidth ?? 1000) - 340 - 24 - 16);
      const ar = data.dims.w / data.dims.h;  /* the window NEVER changes shape */
      let h = Math.min(window.innerHeight * 0.82, 900);
      let w = h * ar;
      if (w > availW) { w = availW; h = w / ar; }
      setBox({ w: Math.round(w), h: Math.round(h) });
    };
    compute();
    window.addEventListener('resize', compute);
    return () => window.removeEventListener('resize', compute);
  }, [data]);

  /* wheel zoom anchored at the cursor; non-passive so we can preventDefault */
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = box.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      setView(v => {
        const k = Math.max(KMIN, Math.min(KMAX, v.k * Math.exp(-e.deltaY * 0.0016)));
        const wx = (px - v.x) / v.k, wy = (py - v.y) / v.k;
        let x = px - wx * k, y = py - wy * k;
        if (k <= KMIN + 1e-3) return HOME;
        x = Math.max(r.width * (1 - k), Math.min(0, x));
        y = Math.max(r.height * (1 - k), Math.min(0, y));
        return { x, y, k };
      });
    };
    box.addEventListener('wheel', onWheel, { passive: false });
    return () => box.removeEventListener('wheel', onWheel);
  }, [data]);

  /* clicking a patch re-analyzes that exact tile at full resolution */
  useEffect(() => {
    if (!sel) { setDetail(null); setDetailErr(false); return; }
    let live = true;
    setDetail(null); setDetailErr(false);
    (async () => {
      try {
        const r = await fetch(`${apiBase}/slide-tile/${sel.row}/${sel.col}.png`);
        if (!r.ok) throw new Error('tile fetch failed');
        const fd = new FormData();
        fd.append('file', await r.blob(), 'tile.png');
        fd.append('model', 'ship');
        const a = await fetch(`${apiBase}/analyze-patch`, { method: 'POST', body: fd });
        if (!a.ok) throw new Error('analyze failed');
        const j: PatchDetail = await a.json();
        if (live) setDetail(j);
      } catch { if (live) setDetailErr(true); }
    })();
    return () => { live = false; };
  }, [sel]);

  /* paint the evidence heat over the tile */
  useEffect(() => {
    const c = camRef.current;
    if (!c || !detail || !showCam) return;
    const g = detail.gradcam;
    if (!Array.isArray(g) || g.length === 0) return;
    const n = g.length;
    c.width = n; c.height = n;
    const ctx = c.getContext('2d');
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
  }, [detail, showCam]);

  /* pan surface: capture + drag, but NEVER steal presses on the control
     buttons or a plain tap on a patch (that's a select, not a drag) */
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const t = e.target as HTMLElement;
    if (t.closest('.vctrl')) return;                       /* controls keep their clicks */
    const cell = t.closest('.vcell');
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = {
      px: e.clientX, py: e.clientY, vx: view.x, vy: view.y, moved: 0,
      id: cell?.getAttribute('data-id') ?? null,
    };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.px, dy = e.clientY - d.py;
    d.moved = Math.max(d.moved, Math.hypot(dx, dy));
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setView(v => ({
      ...v,
      x: Math.max(r.width * (1 - v.k), Math.min(0, d.vx + dx)),
      y: Math.max(r.height * (1 - v.k), Math.min(0, d.vy + dy)),
    }));
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d && d.id && d.moved <= 5) {                       /* tap = select that patch */
      const t = data?.tiles.find(x => x.id === d.id);
      if (t) setSel(t);
    }
  };

  /* visible controls (GitHub-viewer style): the hint that this canvas moves */
  const clamp = (v: View, r: DOMRect): View => ({
    ...v,
    x: Math.max(r.width * (1 - v.k), Math.min(0, v.x)),
    y: Math.max(r.height * (1 - v.k), Math.min(0, v.y)),
  });
  const zoomBy = (f: number) => {
    const b = boxRef.current; if (!b) return;
    const r = b.getBoundingClientRect();
    setView(v => {
      const k = Math.max(KMIN, Math.min(KMAX, v.k * f));
      if (k <= KMIN + 1e-3) return HOME;
      const px = r.width / 2, py = r.height / 2;
      const wx = (px - v.x) / v.k, wy = (py - v.y) / v.k;
      return clamp({ x: px - wx * k, y: py - wy * k, k }, r);
    });
  };
  const panBy = (fx: number, fy: number) => {
    const b = boxRef.current; if (!b) return;
    const r = b.getBoundingClientRect();
    setView(v => clamp({ ...v, x: v.x - fx * r.width * 0.15, y: v.y - fy * r.height * 0.15 }, r));
  };
  const rotate90 = () => { setRot(r => (r + 90) % 360); setView(HOME); };
  const resetAll = () => { setRot(0); setView(HOME); };

  if (err) return <p className="map-err">the map is still warming up — refresh in a minute</p>;
  if (!data) return <p className="map-err mono">tiling the slide… (first load takes a few minutes)</p>;

  const answered = data.tiles.filter(t => !t.declined);
  const declinedN = data.tiles.length - answered.length;
  const guides = [...answered]
    .sort((a, b) => (b.cls === 'TUM' ? b.conf : -1) - (a.cls === 'TUM' ? a.conf : -1))
    .slice(0, 3);
  const showTiles = view.k >= 2;
  const declined = detail ? detail.ood : sel?.declined ?? false;
  const clsName = detail ? detail.top_class : sel?.cls ?? '';

  const cell = (t: SlideTile) => ({
    left: `${((t.col * data.stride) / data.dims.w) * 100}%`,
    top: `${((t.row * data.stride) / data.dims.h) * 100}%`,
    width: `${(data.patch / data.dims.w) * 100}%`,
    height: `${(data.patch / data.dims.h) * 100}%`,
  });

  return (
    <div className="slide-sec">
      <div className="sumband mono">
        <span><b>{data.tiles.length}</b> patches</span>
        <span><b className="ok">{answered.length}</b> answered</span>
        <span><b className="amb">{declinedN}</b> declined — foreign organ</span>
        <span className="vswitch" role="group" aria-label="view mode">
          {(['verdicts', 'familiarity'] as ViewMode[]).map(m => (
            <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}
              aria-pressed={mode === m}>{m}</button>
          ))}
        </span>
        <button className="vtg" onClick={resetAll}
          disabled={view.k <= KMIN + 1e-3 && rot === 0}>reset view</button>
      </div>
      <div className="guides">
        <span className="glab">most tumor-like — verify:</span>
        {guides.map((t, i) => (
          <button key={t.id} className="guide mono" onClick={() => setSel(t)}
            aria-label={`patch ${t.id}: ${t.cls}, ${Math.round(t.conf * 100)}% sure, familiarity ${t.familiarity}`}>
            {i + 1}. {t.cls} {Math.round(t.conf * 100)}% · fam {t.familiarity}
          </button>
        ))}
      </div>
      <div className="panes">
        <div className="pane mosaicwrap">
          <div ref={boxRef} className="fitbox zoombox"
            style={box ? { width: `${box.w}px`, height: `${box.h}px` } : undefined}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove}
            onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
            onDoubleClick={() => setView(HOME)}>
            <p className="ulcap">{view.k > KMIN + 1e-3
              ? 'scroll = zoom · drag = pan · double-click = reset'
              : mode === 'verdicts'
                ? 'scroll to zoom in — the squares become real tiles · colored = answer, dashed amber = declined'
                : 'scroll to zoom in — brightness = familiarity; dim = where it declined'}</p>
            <div className="vctrl" role="group" aria-label="slide controls">
              <button onClick={() => zoomBy(1.4)} aria-label="zoom in" title="zoom in">＋</button>
              <button onClick={() => zoomBy(1 / 1.4)} aria-label="zoom out" title="zoom out">−</button>
              <button onClick={() => panBy(-1, 0)} aria-label="pan left" title="pan left">◀</button>
              <button onClick={() => panBy(1, 0)} aria-label="pan right" title="pan right">▶</button>
              <button onClick={() => panBy(0, -1)} aria-label="pan up" title="pan up">▲</button>
              <button onClick={() => panBy(0, 1)} aria-label="pan down" title="pan down">▼</button>
              <button onClick={rotate90} aria-label="rotate 90 degrees" title="rotate 90°">⟳</button>
              <button onClick={resetAll} aria-label="reset view" title="reset view">⤢</button>
            </div>
            <div className="zoomable"
              style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}>
              <div className="rotwrap" style={{ transform: `rotate(${rot}deg)` }}>
                <img className="underlayer" src={`${apiBase}/slide-overview.png`}
                  alt="the real lymph-node slide, low magnification" draggable={false}
                  onError={e => { (e.target as HTMLInputElement).style.display = 'none'; }} />
                {data.tiles.map(t => (
                  <button key={t.id} data-id={t.id}
                    className={`vcell ${mode === 'familiarity' ? 'fam' : t.declined ? 'dec' : 'ans'} ${sel?.id === t.id ? 'sel' : ''}`}
                    style={{ ...cell(t),
                      borderColor: mode === 'familiarity' ? undefined : t.declined ? undefined : CLASS_COLORS[t.cls] ?? '#666',
                      background: showTiles ? undefined : mode === 'familiarity'
                        ? `rgb(${30 + Math.round(66 * t.familiarity / 100)},${30 + Math.round(73 * t.familiarity / 100)},${46 + Math.round(189 * t.familiarity / 100)})`
                        : t.declined ? undefined : `${CLASS_COLORS[t.cls] ?? '#666'}55` }}
                    onClick={() => setSel(t)}
                    aria-label={`patch ${t.id}: ${t.declined ? 'declined' : t.cls}, familiarity ${t.familiarity}`}
                    aria-pressed={sel?.id === t.id}>
                    {showTiles && (
                      <img className="cellimg" loading="lazy"
                        src={`${apiBase}/slide-tile/${t.row}/${t.col}.png`} alt="" draggable={false} />
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="sidecol">
          <div className="pane inspector" role="status">
            {sel ? (
              <>
                <div className="tilebox">
                  <img className="tileimg" src={`${apiBase}/slide-tile/${sel.row}/${sel.col}.png`}
                    alt={`the actual patch ${sel.id} at full resolution`} />
                  {detail && showCam && !detail.ood && (
                    <canvas ref={camRef} className="cam" aria-hidden="true" />)}
                </div>
                <div className="itxt">
                  <div className="t mono">patch {sel.id} · col {sel.col} row {sel.row}</div>
                  {!detail && !detailErr && <div className="sub2 mono">re-analyzing at full resolution…</div>}
                  {detailErr && <div className="sub2">full-resolution analysis unavailable</div>}
                  {detail && (declined ? (
                    <>
                      <div className="big amber">declined — safe</div>
                      <div className="sub2">familiarity {famPct(detail.energy)}/100 — below its gate</div>
                    </>
                  ) : (
                    <>
                      <div className="big">{detail.top_class}</div>
                      <div className="sub2">{Math.round(detail.confidence * 100)}% sure · familiarity {famPct(detail.energy)}/100</div>
                      <div className="bars">
                        {[...detail.probabilities].sort((a, b) => b.p - a.p).slice(0, 3).map(pr => (
                          <div className="brow" key={pr.class}>
                            <span className="mono blab">{pr.class}</span>
                            <span className="btrack"><i style={{ width: `${Math.round(pr.p * 100)}%`, background: CLASS_COLORS[pr.class] ?? '#666' }} /></span>
                            <span className="mono bpct">{Math.round(pr.p * 100)}%</span>
                          </div>
                        ))}
                      </div>
                      <div className="camrow">
                        <button className="camtog" aria-pressed={showCam} onClick={() => setShowCam(s => !s)}>
                          {showCam ? 'hide evidence' : 'show evidence'}</button>
                        <span className="camnote">model evidence — where it looked · not a pathologist marking</span>
                      </div>
                    </>
                  ))}
                  <details className="why" open><summary>why?</summary>
                    <p>{declined
                      ? 'familiarity below its gate — unlike anything it studied'
                      : 'answered — familiarity above its gate'}</p></details>
                </div>
              </>
            ) : (
              <p className="noclick">click any patch on the slide — its verdict appears here, no scrolling</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
