import { useEffect, useRef, useState } from 'react';
import { apiBase, getSlideMap, type SlideMapData, type SlideTile } from '../lib/api';

/* Whole-slide view: the REAL slide as under-layer, verdict overlays on top,
   wheel-zoom / drag-pan (overlays stay glued to the tissue — one transform
   moves both). Zoom past 2× and the cells show the actual tile images. */
const CLASS_COLORS: Record<string, string> = {
  TUM: '#e46fa8', NORM: '#5dbb7a', MUS: '#9a7ce0', STR: '#6f6ee0',
  LYM: '#4a9ebf', MUC: '#58b6a8', DEB: '#8a8f98', ADI: '#c9a227',
  BACK: '#3a3d42',
};
const KMIN = 1, KMAX = 8;

interface View { x: number; y: number; k: number }
const HOME: View = { x: 0, y: 0, k: 1 };

export default function SlideMap() {
  const [data, setData] = useState<SlideMapData | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState<SlideTile | null>(null);
  type ViewMode = 'verdicts' | 'familiarity';
  const [mode, setMode] = useState<ViewMode>('verdicts');
  const [view, setView] = useState<View>(HOME);
  const [rot, setRot] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ px: number; py: number; vx: number; vy: number; moved: number } | null>(null);

  useEffect(() => {
    getSlideMap().then(setData).catch(e => setErr(String(e)));
  }, []);

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
        /* keep the image covering the box */
        x = Math.max(r.width * (1 - k), Math.min(0, x));
        y = Math.max(r.height * (1 - k), Math.min(0, y));
        return { x, y, k };
      });
    };
    box.addEventListener('wheel', onWheel, { passive: false });
    return () => box.removeEventListener('wheel', onWheel);
  }, [data]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { px: e.clientX, py: e.clientY, vx: view.x, vy: view.y, moved: 0 };
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
  const onPointerUp = () => { drag.current = null; };
  const wasDrag = () => (drag.current?.moved ?? 0) > 5;

  /* visible controls (GitHub-viewer style): the hint that this canvas moves */
  const clamp = (v: View, r: DOMRect): View => ({
    ...v,
    x: Math.max(r.width * (1 - v.k), Math.min(0, v.x)),
    y: Math.max(r.height * (1 - v.k), Math.min(0, v.y)),
  });
  const zoomBy = (f: number) => {
    const b = boxRef.current; if (!b) return;
    const r = b.getBoundingClientRect();
    setView(v => clamp({ ...stepZoom(v, r, f) }, r));
  };
  const stepZoom = (v: View, r: DOMRect, f: number): View => {
    const k = Math.max(KMIN, Math.min(KMAX, v.k * f));
    if (k <= KMIN + 1e-3) return HOME;
    const px = r.width / 2, py = r.height / 2;
    const wx = (px - v.x) / v.k, wy = (py - v.y) / v.k;
    return { x: px - wx * k, y: py - wy * k, k };
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
        {(view.k > KMIN + 1e-3 || rot !== 0) && (
          <button className="vtg" onClick={resetAll}>reset view</button>
        )}
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
            style={{ aspectRatio: rot % 180 ? `${data.dims.h} / ${data.dims.w}` : `${data.dims.w} / ${data.dims.h}`,
              height: 'min(82vh, 900px)', width: 'auto', maxWidth: '100%' }}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove}
            onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
            onDoubleClick={() => setView(HOME)}>
            <div className="vctrl" role="group" aria-label="slide controls">
              <button onClick={() => zoomBy(1.4)} aria-label="zoom in" title="zoom in">＋</button>
              <button onClick={() => zoomBy(1 / 1.4)} aria-label="zoom out" title="zoom out">−</button>
              <button onClick={() => panBy(-1, 0)} aria-label="pan left" title="pan left">◀</button>
              <button onClick={() => panBy(1, 0)} aria-label="pan right" title="pan right">▶</button>
              <button onClick={() => panBy(0, -1)} aria-label="pan up" title="pan up">▲</button>
              <button onClick={() => panBy(0, 1)} aria-label="pan down" title="pan down">▼</button>
              <button onClick={rotate90} aria-label="rotate 90°" title="rotate 90°">⟳</button>
              <button onClick={resetAll} aria-label="reset view" title="reset view">⤢</button>
            </div>
            <div className="zoomable"
              style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}>
            <div className="rotwrap" style={{ transform: `rotate(${rot}deg)` }}>
              <img className="underlayer" src={`${apiBase}/slide-overview.png`}
                alt="the real lymph-node slide, low magnification" draggable={false}
                onError={e => { (e.target as HTMLInputElement).style.display = 'none'; }} />
              {data.tiles.map(t => (
                <button key={t.id}
                  className={`vcell ${mode === 'familiarity' ? 'fam' : t.declined ? 'dec' : 'ans'} ${sel?.id === t.id ? 'sel' : ''}`}
                  style={{ ...cell(t),
                    borderColor: mode === 'familiarity' ? undefined : t.declined ? undefined : CLASS_COLORS[t.cls] ?? '#666',
                    background: showTiles ? undefined : mode === 'familiarity'
                      ? `rgb(${30 + Math.round(66 * t.familiarity / 100)},${30 + Math.round(73 * t.familiarity / 100)},${46 + Math.round(189 * t.familiarity / 100)})`
                      : t.declined ? undefined : `${CLASS_COLORS[t.cls] ?? '#666'}55` }}
                  onClick={() => { if (!wasDrag()) setSel(t); }}
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
          <p className="ulcap">{view.k > KMIN + 1e-3
            ? 'scroll = zoom · drag = pan · double-click = reset'
            : mode === 'verdicts'
              ? 'scroll to zoom in — the squares become real tiles · colored = answer, dashed amber = declined'
              : 'scroll to zoom in — brightness = familiarity; dim = where it declined'}</p>
        </div>
        <div className="sidecol">
          <div className="pane inspector" role="status">
            {sel ? (
              <>
                <img className="tileimg" src={`${apiBase}/slide-tile/${sel.row}/${sel.col}.png`}
                  alt={`the actual patch ${sel.id}`} />
                <div className="itxt">
                  <div className="t mono">patch {sel.id} · col {sel.col} row {sel.row}</div>
                  {sel.declined ? (
                    <>
                      <div className="big amber">declined — safe</div>
                      <div className="sub2">familiarity {sel.familiarity}/100 — below its gate</div>
                    </>
                  ) : (
                    <>
                      <div className="big">{sel.cls}</div>
                      <div className="sub2">{Math.round(sel.conf * 100)}% sure · familiarity {sel.familiarity}/100</div>
                    </>
                  )}
                  <details className="why" open><summary>why?</summary>
                    <p>{sel.declined
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
