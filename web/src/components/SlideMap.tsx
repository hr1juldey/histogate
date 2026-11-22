import { useEffect, useRef, useState } from 'react';
import { apiBase, getSlideMap, type SlideMapData, type SlideTile } from '../lib/api';

/* Whole-slide view: the REAL slide image as the under-layer, verdicts as
   positioned overlays on top, familiarity heat in a synchronized side pane.
   Toggle hides overlays so the tissue is always inspectable. */
const CLASS_COLORS: Record<string, string> = {
  TUM: '#e46fa8', NORM: '#5dbb7a', MUS: '#9a7ce0', STR: '#6f6ee0',
  LYM: '#4a9ebf', MUC: '#58b6a8', DEB: '#8a8f98', ADI: '#c9a227',
  BACK: '#3a3d42',
};

export default function SlideMap() {
  const [data, setData] = useState<SlideMapData | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState<SlideTile | null>(null);
  const [showVerdicts, setShowVerdicts] = useState(true);
  const heatRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    getSlideMap().then(setData).catch(e => setErr(String(e)));
  }, []);

  /* familiarity heat pane: one square per tile, dark → indigo by familiarity */
  useEffect(() => {
    const c = heatRef.current;
    if (!c || !data) return;
    const ctx = c.getContext('2d')!;
    const cw = c.width / data.grid.cols, ch = c.height / data.grid.rows;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#141516';
    ctx.fillRect(0, 0, c.width, c.height);
    for (const t of data.tiles) {
      const v = t.familiarity / 100;
      ctx.fillStyle = `rgb(${Math.round(30 + 66 * v)},${Math.round(30 + 73 * v)},${Math.round(46 + 189 * v)})`;
      ctx.fillRect(t.col * cw, t.row * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }, [data]);

  if (err) return <p className="map-err">the map is still warming up — refresh in a minute</p>;
  if (!data) return <p className="map-err mono">tiling the slide… (first load takes a few minutes)</p>;

  const answered = data.tiles.filter(t => !t.declined);
  const declinedN = data.tiles.length - answered.length;
  const guides = [...answered]
    .sort((a, b) => (b.cls === 'TUM' ? b.conf : -1) - (a.cls === 'TUM' ? a.conf : -1))
    .slice(0, 3);

  /* overlay cell geometry from real pixel dims */
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
        <button className={`vtg ${showVerdicts ? 'on' : ''}`} onClick={() => setShowVerdicts(v => !v)}
          aria-pressed={showVerdicts}>
          {showVerdicts ? 'verdicts on' : 'verdicts off'}
        </button>
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
          <img className="underlayer" src={`${apiBase}/slide-overview.png`}
            alt="the real lymph-node slide, low magnification"
            onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} />
          <p className="ulcap">the real slide underneath — overlays are the model's verdicts</p>
          {showVerdicts && data.tiles.map(t => (
            <button key={t.id} className={`vcell ${t.declined ? 'dec' : 'ans'} ${sel?.id === t.id ? 'sel' : ''}`}
              style={{ ...cell(t), borderColor: t.declined ? undefined : CLASS_COLORS[t.cls] ?? '#666',
                background: t.declined ? undefined : `${CLASS_COLORS[t.cls] ?? '#666'}55` }}
              onClick={() => setSel(t)}
              aria-label={`patch ${t.id}: ${t.declined ? 'declined' : t.cls}, familiarity ${t.familiarity}`}
              aria-pressed={sel?.id === t.id} />
          ))}
        </div>
        <div className="pane heatp">
          <canvas ref={heatRef} width={512} height={512} aria-hidden="true" />
          <p className="hleg mono">brightness = familiarity<br />one square = one patch</p>
          <p className="hleg2">dim squares declined — the model refused to guess there</p>
        </div>
      </div>
      {sel && (
        <div className="pane inspector" role="status">
          <img className="tileimg" src={`${apiBase}/slide-tile/${sel.row}/${sel.col}.png`}
            alt={`the actual patch ${sel.id}`} />
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
          <details className="why"><summary>why?</summary>
            <p>{sel.declined
              ? 'familiarity below its gate — unlike anything it studied'
              : 'answered — familiarity above its gate'}</p></details>
        </div>
      )}
    </div>
  );
}
