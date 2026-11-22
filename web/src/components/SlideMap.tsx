import { useEffect, useRef, useState } from 'react';
import { getSlideMap, type SlideMapData, type SlideTile } from '../lib/api';

/* Whole-slide view: verdict mosaic + synchronized familiarity heat pane.
   Design law: dual panes, never an occluding overlay (DiVA finding). */
const CLASS_COLORS: Record<string, string> = {
  TUM: '#c25b8d', NORM: '#5dbb7a', MUS: '#7c5cd6', STR: '#5e6ad2',
  LYM: '#4a9ebf', MUC: '#58b6a8', DEB: '#8a8f98', ADI: '#c9a227',
  BACK: '#3a3d42',
};

export default function SlideMap() {
  const [data, setData] = useState<SlideMapData | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState<SlideTile | null>(null);
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

  const guides = data.tiles
    .filter(t => !t.declined)
    .sort((a, b) => (b.cls === 'TUM' ? b.conf : -1) - (a.cls === 'TUM' ? a.conf : -1))
    .slice(0, 3);

  return (
    <div className="slide-sec">
      <div className="guides">
        <span className="glab">look here first:</span>
        {guides.map((t, i) => (
          <button key={t.id} className="guide mono" onClick={() => setSel(t)}
            aria-label={`patch ${t.id}: ${t.cls}, ${Math.round(t.conf * 100)}% sure, familiarity ${t.familiarity}`}>
            {i + 1}. {t.cls} {Math.round(t.conf * 100)}% · fam {t.familiarity}
          </button>
        ))}
      </div>
      <div className="panes">
        <div className="pane mosaic" role="grid" aria-label="verdict tiles"
          style={{ gridTemplateColumns: `repeat(${data.grid.cols}, 1fr)` }}>
          {data.tiles.map(t => (
            <button key={t.id} role="gridcell" onClick={() => setSel(t)}
              aria-label={`patch ${t.id}: ${t.declined ? 'declined' : t.cls}, familiarity ${t.familiarity}`}
              aria-pressed={sel?.id === t.id}
              className={`vtile ${t.declined ? 'declined' : ''} ${sel?.id === t.id ? 'sel' : ''}`}
              style={{ background: t.declined ? 'transparent' : CLASS_COLORS[t.cls] ?? '#666' }}
              title={t.declined ? `declined — familiarity ${t.familiarity}` : `${t.cls} ${Math.round(t.conf * 100)}% · fam ${t.familiarity}`} />
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
