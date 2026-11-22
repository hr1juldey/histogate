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
  type ViewMode = 'verdicts' | 'familiarity';
  const [mode, setMode] = useState<ViewMode>('verdicts');

  useEffect(() => {
    getSlideMap().then(setData).catch(e => setErr(String(e)));
  }, []);

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
        <span className="vswitch" role="group" aria-label="view mode">
          {(['verdicts', 'familiarity'] as ViewMode[]).map(m => (
            <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}
              aria-pressed={mode === m}>{m}</button>
          ))}
        </span>
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
        <div className="pane mosaicwrap fitbox"
          style={{ aspectRatio: `${data.dims.w} / ${data.dims.h}`,
            height: 'min(72vh, 700px)', maxWidth: '100%', margin: '0 auto' }}>
          <img className="underlayer" src={`${apiBase}/slide-overview.png`}
            alt="the real lymph-node slide, low magnification"
            onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }} />
          {data.tiles.map(t => (
            <button key={t.id}
              className={`vcell ${mode === 'familiarity' ? 'fam' : t.declined ? 'dec' : 'ans'} ${sel?.id === t.id ? 'sel' : ''}`}
              style={{ ...cell(t),
                borderColor: mode === 'familiarity' ? undefined : t.declined ? undefined : CLASS_COLORS[t.cls] ?? '#666',
                background: mode === 'familiarity'
                  ? `rgb(${30 + Math.round(66 * t.familiarity / 100)},${30 + Math.round(73 * t.familiarity / 100)},${46 + Math.round(189 * t.familiarity / 100)})`
                  : t.declined ? undefined : `${CLASS_COLORS[t.cls] ?? '#666'}55` }}
              onClick={() => setSel(t)}
              aria-label={`patch ${t.id}: ${t.declined ? 'declined' : t.cls}, familiarity ${t.familiarity}`}
              aria-pressed={sel?.id === t.id} />
          ))}
          <p className="ulcap">{mode === 'verdicts'
            ? 'the real slide underneath — colored = its answer, dashed amber = declined'
            : 'same slide, same squares — brightness = familiarity; dim = where it declined'}</p>
        </div>
        <div className="sidecol">
          <div className="pane inspector" role="status">
            {sel ? (
              <>
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
                <details className="why" open><summary>why?</summary>
                  <p>{sel.declined
                    ? 'familiarity below its gate — unlike anything it studied'
                    : 'answered — familiarity above its gate'}</p></details>
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
