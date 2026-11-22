import { useEffect, useState } from 'react';
import { assetUrl, getSlideMap, tileDetail, tileUrl,
         type SlideMapData, type SlideTile } from '../lib/api';
import type { PatchDetail } from '../lib/viewer';
import { CLASS_COLORS } from '../lib/viewer';
import PatchViewer from './PatchViewer';
import InspectorCard from './InspectorCard';

/* Whole-slide view on the shared viewer grammar: the REAL slide under-layer,
   verdict overlays as children of the window, full-res re-analysis in the
   shared inspector. Window/controls/rotation logic lives in PatchViewer. */
export default function SlideMap() {
  const [data, setData] = useState<SlideMapData | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState<SlideTile | null>(null);
  type ViewMode = 'verdicts' | 'familiarity';
  const [mode, setMode] = useState<ViewMode>('verdicts');
  const [zoomed, setZoomed] = useState(false);
  const [detail, setDetail] = useState<PatchDetail | null>(null);
  const [detailErr, setDetailErr] = useState(false);
  const [overview, setOverview] = useState<string | null>(null);
  const [tileSrc, setTileSrc] = useState<string | null>(null);

  useEffect(() => {
    getSlideMap().then(setData).catch(e => setErr(String(e)));
    assetUrl(overviewUrl()).then(setOverview);
  }, []);

  /* clicking a patch re-analyzes that exact tile at full resolution */
  useEffect(() => {
    if (!sel) { setDetail(null); setDetailErr(false); return; }
    let live = true;
    setDetail(null); setDetailErr(false);
    setTileSrc(null);
    (async () => {
      try {
        const [j, u] = await Promise.all([
          tileDetail(sel.row, sel.col) as unknown as Promise<PatchDetail>,
          assetUrl(tileUrl(sel.row, sel.col)),
        ]);
        if (live) { setDetail(j); setTileSrc(u); }
      } catch { if (live) setDetailErr(true); }
    })();
    return () => { live = false; };
  }, [sel]);

  if (err) return <p className="map-err">the map is still warming up — refresh in a minute</p>;
  if (!data) return <p className="map-err mono">tiling the slide… (first load takes a few minutes)</p>;

  const answered = data.tiles.filter(t => !t.declined);
  const declinedN = data.tiles.length - answered.length;
  const guides = [...answered]
    .sort((a, b) => (b.cls === 'TUM' ? b.conf : -1) - (a.cls === 'TUM' ? a.conf : -1))
    .slice(0, 3);
  const showTiles = zoomed;

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
        <div className="pane mosaicwrap">
          <PatchViewer
            src={overview ?? 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'}
            alt="the real lymph-node slide, low magnification"
            captionIdle={mode === 'verdicts'
              ? 'scroll to zoom in — the squares become real tiles · colored = answer, dashed amber = declined'
              : 'scroll to zoom in — brightness = familiarity; dim = where it declined'}
            natural={data.dims}
            reserveWidth={380}
            onView={k => setZoomed(k >= 2)}
            onTapChild={id => { const t = data.tiles.find(x => x.id === id); if (t) setSel(t); }}>
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
                    src={tileUrl(t.row, t.col)} alt="" draggable={false} />
                )}
              </button>
            ))}
          </PatchViewer>
        </div>
        <div className="sidecol">
          <InspectorCard
            title={sel ? `patch ${sel.id} · col ${sel.col} row ${sel.row}` : ''}
            tileSrc={tileSrc}
            detail={detail} loading={!!sel && !detail && !detailErr} error={detailErr}
            placeholder="click any patch on the slide — its verdict appears here, no scrolling" />
        </div>
      </div>
    </div>
  );
}
