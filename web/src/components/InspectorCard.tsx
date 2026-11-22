import { useEffect, useRef, useState } from 'react';
import { famPct } from '../lib/familiarity';
import { CLASS_COLORS, paintCam, type PatchDetail } from '../lib/viewer';

/* The shared verdict panel: full-res tile, three-state verdict headline,
   familiarity, top-3 probability bars, Grad-CAM evidence overlay (model
   evidence — never a pathologist marking), why? — one grammar everywhere. */
interface Props {
  title: string;                       /* e.g. "patch t21 · col 7 row 11" */
  tileSrc: string | null;
  detail: PatchDetail | null;
  loading: boolean;
  error: boolean;
  badge?: React.ReactNode;             /* extra verdict pill (e.g. expert-label comparison) */
  placeholder?: string;
  filename?: string | null;            /* user upload name */
}

export default function InspectorCard({ title, tileSrc, detail, loading, error,
  badge, placeholder, filename }: Props) {
  const camRef = useRef<HTMLCanvasElement>(null);
  const [showCam, setShowCam] = useState(true);

  useEffect(() => {
    const c = camRef.current;
    if (!c || !detail || !showCam || detail.ood) return;
    paintCam(c, detail.gradcam);
  }, [detail, showCam]);

  return (
    <div className="pane inspector" role="status">
      {tileSrc ? (
        <>
          <div className="tilebox">
            <img className="tileimg" src={tileSrc} alt={`${title} at full resolution`} />
            {detail && showCam && !detail.ood && (
              <canvas ref={camRef} className="cam" aria-hidden="true" />)}
          </div>
          <div className="itxt">
            <div className="t mono">{title}{filename ? ` · ${filename}` : ''}</div>
            {loading && <div className="sub2 mono">analyzing at full resolution…</div>}
            {error && <div className="sub2">analysis unavailable</div>}
            {badge}
            {detail && (detail.ood ? (
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
              <p>{detail?.ood
                ? 'familiarity below its gate — unlike anything it studied'
                : 'answered — familiarity above its gate'}</p></details>
          </div>
        </>
      ) : (
        <p className="noclick">{placeholder ?? 'load an image — its verdict appears here'}</p>
      )}
    </div>
  );
}
