import { useEffect, useRef, useState } from 'react';
import { analyze, analyzeTiff, apiBase, get, type Analysis, type Sample } from '../lib/api';
import PatchViewer from './PatchViewer';
import InspectorCard from './InspectorCard';

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

/** Unified demo island — same window, same controls, same inspector as the
    whole-slide page. One grammar everywhere. */
export default function Demo() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [j, setJ] = useState<Analysis | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [natural, setNatural] = useState({ w: 448, h: 448 });
  const [err, setErr] = useState(false);
  const [fname, setFname] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const touched = useRef(false);   /* stop auto-pick once the user acts */

  useEffect(() => { get<Sample[]>('/samples').then(s => setSamples(s.filter(x => x.id !== 'nonorm'))); }, []);

  const measure = (url: string) => {
    const im = new Image();
    im.onload = () => setNatural({ w: im.naturalWidth || 448, h: im.naturalHeight || 448 });
    im.src = url;
  };

  const runBlob = async (blob: Blob, displayUrl: string, isTiff: boolean, name: string | null) => {
    setSrc(displayUrl); setJ(null); setErr(false); setFname(name);
    measure(displayUrl);
    try { setJ(isTiff ? await analyzeTiff(blob, 'ship') : await analyze(blob, 'ship')); }
    catch { setErr(true); }
  };

  const pick = async (id: string) => {
    touched.current = true;
    setSel(id);
    const r = await fetch(`${apiBase}/samples/${id}`);
    await runBlob(await r.blob(), `${apiBase}/samples/${id}.png`, true, null);
  };

  const onFile = (f: File | null | undefined) => {
    if (!f) return;
    touched.current = true;
    setSel(null);
    const tiff = f.type === 'image/tiff' || f.name.toLowerCase().endsWith('.tif') || f.name.toLowerCase().endsWith('.tiff');
    runBlob(f, URL.createObjectURL(f), tiff, f.name);
  };

  useEffect(() => { if (samples.length && sel === null && !touched.current) pick(samples[0].id); }, [samples, sel]); // eslint-disable-line react-hooks/exhaustive-deps

  /* three-state verdict against the doctor's label, when we have one */
  const badge = sel && TRUTH[sel] && j ? (
    j.ood ? (
      <div className="vbadge ref">expert label {TRUTH[sel]} · AI declined — safe</div>
    ) : j.top_class === TRUTH[sel] ? (
      <div className="vbadge match">expert label {TRUTH[sel]} · AI agrees</div>
    ) : (
      <div className="vbadge miss">expert label {TRUTH[sel]} · AI says {j.top_class}</div>
    )
  ) : null;

  return (
    <div className="demo-uni">
      <div className="dgutter">
        <div className="gallery">
          {samples.map(s => (
            <button key={s.id} className={`tile ${sel === s.id ? 'on' : ''}`} onClick={() => pick(s.id)}>
              <img src={`${apiBase}/samples/${s.id}.png`} alt={s.why} />
              <span>{s.id.toUpperCase()}</span>
            </button>
          ))}
        </div>
        {sel && <p className="gloss">{GLOSS[sel]}</p>}
        <div className="own"
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); onFile(e.dataTransfer.files?.[0]); }}>
          <input ref={fileRef} type="file" accept="image/*,.tif,.tiff" hidden
            onChange={e => onFile(e.target.files?.[0])} />
          <button className="upl" onClick={() => fileRef.current?.click()}>Upload your own</button>
          <p className="drophint">or drop an image here</p>
        </div>
      </div>
      <div className="panes">
        {src ? (
          <PatchViewer src={src} alt={fname ?? 'tissue patch'} natural={natural}
            reserveWidth={380}
            captionIdle="scroll to zoom in — drag to look around · rotate if you like" />
        ) : <div className="pane mosaicwrap" />}
        <div className="sidecol">
          <InspectorCard
            title={sel ? `sample ${sel}` : 'your image'}
            tileSrc={src} detail={j} loading={!!src && !j && !err} error={err}
            badge={badge} filename={fname}
            placeholder="pick a sample or upload — the verdict appears here" />
        </div>
      </div>
    </div>
  );
}
