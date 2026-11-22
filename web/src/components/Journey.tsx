import { useEffect, useState } from 'react';
import type { Metrics, SiteConfig } from '../lib/api';

/** Interactive training-journey stepper: click a stage, see its measured bars. */
export default function Journey({ cfg, metrics }: { cfg: SiteConfig; metrics: Metrics }) {
  const [sel, setSel] = useState(cfg.journey.length - 1);
  const stage = cfg.journey[sel];
  const data = metrics.journey.find(j => j.run === stage.run);
  const [w1, setW1] = useState(0);
  useEffect(() => { setW1(0); const t = setTimeout(() => setW1(1), 60); return () => clearTimeout(t); }, [sel]);
  if (!data) return null;
  const bar = (v: number, label: string, cls: string) => (
    <div className="barrow">
      <span className="bl">{label}</span>
      <div className="track"><div className={`fill ${cls}`} style={{ width: `${w1 ? v * 100 : 0}%` }} /></div>
      <span className="bv mono">{v.toFixed(4)}</span>
    </div>
  );
  return (
    <div className="journey">
      <div className="stages">
        {cfg.journey.map((s, i) => (
          <button key={s.run} className={i === sel ? 'on' : ''} onClick={() => setSel(i)}>
            <span className="n mono">{String(i + 1).padStart(2, '0')}</span>
            <span className="t">{s.title}</span>
          </button>
        ))}
      </div>
      <div className="detail card">
        <h3>{stage.title}</h3>
        <p>{stage.note}</p>
        <div className="bars">
          {bar(data.crcval_f1, "score on the test it studied for", 'crc')}
          {bar(data.nonorm_f1, 'score on machines it never saw', 'dom')}
        </div>
        <p className="foot">
          {data.nonorm_f1 < 0.9
            ? 'The gap is the failure: same patients, different machine — and it falls apart.'
            : data.nonorm_f1 > data.crcval_f1
              ? 'The failure flipped: it now does better on machines it never saw than on its own homework.'
              : 'Both hold — but regaining the new machines cost it some edge at home.'}
        </p>
        {stage.run_url && (
          <a className="runlink" href={stage.run_url} target="_blank" rel="noopener">
            Training curves ↗</a>
        )}
      </div>
    </div>
  );
}
