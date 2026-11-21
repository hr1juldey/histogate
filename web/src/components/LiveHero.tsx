import { useEffect, useRef, useState } from 'react';
import { analyze, apiBase, get, type Metrics, type Sample } from '../lib/api';

/** Hero island: cycles real samples through the live model on load. */
export default function LiveHero() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [idx, setIdx] = useState(0);
  const [res, setRes] = useState<{ top: string; conf: number; ood: boolean } | null>(null);
  const busy = useRef(false);

  useEffect(() => {
    get<Sample[]>('/samples').then(s => setSamples(s.filter(x => x.id !== 'nonorm')));
  }, []);

  useEffect(() => {
    if (!samples.length) return;
    let cancelled = false;
    const run = async () => {
      if (busy.current) return;
      busy.current = true;
      const s = samples[idx % samples.length];
      try {
        const r = await fetch(`${apiBase}/samples/${s.id}`);
        const blob = await r.blob();
        const j = await analyze(blob, 'ship');
        if (!cancelled) setRes({ top: j.top_class, conf: j.confidence, ood: j.ood });
      } catch { /* backend cold — keep last */ }
      finally { busy.current = false; }
      if (!cancelled) setTimeout(() => !cancelled && setIdx(i => i + 1), 3800);
    };
    run();
    return () => { cancelled = true; };
  }, [idx, samples]);

  const cur = samples.length ? samples[idx % samples.length] : null;

  return (
    <div className="livehero">
      <div className="stack">
        {cur && <img src={`${apiBase}/samples/${cur.id}.png`} alt={cur.why} key={cur.id} />}
        <div className="readout mono">
          {res ? (
            <>
              <span className={res.ood ? 'warn' : 'ok'}>
                {res.top} · {(res.conf * 100).toFixed(1)}%
              </span>
              <span className="dim">live inference · model pr16-mixed</span>
            </>
          ) : (
            <span className="dim">waking the model…</span>
          )}
        </div>
      </div>
      <div className="rail">
        {samples.map((s, i) => (
          <img
            key={s.id}
            src={`${apiBase}/samples/${s.id}.png`}
            alt=""
            className={i === idx % samples.length ? 'on' : ''}
          />
        ))}
      </div>
    </div>
  );
}
