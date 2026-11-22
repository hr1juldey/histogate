import { useEffect, useRef, useState } from 'react';
import sampleAnalyses from '../data/sample-analyses.json';
import samplesFallback from '../data/samples.json';
import { type Analysis, type Sample } from '../lib/api';

import { ELO, EHI, famPct } from '../lib/familiarity';
const pct = (e: number) => 100 - famPct(e);

/** Hero island: cycles known samples through CACHED model verdicts on load,
 *  with the model's own evidence heatmap over the tissue. Only unknown-data
 *  uploads hit the live backend — known samples are cache, always render. */
export default function LiveHero() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [idx, setIdx] = useState(0);
  const [res, setRes] = useState<Analysis | null>(null);
  const camRef = useRef<HTMLCanvasElement>(null);
  const busy = useRef(false);
  const manual = useRef(false);

  useEffect(() => {
    setSamples((samplesFallback as unknown as Sample[]).filter(x => x.id !== 'nonorm'));
  }, []);

  useEffect(() => {
    if (!samples.length) return;
    let cancelled = false;
    const run = async () => {
      if (busy.current) return;
      busy.current = true;
      const s = samples[idx % samples.length];
      try {
        const j = (sampleAnalyses as unknown as Record<string, Analysis>)[s.id]
          ?? null;   // precomputed verdict — no backend, no wake, instant
        if (!cancelled) setRes(j);
        // draw the model's evidence heatmap where it looked
        const cam = camRef.current?.getContext('2d');
        if (cam && j) {
          const flat = (j.gradcam as unknown as number[][]).flat?.() ?? (j.gradcam as unknown as number[]);
          const w = Math.round(Math.sqrt(flat.length));
          const max = Math.max(...flat) || 1;
          // hot region = cells within 55% of peak; edge cells get a solid outline
          const hi = flat.map(v => v >= 0.55 * max);
          const at = (cx: number, cy: number) =>
            cx < 0 || cy < 0 || cx >= w || cy >= w ? false : hi[cy * w + cx];
          const d = cam.createImageData(448, 448);
          for (let y = 0; y < 448; y++) for (let x = 0; x < 448; x++) {
            const cx = Math.floor((x / 448) * w), cy = Math.floor((y / 448) * w);
            const v = flat[cy * w + cx] / max;
            const i = (y * 448 + x) * 4;
            const isHi = at(cx, cy);
            const isEdge = isHi && (!at(cx - 1, cy) || !at(cx + 1, cy) || !at(cx, cy - 1) || !at(cx, cy + 1));
            d.data[i] = 255; d.data[i + 1] = 40; d.data[i + 2] = 40;
            d.data[i + 3] = isEdge ? 255
              : isHi ? 255 * Math.min(1, 0.35 + v) * 0.65
              : 255 * v * 0.25;
          }
          cam.putImageData(d, 0, 0);
        }
      } catch { /* backend cold — keep last */ }
      finally { busy.current = false; }
      // manual pick holds longer before autoplay resumes
      const hold = manual.current ? 12000 : 4200;
      manual.current = false;
      if (!cancelled) setTimeout(() => !cancelled && setIdx(i => i + 1), hold);
    };
    run();
    return () => { cancelled = true; };
  }, [idx, samples]);

  const pick = (i: number) => {
    manual.current = true;
    setIdx(i);
  };

  const cur = samples.length ? samples[idx % samples.length] : null;

  return (
    <div className="livehero">
      <div className="stack">
        {cur && <img src={`/samples/${cur.id}.png`} alt={cur.why} key={cur.id} />}
        <canvas ref={camRef} width={448} height={448} aria-hidden="true" />
        <div className="readout mono">
          {res ? (
            <>
              <span className={`lamp ${res.ood ? 'amber' : 'green'}`} aria-hidden="true" />
              <span className={res.ood ? 'warn' : 'ok'}>
                {res.ood ? 'declined — safe' : `${res.top_class} · ${(res.confidence * 100).toFixed(1)}%`}
              </span>
              <span className="mini-gauge" aria-hidden="true" title="familiarity gauge: familiar answers, unfamiliar declines — amber tick = its gate">
                <i style={{ width: `${pct(res.energy)}%`, background: res.ood ? '#e3b341' : '#5dbb7a' }} />
                <b style={{ left: `${pct(res.energy_threshold)}%` }} />
              </span>
              <span className="dim">red = where the model looked</span>
            </>
          ) : (
            <span className="dim">waking the model…</span>
          )}
        </div>
      </div>
      <div className="rail" role="group" aria-label="choose a tissue sample">
        {samples.map((s, i) => (
          <button
            key={s.id}
            onClick={() => pick(i)}
            aria-label={`show ${s.id}`}
            aria-pressed={i === idx % samples.length}
            className={i === idx % samples.length ? 'on' : ''}
          >
            <img src={`/samples/${s.id}.png`} alt="" />
          </button>
        ))}
      </div>
    </div>
  );
}
