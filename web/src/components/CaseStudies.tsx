import { useState } from 'react';
import type { Metrics, SiteConfig } from '../lib/api';

const Tick = ({ ok }: { ok: boolean }) =>
  ok ? (
    <span className="tick ok"><svg width="9" height="9" viewBox="0 0 10 10"><path d="M1.5 5.5l2.5 2.5 4.5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none"/></svg></span>
  ) : (
    <span className="tick no"><svg width="8" height="8" viewBox="0 0 10 10"><path d="M2 2l6 6M8 2l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none"/></svg></span>
  );

export function GateGrid({ metrics }: { metrics: Metrics }) {
  return (
    <div className="gate">
      {metrics.gate_checks.map(c => (
        <div className="gcell card" key={c.name}>
          <div className="name"><Tick ok={c.passed} />{c.name}</div>
          <div className="val mono">{c.value}</div>
          <div className="thr mono">{c.threshold}</div>
        </div>
      ))}
    </div>
  );
}

/** Case studies: failure → fix, expandable. */
export function CaseStudies({ cfg, metrics }: { cfg: SiteConfig; metrics: Metrics }) {
  const [open, setOpen] = useState<string | null>(cfg.cases[0]?.id ?? null);
  return (
    <div className="cases">
      {cfg.cases.map(c => {
        const isOpen = open === c.id;
        return (
          <div className={`case card ${isOpen ? 'open' : ''}`} key={c.id}>
            <button className="head" onClick={() => setOpen(isOpen ? null : c.id)}>
              <Tick ok={false} />
              <span className="title">{c.title}</span>
              <span className="chev">{isOpen ? '−' : '+'}</span>
            </button>
            {isOpen && (
              <div className="body">
                <p><b className="lab">What failed.</b> {c.what}</p>
                <p><b className="lab">How it was solved.</b> {c.fix}</p>
                {c.id === 'collapse' && metrics.journey.length >= 4 && (
                  <p className="mono nums">
                    {metrics.journey.find(j => j.run === 'overnight-30ep')?.nonorm_f1} →{' '}
                    {metrics.journey.find(j => j.run === 'pr16-mixed')?.nonorm_f1} raw-scanner F1
                  </p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
