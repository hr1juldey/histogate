export interface Analysis {
  probabilities: { class: string; p: number }[];
  top_class: string;
  confidence: number;
  energy: number;
  energy_threshold: number;
  ood: boolean;
  verdict: string;
  gradcam: number[][];
}
export interface Sample { id: string; why: string }
export interface Metrics {
  crcval_f1: number; domain_f1: number; int8_agree: number;
  throughput: number; ece: number; revision: string;
  journey: { run: string; crcval_f1: number; nonorm_f1: number }[];
  gate_checks: { name: string; value: string; threshold: string; passed: boolean }[];
}
export interface SiteConfig {
  product: { name: string; tagline: string; hero_sub: string;
    primary_cta: { label: string; href: string };
    secondary_cta: { label: string; href: string } };
  journey: { run: string; title: string; note: string; run_url?: string }[];
  cases: { id: string; title: string; what: string; fix: string; metric: string }[];
  gate_note: string; who: string;
  about: {
    name: string; role: string; bio: string;
    background: { dates: string; what: string }[];
    links: { label: string; href: string }[];
  };
}

export const apiBase: string =
  (import.meta.env.PUBLIC_API_BASE as string | undefined) ??
  'http://127.0.0.1:8000';

export const get = async <T>(path: string): Promise<T> => {
  const r = await fetch(`${apiBase}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json() as Promise<T>;
};

export async function analyze(file: Blob, model: 'ship' | 'v1'): Promise<Analysis> {
  const fd = new FormData();
  fd.append('file', new File([file], 'patch.png', { type: 'image/png' }));
  fd.append('model', model);
  const r = await fetch(`${apiBase}/analyze-patch`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`analyze: ${r.status}`);
  return r.json();
}

export async function analyzeTiff(file: Blob, model: 'ship' | 'v1'): Promise<Analysis> {
  const fd = new FormData();
  fd.append('file', new File([file], 'patch.tif', { type: 'image/tiff' }));
  fd.append('model', model);
  const r = await fetch(`${apiBase}/analyze-patch`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`analyze: ${r.status}`);
  return r.json();
}
