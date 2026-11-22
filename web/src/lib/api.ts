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

/* ngrok free tier shows an HTML interstitial to browser requests without
   this header — every API fetch carries it. Images can't (no headers on
   <img>), so live-mode images load via assetUrl() blob fetches. */
export const ngrokHeaders: Record<string, string> = {
  'ngrok-skip-browser-warning': '1',
  // ngrok edge basic-auth (traffic policy) — demo gate, not security
  'authorization': 'Basic bmdyb2s6MmZlZDI1MzAxYzc5NGE2YQ==',
};

const blobCache = new Map<string, string>();
export async function assetUrl(path: string): Promise<string> {
  if (staticDemo) return path;
  const hit = blobCache.get(path);
  if (hit) return hit;
  const r = await fetchAwake(`${apiBase}${path}`, { headers: ngrokHeaders });
  if (!r.ok) return `${apiBase}${path}`;
  const u = URL.createObjectURL(await r.blob());
  blobCache.set(path, u);
  return u;
}

/* Wake-aware fetching: the backend is DOWN by default (on-prem steward
   pattern). Any 503 means "booting" — poll /wake until warm, then retry the
   original request once. Cold start is a normal state, never an error. */
async function wakeLoop(): Promise<void> {
  if (typeof window === 'undefined') return;   // SSR: fail fast, pages fall back
  for (let i = 0; i < 24; i++) {            // ~4 min ceiling
    await new Promise(r => setTimeout(r, 10000));
    try {
      const r = await fetch(`${apiBase}/wake`, { headers: ngrokHeaders });
      if (r.ok) return;
    } catch { /* backend mid-boot; keep polling */ }
  }
  throw new Error('backend did not wake within 4 minutes');
}

async function fetchAwake(url: string, init?: RequestInit): Promise<Response> {
  const r = await fetch(url, init);
  if (r.status !== 503) return r;
  await wakeLoop();
  return fetch(url, init);
}

export const get = async <T>(path: string): Promise<T> => {
  const r = await fetchAwake(`${apiBase}${path}`, { headers: ngrokHeaders });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json() as Promise<T>;
};

export async function analyze(file: Blob, model: 'ship' | 'v1'): Promise<Analysis> {
  const fd = new FormData();
  fd.append('file', new File([file], 'patch.png', { type: 'image/png' }));
  fd.append('model', model);
  const r = await fetchAwake(`${apiBase}/analyze-patch`, { method: 'POST', body: fd, headers: ngrokHeaders });
  if (!r.ok) throw new Error(`analyze: ${r.status}`);
  return r.json();
}

export async function analyzeTiff(file: Blob, model: 'ship' | 'v1'): Promise<Analysis> {
  const fd = new FormData();
  fd.append('file', new File([file], 'patch.tif', { type: 'image/tiff' }));
  fd.append('model', model);
  const r = await fetchAwake(`${apiBase}/analyze-patch`, { method: 'POST', body: fd, headers: ngrokHeaders });
  if (!r.ok) throw new Error(`analyze: ${r.status}`);
  return r.json();
}

export interface SlideTile {
  id: string; col: number; row: number; cls: string;
  conf: number; energy: number; familiarity: number; declined: boolean;
}
export interface SlideMapData {
  slide: string; grid: { cols: number; rows: number };
  dims: { w: number; h: number }; patch: number; stride: number;
  gate: number; tiles: SlideTile[];
}
/* Static demo is now the PRIMARY /slide experience: the whole page is
   precomputed into /demo/slide/ (map + overview + tiles + per-tile analyze
   JSON, ~41 MB). The slide page must render and predict NO MATTER WHAT —
   backend reachability only gates the interactive upload demo, never /slide.
   (Live-first proved fragile: the ngrok free edge rate-limits browser
   connections, so tiles fetched fine but fetches threw and predictions
   never appeared.) */
export let staticDemo = true;

export function tileUrl(row: number, col: number): string {
  return staticDemo
    ? `/demo/slide/${row}_${col}.png`
    : `${apiBase}/slide-tile/${row}/${col}.png`;
}

export function overviewUrl(): string {
  return staticDemo ? '/demo/slide/overview.png'
    : `${apiBase}/slide-overview.png`;
}

export async function getSlideMap(): Promise<SlideMapData> {
  const s = await fetch('/demo/slide/slide-map.json');
  if (!s.ok) throw new Error('slide-map unavailable (static snapshot)');
  staticDemo = true;
  return s.json();
}

export async function tileDetail(row: number, col: number): Promise<Analysis> {
  const r = await fetch(`/demo/slide/${row}_${col}.json`);
  if (!r.ok) throw new Error('static detail missing');
  return r.json();
}
