/* HistoGate service worker — the device's local copy of the demo.
   Cache-first for shell + snapshots (tier-0/1), network-only for live
   inference (unknown uploads). Second visit works fully offline. */
const CACHE = 'histogate-v1';
const SHELL = [
  '/', '/story', '/slide', '/demo', '/contact',
  '/manifest.webmanifest', '/favicon.svg',
  '/samples/tumor.png', '/samples/normal.png',
  '/sample-analyses.json', '/samples.json',
  '/demo/slide/slide-map.json', '/demo/slide/overview-720.jpg',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;        // live API stays online
  if (e.request.method !== 'GET') return;
  const isSnapshot = url.pathname.startsWith('/demo/slide/')
    || url.pathname.startsWith('/samples/')
    || url.pathname.startsWith('/_astro/')
    || url.pathname === '/story-dial-curve.json';
  if (!isSnapshot && !SHELL.includes(url.pathname)) return;

  e.respondWith(
    caches.match(e.request).then((hit) => hit
      ? fetch(e.request).then((r) => {
          if (r.ok) caches.open(CACHE).then((c) => c.put(e.request, r.clone()));
          return r;
        }).catch(() => hit)                               // offline: serve cached
      : fetch(e.request).then((r) => {
          if (r.ok) caches.open(CACHE).then((c) => c.put(e.request, r.clone()));
          return r;
        })));
});
