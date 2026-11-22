import { useEffect, useRef, useState } from 'react';
import { clampView, HOME, KMAX, KMIN, type View } from '../lib/viewer';

/* The shared image window — one grammar everywhere:
   fixed aspect-honest frame (sized in px, never reshaped), visible GitHub-style
   control cluster (zoom / pan / rotate-90 / reset), wheel-zoom anchored at the
   cursor, drag-pan with pointer-capture that never steals clicks from controls
   or children, rigid 90-degree rotation (window crops, never stretches),
   rotation-aware pan bounds, in-frame caption overlay (zero layout shift). */
interface Props {
  src: string;
  alt: string;
  captionIdle: string;                 /* caption shown at rest */
  natural: { w: number; h: number };   /* content aspect reference (px) */
  reserveWidth?: number;               /* px to subtract from available width (a side column) */
  children?: React.ReactNode;          /* overlays (absolute %) inside the rotated layer */
  onTapChild?: (id: string) => void;   /* tap (not drag) on a child carrying data-id */
  onView?: (k: number, rot: number) => void;
  onImgError?: () => void;
}

const CAPTION_GESTURE = 'scroll = zoom · drag = pan · double-click = reset';

export default function PatchViewer({ src, alt, captionIdle, natural,
  reserveWidth = 0, children, onTapChild, onView, onImgError }: Props) {
  const [view, setView] = useState<View>(HOME);
  const [rot, setRot] = useState(0);
  const [box, setBox] = useState<{ w: number; h: number } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ px: number; py: number; vx: number; vy: number; moved: number; id: string | null } | null>(null);

  /* frame sized in PIXELS so its aspect ALWAYS equals the content's aspect */
  useEffect(() => {
    const compute = () => {
      /* measure a container whose width does not depend on this box (avoid
         circular sizing): .panes if present, else the direct parent */
      const container = wrapRef.current?.closest('.panes') ?? wrapRef.current?.parentElement;
      const mobile = window.innerWidth <= 900;
      const availW = Math.max(160, (container?.clientWidth ?? 640) - 16 - (mobile ? 0 : reserveWidth));
      const ar = natural.w / natural.h;
      let h = Math.min(window.innerHeight * 0.82, 900);
      let w = h * ar;
      if (w > availW) { w = availW; h = w / ar; }
      setBox({ w: Math.round(w), h: Math.round(h) });
    };
    compute();
    window.addEventListener('resize', compute);
    return () => window.removeEventListener('resize', compute);
  }, [natural.w, natural.h, reserveWidth]);

  useEffect(() => { onView?.(view.k, rot); }, [view.k, rot]); // eslint-disable-line react-hooks/exhaustive-deps

  /* wheel zoom anchored at the cursor; non-passive so we can preventDefault */
  useEffect(() => {
    const b = boxRef.current;
    if (!b) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = b.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      setView(v => {
        const k = Math.max(KMIN, Math.min(KMAX, v.k * Math.exp(-e.deltaY * 0.0016)));
        const wx = (px - v.x) / v.k, wy = (py - v.y) / v.k;
        const nv = { x: px - wx * k, y: py - wy * k, k };
        return clampView(k <= KMIN + 1e-3 ? HOME : nv, r, rot);
      });
    };
    b.addEventListener('wheel', onWheel, { passive: false });
    return () => b.removeEventListener('wheel', onWheel);
  }, [rot, src]);

  /* pan surface: capture + drag, but NEVER steal presses on the control
     buttons or a plain tap on a child (that's a select, not a drag) */
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const t = e.target as HTMLElement;
    if (t.closest('.vctrl')) return;
    const child = t.closest('[data-id]');
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = {
      px: e.clientX, py: e.clientY, vx: view.x, vy: view.y, moved: 0,
      id: child?.getAttribute('data-id') ?? null,
    };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.px, dy = e.clientY - d.py;
    d.moved = Math.max(d.moved, Math.hypot(dx, dy));
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setView(v => clampView({ ...v, x: d.vx + dx, y: d.vy + dy }, r, rot));
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d && d.id && d.moved <= 5) onTapChild?.(d.id);
  };

  const zoomBy = (f: number) => {
    const b = boxRef.current; if (!b) return;
    const r = b.getBoundingClientRect();
    setView(v => {
      const k = Math.max(KMIN, Math.min(KMAX, v.k * f));
      if (k <= KMIN + 1e-3) return clampView(HOME, r, rot);
      const px = r.width / 2, py = r.height / 2;
      const wx = (px - v.x) / v.k, wy = (py - v.y) / v.k;
      return clampView({ x: px - wx * k, y: py - wy * k, k }, r, rot);
    });
  };
  const panBy = (fx: number, fy: number) => {
    const b = boxRef.current; if (!b) return;
    const r = b.getBoundingClientRect();
    setView(v => clampView({ ...v, x: v.x - fx * r.width * 0.15, y: v.y - fy * r.height * 0.15 }, r, rot));
  };
  const rotate90 = () => {
    const nr = (rot + 90) % 360;
    setRot(nr);
    const b = boxRef.current;
    if (b) { const r = b.getBoundingClientRect(); setView(v => clampView(v, r, nr)); }
  };
  const resetAll = () => { setRot(0); setView(HOME); };

  return (
    <div className="pvpane" ref={wrapRef}>
      <div ref={boxRef} className="fitbox zoombox"
        style={box ? { width: `${box.w}px`, height: `${box.h}px` } : undefined}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove}
        onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
        onDoubleClick={() => { const b = boxRef.current;
          if (b) setView(v => clampView({ ...v, x: 0, y: 0, k: KMIN }, b.getBoundingClientRect(), rot)); }}>
        <p className="ulcap">{view.k > KMIN + 1e-3 ? CAPTION_GESTURE : captionIdle}</p>
        <div className="vctrl" role="group" aria-label="image controls">
          <button onClick={() => zoomBy(1.4)} aria-label="zoom in" title="zoom in">＋</button>
          <button onClick={() => zoomBy(1 / 1.4)} aria-label="zoom out" title="zoom out">−</button>
          <button onClick={() => panBy(-1, 0)} aria-label="pan left" title="pan left">◀</button>
          <button onClick={() => panBy(1, 0)} aria-label="pan right" title="pan right">▶</button>
          <button onClick={() => panBy(0, -1)} aria-label="pan up" title="pan up">▲</button>
          <button onClick={() => panBy(0, 1)} aria-label="pan down" title="pan down">▼</button>
          <button onClick={rotate90} aria-label="rotate 90 degrees" title="rotate 90°">⟳</button>
          <button onClick={resetAll} aria-label="reset view" title="reset view">⤢</button>
        </div>
        <div className="zoomable"
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}>
          <div className="rotwrap" style={{ transform: `rotate(${rot}deg)` }}>
            <img className="underlayer" src={src} alt={alt} draggable={false}
              onError={onImgError} />
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
