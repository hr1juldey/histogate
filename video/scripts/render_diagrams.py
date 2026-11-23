"""Frame-stepped rendering of the diagram HTML pages via CDP captureScreenshot.
Each scene: video/frames/<name>/f%04d.png -> video/scenes/<name>.mp4 (30fps)."""
import asyncio, base64, json, os, subprocess, sys
sys.path.insert(0, "/home/riju279/Documents/code/AIRA/histogate/video/scripts")
import cdp

ROOT = "/home/riju279/Documents/code/AIRA/histogate/video"
DIA = f"{ROOT}/diagrams"; FRM = f"{ROOT}/frames"; SCE = f"{ROOT}/scenes"
FPS = 30
# name: (lead_in seconds, tail_hold seconds)
SCENES = {
    "s3_failure": (0.5, 4.0),
    "s5_pipeline": (0.5, 4.0),
    "s6_trust": (0.5, 4.0),
    "s9_numbers": (0.5, 4.5),
    "s10_outro": (0.5, 3.0),
}

async def render(tab, name, lead, tail):
    await cdp.goto(tab, f"file://{DIA}/{name}.html")
    await asyncio.sleep(2.5)
    # wait for gsap timeline + fonts
    for _ in range(40):
        ok = await cdp.js(tab, "!!(window.tl && window.tl.duration && window.tl.duration()>0) && document.fonts.status==='loaded'")
        if ok: break
        await asyncio.sleep(0.5)
    dur = await cdp.js(tab, "window.tl.duration()+0.01")
    print(f"{name}: tl duration {dur:.2f}s", flush=True)
    frames = int(round((lead + dur + tail) * FPS))
    outdir = f"{FRM}/{name}"; os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir): os.remove(os.path.join(outdir, f))
    for i in range(frames):
        t = (i / FPS) - lead
        p = 0.0 if t <= 0 else (1.0 if t >= dur else t / dur)
        await cdp.js(tab, f"(()=>{{window.tl.progress({p:.6f})}})()")
        r = await tab.send("Page.captureScreenshot", format="png")
        with open(f"{outdir}/f{i:04d}.png", "wb") as fh: fh.write(base64.b64decode(r["data"]))
        if i % 150 == 0: print(f"  frame {i}/{frames}", flush=True)
    mp4 = f"{SCE}/{name.replace('_failure','').replace('_pipeline','').replace('_trust','').replace('_numbers','').replace('_outro','')}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(FPS), "-i", f"{outdir}/f%04d.png",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", mp4], check=True)
    print(f"{name}: -> {mp4} ({frames/FPS:.2f}s)", flush=True)

async def main():
    which = sys.argv[1:] or list(SCENES)
    tab = await cdp.new_tab("about:blank")
    await cdp.activate(tab)
    for name in which:
        if name not in SCENES: continue
        lead, tail = SCENES[name]
        await render(tab, name, lead, tail)
    print("DIAGRAMS DONE", flush=True)

asyncio.run(main())
