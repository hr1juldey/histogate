"""Screen-scene capture: drives the live site via CDP on Xvfb :2 while
ffmpeg x11grab records 1920x1080@30. One mp4 per scene in video/scenes/."""
import asyncio, subprocess, sys, time
sys.path.insert(0, "/home/riju279/Documents/code/AIRA/histogate/video/scripts")
import cdp

BASE = "https://histogate.vercel.app"
OUT = "/home/riju279/Documents/code/AIRA/histogate/video/scenes"

def record(name, seconds):
    """Return an ffmpeg recorder started fresh; caller stops it via 'q'."""
    return cdp.ffmpeg_recorder(f"{OUT}/{name}.mp4")

async def stop(rec):
    rec.stdin.write(b"q"); rec.stdin.flush()
    for _ in range(80):
        if rec.poll() is not None: return
        await asyncio.sleep(0.25)
    rec.kill()

async def wheel(tab, x, y, dy, notches, pause=0.12):
    for _ in range(notches):
        await tab.send("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y,
                       deltaX=0, deltaY=dy)
        await asyncio.sleep(pause)

async def smooth_page_scroll(tab, to_px, seconds):
    steps = int(seconds * 30)
    start = await cdp.js(tab, "window.scrollY")
    for i in range(1, steps + 1):
        v = start + (to_px - start) * (i / steps)
        await cdp.js(tab, f"window.scrollTo(0,{v:.0f})")
        await asyncio.sleep(seconds / steps)

async def scene_s1(tab, rec):
    """Homepage hero + scroll to counters."""
    await cdp.activate(tab); await asyncio.sleep(1)
    t0 = time.time()
    await asyncio.sleep(3.0)                                   # hero hold
    await smooth_page_scroll(tab, 480, 3.5)                    # reveal counters
    await asyncio.sleep(2.2)
    await smooth_page_scroll(tab, 300, 1.5)                    # settle mid
    await asyncio.sleep(2.0)
    print(f"s1 elapsed {time.time()-t0:.1f}s")

async def scene_s2(tab, rec):
    """/demo: tumor verdict, zoom, then DECLINED lymph sample."""
    await cdp.activate(tab); await asyncio.sleep(1)
    # wait for auto-picked tumor verdict to be visible
    for _ in range(40):
        v = await cdp.js(tab, "document.querySelector('.inspector')?.innerText.slice(0,80)")
        if v and "verdict" not in v.lower() and len(v) > 20 and "pick a sample" not in v: break
        await asyncio.sleep(0.5)
    await asyncio.sleep(1.5)
    t0 = time.time()
    await asyncio.sleep(3.5)                                   # tumor verdict hold
    box = await cdp.js(tab, "JSON.stringify(document.querySelector('.tilebox, .pane.mosaicwrap, .panes .pane')?.getBoundingClientRect())")
    print("viewer box", box)
    await wheel(tab, 900, 500, -320, 4, pause=0.15)            # zoom in viewer
    await asyncio.sleep(1.5)
    await wheel(tab, 900, 500, 240, 3, pause=0.15)             # back out
    await asyncio.sleep(1.0)
    await cdp.js(tab, "[...document.querySelectorAll('.tile')].find(b=>b.textContent.includes('LYMPH'))?.click()")
    await asyncio.sleep(4.5)                                   # declined verdict
    print(f"s2 elapsed {time.time()-t0:.1f}s")

async def scene_s4(tab, rec):
    """/story: stain dial hero — drag dial, first model flips wrong."""
    await cdp.activate(tab); await asyncio.sleep(1)
    await cdp.js(tab, "document.querySelector('.dial-sec')?.scrollIntoView({block:'center'})")
    await asyncio.sleep(3.0)   # dial canvases/models settle
    t0 = time.time()
    await asyncio.sleep(2.0)
    await cdp.drag_range(tab, "input[type=range]", 0, 85, seconds=6.0)
    await asyncio.sleep(2.5)                                   # first model wrong/declined
    await cdp.drag_range(tab, "input[type=range]", 85, 45, seconds=2.5)
    await asyncio.sleep(2.0)
    print(f"s4 elapsed {time.time()-t0:.1f}s")

async def scene_s7(tab, rec):
    """Mobile 390x844: /slide viewer with pinch zoom."""
    await tab.send("Emulation.setDeviceMetricsOverride", width=390, height=844,
                   deviceScaleFactor=1, mobile=True)
    await cdp.goto(tab, f"{BASE}/slide")   # reload under mobile metrics -> mobile overview asset
    await cdp.activate(tab); await asyncio.sleep(1)
    for _ in range(60):
        ok = await cdp.js(tab, "!!document.querySelector('.vcell')")
        if ok: break
        await asyncio.sleep(0.5)
    await asyncio.sleep(2.5)   # overview image settle
    t0 = time.time()
    await asyncio.sleep(2.0)
    await cdp.pinch(tab, 195, 380, 2.6, steps=40, speed=200)   # pinch zoom
    await asyncio.sleep(2.0)
    # tap a visible tile -> verdict inspector (mobile layout)
    p = await cdp.js(tab, """(()=>{const v=[...document.querySelectorAll('.vcell')].find(e=>{const r=e.getBoundingClientRect();return r.width>30&&r.left<390&&r.right>0&&r.top<760&&r.bottom>60});if(!v)return null;const r=v.getBoundingClientRect();return JSON.stringify({x:r.left+r.width/2,y:r.top+r.height/2})})()""")
    print("tile at", p)
    if p:
        import json as _j; c = _j.loads(p)
        await tab.send("Input.dispatchMouseEvent", type="mousePressed", x=c["x"], y=c["y"], button="left", clickCount=1)
        await tab.send("Input.dispatchMouseEvent", type="mouseReleased", x=c["x"], y=c["y"], button="left", clickCount=1)
    await asyncio.sleep(4.0)
    print(f"s7 elapsed {time.time()-t0:.1f}s")
    await tab.send("Emulation.clearDeviceMetricsOverride")

async def scene_s8(tab, rec):
    """/slide: foreign-organ map, zoom to real tiles, click verdict."""
    await cdp.activate(tab); await asyncio.sleep(1)
    for _ in range(60):
        ok = await cdp.js(tab, "!!document.querySelector('.vcell')")
        if ok: break
        await asyncio.sleep(0.5)
    await asyncio.sleep(2.5)   # overview settle
    t0 = time.time()
    await asyncio.sleep(2.5)                                   # sumband + map hold
    await wheel(tab, 800, 560, -280, 6, pause=0.18)            # zoom: squares become tiles
    await asyncio.sleep(1.5)
    await cdp.js(tab, "[...document.querySelectorAll('.vcell.dec')][0]?.click()")  # declined tile
    await asyncio.sleep(3.0)
    await cdp.js(tab, "document.querySelector('.guide')?.click()")  # TUM guide verdict
    await asyncio.sleep(3.5)
    print(f"s8 elapsed {time.time()-t0:.1f}s")

async def main(which):
    # fresh tabs, pre-warm every page so nothing cold-loads on camera
    tabs = {}
    for key, url in [("s1", f"{BASE}/"), ("s2", f"{BASE}/demo"),
                     ("s4", f"{BASE}/story"), ("s7", f"{BASE}/slide"), ("s8", f"{BASE}/slide")]:
        tabs[key] = await cdp.new_tab(url)
        await cdp.goto(tabs[key], url)
    await asyncio.sleep(4)
    scenes = {"s1": scene_s1, "s2": scene_s2, "s4": scene_s4, "s7": scene_s7, "s8": scene_s8}
    if which:
        order = [w for w in which.split(",") if w in scenes]
    else:
        order = ["s1", "s2", "s4", "s7", "s8"]
    for name in order:
        print(f"=== recording {name} ===", flush=True)
        rec = record(name, 0)
        await asyncio.sleep(0.8)
        await scenes[name](tabs[name], rec)
        await stop(rec)
        print(f"=== {name} done ===", flush=True)
        await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else ""))
