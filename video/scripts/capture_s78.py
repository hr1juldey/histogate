"""S7 (mobile 390x844, touch pinch) then S8 (desktop slide) — robust version."""
import asyncio, subprocess, sys, time, json
sys.path.insert(0, "/home/riju279/Documents/code/AIRA/histogate/video/scripts")
import cdp

BASE = "https://histogate.vercel.app"
OUT = "/home/riju279/Documents/code/AIRA/histogate/video/scenes"

def xdotool(*args):
    subprocess.run(["env", "DISPLAY=:2", "xdotool", *args], check=False)

def win_size(w, h):
    xdotool("search", "--onlyvisible", "--class", ".", "windowsize", str(w), str(h))
    xdotool("search", "--onlyvisible", "--class", ".", "windowmove", "0", "0")

async def wheel(tab, x, y, dy, notches, pause=0.15):
    for _ in range(notches):
        await tab.send("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y,
                       deltaX=0, deltaY=dy)
        await asyncio.sleep(pause)

async def wait_map(tab, timeout=45):
    for _ in range(timeout * 2):
        if await cdp.js(tab, "!!document.querySelector('.vcell')"): return True
        await asyncio.sleep(0.5)
    return False

def start_rec(out, size, grab=":2"):
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "x11grab", "-video_size", size, "-framerate", "30",
         "-draw_mouse", "0", "-i", grab,
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)

async def stop(rec):
    try:
        rec.stdin.write(b"q"); rec.stdin.flush()
    except Exception: pass
    for _ in range(80):
        if rec.poll() is not None: return
        await asyncio.sleep(0.25)
    rec.kill()

async def s7(tab):
    win_size(390, 844)
    await asyncio.sleep(0.8)
    await tab.send("Page.navigate", url=f"{BASE}/slide")
    await asyncio.sleep(4)
    print("mobile map ready:", await wait_map(tab), flush=True)
    await asyncio.sleep(2.0)
    # bring the viewer into the (small) viewport
    await cdp.js(tab, "document.querySelector('.mosaicwrap')?.scrollIntoView({block:'center',behavior:'smooth'})")
    await asyncio.sleep(1.5)
    print("viewport:", await cdp.js(tab, "JSON.stringify({w:innerWidth,h:innerHeight})"), flush=True)
    try:
        await tab.send("Input.setTouchEmulationEnabled", enabled=True, maxPoints=3)
    except Exception as e:
        print("touch emulation opt:", e, flush=True)
    rec = start_rec(f"{OUT}/s7.mp4", "390x844", ":2+0,0")
    t0 = time.time()
    try:
        await asyncio.sleep(1.5)
        await cdp.pinch(tab, 195, 420, 2.6, steps=40)
        await asyncio.sleep(2.0)
        p = await cdp.js(tab, """(()=>{const v=[...document.querySelectorAll('.vcell')].find(e=>{const r=e.getBoundingClientRect();return r.width>25&&r.left>=0&&r.right<=390&&r.top>150&&r.bottom<800});if(!v)return null;const r=v.getBoundingClientRect();return JSON.stringify({x:Math.round(r.left+r.width/2),y:Math.round(r.top+r.height/2)})})()""")
        print("tile at", p, flush=True)
        if p:
            c = json.loads(p)
            await tab.send("Input.dispatchMouseEvent", type="mousePressed", x=c["x"], y=c["y"], button="left", clickCount=1)
            await tab.send("Input.dispatchMouseEvent", type="mouseReleased", x=c["x"], y=c["y"], button="left", clickCount=1)
            await asyncio.sleep(1.5)
            await cdp.js(tab, "document.querySelector('.sidecol')?.scrollIntoView({block:'center',behavior:'smooth'})")
        await asyncio.sleep(4.5)
    finally:
        print(f"s7 elapsed {time.time()-t0:.1f}s", flush=True)
        await stop(rec)
    win_size(1920, 1080)
    await asyncio.sleep(1)

async def s8(tab):
    await tab.send("Page.navigate", url=f"{BASE}/slide")
    await asyncio.sleep(5)
    print("desktop map ready:", await wait_map(tab), flush=True)
    await asyncio.sleep(2.5)
    rec = start_rec(f"{OUT}/s8.mp4", "1920x1080")
    t0 = time.time()
    try:
        await asyncio.sleep(2.5)
        await wheel(tab, 800, 560, -280, 6, pause=0.18)
        await asyncio.sleep(1.5)
        await cdp.js(tab, "[...document.querySelectorAll('.vcell.dec')][0]?.click()")
        await asyncio.sleep(3.0)
        await cdp.js(tab, "document.querySelector('.guide')?.click()")
        await asyncio.sleep(3.5)
    finally:
        print(f"s8 elapsed {time.time()-t0:.1f}s", flush=True)
        await stop(rec)

async def main():
    tab = await cdp.new_tab(f"{BASE}/slide")
    await cdp.goto(tab, f"{BASE}/slide")
    which = sys.argv[1] if len(sys.argv) > 1 else "s7,s8"
    try:
        if "s7" in which: await s7(tab)
        if "s8" in which: await s8(tab)
    finally:
        win_size(1920, 1080)
    print("ALL DONE", flush=True)

asyncio.run(main())
