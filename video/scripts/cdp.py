"""Minimal CDP client for driving Chromium on a local display (video capture rig)."""
import asyncio, base64, json, subprocess, sys, urllib.request
import websockets

CDP_PORT = 9333

class Tab:
    def __init__(self, ws, target_id):
        self.ws = ws; self.id = target_id; self.mid = 0; self.pending = {}
    async def send(self, method, timeout=10.0, **params):
        self.mid += 1
        mid = self.mid
        async def _call():
            await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                msg = json.loads(await self.ws.recv())
                if msg.get("id") == mid:
                    if "error" in msg: raise RuntimeError(f"{method}: {msg['error']}")
                    return msg.get("result", {})
        try:
            return await asyncio.wait_for(_call(), timeout=timeout)
        except TimeoutError:
            raise RuntimeError(f"CDP timeout: {method}")

def http_json(path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{CDP_PORT}{path}", method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

async def new_tab(url="about:blank"):
    t = http_json(f"/json/new?{url}", method="PUT")
    ws = await websockets.connect(t["webSocketDebuggerUrl"], max_size=200*1024*1024)
    tab = Tab(ws, t["id"])
    await tab.send("Page.enable"); await tab.send("Runtime.enable")
    return tab

async def activate(tab):
    try: http_json(f"/json/activate/{tab.id}")
    except Exception: pass

async def goto(tab, url, wait_s=1.0):
    await tab.send("Page.navigate", url=url)
    await asyncio.sleep(wait_s)
    # wait for load state
    for _ in range(60):
        st = await tab.send("Runtime.evaluate", expression="document.readyState", returnByValue=True)
        if st.get("result", {}).get("value") == "complete": break
        await asyncio.sleep(0.5)
    try:
        await tab.send("Runtime.evaluate", expression="document.fonts.ready.then(()=>1)",
                       awaitPromise=True, returnByValue=True)
    except Exception: pass

async def js(tab, expr, await_promise=False):
    r = await tab.send("Runtime.evaluate", expression=expr, returnByValue=True,
                       awaitPromise=await_promise)
    if r.get("exceptionDetails"): raise RuntimeError(r["exceptionDetails"]["exception"]["description"][:800])
    return r.get("result", {}).get("value")

async def screenshot(tab, path):
    r = await tab.send("Page.captureScreenshot", format="png")
    with open(path, "wb") as f: f.write(base64.b64decode(r["data"]))

async def key(tab, code, raw_key="", key_name="", ty="keyDown", mods=0):
    await tab.send("Input.dispatchKeyEvent", type=ty, key=raw_key or key_name,
                   code=code, windowsVirtualKeyCode=0)

async def pinch(tab, x, y, scale_factor, steps=30, speed=250):
    """Two-finger pinch via raw touch events (synthesizePinchGesture hangs some builds)."""
    def pt(px, py, i=1):
        return {"x": px, "y": py, "id": i, "radiusX": 2.0, "radiusY": 2.0,
                "rotationAngle": 0.0, "force": 1.0}
    d = 60.0
    await tab.send("Input.dispatchTouchEvent", type="touchStart",
                   touchPoints=[pt(x - d, y, 1), pt(x + d, y, 2)])
    for i in range(1, steps + 1):
        dd = d * (1 + (scale_factor - 1) * i / steps)
        await tab.send("Input.dispatchTouchEvent", type="touchMove",
                       touchPoints=[pt(x - dd, y - (dd - d) * 0.3, 1), pt(x + dd, y + (dd - d) * 0.3, 2)])
        await asyncio.sleep(0.033)
    await tab.send("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])

async def tap(tab, x, y):
    await tab.send("Input.dispatchTouchEvent", type="touchStart",
                   touchPoints=[{"x": x, "y": y, "id": 1, "radiusX": 2.0, "radiusY": 2.0, "force": 1.0}])
    await asyncio.sleep(0.08)
    await tab.send("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])

async def smooth_scroll(tab, x, y, dy_total, steps=20, step_pause=0.05):
    for i in range(1, steps+1):
        await tab.send("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y,
                       deltaX=0, deltaY=dy_total/steps)
        await asyncio.sleep(step_pause)

async def drag_range(tab, selector, v0, v1, seconds=6.0, fps=30, change_evt="input"):
    """Drag an <input type=range> from v0 to v1 via native value setter + events."""
    set_v = f"""((v)=>{{
      const el = document.querySelector({json.dumps(selector)}); if(!el) return false;
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', {{bubbles:true}}));
      el.dispatchEvent(new Event('change', {{bubbles:true}}));
      return true;
    }})(arguments[0])"""
    frames = int(seconds*fps)
    for i in range(frames+1):
        t = i/frames
        v = v0 + (v1-v0)*t
        await tab.send("Runtime.evaluate",
                       expression=f"((v)=>{{const el=document.querySelector({json.dumps(selector)});if(!el)return;Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(el,String(v));el.dispatchEvent(new Event('input',{{bubbles:true}}); )}})({v})"
                       if False else set_v.replace("arguments[0]", str(v)), returnByValue=True)
        await asyncio.sleep(1/fps)

def ffmpeg_recorder(out, display=":2", size="1920x1080", fps=30, crf=18, extra=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "x11grab", "-video_size", size, "-framerate", str(fps),
           "-draw_mouse", "0", "-i", display,
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(out)]
    if extra: cmd = cmd[:-1] + extra + [str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
