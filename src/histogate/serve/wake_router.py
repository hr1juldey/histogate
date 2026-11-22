"""HistoGate wake router — the always-light steward gateway (P-deploy).

Boots the heavy backend ON DEMAND and idles it down. Runs permanently on
:8788 (~40 MB RAM, no model); ngrok's static domain points HERE.

  GET /wake            -> boots the heavy backend (tmux hg:api) if down,
                          polls /health, returns {"status":"up","took_s":N}
                          or {"status":"booting"} while warming.
  ANY  /*              -> proxied to :8000 when warm; 503 + Retry-After with
                          {"status":"booting","eta_s":150} when cold (and the
                          wake is triggered automatically).
  Idle shutdown        -> no proxied request for IDLE_S -> kill hg:api.

Frontend contract: on 503 booting, poll /wake every 10 s, then retry.
Steward (Hermes) contract: `tmux new-session -d -s wake 'uv run uvicorn
histogate.serve.wake_router:app --port 8788'` after reboot; nothing else.
"""
from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

BACKEND = "http://127.0.0.1:8000"
IDLE_S = 20 * 60          # shut the heavy box down after 20 min of silence
BOOT_CMD = ("tmux has-session -t hg 2>/dev/null || true; "
            "tmux send-keys -t hg:api C-c 2>/dev/null; sleep 1; "
            "tmux send-keys -t hg:api 'uv run histogate-serve --api' Enter")
WORKDIR = "/home/riju279/Documents/code/AIRA/histogate"

app = FastAPI(title="wake-router")
_state = {"last_used": time.time(), "booting": False}


async def _backend_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{BACKEND}/health")
            return r.status_code == 200
    except Exception:
        return False


async def _boot() -> None:
    if _state["booting"]:
        return
    _state["booting"] = True
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-lc", BOOT_CMD, cwd=WORKDIR,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        deadline = time.time() + 300
        while time.time() < deadline:
            if await _backend_up():
                break
            await asyncio.sleep(5)
    finally:
        _state["booting"] = False


@app.get("/wake")
async def wake() -> JSONResponse:
    if await _backend_up():
        return JSONResponse({"status": "up", "took_s": 0})
    asyncio.create_task(_boot())
    return JSONResponse({"status": "booting", "eta_s": 150}, status_code=202)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "DELETE", "PUT", "OPTIONS", "HEAD", "PATCH"])
async def proxy(path: str, request: Request) -> Response:
    if not await _backend_up():
        asyncio.create_task(_boot())
        return JSONResponse(
            {"status": "booting", "eta_s": 150, "hint": "GET /wake to poll"},
            status_code=503, headers={"Retry-After": "10"})
    _state["last_used"] = time.time()
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.request(
            request.method, f"{BACKEND}/{path}",
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "content-length")},
            content=await request.body(),
            params=dict(request.query_params))
    # Pass the upstream payload through UNTOUCHED — JSON verbatim, binary
    # (PNG tiles, overview, heatmaps) as raw bytes with its content-type.
    # The earlier JSON-only shim turned every image into {"_bytes": N},
    # which the frontend fetched fine but could never render.
    return Response(
        status_code=r.status_code, content=r.content,
        media_type=r.headers.get("content-type", "application/octet-stream"),
        headers={"x-routed": "wake-router"})


@app.on_event("startup")
async def _idle_watch() -> None:
    async def _watch() -> None:
        while True:
            await asyncio.sleep(60)
            if (time.time() - _state["last_used"] > IDLE_S
                    and await _backend_up()
                    and not _state["booting"]):
                p = await asyncio.create_subprocess_exec(
                    "bash", "-lc",
                    "tmux send-keys -t hg:api C-c; "
                    "tmux send-keys -t hg:api 'echo idle-stop' Enter",
                    cwd=WORKDIR)
                await p.wait()
                _state["last_used"] = time.time()
    asyncio.create_task(_watch())
