"""C3: the --space composition serves FastAPI routes AND MCP (tools/list,
initialize) AND the portal custom routes from ONE ASGI app on one port —
the HF Space unit. Wire-true: uvicorn on an ephemeral port, real HTTP."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from histogate.serve.engine import LockedEngine
from histogate.serve.mcp import build_mcp
from tests.test_mcp_http import _StubAnalyzer


def _space_app(tmp_path: Path):
    from contextlib import asynccontextmanager

    from fastmcp.utilities.lifespan import combine_lifespans

    from histogate.serve.api import create_app

    stub = _StubAnalyzer()
    api = create_app(engine=LockedEngine(analyze=None),  # type: ignore[arg-type]
                     slides_root=tmp_path, patch_analyzer=stub)
    mcp = build_mcp(analyze=stub.analyze,  # type: ignore[arg-type]
                    patch_analyzer=stub, uploads_root=tmp_path / "up")
    mcp_app = mcp.http_app(path="/")

    @asynccontextmanager
    async def _noop(app):
        yield

    api.router.lifespan_context = combine_lifespans(_noop, mcp_app.lifespan)
    api.mount("/mcp", mcp_app)
    return api


@pytest.fixture
def space_url(tmp_path: Path):
    import uvicorn

    app = _space_app(tmp_path)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    def _run():
        server.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]  # type: ignore[index]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def test_one_port_serves_api_mcp_and_portal(space_url: str) -> None:
    # 1. FastAPI side on the same port
    r = httpx.get(f"{space_url}/health", timeout=10)
    assert r.status_code == 200, r.text
    # 2. MCP wire handshake + tools through the /mcp mount
    import anyio

    async def _mcp() -> None:
        async with Client(f"{space_url}/mcp") as c:
            names = [t.name for t in await c.list_tools()]
            assert "histogate_start" in names and "model_quality" in names

    anyio.run(_mcp)
    # 3. portal custom routes forward through the mount (unknown session -> 404,
    #    but that 404 comes from OUR portal handler — routing works)
    r = httpx.get(f"{space_url}/mcp/portal/does-not-exist", timeout=10)
    assert r.status_code == 404
    assert "unknown session" in r.text.lower()
