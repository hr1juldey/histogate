# HistoGate — Distribution & Demo Surfaces (DEPLOYMENT)

Version: 0.1 · Constraint: no servers owned; workstation is the only compute. ngrok-class inbound tunnels REJECTED (wasteful, exposes single-GPU box, PC-uptime-coupled).

## Reviewer-access doctrine (layered by effort)

| Layer | Surface | Hosted where | Effort for reviewer | Status |
|---|---|---|---|---|
| L0 | GitHub repo: README technical report + committed sample report HTML/PNG artifacts + gate report | GitHub | click | always |
| L1 | HF model card + weights (+ small demo dataset card) | HF Hub | click | always |
| L2 | **HF Space (Gradio, free CPU tier) running the int8 ONNX model** | Hugging Face | click + interact | PR-15 companion (`space/` dir) |
| L3 | Hermes-mediated demo: reviewer asks in chat, agent calls FastMCP tool, returns report file | this workstation, agent-gated | conversation | personal/demo |
| L4 | Dedicated narrow Telegram bot (long-polling bridge to local FastAPI) | this workstation | chat | optional, passcode-gated |

## Layer 2 — HF Space (the real answer to "no server")
- VERIFIED pricing (2026-11-20, official docs): CPU Basic compute is FREE (2 vCPU / 16 GB / 50 GB ephemeral, sleeps after 48 h idle); GPUs paid ($0.40+/h); **creating** a Gradio/Docker Space needs PRO (~$9/mo) EXCEPT **ZeroGPU Spaces — free accounts get 2**, dynamic GPU slices with daily quota. → CHOICE: **ZeroGPU Space** (free, GPU slice per request) running the fp16/int8 ONNX path. Fallback: PRO one month ($9) for a CPU-Basic Gradio space if ZeroGPU quotas bite.
- HF access is via local `hf` CLI (logged in: riju279); `hf skills add -g` installs the agent skill for it at build time. NO HF MCP connector exists on this profile (verified 2026-11-20).
- Gradio app wrapping the SAME Pydantic contracts; training NEVER runs here — Space serves only the exported quantized artifact (local 3060 trains).
- Demo scope stays small: patch upload → 9-class prediction + artifact flag + TSR/TIL on bundled demo slide crops; full 512-patch WSI demo runs only on bundled fixture slides (upload of multi-GB WSI is not a demo).
- Free Spaces sleep after inactivity — acceptable; link carries "wake ~30s" note.
- Zero inbound exposure of this machine; zero cost; survives PC being off.

## Layer 3 — Hermes gateway (agent-mediated)
- The FastMCP server (PR-14) is consumed by the local agent as an MCP tool: in-chat `analyze_slide`, `run_regression_gate`, `get_report` — reports returned as attachments.
- HARD RULE: external reviewers NEVER get access to the agent itself. The agent holds memory, files, credentials. Agent-mediated means the OWNER drives, the reviewer watches/receives artifacts. This layer demos the integration story ("AI output reachable by agents") without exposing anything.

## Layer 4 — Dedicated Telegram bridge (only if a live chat surface is demanded)
- A SEPARATE bot token + ~100-line poller (python-telegram-bot, long-polling) that forwards whitelisted commands to localhost FastAPI and returns the report.
- Telegram long-polling is OUTBOUND-only: no inbound ports, no tunnel, no ngrok — this is the trick that makes it free.
- Passcode-gated (bot ignores non-authorized chat ids), one job at a time (single GPU), PC uptime caveat documented. No path from bot → shell/filesystem; the ONLY verb is the analysis API.

## Decision log addendum
- AD-9: Public interactive demo = HF Space (L2), not self-hosted tunnel. Rejected: ngrok/cloudflared (wasteful, uptime-coupled, security surface), exposing Hermes agent to externals (never).
- AD-10: Telegram access, if wanted, is a dedicated narrow bot via long-polling bridge — outbound-only by protocol design.
