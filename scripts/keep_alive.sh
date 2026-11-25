#!/usr/bin/env bash
# HistoGate public-demo watchdog: revives wake_router / MCP / ngrok if dead.
# The backend :8000 is NOT restarted here - wake_router spawns it on demand.
set -u
LOG=/tmp/histogate_watchdog.log
ts() { date +"%F %T"; }

# 1. wake_router :8788 (public entry proxy)
if ! curl -sf -o /dev/null -m 5 http://127.0.0.1:8788/site-config.json; then
  if ! pgrep -f "wake_router:app" >/dev/null; then
    echo "$(ts) wake_router dead -> respawning tmux window hg:wake" >>"$LOG"
    tmux new-window -d -t hg -n wake \
      "cd /home/riju279/Documents/code/AIRA/histogate && exec .venv/bin/python -m uvicorn histogate.serve.wake_router:app --port 8788 2>&1 | tee -a /tmp/wake_router.log; exec bash"
  fi
fi

# 2. MCP :8790
if ! curl -sf -o /dev/null -m 5 http://127.0.0.1:8790/; then
  if ! pgrep -f "histogate-serve --mcp" >/dev/null; then
    echo "$(ts) mcp dead -> respawning tmux window hg:mcp" >>"$LOG"
    tmux respawn-window -k -t hg:mcp \
      "cd /home/riju279/Documents/code/AIRA/histogate && .venv/bin/histogate-serve --mcp 2>&1 | tee -a /tmp/mcp.log"
  fi
fi

# 3. ngrok systemd service (public URL)
if ! curl -sf -o /dev/null -m 8 https://gobbler-calm-evenly.ngrok-free.app/site-config.json; then
  if ! systemctl --user is-active --quiet histogate-ngrok.service; then
    echo "$(ts) public URL down -> starting histogate-ngrok.service" >>"$LOG"
    systemctl --user start histogate-ngrok.service
  fi
fi
