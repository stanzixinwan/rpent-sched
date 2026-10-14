#!/usr/bin/env bash
# Start the shared Pi0.5 VLA and SAM3 services once; every sweep reuses them.
#
#   bash scripts/launch_services.sh          # start both (no-op if already healthy)
#   bash scripts/launch_services.sh status   # health check only
#
# Services listen on 127.0.0.1:$VLA_PORT and :$SAM3_PORT (env/.env). Sweeps put
# an rsched proxy in front of each, so never point rpent at these ports directly
# during measurements. Logs and PIDs: results/services/.
set -euo pipefail
source "$(dirname "$0")/common.sh"

healthz() {  # $1 = port
  python - "$1" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
body = json.dumps({"method": "healthz", "args": [], "kwargs": {}, "session_id": None}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/call", data=body,
                             headers={"Content-Type": "application/json"})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(req, timeout=3) as r:
        sys.exit(0 if json.loads(r.read()).get("ok") else 1)
except Exception:
    sys.exit(1)
PY
}

start() {  # $1 = name, $2 = port, rest = command
  local name=$1 port=$2; shift 2
  if healthz "$port"; then
    log "$name already healthy on :$port"
    return
  fi
  log "starting $name on :$port (log: results/services/$name.log)"
  nohup "$@" >"$SERVICE_DIR/$name.log" 2>&1 &
  echo $! >"$SERVICE_DIR/$name.pid"
}

wait_ready() {  # $1 = name, $2 = port, $3 = timeout seconds
  local name=$1 port=$2 timeout=$3 waited=0
  until healthz "$port"; do
    sleep 5; waited=$((waited + 5))
    if [[ -f "$SERVICE_DIR/$name.pid" ]] && ! kill -0 "$(cat "$SERVICE_DIR/$name.pid")" 2>/dev/null; then
      tail -n 30 "$SERVICE_DIR/$name.log" >&2
      die "$name exited during startup (see results/services/$name.log)"
    fi
    (( waited >= timeout )) && die "$name not ready after ${timeout}s"
  done
  log "$name ready on :$port"
}

if [[ "${1:-}" == "status" ]]; then
  healthz "$VLA_PORT" && log "vla  :$VLA_PORT ok" || log "vla  :$VLA_PORT DOWN"
  healthz "$SAM3_PORT" && log "sam3 :$SAM3_PORT ok" || log "sam3 :$SAM3_PORT DOWN"
  exit 0
fi

[[ -n "${PI05_CHECKPOINT_PATH:-}" && -e "$PI05_CHECKPOINT_PATH" ]] || die "PI05_CHECKPOINT_PATH not set or missing (env/.env)"
[[ -n "${SAM3_CHECKPOINT_PATH:-}" && -e "$SAM3_CHECKPOINT_PATH" ]] || die "SAM3_CHECKPOINT_PATH not set or missing (env/.env)"

start vla "$VLA_PORT" python -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --cuda-device "$CUDA_DEVICE" \
  --transport http --host 127.0.0.1 --port "$VLA_PORT"
start sam3 "$SAM3_PORT" python -m rpent.robots.components.sam3_server \
  --cuda-device "$CUDA_DEVICE" \
  --transport http --host 127.0.0.1 --port "$SAM3_PORT"

wait_ready vla "$VLA_PORT" 900
wait_ready sam3 "$SAM3_PORT" 900
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/[rsched] GPU memory after load: /'
