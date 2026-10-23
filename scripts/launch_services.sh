#!/usr/bin/env bash
# Start the shared Pi0.5 VLA and SAM3 services once; every sweep reuses them.
# Molmo starts too when MOLMO_CHECKPOINT_PATH is set (Flash sweeps). It uses
# its own interpreter: transformers>=4.57 conflicts with the LIBERO pin.
#
#   bash scripts/launch_services.sh          # start (no-op if already healthy)
#   bash scripts/launch_services.sh status   # health check only
#
# Services listen on 127.0.0.1:$VLA_PORT, :$SAM3_PORT, and :$MOLMO_PORT
# (env/.env). Sweeps put an rsched proxy in front of each, so never point
# rpent at these ports directly during measurements. Logs and PIDs:
# results/services/.
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
      if dmesg 2>/dev/null | tail -n 50 | grep -qi "out of memory\|oom-kill\|killed process"; then
        log "kernel OOM killer stopped $name: host RAM ran out while loading (not GPU memory)."
        log "Raise WSL2 memory/swap (README, 'WSL2') or stop other programs, then retry."
      fi
      die "$name exited during startup (see results/services/$name.log)"
    fi
    (( waited >= timeout )) && die "$name not ready after ${timeout}s"
  done
  log "$name ready on :$port"
}

if [[ "${1:-}" == "status" ]]; then
  healthz "$VLA_PORT" && log "vla  :$VLA_PORT ok" || log "vla  :$VLA_PORT DOWN"
  healthz "$SAM3_PORT" && log "sam3 :$SAM3_PORT ok" || log "sam3 :$SAM3_PORT DOWN"
  if [[ -n "${MOLMO_CHECKPOINT_PATH:-}" ]]; then
    healthz "$MOLMO_PORT" && log "molmo :$MOLMO_PORT ok" || log "molmo :$MOLMO_PORT DOWN"
  else
    log "molmo skipped (MOLMO_CHECKPOINT_PATH unset)"
  fi
  exit 0
fi

[[ -n "${PI05_CHECKPOINT_PATH:-}" && -e "$PI05_CHECKPOINT_PATH" ]] || die "PI05_CHECKPOINT_PATH not set or missing (env/.env)"
[[ -n "${SAM3_CHECKPOINT_PATH:-}" && -e "$SAM3_CHECKPOINT_PATH" ]] || die "SAM3_CHECKPOINT_PATH not set or missing (env/.env)"

mem_report() {
  free -g | awk '/^Mem:/ {printf "[rsched] host RAM: %s GB total, %s GB available\n", $2, $7}
                 /^Swap:/ {printf "[rsched] swap:     %s GB total\n", $2}'
}
mem_report
total_gb=$(free -g | awk '/^Mem:/ {print $2}')
swap_gb=$(free -g | awk '/^Swap:/ {print $2}')
if (( total_gb + swap_gb < 24 )); then
  log "WARNING: <24 GB RAM+swap. Pi0.5 is built in fp32 on the CPU before moving to"
  log "         the GPU (~13 GB peak); the kernel may kill it. On WSL2, raise memory/swap"
  log "         in %UserProfile%\\.wslconfig (see README, 'WSL2')."
fi

# Start one at a time: both models load weights through host RAM, and loading
# them together roughly doubles the peak (WSL2 default = half of Windows RAM).
start vla "$VLA_PORT" python -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --cuda-device "$CUDA_DEVICE" \
  --transport http --host 127.0.0.1 --port "$VLA_PORT"
wait_ready vla "$VLA_PORT" 900

start sam3 "$SAM3_PORT" python -m rpent.robots.components.sam3_server \
  --cuda-device "$CUDA_DEVICE" \
  --transport http --host 127.0.0.1 --port "$SAM3_PORT"
wait_ready sam3 "$SAM3_PORT" 900

if [[ -n "${MOLMO_CHECKPOINT_PATH:-}" ]]; then
  [[ -e "$MOLMO_CHECKPOINT_PATH" ]] || die "MOLMO_CHECKPOINT_PATH not found: $MOLMO_CHECKPOINT_PATH"
  molmo_py="${MOLMO_PYTHON:-$RS_ROOT/.venv-molmo/bin/python}"
  if [[ ! -x "$molmo_py" ]]; then
    die "Molmo interpreter not found ($molmo_py). From the repo root:
  uv venv --python 3.11 .venv-molmo
  uv pip install --python .venv-molmo/bin/python -e 'third_party/RPent[molmo]' einops"
  fi
  # Own process env: the LIBERO venv's transformers cannot import this server.
  start molmo "$MOLMO_PORT" env \
    PYTHONPATH="$RS_ROOT/third_party/RPent" \
    MOLMO_CHECKPOINT_PATH="$MOLMO_CHECKPOINT_PATH" \
    "$molmo_py" -m rpent.robots.components.molmo_server \
    --cuda-device "$CUDA_DEVICE" \
    --transport http --host 127.0.0.1 --port "$MOLMO_PORT"
  wait_ready molmo "$MOLMO_PORT" 900
else
  log "molmo skipped (set MOLMO_CHECKPOINT_PATH for a Flash sweep)"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/[rsched] GPU memory after load: /'
