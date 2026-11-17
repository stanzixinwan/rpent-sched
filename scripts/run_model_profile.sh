#!/usr/bin/env bash
# Single-model characterization (paper section 4): RPC/host/GPU split, per-stage time,
# prefill vs decode, operator mix, analytic roofline, energy, Pi0.5 batch scaling.
# Runs on any GPU machine, no root needed. ~45-60 min on an A100.
#
#   bash scripts/run_model_profile.sh                  # results/model_profile/<stamp>/
#   bash scripts/run_model_profile.sh <stamp>          # resume: skips the recording and finished services
#
# 1. record: 16 real episodes through proxies that save request bodies
# 2. for each of vla, sam3, molmo: stop everything, start that model alone under
#    rsched.profiled_server, replay the recorded requests with rsched.bench_models
# 3. python analysis/model_profile.py <root>
#
# Overrides: MP_SERVICES ("vla sam3 molmo"), MP_CALLS (100), MP_BATCHES ("1 2 4 8 16"),
# MP_PORT_BASE (9100). Leaves the normal services stopped; restart them with
# scripts/launch_services.sh. Needs nvidia-ml-py in the main venv for energy.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/model_profile/$STAMP"
read -r -a SERVICES <<<"${MP_SERVICES:-vla sam3 molmo}"
CALLS="${MP_CALLS:-100}"
read -r -a BATCHES <<<"${MP_BATCHES:-1 2 4 8 16}"
PORT_BASE="${MP_PORT_BASE:-9100}"
mkdir -p "$ROOT"
python -c "import pynvml" 2>/dev/null || { log "installing nvidia-ml-py for energy readings"; uv pip install -q nvidia-ml-py; }

healthy() {  # $1 = port
  python - "$1" <<'PY'
import json, sys, urllib.request
body = json.dumps({"method": "healthz", "args": [], "kwargs": {}, "session_id": None}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/call", data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=3) as r:
        sys.exit(0 if json.loads(r.read()).get("ok") else 1)
except Exception:
    sys.exit(1)
PY
}

# 1. record requests from real episodes
REC="$ROOT/record"
if [[ ! -f "$REC/N4_r0/manifest.json" ]]; then
  bash scripts/mps.sh stop >/dev/null 2>&1 || true
  bash scripts/launch_services.sh
  log "recording requests: 16 episodes at N=4"
  python -m rsched.sweep configs/record_requests.yaml --out "$REC"
fi
REQ="$REC/N4_r0/requests"
for d in vla sam3 molmo; do
  n=$(find "$REQ/$d" -name '*.json' 2>/dev/null | wc -l)
  log "recorded $d requests: $n"
done
bash scripts/record_platform.sh "$ROOT"
bash scripts/stop_services.sh

# 2. one model at a time, alone on the GPU
i=0
for svc in "${SERVICES[@]}"; do
  port=$(( PORT_BASE + i )); i=$(( i + 1 ))
  out="$ROOT/$svc"
  if [[ -f "$out/client.csv" ]]; then log "skip $svc (done)"; continue; fi
  rm -rf "$out"; mkdir -p "$out"
  prof=(-m rsched.profiled_server --service "$svc" --log "$out/server.jsonl" --)
  case $svc in
    vla)   cmd=(python "${prof[@]}" rpent.robots.components.pi05_vla_server --embodiment libero
                --cuda-device "$CUDA_DEVICE" --transport http --host 127.0.0.1 --port "$port") ;;
    sam3)  cmd=(python "${prof[@]}" rpent.robots.components.sam3_server
                --cuda-device "$CUDA_DEVICE" --transport http --host 127.0.0.1 --port "$port") ;;
    molmo) cmd=(env PYTHONPATH="$RS_ROOT:$RS_ROOT/third_party/RPent" MOLMO_CHECKPOINT_PATH="$MOLMO_CHECKPOINT_PATH"
                "${MOLMO_PYTHON:-$RS_ROOT/.venv-molmo/bin/python}" "${prof[@]}" rsched.molmo_server
                --cuda-device "$CUDA_DEVICE" --transport http --host 127.0.0.1 --port "$port") ;;
    *) die "unknown service $svc" ;;
  esac
  log "$svc: starting profiled server on :$port"
  OMP_NUM_THREADS="${SERVICE_THREADS:-4}" MKL_NUM_THREADS="${SERVICE_THREADS:-4}" \
    nohup "${cmd[@]}" >"$out/server.log" 2>&1 &
  pid=$!
  for _ in $(seq 180); do healthy "$port" && break; kill -0 "$pid" 2>/dev/null || { tail -n 30 "$out/server.log"; die "$svc server exited"; }; sleep 5; done
  healthy "$port" || die "$svc server not ready"
  extra=()
  [[ $svc == vla ]] && extra=(--batches "${BATCHES[@]}")
  python -m rsched.bench_models --service "$svc" --url "http://127.0.0.1:$port" --requests "$REQ/$svc" \
    --out "$out" --gpu "$CUDA_DEVICE" --calls "$CALLS" "${extra[@]}" || { kill "$pid"; die "$svc benchmark failed"; }
  kill "$pid"; wait "$pid" 2>/dev/null || true
done

# 3. tables
python analysis/model_profile.py "$ROOT" | tee "$ROOT/summary.txt"
log "done: $ROOT"
log "copy back: tar czf /workspace/model_profile.tgz -C $RS_ROOT results/model_profile/$STAMP && runpodctl send /workspace/model_profile.tgz"
