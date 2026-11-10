#!/usr/bin/env bash
# Start the local planner (vLLM, OpenAI-compatible) next to the robot services.
#
#   bash scripts/launch_llm.sh          # start (no-op if already healthy)
#   bash scripts/launch_llm.sh status
#
# Start it AFTER scripts/launch_services.sh: vLLM claims LLM_GPU_UTIL of the
# GPU's total memory at startup (weights + KV cache) and needs that much free.
# Pi0.5 + SAM3 + Molmo use ~36 GB of an 80 GB A100, so the default 0.32 (~26 GB)
# fits. Settings (env/.env): LLM_MODEL, LLM_PORT (9003), LLM_GPU_UTIL (0.32),
# LLM_MAX_LEN (16384), LLM_MAX_SEQS (32), VLLM_BIN. Log: results/services/llm.log.
# Under MPS (scripts/mps.sh), LLM_MPS_PCT=30 caps the planner at 30% of the SMs
# (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE); unset = no cap.
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${LLM_PORT:=9003}"
: "${LLM_GPU_UTIL:=0.32}"
: "${LLM_MAX_LEN:=16384}"
: "${LLM_MAX_SEQS:=32}"
: "${LLM_SERVED_NAME:=planner}"
: "${VLLM_BIN:=$RS_ROOT/.venv-vllm/bin/vllm}"
: "${LLM_MODEL:=$RS_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct}"

llm_up() {
  python - "$LLM_PORT" <<'PY'
import sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

if [[ "${1:-}" == "status" ]]; then
  llm_up && log "llm :$LLM_PORT ok" || log "llm :$LLM_PORT DOWN"
  exit 0
fi
if llm_up; then
  log "llm already healthy on :$LLM_PORT"
  exit 0
fi
[[ -x "$VLLM_BIN" ]] || die "vLLM not installed ($VLLM_BIN): bash env/setup_vllm.sh"
[[ -f "$LLM_MODEL/config.json" ]] || die "no model at $LLM_MODEL: bash env/setup_vllm.sh"

log "starting llm ($(basename "$LLM_MODEL")) on :$LLM_PORT, gpu util $LLM_GPU_UTIL (log: results/services/llm.log)"
mps_env=()
[[ -n "${LLM_MPS_PCT:-}" ]] && mps_env=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$LLM_MPS_PCT")
nohup env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS="${SERVICE_THREADS:-4}" "${mps_env[@]}" \
  "$VLLM_BIN" serve "$LLM_MODEL" --served-model-name "$LLM_SERVED_NAME" \
  --host 127.0.0.1 --port "$LLM_PORT" --gpu-memory-utilization "$LLM_GPU_UTIL" \
  --max-model-len "$LLM_MAX_LEN" --max-num-seqs "$LLM_MAX_SEQS" \
  >"$SERVICE_DIR/llm.log" 2>&1 &
echo $! >"$SERVICE_DIR/llm.pid"

waited=0
until llm_up; do
  sleep 5; waited=$((waited + 5))
  if ! kill -0 "$(cat "$SERVICE_DIR/llm.pid")" 2>/dev/null; then
    tail -n 40 "$SERVICE_DIR/llm.log" >&2
    die "llm exited during startup (see results/services/llm.log). Out of GPU memory: lower LLM_GPU_UTIL or LLM_MAX_LEN."
  fi
  (( waited >= 1500 )) && die "llm not ready after ${waited}s"
done
log "llm ready on :$LLM_PORT after ${waited}s"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/[rsched] GPU memory with llm: /' || true
