#!/usr/bin/env bash
# NVIDIA MPS (multi-process service) for the B stage 2a baseline: with MPS the
# processes' kernels share the GPU's SMs at the same time instead of taking turns.
#
#   bash scripts/stop_services.sh      # MPS only applies to processes started after it
#   bash scripts/mps.sh start
#   bash scripts/launch_services.sh && bash scripts/launch_llm.sh
#   bash scripts/mps.sh status         # should list one MPS server once services run
#   ...
#   bash scripts/stop_services.sh && bash scripts/mps.sh stop
#
# Many containers cannot run MPS (no nvidia-cuda-mps-control, or the daemon
# cannot start): "start" then fails and says why.
set -euo pipefail
source "$(dirname "$0")/common.sh"

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"

mps_up() { echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    command -v nvidia-cuda-mps-control >/dev/null || die "nvidia-cuda-mps-control not found: MPS is not available here"
    if mps_up; then log "MPS daemon already running"; exit 0; fi
    for name in vla sam3 molmo llm; do
      pidfile="$SERVICE_DIR/$name.pid"
      if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        die "$name is running; stop the services first (bash scripts/stop_services.sh)"
      fi
    done
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" nvidia-cuda-mps-control -d || die "MPS daemon failed to start"
    sleep 1
    mps_up || die "MPS daemon not answering; see $CUDA_MPS_LOG_DIRECTORY/control.log"
    # a CUDA client must connect for the server to come up; fail now, not in the services
    python - <<'PY' || { echo quit | nvidia-cuda-mps-control; die "a CUDA process cannot run under MPS here (log: $CUDA_MPS_LOG_DIRECTORY/server.log); daemon stopped"; }
import ctypes, sys
cuda = ctypes.CDLL("libcuda.so.1")
ok = cuda.cuInit(0) == 0
ctx = ctypes.c_void_p()
dev = ctypes.c_int()
ok = ok and cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0 and cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0
sys.exit(0 if ok else 1)
PY
    log "MPS daemon running (pipes: $CUDA_MPS_PIPE_DIRECTORY); start the services now"
    ;;
  stop)
    if mps_up; then echo quit | nvidia-cuda-mps-control; log "MPS daemon stopped"; else log "MPS daemon not running"; fi
    ;;
  status)
    if mps_up; then
      servers=$(echo get_server_list | nvidia-cuda-mps-control)
      log "MPS daemon running; servers: ${servers:-none (no CUDA client yet)}"
    else
      log "MPS daemon not running"
    fi
    ;;
  *) die "usage: $0 start|stop|status" ;;
esac
