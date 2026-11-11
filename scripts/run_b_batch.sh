#!/usr/bin/env bash
# Go/no-go for cross-episode VLA batching: how Pi0.5 latency grows with batch
# size, alone and with 4 planner streams, without and with MPS. ~25 min.
#
#   bash scripts/launch_services.sh && bash scripts/launch_llm.sh
#   bash scripts/run_b_batch.sh                 # results/b_batch/<stamp>/
#
# Extra arguments go to rsched.bench_batch (e.g. --batch-sizes 1 2 4 8 16 32).
# B_BATCH_MPS=0 skips the MPS half. Services end up running without MPS.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RS_ROOT/results/b_batch/$STAMP"
mkdir -p "$OUT"
bash scripts/launch_services.sh status
bash scripts/launch_llm.sh status
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader > "$OUT/gpu_info.txt" || true

restart() {  # $1 = with MPS (1) or not (0)
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop; fi
  bash scripts/launch_services.sh
  bash scripts/launch_llm.sh
}

log "1/2 time-slicing"
python -m rsched.bench_batch --out "$OUT" "$@"

if [[ "${B_BATCH_MPS:-1}" != 0 ]]; then
  trap 'log "restarting services without MPS"; restart 0 || true' EXIT
  restart 1
  log "2/2 under MPS"
  python -m rsched.bench_batch --out "$OUT" --tag mps "$@"
fi
log "done: $OUT"
log "copy back: tar czf /workspace/b_batch.tgz -C $RS_ROOT results/b_batch && runpodctl send /workspace/b_batch.tgz"
