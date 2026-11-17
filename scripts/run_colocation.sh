#!/usr/bin/env bash
# Co-location interference (B1) under time-slicing AND under MPS, in one session
# on one machine, so the two columns of the interference matrix and the MPS cost
# on a model running alone are measured on the same hardware. ~45 min.
#
#   bash scripts/run_colocation.sh                 # results/colocation/<stamp>/{timeslice,mps}/
#
# Needs the local planner (env/setup_vllm.sh). Extra arguments go to
# rsched.bench_interference (default: its full scenario list, 60 s each).
# Services end up running without MPS.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="$RS_ROOT/results/colocation/$STAMP"
mkdir -p "$ROOT"
bash scripts/record_platform.sh "$ROOT"

restart() {  # $1 = with MPS (1) or not (0)
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop || true; fi
  bash scripts/launch_services.sh
  bash scripts/launch_llm.sh
}
trap 'log "restarting services without MPS"; restart 0 || true' EXIT

for mode in timeslice mps; do
  log "$mode"
  restart "$([[ $mode == mps ]] && echo 1 || echo 0)"
  bash scripts/mps.sh status > "$ROOT/mps_status_$mode.txt" 2>&1 || true
  python -m rsched.bench_interference --out "$ROOT/$mode" "$@"
done
log "done: $ROOT  (compare $ROOT/timeslice/inflation.csv with $ROOT/mps/inflation.csv; solo rows in summary.csv)"
log "copy back: tar czf /workspace/colocation.tgz -C $RS_ROOT results/colocation/$STAMP && runpodctl send /workspace/colocation.tgz"
