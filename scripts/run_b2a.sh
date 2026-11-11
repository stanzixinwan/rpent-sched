#!/usr/bin/env bash
# B stage 2a: can moving GPU time between the planner and the VLA shorten a batch
# of agent episodes, and does MPS already fix the interference? ~1.5-2 h on an A100.
#
#   bash scripts/launch_services.sh && bash scripts/launch_llm.sh
#   bash scripts/run_b2a.sh                     # results/b2a/<stamp>/
#   bash scripts/run_b2a.sh 20261111-090000     # resume (finished runs are skipped)
#
# 1. phases (time-slicing): rsched.bench_phases with every coordination policy,
#    including vla_first without the starvation guard (how much a frozen planner
#    gives back to the VLA, at the planner's expense)
# 2. MPS: restart all services under the MPS daemon, then rerun the feasibility
#    scenarios that matter (-> interference_mps/) and the uncoordinated batches
#    (label none@mps, paired with stage 1). B2A_MPS_SPLIT=30 adds none@mps30: the
#    planner capped at 30% of the SMs. Skipped with a note if MPS cannot run here.
#    Services are restarted without MPS at the end.
#
# Overrides: B2A_POLICIES, B2A_LEVELS ("4 8"), B2A_REPEATS (3), B2A_MPS=0 skips stage 2.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/b2a/$STAMP"
read -r -a POLICIES <<<"${B2A_POLICIES:-none vla_first vla_first:max_pause=1000,min_run=0 intent:budget=2 history:budget=2 oracle:budget=2}"
read -r -a LEVELS <<<"${B2A_LEVELS:-4 8}"
REPEATS="${B2A_REPEATS:-3}"
PHASES=(python -m rsched.bench_phases --out "$ROOT/phases" --levels "${LEVELS[@]}" --repeats "$REPEATS")
MPS_SCENARIOS=(vla=1 llm=4 molmo=1 vla=1,llm=4 vla=1,llm=16 molmo=1,llm=4 vla=1,molmo=1,llm=4)
mkdir -p "$ROOT"

restart() {  # $1 = with MPS (1) or not (0); $2 = LLM_MPS_PCT or ""
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start || return 1; else bash scripts/mps.sh stop; fi
  bash scripts/launch_services.sh
  LLM_MPS_PCT="$2" bash scripts/launch_llm.sh
  bash scripts/mps.sh status
}

bash scripts/launch_services.sh status
bash scripts/launch_llm.sh status
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader > "$ROOT/gpu_info.txt" || true

log "stage 1: phase-structured batches, ${#POLICIES[@]} policies x levels ${LEVELS[*]} x $REPEATS repeats"
"${PHASES[@]}" --policies "${POLICIES[@]}"

if [[ "${B2A_MPS:-1}" == 0 ]]; then
  log "stage 2 skipped (B2A_MPS=0)"
elif ! restart 1 ""; then
  echo "MPS could not start in this container" > "$ROOT/mps_unavailable.txt"
  log "MPS unavailable here; restarting services without it"
  restart 0 ""
else
  log "stage 2: under MPS"
  [[ -f "$ROOT/interference_mps/summary.csv" ]] || \
    python -m rsched.bench_interference --out "$ROOT/interference_mps" --scenarios "${MPS_SCENARIOS[@]}"
  "${PHASES[@]}" --policies none --tag mps --warmup-episodes 2
  if [[ -n "${B2A_MPS_SPLIT:-}" ]]; then
    log "stage 2b: planner capped at ${B2A_MPS_SPLIT}% of the SMs"
    restart 1 "$B2A_MPS_SPLIT"
    "${PHASES[@]}" --policies none --tag "mps$B2A_MPS_SPLIT" --warmup-episodes 2
  fi
  log "restarting services without MPS"
  restart 0 ""
fi

python -m rsched.bench_phases --out "$ROOT/phases" --summarize-only
log "done: $ROOT"
log "copy back: tar czf /workspace/b2a.tgz -C $RS_ROOT results/b2a && runpodctl send /workspace/b2a.tgz"
