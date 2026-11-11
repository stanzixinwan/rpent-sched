#!/usr/bin/env bash
# B stage 2a, follow-up: does coordination still help once the servers run under
# MPS, and is "intent" better than its budget? ~1.5 h on an A100.
#
#   bash scripts/run_b2a_mps.sh 20261110-212217     # the stamp of the run_b2a.sh run
#
# Adds to results/b2a/<stamp>/phases (same batches, paired):
#   1. under MPS: intent / history / oracle at budget 1, 2, 4 s (labels ...@mps),
#      compared with none@mps (vs_same_pct in summary.csv)
#   2. under MPS with the planner capped at B2A_MPS_SPLIT % of the SMs (default 30):
#      none@mps30. B2A_MPS_SPLIT=0 skips it.
# Services are restarted without MPS at the end.
#
# Overrides: B2A_LEVELS ("8"), B2A_REPEATS (3), B2A_BUDGETS ("1 2 4").
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:?usage: $0 <stamp of the run_b2a.sh run>}"
OUT="$RS_ROOT/results/b2a/$STAMP/phases"
[[ -f "$OUT/runs.jsonl" ]] || die "no $OUT/runs.jsonl: run scripts/run_b2a.sh first"
read -r -a LEVELS <<<"${B2A_LEVELS:-8}"
read -r -a BUDGETS <<<"${B2A_BUDGETS:-1 2 4}"
REPEATS="${B2A_REPEATS:-3}"
SPLIT="${B2A_MPS_SPLIT:-30}"
POLICIES=(none)
for b in "${BUDGETS[@]}"; do POLICIES+=("intent:budget=$b" "history:budget=$b" "oracle:budget=$b"); done
PHASES=(python -m rsched.bench_phases --out "$OUT" --levels "${LEVELS[@]}" --repeats "$REPEATS")

restart() {  # $1 = with MPS (1) or not (0); $2 = LLM_MPS_PCT or ""
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop; fi
  bash scripts/launch_services.sh
  LLM_MPS_PCT="$2" bash scripts/launch_llm.sh
  bash scripts/mps.sh status
}
trap 'log "restarting services without MPS"; restart 0 "" || true' EXIT

restart 1 ""
log "1/2 under MPS: ${#POLICIES[@]} policies x levels ${LEVELS[*]} x $REPEATS repeats (none@mps reused if done)"
"${PHASES[@]}" --policies "${POLICIES[@]}" --tag mps --warmup-episodes 2

if [[ "$SPLIT" != 0 ]]; then
  log "2/2 under MPS, planner capped at $SPLIT% of the SMs"
  restart 1 "$SPLIT"
  "${PHASES[@]}" --policies none --tag "mps$SPLIT" --warmup-episodes 2
fi

"${PHASES[@]}" --summarize-only
log "done: $OUT"
log "copy back: tar czf /workspace/b2a.tgz -C $RS_ROOT results/b2a results/b_feasibility && runpodctl send /workspace/b2a.tgz"
