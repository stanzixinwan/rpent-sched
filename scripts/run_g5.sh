#!/usr/bin/env bash
# G5: plan-based launch order on held-out tasks, against per-task history.
#
#   bash scripts/launch_services.sh
#   bash scripts/run_g5.sh                    # new run under results/g5/<stamp>/
#   bash scripts/run_g5.sh 20261110-140000    # resume (finished cells are skipped)
#
# 1. record   every Flash task once (configs/g5_record.yaml, 78 episodes, N=8);
#             doubles as the warm-up
# 2. history  results/g5/<stamp>/history.json: mean wall time per task (rsched.history_table)
# 3. arms     configs/g5_heldout.yaml (16 unseen tasks x seeds 1, 2), one cell per arm
#             and repeat; every arm at a repeat starts from the same shuffled batch;
#             arm order rotates between repeats
#
# Arms are label=rule:estimate:cold. Default:
#   given=given:plan:0  plan=jit_lpt:plan:0  history=jit_lpt:history:0
#   history_cold50=jit_lpt:history:0.5
# Overrides: G5_ARMS="...", G5_REPEATS=5, G5_LEVEL=8, G5_SKIP_RECORD=1 with
# G5_HISTORY=<existing history.json>, G5_RECORD_CONFIG, G5_CONFIG.
set -euo pipefail
source "$(dirname "$0")/common.sh"

read -r -a ARMS <<<"${G5_ARMS:-given=given:plan:0 plan=jit_lpt:plan:0 history=jit_lpt:history:0 history_cold50=jit_lpt:history:0.5}"
REPEATS="${G5_REPEATS:-5}"
LEVEL="${G5_LEVEL:-8}"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/g5/$STAMP"
HISTORY="${G5_HISTORY:-$ROOT/history.json}"
RECORD_CONFIG="${G5_RECORD_CONFIG:-configs/g5_record.yaml}"
CONFIG="${G5_CONFIG:-configs/g5_heldout.yaml}"

bash scripts/launch_services.sh status
mkdir -p "$ROOT"

if [[ -z "${G5_SKIP_RECORD:-}" ]]; then
  if [[ ! -f "$ROOT/record/N${LEVEL}_r0/manifest.json" ]]; then
    rm -rf "$ROOT/record"
    log "record: every Flash task once at N=$LEVEL"
    python -m rsched.sweep "$RECORD_CONFIG" --levels "$LEVEL" --repeats 1 --out "$ROOT/record"
  fi
  python -m rsched.history_table "$ROOT/record/N${LEVEL}_r0" --out "$HISTORY"
fi
[[ -f "$HISTORY" ]] || die "no history table at $HISTORY"

n=${#ARMS[@]}
total=$(( n * REPEATS ))
k=0
for (( r = 0; r < REPEATS; r++ )); do
  for (( i = 0; i < n; i++ )); do
    arm="${ARMS[$(( (i + r) % n ))]}"
    label="${arm%%=*}"
    IFS=: read -r rule estimate cold <<<"${arm#*=}"
    k=$(( k + 1 ))
    cell="$ROOT/$label/N${LEVEL}_r${r}"
    if [[ -f "$cell/manifest.json" ]]; then
      log "[$k/$total] skip $label r=$r (done)"
      continue
    fi
    if [[ -d "$cell" ]]; then
      mv "$cell" "$cell.partial-$(date +%H%M%S)"
      log "moved an interrupted $label r=$r aside"
    fi
    log "[$k/$total] $label ($rule, estimate=$estimate, cold=$cold) r=$r -> $cell"
    python -m rsched.sweep "$CONFIG" --levels "$LEVEL" --repeats 1 --repeat-index "$r" \
      --launch "$rule" --estimate "$estimate" --cold "$cold" --history-file "$HISTORY" --out "$ROOT/$label"
  done
done
log "G5 done: $ROOT"
log "next: python analysis/launch_compare.py $ROOT --baseline given"
