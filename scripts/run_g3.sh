#!/usr/bin/env bash
# Run every G3 cell: scheduling mode x concurrency x repeat, on one machine.
#
#   bash scripts/launch_services.sh
#   bash scripts/run_g3.sh                 # new run under results/g3/<stamp>/
#   bash scripts/run_g3.sh 20261103-140000 # resume that run (finished cells are skipped)
#
# Layout: results/g3/<stamp>/<mode>/N<level>_r<repeat>/ (one sweep level each), so
# analysis/breakdown.py works per mode and analysis/g3_compare.py on the stamp.
# The mode order rotates every repeat so that slow drift (thermal, host noise)
# does not always favour the same mode. A cell interrupted halfway is moved
# aside to <cell>.partial-<time> and run again.
#
# Overrides: G3_MODES="none intent_srpt", G3_LEVELS="16", G3_REPEATS=1.
set -euo pipefail
source "$(dirname "$0")/common.sh"

read -r -a MODES <<<"${G3_MODES:-none intent_srpt intent_lrpt history_srpt history_lrpt}"
read -r -a LEVELS <<<"${G3_LEVELS:-8 16}"
REPEATS="${G3_REPEATS:-3}"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/g3/$STAMP"
CONFIG="${G3_CONFIG:-configs/g3.yaml}"

bash scripts/launch_services.sh status
n=${#MODES[@]}
total=$(( n * ${#LEVELS[@]} * REPEATS ))
k=0
for (( r = 0; r < REPEATS; r++ )); do
  for (( i = 0; i < n; i++ )); do
    mode="${MODES[$(( (i + r) % n ))]}"
    for level in "${LEVELS[@]}"; do
      k=$(( k + 1 ))
      cell="$ROOT/$mode/N${level}_r${r}"
      if [[ -f "$cell/manifest.json" ]]; then
        log "[$k/$total] skip $mode N=$level r=$r (done)"
        continue
      fi
      if [[ -d "$cell" ]]; then
        mv "$cell" "$cell.partial-$(date +%H%M%S)"
        log "moved an interrupted $mode N=$level r=$r aside"
      fi
      log "[$k/$total] $mode N=$level r=$r -> $cell"
      python -m rsched.sweep "$CONFIG" --levels "$level" --repeats 1 --repeat-index "$r" \
        --sched "$mode" --out "$ROOT/$mode"
    done
  done
done
log "G3 done: $ROOT"
log "next: python analysis/g3_compare.py $ROOT"
