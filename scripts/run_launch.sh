#!/usr/bin/env bash
# G4: launch order by planned work, on the GPU. Resumable.
#
#   bash scripts/launch_services.sh
#   bash scripts/run_launch.sh                  # new run under results/launch/<stamp>/
#   bash scripts/run_launch.sh 20261105-140000  # resume that run
#
# First a short warm-up cell (discarded) so no measured cell runs on cold
# services. Then, for each repeat r, both rules on the same shuffled batch,
# with the rule order alternating between repeats. Layout:
# results/launch/<stamp>/<rule>/N<level>_r<r>/.
#
# Overrides: LAUNCH_RULES="given jit_lpt lpt", LAUNCH_REPEATS=4, LAUNCH_LEVEL=8,
# LAUNCH_CONFIG=configs/g4_launch.yaml, LAUNCH_WARMUP=8 (episodes; 0 to skip).
set -euo pipefail
source "$(dirname "$0")/common.sh"

read -r -a RULES <<<"${LAUNCH_RULES:-given jit_lpt}"
REPEATS="${LAUNCH_REPEATS:-4}"
LEVEL="${LAUNCH_LEVEL:-8}"
CONFIG="${LAUNCH_CONFIG:-configs/g4_launch.yaml}"
WARMUP="${LAUNCH_WARMUP:-8}"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/launch/$STAMP"

bash scripts/launch_services.sh status
if (( WARMUP > 0 )) && [[ ! -f "$ROOT/warmup/N${LEVEL}_r0/manifest.json" ]]; then
  log "warm-up: $WARMUP episodes at N=$LEVEL (discarded)"
  rm -rf "$ROOT/warmup"
  python -m rsched.sweep "$CONFIG" --levels "$LEVEL" --repeats 1 --episodes "$WARMUP" \
    --launch given --out "$ROOT/warmup"
fi

n=${#RULES[@]}
total=$(( n * REPEATS ))
k=0
for (( r = 0; r < REPEATS; r++ )); do
  for (( i = 0; i < n; i++ )); do
    if (( r % 2 == 0 )); then rule="${RULES[$i]}"; else rule="${RULES[$(( n - 1 - i ))]}"; fi
    k=$(( k + 1 ))
    cell="$ROOT/$rule/N${LEVEL}_r${r}"
    if [[ -f "$cell/manifest.json" ]]; then
      log "[$k/$total] skip $rule r=$r (done)"
      continue
    fi
    if [[ -d "$cell" ]]; then
      mv "$cell" "$cell.partial-$(date +%H%M%S)"
      log "moved an interrupted $rule r=$r aside"
    fi
    log "[$k/$total] $rule N=$LEVEL r=$r -> $cell"
    python -m rsched.sweep "$CONFIG" --levels "$LEVEL" --repeats 1 --repeat-index "$r" \
      --launch "$rule" --out "$ROOT/$rule"
  done
done
log "G4 done: $ROOT"
log "next: python analysis/launch_compare.py $ROOT"
