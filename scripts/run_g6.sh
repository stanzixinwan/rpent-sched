#!/usr/bin/env bash
# G6: cross-episode batching of Pi0.5 calls on the real Flash workload (the G5
# held-out batch, 32 episodes), N=8 and N=16, under MPS. ~2-2.5 h on an A100.
#
#   bash scripts/run_g6.sh                    # results/g6/<stamp>/
#   bash scripts/run_g6.sh 20261112-140000    # resume (finished cells are skipped)
#
# Arms (VLA proxy, configs/g6_batch.yaml with the batch settings swapped in):
#   nobatch   max_batch 1: one call at a time, through the same batching code
#   greedy    send everything waiting when Pi0.5 frees up (max 16)
#   intent30  hold the oldest call <= 30 ms while another episode inside a pick
#             is expected to send its next chunk, using the pick's max_chunks
# One cell per arm, level and repeat; every arm at a repeat replays the same
# shuffled batch; arm order rotates between repeats. A short warm-up cell first.
#
# Overrides: G6_ARMS, G6_LEVELS ("8 16"), G6_REPEATS (3), G6_MPS=0 (as is).
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/g6/$STAMP"
read -r -a ARMS <<<"${G6_ARMS:-nobatch greedy intent30}"
read -r -a LEVELS <<<"${G6_LEVELS:-8 16}"
REPEATS="${G6_REPEATS:-3}"
mkdir -p "$ROOT/configs"

restart() {  # $1 = with MPS (1) or not (0)
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop; fi
  bash scripts/launch_services.sh
}
if [[ "${G6_MPS:-1}" == 1 ]]; then
  trap 'log "restarting services without MPS"; restart 0 || true' EXIT
  bash scripts/mps.sh status | grep -q "daemon running" || restart 1
else
  bash scripts/launch_services.sh
fi
bash scripts/mps.sh status | tee "$ROOT/mps_status.txt"

for arm in "${ARMS[@]}"; do
  python - "$arm" "$ROOT/configs/$arm.yaml" <<'PY'
import sys, yaml
arm, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/g6_batch.yaml"))
vla = cfg["services"]["vla"]
settings = {"nobatch": dict(max_batch=1, batch_mode="greedy", batch_window_ms=0, batch_intent=False),
            "greedy": dict(max_batch=16, batch_mode="greedy", batch_window_ms=0, batch_intent=False),
            "window30": dict(max_batch=16, batch_mode="window", batch_window_ms=30, batch_intent=False),
            "predict30": dict(max_batch=16, batch_mode="predict", batch_window_ms=30, batch_intent=False),
            "intent30": dict(max_batch=16, batch_mode="predict", batch_window_ms=30, batch_intent=True)}
if arm not in settings:
    sys.exit(f"unknown arm {arm}; one of {list(settings)}")
vla.update(settings[arm])
cfg["name"] = f"g6_{arm}"
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
done

if [[ ! -f "$ROOT/warmup/N8_r0/manifest.json" ]]; then
  rm -rf "$ROOT/warmup"
  log "warm-up: 8 episodes"
  python -m rsched.sweep "$ROOT/configs/${ARMS[0]}.yaml" --levels 8 --repeats 1 --episodes 8 --out "$ROOT/warmup"
fi

n=${#ARMS[@]}; total=$(( n * REPEATS * ${#LEVELS[@]} )); k=0
for (( r = 0; r < REPEATS; r++ )); do
  for level in "${LEVELS[@]}"; do
    for (( i = 0; i < n; i++ )); do
      arm="${ARMS[$(( (i + r) % n ))]}"
      k=$(( k + 1 ))
      cell="$ROOT/$arm/N${level}_r${r}"
      if [[ -f "$cell/manifest.json" ]]; then log "[$k/$total] skip $arm N=$level r=$r (done)"; continue; fi
      [[ -d "$cell" ]] && mv "$cell" "$cell.partial-$(date +%H%M%S)"
      log "[$k/$total] $arm N=$level r=$r"
      python -m rsched.sweep "$ROOT/configs/$arm.yaml" --levels "$level" --repeats 1 --repeat-index "$r" \
        --out "$ROOT/$arm"
    done
  done
done
log "G6 done: $ROOT"
log "next: python analysis/launch_compare.py $ROOT --baseline nobatch"
log "copy back: tar czf /workspace/g6.tgz -C $RS_ROOT results/g6 && runpodctl send /workspace/g6.tgz"
