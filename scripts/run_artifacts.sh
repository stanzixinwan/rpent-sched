#!/usr/bin/env bash
# X8 (README, Experiments): what does RPent's per-step artifact saving cost?
#
#   bash scripts/run_artifacts.sh                 # results/artifacts/<stamp>/
#   bash scripts/run_artifacts.sh <stamp>         # resume
#
# Arms: store (RPent unchanged) | lazy (RS_ARTIFACTS=lazy, rsched/artifacts.py:
# images and depth maps stay in memory and are written only if a tool reads
# them). Same episodes as X0, fully traced, MPS, at X8_LEVELS (1 8 16),
# X8_REPEATS (3) repeats, arm order rotating. N=1 runs the first X8_N1_EPISODES (8).
# Services must be up (MPS).
# Afterwards:
#   python analysis/overhead.py results/artifacts/<stamp> --baseline store --arms lazy
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/artifacts/$STAMP"
ARMS=(store lazy)
read -r -a LEVELS <<<"${X8_LEVELS:-1 8 16}"
REPEATS="${X8_REPEATS:-3}"
N1_EPISODES="${X8_N1_EPISODES:-8}"
mkdir -p "$ROOT/configs"
bash scripts/record_platform.sh "$ROOT"
bash scripts/launch_services.sh status

python - "$ROOT/configs" <<'PY'
import copy, sys, yaml
out = sys.argv[1]
base = yaml.safe_load(open("configs/overhead.yaml"))
for arm in ("store", "lazy"):
    cfg = copy.deepcopy(base)
    cfg["name"] = f"x8_{arm}"
    cfg.setdefault("episode_env", {})["RS_ARTIFACTS"] = arm
    yaml.safe_dump(cfg, open(f"{out}/{arm}.yaml", "w"), sort_keys=False)
PY

if [[ ! -f "$ROOT/warmup/N8_r0/manifest.json" ]]; then
  log "warm-up: 8 episodes"
  python -m rsched.sweep "$ROOT/configs/store.yaml" --levels 8 --repeats 1 --episodes 8 --out "$ROOT/warmup"
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
      eps=(); [[ "$level" == 1 ]] && eps=(--episodes "$N1_EPISODES")
      log "[$k/$total] $arm N=$level r=$r"
      python -m rsched.sweep "$ROOT/configs/$arm.yaml" --levels "$level" --repeats 1 --repeat-index "$r" \
        "${eps[@]}" --out "$ROOT/$arm"
      # episode outputs are not analysed and fill the disk; keep traces and manifests
      rm -rf "$cell/episodes"
    done
  done
done
log "X8 done: python analysis/overhead.py $ROOT --baseline store --arms lazy"
