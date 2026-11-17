#!/usr/bin/env bash
# X0 (README, Experiments): cost of the tracing itself.
#
#   bash scripts/run_overhead.sh                  # results/overhead/<stamp>/
#   bash scripts/run_overhead.sh <stamp>          # resume
#
# Arms: full | proxy_only (RS_TRACE=off) | bare (RS_TRACE=off, no proxies), at
# N=1 and N=8, X0_REPEATS (3) repeats, arm order rotating. Services must be up.
# Afterwards: python analysis/overhead.py results/overhead/<stamp>
set -euo pipefail
source "$(dirname "$0")/../common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/overhead/$STAMP"
ARMS=(full proxy_only bare)
read -r -a LEVELS <<<"${X0_LEVELS:-1 8}"
REPEATS="${X0_REPEATS:-3}"
mkdir -p "$ROOT/configs"
bash scripts/record_platform.sh "$ROOT"
bash scripts/launch_services.sh status

python - "$ROOT/configs" <<'PY'
import copy, sys, yaml
out = sys.argv[1]
base = yaml.safe_load(open("configs/overhead.yaml"))
for arm in ("full", "proxy_only", "bare"):
    cfg = copy.deepcopy(base)
    cfg["name"] = f"x0_{arm}"
    if arm != "full":
        cfg.setdefault("episode_env", {})["RS_TRACE"] = "off"
    if arm == "bare":
        for svc in cfg["services"].values():
            svc["proxy"] = False
    yaml.safe_dump(cfg, open(f"{out}/{arm}.yaml", "w"), sort_keys=False)
PY

if [[ ! -f "$ROOT/warmup/N8_r0/manifest.json" ]]; then
  log "warm-up: 8 episodes"
  python -m rsched.sweep "$ROOT/configs/full.yaml" --levels 8 --repeats 1 --episodes 8 --out "$ROOT/warmup"
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
log "X0 done: python analysis/overhead.py $ROOT"
