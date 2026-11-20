#!/usr/bin/env bash
# X9 (README, Experiments): is Molmo's serial lock the throughput limit?
#
#   bash scripts/run_molmo_replicas.sh            # results/molmo_replicas/<stamp>/
#   bash scripts/run_molmo_replicas.sh <stamp>    # resume
#
# Arms: one  (molmo behind the proxy as everywhere else: one server, FIFO, 1 in flight)
#       two  (two Molmo servers on MOLMO_PORT and MOLMO_PORT+1, one comma-separated
#             backend, 2 in flight; each request takes a free replica)
# Both arms run with both servers loaded (same GPU memory, same MPS clients);
# only the proxy's backend list differs. E8 episode set (32 per cell, random
# launch order shared by both arms at a repeat), MPS, X9_LEVELS (8 16 32),
# X9_REPEATS (3), arm order rotating. Restarts the services with two Molmo
# replicas under MPS.
# Afterwards:
#   python analysis/launch_compare.py results/molmo_replicas/<stamp> --baseline one
#   python analysis/anatomy.py --set one_N16="<root>/one/N16_r*" ... --out <root>/anatomy
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/molmo_replicas/$STAMP"
ARMS=(one two)
read -r -a LEVELS <<<"${X9_LEVELS:-8 16 32}"
REPEATS="${X9_REPEATS:-3}"
mkdir -p "$ROOT/configs"

export MOLMO_REPLICAS=2
bash scripts/stop_services.sh
bash scripts/mps.sh start
bash scripts/launch_services.sh
bash scripts/launch_services.sh status
bash scripts/record_platform.sh "$ROOT"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >"$ROOT/gpu_memory.txt"

python - "$ROOT/configs" "$MOLMO_PORT" <<'PY'
import copy, sys, yaml
out, port = sys.argv[1], int(sys.argv[2])
base = yaml.safe_load(open("configs/cpu_scaling.yaml"))
base.pop("cpu", None)  # RS_PIN layout from env/.env (18 episode vCPUs)
for arm, k in (("one", 1), ("two", 2)):
    cfg = copy.deepcopy(base)
    cfg["name"] = f"x9_molmo_{arm}"
    cfg["services"]["molmo"] = {
        "backend": ",".join(f"http://127.0.0.1:{port + i}" for i in range(k)),
        "policy": "fifo", "max_inflight": k}
    yaml.safe_dump(cfg, open(f"{out}/{arm}.yaml", "w"), sort_keys=False)
PY

if [[ ! -f "$ROOT/warmup/N8_r0/manifest.json" ]]; then
  log "warm-up: 8 episodes on two replicas"
  python -m rsched.sweep "$ROOT/configs/two.yaml" --levels 8 --repeats 1 --episodes 8 --out "$ROOT/warmup"
  rm -rf "$ROOT/warmup/N8_r0/episodes"
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
      rm -rf "$cell/episodes"   # not analysed; fills the disk
    done
  done
done
log "X9 done: python analysis/launch_compare.py $ROOT --baseline one"
