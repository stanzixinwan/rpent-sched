#!/usr/bin/env bash
# X2 (README, Experiments): concurrency x CPU budget, plan-replay episodes, one task set.
#
#   bash scripts/run_cpu_scaling.sh                    # results/cpu_scaling/<stamp>/
#   bash scripts/run_cpu_scaling.sh 20261124-090000    # resume (finished cells are skipped)
#
# One cell per (CPU budget, level, repeat): results/cpu_scaling/<stamp>/c<budget>/N<level>_r<rep>.
# Episodes are pinned to <budget> cores starting at X2_FIRST_CPU; the model
# services are (re)started pinned to the cores below that. Order rotates
# between repeats so that drift does not line up with one budget.
#
# Overrides:
#   X2_BUDGETS="8 16 32 48"   cores for the episodes
#   X2_LEVELS="1 4 8 16 24 32"
#   X2_REPEATS=3
#   X2_FIRST_CPU=8            episodes use cores X2_FIRST_CPU .. X2_FIRST_CPU+budget-1
#   X2_MPS=1                  run under MPS (0 = default time-slicing)
#   X2_SKIP_OVERSUB=1         skip cells with more than 2 episodes per core (0 = run them too)
# Short levels use fewer episodes (N=1: 8, N=4: 16) to keep the run within a day.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/cpu_scaling/$STAMP"
read -r -a BUDGETS <<<"${X2_BUDGETS:-8 16 32 48}"
read -r -a LEVELS <<<"${X2_LEVELS:-1 4 8 16 24 32}"
REPEATS="${X2_REPEATS:-3}"
FIRST="${X2_FIRST_CPU:-8}"
CFG="configs/cpu_scaling.yaml"
mkdir -p "$ROOT"

# The quota must cover the largest budget plus the service cores, or the cells
# measure throttling instead of the budget.
max_budget=0; for b in "${BUDGETS[@]}"; do (( b > max_budget )) && max_budget=$b; done
need=$(( FIRST + max_budget ))
python - "$need" <<'PY' || die "CPU check failed (see above); lower X2_BUDGETS or use a larger machine"
import os, sys
sys.path.insert(0, ".")
from rsched.sweep import cpu_info
need = int(sys.argv[1])
info = cpu_info()
avail = sorted(os.sched_getaffinity(0))
print(f"[rsched] CPUs visible: {len(avail)}, cgroup cpu.max: {info['cgroup_cpu_max']}")
if len(avail) < need or max(avail) < need - 1:
    sys.exit(f"need CPUs 0..{need - 1}, have {len(avail)}")
quota = (info["cgroup_cpu_max"] or "max").split()
if quota[0] != "max" and int(quota[0]) / int(quota[1]) < need:
    sys.exit(f"cgroup quota {int(quota[0]) / int(quota[1]):.1f} CPUs < {need} needed: cells would be throttled")
PY

restart() {  # $1 = MPS on (1) / off (0); services pinned below the episode cores
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop || true; fi
  SERVICE_CPUS="0-$(( FIRST - 1 ))" bash scripts/launch_services.sh
}
if [[ ! -f "$ROOT/services.txt" ]]; then
  restart "${X2_MPS:-1}"
  { echo "mps=${X2_MPS:-1} service_cpus=0-$(( FIRST - 1 )) first_episode_cpu=$FIRST"; bash scripts/mps.sh status || true
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; } | tee "$ROOT/services.txt"
else
  log "resuming: services are expected to be up as recorded in $ROOT/services.txt"
  bash scripts/launch_services.sh status
fi

episodes_for() { case "$1" in 1) echo 8 ;; 4) echo 16 ;; *) echo 32 ;; esac; }

if [[ ! -f "$ROOT/warmup/N8_r0/manifest.json" ]]; then
  rm -rf "$ROOT/warmup" 2>/dev/null || mv "$ROOT/warmup" "$ROOT/warmup.partial-$(date +%H%M%S)" 2>/dev/null || true
  log "warm-up: 8 episodes"
  python -m rsched.sweep "$CFG" --levels 8 --repeats 1 --episodes 8 --out "$ROOT/warmup" \
    --episode-cpus "$FIRST-$(( FIRST + BUDGETS[0] - 1 ))"
fi

nb=${#BUDGETS[@]}; total=$(( nb * REPEATS * ${#LEVELS[@]} )); k=0
for (( r = 0; r < REPEATS; r++ )); do
  for level in "${LEVELS[@]}"; do
    for (( i = 0; i < nb; i++ )); do
      budget="${BUDGETS[$(( (i + r) % nb ))]}"
      k=$(( k + 1 ))
      cell="$ROOT/c$budget/N${level}_r${r}"
      if [[ -f "$cell/manifest.json" ]]; then log "[$k/$total] skip c=$budget N=$level r=$r (done)"; continue; fi
      if [[ "${X2_SKIP_OVERSUB:-1}" == 1 ]] && (( level > 2 * budget )); then
        log "[$k/$total] skip c=$budget N=$level r=$r (more than 2 episodes per core)"; continue
      fi
      [[ -d "$cell" ]] && mv "$cell" "$cell.partial-$(date +%H%M%S)"
      log "[$k/$total] cores=$budget N=$level r=$r"
      python -m rsched.sweep "$CFG" --levels "$level" --repeats 1 --repeat-index "$r" \
        --episodes "$(episodes_for "$level")" --out "$ROOT/c$budget" \
        --episode-cpus "$FIRST-$(( FIRST + budget - 1 ))"
    done
  done
done
log "X2 done: $ROOT"
log "next: python analysis/anatomy.py \$(for d in $ROOT/c*/N*_r0; do b=\$(basename \$(dirname \$d)); n=\$(basename \$d | cut -d_ -f1); echo --set \${b}_\${n}=$ROOT/\$b/\${n}_r\\*; done) --out $ROOT/anatomy"
