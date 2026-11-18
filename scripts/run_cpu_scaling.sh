#!/usr/bin/env bash
# X2 / E8: concurrency x CPU budget, plan-replay episodes, one task set, one machine.
#
#   bash scripts/run_cpu_scaling.sh                    # results/cpu_scaling/<stamp>/
#   bash scripts/run_cpu_scaling.sh 20261124-090000    # resume (finished cells are skipped)
#
# One cell per (episode CPUs, level, repeat): results/cpu_scaling/<stamp>/c<budget>/N<level>_r<rep>.
# CPUs come from rsched/cpu_layout.py in topology order: the model servers, proxies
# and GPU sampler on the first X2_SERVICE_CORES, episodes on the next <budget>.
# A budget of k is a prefix of the next one, so budgets only add CPUs. The order
# of budgets rotates between repeats so that drift does not line up with one budget.
#
# Overrides:
#   X2_BUDGETS="4 8 16 24"     episode CPUs (default fits a 32-vCPU pod with 8 service CPUs)
#   X2_SERVICE_CORES=8
#   X2_LEVELS="1 4 8 16 24 32"
#   X2_REPEATS=2
#   X2_MPS=1                   run under MPS (0 = default time-slicing)
#   X2_SKIP_OVERSUB=1          skip cells with more than 2 episodes per CPU (0 = run them too)
#   X2_N1_ALL=0                N=1 only at the largest budget (1 = at every budget)
# Short levels use fewer episodes (N=1: 8, N=4: 16).
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/cpu_scaling/$STAMP"
read -r -a BUDGETS <<<"${X2_BUDGETS:-4 8 16 24}"
read -r -a LEVELS <<<"${X2_LEVELS:-1 4 8 16 24 32}"
REPEATS="${X2_REPEATS:-2}"
SVC="${X2_SERVICE_CORES:-8}"
CFG="configs/cpu_scaling.yaml"
mkdir -p "$ROOT"
bash scripts/record_platform.sh "$ROOT"

max_budget=0; for b in "${BUDGETS[@]}"; do (( b > max_budget )) && max_budget=$b; done
python - "$SVC" "$max_budget" <<'PY' || die "CPU check failed (see above); lower X2_BUDGETS or X2_SERVICE_CORES"
import sys
from rsched.cpu_layout import allowed_cpus, layout, quota_cores
svc, eps = int(sys.argv[1]), int(sys.argv[2])
layout(svc, eps)  # raises if there are not enough CPUs
q = quota_cores()
print(f"[rsched] CPUs available: {len(allowed_cpus())}, cgroup quota: {q if q is not None else 'none'}")
if q is not None and svc + eps > q + 1e-6:
    sys.exit(f"{svc} service + {eps} episode CPUs exceed the quota of {q:g}: the largest budget would be throttled")
PY

# fixed layout for services, proxies and the sampler; episodes get --episode-cpus per cell
eval "export $(python -m rsched.cpu_layout --service-cores "$SVC" --episode-cores "$max_budget" --shell)"
export RS_PIN=1
episode_cpus() { python -m rsched.cpu_layout --service-cores "$SVC" --episode-cores "$1" --episodes-only; }
{ echo "service CPUs: $SERVICE_CPUS"; for b in "${BUDGETS[@]}"; do echo "episode CPUs c$b: $(episode_cpus "$b")"; done; } \
  | tee "$ROOT/cpu_layout.txt"

restart() {  # $1 = MPS on (1) / off (0); services on SERVICE_CPUS (scripts/launch_services.sh)
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop || true; fi
  bash scripts/launch_services.sh
}
if [[ ! -f "$ROOT/services.txt" ]]; then
  restart "${X2_MPS:-1}"
  { echo "mps=${X2_MPS:-1} service_cpus=$SERVICE_CPUS"; bash scripts/mps.sh status || true; } | tee "$ROOT/services.txt"
else
  log "resuming: services are expected to be up as recorded in $ROOT/services.txt"
  bash scripts/launch_services.sh status
fi

episodes_for() { case "$1" in 1) echo 8 ;; 4) echo 16 ;; *) echo 32 ;; esac; }

if [[ ! -f "$ROOT/warmup/N8_r0/manifest.json" ]]; then
  rm -rf "$ROOT/warmup"
  log "warm-up: 8 episodes"
  python -m rsched.sweep "$CFG" --levels 8 --repeats 1 --episodes 8 --out "$ROOT/warmup" \
    --episode-cpus "$(episode_cpus "$max_budget")"
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
        log "[$k/$total] skip c=$budget N=$level r=$r (more than 2 episodes per CPU)"; continue
      fi
      if [[ "${X2_N1_ALL:-0}" != 1 ]] && (( level == 1 && budget != max_budget )); then
        log "[$k/$total] skip c=$budget N=1 r=$r (N=1 only at c=$max_budget)"; continue
      fi
      [[ -d "$cell" ]] && mv "$cell" "$cell.partial-$(date +%H%M%S)"
      log "[$k/$total] CPUs=$budget N=$level r=$r"
      python -m rsched.sweep "$CFG" --levels "$level" --repeats 1 --repeat-index "$r" \
        --episodes "$(episodes_for "$level")" --out "$ROOT/c$budget" --episode-cpus "$(episode_cpus "$budget")"
    done
  done
done
log "X2 done: $ROOT"
log "next: see the RunPod runbook (anatomy.py and gpu_util.py over $ROOT/c*/N*_r*)"
