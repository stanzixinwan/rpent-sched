#!/usr/bin/env bash
# Plan B feasibility: interference between a local VLM planner (vLLM) and the
# robot models (Pi0.5, Molmo) on one GPU. ~20 min with the default scenarios.
#
#   bash scripts/launch_services.sh && bash scripts/launch_llm.sh
#   bash scripts/run_b_feasibility.sh                 # results/b_feasibility/<stamp>/
#
# Extra arguments go to rsched.bench_interference, e.g. --duration 30 or
# --scenarios vla=1 llm=4 vla=1,llm=4. See that module for what each worker does.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RS_ROOT/results/b_feasibility/$STAMP"
bash scripts/launch_services.sh status
bash scripts/launch_llm.sh status
python -m rsched.bench_interference --out "$OUT" "$@"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader > "$OUT/gpu_info.txt" || true
log "done: $OUT"
log "copy back: tar czf /workspace/b_feasibility.tgz -C $RS_ROOT results/b_feasibility && runpodctl send /workspace/b_feasibility.tgz"
