#!/usr/bin/env bash
# B step 3: cross-episode batching of Pi0.5 calls on agent-like batches, under MPS.
# ~2 h on an A100.
#
#   bash scripts/run_b3.sh                      # results/b3/<stamp>/
#   bash scripts/run_b3.sh 20261112-090000      # resume (finished runs are skipped)
#
# Restarts the services under MPS (the setup of record after 2a), then for each
# variant starts an rsched proxy (policy batch) in front of Pi0.5 and runs the
# rsched.bench_phases batches (no pausing) through it:
#   none (baseline)  --max-batch 1: one request at a time, same code path
#   greedy           send everything waiting when the backend frees up
#   window10/30      hold the oldest request up to 10 / 30 ms
#   predict30        hold it (<= 30 ms) only while another episode inside a burst
#                    is expected to send its next chunk in time
#   intent30         the same, minus episodes whose pick has used its max_chunks
# summary.csv: vs_none_pct is against the unbatched baseline on the same batches.
# Proxy traces (batch sizes, waits) in proxy_<variant>.jsonl.
#
# Overrides: B3_VARIANTS, B3_LEVELS ("8 16"), B3_REPEATS (3), B3_MAX_BATCH (16),
# B3_MPS=0 keeps the services as they are.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/b3/$STAMP"
read -r -a VARIANTS <<<"${B3_VARIANTS:-none greedy window10 window30 predict30 intent30}"
read -r -a LEVELS <<<"${B3_LEVELS:-8 16}"
REPEATS="${B3_REPEATS:-3}"
MAXB="${B3_MAX_BATCH:-16}"
PPORT="${B3_PROXY_PORT:-9150}"
mkdir -p "$ROOT"

restart() {  # $1 = with MPS (1) or not (0)
  bash scripts/stop_services.sh
  if [[ $1 == 1 ]]; then bash scripts/mps.sh start; else bash scripts/mps.sh stop; fi
  bash scripts/launch_services.sh
  bash scripts/launch_llm.sh
}

if [[ "${B3_MPS:-1}" == 1 ]]; then
  trap 'log "restarting services without MPS"; restart 0 || true' EXIT
  restart 1
else
  bash scripts/launch_services.sh && bash scripts/launch_llm.sh
fi
bash scripts/mps.sh status | tee "$ROOT/mps_status.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader > "$ROOT/gpu_info.txt" || true

proxy_args() {  # variant -> proxy flags
  case "$1" in
    none)      echo "--max-batch 1 --batch-mode greedy" ;;
    greedy)    echo "--max-batch $MAXB --batch-mode greedy" ;;
    window10)  echo "--max-batch $MAXB --batch-mode window --batch-window-ms 10" ;;
    window30)  echo "--max-batch $MAXB --batch-mode window --batch-window-ms 30" ;;
    predict30) echo "--max-batch $MAXB --batch-mode predict --batch-window-ms 30" ;;
    intent30)  echo "--max-batch $MAXB --batch-mode predict --batch-window-ms 30 --batch-intent" ;;
    *) die "unknown variant $1" ;;
  esac
}

for v in "${VARIANTS[@]}"; do
  # shellcheck disable=SC2046
  python -m rsched.proxy --name vla --listen "$PPORT" --backend "http://127.0.0.1:$VLA_PORT" \
    --policy batch $(proxy_args "$v") --log "$ROOT/proxy_$v.jsonl" > "$ROOT/proxy_$v.log" 2>&1 &
  PROXY=$!
  for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PPORT/rs/stats" >/dev/null && break; sleep 0.2; done
  tag=(); [[ $v != none ]] && tag=(--tag "$v")
  log "variant $v ($(proxy_args "$v"))"
  python -m rsched.bench_phases --out "$ROOT/phases" --policies none "${tag[@]}" \
    --levels "${LEVELS[@]}" --repeats "$REPEATS" --vla-endpoint "http://127.0.0.1:$PPORT" || { kill $PROXY; exit 1; }
  curl -s "http://127.0.0.1:$PPORT/rs/stats" > "$ROOT/proxy_${v}_stats.json" || true
  kill $PROXY; wait $PROXY 2>/dev/null || true
done

python -m rsched.bench_phases --out "$ROOT/phases" --summarize-only
log "done: $ROOT"
log "copy back: tar czf /workspace/b3.tgz -C $RS_ROOT results/b3 && runpodctl send /workspace/b3.tgz"
