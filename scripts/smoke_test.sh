#!/usr/bin/env bash
# One-command smoke test.
#
#   bash scripts/smoke_test.sh         # CPU: fake services on RPent's real RPC stack,
#                                      # full sweep -> proxies -> hooks -> analysis (~1 min)
#   bash scripts/smoke_test.sh --gpu   # one real LIBERO-PRO episode through the whole path
#                                      # (needs launch_services.sh and a planner API key)
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ "${1:-}" != "--gpu" ]]; then
  log "CPU smoke test"
  python -m pytest -q
  log "CPU smoke test passed"
  exit 0
fi

log "GPU smoke test: checking services"
bash scripts/launch_services.sh status
out="results/smoke_gpu/$(date +%Y%m%d-%H%M%S)"
python -m rsched.sweep configs/smoke_gpu.yaml --out "$out"
python analysis/breakdown.py "$out"

trace=$(ls "$out"/N1_r0/traces/episode_*.jsonl | head -1)
python - "$trace" "$out/N1_r0/proxy_vla.jsonl" <<'PY'
import sys
from rsched.traceio import read_jsonl
trace, proxy = read_jsonl(sys.argv[1]), read_jsonl(sys.argv[2])
tools = [r for r in trace if r["type"] == "tool"]
outcome = next((r for r in trace if r["type"] == "outcome"), None)
predicts = [r for r in proxy if r["method"] == "vla.predict"]
print(f"tool calls: {len(tools)}  ({', '.join(sorted({t['tool'] for t in tools}))})")
print(f"vla.predict through proxy: {len(predicts)}, attributed: {sum(bool(r['call_id']) for r in predicts)}")
print(f"outcome: success={outcome and outcome['environment_success']} error={outcome and outcome['agent_error']}")
assert tools, "no tool calls traced -- is the planner key set? see logs/episode_*.log"
assert predicts, "no VLA requests reached the proxy"
PY
log "GPU smoke test passed: $out"
