#!/usr/bin/env bash
# X3 (README, Experiments): py-spy profiles of 8 plan-replay episodes at N=1 and N=8.
#
#   bash scripts/run_profile.sh        # results/profile/<stamp>/
#   python analysis/profile_summary.py results/profile/<stamp>/N*_r0/traces
set -euo pipefail
source "$(dirname "$0")/common.sh"
command -v py-spy >/dev/null || die "py-spy not found: uv pip install py-spy"
STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$RS_ROOT/results/profile/$STAMP"
bash scripts/launch_services.sh status
python -m rsched.sweep configs/profile.yaml --out "$ROOT"
log "X3 done: python analysis/profile_summary.py $ROOT/N*_r0/traces"
