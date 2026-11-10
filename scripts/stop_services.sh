#!/usr/bin/env bash
# Stop the services started by launch_services.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

tree() {  # a pid and all its descendants
  echo "$1"
  for c in $(ps -A -o pid=,ppid= | awk -v p="$1" '$2 == p {print $1}'); do tree "$c"; done
}

for name in vla sam3 molmo llm; do
  pidfile="$SERVICE_DIR/$name.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    # rsched.bench_phases may have left the planner frozen (SIGSTOP): thaw it so it can exit
    [[ $name == llm ]] && kill -CONT $(tree "$(cat "$pidfile")") 2>/dev/null || true
    pid=$(cat "$pidfile")
    kill "$pid" && log "stopping $name"
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { kill -9 "$pid"; log "killed $name (did not exit in 60 s)"; }
  fi
  rm -f "$pidfile"
done
