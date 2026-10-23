#!/usr/bin/env bash
# Stop the services started by launch_services.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

for name in vla sam3 molmo; do
  pidfile="$SERVICE_DIR/$name.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    kill "$(cat "$pidfile")" && log "stopped $name"
  fi
  rm -f "$pidfile"
done
