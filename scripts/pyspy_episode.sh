#!/usr/bin/env bash
# Run one traced rpent episode under py-spy (configs/profile.yaml uses
# this as rpent.command). Arguments are rpent's; RS_EPISODE_ID and RS_TRACE_DIR
# come from the sweep. Output: $RS_TRACE_DIR/pyspy_<episode>.txt (collapsed
# stacks, one line per stack with its sample count), covering the episode
# process and the env_server it spawns (--subprocesses).
#
# Needs py-spy (uv pip install py-spy) and ptrace rights; in a container that
# usually means running as root or with SYS_PTRACE. --idle keeps samples of
# threads that are waiting (sleep, socket reads), which is the point here: we
# want to know whether client-side time is computing or waiting.
set -euo pipefail
command -v py-spy >/dev/null || { echo "py-spy not found: uv pip install py-spy" >&2; exit 127; }
out="${RS_TRACE_DIR:-results/traces}/pyspy_${RS_EPISODE_ID:-episode}.txt"
exec py-spy record --format raw --rate "${PYSPY_RATE:-50}" --subprocesses --idle \
  --output "$out" -- python -m rsched.launch "$@"
