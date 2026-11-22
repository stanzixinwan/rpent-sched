#!/usr/bin/env bash
# Open a shell in the Thor container with the repo, checkpoints and results mounted.
#   bash env/thor/run.sh            # interactive shell (start tmux inside)
#   bash env/thor/run.sh <cmd ...>  # run one command
# The repo lives on the host at its own path (default /data/stan/rpent-sched) and is
# mounted at the same path, so result paths are identical inside and outside.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${THOR_IMAGE:-rpent-thor}"
args=(--rm --runtime nvidia --gpus all --ipc=host --network host
      --ulimit memlock=-1 --ulimit stack=67108864
      -v "$REPO:$REPO" -w "$REPO" -e HOME="$REPO/.thor-home"
      --name "rpent-thor-$USER-$$")
# tegrastats and power rails for energy measurements
[[ -e /usr/bin/tegrastats ]] && args+=(-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro --privileged)
mkdir -p "$REPO/.thor-home"
if (( $# )); then exec sudo docker run "${args[@]}" "$IMAGE" "$@"; fi
exec sudo docker run -it "${args[@]}" "$IMAGE" bash
