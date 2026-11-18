# Sourced by the other scripts: repo root, venv, and env/.env.
RS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RS_ROOT
cd "$RS_ROOT"

if [[ -f "$RS_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$RS_ROOT/.venv/bin/activate"
fi

if [[ -f "$RS_ROOT/env/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$RS_ROOT/env/.env"
  set +a
fi

# RS_PIN=1 (env/.env): one CPU layout for every experiment on this machine.
# Model servers, proxies and the GPU sampler get RS_SERVICE_CORES CPUs (default 8),
# episodes and their simulators RS_EPISODE_CORES (default 24); see rsched/cpu_layout.py.
if [[ "${RS_PIN:-0}" == 1 && -z "${RS_EPISODE_CPUS:-}" ]]; then
  _layout="$(cd "$RS_ROOT" && python -m rsched.cpu_layout --service-cores "${RS_SERVICE_CORES:-8}" \
    --episode-cores "${RS_EPISODE_CORES:-24}" --shell)" || { echo "[rsched] CPU layout failed" >&2; exit 1; }
  eval "export $_layout"
  unset _layout
fi

: "${VLA_PORT:=9000}"
: "${SAM3_PORT:=9001}"
: "${MOLMO_PORT:=9002}"
: "${CUDA_DEVICE:=0}"
: "${LIBERO_TYPE:=pro}"
export VLA_PORT SAM3_PORT MOLMO_PORT CUDA_DEVICE LIBERO_TYPE

SERVICE_DIR="$RS_ROOT/results/services"
mkdir -p "$SERVICE_DIR"

log() { printf '\033[1;34m[rsched]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[rsched] %s\033[0m\n' "$*" >&2; exit 1; }
