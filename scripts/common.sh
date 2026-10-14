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

: "${VLA_PORT:=9000}"
: "${SAM3_PORT:=9001}"
: "${CUDA_DEVICE:=0}"
: "${LIBERO_TYPE:=pro}"
export VLA_PORT SAM3_PORT CUDA_DEVICE LIBERO_TYPE

SERVICE_DIR="$RS_ROOT/results/services"
mkdir -p "$SERVICE_DIR"

log() { printf '\033[1;34m[rsched]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[rsched] %s\033[0m\n' "$*" >&2; exit 1; }
