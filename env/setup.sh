#!/usr/bin/env bash
# One-time setup on a Linux (or WSL2 Ubuntu) machine with an NVIDIA GPU.
#
#   bash env/setup.sh            # full install: RPent + LIBERO-PRO + checkpoints
#   bash env/setup.sh --no-gpu   # rsched only (proxy, analysis, CPU tests)
#
# Requirements (from RPent's quick start): Linux, NVIDIA GPU with a CUDA 12
# driver, git, a C/C++ toolchain, and uv (https://docs.astral.sh/uv/).
set -euo pipefail
source "$(dirname "$0")/../scripts/common.sh"

NO_GPU=0
[[ "${1:-}" == "--no-gpu" ]] && NO_GPU=1

if [[ "$RS_ROOT" == /mnt/* ]]; then
  die "repo is on a Windows drive ($RS_ROOT). Under WSL, clone into the Linux filesystem instead,
  e.g.  cd ~ && git clone --recursive https://github.com/stanzixinwan/rpent-sched.git
  (/mnt/* breaks git file modes and line endings, and is much slower for pip and MuJoCo)"
fi

command -v uv >/dev/null || die "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"

log "submodules (RPent pinned at $(git -C third_party/RPent rev-parse --short HEAD 2>/dev/null || echo '?'))"
git submodule update --init --recursive

if [[ ! -d .venv ]]; then
  log "creating .venv (Python 3.11)"
  uv venv --python 3.11
fi
source .venv/bin/activate

log "installing rsched"
uv pip install -e ".[test]"

if [[ $NO_GPU -eq 1 ]]; then
  log "done (no-GPU mode). Run: bash scripts/smoke_test.sh"
  exit 0
fi

# System libraries the simulator stack loads at import time (cv2 needs libGL;
# MuJoCo offscreen rendering needs EGL). Minimal Ubuntu / WSL images lack them.
missing=()
for lib in libGL.so.1 libEGL.so.1 libglib-2.0.so.0; do
  ldconfig -p | grep -q "$lib" || missing+=("$lib")
done
if (( ${#missing[@]} )); then
  die "missing system libraries: ${missing[*]}
  install with:  sudo apt update && sudo apt install -y libgl1 libegl1 libglib2.0-0"
fi

command -v nvidia-smi >/dev/null || die "nvidia-smi not found; install the NVIDIA driver (WSL2: Windows driver + CUDA on WSL)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "installing RPent with LIBERO-PRO extras (this pulls torch, openpi, LIBERO; takes a while)"
uv pip install -e "third_party/RPent[libero-pro]"
uv pip install -e ".[gpu]" "huggingface_hub>=0.34,<1.0" modelscope

log "LIBERO-PRO assets"
liberopro-download-assets --skip-existing

mkdir -p checkpoints
if [[ ! -d checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT ]]; then
  log "Pi0.5 checkpoint"
  hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --exclude optimizer.pt \
    --local-dir checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
fi
if [[ ! -f checkpoints/sam3/sam3.pt ]]; then
  log "SAM3 checkpoint"
  modelscope download --model facebook/sam3 sam3.pt --local_dir checkpoints/sam3
fi

if [[ ! -f env/.env ]]; then
  cp env/.env.example env/.env
  log "created env/.env -- add your planner API key there"
fi

log "setup complete. Next: bash scripts/launch_services.sh && bash scripts/smoke_test.sh --gpu"
