#!/usr/bin/env bash
# Inside the Thor container (bash env/thor/run.sh): the same environment as
# env/setup.sh + env/setup_pod.sh, but PyTorch is the container's (sm_110 build).
# Safe to rerun. Writes .venv and .venv-molmo, downloads checkpoints, fills env/.env.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
log() { printf '\033[1;34m[thor-setup] %s\033[0m\n' "$*"; }

python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
git submodule update --init --recursive

# never let a dependency replace the container's torch stack
cat > .thor-overrides.txt <<'OV'
torch ; sys_platform == 'never'
torchvision ; sys_platform == 'never'
torchaudio ; sys_platform == 'never'
triton ; sys_platform == 'never'
OV

if [[ ! -d .venv ]]; then
  log "creating .venv (container Python, system site packages for torch)"
  uv venv --python "$(command -v python)" --system-site-packages .venv
fi
source .venv/bin/activate
log "rsched + RPent[libero-pro] (torch from the container)"
uv pip install --overrides .thor-overrides.txt -e ".[test,gpu]" "huggingface_hub>=0.34,<1.0" modelscope
uv pip install --overrides .thor-overrides.txt -e "third_party/RPent[libero-pro]"
python -c "import torch, mujoco; print('venv torch', torch.__version__, 'mujoco', mujoco.__version__)"
liberopro-download-assets --skip-existing
deactivate

if [[ ! -d .venv-molmo ]]; then
  log "creating .venv-molmo"
  uv venv --python "$(command -v python)" --system-site-packages .venv-molmo
fi
uv pip install --python .venv-molmo/bin/python --overrides .thor-overrides.txt \
  -e "third_party/RPent[molmo]" "transformers==4.57.1" einops accelerate
.venv-molmo/bin/python -c "import torch, transformers; print('molmo venv', torch.__version__, transformers.__version__)"

mkdir -p checkpoints
source .venv/bin/activate
[[ -d checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT ]] || hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
  --exclude optimizer.pt --local-dir checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
[[ -f checkpoints/sam3/sam3.pt ]] || modelscope download --model facebook/sam3 sam3.pt --local_dir checkpoints/sam3
[[ -d checkpoints/Molmo2-8B ]] || hf download allenai/Molmo2-8B --local-dir checkpoints/Molmo2-8B

[[ -f env/.env ]] || cp env/.env.example env/.env
add() { grep -q "^$1=" env/.env || echo "$1=$2" >> env/.env; }
add MOLMO_CHECKPOINT_PATH '${RS_ROOT}/checkpoints/Molmo2-8B'
add MOLMO_PYTHON '${RS_ROOT}/.venv-molmo/bin/python'
add SERVICE_THREADS 2
add CUDA_DEVICE 0
log "done. Check: python -m pytest tests -q; bash scripts/launch_services.sh"
