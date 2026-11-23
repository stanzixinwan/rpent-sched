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

# cuDNN 9.28 in this container segfaults on Thor in Pi0.5's first conv (SigLIP
# patch embedding, fp32 and bf16 alike); PyTorch's own conv kernels work.
# env/thor/rs_thor_cudnn.py (loaded by a .pth in both venvs) turns cuDNN off as
# torch loads it when RS_DISABLE_CUDNN=1 (env/.env).
for v in .venv .venv-molmo; do
  sp=$($v/bin/python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
  rm -f "$sp/sitecustomize.py"
  cp env/thor/rs_thor_cudnn.py "$sp/rs_thor_cudnn.py"
  echo "import rs_thor_cudnn" > "$sp/rs_thor_cudnn.pth"
done
add RS_DISABLE_CUDNN 1
add RS_GPU_SAMPLER tegra
RS_DISABLE_CUDNN=1 .venv/bin/python -c "import torch; assert not torch.backends.cudnn.enabled; c=torch.nn.Conv2d(3,8,14,stride=14).cuda(); c(torch.randn(1,3,224,224,device='cuda')); torch.cuda.synchronize(); print('conv ok without cuDNN')"
log "done. Check: python -m pytest tests -q; bash scripts/launch_services.sh"
