#!/usr/bin/env bash
# Everything a fresh cloud GPU pod (RunPod PyTorch template, root, A100 80GB)
# needs before a sweep, in one go. Safe to rerun: finished steps are skipped.
#
#   cd /workspace && git clone --recursive https://github.com/stanzixinwan/rpent-sched.git
#   cd rpent-sched && bash env/setup_pod.sh            # ~1-1.5 h, mostly downloads
#   bash env/setup_pod.sh --launch                     # same, then start the services
#
# Steps: system packages and uv -> HF_HUB_ENABLE_HF_TRANSFER off for good ->
# env/setup.sh (RPent, LIBERO-PRO, Pi0.5, SAM3) -> MuJoCo EGL check -> Molmo
# venv and Molmo2-8B -> env/.env (Molmo paths, SERVICE_THREADS) -> uv cache
# cleanup -> summary (CPU quota, GPU, disk). See docs/troubleshooting.md.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LAUNCH=0
[[ "${1:-}" == "--launch" ]] && LAUNCH=1

step() { printf '\n\033[1;34m[setup_pod] %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[setup_pod] %s\033[0m\n' "$*" >&2; exit 1; }
SUDO=""; [[ $(id -u) -ne 0 ]] && SUDO="sudo"

# Settings that must hold in every later shell, tmux included.
BASHRC_LINES=(
  'unset HF_HUB_ENABLE_HF_TRANSFER'   # RunPod sets it without the package: every HF download fails
  'export PATH="$HOME/.local/bin:$PATH"'
)

step "1/7 system packages and uv"
if ! command -v tmux >/dev/null || ! ldconfig -p | grep -q libEGL.so.1 || ! ldconfig -p | grep -q libGL.so.1; then
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq tmux git build-essential libgl1 libegl1 libglib2.0-0 >/dev/null
fi
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "2/7 shell settings (HF_HUB_ENABLE_HF_TRANSFER off in every new shell)"
for line in "${BASHRC_LINES[@]}"; do
  grep -qxF "$line" "$HOME/.bashrc" 2>/dev/null || echo "$line" >> "$HOME/.bashrc"
done
unset HF_HUB_ENABLE_HF_TRANSFER
avail_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
if (( avail_gb < 120 )); then
  die "only ${avail_gb} GB free under $ROOT; checkpoints and venvs need ~60 GB and each sweep level ~2-3 GB. Use a pod with a 250 GB container disk."
fi

step "3/7 RPent, LIBERO-PRO, Pi0.5, SAM3 (env/setup.sh)"
bash env/setup.sh

step "4/7 MuJoCo EGL check"
egl_out=$(MUJOCO_GL=egl .venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom size=\".1\"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); r.render()
print('EGL ok', flush=True)" 2>&1 || true)
if grep -q "EGL ok" <<<"$egl_out"; then
  echo "EGL ok (EGLError tracebacks at interpreter exit are harmless)"
else
  echo "$egl_out" | tail -15
  ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia* 2>/dev/null || true
  die "MuJoCo cannot render with EGL. If libEGL_nvidia.so exists, add the glvnd vendor file:
  mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
and rerun; if it does not exist, this pod exposes no NVIDIA graphics: use another pod."
fi

step "5/7 Molmo venv and Molmo2-8B"
if [[ ! -x .venv-molmo/bin/python ]] || ! .venv-molmo/bin/python -c "import transformers, einops, accelerate" 2>/dev/null; then
  uv venv --python 3.11 .venv-molmo
  uv pip install --python .venv-molmo/bin/python -e "third_party/RPent[molmo]" "transformers==4.57.1" einops accelerate
fi
if [[ ! -f checkpoints/Molmo2-8B/config.json ]]; then
  .venv/bin/hf download allenai/Molmo2-8B --local-dir checkpoints/Molmo2-8B
fi

step "6/7 env/.env"
set_env() {  # KEY VALUE: replace the line, or append it
  if grep -q "^$1=" env/.env; then
    sed -i "s|^$1=.*|$1=$2|" env/.env
  else
    echo "$1=$2" >> env/.env
  fi
}
[[ -f env/.env ]] || cp env/.env.example env/.env
set_env MOLMO_CHECKPOINT_PATH "$ROOT/checkpoints/Molmo2-8B"
set_env MOLMO_PYTHON "$ROOT/.venv-molmo/bin/python"
set_env SERVICE_THREADS "${SERVICE_THREADS:-2}"
grep -E "^(MOLMO_|SERVICE_THREADS|VLA_PORT|SAM3_PORT|CUDA_DEVICE)" env/.env

step "7/7 cleanup and summary"
uv cache clean >/dev/null 2>&1 || true
.venv/bin/python - <<'PY'
import os
from pathlib import Path
def read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None
quota = read("/sys/fs/cgroup/cpu.max")
if quota is None:
    q, per = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    quota = f"{q} {per}" if q and per else None
cpus = None
if quota and not quota.startswith("max") and not quota.startswith("-1"):
    q, per = quota.split()
    cpus = int(q) / int(per)
print(f"os.cpu_count={os.cpu_count()}  cgroup quota={quota}  -> usable CPUs: {cpus or 'unlimited'}")
if cpus and cpus < 12:
    print("  WARNING: under 12 usable CPUs; N=8 may be CPU-limited (check throttling in the manifests)")
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "$ROOT" | tail -1
du -sh checkpoints .venv .venv-molmo 2>/dev/null || true

if (( LAUNCH )); then
  step "starting services"
  bash scripts/launch_services.sh
  bash scripts/launch_services.sh status
fi

cat <<'EOF'

[setup_pod] done. Next, in a tmux session (tmux new -s run; reattach with tmux attach -t run):
  source .venv/bin/activate
  bash scripts/launch_services.sh && bash scripts/launch_services.sh status
  bash scripts/run_g5.sh                     # or another experiment script
EOF
