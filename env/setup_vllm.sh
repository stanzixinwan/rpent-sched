#!/usr/bin/env bash
# Plan B: a local VLM planner served by vLLM on the same GPU as Pi0.5 and Molmo.
# Run after env/setup_pod.sh. Safe to rerun.
#
#   bash env/setup_vllm.sh                              # Qwen2.5-VL-7B-Instruct (~17 GB)
#   LLM_HF_ID=Qwen/Qwen2.5-VL-3B-Instruct bash env/setup_vllm.sh
#
# vLLM gets its own venv: it pins its own torch, which the LIBERO venv must not see.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PATH="$HOME/.local/bin:$PATH"
unset HF_HUB_ENABLE_HF_TRANSFER
log() { printf '\033[1;34m[setup_vllm] %s\033[0m\n' "$*"; }

HF_ID="${LLM_HF_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
DIR="$ROOT/checkpoints/$(basename "$HF_ID")"

if [[ ! -x .venv-vllm/bin/vllm ]]; then
  log "creating .venv-vllm and installing vLLM (pulls its own torch; a few minutes)"
  uv venv --python 3.11 .venv-vllm
  uv pip install --python .venv-vllm/bin/python vllm
fi
.venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

if [[ ! -f "$DIR/config.json" ]]; then
  log "downloading $HF_ID"
  .venv/bin/hf download "$HF_ID" --local-dir "$DIR"
fi

set_env() {
  if grep -q "^$1=" env/.env; then sed -i "s|^$1=.*|$1=$2|" env/.env; else echo "$1=$2" >> env/.env; fi
}
[[ -f env/.env ]] || cp env/.env.example env/.env
set_env LLM_MODEL "$DIR"
set_env VLLM_BIN "$ROOT/.venv-vllm/bin/vllm"
grep -q "^LLM_PORT=" env/.env || set_env LLM_PORT 9003
grep -q "^LLM_GPU_UTIL=" env/.env || set_env LLM_GPU_UTIL 0.32
grep -E "^(LLM_|VLLM_BIN)" env/.env
log "done. Start it with: bash scripts/launch_llm.sh"
