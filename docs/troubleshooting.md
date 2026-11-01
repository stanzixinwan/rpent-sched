# Troubleshooting

- **`Killed` while loading services (WSL2).** Pi0.5 is built in fp32 on the CPU
  first (~13 GB peak). In `%UserProfile%\.wslconfig` set `memory=26GB` and
  `swap=32GB` under `[wsl2]`, run `wsl --shutdown`, check `free -g`.
- **`ImportError: libGL.so.1` from `env_server`.**
  `sudo apt install -y libgl1 libegl1 libglib2.0-0` (setup checks for it).
- **`Not logged in` / `authentication_failed`.** Put the key in `env/.env`.
- **Clone location on WSL.** Use the Linux filesystem (`~/`), not `/mnt/c`.
- **GPU memory numbers on WSL2** from `nvidia-smi` are unreliable; measure on
  native Linux. Laptop GPUs also throttle, so treat laptop timings as development data.
- **`No module named 'einops'` (or another module) when Molmo starts.** RPent's
  `[molmo]` extra omits packages that Molmo2's remote code imports. Install them
  into `.venv-molmo`: `uv pip install --python .venv-molmo/bin/python einops`.
- **`Unexpected keyword argument image_use_col_tokens` when Molmo starts.**
  `[molmo]` allows transformers v5; Molmo2's remote code needs 4.57 (its model
  card pins `transformers==4.57.1`). Pin it in `.venv-molmo`.
- **`hf_transfer` package is not available** during setup or `hf download`.
  The image sets `HF_HUB_ENABLE_HF_TRANSFER=1`; `unset HF_HUB_ENABLE_HF_TRANSFER`
  (setup.sh does this itself).
- **Episodes get slower far faster than N grows; GPU mostly idle.** CPU thread
  oversubscription in a container. Keep `episode_env` thread caps in the config
  and `SERVICE_THREADS` in `env/.env`; check `nr_throttled` in the manifest.
- **No `/sys/fs/cgroup/cpu.max` (RunPod).** The host uses cgroup v1; the quota is
  `cpu/cpu.cfs_quota_us` / `cpu.cfs_period_us`. The sweep reads both. The quota
  can be below the advertised vCPU count (one "16 vCPU" pod had 13.6).
- **`EGLError` tracebacks when the MuJoCo EGL check exits.** Harmless: the GL
  context is freed after Python has torn down OpenGL. The check passed if it
  printed `EGL ok`.
- **No network volume on RunPod.** Container disk is wiped when the pod stops;
  copy `results/` off before stopping. With a network volume mounted at
  `/workspace`, keep uv's Python, uv cache, and `HF_HOME` there too.
