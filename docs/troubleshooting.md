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
