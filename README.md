# rsched: Look Before You Act

Tracing and planner-aware scheduling for concurrent embodied agents.
COMS 6998 AI-Native Computing (Fall 2026), Zixin Wan.

[RPent](https://github.com/RLinf/RPent) runs an LLM planner that calls embodied
tools backed by shared GPU services (Pi0.5 VLA, SAM3). This repo measures where
episode time goes as concurrency grows, and tests whether the planner's
**pre-execution intent** (which primitive, with what arguments, e.g. `max_chunks`)
predicts downstream VLA demand better than the history-based estimates used by
current VLA serving systems.

Current status, decisions, and next steps: [`docs/STATUS.md`](docs/STATUS.md).

```
 rpent episode ──(rsched.launch hooks)──► rsched proxy ──► Pi0.5 VLA service
 rpent episode ──────────────────────────►  (trace +   ──► SAM3 service
 rpent episode ──────────────────────────►   schedule)
       │ per-episode env_server (simulator), traced client-side
       ▼
 results/<sweep>/N<k>_r<rep>/  traces ─► analysis/breakdown.py ─► CSVs + figures
```

RPent is a pinned submodule (`third_party/RPent`, commit `ef70215`) and is
**not modified**. All tracing goes through the proxy and runtime hooks.

## Quick start

Requirements: Linux or WSL2 Ubuntu, NVIDIA GPU with a CUDA 12 driver, `git`,
a C/C++ toolchain, [uv](https://docs.astral.sh/uv/). A planner API key only for
live LLM-planner runs.

```bash
git clone --recursive https://github.com/stanzixinwan/rpent-sched.git && cd rpent-sched
bash env/setup.sh                 # RPent + LIBERO-PRO + checkpoints (long)
$EDITOR env/.env                  # API key, ports, CUDA device
bash scripts/smoke_test.sh        # CPU only: proxy, hooks, sweep, analysis (~1 min)
bash scripts/launch_services.sh   # load Pi0.5, then SAM3 (minutes)
bash scripts/smoke_test.sh --gpu  # one real episode end to end
```

Without a GPU: `bash env/setup.sh --no-gpu && uv pip install -e third_party/RPent`,
then `bash scripts/smoke_test.sh`.

## Running a sweep

```bash
bash scripts/launch_services.sh
python -m rsched.sweep configs/p1_laptop.yaml      # see configs/ for others
python analysis/breakdown.py results/p1_laptop/<stamp>
```

`rsched.sweep` loads `env/.env`, checks planner credentials with
`rpent-check-llm`, then keeps N episodes in flight per level. It aborts at once
on planner billing/auth errors and after two consecutive episodes with no tool
call. Each level writes `manifest.json` (config, commits, host, GPU, timings).
Configs cap planner spend per episode (`--claude-code-max-budget-usd`).

`breakdown.py` splits each episode's wall time into components that sum to it:

| component | meaning |
|---|---|
| `startup` | launch until first tool call (simulator spawn, memory, first plan) |
| `planner` | gaps between tool calls (LLM time) |
| `vla_wait`, `vla_service` | VLA requests queued at the proxy / running at the backend |
| `sam3_wait`, `sam3_service` | same for SAM3 |
| `sim` | simulator RPCs made inside tools |
| `rpc_overhead` | client RPC time not explained by wait + service |
| `tool_other` | remaining tool time (scripted control, Python) |

Outputs: `episodes.csv`, `levels.csv`, `tools.csv`, `breakdown.png`,
`vla_wait_cdf.png`. Proxy policies: `passthrough` (RPent's own unordered
behaviour), `fifo`, `priority` with aging (reads the `X-RS-Priority` header).

## First measurements

LIBERO-PRO, Claude Code planner, RTX 3080 laptop (WSL2), N = 1 and 2, six valid
episodes. Mean episode 547 s (171 to 1069 s).

| planner | simulator | startup | other tool | VLA | SAM3 | VLA queueing |
|---|---|---|---|---|---|---|
| 57% | 20% | 14% | 5% | 2.7% | 1.7% | ~0 |

VLA busy 2.5% (N=1) to 7.6% (N=2) of wall time; 465 ms per `vla.predict` on the
laptop. Planner cost $1 to 9 per episode. With a per-step LLM planner the
bottleneck is the planner, not the VLA. See `docs/STATUS.md` for what follows.

## Layout

```
rsched/proxy.py     tracing + scheduling proxy
rsched/launch.py    `python -m rsched.launch <rpent args>`: rpent with tracing hooks
rsched/sweep.py     concurrency sweeps, preflight, fail-fast, manifests
rsched/nvml.py      GPU memory / utilization sampler
analysis/           time breakdown and figures
configs/            one YAML per experiment
scripts/            services, smoke test
env/                setup script, .env template
tests/              CPU tests on RPent's real RPC classes with fake services
docs/STATUS.md      project status, decisions, gates, known issues
third_party/RPent   pinned upstream (read-only)
```

## Facts from RPent's code (commit ef70215)

1. **The VLA is serialized but not FIFO.** Every `vla.predict` takes
   `RpcFacade`'s exclusive write lock; waiters wake via `notify_all` and race.
   The proxy's `fifo` policy makes the baseline well defined.
2. **One `vla.predict` = one action chunk** (5 actions on LIBERO).
   `pi0_pick` / `pi0_doubled` loop chunks until success or `max_chunks`
   (default 24 / 20), which the planner writes before the primitive starts.
3. **SAM3 is concurrent** (`sam3.segment` is a read-only method).
4. **Pi0.5 batches natively** (`predict_action_batch`); clients send batch 1.
5. **Scripted tools** (`move_to`, `move_pose`, `rotate_*`, `release`,
   `set_gripper`) use only the simulator.
6. **One funnel for all planners**: `Toolkit.execute_tool`.
7. **Flash Mode** (`--planner flash`) replays stored plans with no LLM; Molmo-8B
   (separate venv) localizes objects, then VLA and SAM3 execute. 78 published
   plans for LIBERO-PRO.

## Troubleshooting

- **`Killed` while loading services (WSL2).** Pi0.5 is built in fp32 on the CPU
  first (~13 GB peak). In `%UserProfile%\.wslconfig` set `memory=26GB` and
  `swap=32GB` under `[wsl2]`, run `wsl --shutdown`, check `free -g`.
- **`ImportError: libGL.so.1` from `env_server`.**
  `sudo apt install -y libgl1 libegl1 libglib2.0-0` (setup checks for it).
- **`Not logged in` / `authentication_failed`.** Put the key in `env/.env`.
- **Clone location on WSL.** Use the Linux filesystem (`~/`), not `/mnt/c`.
- **GPU memory numbers on WSL2** from `nvidia-smi` are unreliable; measure on
  native Linux. Laptop GPUs also throttle, so treat laptop timings as development data.
