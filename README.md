# rsched — Tool-Context-Aware Scheduling for Concurrent Embodied Agents

COMS 6998 AI-Native Computing (Fall 2026) project, Zixin Wan.

Many [RPent](https://github.com/RLinf/RPent) episodes share one Pi0.5 VLA and one
SAM3 service. This repo measures where episode time goes as concurrency grows,
and (later) schedules shared-service requests using the tool context the LLM
planner writes before each call.

```
 rpent episode ──(rsched.launch hooks)──► rsched proxy ──► Pi0.5 VLA service
 rpent episode ──────────────────────────►  (trace +   ──► SAM3 service
 rpent episode ──────────────────────────►   schedule)
       │ per-episode env_server (simulator), traced client-side
       ▼
 results/<sweep>/N<k>_r<rep>/  traces ─► analysis/breakdown.py ─► CSVs + figures
```

RPent is pinned as a submodule (`third_party/RPent`, commit `ef70215`) and is
**not modified**: all tracing happens through a proxy and runtime hooks.

## Quick start

Under WSL2, clone into the Linux filesystem (e.g. `~/rpent-sched`), **not** under
`/mnt/c` or `/mnt/d`: Windows drives break git's file modes and line endings
(the RPent submodule checkout fails) and are much slower.

Requirements: Linux (or WSL2 Ubuntu) with an NVIDIA GPU and CUDA 12 driver,
`git`, a C/C++ toolchain, and [uv](https://docs.astral.sh/uv/). A planner API
key (Anthropic for `claude_code`, OpenAI for `codex`).

```bash
git clone --recursive <this repo> rpent-sched && cd rpent-sched
bash env/setup.sh                 # RPent + LIBERO-PRO + checkpoints (long)
$EDITOR env/.env                  # add API key; check ports and CUDA device
bash scripts/smoke_test.sh        # CPU: proves proxy/hooks/analysis (~1 min)
bash scripts/launch_services.sh   # load Pi0.5 + SAM3 once (minutes)
bash scripts/smoke_test.sh --gpu  # one real episode end to end
```

No GPU (laptop / Mac, for development): `bash env/setup.sh --no-gpu`, then
`uv pip install -e third_party/RPent` (RPent core only) and
`bash scripts/smoke_test.sh`.

## P1 baseline measurement

```bash
bash scripts/launch_services.sh
python -m rsched.sweep configs/p1_fifo_sweep.yaml          # N = 1,2,4,8,16
python analysis/breakdown.py results/p1_fifo_sweep/<stamp>
```

`breakdown.py` splits each episode's wall-clock time into components that add up:

| component | meaning |
|---|---|
| `startup` | launch until first tool call (simulator spawn + first plan) |
| `planner` | gaps between tool calls (LLM time) |
| `vla_wait`, `vla_service` | VLA requests queued at the proxy / running at the backend |
| `sam3_wait`, `sam3_service` | same for segmentation |
| `sim` | simulator RPCs (`env_server`) made inside tools |
| `rpc_overhead` | client RPC time not explained by wait + service (serialization, HTTP) |
| `tool_other` | the rest of tool time (scripted control, Python) |

Outputs: `episodes.csv`, `levels.csv` (success/hour, latency P50/P95, component
fractions, VLA busy fraction, GPU memory peak), `tools.csv` (per-tool duration
and `chunks_used` statistics), `breakdown.png`, `vla_wait_cdf.png`.

To compare against RPent's unmodified ordering, run
`configs/p1_passthrough_sweep.yaml` (the proxy only observes).

## Layout

```
rsched/proxy.py     tracing + scheduling proxy (policies: passthrough, fifo, priority+aging)
rsched/launch.py    `python -m rsched.launch <rpent args>`: rpent with tracing hooks
rsched/sweep.py     concurrency sweeps (closed loop or Poisson), per-level manifest
rsched/nvml.py      GPU memory / utilization sampler
analysis/           time breakdown and figures
configs/            experiment configs (one file per experiment)
scripts/            setup, services, smoke test
tests/              CPU tests using fake services built on RPent's real RPC classes
third_party/RPent   pinned upstream
```

Every level directory has a `manifest.json` with the config, git commits of this
repo and RPent, host, Python version, GPU model and driver, and start/end times.

## What RPent's code implies for this project

Read from the pinned commit; each point shapes a design choice here.

1. **The VLA is serialized, but not FIFO.** `pi05_vla_server` runs on
   `ThreadingHTTPServer`, and every `vla.predict` takes `RpcFacade`'s exclusive
   write lock (`RWLock.write`). Waiters are woken by `notify_all`, so the next
   request served is whichever thread wins the race. RPent's "default" is
   therefore unordered; the `fifo` policy makes the baseline well defined, and
   `passthrough` measures the unordered default.
2. **One `vla.predict` = one action chunk** (5 actions for LIBERO). A VLA
   primitive (`pi0_pick`, `pi0_doubled`) loops chunk by chunk until a success
   condition or `max_chunks` (default 24 / 20). **The planner writes
   `max_chunks` and the prompt before the primitive starts**: an explicit upper
   bound on the call's demand and the core "tool context" signal.
3. **SAM3 segmentation is concurrent**: `sam3.segment` is registered as a
   read-only method, so requests run in parallel on the GPU.
4. **Batching is available at the model level**: Pi0.5's
   `predict_action_batch` takes a batch dimension and the client sends `[1, ...]`
   tensors, so the batched-serving baseline can merge requests in the proxy.
5. **Most LIBERO tools never touch the GPU**: `move_to`, `rotate_*`, `release`,
   `set_gripper` drive the simulator directly.
6. **One funnel for all planners**: Claude Code, Codex, and the API planner all
   call `Toolkit.execute_tool` in the rpent process (via `asyncio.to_thread`), so
   one hook sees every tool call with its arguments.
7. Client timeouts: `vla.predict` 120 s, `sam3.segment` 120 s. Very high
   concurrency can hit these; the proxy logs every failure.

## Roadmap (course milestones)

- **P1 (Oct 19)** — this scaffold; FIFO sweep N=1..16; first breakdown.
- **P2 (Nov 2)** — trace set across suites; predict per-primitive VLA calls from
  tool context vs. tool-type mean and history, on held-out suites; oracle
  headroom by replay.
- **P3 (Nov 23)** — predictor sets `X-RS-Priority` (hook point in
  `rsched/launch.py`); `priority` policy with aging; batched-serving baseline;
  admission control in `sweep.py`.

## WSL2

WSL2 gets half of Windows RAM by default. Pi0.5 is built in fp32 on the CPU
before moving to the GPU (~13 GB peak), so the kernel can kill it during
startup (`Killed` in `launch_services.sh`). Raise the limits in
`%UserProfile%\.wslconfig` on Windows:

```ini
[wsl2]
memory=26GB
swap=32GB
```

Then run `wsl --shutdown` in PowerShell, reopen Ubuntu, and check `free -g`.
Close large Windows programs (browsers, IDEs) while services load; after
loading, host RAM use drops back.

## Hardware notes

A 16 GB laptop GPU can run one episode for development; WSL2 needs the Windows
NVIDIA driver with CUDA on WSL, and MuJoCo's EGL rendering (`MUJOCO_GL=egl`) may
need extra setup there. Measurements should run on a dedicated 48–80 GB GPU:
laptop GPUs throttle, which adds noise to exactly the timings this project measures.
