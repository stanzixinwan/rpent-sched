# rsched: Look Before You Act

Tracing and planner-aware scheduling for concurrent embodied agents.
COMS 6998 AI-Native Computing (Fall 2026), Zixin Wan.

[RPent](https://github.com/RLinf/RPent) runs an LLM planner that calls embodied
tools backed by shared GPU services (Pi0.5 VLA, SAM3). This repo measures where
episode time goes as concurrency grows, and tests whether the planner's
**pre-execution intent** (which primitive, with what arguments, e.g. `max_chunks`)
predicts downstream VLA demand better than the history-based estimates used by
current VLA serving systems.

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

**CPU threads.** In a container, `os.cpu_count()` reports the host (e.g. 128)
while the cgroup grants far fewer CPUs, so every process sizes its OpenMP/BLAS
pools for cores it does not have. In G2 this made `env.step` 60x slower at N=4.
Configs therefore set `episode_env` (thread caps for each episode and its
`env_server`), and `launch_services.sh` caps each service with `SERVICE_THREADS`
(default 4; 2 on a 13.6-CPU pod). Each manifest records the cgroup CPU quota
and throttle counters at level start and end (`cpu_start`, `cpu_end`).

`breakdown.py` splits each episode's wall time into components that sum to it:

| component | meaning |
|---|---|
| `startup` | launch until first tool call or model RPC (simulator spawn, memory, first plan) |
| `planner` | time outside tools and RPCs (LLM time; for Flash, replay logic) |
| `vla_wait`, `vla_service` | VLA requests queued at the proxy / running at the backend |
| `sam3_wait`, `sam3_service` | same for SAM3 |
| `molmo_wait`, `molmo_service` | same for Molmo grounding (Flash only) |
| `sim` | simulator RPCs made inside tools |
| `rpc_overhead` | client RPC time not explained by wait + service |
| `tool_other` | remaining tool time (scripted control, Python) |

Outputs: `episodes.csv`, `levels.csv`, `tools.csv`, `breakdown.png`,
`vla_wait_cdf.png`. Proxy policies: `passthrough` (RPent's own unordered
behaviour), `fifo`, `priority` with aging (reads the `X-RS-Priority` header).

## Intent scheduling (G3)

A Flash plan states, before the episode starts, every primitive it will run,
each pick's chunk budget (`max_chunks`), and every anchor Molmo will ground.
`rsched/intent.py` counts that work down as the episode runs and prices it in
seconds (per-primitive costs from G2 at N=1). With `scheduling.mode` (or
`--sched`) set, every model request carries the estimate as `X-RS-Priority`,
and the VLA and Molmo proxies (`policy: priority`) serve the smallest first:

| mode | order |
|---|---|
| `none` | arrival order (FIFO; no header) |
| `intent_srpt` / `intent_lrpt` | least / most remaining work in the plan first |
| `history_srpt` / `history_lrpt` | same, but remaining = mean work of episodes already finished in this run minus work done |

```bash
bash scripts/launch_services.sh
bash scripts/run_g3.sh                       # modes x N=8,16 x 3 repeats; resumable
python analysis/g3_compare.py results/g3/<stamp>
```

`g3_compare.py` compares each mode with FIFO at the same N and repeat
(`summary.csv`, `g3_compare.png`), drops cells whose CPU throttling exceeds 5%
of the makespan, and scores how well each estimate ranks episodes by their
true remaining time (`predict.csv`). Episode traces log `plan_work` once and a
`remaining` record at every tool start, in every mode. `analysis/intent.py`
now fits on other tasks (`--holdout task`, the default), not just other episodes.

## Launch order (G4)

For a fixed batch, `launch.rule` decides which episode enters a free slot,
from each episode's planned work (the Flash plan priced with `rsched/intent.py`,
known before launch): `given` (list order; `launch.shuffle` randomizes it per
repeat), `lpt`, `spt`, `front`, `front_back`, or `jit_lpt`, which starts a long
episode only once it would otherwise become the batch's tail. The rules live in
`rsched/launch_order.py`, shared with the trace-driven simulator
(`analysis/g3_sim.py launch`), which predicts about -14% makespan for 32
episodes at N=8. Plans come from RPent's memory sync unless `launch.plans_dir`
is set.

```bash
bash scripts/launch_services.sh
bash scripts/run_launch.sh                   # warm-up, then given vs jit_lpt x 4 repeats; resumable
python analysis/launch_compare.py results/launch/<stamp>
```

## Layout

```
rsched/proxy.py     tracing + scheduling proxy
rsched/launch.py    `python -m rsched.launch <rpent args>`: rpent with tracing hooks
rsched/sweep.py     concurrency sweeps, preflight, fail-fast, manifests
rsched/intent.py    remaining work from a Flash plan; scheduling priorities
rsched/launch_order.py  launch-order rules (sweep and simulator)
rsched/nvml.py      GPU memory / utilization sampler
analysis/           time breakdown, intent prediction, G3 mode comparison
configs/            one YAML per experiment
scripts/            services, smoke test, G3 driver
env/                setup script, .env template
tests/              CPU tests on RPent's real RPC classes with fake services
docs/               RPent notes, troubleshooting
third_party/RPent   pinned upstream (read-only)
```

## Docs

- [`docs/rpent_notes.md`](docs/rpent_notes.md): RPent behaviour this project relies on, with source references
- [`docs/troubleshooting.md`](docs/troubleshooting.md): WSL2, missing libraries, planner credentials
