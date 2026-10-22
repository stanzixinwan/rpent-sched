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
docs/               status, RPent notes, troubleshooting
third_party/RPent   pinned upstream (read-only)
```

## Docs

- [`docs/STATUS.md`](docs/STATUS.md): research question, latest results, decisions, gates, known issues
- [`docs/rpent_notes.md`](docs/rpent_notes.md): RPent behaviour this project relies on, with source references
- [`docs/troubleshooting.md`](docs/troubleshooting.md): WSL2, missing libraries, planner credentials
