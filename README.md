# rsched: end-to-end tracing of harness-style embodied agents on a shared GPU

Tools, configurations and analysis for measuring where the time goes when many
episodes of an embodied-agent harness share one GPU. The harness under study is
[RPent](https://github.com/RLinf/RPent): a planner, or a plan an LLM wrote once,
calls a frozen VLA ($\pi_{0.5}$), a grounding VLM (Molmo2-8B) and a segmentation
model (SAM3) as tools inside a LIBERO-PRO simulator.

RPent is a pinned submodule (`third_party/RPent`, commit `ef70215`) and is **not
modified**. Everything is measured through runtime hooks and per-model proxies:

```
 episode ──(rsched.launch hooks)──► rsched.proxy ──► Pi0.5 server
 episode ─────────────────────────► (trace, queue, ──► Molmo2 server
 episode ─────────────────────────►  order, batch) ──► SAM3 server
    │ own env_server (MuJoCo), traced client-side
    ▼
 results/<experiment>/<stamp>/N<k>_r<rep>/   traces, proxy logs, GPU samples, manifest
    └─► analysis/*.py ─► CSVs ─► paper/figures/make_figures.py
```

## Requirements

- Linux (or WSL2 Ubuntu for development), an NVIDIA GPU with a CUDA 12 driver.
  The measurements use one A100-SXM4-80GB; Pi0.5, SAM3 and Molmo2 together take
  about 36 GB.
- `git`, a C/C++ toolchain, [uv](https://docs.astral.sh/uv/), about 250 GB of disk
  for checkpoints and assets.
- A planner API key only for live-planner runs (`configs/live_planner.yaml`).
- NVIDIA MPS (`nvidia-cuda-mps-control`) for the MPS experiments.
- `py-spy` with ptrace rights for the profiling run; `nvidia-ml-py` for energy.

## Setup

```bash
git clone --recursive <repository-url> rpent-sched && cd rpent-sched
bash env/setup.sh                 # RPent, LIBERO-PRO, checkpoints (long)
cp env/.env.example env/.env      # API key, ports, checkpoint paths, CUDA device
bash scripts/smoke_test.sh        # CPU only: proxy, hooks, sweep, analysis (~1 min)
bash scripts/launch_services.sh   # load Pi0.5, SAM3 and (with MOLMO_CHECKPOINT_PATH) Molmo2
bash scripts/smoke_test.sh --gpu  # one real episode end to end
```

On a fresh cloud GPU machine, `bash env/setup_pod.sh` does all of the above,
including the Molmo2 environment and the MuJoCo EGL check (`--launch` also
starts the services). Without a GPU: `bash env/setup.sh --no-gpu && uv pip
install -e third_party/RPent`, then `bash scripts/smoke_test.sh`. Tests:
`pytest tests` (CPU only; RPent's RPC classes against fake services).

## Running an experiment

```bash
bash scripts/launch_services.sh
python -m rsched.sweep configs/g2_flash.yaml          # N episodes in flight per level
python analysis/anatomy.py --set "N8=results/g2_flash/<stamp>/N8_r0" --out results/paper/anatomy
```

`rsched.sweep` keeps N episodes in flight from a fixed batch, starts a proxy per
model service and a GPU sampler (utilization, memory, power) per cell, and
writes a `manifest.json` per cell with the configuration, commits, GPU, cgroup
CPU quota and throttling counters. It checks planner credentials first, aborts
on billing or authentication errors, and stops after repeated episodes with no
tool call. Drivers under `scripts/run_*.sh` run whole experiments: they pair
arms on the same shuffled batch, rotate arm order between repeats, discard a
warm-up cell, and resume where they stopped.

Configuration knobs used by the experiments:

| knob | effect |
|---|---|
| `episode_env` | environment of every episode and its simulator (thread caps: see below) |
| `cpu.episode_cpus`, `--episode-cpus "8-23"` | pin episodes and their simulators to these cores |
| `SERVICE_CPUS=0-7` (`launch_services.sh`) | pin the model servers to their own cores |
| `services.<name>.policy` | proxy policy: `passthrough`, `fifo`, `priority` (with aging), `batch` |
| `services.<name>.proxy: false` | no proxy: episodes call the model server directly |
| `RS_TRACE=off` (in `episode_env`) | run RPent without hooks (overhead control) |
| `services.<name>.dump_per_episode` | the proxy saves the first request bodies of each episode (replayed by `rsched.bench_models`) |
| `scheduling.mode` | request priority from the plan's or the history's remaining work |
| `launch.rule`, `launch.estimate` | episode launch order (`given`, `lpt`, `jit_lpt`, ...) and its length estimate (`plan`, `history`, `hybrid`) |

**CPU threads.** In a container, `os.cpu_count()` reports the host while the
cgroup grants far fewer CPUs, so every process sizes its OpenMP/BLAS pools for
cores it does not have; this made `env.step` 60x slower in our runs. Configs
cap threads per episode (`episode_env`), `launch_services.sh` caps each service
(`SERVICE_THREADS`, default 4), and manifests record the quota and throttling.

## Time accounting

`analysis/anatomy.py` splits every episode's wall time into parts that add up to
it exactly (RPCs within an episode are sequential):

| component | meaning |
|---|---|
| `startup`, `teardown` | process start to first RPC; last RPC to exit |
| `sim_reset`, `sim_step`, `sim_chunk_step`, `sim_render`, `sim_other` | simulator RPCs by method |
| `local_tools`, `local_between` | client-side time between RPCs, inside a tool / between tools (plan replay or LLM) |
| `<model>_wait`, `<model>_svc` | model requests (vla, molmo, sam3): queued at the proxy / in service |

It also writes RPC counts and latencies per method, client-side time per tool,
and the split by task outcome. `analysis/breakdown.py` is the earlier
per-level summary (throughput, cost, queueing CDFs).

## Experiments

Results go to `results/<name>/<stamp>/` (not version-controlled). Paper
sections refer to `paper/main.tex`.

| id | question | config | driver | analysis | paper |
|---|---|---|---|---|---|
| G2 | episode anatomy vs. concurrency, N = 1, 4, 8, 16 | `g2_flash.yaml` (`g2_flash_uncapped.yaml`: thread-cap control) | `python -m rsched.sweep` | `anatomy.py`, `breakdown.py`, `intent.py` | §4 |
| G3 | ordering model requests by remaining work | `g3.yaml` | `run_g3.sh` | `g3_compare.py`, `g3_sim.py` (trace replay) | §7 |
| G4 | episode launch order, tuning tasks | `g4_launch.yaml` | `run_launch.sh` | `launch_compare.py` | §7 |
| G5 | launch order on held-out tasks; plan vs. history estimates | `g5_record.yaml`, `g5_heldout.yaml` | `run_g5.sh` | `launch_compare.py`, `holdout.py`, `history_baseline.py` | §4, §7 |
| G6 | cross-episode VLA batching on real episodes, under MPS | `g6_batch.yaml` | `run_g6.sh` | `launch_compare.py --baseline nobatch`, `anatomy.py` | §4, §7 |
| B1 | interference of co-located planner, VLA, grounding model | — | `run_b_feasibility.sh` (`rsched.bench_interference`) | `inflation.csv` | §6 |
| B1b | B1 under time-slicing and MPS in one session (same machine) | — | `run_colocation.sh` | `inflation.csv` | §6 |
| B2 | MPS vs. time-slicing; pausing the planner during VLA bursts | — | `run_b2a.sh`, `run_b2a_mps.sh` (`rsched.bench_phases`) | `summary.csv` | §6, §7 |
| B3 | Pi0.5 batch cost; batching on an agent-like synthetic workload | — | `run_b_batch.sh` (`rsched.bench_batch`), `run_b3.sh` | `batch.csv`, `summary.csv` | §7 |
| M1 | single models: RPC / host / GPU split, stages, prefill vs decode, operator mix, analytic roofline, energy, Pi0.5 batch scaling | `record_requests.yaml` | `run_model_profile.sh` (`rsched.profiled_server`, `rsched.bench_models`) | `model_profile.py` | §4 |
| X0 | cost of the tracing (hooks, proxies) | `overhead.yaml` | `run_overhead.sh` | `overhead.py` | §3 |
| X1 | the same tasks with an LLM planner in the loop | `live_planner.yaml` | `python -m rsched.sweep` | `anatomy.py` | §4 |
| X2 | concurrency × CPU budget (bottleneck migration, energy) | `cpu_scaling.yaml` | `run_cpu_scaling.sh` | `anatomy.py`, `gpu_util.py` | §5 |
| X3 | sampling profile of episodes and simulators | `profile.yaml` | `run_profile.sh` | `profile_summary.py` | §4 |
| X7 | real episodes with a local planner on the same GPU | `local_planner.yaml` | `launch_llm.sh`, then `python -m rsched.sweep` | `anatomy.py` | §6 |

Cross-experiment analysis: `analysis/bounds.py` (first-order bound vs. measured
effect of each lever), `analysis/gpu_util.py` (GPU busy fraction, memory,
energy per episode). Figures: `python paper/figures/make_figures.py`.

The co-location experiments (B1, B2, X7) serve a local VLM planner with vLLM
next to the robot models: `bash env/setup_vllm.sh` once, then
`bash scripts/launch_llm.sh`; `bash scripts/mps.sh start` before starting the
services switches to MPS. The proxy's `batch` policy merges concurrent
`vla.predict` calls into one batched call (`rsched/batching.py`; dispatch
`greedy`, `window`, or `predict`, optionally using the pick's `max_chunks`).

## Layout

```
rsched/proxy.py           tracing proxy: queueing, ordering, batching policies
rsched/batching.py        cross-episode batching of vla.predict (proxy --policy batch)
rsched/launch.py          python -m rsched.launch <rpent args>: rpent with tracing hooks
rsched/sweep.py           concurrency sweeps, CPU pinning, preflight, fail-fast, manifests
rsched/intent.py          remaining work from a stored plan; request priorities
rsched/launch_order.py    launch-order rules and length estimates (sweep and replay)
rsched/history_table.py   per-task runtime history from earlier runs
rsched/bench_*.py         microbenchmarks: interference, agent-like phases, batch cost, model replay
rsched/profiled_server.py RPent model server with per-call timing, stage hooks, torch.profiler
rsched/nvml.py            GPU utilization, memory and power sampler
rsched/molmo_server.py    Molmo2 server with a point parser for both output formats
analysis/                 per-episode accounting, experiment comparisons, bounds, replay
configs/                  one YAML per experiment
scripts/                  services, MPS, experiment drivers, smoke test
env/                      setup scripts, .env template
tests/                    CPU tests on RPent's RPC classes with fake services
paper/                    paper draft and figure script
docs/                     RPent behaviour this project relies on; troubleshooting
third_party/RPent         pinned upstream (read-only)
```

See [`docs/rpent_notes.md`](docs/rpent_notes.md) for the RPent internals the
tracing depends on (with source references) and
[`docs/troubleshooting.md`](docs/troubleshooting.md) for setup problems.

## License

Apache License 2.0 (`LICENSE`), the same as RPent.
