# RPent behaviour notes

Facts about the pinned RPent that this project's design or measurements depend
on. Each is checked against the source at the pinned commit; paths are relative
to `third_party/RPent`.

- **Pinned commit:** `ef70215` (2026-10-13). When the submodule is bumped,
  re-check every row and update this commit hash.
- Add a row only when a design decision or a measurement depends on it.

## Service model

| fact | source | why it matters here |
|---|---|---|
| RPC servers run on `ThreadingHTTPServer` (one thread per request) | `rpent/utils/rpc/http_rpc.py:207` | concurrent requests reach the model at once |
| Non-read-only RPC methods hold an exclusive write lock | `rpent/utils/rpc/rpc_facade.py:150` | `vla.predict` is serialized |
| Lock waiters wake via `notify_all` and race | `rpent/utils/rwlock.py:59` | RPent's order is not FIFO; the proxy's `fifo` policy defines the baseline, `passthrough` measures RPent as is |
| `vla.predict` is registered as a mutating method | `rpent/robots/components/vla_facade_base.py:56` | serialized as above |
| `sam3.segment` is read-only | `rpent/robots/components/sam3_server.py:125` | SAM3 requests run concurrently; little queueing expected |
| Pi0.5 inference is batched (`predict_action_batch`); clients send batch 1 | `rpent/robots/components/pi05_vla_server.py:259` | a batching baseline can merge requests in the proxy |
| Pi0.5 is built on CPU, then moved to GPU | `rpent/robots/components/pi05_vla_server.py:227` | ~13 GB host RAM peak at load (WSL2 OOM) |
| Client timeouts: `vla.predict` 120 s, `sam3.segment` 120 s | `rpent/robots/components/vla_client_base.py:23`, `rpent/robots/components/sam3_client.py:45` | very high concurrency can time out; the proxy logs failures |

## Tools and VLA demand

| fact | source | why it matters here |
|---|---|---|
| One `vla.predict` returns one action chunk; LIBERO chunk = 5 actions | `robots/libero/tools.py:169`, `rpent/robots/components/pi05_vla_server.py:93` | VLA call count = chunks executed |
| `pi0_pick` loops chunks until success or `max_chunks` (default 24) | `robots/libero/tools.py:206`, `:210` | the planner writes an upper bound on VLA demand before execution: the core signal |
| `pi0_doubled` loops up to `max_chunks` (default 20) | `robots/libero/tools.py:293` | same |
| Scripted tools (`move_to`, `move_pose`, `rotate_*`, `release`, `set_gripper`) drive the simulator only | `robots/libero/tools.py` | no GPU-model demand; simulator time |
| All planners execute tools through `Toolkit.execute_tool`, in a thread | `rpent/tools/toolkit.py:215`, `rpent/planner/base.py:86` | one hook in `rsched/launch.py` sees every tool call and its arguments |

## Runtime and episodes

| fact | source | why it matters here |
|---|---|---|
| Each episode spawns its own `env_server` with `MUJOCO_GL=egl` | `robots/libero/robot_spec.py:385`, `:419` | simulator cost scales with N; needs EGL and libGL |
| `finalize_run` is optional and LIBERO does not define one | `rpent/robots/robot_spec.py:88`, `rpent/cli/main.py:700` | issue K1: the `outcome` hook in `rsched/launch.py` never fires |
| Claude Code planner stats include `total_cost_usd` | `rpent/planner/claude_code.py:632` | per-episode cost is available once K1 is fixed |
| `--memory-version auto` falls back to `GPT_5.5_xhigh` for unmapped models | `robots/libero/memory.py:76` | configs pin it explicitly |

## Flash Mode (plan-once regime)

| fact | source | why it matters here |
|---|---|---|
| `--planner flash` replays a stored plan; no LLM decisions | `docs/source-en/rst_source/guides/flash.rst` | zero API cost; VLA/simulator-bound workload |
| Molmo2-8B localizes plan anchors; needs its own venv (newer `transformers`) | same | a third GPU model; extra service to launch |
| Plans for 78 LIBERO-PRO tasks under `GPT_5.5_xhigh/flash/`; `goal_swap_t0` and `10_swap_t9` absent | same | task list for Flash sweeps |
| Supported suites | `robots/libero/robot_spec.py:62` | config validation |
