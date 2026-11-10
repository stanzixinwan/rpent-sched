"""B stage 2a: does moving GPU time between the planner and the VLA shorten a batch of agent episodes?

    python -m rsched.bench_phases --out results/b2a/<stamp> \
        --policies none vla_first intent:budget=2 history:budget=2 oracle:budget=2 \
        --levels 4 8 --repeats 3

The feasibility run (rsched.bench_interference) showed the planner (vLLM) and
Pi0.5 slow each other ~2x when they share the GPU by time-slicing. This runs a
batch of synthetic agent episodes against the real servers and coordinates them.

Episode: ``plan -> [molmo] -> vla burst`` repeated 3-6 times, then a final plan.
    plan    one streamed planner call (shared ~4000-token prompt, image, 60-200
            tokens out); the episode waits for it
    molmo   one ``molmo.ground`` call (probability 0.5 per step)
    vla     a ``pi0_pick``: the plan announces ``max_chunks`` (4-16) but the burst
            stops after ``actual`` chunks (30-100% of it: the pick succeeds early);
            each chunk is one ``vla.predict`` and a ``--vla-gap`` env step

A batch of ``--episodes-per-slot * level`` episodes runs ``level`` at a time in
order. A coordinator thread (10 ms tick) can freeze the planner server
(SIGSTOP/SIGCONT to the vLLM process tree: in-flight kernels finish, nothing new
is submitted) so a VLA burst has the GPU to itself. Policies:

    none        never pause (CUDA time-slicing; run under MPS with --tag mps)
    vla_first   pause whenever a burst is running and a planner call is in flight
    intent      pause only while some running burst has at most ``budget`` seconds
                left by the announced ``max_chunks`` (the plan's own number)
    history     the same, but a burst's length is the mean actual burst length
                (what a per-task history knows; no announcement)
    oracle      the same with the true burst length (upper bound for intent)

Every policy except ``none`` has an anti-starvation guard: after ``max_pause``
seconds paused the planner runs for at least ``min_run`` seconds. Parameters go
after a colon: ``intent:budget=2,max_pause=3,min_run=0.5``. ``--tag mps`` labels
runs made with the servers under MPS (scripts/mps.sh) so they pair with the same
batches without MPS. Every policy at a (level, repeat) runs the same episodes;
policy order rotates between repeats; finished runs are skipped on rerun.

Outputs in ``--out``: runs.jsonl (one per batch), steps.jsonl (every planner call,
Molmo call and VLA chunk), events.jsonl (pause/resume), gpu.jsonl, summary.csv.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rsched.bench_interference import LlmWorker, MolmoWorker, VlaWorker, filler
from rsched.traceio import JsonlWriter, now, read_jsonl

POLICIES = ("none", "vla_first", "intent", "history", "oracle")
DEFAULT_POLICIES = ["none", "vla_first", "intent:budget=2", "history:budget=2", "oracle:budget=2"]
POLICY_DEFAULTS = {"budget": 2.0, "max_pause": 3.0, "min_run": 0.5}


# ---------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------


@dataclass
class Step:
    kind: str            # plan | molmo | vla
    tokens: int = 0      # plan: output tokens
    announced: int = 0   # vla: max_chunks in the tool call
    chunks: int = 0      # vla: chunks actually run


def make_workload(n: int, seed: int, steps=(3, 6), tokens=(60, 200), molmo_p: float = 0.5,
                  chunk_choices=(4, 6, 8, 10, 12, 16), ratio=(0.3, 1.0)) -> list[list[Step]]:
    rng = random.Random(seed)
    episodes = []
    for _ in range(n):
        ep = []
        for _ in range(rng.randint(*steps)):
            ep.append(Step("plan", tokens=rng.randint(*tokens)))
            if rng.random() < molmo_p:
                ep.append(Step("molmo"))
            k = rng.choice(chunk_choices)
            ep.append(Step("vla", announced=k, chunks=max(1, round(k * rng.uniform(*ratio)))))
        ep.append(Step("plan", tokens=rng.randint(*tokens)))
        episodes.append(ep)
    return episodes


def mean_burst(episodes: list[list[Step]]) -> float:
    return float(np.mean([s.chunks for ep in episodes for s in ep if s.kind == "vla"]))


def parse_policy(text: str) -> tuple[str, dict]:
    name, _, rest = text.partition(":")
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; one of {POLICIES}")
    params = dict(POLICY_DEFAULTS)
    for part in filter(None, rest.split(",")):
        key, _, val = part.partition("=")
        if key not in POLICY_DEFAULTS:
            raise ValueError(f"unknown policy parameter {key!r} in {text!r}")
        params[key] = float(val)
    return name, params


# ---------------------------------------------------------------------------
# freezing the planner server
# ---------------------------------------------------------------------------


def process_tree(root: int) -> list[int]:
    """``root`` and all its descendants (via ps, so it works on Linux and macOS)."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        pid, ppid = (int(x) for x in line.split())
        children.setdefault(ppid, []).append(pid)
    tree, stack = [], [root]
    while stack:
        p = stack.pop()
        tree.append(p)
        stack.extend(children.get(p, []))
    return tree


class Pauser:
    """SIGSTOP/SIGCONT a process tree; always resumes on exit."""

    def __init__(self, root_pid: int | None) -> None:
        self.root = root_pid
        self.pids: list[int] = []
        self.paused = False
        self._lock = threading.Lock()
        if root_pid is not None:
            os.kill(root_pid, 0)  # exists and is ours
            self.pids = process_tree(root_pid)
            atexit.register(self.resume)

    def _signal(self, sig) -> None:
        for pid in self.pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def pause(self) -> None:
        with self._lock:
            if self.root is None or self.paused:
                return
            self.pids = process_tree(self.root)
            self._signal(signal.SIGSTOP)
            self.paused = True

    def resume(self) -> None:
        with self._lock:
            if self.root is None or not self.paused:
                return
            self._signal(signal.SIGCONT)
            self.paused = False


# ---------------------------------------------------------------------------
# coordinator
# ---------------------------------------------------------------------------


class Coordinator:
    def __init__(self, name: str, params: dict, pauser: Pauser, hist_chunks: float, chunk_period_s: float,
                 events: JsonlWriter, run_id: str, tick_s: float = 0.01) -> None:
        self.name, self.p = name, params
        self.pauser, self.hist, self.period = pauser, hist_chunks, chunk_period_s
        self.events, self.run_id, self.tick_s = events, run_id, tick_s
        self.lock = threading.Lock()
        self.bursts: dict[int, dict] = {}
        self.planner_calls = 0
        self.pause_t0: float | None = None
        self.run_until = 0.0
        self.paused_s = 0.0
        self.n_pauses = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # episode hooks
    def planner_begin(self) -> None:
        with self.lock:
            self.planner_calls += 1

    def planner_end(self) -> None:
        with self.lock:
            self.planner_calls -= 1

    def burst_begin(self, ep: int, step: Step) -> None:
        with self.lock:
            self.bursts[ep] = {"announced": step.announced, "chunks": step.chunks, "done": 0}

    def burst_chunk(self, ep: int) -> None:
        with self.lock:
            self.bursts[ep]["done"] += 1

    def burst_end(self, ep: int) -> None:
        with self.lock:
            self.bursts.pop(ep, None)

    # policy
    def _remaining_s(self, b: dict) -> float:
        if self.name == "vla_first":
            return 0.0
        total = {"intent": b["announced"], "oracle": b["chunks"], "history": self.hist}[self.name]
        return max(total - b["done"], 1.0) * self.period

    def want_pause(self) -> bool:
        if self.name == "none" or self.planner_calls == 0 or not self.bursts:
            return False
        return min(self._remaining_s(b) for b in self.bursts.values()) <= self.p["budget"]

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = now()
            with self.lock:
                want = self.want_pause()
                n_bursts, n_calls = len(self.bursts), self.planner_calls
            if self.pause_t0 is not None:
                if not want or t - self.pause_t0 >= self.p["max_pause"]:
                    forced = want
                    self._resume(t, "starvation guard" if forced else "done")
                    if forced:
                        self.run_until = t + self.p["min_run"]
            elif want and t >= self.run_until:
                self.pauser.pause()
                self.pause_t0 = t
                self.n_pauses += 1
                self.events.write({"run": self.run_id, "event": "pause", "t": t,
                                   "bursts": n_bursts, "planner_calls": n_calls})
            time.sleep(self.tick_s)

    def _resume(self, t: float, why: str) -> None:
        self.pauser.resume()
        self.paused_s += t - self.pause_t0
        self.events.write({"run": self.run_id, "event": "resume", "t": t, "why": why,
                           "paused_s": t - self.pause_t0})
        self.pause_t0 = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self.pause_t0 is not None:
            self._resume(now(), "end")
        self.pauser.resume()


# ---------------------------------------------------------------------------
# one batch
# ---------------------------------------------------------------------------


def run_batch(run_id: str, episodes: list[list[Step]], level: int, coord: Coordinator, make,
              steps_log: JsonlWriter) -> dict:
    todo: queue.Queue = queue.Queue()
    for i in range(len(episodes)):
        todo.put(i)
    ep_times: dict[int, tuple[float, float]] = {}
    errors = []

    def slot(idx: int) -> None:
        w = {k: make(k, idx) for k in ("vla", "molmo", "llm")}
        while True:
            try:
                ep = todo.get_nowait()
            except queue.Empty:
                return
            t_ep = now()
            for j, step in enumerate(episodes[ep]):
                base = {"run": run_id, "episode": ep, "step": j, "kind": step.kind}
                try:
                    if step.kind == "plan":
                        coord.planner_begin()
                        try:
                            rec = w["llm"].once(step.tokens)
                        finally:
                            coord.planner_end()
                        steps_log.write({**base, "tokens": step.tokens, **rec})
                    elif step.kind == "molmo":
                        steps_log.write({**base, **w["molmo"].once()})
                    else:
                        coord.burst_begin(ep, step)
                        t_b = now()
                        try:
                            for c in range(step.chunks):
                                rec = w["vla"].once()
                                coord.burst_chunk(ep)
                                steps_log.write({**base, "chunk": c, "announced": step.announced,
                                                 "chunks": step.chunks, **rec})
                        finally:
                            coord.burst_end(ep)
                        steps_log.write({**base, "kind": "burst", "t_start": t_b, "lat_s": now() - t_b,
                                         "announced": step.announced, "chunks": step.chunks})
                except Exception as exc:  # record and move on
                    errors.append(f"ep {ep} step {j}: {type(exc).__name__}: {exc}"[:300])
                    steps_log.write({**base, "error": errors[-1]})
            ep_times[ep] = (t_ep, now())

    t0 = now()
    coord.start()
    threads = [threading.Thread(target=slot, args=(i,), daemon=True) for i in range(level)]
    try:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        coord.stop()
    t1 = now()
    durs = [b - a for a, b in ep_times.values()]
    return {"t_start": t0, "t_end": t1, "makespan_s": t1 - t0, "episodes": len(episodes),
            "mean_episode_s": float(np.mean(durs)) if durs else None,
            "paused_s": coord.paused_s, "n_pauses": coord.n_pauses, "errors": errors[:20],
            "n_errors": len(errors)}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def summarize(out: Path) -> None:
    import pandas as pd

    runs = pd.DataFrame(read_jsonl(out / "runs.jsonl"))
    steps = pd.DataFrame(read_jsonl(out / "steps.jsonl"))
    if "error" not in steps:
        steps["error"] = None
    ok = steps[steps["error"].isna()]
    per_run = []
    for r in runs.itertuples(index=False):
        s = ok[ok["run"] == r.run]
        vla = s[s["kind"] == "vla"]
        plan = s[s["kind"] == "plan"]
        burst = s[s["kind"] == "burst"]
        per_run.append({"run": r.run, "vla_p50_s": vla["lat_s"].median(), "vla_p95_s": vla["lat_s"].quantile(.95),
                        "plan_p50_s": plan["lat_s"].median(), "burst_mean_s": burst["lat_s"].mean(),
                        "molmo_p50_s": s[s["kind"] == "molmo"]["lat_s"].median()})
    runs = runs.merge(pd.DataFrame(per_run), on="run")
    base = runs[runs["label"] == "none"].set_index(["level", "repeat"])["makespan_s"]
    runs["vs_none_pct"] = [(m / base.get((lv, rp), np.nan) - 1) * 100
                           for m, lv, rp in zip(runs["makespan_s"], runs["level"], runs["repeat"])]
    runs["paused_frac"] = runs["paused_s"] / runs["makespan_s"]

    def ci(x):
        x = x.dropna()
        if len(x) < 2:
            return np.nan
        from math import sqrt
        tq = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(len(x), 2.0)
        return tq * x.std(ddof=1) / sqrt(len(x))

    g = runs.groupby(["level", "label"], sort=False)
    table = g.agg(n=("makespan_s", "size"), makespan_s=("makespan_s", "mean"),
                  vs_none_pct=("vs_none_pct", "mean"), vs_none_ci=("vs_none_pct", ci),
                  mean_episode_s=("mean_episode_s", "mean"), paused_frac=("paused_frac", "mean"),
                  vla_p50_s=("vla_p50_s", "mean"), plan_p50_s=("plan_p50_s", "mean"),
                  burst_mean_s=("burst_mean_s", "mean"), molmo_p50_s=("molmo_p50_s", "mean"),
                  errors=("n_errors", "sum")).reset_index()
    table.to_csv(out / "summary.csv", index=False)
    runs.drop(columns=["errors"]).to_csv(out / "runs.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print(f"\nvs_none_pct: batch makespan against the same batch with no coordination (negative is faster), "
          f"mean and 95% t-CI over repeats\nwrote {out}/summary.csv, runs.csv")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    ap.add_argument("--tag", default="", help="label suffix for this server setup, e.g. mps -> none@mps")
    ap.add_argument("--levels", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--episodes-per-slot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs=2, default=[3, 6])
    ap.add_argument("--tokens", type=int, nargs=2, default=[60, 200])
    ap.add_argument("--molmo-p", type=float, default=0.5)
    ap.add_argument("--chunk-choices", type=int, nargs="*", default=[4, 6, 8, 10, 12, 16])
    ap.add_argument("--ratio", type=float, nargs=2, default=[0.3, 1.0], help="actual/announced chunks range")
    ap.add_argument("--vla-alone-s", type=float, default=0.165, help="Pi0.5 predict alone (feasibility: 0.165)")
    ap.add_argument("--warmup-episodes", type=int, default=2)
    ap.add_argument("--llm-pid-file", default=None, help="default: results/services/llm.pid")
    ap.add_argument("--vla-endpoint", default=f"http://127.0.0.1:{os.environ.get('VLA_PORT', '9000')}")
    ap.add_argument("--molmo-endpoint", default=f"http://127.0.0.1:{os.environ.get('MOLMO_PORT', '9002')}")
    ap.add_argument("--llm-url", default=f"http://127.0.0.1:{os.environ.get('LLM_PORT', '9003')}")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_SERVED_NAME", "planner"))
    ap.add_argument("--llm-prompt-tokens", type=int, default=4000)
    ap.add_argument("--llm-no-image", action="store_true")
    ap.add_argument("--vla-gap", type=float, default=0.17)
    ap.add_argument("--no-gpu-sampler", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--resume-llm", action="store_true", help="only SIGCONT the planner tree (after a crash)")
    args = ap.parse_args(argv)
    if not args.out and not args.resume_llm:
        ap.error("--out is required")
    if args.summarize_only:
        summarize(Path(args.out))
        return
    policies = [(text, *parse_policy(text)) for text in args.policies]
    pid_file = Path(args.llm_pid_file or Path(__file__).resolve().parents[1] / "results/services/llm.pid")
    needs_pause = any(name != "none" for _, name, _ in policies) or args.resume_llm
    pauser = Pauser(int(pid_file.read_text().strip()) if needs_pause else None)
    if args.resume_llm:
        pauser.paused = True
        pauser.resume()
        print(f"SIGCONT sent to {pauser.pids}")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (pauser.resume(), sys.exit(130)))

    shared = filler(int(args.llm_prompt_tokens * 0.6 * 0.75), random.Random(0))

    def make(kind: str, idx: int):
        if kind == "vla":
            return VlaWorker(args.vla_endpoint, args.vla_gap, seed=idx)
        if kind == "molmo":
            return MolmoWorker(args.molmo_endpoint, 0.0)
        return LlmWorker(args.llm_url, args.llm_model, args.llm_prompt_tokens, 128,
                         not args.llm_no_image, seed=1000 + idx, shared_prefix=shared)

    def workload(n: int, seed: int):
        return make_workload(n, seed, tuple(args.steps), tuple(args.tokens), args.molmo_p,
                             tuple(args.chunk_choices), tuple(args.ratio))

    (out / f"args{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps(vars(args), indent=2))
    done = {r["run"] for r in read_jsonl(out / "runs.jsonl")} if (out / "runs.jsonl").exists() else set()
    sampler = None
    if not args.no_gpu_sampler:
        sampler = subprocess.Popen([sys.executable, "-m", "rsched.nvml", "--interval", "0.25",
                                    "--log", str(out / "gpu.jsonl")])
    runs_log, steps_log, events = (JsonlWriter(out / f) for f in ("runs.jsonl", "steps.jsonl", "events.jsonl"))
    period = args.vla_alone_s + args.vla_gap
    try:
        if args.warmup_episodes:
            print(f"warm-up: {args.warmup_episodes} episodes", flush=True)
            c = Coordinator("none", POLICY_DEFAULTS, Pauser(None), 1.0, period, events, "warmup")
            wlog = JsonlWriter(out / "warmup_steps.jsonl")
            run_batch("warmup", workload(args.warmup_episodes, 10_000), min(args.warmup_episodes, 2), c, make, wlog)
            wlog.close()
        total = len(args.levels) * args.repeats * len(policies)
        k = 0
        for r in range(args.repeats):
            for level in args.levels:
                seed = args.seed + 1000 * r + level
                eps = workload(args.episodes_per_slot * level, seed)
                hist = mean_burst(workload(args.episodes_per_slot * level, seed + 500_000))  # an earlier batch
                for i in range(len(policies)):
                    text, name, params = policies[(i + r) % len(policies)]
                    label = f"{text}@{args.tag}" if args.tag else text
                    run_id = f"{label}|N{level}|r{r}"
                    k += 1
                    if run_id in done:
                        print(f"[{k}/{total}] skip {run_id} (done)", flush=True)
                        continue
                    print(f"[{k}/{total}] {run_id}: {len(eps)} episodes", flush=True)
                    coord = Coordinator(name, params, pauser, hist, period, events, run_id)
                    res = run_batch(run_id, eps, level, coord, make, steps_log)
                    runs_log.write({"run": run_id, "label": label, "policy": name, "params": params,
                                    "tag": args.tag, "level": level, "repeat": r, "seed": seed,
                                    "hist_chunks": hist, **res})
                    print(f"    makespan {res['makespan_s']:.1f} s, paused {res['paused_s']:.1f} s "
                          f"({res['n_pauses']} pauses), errors {res['n_errors']}", flush=True)
    finally:
        pauser.resume()
        for w in (runs_log, steps_log, events):
            w.close()
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
    summarize(out)


if __name__ == "__main__":
    main()
