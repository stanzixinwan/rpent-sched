"""Trace-driven simulation of the G3 scheduling modes.

Real G3 runs could not separate the scheduling effect from noise: execution
paths differ between runs, the simulator competes for CPU at N=16, and there is
little queueing at N=8. This replays *recorded* episodes against two simulated
single-server queues (VLA and Molmo, like the proxies with ``max_inflight: 1``),
so every policy sees exactly the same work, and N can go past what the CPU allows.

    # 1. where the traces are (writes one compact file)
    python analysis/g3_sim.py extract results/g3/<stamp> --out results/g3/<stamp>/sim/library.json.gz
    # 2. anywhere
    python analysis/g3_sim.py validate results/g3/<stamp>/sim/library.json.gz
    python analysis/g3_sim.py run results/g3/<stamp>/sim/library.json.gz --levels 1 4 8 16 32 64

Episode model. A recorded episode is a sequence of model requests (``vla.predict``,
``molmo.ground``) separated by *local* time: everything that is not waiting or
being served at a model queue (simulator, scripted motion, SAM3, Python, HTTP).
Local time before request i is the client gap since the previous request
returned, plus the previous request's client overhead (client latency minus
proxy wait and service). Service time is the proxy's measured service time.
Local time does not depend on the policy; only waiting does.

Policies, each applied at both queues (smallest key served first, ties FIFO):

    fifo            arrival order
    intent_srpt     least remaining plan work (the ``intent_s`` the episode logged
                    at its latest tool start before this request)
    history_srpt    least (mean work of episodes finished so far in this simulated
                    run - work done); FIFO until one episode has finished
    oracle_srpt     least true remaining time (sum of the rest of the episode's
                    local and service time): the best any predictor could do
    *_lrpt          the same estimates, most remaining first

Caveat: local time was recorded at the concurrency it ran at (default: N=8
cells, CPU unthrottled), so the simulation assumes CPU and GPU interference do
not grow with N; it isolates queueing. Service times include whatever GPU
sharing existed at N=8.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SERVERS = ("vla", "molmo")
POLICIES = ("fifo", "intent_srpt", "history_srpt", "oracle_srpt",
            "intent_lrpt", "history_lrpt", "oracle_lrpt")


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    key: str                       # "<mode>/N<level>_r<rep>/<episode id>"
    task: str                      # "<suite>_t<task>"
    seed: int
    level: int
    mode: str
    local: list[float]             # len(reqs) + 1: before each request, then the tail
    server: list[int]              # 0 = vla, 1 = molmo
    service: list[float]
    intent: list[float]            # intent_s at request time (logged by the episode)
    done: list[float]              # done_s at request time
    work_s: float                  # total priced work (history estimate's unit)
    real: dict = field(default_factory=dict)  # wall_s, wait_s as recorded
    plan_s: float | None = None    # the plan's whole-episode estimate, known before launch
    rest: list[float] = field(default_factory=list)  # true remaining time at each request

    def __post_init__(self) -> None:
        if not self.rest:
            tail = 0.0
            rest = [0.0] * len(self.server)
            for i in range(len(self.server) - 1, -1, -1):
                tail += self.service[i] + self.local[i + 1]
                rest[i] = tail
            self.rest = rest

    @property
    def plan_est(self) -> float:
        """Planned work before launch (``plan_work.intent_s``; older libraries: first request's)."""
        if self.plan_s is not None:
            return self.plan_s
        return self.intent[0] if self.intent else 0.0

    @property
    def work_time(self) -> float:
        """Wall time with no waiting at all."""
        return sum(self.local) + sum(self.service)

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d.pop("rest")
        return d


def save_library(episodes: list[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump([e.to_json() for e in episodes], fh)


def load_library(path: Path) -> list[Episode]:
    with gzip.open(path, "rt") as fh:
        return [Episode(**d) for d in json.load(fh)]


def _jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def extract_cell(mode: str, cell: Path) -> tuple[list[Episode], list[str]]:
    """Episodes of one sweep cell; second value lists skipped episodes and why."""
    manifest = json.loads((cell / "manifest.json").read_text())
    spec = {s["episode_id"]: s for s in manifest.get("specs") or []}
    proxy = {svc: {} for svc in SERVERS}
    for svc in SERVERS:
        for r in _jsonl(cell / f"proxy_{svc}.jsonl"):
            if r.get("gated") and r.get("episode"):
                proxy[svc].setdefault(r["episode"], []).append(r)
    for svc in SERVERS:
        for lst in proxy[svc].values():
            lst.sort(key=lambda r: r["t_arrive"])
    history = {r["episode"]: float(r["work_s"]) for r in _jsonl(cell / "history.jsonl")}

    out, skipped = [], []
    for path in sorted((cell / "traces").glob("episode_*.jsonl")):
        recs = _jsonl(path)
        start = next((r for r in recs if r["type"] == "episode_start"), None)
        end = next((r for r in recs if r["type"] == "episode_end"), None)
        plan = next((r for r in recs if r["type"] == "plan_work"), None)
        tools = [r for r in recs if r["type"] == "tool"]
        if not (start and end and plan and tools):
            skipped.append(f"{path.name}: incomplete (crashed or no plan)")
            continue
        ep = start["episode"]
        calls = sorted((r for r in recs if r["type"] == "client_rpc"
                        and r.get("method") in ("vla.predict", "molmo.ground") and not r.get("error")),
                       key=lambda r: r["t_start"])
        rems = sorted((r for r in recs if r["type"] == "remaining"), key=lambda r: r["t"])
        cursor = {svc: 0 for svc in SERVERS}
        local, server, service, intent, done = [], [], [], [], []
        prev_end, prev_ovh, ok = start["t"], 0.0, True
        wait_total = 0.0
        j = -1
        for c in calls:
            svc = "vla" if c["method"].startswith("vla") else "molmo"
            plist = proxy[svc].get(ep, [])
            if cursor[svc] >= len(plist):
                ok = False
                break
            p = plist[cursor[svc]]
            cursor[svc] += 1
            while j + 1 < len(rems) and rems[j + 1]["t"] <= c["t_start"]:
                j += 1
            rem = rems[j] if j >= 0 else plan
            local.append(max(c["t_start"] - prev_end, 0.0) + prev_ovh)
            server.append(SERVERS.index(svc))
            service.append(float(p["service_s"]))
            intent.append(float(rem["intent_s"]))
            done.append(float(rem.get("done_s") or 0.0))
            wait_total += float(p["wait_s"])
            prev_ovh = max(c["dur_s"] - p["wait_s"] - p["service_s"], 0.0)
            prev_end = c["t_end"]
        if not ok:
            skipped.append(f"{path.name}: client and proxy request counts differ")
            continue
        local.append(max(end["t"] - prev_end, 0.0) + prev_ovh)
        s = spec.get(ep, {})
        out.append(Episode(
            key=f"{mode}/{cell.name}/{ep}", task=f"{s.get('suite')}_t{s.get('task')}",
            seed=int(s.get("seed", 0)), level=int(manifest["level"]), mode=mode,
            local=local, server=server, service=service, intent=intent, done=done,
            work_s=history.get(ep, done[-1] if done else 0.0), plan_s=float(plan["intent_s"]),
            real={"wall_s": end["t"] - start["t"], "wait_s": wait_total,
                  "repeat": manifest["repeat"], "launch": ep},
        ))
    return out, skipped


def extract(root: Path, levels: list[int] | None) -> tuple[list[Episode], list[str]]:
    episodes, skipped = [], []
    for mode_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("analysis", "sim")):
        for cell in sorted(mode_dir.iterdir()):
            if not (cell / "manifest.json").exists():
                continue
            level = json.loads((cell / "manifest.json").read_text())["level"]
            if levels and level not in levels:
                continue
            eps, skip = extract_cell(mode_dir.name, cell)
            episodes += eps
            skipped += [f"{mode_dir.name}/{cell.name}/{s}" for s in skip]
    return episodes, skipped


# ---------------------------------------------------------------------------
# discrete-event simulation
# ---------------------------------------------------------------------------


def _key(policy: str, ep: Episode, i: int, hist_mean: float | None) -> float:
    if policy == "fifo":
        return 0.0
    kind, order = policy.split("_")
    if kind == "intent":
        est = ep.intent[i]
    elif kind == "oracle":
        est = ep.rest[i]
    else:  # history
        if hist_mean is None:
            return math.inf  # nothing finished yet: served after any known estimate, FIFO among these
        est = max(hist_mean - ep.done[i], 0.0)
    return est if order == "srpt" else -est


def simulate(episodes: list[Episode], n: int, policy: str) -> dict:
    """Closed loop: keep ``n`` episodes in flight, launch the next when one ends."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    events: list = []  # (time, seq, kind, payload)
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, payload))
        seq += 1

    pending = deque(range(len(episodes)))
    pos = [0] * len(episodes)
    t_start = [0.0] * len(episodes)
    t_end = [0.0] * len(episodes)
    wait = [0.0] * len(episodes)
    queues = [[] for _ in SERVERS]  # heap of (key, seq, ep, arrive)
    busy = [False] * len(SERVERS)
    busy_time = [0.0] * len(SERVERS)
    finished_work: list[float] = []

    def launch(t):
        e = pending.popleft()
        t_start[e] = t
        push(t + episodes[e].local[0], "arrive", e)

    def dispatch(s, t):
        if busy[s] or not queues[s]:
            return
        _, _, e, arrive = heapq.heappop(queues[s])
        wait[e] += t - arrive
        busy[s] = True
        svc = episodes[e].service[pos[e]]
        busy_time[s] += svc
        push(t + svc, "served", (s, e))

    for _ in range(min(n, len(episodes))):
        launch(0.0)
    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == "arrive":
            e = payload
            ep = episodes[e]
            i = pos[e]
            if i == len(ep.server):  # tail done: episode ends
                t_end[e] = t
                finished_work.append(ep.work_s)
                if pending:
                    launch(t)
                continue
            s = ep.server[i]
            mean = sum(finished_work) / len(finished_work) if finished_work else None
            heapq.heappush(queues[s], (_key(policy, ep, i, mean), seq, e, t))
            seq += 1
            dispatch(s, t)
        else:  # served
            s, e = payload
            busy[s] = False
            pos[e] += 1
            push(t + episodes[e].local[pos[e]], "arrive", e)
            dispatch(s, t)

    wall = np.array(t_end) - np.array(t_start)
    waits = np.array(wait)
    makespan = float(max(t_end))
    return {
        "makespan_s": makespan,
        "wall_mean_s": float(wall.mean()),
        "wall_p50_s": float(np.median(wall)),
        "wall_p95_s": float(np.quantile(wall, 0.95)),
        "wait_mean_s": float(waits.mean()),
        "wait_frac": float(waits.sum() / wall.sum()),
        "episodes_per_hour": len(episodes) / makespan * 3600,
        **{f"{s}_util": busy_time[k] / makespan for k, s in enumerate(SERVERS)},
    }


# ---------------------------------------------------------------------------
# launch order (which episode enters when a slot frees up)
# ---------------------------------------------------------------------------

LAUNCH_RULES = ("given", "lpt", "spt", "front", "front_back", "oracle_lpt", "oracle_front_back")


def launch_order(work: list[Episode], rule: str, n: int) -> list[Episode]:
    """Order a batch before it starts, from each episode's planned work.

    given        the batch as listed (random in the experiments): the baseline
    lpt          longest planned work first
    spt          shortest planned work first
    front        the ``n`` longest first (longest leading), the rest as listed
    front_back   like ``front``, and the ``n`` shortest last (shortest at the very end)
    oracle_*     the same with the true work (local + service time), for reference
    """
    if rule not in LAUNCH_RULES:
        raise ValueError(f"unknown launch rule {rule!r}; choose from {LAUNCH_RULES}")
    if rule == "given":
        return list(work)
    oracle = rule.startswith("oracle_")
    base = rule.removeprefix("oracle_")
    est = [e.work_time if oracle else e.plan_est for e in work]
    by_len = sorted(range(len(work)), key=lambda i: est[i], reverse=True)
    if base == "lpt":
        return [work[i] for i in by_len]
    if base == "spt":
        return [work[i] for i in reversed(by_len)]
    k = min(n, len(work))
    front = by_len[:k]
    taken = set(front)
    back = [i for i in reversed(by_len) if i not in taken][:k] if base == "front_back" else []
    taken |= set(back)
    middle = [i for i in range(len(work)) if i not in taken]
    return [work[i] for i in front + middle + list(reversed(back))]


def sample_workload(lib: list[Episode], k: int, rng: np.random.Generator) -> list[Episode]:
    """``k`` episodes in random order: every (task, seed) cell in turn, a different recording each time."""
    by_cell: dict[tuple, list[Episode]] = {}
    for e in lib:
        by_cell.setdefault((e.task, e.seed), []).append(e)
    cells = list(by_cell.values())
    pools = [list(rng.permutation(len(c))) for c in cells]
    out: list[Episode] = []
    rnd = 0
    while len(out) < k:
        for ci in rng.permutation(len(cells)):
            if len(out) == k:
                break
            out.append(cells[ci][pools[ci][rnd % len(cells[ci])]])
        rnd += 1
    return [out[i] for i in rng.permutation(len(out))]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_extract(args) -> None:
    root = Path(args.g3_dir)
    episodes, skipped = extract(root, args.levels)
    out = Path(args.out) if args.out else root / "sim" / "library.json.gz"
    save_library(episodes, out)
    print(f"{len(episodes)} episodes, {sum(len(e.server) for e in episodes)} model requests -> {out}")
    for s in skipped:
        print("  skipped", s)


def cmd_validate(args) -> None:
    """Replay each real FIFO cell (same episodes, same launch order, same N) and compare."""
    import pandas as pd

    lib = load_library(Path(args.library))
    cells: dict[tuple, list[Episode]] = {}
    for e in lib:
        if e.mode == "none":
            cells.setdefault((e.level, e.real["repeat"]), []).append(e)
    rows = []
    for (level, rep), eps in sorted(cells.items()):
        eps.sort(key=lambda e: e.real["launch"])
        sim = simulate(eps, level, "fifo")
        real_wall = np.mean([e.real["wall_s"] for e in eps])
        real_wait = np.mean([e.real["wait_s"] for e in eps])
        rows.append({"level": level, "repeat": rep, "episodes": len(eps),
                     "real_wall_mean_s": real_wall, "sim_wall_mean_s": sim["wall_mean_s"],
                     "real_wait_mean_s": real_wait, "sim_wait_mean_s": sim["wait_mean_s"],
                     "no_wait_wall_mean_s": np.mean([e.work_time for e in eps])})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 160, "display.float_format", "{:.1f}".format):
        print(df.to_string(index=False))
    print("A cell matches if sim and real agree on mean wall and mean wait; at N=16 the real local"
          " times are CPU-inflated, so the replay uses them as recorded.")


def cmd_run(args) -> None:
    import pandas as pd

    lib = [e for e in load_library(Path(args.library)) if e.level in args.source_levels]
    rng = np.random.default_rng(args.seed)
    rows = []
    for w in range(args.workloads):
        # A workload: one recording per (task, seed) cell, picked at random, in random order.
        by_cell: dict[tuple, list[Episode]] = {}
        for e in lib:
            by_cell.setdefault((e.task, e.seed), []).append(e)
        picks = [cands[rng.integers(len(cands))] for cands in by_cell.values()]
        reps = max(1, math.ceil(args.episodes / len(picks)))
        work = (picks * reps)[: args.episodes]
        order = rng.permutation(len(work))
        work = [work[k] for k in order]
        for n in args.levels:
            for policy in args.policies:
                rows.append({"workload": w, "level": n, "policy": policy, **simulate(work, n, policy)})
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(args.library).parent
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sim_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "sim_summary.csv", index=False)
    plot(summary, out / "sim_gain.png")
    cols = ["level", "policy", "fifo_wait_frac", "wall_mean_s_vs_fifo_pct", "wall_mean_s_ci",
            "makespan_s_vs_fifo_pct", "makespan_s_ci", "wall_p95_s_vs_fifo_pct", "wait_mean_s_vs_fifo_pct"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(summary[cols].to_string(index=False))
    print(f"\n{args.workloads} workloads x {args.episodes} episodes from N={args.source_levels} recordings"
          f" -> {out}/sim_runs.csv, sim_summary.csv, sim_gain.png")


def cmd_launch(args) -> None:
    """Launch-order rules x queue policies, against (given order, FIFO) on the same batch."""
    import pandas as pd

    lib = [e for e in load_library(Path(args.library)) if e.level in args.source_levels]
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in args.batch:
        for n in args.levels:
            for w in range(args.workloads):
                work = sample_workload(lib, k, rng)
                base = simulate(work, n, "fifo")
                for rule in args.rules:
                    ordered = launch_order(work, rule, n)
                    for queue in args.queues:
                        r = simulate(ordered, n, queue) if (rule, queue) != ("given", "fifo") else base
                        rows.append({"batch": k, "level": n, "workload": w, "rule": rule, "queue": queue,
                                     "makespan_s": r["makespan_s"], "wall_mean_s": r["wall_mean_s"],
                                     "makespan_pct": (r["makespan_s"] / base["makespan_s"] - 1) * 100,
                                     "wall_mean_pct": (r["wall_mean_s"] / base["wall_mean_s"] - 1) * 100,
                                     "base_makespan_s": base["makespan_s"]})
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(args.library).parent
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "launch_runs.csv", index=False)
    g = df.groupby(["batch", "level", "rule", "queue"])
    summary = g.agg(makespan_pct=("makespan_pct", "mean"), makespan_sd=("makespan_pct", "std"),
                    wall_mean_pct=("wall_mean_pct", "mean"), base_makespan_s=("base_makespan_s", "mean"),
                    n=("makespan_pct", "size")).reset_index()
    summary["makespan_ci"] = 1.96 * summary["makespan_sd"] / np.sqrt(summary["n"])
    summary.to_csv(out / "launch_summary.csv", index=False)
    plot_launch(summary, out / "launch_gain.png")
    piv = summary.pivot_table(index=["batch", "level"], columns=["rule", "queue"], values="makespan_pct")
    with pd.option_context("display.width", 250, "display.float_format", "{:.1f}".format):
        print("makespan change vs (given order, FIFO), %:")
        print(piv.to_string())
    print(f"\n{args.workloads} batches per cell from N={args.source_levels} recordings -> "
          f"{out}/launch_runs.csv, launch_summary.csv, launch_gain.png")


def plot_launch(summary, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches = sorted(summary["batch"].unique())
    fig, axes = plt.subplots(1, len(batches), figsize=(3.6 * len(batches), 3.4), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], batches):
        part = summary[summary["batch"] == k]
        for (rule, queue), g in part.groupby(["rule", "queue"]):
            if (rule, queue) == ("given", "fifo") or rule == "spt":
                continue
            g = g.sort_values("level")
            ls = ":" if rule.startswith("oracle_") else ("-" if queue == "fifo" else "--")
            ax.errorbar(g["level"], g["makespan_pct"], yerr=g["makespan_ci"], ls=ls, marker="o", ms=3,
                        capsize=2, label=f"{rule} + {queue}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"batch of {k} episodes")
        ax.set_xlabel("concurrent episodes (N)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("makespan vs random launch + FIFO (%)")
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize(df):
    import pandas as pd

    base = df[df["policy"] == "fifo"].set_index(["workload", "level"])
    rows = []
    for (level, policy), g in df.groupby(["level", "policy"]):
        g = g.set_index(["workload", "level"])
        b = base.loc[g.index]
        row = {"level": level, "policy": policy, "fifo_wait_frac": b["wait_frac"].mean(),
               "wall_mean_s": g["wall_mean_s"].mean(), "makespan_s": g["makespan_s"].mean()}
        for m in ("wall_mean_s", "makespan_s", "wall_p95_s", "wait_mean_s"):
            pct = (g[m] / b[m] - 1) * 100
            row[f"{m}_vs_fifo_pct"] = pct.mean()
            half = 1.96 * pct.std(ddof=1) / math.sqrt(len(pct)) if len(pct) > 1 else float("nan")
            row[f"{m}_ci"] = half
        rows.append(row)
    out = pd.DataFrame(rows)
    out["order"] = out["policy"].map({p: k for k, p in enumerate(POLICIES)})
    return out.sort_values(["level", "order"]).drop(columns="order")


def plot(summary, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {"intent": "-", "history": "--", "oracle": ":"}
    for ax, metric, title in ((axes[0], "wall_mean_s", "mean episode time (SRPT)"),
                              (axes[1], "makespan_s", "makespan (LRPT)")):
        order = "srpt" if metric == "wall_mean_s" else "lrpt"
        for kind, ls in styles.items():
            part = summary[summary["policy"] == f"{kind}_{order}"].sort_values("level")
            if part.empty:
                continue
            ax.errorbar(part["level"], part[f"{metric}_vs_fifo_pct"], yerr=part[f"{metric}_ci"],
                        ls=ls, marker="o", ms=3, capsize=2, label=f"{kind}_{order}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log", base=2)
        levels = sorted(summary["level"].unique())
        ax.set_xticks(levels, [str(v) for v in levels])
        ax.set_xlabel("concurrent episodes (N)")
        ax.set_title(f"{title} vs FIFO")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("change vs FIFO (%), lower is better")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="build an episode library from a G3 run")
    e.add_argument("g3_dir")
    e.add_argument("--out", default=None)
    e.add_argument("--levels", type=int, nargs="*", help="only cells at these N (default: all)")
    v = sub.add_parser("validate", help="replay real FIFO cells and compare with what was measured")
    v.add_argument("library")
    r = sub.add_parser("run", help="simulate policies over N")
    r.add_argument("library")
    r.add_argument("--levels", type=int, nargs="*", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    r.add_argument("--policies", nargs="*", default=list(POLICIES))
    r.add_argument("--source-levels", type=int, nargs="*", default=[8],
                   help="use recordings made at these N (default 8: CPU unthrottled)")
    r.add_argument("--episodes", type=int, default=128, help="episodes per workload")
    r.add_argument("--workloads", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None)
    la = sub.add_parser("launch", help="launch-order rules for batch makespan")
    la.add_argument("library")
    la.add_argument("--batch", type=int, nargs="*", default=[16, 32, 64, 128])
    la.add_argument("--levels", type=int, nargs="*", default=[4, 8, 12, 16])
    la.add_argument("--rules", nargs="*", default=list(LAUNCH_RULES))
    la.add_argument("--queues", nargs="*", default=["fifo", "intent_lrpt"])
    la.add_argument("--source-levels", type=int, nargs="*", default=[8])
    la.add_argument("--workloads", type=int, default=40)
    la.add_argument("--seed", type=int, default=0)
    la.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    {"extract": cmd_extract, "validate": cmd_validate, "run": cmd_run, "launch": cmd_launch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
