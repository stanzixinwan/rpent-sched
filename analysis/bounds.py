"""First-order bounds: how much can each lever shorten a batch, given where time goes?

    python analysis/bounds.py --results results --anatomy results/paper/anatomy \
        --out results/paper/bounds.csv

The paper's argument is that a lever's payoff follows from the anatomy of an
episode. This script states that as arithmetic and puts the measured effect
next to it. With s_x the share of summed episode time spent in component x:

  batching the VLA      <= s_vla_wait + s_vla_svc * (1 - c(b)/b),  b = min(N, max_batch)
                           c(b): latency of a batch of b relative to batch 1 (bench_batch)
  reordering requests   <= s_vla_wait + s_molmo_wait   (mean episode time; total work unchanged)
  launch ordering       <= 1 - sum(wall) / (N * makespan)   (slot time left idle at the tail)
  removing a component  =  its share (Amdahl), for the levers not built yet

Bounds are first order: they ignore that a faster component raises the load on
the others (closed loop), so they are upper bounds on the relative reduction.
Measured values come from the experiment summaries already under results/.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import pandas as pd

MAX_BATCH = 16


def latest(results: Path, pattern: str) -> Path | None:
    hits = sorted(results.glob(pattern))
    return hits[-1] if hits else None


def batch_cost(results: Path, setup: str = "mps", llm_streams: int = 0) -> dict[int, float]:
    """c(b): p50 latency of a batch of b relative to batch 1."""
    f = latest(results, "b_batch/*/batch.csv")
    if f is None:
        return {}
    df = pd.read_csv(f)
    df = df[(df["setup"] == setup) & (df["llm_streams"] == llm_streams)]
    return dict(zip(df["batch"].astype(int), df["lat_x_vs_b1"].astype(float)))


def cost_at(c: dict[int, float], b: int) -> float:
    """Interpolate c(b) linearly between measured batch sizes."""
    if not c:
        return math.nan
    if b in c:
        return c[b]
    xs = sorted(c)
    lo = max((x for x in xs if x <= b), default=xs[0])
    hi = min((x for x in xs if x >= b), default=xs[-1])
    return c[lo] if lo == hi else c[lo] + (c[hi] - c[lo]) * (b - lo) / (hi - lo)


def batching_bound(s_wait: float, s_svc: float, n: int, c: dict[int, float]) -> float:
    b = min(n, MAX_BATCH)
    return s_wait + s_svc * (1 - cost_at(c, b) / b)


def measured(summary: Path | None, key: str, value, col: str, level: int) -> tuple[float, float, float, int]:
    """(mean %, CI low, CI high, repeats) of a row in a launch_compare-style summary."""
    if summary is None or not summary.exists():
        return (math.nan,) * 3 + (0,)
    df = pd.read_csv(summary)
    row = df[(df[key] == value) & (df["level"] == level)]
    if row.empty:
        return (math.nan,) * 3 + (0,)
    r = row.iloc[0]
    return float(r[col]), float(r["ci95_low"]), float(r["ci95_high"]), int(r["repeats"])


def real_rows(results: Path, anatomy: Path, c: dict[int, float]) -> list[dict]:
    eps = pd.read_csv(anatomy / "episodes.csv")
    rows = []
    g6 = latest(results, "g6/*/analysis/summary.csv")
    for label, n in (("replay_N8_mps", 8), ("replay_N16_mps", 16)):
        sub = eps[eps["set"] == label]
        if sub.empty:
            continue
        total = sub["wall_s"].sum()
        s = {k: sub[k].sum() / total for k in ("vla_wait", "vla_svc", "molmo_wait", "molmo_svc")}
        m = measured(g6, "rule", "greedy", "makespan_pct_mean", n)
        rows.append({"workload": "plan replay (real)", "N": n, "lever": "batch VLA calls (greedy)",
                     "bound_pct": -100 * batching_bound(s["vla_wait"], s["vla_svc"], n, c),
                     "measured_pct": m[0], "ci_low": m[1], "ci_high": m[2], "repeats": m[3],
                     "metric": "makespan", "basis": f"s_vla_wait={s['vla_wait']:.3f}, s_vla_svc={s['vla_svc']:.3f}"})
        rows.append({"workload": "plan replay (real)", "N": n, "lever": "reorder model queues (any rule)",
                     "bound_pct": -100 * (s["vla_wait"] + s["molmo_wait"]),
                     "measured_pct": math.nan, "ci_low": math.nan, "ci_high": math.nan, "repeats": 0,
                     "metric": "mean episode time",
                     "basis": f"s_vla_wait={s['vla_wait']:.3f}, s_molmo_wait={s['molmo_wait']:.3f}"})
    # Levers nobody built: the share is the ceiling.
    for label, n in (("replay_N1", 1), ("replay_N8_mps", 8), ("replay_N16_mps", 16)):
        sub = eps[eps["set"] == label]
        if sub.empty:
            continue
        total = sub["wall_s"].sum()
        for lever, cols in (("skip scene reset (warm environments)", ["sim_reset"]),
                            ("reuse processes (startup + teardown)", ["startup", "teardown"]),
                            ("free client-side scripted motion", ["local_tools"]),
                            ("free simulator stepping + rendering", ["sim_step", "sim_chunk_step", "sim_render"]),
                            ("free all GPU model time (service + queue)",
                             ["vla_wait", "vla_svc", "molmo_wait", "molmo_svc", "sam3_wait", "sam3_svc"])):
            rows.append({"workload": "plan replay (real)", "N": n, "lever": lever,
                         "bound_pct": -100 * sum(sub[k].sum() for k in cols) / total,
                         "measured_pct": math.nan, "ci_low": math.nan, "ci_high": math.nan, "repeats": 0,
                         "metric": "episode time (Amdahl)", "basis": "+".join(cols)})
    return rows


def launch_rows(results: Path) -> list[dict]:
    rows = []
    for name, root, arms in (("G4 tuning tasks", latest(results, "launch/*/analysis"), [("jit_lpt", "plan")]),
                             ("G5 held-out tasks", latest(results, "g5/*/analysis"),
                              [("history", "history"), ("plan", "plan"), ("history_cold50", "history, half cold")])):
        if root is None or not (root / "cells.csv").exists():
            continue
        cells = pd.read_csv(root / "cells.csv")
        base = cells[cells["rule"] == "given"]
        if base.empty:
            continue
        ok = base["episodes"] - base["crashed"]
        slack = (1 - ok * base["wall_mean_s"] / (base["level"] * base["makespan_s"])).mean()
        for arm, text in arms:
            m = measured(root / "summary.csv", "rule", arm, "makespan_pct_mean", int(base["level"].iloc[0]))
            rows.append({"workload": f"plan replay (real), {name}", "N": int(base["level"].iloc[0]),
                         "lever": f"launch order JIT-LPT ({text})", "bound_pct": -100 * slack,
                         "measured_pct": m[0], "ci_low": m[1], "ci_high": m[2], "repeats": m[3],
                         "metric": "makespan", "basis": "idle slot time under random order"})
    return rows


def synthetic_rows(results: Path, c: dict[int, float]) -> list[dict]:
    """The agent-like synthetic workload of bench_phases (B3): VLA share from its proxy log."""
    root = latest(results, "b3/*")
    if root is None or not (root / "phases" / "runs.csv").exists() or not (root / "proxy_none.jsonl").exists():
        return []
    runs = pd.read_csv(root / "phases" / "runs.csv")
    summary = pd.read_csv(root / "phases" / "summary.csv")
    reqs = []
    with (root / "proxy_none.jsonl").open() as fh:
        for line in fh:
            if '"vla.predict"' in line:
                r = json.loads(line)
                reqs.append((float(r["t_arrive"]), float(r.get("wait_s") or 0), float(r.get("service_s") or 0)))
    rows = []
    for n in sorted(runs["level"].unique()):
        base = runs[(runs["label"] == "none") & (runs["level"] == n)]
        if base.empty:
            continue
        wait = svc = 0.0
        for _, run in base.iterrows():
            for t, w, s in reqs:
                if run["t_start"] <= t <= run["t_end"]:
                    wait, svc = wait + w, svc + s
        total = float((base["episodes"] * base["mean_episode_s"]).sum())
        if total <= 0 or wait + svc <= 0:
            continue
        hit = summary[(summary["label"] == "none@greedy") & (summary["level"] == n)]
        if hit.empty:
            continue
        pct, ci = float(hit["vs_none_pct"].iloc[0]), float(hit["vs_none_ci"].iloc[0])
        rows.append({"workload": "agent-like synthetic (B3)", "N": int(n), "lever": "batch VLA calls (greedy)",
                     "bound_pct": -100 * batching_bound(wait / total, svc / total, int(n), c),
                     "measured_pct": pct, "ci_low": pct - ci, "ci_high": pct + ci, "repeats": int(hit["n"].iloc[0]),
                     "metric": "makespan",
                     "basis": f"s_vla_wait={wait / total:.3f}, s_vla_svc={svc / total:.3f}"})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--anatomy", default="results/paper/anatomy")
    ap.add_argument("--out", default="results/paper/bounds.csv")
    args = ap.parse_args(argv)
    results, anatomy = Path(args.results), Path(args.anatomy)
    if not (anatomy / "episodes.csv").exists():
        sys.exit(f"{anatomy}/episodes.csv missing: run analysis/anatomy.py first (see README, Experiments)")
    c_real = batch_cost(results, "mps", 0)
    c_syn = batch_cost(results, "mps", 4) or c_real
    rows = real_rows(results, anatomy, c_real) + launch_rows(results) + synthetic_rows(results, c_syn)
    df = pd.DataFrame(rows)
    df["within_bound"] = [None if math.isnan(m) else bool(m >= b - 1e-9 or lo >= b - 1e-9)
                          for m, b, lo in zip(df["measured_pct"], df["bound_pct"], df["ci_high"])]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 48)
    show = df[["workload", "N", "lever", "bound_pct", "measured_pct", "ci_low", "ci_high", "repeats", "metric"]]
    print(show.round(1).to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
