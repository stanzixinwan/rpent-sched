"""Plan intent vs per-task history as the launch-order estimate (simulated).

The obvious objection to plan-intent launch ordering: batch evaluation reruns
the same tasks (other checkpoints, other seeds), so a table "task -> work last
time" could replace the plan. This replays G3 recordings and compares, for
JIT-LPT launch order, estimates of each episode's work:

    intent      the Flash plan's whole-episode estimate (one value per task)
    hist_h      mean work of h earlier recordings of the same task (h=1: "last run")
    oracle      the episode's true work

History is drawn only from recordings outside the simulated batch. Cold start:
a fraction f of the batch's tasks has no history. Then

    hist_h      unknown tasks get the mean of the known tasks' estimates
    hybrid_h    known tasks use history; unknown tasks use the plan, put in
                seconds with the work/plan ratio of the known tasks

Every estimate is rescaled so its batch sum equals the plan's (rank-preserving),
so jit_pick's prior work/plan ratio fits all of them equally.

    python analysis/history_baseline.py results/g3/<stamp>/sim/library.json.gz
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_sim import load_library, sample_workload, simulate  # noqa: E402
from rsched.launch_order import jit_pick  # noqa: E402


def jit(est: list[float], slack: float = 1.0):
    def pick(t, pending, state):
        return jit_pick(t, [est[i] for i in pending],
                        [(est[i], t0) for i, t0 in state["running"].items()],
                        [(est[i], w) for i, w in state["finished"]], state["n"], slack)
    return pick


def rescale(est: list[float], plan: list[float]) -> list[float]:
    s = sum(est)
    return [e * sum(plan) / s for e in est] if s > 0 else list(plan)


def history_table(lib, exclude: set, h: int | None, rng) -> dict[str, float]:
    """task -> mean work of h random recordings not in ``exclude`` (None: all)."""
    by = defaultdict(list)
    for e in lib:
        if e.key not in exclude:
            by[e.task].append(e.work_time)
    out = {}
    for task, w in by.items():
        if h is not None and h < len(w):
            w = list(rng.choice(w, size=h, replace=False))
        out[task] = float(np.mean(w))
    return out


def estimates(work, table, cold: set) -> dict[str, list[float]]:
    plan = [e.plan_est for e in work]
    known = [e for e in work if e.task not in cold]
    if known:
        fill = float(np.mean([table[e.task] for e in known]))
        ratio = sum(table[e.task] for e in known) / sum(e.plan_est for e in known)
    else:
        fill, ratio = 1.0, 1.0
    hist = [fill if e.task in cold else table[e.task] for e in work]
    hybrid = [e.plan_est * ratio if e.task in cold else table[e.task] for e in work]
    return {"hist": rescale(hist, plan), "hybrid": rescale(hybrid, plan)}


def spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def predict_table(lib, rng) -> pd.DataFrame:
    """Rank correlation with true work over all recordings (history: leave this one out)."""
    true = [e.work_time for e in lib]
    rows = {"intent": [e.plan_est for e in lib]}
    by = defaultdict(list)
    for e in lib:
        by[e.task].append(e)
    for h in (1, 3, None):
        est = []
        for e in lib:
            others = [o.work_time for o in by[e.task] if o.key != e.key]
            if h is not None:
                others = list(rng.choice(others, size=h, replace=False))
            est.append(float(np.mean(others)))
        rows[f"hist_{h or 'all'}"] = est
    out = [{"estimate": name, "spearman": spearman(est, true)} for name, est in rows.items()]
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("library")
    ap.add_argument("--batch", type=int, nargs="*", default=[32, 64, 128])
    ap.add_argument("--levels", type=int, nargs="*", default=[8, 16])
    ap.add_argument("--history", nargs="*", default=["1", "3", "all"])
    ap.add_argument("--cold", type=float, nargs="*", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--source-levels", type=int, nargs="*", default=[8])
    ap.add_argument("--workloads", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lib = [e for e in load_library(Path(args.library)) if e.level in args.source_levels]
    tasks = sorted({e.task for e in lib})
    rng = np.random.default_rng(args.seed)
    out = Path(args.out) if args.out else Path(args.library).parent
    out.mkdir(parents=True, exist_ok=True)

    pred = predict_table(lib, rng)
    pred.to_csv(out / "history_predict.csv", index=False)
    print("rank correlation with true episode work (N=%s recordings):" % args.source_levels)
    print(pred.to_string(index=False, float_format="{:.2f}".format), "\n")

    rows = []
    for k in args.batch:
        for n in args.levels:
            for w in range(args.workloads):
                work = sample_workload(lib, k, rng)
                plan = [e.plan_est for e in work]
                base = simulate(work, n, "fifo")["makespan_s"]

                def run(name, est, h="-", cold=0.0):
                    r = simulate(work, n, "fifo", pick=jit(est))
                    rows.append({"batch": k, "level": n, "workload": w, "estimate": name, "history": h,
                                 "cold": cold, "makespan_pct": (r["makespan_s"] / base - 1) * 100})

                run("intent", plan)
                run("oracle", rescale([e.work_time for e in work], plan))
                exclude = {e.key for e in work}
                for h in args.history:
                    table = history_table(lib, exclude, None if h == "all" else int(h), rng)
                    for f in args.cold:
                        cold = set(rng.choice(tasks, size=round(f * len(tasks)), replace=False))
                        est = estimates(work, table, cold)
                        run("hist", est["hist"], h, f)
                        if f > 0:
                            run("hybrid", est["hybrid"], h, f)
            print(f"batch {k}, N={n}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "history_runs.csv", index=False)
    g = df.groupby(["batch", "level", "estimate", "history", "cold"])["makespan_pct"]
    s = g.agg(["mean", "std", "size"]).reset_index()
    s["ci"] = 1.96 * s["std"] / np.sqrt(s["size"])
    s.to_csv(out / "history_summary.csv", index=False)

    # Warm start: intent vs history vs oracle.
    warm = s[(s["cold"] == 0)].copy()
    warm["col"] = warm["estimate"] + warm["history"].map(lambda h: "" if h == "-" else f"_{h}")
    piv = warm.pivot_table(index=["batch", "level"], columns="col", values="mean")
    cols = ["intent"] + [f"hist_{h}" for h in args.history] + ["oracle"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print("JIT-LPT makespan vs random launch + FIFO, %, every task has history:")
        print(piv[cols].to_string(), "\n")
        for h in args.history:
            c = s[(s["history"] == h) & (s["estimate"].isin(["hist", "hybrid"]))]
            p = c.pivot_table(index=["batch", "level"], columns=["estimate", "cold"], values="mean")
            print(f"cold start, history = {h} recording(s) per known task (cold = share of tasks never run):")
            print(p.to_string(), "\n")
    # paired: intent minus hist (warm), per cell
    pw = df[(df["cold"] == 0) | (df["estimate"].isin(["intent", "oracle"]))]
    piv2 = pw.pivot_table(index=["batch", "level", "workload"], columns=["estimate", "history"],
                          values="makespan_pct")
    print("paired difference intent - hist_h (pp; positive = history better), mean ± 95% CI:")
    for h in args.history:
        d = piv2[("intent", "-")] - piv2[("hist", h)]
        gg = d.groupby(level=[0, 1])
        print(f"  h={h}: " + "  ".join(f"B{b}/N{n} {m:+.1f}±{1.96*sd/np.sqrt(c):.1f}"
                                         for (b, n), m, sd, c in zip(gg.mean().index, gg.mean(), gg.std(), gg.size())))
    print(f"\n-> {out}/history_predict.csv, history_runs.csv, history_summary.csv")


if __name__ == "__main__":
    main()
