"""Out-of-sample check of plan-based JIT-LPT: are its gains from fitting to these tasks?

Two things were set with the same 16 G3 tasks the results are reported on:
the cost table that turns a plan into seconds (``rsched.intent.DEFAULT_COSTS``,
from G2 N=1 traces of 4 of these tasks) and JIT-LPT's two knobs (``slack``,
``prior_scale``). This replays G3 recordings (``analysis/g3_sim.py``) in three
out-of-sample settings:

    unseen      batches drawn only from the 12 tasks that did not feed the
                default cost table; default costs and knobs (no refitting)
    split       many random 8/8 splits of the tasks: refit the costs (three
                non-negative rates: per pi0 chunk, per Molmo call, and a
                multiplier on the default scripted-step costs) and pick the
                knobs on one half, evaluate on batches of the other half
    grid        knob sensitivity: every (slack, prior_scale) on all tasks

Each arm is paired with the same batch launched in random order (FIFO
queues); numbers are batch makespan change in percent (negative is better).

    python analysis/holdout.py results/g3/<stamp>/sim/library.json.gz \
        --plans results/flash_plans/libero/GPT_5.5_xhigh/flash
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_sim import load_library, sample_workload, simulate  # noqa: E402

from rsched.intent import DEFAULT_COSTS, plan_work, step_cost  # noqa: E402
from rsched.launch_order import jit_pick, load_program, plan_name  # noqa: E402

# Tasks whose G2 N=1 traces produced DEFAULT_COSTS (configs/g2_flash.yaml).
COST_TASKS = {"libero_object_swap_t2", "libero_goal_swap_t1", "libero_spatial_swap_t1", "libero_10_swap_t3"}
SLACKS = (0.5, 1.0, 2.0)
PRIORS = (1.0, 1.5, 2.0)


def task_features(plans: Path, tasks: list[str]) -> pd.DataFrame:
    """Per task: pi0 chunk budget, Molmo calls, and scripted-step seconds at default costs."""
    rows = []
    for task in tasks:
        suite, t = task.rsplit("_t", 1)
        w = plan_work(load_program(plans, plan_name(suite, int(t))))
        steps_s = sum(step_cost(DEFAULT_COSTS, n) * k for n, k in w.steps.items())
        rows.append({"task": task, "chunks": w.chunks, "molmo": w.molmo_calls, "steps_s": steps_s})
    return pd.DataFrame(rows).set_index("task")


def nnls3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares for a handful of columns (exact, by active sets)."""
    best, best_err = np.zeros(x.shape[1]), float(np.sum(y ** 2))
    for k in range(1, x.shape[1] + 1):
        for cols in itertools.combinations(range(x.shape[1]), k):
            coef, *_ = np.linalg.lstsq(x[:, cols], y, rcond=None)
            if (coef < 0).any():
                continue
            full = np.zeros(x.shape[1])
            full[list(cols)] = coef
            err = float(np.sum((x @ full - y) ** 2))
            if err < best_err:
                best, best_err = full, err
    return best


def default_est(feat: pd.DataFrame) -> pd.Series:
    return feat["chunks"] * DEFAULT_COSTS["chunk_s"] + feat["molmo"] * DEFAULT_COSTS["molmo_s"] + feat["steps_s"]


def fit_est(feat: pd.DataFrame, lib, fit_tasks: list[str]) -> pd.Series:
    """Refit (chunk_s, molmo_s, step multiplier) on mean episode work of ``fit_tasks``."""
    work = pd.Series({t: np.mean([e.work_time for e in lib if e.task == t]) for t in fit_tasks})
    x = feat.loc[fit_tasks, ["chunks", "molmo", "steps_s"]].to_numpy(float)
    coef = nnls3(x, work.loc[fit_tasks].to_numpy(float))
    return feat[["chunks", "molmo", "steps_s"]] @ coef


def jit(est: list[float], slack: float, prior: float):
    def pick(t, pending, state):
        return jit_pick(t, [est[i] for i in pending], [(est[i], t0) for i, t0 in state["running"].items()],
                        [(est[i], w) for i, w in state["finished"]], state["n"], slack, prior)
    return pick


def gain(work, n, est: list[float], slack: float, prior: float, base: float) -> float:
    return (simulate(work, n, "fifo", pick=jit(est, slack, prior))["makespan_s"] / base - 1) * 100


def spearman(a, b) -> float:
    return float(pd.Series(list(a)).rank().corr(pd.Series(list(b)).rank()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("library")
    ap.add_argument("--plans", required=True, help="directory with <suite>_t<task>_plan.json")
    ap.add_argument("--splits", type=int, default=40)
    ap.add_argument("--tune-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--workloads", type=int, default=40, help="batches per cell for 'unseen' and 'grid'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lib = [e for e in load_library(Path(args.library)) if e.level == 8]
    tasks = sorted({e.task for e in lib})
    feat = task_features(Path(args.plans), tasks)
    d_est = default_est(feat)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out) if args.out else Path(args.library).parent
    out.mkdir(parents=True, exist_ok=True)
    true_work = pd.Series({t: np.mean([e.work_time for e in lib if e.task == t]) for t in tasks})

    # -- unseen: tasks that did not shape the cost table, nothing refit --------
    unseen_tasks = [t for t in tasks if t not in COST_TASKS]
    sub = [e for e in lib if e.task in unseen_tasks]
    rows = []
    for k, n in itertools.product((32, 64, 128), (8, 16)):
        for w in range(args.workloads):
            work = sample_workload(sub, k, rng)
            base = simulate(work, n, "fifo")["makespan_s"]
            rows.append({"batch": k, "level": n, "workload": w, "arm": "plan_default",
                         "pct": gain(work, n, [d_est[e.task] for e in work], 1.0, 1.5, base)})
            rows.append({"batch": k, "level": n, "workload": w, "arm": "oracle",
                         "pct": gain(work, n, [e.work_time for e in work], 1.0, 1.5, base)})
    unseen = pd.DataFrame(rows)
    unseen.to_csv(out / "holdout_unseen_runs.csv", index=False)
    rho_unseen = spearman(d_est[unseen_tasks], true_work[unseen_tasks])

    # -- split: refit costs and knobs on half, test on the other half ----------
    rows, rho_rows = [], []
    for s in range(args.splits):
        perm = list(rng.permutation(tasks))
        fit_t, test_t = sorted(perm[:8]), sorted(perm[8:])
        f_est = fit_est(feat, lib, fit_t)
        fit_lib = [e for e in lib if e.task in fit_t]
        test_lib = [e for e in lib if e.task in test_t]
        # choose knobs on the fit half (fitted costs)
        scores = {}
        tune = [(sample_workload(fit_lib, 32, rng), n) for n in (8, 16) for _ in range(args.tune_batches)]
        tune = [(w, n, simulate(w, n, "fifo")["makespan_s"]) for w, n in tune]
        for sl, pr in itertools.product(SLACKS, PRIORS):
            scores[(sl, pr)] = np.mean([gain(w, n, [f_est[e.task] for e in w], sl, pr, b) for w, n, b in tune])
        sl, pr = min(scores, key=scores.get)
        rho_rows.append({"split": s, "slack": sl, "prior": pr,
                         "rho_default": spearman(d_est[test_t], true_work[test_t]),
                         "rho_fitted": spearman(f_est[test_t], true_work[test_t])})
        for k, n in itertools.product((32, 64), (8, 16)):
            for w in range(args.eval_batches):
                work = sample_workload(test_lib, k, rng)
                base = simulate(work, n, "fifo")["makespan_s"]
                common = {"split": s, "batch": k, "level": n, "workload": w}
                rows.append({**common, "arm": "plan_default",
                             "pct": gain(work, n, [d_est[e.task] for e in work], 1.0, 1.5, base)})
                rows.append({**common, "arm": "plan_fitted_tuned",
                             "pct": gain(work, n, [f_est[e.task] for e in work], sl, pr, base)})
                rows.append({**common, "arm": "oracle",
                             "pct": gain(work, n, [e.work_time for e in work], 1.0, 1.5, base)})
    split = pd.DataFrame(rows)
    split.to_csv(out / "holdout_split_runs.csv", index=False)
    picks = pd.DataFrame(rho_rows)
    picks.to_csv(out / "holdout_split_picks.csv", index=False)

    # -- grid: knob sensitivity on all tasks -----------------------------------
    rows = []
    for k, n in itertools.product((32, 128), (8, 16)):
        for w in range(args.workloads // 2):
            work = sample_workload(lib, k, rng)
            base = simulate(work, n, "fifo")["makespan_s"]
            for sl, pr in itertools.product(SLACKS, PRIORS):
                rows.append({"batch": k, "level": n, "workload": w, "slack": sl, "prior": pr,
                             "pct": gain(work, n, [d_est[e.task] for e in work], sl, pr, base)})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "holdout_grid_runs.csv", index=False)

    # -- report ----------------------------------------------------------------
    def ci(x):
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(f"UNSEEN: {len(unseen_tasks)} tasks that did not shape the cost table; default costs and knobs")
        print(f"  Spearman(plan estimate, mean work) over these tasks: {rho_unseen:.2f}")
        t = unseen.groupby(["batch", "level", "arm"])["pct"].agg(["mean", ci]).unstack("arm")
        print(t.to_string(), "\n")

        print(f"SPLIT: {args.splits} random 8/8 task splits; costs and knobs chosen on one half, tested on the other")
        t = split.groupby(["batch", "level", "arm"])["pct"].agg(["mean", ci]).unstack("arm")
        print(t.to_string())
        sw = split.pivot_table(index=["split", "batch", "level", "workload"], columns="arm", values="pct")
        d = sw["plan_fitted_tuned"] - sw["plan_default"]
        print(f"  fitted+tuned minus default: {d.mean():+.2f} pp (95% CI ±{ci(d):.2f}; negative = refitting helps)")
        print(f"  share of oracle gain, default: {sw['plan_default'].mean() / sw['oracle'].mean():.0%}")
        print(f"  test-half Spearman: default {picks['rho_default'].mean():.2f}, "
              f"fitted {picks['rho_fitted'].mean():.2f}")
        print("  knobs picked on the fit half:", picks.groupby(["slack", "prior"]).size().to_dict(), "\n")

        print("GRID: makespan change by (slack, prior_scale), all tasks, default costs")
        g = grid.groupby(["batch", "level", "slack", "prior"])["pct"].mean().unstack(["slack", "prior"])
        print(g.to_string())
        spread = grid.groupby(["batch", "level", "slack", "prior"])["pct"].mean().groupby(level=[0, 1])
        print("  best - worst knob setting per cell (pp):", (spread.max() - spread.min()).round(1).to_dict())
    print(f"\n-> {out}/holdout_unseen_runs.csv, holdout_split_runs.csv, holdout_split_picks.csv, "
          "holdout_grid_runs.csv")


if __name__ == "__main__":
    main()
