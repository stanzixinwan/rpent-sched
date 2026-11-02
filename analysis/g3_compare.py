"""Compare G3 scheduling modes: does ordering queued model requests by planner
intent beat FIFO and history?

    python analysis/g3_compare.py results/g3/<stamp> [--out <dir>]

Input layout (scripts/run_g3.sh): ``<stamp>/<mode>/N<level>_r<repeat>/``.
Every cell is summarized with ``analysis/breakdown.py``; modes are then compared
**paired** with FIFO (mode ``none``) at the same N and repeat, since repeats
share the machine state of their time slot.

Outputs (default ``<stamp>/analysis``):

    runs.csv       one row per cell: makespan, throughput, episode time
                   (mean, p50, p95), VLA/Molmo queueing, success, CPU throttling
    summary.csv    mean and std over repeats per (mode, N), plus the paired
                   change vs FIFO in percent
    predict.csv    how well each remaining-work estimate ranks episodes by
                   their true remaining time (Spearman), and plan vs actual
                   VLA/Molmo calls per episode
    g3_compare.png makespan, mean and p95 episode time vs FIFO, per N

A cell whose CPU throttling exceeds 5% of its makespan is marked
``cpu_limited``: it measured CPU contention, not model queueing, and is left
out of the comparison (it stays in runs.csv).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from breakdown import load_level, summarize_level  # noqa: E402

from rsched.traceio import read_jsonl  # noqa: E402

BASELINE = "none"
MODE_ORDER = ["none", "history_srpt", "intent_srpt", "history_lrpt", "intent_lrpt"]
MODE_LABEL = {"none": "FIFO", "history_srpt": "History-SRPT", "intent_srpt": "Intent-SRPT",
              "history_lrpt": "History-LRPT", "intent_lrpt": "Intent-LRPT"}
METRICS = ["makespan_s", "episodes_per_hour", "wall_mean_s", "wall_p50_s", "wall_p95_s",
           "vla_wait_mean_ms", "molmo_wait_mean_ms", "success_rate"]
CPU_LIMIT = 0.05


def cells(root: Path) -> list[tuple[str, Path]]:
    out = []
    for mode_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis"):
        for cell in sorted(mode_dir.iterdir()):
            if (cell / "manifest.json").exists():
                out.append((mode_dir.name, cell))
    return out


def throttle(manifest: dict) -> tuple[float, float]:
    """(throttled seconds, throttled periods) during the level, from cgroup cpu.stat."""
    a = (manifest.get("cpu_start") or {}).get("cgroup_cpu_stat") or {}
    b = (manifest.get("cpu_end") or {}).get("cgroup_cpu_stat") or {}
    if "throttled_time" in a and "throttled_time" in b:  # v1: nanoseconds
        secs = (b["throttled_time"] - a["throttled_time"]) / 1e9
    elif "throttled_usec" in a and "throttled_usec" in b:  # v2: microseconds
        secs = (b["throttled_usec"] - a["throttled_usec"]) / 1e6
    else:
        return np.nan, np.nan
    return secs, b.get("nr_throttled", 0) - a.get("nr_throttled", 0)


def summarize_cell(mode: str, cell: Path) -> tuple[dict, pd.DataFrame]:
    eps, rpc, _tools, manifest = load_level(cell)
    manifest["_dir"] = str(cell)
    row = summarize_level(eps, rpc, manifest)
    row["mode"] = mode
    if not rpc.empty:
        molmo = rpc[(rpc["service"] == "molmo") & rpc["gated"]]
        if not molmo.empty:
            row["molmo_wait_mean_ms"] = molmo["wait_s"].mean() * 1e3
            row["molmo_wait_p95_ms"] = molmo["wait_s"].quantile(0.95) * 1e3
    secs, periods = throttle(manifest)
    row["throttled_s"], row["throttled_periods"] = secs, periods
    row["throttle_frac"] = secs / row["makespan_s"] if row.get("makespan_s") else np.nan
    row["cpu_limited"] = bool(row["throttle_frac"] > CPU_LIMIT) if row["throttle_frac"] == row["throttle_frac"] else False
    row["cpu_quota"] = (manifest.get("cpu_start") or {}).get("cgroup_cpu_max")
    row["cell"] = str(cell)
    eps = eps.assign(mode=mode)
    return row, eps


def paired_change(runs: pd.DataFrame) -> pd.DataFrame:
    """Percent change of each metric vs FIFO at the same (level, repeat)."""
    ok = runs[~runs["cpu_limited"]]
    base = ok[ok["mode"] == BASELINE].set_index(["level", "repeat"])
    rows = []
    for r in ok.itertuples(index=False):
        key = (r.level, r.repeat)
        if r.mode == BASELINE or key not in base.index:
            continue
        b = base.loc[key]
        rec = {"mode": r.mode, "level": r.level, "repeat": r.repeat}
        for m in METRICS:
            v, bv = getattr(r, m, np.nan), b.get(m, np.nan)
            rec[f"{m}_vs_fifo_pct"] = (v / bv - 1) * 100 if bv and bv == bv and v == v else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[~runs["cpu_limited"]]
    agg = ok.groupby(["mode", "level"])[[m for m in METRICS if m in ok]].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    change = paired_change(runs)
    if not change.empty:
        ch = change.groupby(["mode", "level"]).agg(["mean", "std"])
        ch = ch.drop(columns=[c for c in ch.columns if c[0] == "repeat"])
        ch.columns = ["_".join(c) for c in ch.columns]
        agg = agg.join(ch, how="left")
    agg = agg.reset_index()
    agg["order"] = agg["mode"].map({m: i for i, m in enumerate(MODE_ORDER)})
    return agg.sort_values(["level", "order"]).drop(columns="order")


def prediction_quality(cell_list: list[tuple[str, Path]]) -> pd.DataFrame:
    """Rank agreement of remaining-work estimates with true remaining episode time."""
    rows = []
    for mode, cell in cell_list:
        manifest = json.loads((cell / "manifest.json").read_text())
        rem, plan_vs_actual = [], []
        for path in sorted((cell / "traces").glob("episode_*.jsonl")):
            recs = read_jsonl(path)
            end = next((r for r in recs if r["type"] == "episode_end"), None)
            work = next((r for r in recs if r["type"] == "plan_work"), None)
            if end is None or work is None:
                continue
            for r in recs:
                if r["type"] == "remaining":
                    rem.append({"true_s": end["t"] - r["t"], "intent_s": r.get("intent_s"),
                                "history_s": r.get("history_s")})
            calls = [r.get("method") for r in recs if r["type"] == "client_rpc"]
            plan_vs_actual.append({"plan_chunks": work["chunks"], "vla": calls.count("vla.predict"),
                                   "plan_molmo": work["molmo_calls"], "molmo": calls.count("molmo.ground")})
        if not rem:
            continue
        df = pd.DataFrame(rem)
        pa = pd.DataFrame(plan_vs_actual)
        row = {"mode": mode, "level": manifest["level"], "repeat": manifest["repeat"], "n_points": len(df)}
        for est in ("intent_s", "history_s"):
            part = df.dropna(subset=[est])
            row[f"spearman_{est}"] = part["true_s"].corr(part[est], method="spearman") if len(part) > 2 else np.nan
            row[f"coverage_{est}"] = len(part) / len(df)
        row["vla_plan_minus_actual_mean"] = (pa["plan_chunks"] - pa["vla"]).mean()
        row["vla_plan_mae"] = (pa["plan_chunks"] - pa["vla"]).abs().mean()
        row["molmo_plan_mae"] = (pa["plan_molmo"] - pa["molmo"]).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot(change: pd.DataFrame, path: Path) -> None:
    if change.empty:
        return
    metrics = [("makespan_s", "makespan"), ("wall_mean_s", "mean episode time"),
               ("wall_p95_s", "p95 episode time")]
    levels = sorted(change["level"].unique())
    modes = [m for m in MODE_ORDER if m in set(change["mode"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), sharey=True)
    width = 0.8 / max(len(modes), 1)
    for ax, (m, title) in zip(np.atleast_1d(axes), metrics):
        col = f"{m}_vs_fifo_pct"
        for j, mode in enumerate(modes):
            part = change[change["mode"] == mode].groupby("level")[col].agg(["mean", "std"]).reindex(levels)
            x = np.arange(len(levels)) + (j - (len(modes) - 1) / 2) * width
            ax.bar(x, part["mean"], width, yerr=part["std"], capsize=2, label=MODE_LABEL.get(mode, mode))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(levels)), [f"N={lv}" for lv in levels])
        ax.set_title(f"{title} vs FIFO")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("change vs FIFO (%), lower is better")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("g3_dir", help="results/g3/<stamp>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.g3_dir)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    cell_list = cells(root)
    if not cell_list:
        sys.exit(f"no <mode>/N<level>_r<repeat>/manifest.json under {root}")
    rows, all_eps = [], []
    for mode, cell in cell_list:
        row, eps = summarize_cell(mode, cell)
        rows.append(row)
        all_eps.append(eps)
    runs = pd.DataFrame(rows).sort_values(["level", "repeat", "mode"])
    runs.to_csv(out / "runs.csv", index=False)
    pd.concat(all_eps, ignore_index=True).to_csv(out / "episodes.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(out / "summary.csv", index=False)
    pred = prediction_quality(cell_list)
    pred.to_csv(out / "predict.csv", index=False)
    plot(paired_change(runs), out / "g3_compare.png")

    limited = runs[runs["cpu_limited"]]
    if len(limited):
        print(f"!!! {len(limited)} cell(s) CPU-limited (throttled > {CPU_LIMIT:.0%} of makespan), "
              "left out of the comparison:")
        print(limited[["mode", "level", "repeat", "throttle_frac"]].to_string(index=False))
    crashed = pd.concat(all_eps)["crashed"].sum()
    if crashed:
        print(f"!!! {crashed} crashed episode(s) excluded from time stats (see episodes.csv)")
    cols = ["mode", "level", "makespan_s_mean", "makespan_s_vs_fifo_pct_mean", "wall_mean_s_mean",
            "wall_mean_s_vs_fifo_pct_mean", "wall_p95_s_mean", "wall_p95_s_vs_fifo_pct_mean",
            "molmo_wait_mean_ms_mean", "vla_wait_mean_ms_mean", "success_rate_mean", "makespan_s_count"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(summary[[c for c in cols if c in summary]].to_string(index=False))
        if not pred.empty:
            print("\nremaining-work estimates vs true remaining time (Spearman, mean over cells):")
            print(pred[["spearman_intent_s", "spearman_history_s", "vla_plan_mae",
                        "molmo_plan_mae"]].mean().to_string())
    print(f"\nwrote {out}/runs.csv, summary.csv, predict.csv, episodes.csv, g3_compare.png")


if __name__ == "__main__":
    main()
