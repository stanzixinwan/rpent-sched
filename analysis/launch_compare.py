"""G4: did launching by planned work shorten the batch on the GPU?

    python analysis/launch_compare.py results/launch/<stamp> [--baseline given] [--out <dir>]

Input layout (scripts/run_launch.sh): ``<stamp>/<rule>/N<level>_r<repeat>/``;
``warmup/`` is ignored. Each rule is compared with the baseline at the same
repeat, where both started from the same shuffled batch.

Outputs (default ``<stamp>/analysis``):

    cells.csv     per cell: makespan, mean episode time, success, CPU throttling,
                  and how well launch order followed the plan (Spearman of
                  launch position vs planned work; negative = long first)
    paired.csv    per rule and repeat: makespan change vs the baseline (%)
    summary.csv   per rule: mean change, 95% CI (t, repeats - 1 df), repeats
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from breakdown import load_level  # noqa: E402

# Two-sided 97.5% t quantiles by degrees of freedom (no scipy).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}
CPU_LIMIT = 0.05


def t975(df: int) -> float:
    if df <= 0:
        return math.nan
    keys = sorted(k for k in T975 if k <= df)
    return T975[keys[-1]] if df <= 30 else 1.96


def throttle_frac(manifest: dict) -> float:
    a = (manifest.get("cpu_start") or {}).get("cgroup_cpu_stat") or {}
    b = (manifest.get("cpu_end") or {}).get("cgroup_cpu_stat") or {}
    span = manifest["t_end"] - manifest["t_start"]
    if "throttled_time" in a and "throttled_time" in b:
        return (b["throttled_time"] - a["throttled_time"]) / 1e9 / span
    if "throttled_usec" in a and "throttled_usec" in b:
        return (b["throttled_usec"] - a["throttled_usec"]) / 1e6 / span
    return math.nan


def cell_row(rule: str, cell: Path) -> dict:
    eps, _rpc, _tools, manifest = load_level(cell)
    ok = eps[~eps["crashed"]] if not eps.empty else eps
    launched = manifest.get("launched") or []
    est = [x["estimate_s"] for x in launched if x.get("estimate_s") is not None]
    order_rho = (pd.Series(range(len(est))).rank().corr(pd.Series(est).rank())
                 if len(est) > 2 else math.nan)
    return {
        "rule": rule, "level": manifest["level"], "repeat": manifest["repeat"],
        "makespan_s": manifest["t_end"] - manifest["t_start"],
        "episodes": len(eps), "crashed": int(eps["crashed"].sum()) if not eps.empty else 0,
        "wall_mean_s": float(ok["wall_s"].mean()) if len(ok) else math.nan,
        "success_rate": float((eps["success"] == True).mean()) if len(eps) else math.nan,  # noqa: E712
        "throttle_frac": throttle_frac(manifest),
        "launch_order_rho": order_rho,
        "planned_total_s": sum(est) if est else math.nan,
        "cell": str(cell),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("launch_dir", help="results/launch/<stamp>")
    ap.add_argument("--baseline", default="given")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.launch_dir)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rule_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("analysis", "warmup")):
        for cell in sorted(rule_dir.iterdir()):
            if (cell / "manifest.json").exists() and ".partial-" not in cell.name:
                rows.append(cell_row(rule_dir.name, cell))
    if not rows:
        sys.exit(f"no <rule>/N<level>_r<repeat>/manifest.json under {root}")
    cells = pd.DataFrame(rows).sort_values(["repeat", "rule"])
    cells["cpu_limited"] = cells["throttle_frac"] > CPU_LIMIT
    cells.to_csv(out / "cells.csv", index=False)

    base = cells[cells["rule"] == args.baseline].set_index(["level", "repeat"])
    paired = []
    for r in cells[cells["rule"] != args.baseline].itertuples(index=False):
        key = (r.level, r.repeat)
        if key not in base.index:
            continue
        b = base.loc[key]
        paired.append({"rule": r.rule, "level": r.level, "repeat": r.repeat,
                       "makespan_pct": (r.makespan_s / b["makespan_s"] - 1) * 100,
                       "wall_mean_pct": (r.wall_mean_s / b["wall_mean_s"] - 1) * 100,
                       "success_diff": r.success_rate - b["success_rate"],
                       "either_cpu_limited": bool(r.cpu_limited or b["cpu_limited"])})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)

    summary = []
    for (rule, level), g in paired.groupby(["rule", "level"]) if not paired.empty else []:
        x = g["makespan_pct"].to_numpy()
        half = t975(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else math.nan
        summary.append({"rule": rule, "level": level, "repeats": len(x),
                        "makespan_pct_mean": x.mean(), "ci95_low": x.mean() - half, "ci95_high": x.mean() + half,
                        "repeats_faster": int((x < 0).sum()),
                        "wall_mean_pct_mean": g["wall_mean_pct"].mean(),
                        "success_diff_mean": g["success_diff"].mean(),
                        "cpu_limited_pairs": int(g["either_cpu_limited"].sum())})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(cells[["rule", "level", "repeat", "makespan_s", "wall_mean_s", "success_rate", "crashed",
                     "throttle_frac", "launch_order_rho"]].to_string(index=False))
        if not summary.empty:
            print("\nmakespan vs", args.baseline, "(paired by repeat):")
            print(summary.to_string(index=False))
    if cells["cpu_limited"].any():
        print(f"!!! {int(cells['cpu_limited'].sum())} cell(s) throttled > {CPU_LIMIT:.0%} of makespan")
    print(f"\nwrote {out}/cells.csv, paired.csv, summary.csv")


if __name__ == "__main__":
    main()
