"""X0: episode wall time with and without the tracing, from the start/end records.

    python analysis/overhead.py results/overhead/<stamp>
    python analysis/overhead.py results/artifacts/<stamp> --baseline store --arms lazy

Expects <root>/<arm>/N<level>_r<rep>/traces/episode_*.jsonl for the arms
``full``, ``proxy_only`` and ``bare`` (scripts/run_overhead.sh).
Every arm writes ``episode_start`` / ``episode_end``, so wall time is measured
the same way in all three. Reports, per level, the mean episode time of each
arm and the paired difference to ``bare``: per repeat (cell means) and per
episode (same suite, task, seed and repeat; geometric mean of the time ratios),
each with a 95% interval, and writes overhead.csv next to the cells.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _arg(argv: list, flag: str):
    return argv[argv.index(flag) + 1] if flag in argv else None


def cell_walls(cell: Path) -> dict[tuple, float]:
    """Episode wall time keyed by (suite, task, seed)."""
    walls = {}
    for f in (cell / "traces").glob("episode_*.jsonl"):
        start = end = key = None
        for line in f.open():
            if '"episode_start"' in line:
                rec = json.loads(line)
                start = rec["t"]
                argv = rec.get("argv") or []
                key = (_arg(argv, "--suite"), _arg(argv, "--task"), _arg(argv, "--seed"), f.stem)
            elif '"episode_end"' in line:
                rec = json.loads(line)
                end = rec["t"] if rec.get("exit_code") is not None else None
        if start is not None and end is not None:
            # same task and seed may appear twice in a cell; keep them apart by order
            k = key[:3] if key[:3] not in walls else key
            walls[k] = float(end) - float(start)
    return walls


def cell_mean(cell: Path) -> tuple[float, int]:
    walls = list(cell_walls(cell).values())
    return (sum(walls) / len(walls) if walls else math.nan), len(walls)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--baseline", default="bare")
    ap.add_argument("--arms", nargs="+", default=["proxy_only", "full"])
    args = ap.parse_args(argv)
    root = Path(args.root)
    base_arm, arms = args.baseline, args.arms
    known = {base_arm, *arms}
    rows = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in known):
        for cell in sorted(arm_dir.glob("N*_r*")):
            level, rep = cell.name[1:].split("_r")
            mean, n = cell_mean(cell)
            rows.append({"arm": arm_dir.name, "level": int(level), "repeat": int(rep),
                         "episodes": n, "wall_mean_s": mean})
    if not rows:
        sys.exit(f"no cells under {root}")
    cells = pd.DataFrame(rows)
    wide = cells.pivot_table(index=["level", "repeat"], columns="arm", values="wall_mean_s")
    out = []
    for level, g in wide.groupby(level="level"):
        for arm in arms:
            if arm not in g or base_arm not in g:
                continue
            d = ((g[arm] - g[base_arm]) / g[base_arm] * 100).dropna()
            n = len(d)
            half = T95.get(n - 1, 1.96) * d.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
            out.append({"level": level, "arm": arm, "repeats": n, "base_wall_s": g[base_arm].mean(),
                        "arm_wall_s": g[arm].mean(), "overhead_pct": d.mean(), "ci95_half": half})
    # per-episode pairing: same task, seed and repeat in both arms
    eps = {}
    for arm_dir in (p for p in root.iterdir() if p.is_dir() and p.name in known):
        for cell in arm_dir.glob("N*_r*"):
            level, rep = cell.name[1:].split("_r")
            for k, w in cell_walls(cell).items():
                eps.setdefault((int(level), int(rep), k), {})[arm_dir.name] = w
    for row in out:
        lr = [math.log(v[row["arm"]] / v[base_arm]) for (lvl, _, _), v in eps.items()
              if lvl == row["level"] and row["arm"] in v and base_arm in v]
        n = len(lr)
        if n > 1:
            m = sum(lr) / n
            se = math.sqrt(sum((x - m) ** 2 for x in lr) / (n - 1) / n)
            row.update({"paired_episodes": n, "ep_overhead_pct": 100 * (math.exp(m) - 1),
                        "ep_ci95_lo": 100 * (math.exp(m - 1.96 * se) - 1),
                        "ep_ci95_hi": 100 * (math.exp(m + 1.96 * se) - 1)})
    res = pd.DataFrame(out)
    res.to_csv(root / "overhead.csv", index=False)
    print(cells.groupby(["level", "arm"])[["episodes", "wall_mean_s"]].mean().round(2).to_string())
    print(res.round(2).to_string(index=False))
    print(f"-> {root / 'overhead.csv'}")


if __name__ == "__main__":
    main()
