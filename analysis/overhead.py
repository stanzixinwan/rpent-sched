"""X0: episode wall time with and without the tracing, from the start/end records.

    python analysis/overhead.py results/overhead/<stamp>

Expects <root>/<arm>/N<level>_r<rep>/traces/episode_*.jsonl for the arms
``full``, ``proxy_only`` and ``bare`` (scripts/run_overhead.sh).
Every arm writes ``episode_start`` / ``episode_end``, so wall time is measured
the same way in all three. Reports, per level, the mean episode time of each
arm and the paired difference to ``bare`` (same repeat) with a 95% t-interval,
and writes overhead.csv next to the cells.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def cell_mean(cell: Path) -> tuple[float, int]:
    walls = []
    for f in (cell / "traces").glob("episode_*.jsonl"):
        start = end = None
        for line in f.open():
            if '"episode_start"' in line:
                start = json.loads(line)["t"]
            elif '"episode_end"' in line:
                rec = json.loads(line)
                end = rec["t"] if rec.get("exit_code") is not None else None
        if start is not None and end is not None:
            walls.append(float(end) - float(start))
    return (sum(walls) / len(walls) if walls else math.nan), len(walls)


def main(argv=None) -> None:
    root = Path((argv or sys.argv[1:])[0])
    rows = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in ("full", "proxy_only", "bare")):
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
        for arm in ("proxy_only", "full"):
            if arm not in g or "bare" not in g:
                continue
            d = ((g[arm] - g["bare"]) / g["bare"] * 100).dropna()
            n = len(d)
            half = T95.get(n - 1, 1.96) * d.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
            out.append({"level": level, "arm": arm, "repeats": n, "bare_wall_s": g["bare"].mean(),
                        "arm_wall_s": g[arm].mean(), "overhead_pct": d.mean(), "ci95_half": half})
    res = pd.DataFrame(out)
    res.to_csv(root / "overhead.csv", index=False)
    print(cells.groupby(["level", "arm"])[["episodes", "wall_mean_s"]].mean().round(2).to_string())
    print(res.round(2).to_string(index=False))
    print(f"-> {root / 'overhead.csv'}")


if __name__ == "__main__":
    main()
