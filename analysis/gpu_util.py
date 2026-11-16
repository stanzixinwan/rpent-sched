"""GPU utilization, memory and energy per sweep cell, from the sampler's gpu.jsonl.

    python analysis/gpu_util.py --set NAME=GLOB [--set ...] --out results/paper/gpu.csv

Per cell: mean / p50 / p95 utilization, share of samples with the GPU idle
(util < 5%), peak memory, and, when the sampler logged ``power_w`` (rsched.nvml
from the ISPASS branch on), mean power and energy per finished episode.
Utilization is NVML's coarse "a kernel was running" figure, so read it as a
busy fraction, not as how much of the GPU was used.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd


def cell_row(cell: Path, label: str) -> dict | None:
    f = cell / "gpu.jsonl"
    if not f.exists():
        return None
    recs = []
    with f.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "gpu" and r.get("gpu", 0) == 0:
                recs.append(r)
    if not recs:
        return None
    df = pd.DataFrame(recs)
    t0, t1 = df["t"].min(), df["t"].max()
    episodes = None
    manifest = cell / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        t0, t1 = float(m.get("t_start", t0)), float(m.get("t_end", t1))
        episodes = sum(1 for e in m.get("episodes", []) if e.get("tool_calls", 0) > 0)
        df = df[(df["t"] >= t0) & (df["t"] <= t1)]
    row = {"set": label, "cell": str(cell), "duration_s": t1 - t0, "episodes": episodes,
           "util_mean": df["util_gpu"].mean(), "util_p50": df["util_gpu"].median(),
           "util_p95": df["util_gpu"].quantile(0.95), "idle_frac": float((df["util_gpu"] < 5).mean()),
           "mem_peak_mb": df["used_mb"].max(), "power_mean_w": None, "energy_per_episode_kj": None}
    if "power_w" in df and df["power_w"].notna().any():
        p = df.dropna(subset=["power_w"]).sort_values("t")
        energy_j = float(((p["power_w"].shift() + p["power_w"]) / 2 * p["t"].diff()).sum())
        row["power_mean_w"] = p["power_w"].mean()
        if episodes:
            row["energy_per_episode_kj"] = energy_j / episodes / 1000
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME=GLOB", dest="sets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = []
    for spec in args.sets:
        label, pattern = spec.split("=", 1)
        rows += [r for c in sorted(glob.glob(pattern)) if (r := cell_row(Path(c), label))]
    if not rows:
        sys.exit("no gpu.jsonl found")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    num = ["util_mean", "util_p95", "idle_frac", "mem_peak_mb", "power_mean_w", "energy_per_episode_kj"]
    print(df.groupby("set", sort=False)[num].mean(numeric_only=True).round(2).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
