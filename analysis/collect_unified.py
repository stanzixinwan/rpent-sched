"""Collect every unified-platform result into the tables the paper uses.

    python analysis/collect_unified.py [--out results/paper/unified]

All inputs come from one RunPod A100-SXM4-80GB host (platform.txt in each
result directory; the script checks the host). Each experiment directory must
already have its own analysis (analysis/*.csv from launch_compare.py or
overhead.py, anatomy/ from anatomy.py); the script only gathers and pairs.

Outputs (CSV):
  inventory     experiment, directory, host, cells, what it measures
  model         per model: call time, kernel-busy share, launches, copies, energy (M1)
  batch_cost    Pi0.5 batch latency c(b) relative to an ordinary single call (M1)
  scaling       E8 per (episode vCPUs, N): throughput, time shares, GPU use, energy
  levers        every paired lever result: makespan (paired by repeat) and episode
                time (paired by episode: same task, seed and repeat; geometric mean),
                95% intervals, and the first-order bound from the baseline's shares
  migration     per lever experiment and arm: seconds per episode in each part
  colocation    time-slice vs MPS inflation (B1b)
  overhead      tracing cost (X0)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import pandas as pd

R = Path("results")
HOST = "8d12c4d9216d"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

DIRS = {
    "M1": "model_profile/20261118-175825",
    "X0": "overhead/20261118-184851",
    "E1": "profile/20261118-215357",
    "E8": "cpu_scaling/20261119-002514",
    "G3": "g3/20261119-161253",
    "G6": "g6/20261119-180501",
    "B1b": "colocation/20261120-155756",
    "X8": "artifacts/20261120-182424",
    "X9": "molmo_replicas/20261120-203315",
    "X10": "molmo_replicas/20261121-024728",
}
WHAT = {
    "M1": "single-model profiling: host vs device, stages, ops, roofline, batch, energy",
    "X0": "tracing cost: full / proxy only / bare",
    "E1": "py-spy of the episode and env_server processes",
    "E8": "concurrency x episode vCPUs (4/8/12/18), N=1..32",
    "G3": "request reordering at N=16: none / history_srpt / intent_srpt",
    "G6": "cross-episode VLA batching: nobatch / greedy, 8 repeats",
    "B1b": "co-location interference: time-slice and MPS in one session",
    "X8": "RPent artifact saving: store / lazy",
    "X9": "Molmo serial lock: one / two replicas",
    "X10": "lever composition: Molmo replicas x VLA batching (2x2)",
}
# lever experiments: baseline arm, the arms, and where the arms live
LEVERS = {
    "G3": ("none", ["history_srpt", "intent_srpt"]),
    "G6": ("nobatch", ["greedy"]),
    "X8": ("store", ["lazy"]),
    "X9": ("one", ["two"]),
    "X10": ("one", ["one_batch", "two", "two_batch"]),
}
LABEL = {
    ("G3", "history_srpt"): "request reordering, history SRPT",
    ("G3", "intent_srpt"): "request reordering, plan (intent) SRPT",
    ("G6", "greedy"): "VLA batching (greedy)",
    ("X8", "lazy"): "lazy artifact saving",
    ("X9", "two"): "second Molmo replica",
    ("X10", "one_batch"): "VLA batching (greedy)",
    ("X10", "two"): "second Molmo replica",
    ("X10", "two_batch"): "second Molmo replica + VLA batching",
}
CELL = re.compile(r"N(\d+)_r(\d+)")
MAX_BATCH = 16


def host_of(d: Path) -> str | None:
    p = d / "platform.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    for m in sorted(d.glob("**/manifest.json"))[:1]:
        return json.loads(m.read_text()).get("host")
    return None


def cells(root: Path, arm: str):
    for c in sorted((root / arm).iterdir()):
        m = CELL.fullmatch(c.name)
        if m and (c / "manifest.json").exists():
            yield int(m[1]), int(m[2]), c


def episode_walls(cell: Path) -> dict[tuple, float]:
    out = {}
    for f in (cell / "traces").glob("episode_*.jsonl"):
        start = end = key = None
        with f.open() as fh:
            for line in fh:
                if '"episode_start"' in line:
                    r = json.loads(line)
                    a = r.get("argv") or []
                    g = lambda k: a[a.index(k) + 1] if k in a else None  # noqa: E731
                    key, start = (g("--suite"), g("--task"), g("--seed")), r["t"]
                elif '"episode_end"' in line:
                    r = json.loads(line)
                    end = r["t"] if r.get("exit_code") == 0 else None
        if key and start and end:
            k = key if key not in out else key + (f.stem,)
            out[k] = end - start
    return out


def makespan(cell: Path) -> float:
    m = json.loads((cell / "manifest.json").read_text())
    return m["t_end"] - m["t_start"]


def t_ci(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    m = st.mean(xs)
    h = T95.get(n - 1, 1.96) * st.stdev(xs) / math.sqrt(n) if n > 1 else math.nan
    return m, m - h, m + h


def geo_ci(ratios: list[float]) -> tuple[float, float, float, int]:
    lr = [math.log(x) for x in ratios]
    n = len(lr)
    m = st.mean(lr)
    se = st.stdev(lr) / math.sqrt(n)
    f = lambda v: 100 * (math.exp(v) - 1)  # noqa: E731
    return f(m), f(m - 1.96 * se), f(m + 1.96 * se), n


def shares(root: Path) -> pd.DataFrame:
    p = root / "anatomy" / "shares.csv"
    if not p.exists():
        sys.exit(f"{p} missing: run analysis/anatomy.py over {root} first (README, Experiments)")
    s = pd.read_csv(p)
    return s[s["outcome"] == "all"].set_index("set")


def batch_cost(m1: Path) -> dict[int, float]:
    b = pd.read_csv(m1 / "analysis" / "batch.csv")
    e = pd.read_csv(m1 / "analysis" / "energy.csv")
    solo = float(e[(e.service == "vla") & (e.phase == "timing")]["lat_p50_ms"].iloc[0])
    return {int(r.batch): float(r.lat_p50_ms) / solo for r in b.itertuples()}


def c_of(c: dict[int, float], b: int) -> float:
    ks = sorted(c)
    if b in c:
        return c[b]
    lo = max(k for k in ks if k < b)
    hi = min(k for k in ks if k > b)
    return c[lo] + (c[hi] - c[lo]) * (b - lo) / (hi - lo)


def bound(exp: str, arm: str, s: pd.Series, n: int, c: dict[int, float]) -> tuple[float, str]:
    """First-order upper bound (% of mean episode time) from the baseline's shares."""
    b = min(n, MAX_BATCH)
    batch = s.vla_wait + s.vla_svc * (1 - c_of(c, b) / b)
    if arm.endswith("srpt"):
        return 100 * (s.vla_wait + s.molmo_wait), "all VLA and Molmo queueing"
    if arm in ("greedy", "one_batch"):
        return 100 * batch, f"VLA queueing + VLA service x (1 - c({b})/{b})"
    if arm == "two":
        return 100 * s.molmo_wait, "all Molmo queueing"
    if arm == "two_batch":
        return 100 * (batch + s.molmo_wait), "both of the above"
    if arm == "lazy":
        return math.nan, "E1: compression + PNG encoding = 21-27% of the episode main thread"
    return math.nan, ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper/unified")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inv = []
    for exp, d in DIRS.items():
        root = R / d
        h = host_of(root)
        if h != HOST:
            sys.exit(f"{root}: host {h}, expected {HOST}")
        n = len(list(root.glob("**/manifest.json")))
        inv.append({"experiment": exp, "dir": str(root), "host": h, "cells": n, "measures": WHAT[exp]})
    pd.DataFrame(inv).to_csv(out / "inventory.csv", index=False)

    # M1
    m1 = R / DIRS["M1"]
    hd = pd.read_csv(m1 / "analysis" / "host_device.csv")
    en = pd.read_csv(m1 / "analysis" / "energy.csv")
    en = en[en.phase == "timing"].set_index("service")
    model = hd[["service", "dtype", "method_ms", "kernel_busy_ms", "launches_per_call",
                "launch_cpu_ms", "copy_ms", "host_other_ms"]].copy()
    model["kernel_busy_share"] = model.kernel_busy_ms / model.method_ms
    model["copy_share"] = model.copy_ms / model.method_ms
    model["power_W"] = model.service.map(en.power_mean_W)
    model["J_per_call_net"] = model.service.map(en.J_per_item_net)
    model.to_csv(out / "model.csv", index=False)
    c = batch_cost(m1)
    pd.DataFrame({"batch": list(c), "c_b_vs_single_call": list(c.values())}).to_csv(out / "batch_cost.csv", index=False)

    # E8
    e8 = R / DIRS["E8"]
    sh = shares(e8)
    gpu = pd.read_csv(e8 / "gpu.csv").groupby("set").agg(
        gpu_util=("util_mean", "mean"), kJ_per_episode=("energy_per_episode_kj", "mean"))
    rows = []
    for m in sorted(e8.glob("c*/N*_r*/manifest.json")):
        d = json.loads(m.read_text())
        a, b = d["cpu_start"]["cgroup_cpu_stat"], d["cpu_end"]["cgroup_cpu_stat"]
        rows.append({"set": f"{m.parts[-3]}_N{d['level']}", "vcpus": int(m.parts[-3][1:]), "N": d["level"],
                     "makespan_s": d["t_end"] - d["t_start"], "episodes": len(d["episodes"]),
                     "throttled_periods": b["nr_throttled"] - a["nr_throttled"]})
    sc = pd.DataFrame(rows).groupby(["set", "vcpus", "N"], as_index=False).agg(
        repeats=("makespan_s", "size"), makespan_s=("makespan_s", "mean"),
        episodes=("episodes", "mean"), throttled_periods=("throttled_periods", "sum"))
    sc["episodes_per_hour"] = 3600 * sc.episodes / sc.makespan_s
    keep = ["wall_mean_s", "g_process", "g_simulator", "g_client", "g_gpu_service", "g_gpu_queue",
            "molmo_wait", "vla_wait", "sam3_wait", "sim_reset", "local_tools"]
    sc = sc.join(sh[keep], on="set").join(gpu, on="set").sort_values(["vcpus", "N"])
    sc.to_csv(out / "scaling.csv", index=False)

    # levers
    lev, mig = [], []
    for exp, (base, arms) in LEVERS.items():
        root = R / DIRS[exp]
        sh = shares(root)
        walls = {}
        for arm in [base, *arms]:
            for n, r, cell in cells(root, arm):
                walls[(arm, n, r)] = (makespan(cell), episode_walls(cell))
        levels = sorted({n for (_, n, _) in walls})
        for arm in [base, *arms]:
            for n in levels:
                key = f"{arm}_N{n}"
                if key in sh.index:
                    s = sh.loc[key]
                    w = s.wall_mean_s
                    mig.append({"experiment": exp, "arm": arm, "N": n, "wall_mean_s": w,
                                "molmo_wait_s": w * s.molmo_wait, "vla_wait_s": w * s.vla_wait,
                                "vla_svc_s": w * s.vla_svc, "molmo_svc_s": w * s.molmo_svc,
                                "sim_s": w * s.g_simulator, "client_s": w * s.g_client,
                                "process_s": w * s.g_process})
        for arm in arms:
            for n in levels:
                reps = sorted(r for (a, nn, r) in walls if a == arm and nn == n and (base, n, r) in walls)
                if not reps:
                    continue
                mk = [100 * (walls[(arm, n, r)][0] / walls[(base, n, r)][0] - 1) for r in reps]
                ratios = []
                for r in reps:
                    bw, aw = walls[(base, n, r)][1], walls[(arm, n, r)][1]
                    ratios += [aw[k] / bw[k] for k in aw if k in bw]
                m, lo, hi = t_ci(mk)
                g, glo, ghi, ne = geo_ci(ratios)
                bkey = f"{base}_N{n}"
                bd, how = bound(exp, arm, sh.loc[bkey], n, c) if bkey in sh.index else (math.nan, "")
                lev.append({"experiment": exp, "lever": LABEL[(exp, arm)], "arm": arm, "baseline": base,
                            "N": n, "repeats": len(reps),
                            "makespan_pct": m, "makespan_ci_lo": lo, "makespan_ci_hi": hi,
                            "repeats_faster": sum(x < 0 for x in mk),
                            "episode_pct": g, "episode_ci_lo": glo, "episode_ci_hi": ghi, "paired_episodes": ne,
                            "bound_pct": -bd if not math.isnan(bd) else math.nan, "bound_from": how})
    pd.DataFrame(lev).to_csv(out / "levers.csv", index=False)
    pd.DataFrame(mig).to_csv(out / "migration.csv", index=False)

    # B1b, X0
    b1 = R / DIRS["B1b"]
    col = []
    for mode in ("timeslice", "mps"):
        t = pd.read_csv(b1 / mode / "inflation.csv")
        t.insert(0, "mode", mode)
        col.append(t)
    pd.concat(col).to_csv(out / "colocation.csv", index=False)
    pd.read_csv(R / DIRS["X0"] / "overhead.csv").to_csv(out / "overhead.csv", index=False)

    lv = pd.DataFrame(lev)
    pd.set_option("display.width", 200)
    print(lv[["experiment", "arm", "N", "makespan_pct", "makespan_ci_lo", "makespan_ci_hi",
              "episode_pct", "episode_ci_lo", "episode_ci_hi", "bound_pct"]].round(1).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
