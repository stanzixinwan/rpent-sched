"""Episode anatomy for the ISPASS study: where wall time goes, one level deeper.

    python analysis/anatomy.py --set NAME=GLOB [--set NAME=GLOB ...] --out DIR

A set is a label plus a glob of sweep cells (directories with ``traces/`` and
``proxy_*.jsonl``), e.g. ``--set "replay_N8=results/g6/*/nobatch/N8_r*"``.

It extends ``analysis/episode_breakdown.py`` (same accounting; parts add up to
the wall time) with what the paper needs on top:

* simulator time by RPC method (``env.reset``, ``env.step``, ``env.chunk_step``,
  ``env.render_camera``, other ``env.*``),
* client-side time (gaps between RPCs) attributed to the tool that was running,
  or to ``between_tools`` (plan replay / planner),
* outcome (``environment_success``), suite/task/seed, planner tokens and turns,
* call counts and mean latency per RPC method.

Outputs under --out:
    episodes.csv   one row per episode (all components, seconds)
    shares.csv     per set: share of summed wall time per component, plus the
                   same split by outcome (success / failure)
    rpc.csv        per set and RPC method: calls/episode, ms/call, share
    local.csv      per set: client-side time by tool
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_breakdown import MODEL, load_proxy  # noqa: E402

SIM_PARTS = {"env.reset": "sim_reset", "env.step": "sim_step", "env.chunk_step": "sim_chunk_step",
             "env.render_camera": "sim_render"}
# Order used for tables and figures; groups are what the paper's text talks about.
COMPONENTS = ["startup", "teardown", "sim_reset", "sim_step", "sim_chunk_step", "sim_render", "sim_other",
              "local_tools", "local_between", "molmo_wait", "molmo_svc", "vla_wait", "vla_svc",
              "sam3_wait", "sam3_svc", "other_rpc"]
GROUPS = {
    "process": ["startup", "teardown"],
    "simulator": ["sim_reset", "sim_step", "sim_chunk_step", "sim_render", "sim_other"],
    "client": ["local_tools", "local_between"],
    "gpu_service": ["molmo_svc", "vla_svc", "sam3_svc"],
    "gpu_queue": ["molmo_wait", "vla_wait", "sam3_wait"],
}


def _load(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off by a killed episode
    return out


def episode_row(trace: Path, proxy: dict, cell: Path, label: str) -> tuple[dict, list[dict]] | None:
    recs = _load(trace)
    start = next((float(r["t"]) for r in recs if r.get("type") == "episode_start"), None)
    end = next((float(r["t"]) for r in recs if r.get("type") == "episode_end"), None)
    rpc = sorted((r for r in recs if r.get("type") == "client_rpc" and not r.get("error")),
                 key=lambda r: float(r["t_start"]))
    tools = sorted((r for r in recs if r.get("type") == "tool"), key=lambda r: float(r["t_start"]))
    if start is None or end is None or not rpc or not tools:
        return None  # crashed before running anything
    ep = recs[0]["episode"]
    outcome = next((r for r in recs if r.get("type") == "outcome"), {})
    parts: collections.Counter = collections.Counter()
    model_time: collections.Counter = collections.Counter()
    per_method = collections.defaultdict(lambda: [0, 0.0])
    for r in rpc:
        m, d = r["method"], float(r["dur_s"])
        per_method[m][0] += 1
        per_method[m][1] += d
        if m in MODEL:
            model_time[MODEL[m]] += d
        elif m in SIM_PARTS:
            parts[SIM_PARTS[m]] += d
        elif m.startswith("env."):
            parts["sim_other"] += d
        else:
            parts["other_rpc"] += d
    for kind in ("vla", "molmo", "sam3"):
        wait = min(sum(float(p.get("wait_s") or 0) for p in proxy.get((ep, kind), [])), model_time[kind])
        parts[f"{kind}_wait"] = wait
        parts[f"{kind}_svc"] = model_time[kind] - wait

    # Client-side time: gaps between consecutive RPCs, charged to the running tool.
    starts = [float(t["t_start"]) for t in tools]
    local_by_tool: collections.Counter = collections.Counter()
    for a, b in zip(rpc, rpc[1:]):
        gap = float(b["t_start"]) - float(a["t_end"])
        if gap <= 0:
            continue
        mid = float(a["t_end"]) + gap / 2
        i = bisect.bisect_right(starts, mid) - 1
        inside = i >= 0 and mid <= float(tools[i]["t_end"])
        local_by_tool[tools[i]["tool"] if inside else "between_tools"] += gap
    parts["local_between"] = local_by_tool.get("between_tools", 0.0)
    parts["local_tools"] = sum(v for k, v in local_by_tool.items() if k != "between_tools")

    first, last = float(rpc[0]["t_start"]), float(rpc[-1]["t_end"])
    task = outcome.get("task") or {}
    stats = outcome.get("stats") or {}
    row = {"set": label, "cell": str(cell), "episode": ep, "wall_s": end - start,
           "startup": first - start, "teardown": end - last,
           "success": outcome.get("environment_success"), "planner": outcome.get("planner"),
           "suite": task.get("suite"), "task": task.get("task"), "seed": task.get("seed"),
           "tool_calls": len(tools), "turns": stats.get("turns_used"),
           "tokens_in": stats.get("total_input_tokens"), "tokens_out": stats.get("total_output_tokens"),
           "vla_calls": per_method["vla.predict"][0], "molmo_calls": per_method["molmo.ground"][0],
           **{c: float(parts.get(c, 0.0)) for c in COMPONENTS if c not in ("startup", "teardown")}}
    row["unaccounted_s"] = row["wall_s"] - sum(row[c] for c in COMPONENTS)
    extra = [{"set": label, "episode": ep, "kind": "rpc", "name": m, "calls": n, "seconds": s}
             for m, (n, s) in per_method.items()]
    extra += [{"set": label, "episode": ep, "kind": "local", "name": k, "calls": 0, "seconds": v}
              for k, v in local_by_tool.items()]
    return row, extra


def collect(sets: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, extras = [], []
    for label, pattern in sets:
        cells = [Path(p) for p in sorted(glob.glob(pattern)) if (Path(p) / "traces").is_dir()]
        if not cells:
            print(f"warning: no cells match {pattern!r} (set {label})", file=sys.stderr)
        for cell in cells:
            proxy = load_proxy(cell)
            for trace in sorted((cell / "traces").glob("*.jsonl")):
                got = episode_row(trace, proxy, cell, label)
                if got:
                    rows.append(got[0])
                    extras += got[1]
    return pd.DataFrame(rows), pd.DataFrame(extras)


def shares(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    splits = [("all", df)]
    if df["success"].notna().any():
        splits += [("success", df[df["success"] == True]), ("failure", df[df["success"] == False])]  # noqa: E712
    for label, sub in ((s, d) for s, d in splits if len(d)):
        total = sub["wall_s"].sum()
        row = {"outcome": label, "episodes": len(sub), "wall_mean_s": sub["wall_s"].mean(),
               "wall_p95_s": sub["wall_s"].quantile(0.95)}
        row.update({c: sub[c].sum() / total for c in COMPONENTS})
        row.update({f"g_{g}": sum(sub[c].sum() for c in cs) / total for g, cs in GROUPS.items()})
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME=GLOB", dest="sets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sets = [tuple(s.split("=", 1)) for s in args.sets]
    eps, extra = collect(sets)
    if eps.empty:
        sys.exit("no episodes found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps.to_csv(out / "episodes.csv", index=False)

    sh = pd.concat([shares(g).assign(set=k) for k, g in eps.groupby("set", sort=False)])
    sh = sh[["set"] + [c for c in sh.columns if c != "set"]]
    sh.to_csv(out / "shares.csv", index=False)

    n_eps = eps.groupby("set")["episode"].count()
    wall = eps.groupby("set")["wall_s"].sum()
    agg = extra.groupby(["set", "kind", "name"], sort=False)[["calls", "seconds"]].sum().reset_index()
    agg["per_episode_s"] = agg["seconds"] / agg["set"].map(n_eps)
    agg["share"] = agg["seconds"] / agg["set"].map(wall)
    rpc = agg[agg["kind"] == "rpc"].copy()
    rpc["calls_per_episode"] = rpc["calls"] / rpc["set"].map(n_eps)
    rpc["ms_per_call"] = 1000 * rpc["seconds"] / rpc["calls"].clip(lower=1)
    rpc.drop(columns=["kind", "calls", "seconds"]).sort_values(["set", "share"], ascending=[True, False]).to_csv(
        out / "rpc.csv", index=False)
    agg[agg["kind"] == "local"].drop(columns=["kind", "calls", "seconds"]).sort_values(
        ["set", "share"], ascending=[True, False]).to_csv(out / "local.csv", index=False)

    pd.set_option("display.width", 200)
    show = sh[sh["outcome"] == "all"].set_index("set")
    print((show[[f"g_{g}" for g in GROUPS]] * 100).round(1).assign(
        episodes=show["episodes"], wall_mean_s=show["wall_mean_s"].round(1)).to_string())
    worst = eps["unaccounted_s"].abs().max()
    print(f"max |unaccounted| per episode: {worst:.2f} s  -> {out}")


if __name__ == "__main__":
    main()
