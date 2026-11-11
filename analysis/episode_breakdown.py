"""Where does an episode's wall time go? Per-episode breakdown from rsched traces and proxy logs.

    python analysis/episode_breakdown.py <cell dir> [<cell dir> ...] --out <csv>

A cell directory is one sweep level (``.../N8_r0``) with ``traces/`` and
``proxy_*.jsonl``. Every episode's wall time (``episode_start`` to
``episode_end``) is split into:

    startup     process start to its first RPC (imports, building the robot)
    env_reset   ``env.reset`` (LIBERO scene load and reset)
    sim         other simulator RPCs (step, chunk_step, render_camera, raw_obs, ...)
    molmo_wait / molmo_svc, vla_wait / vla_svc, sam3_wait / sam3_svc
                model calls split into proxy queueing and service (the rest of the
                client-side time, HTTP and encoding, counts as service)
    local       time between RPCs inside the episode (Python: scripted motion,
                plan replay, image handling)
    teardown    last RPC to process exit

RPCs inside one episode run one at a time, so the parts add up to the wall time
(checked: ``overlap_s`` reports any excess).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

MODEL = {"vla.predict": "vla", "molmo.ground": "molmo", "sam3.segment": "sam3"}


def load_proxy(cell: Path) -> dict[tuple[str, str], list[dict]]:
    out = collections.defaultdict(list)
    for f in cell.glob("proxy_*.jsonl"):
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("episode") and r.get("method") in MODEL:
                out[(r["episode"], MODEL[r["method"]])].append(r)
    return out


def episode_rows(cell: Path) -> list[dict]:
    proxy = load_proxy(cell)
    rows = []
    for f in sorted((cell / "traces").glob("*.jsonl")):
        recs = [json.loads(line) for line in f.open() if line.strip()]
        start = next((float(r["t"]) for r in recs if r.get("type") == "episode_start"), None)
        end = next((float(r["t"]) for r in recs if r.get("type") == "episode_end"), None)
        rpc = sorted((r for r in recs if r.get("type") == "client_rpc" and not r.get("error")),
                     key=lambda r: float(r["t_start"]))
        tools = [r for r in recs if r.get("type") == "tool"]
        if start is None or end is None or not rpc or not tools:
            continue  # crashed before running anything
        ep = recs[0]["episode"]
        parts = collections.Counter()
        for r in rpc:
            m, d = r["method"], float(r["dur_s"])
            if m in MODEL:
                parts[MODEL[m]] += d
            elif m == "env.reset":
                parts["env_reset"] += d
            elif m.startswith("env."):
                parts["sim"] += d
            else:
                parts["other_rpc"] += d
        for kind in ("vla", "molmo", "sam3"):
            wait = sum(float(p.get("wait_s") or 0) for p in proxy.get((ep, kind), []))
            wait = min(wait, parts[kind])
            parts[f"{kind}_wait"] = wait
            parts[f"{kind}_svc"] = parts.pop(kind, 0.0) - wait
        first, last = float(rpc[0]["t_start"]), float(rpc[-1]["t_end"])
        rpc_total = sum(float(r["dur_s"]) for r in rpc)
        inside = last - first
        row = {"cell": str(cell), "episode": ep, "wall_s": end - start, "startup": first - start,
               "teardown": end - last, "local": max(inside - rpc_total, 0.0),
               "overlap_s": max(rpc_total - inside, 0.0), "tool_calls": len(tools),
               "vla_calls": sum(1 for r in rpc if r["method"] == "vla.predict"),
               "molmo_calls": sum(1 for r in rpc if r["method"] == "molmo.ground"), **parts}
        rows.append(row)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    df = pd.DataFrame([r for c in args.cells for r in episode_rows(Path(c))]).fillna(0.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    parts = ["startup", "env_reset", "sim", "local", "molmo_wait", "molmo_svc", "vla_wait", "vla_svc",
             "sam3_wait", "sam3_svc", "other_rpc", "teardown"]
    parts = [p for p in parts if p in df]
    share = df[parts].sum() / df["wall_s"].sum()
    print(f"{len(df)} episodes, mean wall {df['wall_s'].mean():.1f} s, max overlap {df['overlap_s'].max():.2f} s")
    print((share * 100).round(1).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
