"""Where does episode time go? Join episode, proxy, and GPU traces for a sweep.

    python analysis/breakdown.py results/p1_fifo_sweep/<stamp> [--out figures/p1]

For every episode, wall-clock time is split into components that add up:

    startup      launch until the first tool call or model RPC (simulator spawn + first plan)
    planner      remaining time outside tools and RPCs (LLM time; for Flash, replay logic)
    vla_wait     VLA requests queued at the proxy
    vla_service  VLA inference at the backend
    sam3_wait / sam3_service   same for segmentation
    molmo_wait / molmo_service   same for Molmo grounding (Flash only; zero otherwise)

Model and simulator RPCs made outside any tool after work starts (Flash
replay calls molmo.ground and reads observations between tools) are charged
to their service, not to planner.
    sim          simulator RPCs (env.step, chunk_step, ...)
    rpc_overhead client-side RPC time not explained by wait + service
                 (serialization, HTTP, image encoding)
    tool_other   remaining time inside tools (scripted control, Python)

Outputs: episodes.csv, levels.csv, tools.csv, breakdown.png, vla_wait_cdf.png.
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
from rsched.traceio import read_jsonl  # noqa: E402

FATAL_ERRORS = ("Credit balance is too low", "billing_error", "authentication_failed", "Not logged in")

COMPONENTS = [
    "startup", "planner", "vla_wait", "vla_service", "sam3_wait", "sam3_service",
    "molmo_wait", "molmo_service", "sim", "rpc_overhead", "tool_other",
]
MODELS = ("vla", "sam3", "molmo")
COLORS = {
    "startup": "#b8b8b8", "planner": "#4c72b0", "vla_wait": "#dd8452", "vla_service": "#c44e52",
    "sam3_wait": "#ccb974", "sam3_service": "#8c8c3c", "molmo_wait": "#e6a0c4",
    "molmo_service": "#c17c9c", "sim": "#55a868", "rpc_overhead": "#8172b3", "tool_other": "#64b5cd",
}


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -np.inf
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def service_of(method: str | None) -> str:
    if not method:
        return "other"
    if method.startswith("vla."):
        return "vla"
    if method.startswith("sam3."):
        return "sam3"
    if method.startswith("molmo."):
        return "molmo"
    if method in ("healthz", "shutdown") or method.startswith("session."):
        return "control"
    return "sim"


def load_level(level_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((level_dir / "manifest.json").read_text())
    rpc = []
    for f in level_dir.glob("proxy_*.jsonl"):
        rpc += read_jsonl(f)
    rpc_df = pd.DataFrame(rpc)

    ep_records, tool_records = [], []
    for f in sorted((level_dir / "traces").glob("episode_*.jsonl")):
        recs = read_jsonl(f)
        start = next((r for r in recs if r["type"] == "episode_start"), None)
        end = next((r for r in recs if r["type"] == "episode_end"), None)
        outcome = next((r for r in recs if r["type"] == "outcome"), {})
        if start is None or end is None:
            continue
        ep = start["episode"]
        tools = [r for r in recs if r["type"] == "tool"]
        crpc = [r for r in recs if r["type"] == "client_rpc"]
        tool_records += tools

        wall = end["t"] - start["t"]
        tool_iv = [(t["t_start"], t["t_end"]) for t in tools]
        tool_union = union_length(tool_iv)
        in_tool = [r for r in crpc if r.get("call_id")]
        outside = [r for r in crpc if not r.get("call_id")
                   and service_of(r.get("method")) not in ("control", "other")]
        # Work starts at the first tool call or the first model RPC, whichever is
        # earlier (Flash grounds its anchors with Molmo before the first tool).
        firsts = [s for s, _ in tool_iv] + [
            r["t_start"] for r in outside if service_of(r.get("method")) in MODELS]
        work_start = min(firsts) if firsts else None
        startup = (work_start - start["t"]) if work_start is not None else wall
        between = [r for r in outside if work_start is not None and r["t_start"] >= work_start]
        busy = union_length(tool_iv + [(r["t_start"], r["t_end"]) for r in between])
        planner = max(wall - startup - busy, 0.0)

        comp = dict.fromkeys(COMPONENTS, 0.0)
        comp["startup"], comp["planner"] = startup, planner
        client_by_svc: dict[str, float] = {}
        for r in in_tool + between:
            svc = service_of(r.get("method"))
            client_by_svc[svc] = client_by_svc.get(svc, 0.0) + r["dur_s"]
        comp["sim"] = client_by_svc.get("sim", 0.0)
        if not rpc_df.empty:
            mine = rpc_df[(rpc_df["episode"] == ep) & rpc_df["gated"]]
            for svc in MODELS:
                part = mine[mine["service"] == svc]
                comp[f"{svc}_wait"] = float(part["wait_s"].sum())
                comp[f"{svc}_service"] = float(part["service_s"].sum())
        explained = sum(comp[f"{s}_{k}"] for s in MODELS for k in ("wait", "service"))
        model_client = sum(client_by_svc.get(s, 0.0) for s in MODELS)
        comp["rpc_overhead"] = max(model_client - explained, 0.0)
        in_tool_rpc = sum(r["dur_s"] for r in in_tool)
        comp["tool_other"] = max(tool_union - in_tool_rpc, 0.0)

        task = outcome.get("task") or {}
        stats = outcome.get("stats") or {}
        ep_records.append(
            {
                "level": manifest["level"], "repeat": manifest["repeat"], "episode": ep,
                "suite": task.get("suite"), "task": task.get("task"),
                "wall_s": wall, "success": outcome.get("environment_success"),
                "agent_error": outcome.get("agent_error"), "exit_code": end.get("exit_code"),
                "n_tools": len(tools),
                "n_vla_calls": sum(1 for r in in_tool if service_of(r.get("method")) == "vla"),
                "n_molmo_calls": sum(1 for r in in_tool + between
                                     if service_of(r.get("method")) == "molmo"),
                "input_tokens": stats.get("total_input_tokens"),
                "output_tokens": stats.get("total_output_tokens"),
                "cost_usd": stats.get("total_cost_usd"),
                "turns": stats.get("turns_used"),
                # Not a measurement: never ran a tool, never finished, or cut off by a
                # planner billing/auth failure. Non-zero exit alone is NOT a crash: rpent
                # exits 1 when the planner hits --max-turns, a valid (unsolved) episode.
                "crashed": (not tools or not outcome
                            or any(p in str(outcome.get("agent_error") or "") for p in FATAL_ERRORS)),
                **comp,
            }
        )
    tools_df = pd.DataFrame(tool_records)
    if not tools_df.empty:
        tools_df["level"] = manifest["level"]
    return pd.DataFrame(ep_records), rpc_df, tools_df, manifest


def summarize_level(eps: pd.DataFrame, rpc: pd.DataFrame, manifest: dict) -> dict:
    makespan = manifest["t_end"] - manifest["t_start"]
    n_ok = int((eps["success"] == True).sum()) if not eps.empty else 0  # noqa: E712
    n_crashed = int(eps["crashed"].sum()) if not eps.empty else 0
    all_eps, eps = eps, eps[~eps["crashed"]] if not eps.empty else eps  # time stats: valid runs only
    row = {
        "level": manifest["level"], "repeat": manifest["repeat"], "episodes": len(all_eps),
        "crashed": n_crashed, "successes": n_ok,
        "success_rate": n_ok / len(all_eps) if len(all_eps) else np.nan,
        "makespan_s": makespan, "success_per_hour": n_ok / makespan * 3600 if makespan else np.nan,
        "episodes_per_hour": len(all_eps) / makespan * 3600 if makespan else np.nan,
        "cost_usd": all_eps["cost_usd"].sum() if "cost_usd" in all_eps else np.nan,
        "cost_per_success_usd": (all_eps["cost_usd"].sum() / n_ok
                                 if n_ok and "cost_usd" in all_eps else np.nan),
        "wall_mean_s": eps["wall_s"].mean(), "wall_p50_s": eps["wall_s"].median(),
        "wall_p95_s": eps["wall_s"].quantile(0.95),
    }
    for c in COMPONENTS:
        row[f"{c}_mean_s"] = eps[c].mean()
        row[f"{c}_frac"] = (eps[c].sum() / eps["wall_s"].sum()) if eps["wall_s"].sum() else np.nan
    if not rpc.empty:
        vla = rpc[(rpc["service"] == "vla") & rpc["gated"]]
        if not vla.empty:
            row.update(
                vla_calls=len(vla), vla_service_mean_ms=vla["service_s"].mean() * 1e3,
                vla_wait_mean_ms=vla["wait_s"].mean() * 1e3,
                vla_wait_p95_ms=vla["wait_s"].quantile(0.95) * 1e3,
                vla_busy_frac=vla["service_s"].sum() / makespan if makespan else np.nan,
            )
    gpu_file = Path(manifest.get("_dir", "")) / "gpu.jsonl"
    if gpu_file.exists():
        g = pd.DataFrame([r for r in read_jsonl(gpu_file) if r["type"] == "gpu"])
        if not g.empty:
            row.update(gpu_mem_peak_mb=g["used_mb"].max(), gpu_util_mean=g["util_gpu"].mean())
    return row


def tool_table(tools: pd.DataFrame) -> pd.DataFrame:
    if tools.empty:
        return tools
    res = pd.json_normalize(tools["result"]).add_prefix("res_") if "result" in tools else None
    t = pd.concat([tools.reset_index(drop=True), res], axis=1) if res is not None else tools
    agg = {"dur_s": ["count", "mean", "std", "median", "max"]}
    if "res_chunks_used" in t:
        agg["res_chunks_used"] = ["mean", "std", "max"]
    out = t.groupby(["level", "tool"]).agg(agg)
    out.columns = ["_".join(c) for c in out.columns]
    out["dur_cv"] = out["dur_s_std"] / out["dur_s_mean"]
    return out.reset_index()


def plot_breakdown(levels: pd.DataFrame, path: Path) -> None:
    lv = levels.groupby("level").mean(numeric_only=True).sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bottom = np.zeros(len(lv))
    x = np.arange(len(lv))
    for c in COMPONENTS:
        vals = lv[f"{c}_mean_s"].fillna(0).to_numpy()
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=c, color=COLORS[c], width=0.65)
        bottom += vals
    ax.set_xticks(x, [str(i) for i in lv.index])
    ax.set_xlabel("concurrent episodes (N)")
    ax.set_ylabel("mean seconds per episode")
    ax.set_title("Episode time breakdown vs. concurrency")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_wait_cdf(rpc_all: pd.DataFrame, path: Path) -> None:
    if rpc_all.empty:
        return
    vla = rpc_all[(rpc_all["service"] == "vla") & rpc_all["gated"]]
    if vla.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for level, part in sorted(vla.groupby("level")):
        w = np.sort(part["wait_s"].to_numpy() * 1e3)
        ax.plot(w, np.arange(1, len(w) + 1) / len(w), label=f"N={level}")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("VLA queueing delay per request (ms)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=None, help="default: <sweep_dir>/analysis")
    args = ap.parse_args(argv)

    root = Path(args.sweep_dir)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    all_eps, all_rpc, all_tools, level_rows = [], [], [], []
    for level_dir in sorted(p for p in root.iterdir() if (p / "manifest.json").exists()):
        eps, rpc, tools, manifest = load_level(level_dir)
        manifest["_dir"] = str(level_dir)
        if not rpc.empty:
            rpc["level"] = manifest["level"]
        all_eps.append(eps)
        all_rpc.append(rpc)
        all_tools.append(tools)
        level_rows.append(summarize_level(eps, rpc, manifest))

    if not level_rows:
        sys.exit(f"no level directories with manifest.json under {root}")
    eps = pd.concat(all_eps, ignore_index=True)
    rpc = pd.concat(all_rpc, ignore_index=True)
    tools = pd.concat(all_tools, ignore_index=True)
    levels = pd.DataFrame(level_rows).sort_values(["level", "repeat"])

    eps.to_csv(out / "episodes.csv", index=False)
    levels.to_csv(out / "levels.csv", index=False)
    tool_table(tools).to_csv(out / "tools.csv", index=False)
    plot_breakdown(levels, out / "breakdown.png")
    plot_wait_cdf(rpc, out / "vla_wait_cdf.png")

    if eps["crashed"].any():
        bad = eps[eps["crashed"]]
        print(f"\n!!! {len(bad)}/{len(eps)} episodes are NOT measurements (no tool calls, no outcome, or "
              "planner billing/auth failure); "
              "they are excluded from time stats and are not measurements.")
        print("!!! inspect: <level_dir>/logs/episode_<id>.log  e.g. " + ", ".join(bad["episode"].head(3)))
        if len(bad) == len(eps):
            print("!!! no valid episodes: fix the failure before using these numbers.\n")

    if "cost_usd" in eps and eps["cost_usd"].notna().any():
        print(f"planner cost: ${eps['cost_usd'].sum():.2f} total, "
              f"${eps['cost_usd'].mean():.2f}/episode (max ${eps['cost_usd'].max():.2f})")

    cols = ["level", "episodes", "crashed", "cost_usd", "success_rate", "success_per_hour", "wall_mean_s", "wall_p95_s",
            "planner_frac", "vla_wait_frac", "vla_service_frac", "sim_frac"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(levels[[c for c in cols if c in levels]].to_string(index=False))
    print(f"\nwrote {out}/{{episodes,levels,tools}}.csv, breakdown.png, vla_wait_cdf.png")


if __name__ == "__main__":
    main()
