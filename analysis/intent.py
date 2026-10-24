"""Does planner intent predict VLA demand better than history?

    python analysis/intent.py results/p1_laptop/<stamp> [--out dir]

One row per primitive (one tool call). Two labels, both known only after the
call returns:

    n_vla   client RPCs whose method starts with ``vla.`` for that call
    dur_s   the primitive's wall time (includes proxy queueing when N > 1)

Four predictors. Intent and the fixed ratio are fit on the other episodes at
the same concurrency level only (``--pool-levels`` fits across levels), so
the scored episode's labels are not in the fit and queueing at another level
does not leak into the estimate.

    intent       mean label of the same (tool, max_chunks) in the other
                 episodes. Missing bucket falls back to the same tool, then
                 to the global mean. ``max_chunks`` is the planner's argument;
                 tools that omit it share one bucket per tool.
    last_round   the previous primitive in this episode (Kairos-style: recent
                 demand, not the primitive that is about to run). The first
                 primitive has no previous call and uses the global mean.
    fixed_ratio  one mean for every primitive (ROSA-style constant
                 planner:VLA ratio).
    intent_rate  dur_s only: intent's n_vla times this episode's latest
                 observed seconds per VLA call (from its last primitive that
                 called the VLA). Intent says how much work; the runtime says
                 how fast it currently goes. Falls back to intent.

A boundary is the first primitive of an episode, or a primitive whose tool
differs from the previous one. The ``vla`` slice is every call of a tool that
issues at least one ``vla.`` RPC in this sweep; it is a reporting cut, not a
feature. Outcome records are not required, so traces from before the K1 fix
still count. Episodes with no tool call are skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsched.traceio import read_jsonl  # noqa: E402

PREDICTORS = ("intent", "last_round", "fixed_ratio", "intent_rate")
TARGETS = ("n_vla", "dur_s")


def level_dirs(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def _max_chunks(args: object) -> float:
    if not isinstance(args, dict) or args.get("max_chunks") is None:
        return math.nan
    try:
        return float(args["max_chunks"])
    except (TypeError, ValueError):
        return math.nan


def load_primitives(root: Path) -> tuple[pd.DataFrame, int, int]:
    """Return primitives, the number of toolless episodes, and unattributed VLA RPCs."""
    rows = []
    skipped = 0
    unattributed = 0
    for level_dir in level_dirs(root):
        manifest = json.loads((level_dir / "manifest.json").read_text())
        trace_dir = level_dir / "traces"
        files = sorted(trace_dir.glob("episode_*.jsonl")) if trace_dir.is_dir() else []
        for path in files:
            recs = read_jsonl(path)
            tools = [r for r in recs if r.get("type") == "tool"]
            if not tools:
                skipped += 1
                continue
            counts: dict[str, int] = {}
            for r in recs:
                if r.get("type") != "client_rpc" or not str(r.get("method") or "").startswith("vla."):
                    continue
                call_id = r.get("call_id")
                if not call_id:
                    unattributed += 1
                    continue
                counts[call_id] = counts.get(call_id, 0) + 1
            for tool in tools:
                rows.append(
                    {
                        "level": manifest.get("level"),
                        "repeat": manifest.get("repeat"),
                        "episode": tool.get("episode"),
                        "call_id": tool.get("call_id"),
                        "t_start": tool.get("t_start"),
                        "tool": tool.get("tool"),
                        "max_chunks": _max_chunks(tool.get("args")),
                        "n_vla": counts.get(tool.get("call_id"), 0),
                        "dur_s": float(tool.get("dur_s") or 0.0),
                    }
                )
    return pd.DataFrame(rows), skipped, unattributed


def _from_train(train: pd.DataFrame, tool: str, max_chunks: float, col: str) -> float:
    if max_chunks == max_chunks:
        bucket = train[(train["tool"] == tool) & (train["max_chunks"] == max_chunks)]
        if len(bucket):
            return float(bucket[col].mean())
    same = train[train["tool"] == tool]
    if len(same):
        return float(same[col].mean())
    return float(train[col].mean())


def add_predictions(df: pd.DataFrame, pool_levels: bool = False) -> pd.DataFrame:
    df = df.sort_values(["episode", "t_start"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("episode", sort=False)
    df["prev_tool"] = grouped["tool"].shift(1)
    df["prev_n_vla"] = grouped["n_vla"].shift(1)
    df["prev_dur_s"] = grouped["dur_s"].shift(1)
    df["boundary"] = df["prev_tool"].isna() | (df["prev_tool"] != df["tool"])

    columns = {name: [] for name in (
        "pred_intent_n_vla", "pred_intent_dur_s",
        "pred_last_n_vla", "pred_last_dur_s",
        "pred_fixed_n_vla", "pred_fixed_dur_s",
        "pred_intent_rate_n_vla", "pred_intent_rate_dur_s",
    )}
    for episode, part in df.groupby("episode", sort=False):
        train = df[df["episode"] != episode]
        if not pool_levels:
            same = train[train["level"] == part["level"].iloc[0]]
            train = same if len(same) else train
        sec_per_call = math.nan
        fixed_n = float(train["n_vla"].mean())
        fixed_d = float(train["dur_s"].mean())
        for row in part.itertuples(index=False):
            columns["pred_intent_n_vla"].append(_from_train(train, row.tool, row.max_chunks, "n_vla"))
            columns["pred_intent_dur_s"].append(_from_train(train, row.tool, row.max_chunks, "dur_s"))
            columns["pred_fixed_n_vla"].append(fixed_n)
            columns["pred_fixed_dur_s"].append(fixed_d)
            columns["pred_last_n_vla"].append(fixed_n if pd.isna(row.prev_n_vla) else float(row.prev_n_vla))
            columns["pred_last_dur_s"].append(fixed_d if pd.isna(row.prev_dur_s) else float(row.prev_dur_s))
            intent_n = columns["pred_intent_n_vla"][-1]
            columns["pred_intent_rate_n_vla"].append(intent_n)
            rate_ok = intent_n > 0 and sec_per_call == sec_per_call
            columns["pred_intent_rate_dur_s"].append(
                intent_n * sec_per_call if rate_ok else columns["pred_intent_dur_s"][-1])
            if row.n_vla > 0:  # observed after the call returns; used by later calls only
                sec_per_call = row.dur_s / row.n_vla
    for name, values in columns.items():
        df[name] = values
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """MAE and bias per target, predictor, and slice; per level and pooled."""
    vla_tools = set(df.loc[df["n_vla"] > 0, "tool"])
    groups = [("all", df)]
    if df["level"].nunique() > 1:
        groups += [(str(lv), part) for lv, part in df.groupby("level", sort=True)]
    out = []
    for level, part in groups:
        rows = _summarize_one(part.reset_index(drop=True), vla_tools)
        rows.insert(0, "level", level)
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def _summarize_one(df: pd.DataFrame, vla_tools: set) -> pd.DataFrame:
    slices = {
        "all": np.ones(len(df), dtype=bool),
        "boundary": df["boundary"].to_numpy(dtype=bool),
        "interior": ~df["boundary"].to_numpy(dtype=bool),
        "vla": df["tool"].isin(vla_tools).to_numpy(dtype=bool),
    }
    pred = {
        "n_vla": ("pred_intent_n_vla", "pred_last_n_vla", "pred_fixed_n_vla", "pred_intent_rate_n_vla"),
        "dur_s": ("pred_intent_dur_s", "pred_last_dur_s", "pred_fixed_dur_s", "pred_intent_rate_dur_s"),
    }
    rows = []
    for target in TARGETS:
        actual = df[target].to_numpy(dtype=float)
        for name, column in zip(PREDICTORS, pred[target]):
            if target == "n_vla" and name == "intent_rate":
                continue  # identical to intent for counts
            error = df[column].to_numpy(dtype=float) - actual
            for slice_name, mask in slices.items():
                if not mask.any():
                    continue
                part = error[mask]
                rows.append(
                    {
                        "target": target,
                        "slice": slice_name,
                        "predictor": name,
                        "n": int(mask.sum()),
                        "mae": float(np.abs(part).mean()),
                        "bias": float(part.mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_mae(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    x = np.arange(len(PREDICTORS))
    width = 0.36
    for ax, target, ylabel in zip(axes, TARGETS, ("MAE (VLA calls)", "MAE (seconds)")):
        part = summary[(summary["target"] == target) & (summary["level"] == "all")]
        for offset, slice_name, color in ((-width / 2, "all", "#4c72b0"), (width / 2, "boundary", "#c44e52")):
            rows = part[part["slice"] == slice_name].set_index("predictor").reindex(list(PREDICTORS))
            ax.bar(x + offset, rows["mae"].to_numpy(), width=width, label=slice_name, color=color)
        ax.set_xticks(x, list(PREDICTORS))
        ax.set_ylabel(ylabel)
        ax.set_title(target)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare intent, last-round, and fixed-ratio VLA-demand predictors.")
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=None, help="default: <sweep_dir>/analysis")
    ap.add_argument("--pool-levels", action="store_true",
                    help="fit intent and fixed_ratio across concurrency levels (old behaviour)")
    args = ap.parse_args(argv)

    root = Path(args.sweep_dir)
    out = Path(args.out) if args.out else root / "analysis"
    primitives, skipped, unattributed = load_primitives(root)
    if primitives.empty or primitives["episode"].nunique() < 2:
        n = 0 if primitives.empty else int(primitives["episode"].nunique())
        sys.exit(f"need tool calls from at least 2 episodes under {root} (found {n})")

    predicted = add_predictions(primitives, pool_levels=args.pool_levels)
    summary = summarize(predicted)
    out.mkdir(parents=True, exist_ok=True)
    keep = [c for c in predicted.columns if not c.startswith("prev_")]
    predicted[keep].to_csv(out / "intent_primitives.csv", index=False)
    summary.to_csv(out / "intent_summary.csv", index=False)
    plot_mae(summary, out / "intent_mae.png")

    print(
        f"episodes: {predicted['episode'].nunique()}  primitives: {len(predicted)}  "
        f"skipped_no_tools: {skipped}  unattributed_vla_rpcs: {unattributed}"
    )
    print("intent = mean of the same (tool, max_chunks) on other episodes")
    print("last_round = previous primitive in this episode; fixed_ratio = one mean for every primitive")
    print("intent_rate = intent n_vla x this episode's latest seconds per VLA call (dur_s only)")
    print("fit: " + ("pooled across levels" if args.pool_levels else "same concurrency level only"))
    print("boundary = first primitive, or a different tool from the previous one")
    print("vla = calls of a tool that makes at least one vla.* RPC in this sweep")
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\nwrote {out}/{{intent_primitives,intent_summary}}.csv, intent_mae.png")


if __name__ == "__main__":
    main()
