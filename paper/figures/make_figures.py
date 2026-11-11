"""Paper figures from the result CSVs.

    python paper/figures/make_figures.py --results results --out paper/figures

Inputs (under --results): paper/breakdown_*.csv (analysis/episode_breakdown.py),
b_feasibility/*/inflation.csv, b2a/*/interference_mps/inflation.csv,
b_batch/*/batch.csv, b3/*/phases/summary.csv, g6/*/analysis/summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette (light), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COL_W = 3.33  # one ACM column, inches


def one(results: Path, pattern: str) -> Path:
    hits = sorted(results.glob(pattern))
    if not hits:
        raise SystemExit(f"missing {pattern} under {results}")
    return hits[-1]


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def fig_breakdown(results: Path, out: Path) -> None:
    sets = [("G5, N=8\n(time-slicing)", "paper/breakdown_g5_N8.csv"),
            ("G6, N=8\n(MPS)", "paper/breakdown_g6_N8.csv"),
            ("G6, N=16\n(MPS)", "paper/breakdown_g6_N16.csv")]
    parts = [  # label, columns, color, hatch
        ("Simulator: reset", ["env_reset"], C[0], None),
        ("Simulator: step, render", ["sim"], C[0], "////"),
        ("Client-side Python", ["local"], C[1], None),
        ("Process start/exit", ["startup", "teardown"], GRAY, None),
        ("Molmo", ["molmo_svc"], C[2], None),
        ("Molmo queueing", ["molmo_wait"], C[2], "////"),
        ("Pi0.5 (VLA)", ["vla_svc"], C[6], None),
        ("Pi0.5 queueing", ["vla_wait"], C[6], "////"),
        ("SAM3", ["sam3_svc", "sam3_wait"], C[3], None),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    for y, (name, f) in enumerate(sets):
        df = pd.read_csv(results / f)
        total = df["wall_s"].sum()
        left = 0.0
        for label, cols, color, hatch in parts:
            w = sum(df[c].sum() for c in cols if c in df) / total * 100
            ax.barh(y, w, left=left, color=color if hatch is None else "white", edgecolor=color if hatch
                    else "white", hatch=hatch, linewidth=0.8 if hatch else 0.6, height=0.62,
                    label=label if y == 0 else None)
            if w >= 7:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if hatch is None and color != GRAY else INK)
            left += w
        ax.text(101, y, f"{df['wall_s'].mean():.0f} s", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(sets)), [s for s, _ in sets])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of episode wall time (%)   [right: mean episode time]")
    style(ax)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.42, -0.42), frameon=False,
              handlelength=1.4, columnspacing=0.8)
    fig.savefig(out / "breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_colocation(results: Path, out: Path) -> None:
    ts = pd.read_csv(one(results, "b_feasibility/*/inflation.csv"))
    mps = pd.read_csv(one(results, "b2a/*/interference_mps/inflation.csv"))

    def get(df, kind, workers, with_, col):
        r = df[(df["kind"] == kind) & (df["workers"] == workers) & (df["with"] == with_)]
        return float(r[col].iloc[0])

    rows = [("Pi0.5\n+planner", "vla", 1, "llm=4", "lat_p50_x"),
            ("Pi0.5\n+both", "vla", 1, "molmo=1,llm=4", "lat_p50_x"),
            ("Molmo\n+planner", "molmo", 1, "llm=4", "lat_p50_x"),
            ("Planner\n+Pi0.5", "llm", 4, "vla=1", "tpot_p50_x"),
            ("Planner\n+Molmo", "llm", 4, "molmo=1", "tpot_p50_x"),
            ("Planner\n+both", "llm", 4, "vla=1,molmo=1", "tpot_p50_x")]
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    w = 0.38
    for i, (lab, k, n, wth, col) in enumerate(rows):
        a, b = get(ts, k, n, wth, col), get(mps, k, n, wth, col)
        ax.bar(i - w / 2 - 0.01, a, w, color=C[0], label="time-slicing (default)" if i == 0 else None)
        ax.bar(i + w / 2 + 0.01, b, w, color=C[1], label="MPS" if i == 0 else None)
        ax.text(i - w / 2 - 0.01, a + 0.05, f"{a:.1f}", ha="center", va="bottom", fontsize=6.2, color=INK)
        ax.text(i + w / 2 + 0.01, b + 0.05, f"{b:.1f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    ax.axhline(1, color=INK2, linewidth=0.6, linestyle=":")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=6.6)
    ax.set_ylabel("slowdown vs alone (×)")
    ax.set_ylim(0, 3.7)
    style(ax, "y")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(out / "colocation.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_batching(results: Path, out: Path) -> None:
    bb = pd.read_csv(one(results, "b_batch/*/batch.csv"))
    bb = bb[(bb["setup"] == "mps") & (bb["llm_streams"] == 4)].sort_values("batch")
    b3 = pd.read_csv(one(results, "b3/*/phases/summary.csv"))
    g6 = pd.read_csv(one(results, "g6/*/analysis/summary.csv"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"width_ratios": [1, 1.2]})
    a1.plot(bb["batch"], bb["lat_x_vs_b1"], color=C[0], marker="o", markersize=3.2, linewidth=1.4,
            label="latency")
    a1.plot(bb["batch"], bb["throughput_x_vs_b1"], color=C[1], marker="s", markersize=3.2, linewidth=1.4,
            label="throughput")
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    a1.set_xlabel("batch size")
    a1.set_ylabel("× of batch 1")
    a1.set_title("(a) Pi0.5 batch cost", loc="left")
    style(a1, "y")
    a1.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    a1.set_ylim(0.5, 6.2)

    bars = []
    for lv in (8, 16):
        r = b3[(b3["label"] == "none@greedy") & (b3["level"] == lv)].iloc[0]
        bars.append((f"synth.\nN={lv}", r["vs_none_pct"], r["vs_none_ci"], C[2]))
    for lv in (8, 16):
        r = g6[(g6["rule"] == "greedy") & (g6["level"] == lv)].iloc[0]
        bars.append((f"Flash\nN={lv}", r["makespan_pct_mean"],
                     (r["ci95_high"] - r["ci95_low"]) / 2, C[6]))
    for i, (lab, m, ci, col) in enumerate(bars):
        a2.bar(i, m, 0.62, color=col, yerr=ci, error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": INK2})
        a2.text(i, (m - ci - 3) if m < 0 else (m + ci + 2), f"{m:+.0f}%", ha="center",
                va="top" if m < 0 else "bottom", fontsize=6.3, color=INK)
    a2.axhline(0, color=INK2, linewidth=0.6)
    a2.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=6.4)
    a2.set_ylabel("makespan change (%)")
    a2.set_ylim(-74, 25)
    a2.set_title("(b) batching vs none", loc="left")
    style(a2, "y")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "batching.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args(argv)
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_breakdown(res, out)
    fig_colocation(res, out)
    fig_batching(res, out)
    print(f"wrote {out}/breakdown.pdf, colocation.pdf, batching.pdf")


if __name__ == "__main__":
    main()
