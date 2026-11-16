"""Paper figures, from the CSVs the analysis scripts write under results/.

    python paper/figures/make_figures.py [--results results] [--out paper/figures]

    anatomy.pdf      episode wall time by component     results/paper/anatomy/shares.csv  (analysis/anatomy.py)
    bounds.pdf       first-order bound vs. measured     results/paper/bounds.csv          (analysis/bounds.py)
    colocation.pdf   slowdown of co-located models      results/b_feasibility/*/inflation.csv,
                                                         results/b2a/*/interference_mps/inflation.csv

Colors: one fixed categorical order; queueing is the hatched version of its
model's color, so the figures also read in grayscale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.6,
})
COL_W = 3.45  # one IEEE column, inches


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


def fig_anatomy(results: Path, out: Path) -> None:
    sh = pd.read_csv(results / "paper" / "anatomy" / "shares.csv")
    sh = sh[sh["outcome"] == "all"].set_index("set")
    rows = [("replay_N1", "N=1"), ("replay_N8_ts", "N=8\ntime-sliced"), ("replay_N8_mps", "N=8\nMPS"),
            ("replay_N16_mps", "N=16\nMPS")]
    rows = [(k, lab) for k, lab in rows if k in sh.index]
    parts = [  # label, columns, color, hatch
        ("Sim: scene reset", ["sim_reset"], C[0], None),
        ("Sim: step, render", ["sim_step", "sim_chunk_step", "sim_render", "sim_other"], C[0], "////"),
        ("Client: inside tools", ["local_tools"], C[1], None),
        ("Client: between tools", ["local_between"], C[1], "////"),
        ("Process start/exit", ["startup", "teardown"], GRAY, None),
        ("Grounding VLM", ["molmo_svc"], C[2], None),
        ("Grounding VLM queue", ["molmo_wait"], C[2], "////"),
        ("VLA", ["vla_svc"], C[6], None),
        ("VLA queue", ["vla_wait"], C[6], "////"),
        ("Segmentation", ["sam3_svc", "sam3_wait"], C[3], None),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    for y, (key, _) in enumerate(rows):
        left = 0.0
        for label, cols, color, hatch in parts:
            w = float(sh.loc[key, cols].sum()) * 100
            ax.barh(y, w, left=left, height=0.62, color=color if hatch is None else "white",
                    edgecolor="white" if hatch is None else color, hatch=hatch,
                    linewidth=0.8, label=label if y == 0 else None)
            if w >= 7:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=6.3,
                        color="white" if hatch is None and color != GRAY else INK,
                        bbox=None if hatch is None else dict(facecolor="white", edgecolor="none", pad=0.4))
            left += w
        ax.text(101, y, f"{sh.loc[key, 'wall_mean_s']:.0f} s", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of episode wall time (%); right: mean episode")
    style(ax)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.46, 1.0), frameon=False, columnspacing=0.9,
              handlelength=1.1, handletextpad=0.4)
    fig.savefig(out / "anatomy.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_bounds(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "paper" / "bounds.csv")
    df = df[df["measured_pct"].notna()].reset_index(drop=True)

    def name(r):
        if "synthetic" in r["workload"]:
            return f"Batch VLA, synthetic, N={r['N']}"
        if r["lever"].startswith("batch"):
            return f"Batch VLA, real, N={r['N']}"
        est = r["lever"][r["lever"].index("(") + 1:-1]
        return f"Launch, {'tuning' if 'G4' in r['workload'] else 'held-out'}: {est}"

    df["name"] = [name(r) for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(COL_W, 0.27 * len(df) + 0.55))
    y = range(len(df))
    ax.barh(y, -df["bound_pct"], height=0.6, color="#cfcdc6", edgecolor="none", label="first-order bound")
    ax.errorbar(-df["measured_pct"], y, xerr=[(df["ci_high"] - df["measured_pct"]).clip(lower=0),
                                              (df["measured_pct"] - df["ci_low"]).clip(lower=0)],
                fmt="o", color=C[0], markersize=3.2, elinewidth=0.9, capsize=1.8, label="measured (95% CI)")
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_yticks(list(y), df["name"])
    ax.invert_yaxis()
    ax.set_xlabel("makespan reduction (%)")
    style(ax)
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.savefig(out / "bounds.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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
    fig.savefig(out / "colocation.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    args = ap.parse_args(argv)
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.png:
        plt.rcParams["savefig.dpi"] = 220
    for f in (fig_anatomy, fig_bounds, fig_colocation):
        f(results, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
