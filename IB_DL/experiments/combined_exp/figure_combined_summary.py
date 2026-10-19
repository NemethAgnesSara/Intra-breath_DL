"""Manuscript figure: the combined-experiment summary (four panels + key).

Rebuilds the published figure from the latest results
(``result/combined_and_4_ch``), which add the four-channel runs of Models A
and B. Layout and encodings follow the published version:

    a)  every configuration, accuracy vs macro F1, coloured by aggregation
    b)  the ten best configurations by soft-vote accuracy
    c)  soft vote, coloured by augmentation strength
    d)  loop-wise, coloured by augmentation strength

Every point is a configuration's mean over its five seeds. The PDF is written
at a fixed size with no tight bounding box, so at full text width the type is
the size set here.

Usage:
    python -m IB_DL.experiments.combined_exp.figure_combined_summary
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from IB_DL.experiments.combined_exp.figure_top10_ranking import (
    CH_FILLED, CH_FOUR, CH_HOLLOW, MODEL_COLOR, MODEL_LABEL, lighten)
from IB_DL.experiments.combined_exp.results import (
    DEFAULT_CSV, config_means, load_runs, ranked_soft_vote)

IMAGES = Path(__file__).resolve().parents[2] / "images"

# Colours of the published version.
AGG_COLOR = {"loop_wise": "#f77189", "majority_vote": "#50b131",
             "median_loop": "#3ba3ec"}
AGG_LABEL = {"loop_wise": "Loop-wise", "majority_vote": "Soft vote",
             "median_loop": "Median loop"}
AUG_COLOR = {"1": "#66c2a5", "2": "#fc8d62"}
AUG_LABEL = {"1": "Mild", "2": "Strong"}
CH_LEGEND = {
    CH_FILLED: r"Filled : $\dot{V}$; abs, $\dot{V}$; rel, $V$; rel",
    CH_HOLLOW: r"Hollow : $\dot{V}$; abs, $V$; rel, $V$; abs",
    CH_FOUR: r"Light : $\dot{V}$; abs, $\dot{V}$; rel, $V$; abs, $V$; rel",
}


def style(fs: float):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": fs,
        "axes.labelsize": fs, "axes.labelweight": "bold",
        "axes.titlesize": fs + 1, "axes.titleweight": "bold",
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# View limits of the published figure, widened where the new best configuration
# (83.1 %) would otherwise sit on the edge. Configurations that failed to train
# (SGD runs of some architectures, ~30-57 %) fall outside these views - as
# they did in the published version; main() reports how many.
VIEW = {"a)": (63, 85), "c)": (75, 85), "d)": (63, 74)}


def scatter_panel(ax, d, key, colors, title):
    for k, c in colors.items():
        s = d[d[key] == k]
        ax.scatter(s.macro_f1, s.accuracy, s=26, color=c, alpha=0.8,
                   edgecolor="white", linewidth=0.5, zorder=3)
    lo, hi = VIEW[title]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, lw=1, zorder=0)
    ax.set_title(title, pad=6)
    out = ~(d.macro_f1.between(lo, hi) & d.accuracy.between(lo, hi))
    print(f"  panel {title} {int(out.sum())} of {len(d)} configurations outside "
          f"the {lo}-{hi} % view")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=float, default=7.16,
                   help="inches (default: 7.16, IEEE two-column width)")
    p.add_argument("--height", type=float, default=4.75)
    p.add_argument("--fontsize", type=float, default=9.0)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="test_results.csv of a combined_exp run")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", type=Path, default=IMAGES / "combined_summary.pdf")
    args = p.parse_args()

    style(args.fontsize)
    runs = load_runs(args.csv)
    means = config_means(runs)
    top = ranked_soft_vote(runs).head(args.top)
    print(f"{means.groupby(['model','opt','tf','ch']).ngroups} configurations, "
          f"top soft vote {top.accuracy.iloc[0]:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(args.width, args.height))
    fig.subplots_adjust(left=0.07, right=0.66, bottom=0.085, top=0.955,
                        wspace=0.38, hspace=0.30)
    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.set_xlabel("Macro F1 Score (%)")
        ax.set_ylabel("Accuracy (%)")

    # a) every configuration, by aggregation
    scatter_panel(axes[0, 0], means, "aggregation_method", AGG_COLOR, "a)")

    # b) top ten by soft vote
    ax = axes[0, 1]
    for i, r in top.iterrows():
        c = MODEL_COLOR.get(r.model, "#7f7f7f")
        face = (lighten(c) if r.ch == CH_FOUR
                else "none" if r.ch == CH_HOLLOW else c)
        ax.scatter(i + 1, r.accuracy, s=120 if r.tf == "2" else 45,
                   marker="o" if r.opt == "Adam" else "s",
                   facecolors=face, edgecolors=c, linewidths=1.5, zorder=3)
        # A label is wider than one rank step, so horizontal labels collide
        # whenever neighbours are within ~0.1 %. Vertical labels take one text
        # height of width instead. They sit above the point; only an exact tie
        # with the previous rank goes below (ranks 8 and 9 are both 81.1 %).
        tie = i > 0 and abs(r.accuracy - top.accuracy.iloc[i - 1]) < 0.05
        gap = 8 if r.tf == "2" else 5
        ax.annotate(f"{r.accuracy:.1f}%", (i + 1, r.accuracy),
                    textcoords="offset points", rotation=90,
                    xytext=(0, -gap) if tie else (0, gap), ha="center",
                    va="top" if tie else "bottom",
                    fontsize=args.fontsize - 2.5, fontweight="bold")
    ax.set_xticks(range(1, len(top) + 1))
    ax.set_xlim(0.3, len(top) + 0.7)
    lo, hi = top.accuracy.min(), top.accuracy.max()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.4 * (hi - lo))
    ax.set_xlabel("Rank")
    ax.set_title("b)", pad=6)

    # c) and d) by augmentation strength
    sv = means[means.aggregation_method == "majority_vote"]
    lw = means[means.aggregation_method == "loop_wise"]
    scatter_panel(axes[1, 0], sv, "tf", AUG_COLOR, "c)")
    scatter_panel(axes[1, 1], lw, "tf", AUG_COLOR, "d)")

    # --- one key for all panels, lengthwise, matplotlib's default frame -----
    blank = Line2D([], [], linestyle="none", marker="none")
    def dot(**kw):
        kw.setdefault("marker", "o"); kw.setdefault("markersize", 6)
        kw.setdefault("linestyle", "none")
        return Line2D([], [], **kw)
    grey = "#555555"
    H, L, B = [], [], []
    def section(title, items):
        H.append(blank); L.append(title); B.append(True)
        for h, l in items:
            H.append(h); L.append(l); B.append(False)
    def gap():        # empty row between the a), b) and c/d) groups
        H.append(blank); L.append(" "); B.append(False)
    section("a) AGGREGATION", [(dot(color=c), AGG_LABEL[k]) for k, c in AGG_COLOR.items()])
    gap()
    section("b) MODELS", [(dot(color=MODEL_COLOR[m]), MODEL_LABEL[m]) for m in MODEL_COLOR
                          if m in set(top.model)])
    section("OPTIMIZER", [(dot(color=grey), "Adam"), (dot(color=grey, marker="s"), "SGD")])
    section("CHANNELS", [(dot(markerfacecolor={CH_FILLED: grey, CH_HOLLOW: "none",
                                               CH_FOUR: lighten(grey)}[ch],
                              markeredgecolor=grey, markeredgewidth=1.3), CH_LEGEND[ch])
                         for ch in (CH_FILLED, CH_HOLLOW, CH_FOUR) if ch in set(top.ch)])
    section("AUGMENTATION", [(dot(color=grey, markersize=4), "Mild"),
                             (dot(color=grey, markersize=9), "Strong")])
    gap()
    section("c and d) AUGMENTATION", [(dot(color=c), AUG_LABEL[k]) for k, c in AUG_COLOR.items()])
    leg = fig.legend(H, L, loc="center left", bbox_to_anchor=(0.675, 0.52),
                     ncol=1, frameon=True, fontsize=args.fontsize - 1,
                     handletextpad=0.4, labelspacing=0.28)
    for t, b in zip(leg.get_texts(), B):
        if b:
            t.set_fontweight("bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
