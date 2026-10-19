"""Manuscript figure: sign consistency of the temporal-encoding effect per arm.

One point per (aggregation x model) cell, six per arm. The gradient helps in
every cell of the volume-only and flow-only arms, but in only two of six when
both direction cues are present - the evidence that needs no independence
assumption about the 30 paired deltas.

Each cell is identified twice over, so the six points of a row can be told
apart: the **fill** carries the test aggregation method (three levels - solid,
open, hatched) and the **marker shape** carries the architecture (two levels -
circle, square). Colour repeats the row and is therefore redundant by design,
which is what lets the figure survive greyscale printing.

The PDF is written at exactly one IEEE Access column (3.5 in), with the layout
fixed by explicit margins rather than a tight bounding box. That matters:
``bbox_inches="tight"`` grows the canvas to fit the annotations that sit outside
the axes, so ``\\includegraphics[width=\\columnwidth]`` then scales the figure
*down* and the type comes out smaller than the caption. At a true 3.5 in the
scale factor is 1.0 and the point sizes below are what the page shows.

Usage:
    python -m IB_DL.experiments.gradient_channels.figure_sign_consistency
    python -m IB_DL.experiments.gradient_channels.figure_sign_consistency --fontsize 11
"""

import IB_DL.experiments.gradient_channels.analysis as ch
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


COLUMN_WIDTH_IN = 5.5          # one IEEE Access column
DEFAULT_DATE = "2026-09-14_15-57-15"
IMAGES = Path(__file__).resolve().parents[2] / "images"

# Manuscript notation: V is volume, \dot{V} is flow.
ARM_LABEL = {
    "A": r"$V$; abs, $V$; rel",
    "B": r"$\dot{V}$; abs, $\dot{V}$; rel",
    "C": r"$\dot{V}$; abs, $V$; abs",
}
# Validated categorical slots 1-3, in fixed order.
ARM_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, MUTED, GRID = "#1a1a19", "#52514e", "#d8d7d0"

# Three fills for the three aggregation methods, in the manuscript's order.
AGG_FILL = {"loop_wise": "solid", "majority_vote": "open",
            "median_loop": "hatched"}
AGG_LABEL = {"loop_wise": "Loop-wise", "majority_vote": "Soft vote",
             "median_loop": "Median loop"}
# Two shapes for the two architectures.
MODEL_MARKER = {"flexicnn": "o", "linearcnn": "s"}
MODEL_LABEL = {"flexicnn": "Model A", "linearcnn": "Model B"}

# One sub-row per (aggregation, model) cell. Six offsets rather than three,
# because two cells can land on identical values - Arm A soft vote gives
# +0.115 for both architectures - and would otherwise be drawn one exactly on
# top of the other, leaving a row looking as though it held five points.
# The vertical axis is categorical, so these offsets carry no data.
CELL_OFFSET = {
    ("loop_wise", "flexicnn"): -0.30, ("loop_wise", "linearcnn"): -0.18,
    ("majority_vote", "flexicnn"): -0.06, ("majority_vote", "linearcnn"): 0.06,
    ("median_loop", "flexicnn"): 0.18, ("median_loop", "linearcnn"): 0.30,
}


def fill_kwargs(style: str, color: str) -> dict:
    """Face/edge/hatch for one of the three aggregation fills."""
    if style == "solid":
        return dict(facecolor=color, edgecolor=color, linewidth=0.6, hatch=None)
    if style == "open":
        return dict(facecolor="white", edgecolor=color, linewidth=0.9, hatch=None)
    return dict(facecolor="white", edgecolor=color, linewidth=0.9, hatch="/////")


def style(fs: float):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "axes.linewidth": 0.5,
        "hatch.linewidth": 0.4,    # fine stripes; the default is far too heavy here
        "pdf.fonttype": 42,        # embed TrueType, not Type 3
        "ps.fonttype": 42,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=DEFAULT_DATE)
    p.add_argument("--width", type=float, default=COLUMN_WIDTH_IN)
    p.add_argument("--height", type=float, default=2.5)
    p.add_argument("--fontsize", type=float, default=10.0,
                   help="point size of the axis text as it appears on the page "
                        "(default: 10, against an ~8 pt caption)")
    p.add_argument("--markersize", type=float, default=40.0)
    p.add_argument("--out", type=Path, default=IMAGES / "sign_consistency.pdf")
    args = p.parse_args()

    style(args.fontsize)
    m = ch.pair(ch.load(args.date))
    cells = (m.groupby(["arm", "aggregation_method", "model"])["delta"]
              .mean().reset_index())

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    # Explicit margins: the y labels, the count column and the legend all live
    # outside the axes, and a tight bbox would change the published width.
    fig.subplots_adjust(left=0.245, right=0.590, bottom=0.185, top=0.955)

    small = args.fontsize - 1.5
    for i, arm in enumerate("ABC"):
        d = cells[cells.arm == arm]
        for _, r in d.iterrows():
            ax.scatter(r["delta"],
                       i + CELL_OFFSET[(r["aggregation_method"], r["model"])],
                       s=args.markersize,
                       marker=MODEL_MARKER[r["model"]],
                       zorder=3, clip_on=False,
                       **fill_kwargs(AGG_FILL[r["aggregation_method"]],
                                     ARM_COLOR[arm]))
        n = int((d["delta"] > 0).sum())
        ax.annotate(f"{n}/{len(d)}", (1.04, i), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=small, color=MUTED,
                    annotation_clip=False)
        print(f"  {ARM_LABEL[arm]:<30} {n}/{len(d)} positive   "
              f"mean {d['delta'].mean():+.2f}")

    # Header for the count column, so "6/6" is not cryptic without the caption.
    ax.annotate("cells\n$>0$", (1.03, -0.62), xycoords=("axes fraction", "data"),
                va="center", ha="left", fontsize=small - 0.5, color=MUTED,
                annotation_clip=False, linespacing=1.1)

    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_yticks(range(3))
    ax.set_yticklabels([ARM_LABEL[a] for a in "ABC"])
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change with temporal encoding (%)", labelpad=2)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=2.5, width=0.5, pad=2)
    for lab in ax.get_yticklabels():
        lab.set_color(INK)

    # --- legend: the two encodings, explained above the plot ----------------
    # Proxies are drawn in neutral ink so they read as encoding, not as a row.
    agg_h = [ax.scatter([], [], s=args.markersize, marker="o",
                        **fill_kwargs(AGG_FILL[k], MUTED), label=AGG_LABEL[k])
             for k in ("loop_wise", "majority_vote", "median_loop")]
    mdl_h = [ax.scatter([], [], s=args.markersize, marker=MODEL_MARKER[k],
                        **fill_kwargs("solid", MUTED), label=MODEL_LABEL[k])
             for k in ("flexicnn", "linearcnn")]

    # One boxed key with both groups, matching the legend panel of Fig. 13:
    # a single lengthwise column, group headings carried by legend rows whose
    # handle is invisible. The frame is matplotlib's default.
    blank = Line2D([], [], linestyle="none", marker="none")
    handles = [blank] + agg_h + [blank] + mdl_h
    labels = (["Aggregation"] + [AGG_LABEL[k] for k in
                                 ("loop_wise", "majority_vote", "median_loop")]
              + ["Architecture"] + [MODEL_LABEL[k] for k in
                                    ("flexicnn", "linearcnn")])
    is_head = [True, False, False, False, True, False, False]

    leg = ax.legend(handles, labels, loc="center left",
                    bbox_to_anchor=(1.20, 0.5), ncol=1, frameon=True,
                    fontsize=small, handletextpad=0.3, handlelength=1.2)
    for text, head in zip(leg.get_texts(), is_head):
        text.set_color(MUTED if head else INK)
        if head:
            text.set_fontweight("bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)                      # no tight bbox: width stays 3.5 in
    fig.savefig(args.out.with_suffix(".png"), dpi=400)
    print(f"\nwrote {args.out}  ({args.width} x {args.height} in, "
          f"{args.fontsize:g} pt text, scale 1.0 at \\columnwidth)")
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
