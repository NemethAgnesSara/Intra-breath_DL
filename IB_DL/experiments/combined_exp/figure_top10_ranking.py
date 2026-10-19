"""Manuscript figure: the ten best combined_exp configurations by soft-vote accuracy.

Recreates panel (b) of the combined-experiment figure from the latest results,
``result/combined_and_4_ch/test_results.csv``, which adds the four-channel
flexicnn and linearcnn runs to the original grid. Each point is one
configuration, ranked by its soft-vote accuracy averaged over five seeds.

Encodings follow the original panel:
    colour     architecture
    shape      optimizer (circle Adam, square SGD)
    fill       channels (filled, hollow, or light for the four-channel set)
    size       augmentation (small mild, large strong)

Usage:
    python -m IB_DL.experiments.combined_exp.figure_top10_ranking
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "result" / "combined_and_4_ch" / "test_results.csv"
IMAGES = HERE.parents[1] / "images"

# The original panel used tab10 per architecture; kept so this panel still
# matches the other panels of the same figure.
MODEL_COLOR = {
    "flexicnn": "#1f77b4", "linearcnn": "#ff7f0e",
    "resnet50_pretrained": "#2ca02c", "shufflenet-pretrained": "#d62728",
    "vgg13-pretrained": "#9467bd", "efficientnet_b1": "#8c564b",
    "vit-b-16-pretrained": "#e377c2",
}
MODEL_LABEL = {
    "flexicnn": "Model A", "linearcnn": "Model B",
    "resnet50_pretrained": "ResNet50 (pretrained)",
    "shufflenet-pretrained": "ShuffleNet (pretrained)",
    "vgg13-pretrained": "VGG13 (pretrained)", "efficientnet_b1": "EfficientNet-B1",
    "vit-b-16-pretrained": "ViT-B/16 (pretrained)",
}
CH_FILLED = "[f;abs,f;rel,v;rel]"
CH_HOLLOW = "[f;abs,v;rel,v;abs]"
CH_FOUR = "[f;abs,f;rel,v;abs,v;rel]"
CH_LABEL = {
    CH_FILLED: r"$\dot{V}$; abs, $\dot{V}$; rel, $V$; rel",
    CH_HOLLOW: r"$\dot{V}$; abs, $V$; rel, $V$; abs",
    CH_FOUR: r"$\dot{V}$; abs, $\dot{V}$; rel, $V$; abs, $V$; rel",
}


def load(csv: Path) -> pd.DataFrame:
    """Mean soft-vote accuracy per configuration, one row per configuration.

    Channel lists are normalised before grouping: the merged CSV stores some
    configurations under two spellings (with and without spaces), which would
    otherwise split one configuration in two.
    """
    df = pd.read_csv(csv)
    lab = df.model_label
    df = df.assign(
        model=lab.str.extract(r"model_(.*?)_optimizer")[0],
        opt=lab.str.extract(r"optimizer_(.*?)_tf")[0],
        tf=lab.str.extract(r"tf-trength_(\d)")[0],
        ch=lab.str.extract(r"img-channels_(\[.*?\])_seed")[0]
              .str.replace(r"[\s']", "", regex=True),
        seed=lab.str.extract(r"_seed_(\d+)_")[0].astype(int))
    df = df.drop_duplicates(["model", "opt", "tf", "ch", "seed",
                             "aggregation_method"])
    sv = df[df.aggregation_method == "majority_vote"]
    return (sv.groupby(["model", "opt", "tf", "ch"]).accuracy
              .agg(["mean", "std", "count"]).reset_index()
              .sort_values("mean", ascending=False).reset_index(drop=True))


def lighten(color: str, factor: float = 1.6):
    h, s, v = rgb_to_hsv(to_rgb(color))
    return hsv_to_rgb((h, s / factor, min(1.0, v * factor)))


def style(fs: float):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": fs,
        "axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--width", type=float, default=7.16,
                   help="figure width in inches (default: 7.16, two columns)")
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--fontsize", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=IMAGES / "top10_ranking.pdf")
    args = p.parse_args()

    style(args.fontsize)
    ranked = load(args.csv)
    top = ranked.head(args.top)
    print(top.assign(rank=np.arange(1, len(top) + 1))
             [["rank", "model", "opt", "tf", "ch", "mean", "std", "count"]]
             .round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.subplots_adjust(left=0.085, right=0.60, bottom=0.17, top=0.95)

    for i, r in top.iterrows():
        c = MODEL_COLOR.get(r["model"], "#7f7f7f")
        face = (lighten(c) if r["ch"] == CH_FOUR
                else "none" if r["ch"] == CH_HOLLOW else c)
        ax.scatter(i + 1, r["mean"], s=170 if r["tf"] == "2" else 60,
                   marker="o" if r["opt"] == "Adam" else "s",
                   facecolors=face, edgecolors=c, linewidths=1.6, zorder=3)
        ax.annotate(f"{r['mean']:.1f}%", (i + 1, r["mean"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=args.fontsize - 2)

    ax.set_xticks(range(1, len(top) + 1))
    ax.set_xlim(0.4, len(top) + 0.6)
    lo, hi = top["mean"].min(), top["mean"].max()
    ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.35 * (hi - lo))
    ax.set_xlabel("Rank")
    ax.set_ylabel("Soft vote accuracy (%)")
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)

    # Lengthwise key in matplotlib's default frame; headings are rows whose
    # handle is invisible.
    blank = Line2D([], [], linestyle="none", marker="none")
    def dot(**kw):
        kw.setdefault("marker", "o"); kw.setdefault("markersize", 7)
        kw.setdefault("linestyle", "none")
        return Line2D([], [], **kw)
    grey = "#555555"
    models = [m for m in MODEL_COLOR if m in set(top["model"])]
    handles, labels, head = [], [], []
    def section(title, items):
        handles.append(blank); labels.append(title); head.append(True)
        for h, l in items:
            handles.append(h); labels.append(l); head.append(False)
    section("Architecture", [(dot(color=MODEL_COLOR[m]), MODEL_LABEL.get(m, m))
                             for m in models])
    section("Optimizer", [(dot(color=grey), "Adam"),
                          (dot(color=grey, marker="s"), "SGD")])
    channels = [ch for ch in (CH_FILLED, CH_HOLLOW, CH_FOUR) if ch in set(top["ch"])]
    section("Channels", [(dot(markerfacecolor={CH_FILLED: grey, CH_HOLLOW: "none",
                                               CH_FOUR: lighten(grey)}[ch],
                              markeredgecolor=grey, markeredgewidth=1.4),
                          CH_LABEL[ch]) for ch in channels])
    section("Augmentation", [(dot(color=grey, markersize=5), "Mild"),
                             (dot(color=grey, markersize=10), "Strong")])
    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    ncol=1, frameon=True, fontsize=args.fontsize - 1.5,
                    handletextpad=0.4, labelspacing=0.3)
    for t, h in zip(leg.get_texts(), head):
        if h:
            t.set_fontweight("bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
