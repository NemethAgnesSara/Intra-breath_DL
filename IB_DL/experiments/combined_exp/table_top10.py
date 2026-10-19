"""Appendix tables: complete settings and results of the ten best configurations.

Answers the request to report the final configuration explicitly rather than
only through the ranking figure. Emits two LaTeX tables (booktabs):

    settings  - every hyperparameter that differs between the ten, read from
                the training logs, with the settings they share in a note
    results   - mean ± SD over five seeds for all aggregations, macro F1 and
                class-wise recall

Settings come from the logs of the runs that produced the results: a log is
matched to a configuration by the run timestamps recorded in the CSV, so stray
test runs of the same configuration are never picked up. Where a run's log no
longer exists, the setting is taken from a logged run of the same architecture
in the same batch and marked with a dagger.

Usage:
    python -m IB_DL.experiments.combined_exp.table_top10
"""

import argparse
import ast
import glob
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from IB_DL.experiments.combined_exp.results import (
    CONFIG, DEFAULT_CSV, HERE, load_runs, ranked_soft_vote)

OUT = HERE / "result" / "combined_and_4_ch" / "top10_appendix.tex"

ARCH = {"flexicnn": "Model A", "linearcnn": "Model B",
        "resnet50_pretrained": "ResNet-50", "shufflenet-pretrained": "ShuffleNet V2",
        "vgg13-pretrained": "VGG-13", "efficientnet_b1": "EfficientNet-B1",
        "vit-b-16-pretrained": "ViT-B/16"}
CHANNELS = {"f;abs": r"$\dot{V}$;abs", "f;rel": r"$\dot{V}$;rel",
            "v;abs": r"$V$;abs", "v;rel": r"$V$;rel"}
AUG = {"1": "Mild", "2": "Strong"}
AUG_NAMES = ("AWG", "SaltAndPepperNoise", "StretchLoops", "ShiftLoops")


def _stamp(s: str) -> datetime:
    return datetime.strptime(s.strip("_")[:19], "%Y-%m-%d_%H-%M-%S")


def index_logs(pattern: str = str(HERE / "result" / "*" / "*.log")) -> pd.DataFrame:
    rows = []
    for f in glob.glob(pattern):
        p = {}
        for line in open(f, errors="ignore"):
            parts = line.rstrip("\n").split(", ", 2)
            if len(parts) == 3 and parts[0] == "param":
                p[parts[1]] = parts[2]
        if "model" not in p or "img_channels" not in p:
            continue
        m = re.search(r"__log__(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)\.log$", f)
        rows.append({"file": f, "stamp": _stamp(m.group(1)) if m else None,
                     "model": p["model"], "opt": p.get("optimizer"),
                     "tf": p.get("tf_strength"),
                     "ch": re.sub(r"[\s']", "", p["img_channels"]), **p})
    return pd.DataFrame(rows)


def settings_for(cfg: pd.Series, runs: pd.DataFrame, logs: pd.DataFrame) -> dict:
    stamps = [_stamp(d) for d in runs.date.unique()]
    same = logs[(logs.model == cfg.model) & (logs.opt == cfg.opt) &
                (logs.tf == cfg.tf) & (logs.ch == cfg.ch)]
    exact = same[same.stamp.isin(stamps)]
    inferred = exact.empty
    if inferred:
        # Nearest logged run of the same architecture: same grid batch, so
        # the same image size and linewidth.
        pool = logs[logs.model == cfg.model].copy()
        if pool.empty:
            raise SystemExit(f"No log at all for {cfg.model}")
        pool["gap"] = pool.stamp.map(lambda s: min(abs((s - t).total_seconds())
                                                   for t in stamps))
        log = pool.sort_values("gap").iloc[0]
    else:
        log = exact.iloc[0]
    tr = ast.literal_eval(log["transformations"]) if not inferred else []
    ps = sorted({t["p"] for t in tr if t.get("name") in AUG_NAMES})
    return {"img_size": log["img_size"], "lw": log["lw"],
            "fe": log.get("fe_layers"), "fc": log.get("fc_layers"),
            "base": log.get("base_channels"), "growth": log.get("growth_factor"),
            "bottleneck": log.get("bottleneck_channels"),
            "lr": log.get("learning_rate"),
            "p": ps[0] if len(ps) == 1 else {"1": 0.25, "2": 0.75}[cfg.tf],
            "inferred": inferred}


# The channel sets the top ten draw on, named as the statistics table names
# them, so the two tables can share one definition in the text.
CHANNEL_CONFIG = {
    "[f;abs,f;rel,v;rel]": "Config 1",
    "[f;abs,v;rel,v;abs]": "Config 2",
    "[f;abs,f;rel,v;abs,v;rel]": "Config 3",
}


# Short architecture names for the narrow table. "ShuffleNet V2" is the widest
# value in its column and alone sets that column's width, so dropping the
# version suffix buys more than any header change.
ARCH_SHORT = dict(ARCH, **{"shufflenet-pretrained": "ShuffleNet"})

HEADERS = [r"\#", "Architecture", "Opt.", "Aug.", "Channels", "Acc.", "F1"]


def compact_settings(top: pd.DataFrame) -> list:
    """Single-column settings table: one row per rank, one column per setting.

    The wide table lists every hyperparameter and needs the full page. Here
    only the settings that actually differ between the ten are columns, the
    channel set is named rather than spelled out (three sets occur, and the
    statistics table already defines those names), and the constants move to
    the note. Mean accuracy is kept so the table stands alone beside the
    ranking figure.

    Width comes from four decisions, in descending order of what they save:

      * No initialisation column. Among these ten it is a restatement of the
        architecture -- the two custom models start at random, every other one
        from ImageNet -- so the column carried no information of its own.
      * "ShuffleNet V2" shortened, which is what set the widest column.
      * Headers abbreviated wherever the header, not the values, set the width:
        Optimizer, Augmentation, Accuracy and Rank.
      * Rank written as #, since the caption already says the rows are ranked.
    """
    rows = []
    for rank, (_, cfg) in enumerate(top.iterrows(), start=1):
        cells = [str(rank),
                 str(ARCH_SHORT.get(cfg.model, cfg.model)),
                 str(cfg.opt),
                 AUG[str(cfg.tf)],
                 CHANNEL_CONFIG.get(str(cfg.ch), str(cfg.ch)),
                 f"{cfg.accuracy:.2f}",
                 f"{cfg.macro_f1:.2f}"]
        rows.append(" & ".join(cells) + r" \\")

    used = [c for c in CHANNEL_CONFIG if c in set(top.ch)]
    legend = "; ".join(
        f"{CHANNEL_CONFIG[c]} = "
        + "[" + ", ".join(CHANNELS[p] for p in c.strip("[]").split(",")) + "]"
        for c in used)

    return [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Settings of the ten best configurations under soft voting, "
        r"ranked by mean accuracy over five random seeds.}",
        r"\label{tab:top10_settings}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}cllllcc@{}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{h}}}" for h in HEADERS) + r" \\",
        r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}", "",
        r"{\raggedright \small Note: " + legend + r". Opt.: optimizer. Aug.: "
        r"augmentation, Mild denoting strength 1 and Strong strength 2. Acc.\ "
        r"and F1: mean soft-vote accuracy and mean macro F1 score over the "
        r"five seeds, in percent. Model A and "
        r"Model B are trained from random initialisation, the remaining "
        r"architectures from ImageNet weights. Shared by all ten: learning "
        r"rate 0.001; batch size 64; cross-entropy loss with class-balanced "
        r"batch sampling; at most 100 epochs with early stopping; patient-wise "
        r"60/20/20 split, each seed drawing its own.\par}",
        r"\end{table}"]


def pm(v: pd.Series, bold=False, digits=2) -> str:
    s = f"{v.mean():.{digits}f} $\\pm$ {v.std(ddof=1):.{digits}f}"
    return f"\\textbf{{{s}}}" if bold else s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="test_results.csv of a combined_exp run")
    ap.add_argument("--logs", default=str(HERE / "result" / "*" / "*.log"),
                    help="glob of the run logs the settings are read from")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default: {OUT.name} next to the csv")
    ap.add_argument("--compact", action="store_true",
                    help="emit only the single-column settings table, for one "
                         "column of a two-column paper")
    args = ap.parse_args()
    args.out = args.out or args.csv.parent / OUT.name

    if args.compact:
        tex = "\n".join(["% Requires \\usepackage{booktabs}",
                         "% Generated by combined_exp/table_top10.py --compact",
                         ""] + compact_settings(
                             ranked_soft_vote(load_runs(args.csv)).head(args.top))) + "\n"
        args.out.write_text(tex)
        print(tex)
        print(f"wrote {args.out}")
        return

    runs = load_runs(args.csv)
    top = ranked_soft_vote(runs).head(args.top).reset_index(drop=True)
    logs = index_logs(args.logs)
    dag = r"\textsuperscript{\dag}"

    # ---------------- settings table ---------------------------------------
    srows, any_inferred = [], False
    for i, cfg in top.iterrows():
        cr = runs[(runs[CONFIG] == cfg[CONFIG]).all(axis=1)]
        s = settings_for(cfg, cr, logs)
        any_inferred |= s["inferred"]
        mark = dag if s["inferred"] else ""
        chans = ", ".join(CHANNELS[c] for c in cfg.ch.strip("[]").split(","))
        if cfg.model == "flexicnn":
            arch_detail = f"{s['fe']} / {s['fc']}"
        elif cfg.model == "linearcnn":
            arch_detail = f"{s['fe']} / {s['fc']}"
        else:
            arch_detail = "--"
        ep = cr[cr.aggregation_method == "majority_vote"].epoch
        srows.append(" & ".join([
            str(i + 1), ARCH.get(cfg.model, cfg.model),
            "ImageNet" if "pretrained" in cfg.model else "Random",
            cfg.opt, chans, f"{AUG[cfg.tf]} ($p={s['p']:g}$)",
            f"{s['img_size']}{mark}", f"{s['lw']}{mark}", arch_detail,
            f"{ep.mean():.0f} $\\pm$ {ep.std(ddof=1):.0f}"]) + r" \\")

    settings = [
        r"\begin{table*}[!t]",
        r"\caption{Settings of the ten best configurations of the combined "
        r"experiment, ranked by mean soft-vote accuracy over five seeds.}",
        r"\label{tab:top10_settings}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rllllllrcc}",
        r"\toprule",
        r"Rank & Architecture & Init. & Optimizer & Input channels & Augmentation "
        r"& Res.\ (px) & LW (px) & FE / FC & Epochs \\",
        r"\midrule", *srows, r"\bottomrule", r"\end{tabular}",
        r"\vspace{2pt}",
        r"\parbox{\textwidth}{\footnotesize Shared by all configurations: learning "
        r"rate 0.001; batch size 64; cross-entropy loss with class-balanced batch "
        r"sampling; 3 warm-up epochs followed by reduce-on-plateau scheduling; at "
        r"most 100 epochs with early stopping on validation loss (patience 10, "
        r"minimum 15 epochs); weights taken from the epoch of highest validation "
        r"accuracy; patient-wise 60/20/20 train/validation/test split; five "
        r"random seeds, each also drawing its own split. Augmentation $p$ is the "
        r"execution probability of each pre-render transformation. FE / FC: "
        r"feature-extractor and fully connected layers of the custom models; "
        r"Epochs: epochs trained before early stopping."
        + (r" \dag~Taken from a logged run of the same architecture in the same "
           r"training batch; the log of this configuration was not retained."
           if any_inferred else "") + "}",
        r"\end{table*}"]

    # ---------------- results table ----------------------------------------
    metric_cols = [("majority_vote", "accuracy"), ("majority_vote", "macro_f1"),
                   ("median_loop", "accuracy"), ("loop_wise", "accuracy"),
                   ("majority_vote", "recall_healthy"), ("majority_vote", "recall_copd"),
                   ("majority_vote", "recall_ild")]
    vals = {}
    for i, cfg in top.iterrows():
        cr = runs[(runs[CONFIG] == cfg[CONFIG]).all(axis=1)]
        vals[i] = [cr[cr.aggregation_method == a][m] for a, m in metric_cols]
    # Bold every entry that ties the best at the printed precision.
    best = [max(round(vals[i][j].mean(), 2) for i in range(len(top)))
            for j in range(len(metric_cols))]
    rrows = [" & ".join([str(i + 1)] + [pm(vals[i][j],
                                           bold=round(vals[i][j].mean(), 2) == best[j])
                                        for j in range(len(metric_cols))]) + r" \\"
             for i in range(len(top))]
    results = [
        r"\begin{table*}[!t]",
        r"\caption{Performance of the ten best configurations (ranks as in "
        r"Table~\ref{tab:top10_settings}): mean $\pm$ standard deviation over "
        r"five seeds, in percent. Class-wise values are recalls under soft "
        r"voting. The best value in each column is in bold.}",
        r"\label{tab:top10_results}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rccccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Soft vote} & Median loop & Loop-wise "
        r"& \multicolumn{3}{c}{Soft-vote recall} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){7-9}",
        r"Rank & Accuracy & Macro F1 & Accuracy & Accuracy & Healthy & COPD & ILD \\",
        r"\midrule", *rrows, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    tex = "\n".join(["% Requires \\usepackage{booktabs}",
                     "% Generated by combined_exp/table_top10.py", ""]
                    + settings + [""] + results) + "\n"
    args.out.write_text(tex)
    print(tex)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
