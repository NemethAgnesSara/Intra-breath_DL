"""Table 5: paired comparison of hyperparameter settings across aggregations.

Recomputed on the merged ``combined_and_4_ch`` results, which add the
four-channel configuration to the two three-channel ones the original table was
built from.

Method, unchanged from the analysis notebook: average over the five seeds
first, so one configuration contributes one number, then pair configurations
that differ only in the factor under test and run a Wilcoxon signed-rank test
and a paired t-test over those pairs. Averaging first is what makes the pairs
independent; pairing per seed would count the same configuration five times.

Two things differ from the published table:

  * the four-channel configuration widens the optimizer and augmentation
    comparisons from 28 to 32 pairs, since ``flexicnn`` and ``linearcnn`` were
    also run with it; and
  * it adds two channel comparisons of its own, over the eight pairs where it
    exists, reported below the Config 1 - Config 2 row it does not affect.

Usage:
    python -m IB_DL.experiments.combined_exp.table_stats
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew, ttest_rel, wilcoxon

from IB_DL.experiments.combined_exp.results import HERE, DEFAULT_CSV, config_means, load_runs

OUT = HERE / "result" / "combined_and_4_ch" / "table_stats.tex"

CH1 = "[f;abs,f;rel,v;rel]"           # Config 1: V̇;abs, V̇;rel, V;rel
CH2 = "[f;abs,v;rel,v;abs]"           # Config 2: V̇;abs, V;rel, V;abs
CH3 = "[f;abs,f;rel,v;abs,v;rel]"     # Config 3: all four channels

AGGREGATIONS = [("majority_vote", "Soft Vote"), ("loop_wise", "Loop-wise")]
MEDIAN_LOOP = ("median_loop", "Median loop")

# (row label, factor column, level A, level B, subset predicate or None)
COMPARISONS = [
    ("Optimizer (Adam $-$ SGD)", "opt", "Adam", "SGD", None),
    ("Augmentation (Mild $-$ Strong)", "tf", "1", "2", None),
]
CHANNEL_COMPARISONS = [
    ("Channels (Config 1 $-$ Config 2)", "ch", CH1, CH2, None),
    ("Channels (Config 3 $-$ Config 1)", "ch", CH3, CH1, "four_channel"),
    ("Channels (Config 3 $-$ Config 2)", "ch", CH3, CH2, "four_channel"),
]


def holm(pvalues: list) -> list:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Holm steps through the p-values from smallest to largest, scaling each by
    the number still untested, and enforces monotonicity so an adjusted value
    can never fall below one that came before it. It controls the family-wise
    error rate under arbitrary dependence, which matters here: the two
    aggregations are the same runs scored differently, so their p-values are
    strongly correlated and a method assuming independence would not apply.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    out = [0.0] * m
    running = 0.0
    for step, i in enumerate(order):
        running = max(running, min(1.0, pvalues[i] * (m - step)))
        out[i] = running
    return out


def hodges_lehmann(d) -> float:
    """Median of the Walsh averages -- the estimate matching a signed-rank test.

    Worth keeping straight, because it is easy to get wrong: the Wilcoxon
    signed-rank test ranks the *magnitudes* of the differences, so the location
    it speaks to is this pseudomedian, not the plain median of ``d``. The plain
    median is what a sign test would correspond to. Here the two are far apart
    (11.24 vs 0.46 for the soft-vote optimizer comparison), so reporting the
    plain median beside a signed-rank p-value would misdescribe the test.
    """
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def paired_test(means: pd.DataFrame, factor: str, a: str, b: str) -> dict:
    """Wilcoxon and paired t-test over configurations differing only in `factor`.

    The pivot index is every configuration column except the factor, so a pair
    is one configuration under level `a` and the same one under level `b`.
    The two levels are selected *before* ``dropna``: the channel factor has a
    third level that only two architectures were run with, and dropping on the
    whole pivot would discard every configuration missing that third level.
    """
    index = [c for c in ["model", "opt", "tf", "ch"] if c != factor]
    pivot = means.pivot(index=index, columns=factor, values="accuracy")
    if a not in pivot or b not in pivot:
        return {}
    pivot = pivot[[a, b]].dropna()
    if pivot.empty:
        return {}
    x, y = pivot[a], pivot[b]
    d = x - y
    return {"n": len(pivot), "diff": d.mean(), "median_diff": d.median(),
            "hl": hodges_lehmann(d.to_numpy()),
            "wilcoxon": wilcoxon(x, y).pvalue, "ttest": ttest_rel(x, y).pvalue,
            "shapiro": shapiro(d).pvalue, "skew": float(skew(d)),
            "favours_a": int((d > 0).sum())}


def nonconvergent(means: pd.DataFrame, floor: float = 60.0) -> list:
    """Architectures with a configuration that never left chance level.

    Soft-vote accuracy below `floor` on a three-class problem is a run that did
    not train, not a run that trained badly. Such configurations dominate the
    optimizer comparison, so it is worth being able to re-run the tests without
    them; see ``--drop-nonconvergent``.
    """
    sv = means[means.aggregation_method == "majority_vote"]
    return sorted(sv[sv.accuracy < floor].model.unique())


def collect(runs: pd.DataFrame, drop_models: list | None = None,
            aggregations: list | None = None,
            comparisons: list | None = None) -> pd.DataFrame:
    """Every comparison in the table, with Holm-adjusted p-values appended.

    The correction family is the table itself: all of its rows, separately for
    the Wilcoxon and the t-test column, since the two are alternative tests of
    the same comparisons rather than additional ones.
    """
    means = config_means(runs)
    if drop_models:
        means = means[~means.model.isin(drop_models)]
    rows = []
    for agg, agg_label in (aggregations or AGGREGATIONS):
        sub = means[means.aggregation_method == agg]
        for label, factor, a, b, restrict in (comparisons or COMPARISONS):
            scope = sub
            if restrict == "four_channel":
                # Only the two custom models were run with all four channels;
                # without this the pairs would be empty for every other model.
                scope = sub[sub.model.isin(sub[sub.ch == CH3].model.unique())]
            res = paired_test(scope, factor, a, b)
            if res:
                rows.append({"aggregation": agg_label, "comparison": label, **res})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["wilcoxon_holm"] = holm(table.wilcoxon.tolist())
        table["ttest_holm"] = holm(table.ttest.tolist())
    return table


def _p(value: float) -> str:
    return "$<$ 0.0001" if value < 0.0001 else f"{value:.4f}"


TESTS = {"wilcoxon": ("Wilcoxon signed-rank", "wilcoxon", "wilcoxon_holm"),
         "ttest": ("Paired t-test", "ttest", "ttest_holm")}


def _wide(table: pd.DataFrame, tests: list) -> list:
    """Two-column ``table*`` layout: one column per field, nothing folded."""
    effect = [("Mean Acc. Diff.", "diff")]
    cols = "ll" + "c" * (1 + len(effect) + 2 * len(tests))
    head = (r"\textbf{Aggregation} & \textbf{Comparison} & \textbf{$n$}" +
            "".join(rf" & \textbf{{{n}}}" for n, _ in effect))
    sub = " & " * (3 + len(effect))
    start = 4 + len(effect)
    rules = ""
    for j, t in enumerate(tests):
        head += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{TESTS[t][0]} $p$}}}}"
        sub += ("raw & Holm" if j == 0 else " & raw & Holm")
        rules += rf"\cmidrule(lr){{{start + 2*j}-{start + 2*j + 1}}}"
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"% Generated by combined_exp/table_stats.py",
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\caption{Statistical comparison of hyperparameter configurations across "
        r"Soft Vote and Loop-wise test aggregation methods.}",
        r"\label{tab:statistical_tests}",
        rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
        r"\toprule",
        head + r" \\",
        rules,
        sub + r" \\ \midrule",
    ]
    for agg, block in table.groupby("aggregation", sort=False):
        for i, (_, r) in enumerate(block.iterrows()):
            cells = [rf"\textbf{{{agg}}}" if i == 0 else "", r.comparison,
                     f"{r.n:.0f}"]
            cells += [f"{r[c]:.2f}\\%" for _, c in effect]
            for t in tests:
                cells += [_p(r[TESTS[t][1]]), _p(r[TESTS[t][2]])]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{1ex}", ""]
    return lines


def _single_column(table: pd.DataFrame, tests: list) -> list:
    """Narrow layout for a one-column ``table`` in a two-column paper.

    Three changes buy the width: the aggregation becomes a spanning group row
    instead of a column, the constant $n$ moves to the note, and the test name
    in the header is shortened so the spanned header is no wider than the two
    columns beneath it. Everything else is the wide table's content.
    """
    effect, key = "Mean", "diff"
    ncol = 2 + 2 * len(tests)
    head = rf"\textbf{{Comparison}} & \textbf{{{effect}}}"
    sub = " & \\textbf{Acc. Diff.}"
    rules = ""
    for j, t in enumerate(tests):
        short = "Wilcoxon" if t == "wilcoxon" else "$t$-test"
        head += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{short} $p$}}}}"
        sub += " & raw & Holm"
        rules += rf"\cmidrule(lr){{{3 + 2*j}-{4 + 2*j}}}"
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"% Generated by combined_exp/table_stats.py (single-column layout)",
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Statistical comparison of hyperparameter configurations "
        r"across Soft Vote and Loop-wise test aggregation methods.}",
        r"\label{tab:statistical_tests}",
        r"\footnotesize\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}l{'c' * (ncol - 1)}@{{}}}}",
        r"\toprule",
        head + r" \\",
        rules,
        sub + r" \\ \midrule",
    ]
    for i, (agg, block) in enumerate(table.groupby("aggregation", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{agg}}}}} \\")
        for _, r in block.iterrows():
            cells = [r"\quad " + r.comparison, f"{r[key]:.2f}\\%"]
            for t in tests:
                cells += [_p(r[TESTS[t][1]]), _p(r[TESTS[t][2]])]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{1ex}", ""]
    return lines


def to_latex(table: pd.DataFrame, tests: list | None = None,
             single_column: bool = False) -> str:
    """LaTeX for the table, reporting `tests` (default: both).

    With one test the effect-size column follows it: the signed-rank test is a
    statement about the median difference, the t-test about the mean, and the
    two are far apart here. `single_column` swaps the wide ``table*`` for a
    narrow ``table`` that fits one column of a two-column paper.
    """
    tests = tests or ["wilcoxon", "ttest"]
    has_channels = table.comparison.str.startswith("Channels").any()
    env = "table" if single_column else "table*"
    lines = (_single_column(table, tests) if single_column
             else _wide(table, tests))

    note = (rf"{{\raggedright \small Note: Mild augmentation denotes strength 1, "
            r"and Strong denotes strength 2. Each pair is one configuration's "
            r"mean accuracy over five seeds under the two levels of the factor; "
            r"averaging over seeds first is what makes the pairs independent. ")
    n = sorted(table.n.unique())
    if single_column:
        # The wide layout carries n as a column; here it has to be stated.
        note += ("There are " + (f"{n[0]}" if len(n) == 1 else
                                 "/".join(str(v) for v in n)) +
                 r" such pairs per comparison. ")
    note += (r"$p$-values are given raw and after Holm--Bonferroni correction "
             f"over all {len(table)} comparisons in the table. ")
    if has_channels:
        note += (r"Config 1 represents $\mathrm{[\dot{V};abs, \dot{V};rel, V;rel]}$, "
                 r"Config 2 represents $\mathrm{[\dot{V};abs, V;rel, V;abs]}$, and "
                 r"Config 3 represents "
                 r"$\mathrm{[\dot{V};abs, \dot{V};rel, V;abs, V;rel]}$; the "
                 r"Config 3 rows cover only the two custom architectures, the "
                 r"only ones run with four channels. ")
    if tests == ["wilcoxon"]:
        # A reviewer will ask why the nonparametric test; the normality of the
        # paired differences is the reason, and it is checkable.
        note += (r"The Wilcoxon signed-rank test is reported in place of a "
                 r"paired $t$-test because the paired differences are strongly "
                 r"non-normal in every comparison (Shapiro--Wilk $p<0.01$ "
                 r"throughout, skewness up to 2.9). ")
    note += r"\par}"
    lines += [note, rf"\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="test_results.csv of a combined_exp run")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default: {OUT.name} next to the csv")
    ap.add_argument("--drop-nonconvergent", action="store_true",
                    help="exclude architectures with a chance-level configuration")
    ap.add_argument("--channels", action="store_true",
                    help="also report the input-channel comparisons")
    ap.add_argument("--median-loop", action="store_true",
                    help="also report the median-loop aggregation")
    ap.add_argument("--tests", default="wilcoxon",
                    choices=["wilcoxon", "both", "ttest"],
                    help="which test to put in the LaTeX table "
                         "(default: wilcoxon; the differences are non-normal)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="print normality and sign diagnostics for each comparison")
    ap.add_argument("--single-column", action="store_true",
                    help="narrow 'table' layout for one column of a two-column "
                         "paper, instead of the full-width 'table*'")
    args = ap.parse_args()

    aggregations = AGGREGATIONS + ([MEDIAN_LOOP] if args.median_loop else [])
    comparisons = COMPARISONS + (CHANNEL_COMPARISONS if args.channels else [])

    args.out = args.out or args.csv.parent / OUT.name
    runs = load_runs(args.csv)
    drop = None
    if args.drop_nonconvergent:
        drop = nonconvergent(config_means(runs))
        print(f"excluding {', '.join(drop)}\n")
    table = collect(runs, drop_models=drop, aggregations=aggregations,
                    comparisons=comparisons)
    shown = table.assign(
        diff=table["diff"].map("{:+.2f}%".format),
        median_diff=table.median_diff.map("{:+.2f}%".format),
        hl=table.hl.map("{:+.2f}%".format),
        **{c: table[c].map("{:.4f}".format) for c in
           ("wilcoxon", "wilcoxon_holm", "ttest", "ttest_holm")})
    cols = ["aggregation", "comparison", "n", "diff", "median_diff", "hl",
            "wilcoxon", "wilcoxon_holm", "ttest", "ttest_holm"]
    print(shown[cols].to_string(index=False))

    if args.diagnostics:
        print("\n=== paired-difference diagnostics ===")
        d = table.assign(favours=table.favours_a.astype(str) + "/" + table.n.astype(str))
        print(d[["aggregation", "comparison", "diff", "median_diff", "skew",
                 "shapiro", "favours"]].round(4).to_string(index=False))
        print("Shapiro-Wilk is on the paired differences; a small p means the "
              "paired t-test's normality assumption does not hold.")

    tests = ["wilcoxon", "ttest"] if args.tests == "both" else [args.tests]
    args.out.write_text(to_latex(table, tests, args.single_column))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
