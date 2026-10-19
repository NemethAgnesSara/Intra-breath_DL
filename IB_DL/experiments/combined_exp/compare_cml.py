"""Best deep-learning configuration against the classical MLP baseline.

The classical results come from a scikit-learn ``MLPClassifier`` with a grid
search per seed, run outside this repository; its per-seed confusion matrices
are transcribed below so the comparison is reproducible here. They are on the
same task: every matrix has row sums 61/41/72 and every accuracy is an integer
count out of 174, matching this project's test split exactly.

What the tests can and cannot say, with five seeds per side:

  * A Wilcoxon signed-rank test on five pairs has a smallest attainable
    two-sided p of 0.0625. It can never reject at 0.05, whatever the data, so
    a non-significant result from it means nothing on its own.
  * Mann-Whitney U on 5 vs 5 bottoms out at 0.0079, so it can reject.
  * The paired tests assume seed N drew the same patient split on both sides.
    That is how this project's splits work and the class counts agree, but it
    was not verified patient by patient, so the unpaired tests are reported
    alongside as the assumption-free version.

The deep side is the top-ranked configuration of the combined experiment,
which was selected as the best of 64 on this same test set. The classical side
is one pipeline. That asymmetry flatters the deep side and no test here
corrects for it; ``--reference`` re-runs the comparison against the median
configuration instead, as a selection-free sanity check.

Usage:
    python -m IB_DL.experiments.combined_exp.compare_cml
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, ttest_rel, wilcoxon

from IB_DL.experiments.combined_exp.results import (
    DEFAULT_CSV, config_means, load_runs, ranked_soft_vote)

CLASSES = ["Healthy", "COPD", "ILD"]

# scikit-learn MLPClassifier, one grid search per seed. Rows are the true
# class, columns the predicted one, in CLASSES order.
CML_CONFUSION = {
    1: [[42, 4, 15], [2, 30, 9], [10, 3, 59]],
    2: [[46, 1, 14], [0, 34, 7], [8, 2, 62]],
    3: [[32, 1, 28], [0, 37, 4], [11, 3, 58]],
    4: [[43, 2, 16], [6, 30, 5], [8, 2, 62]],
    5: [[39, 3, 19], [2, 34, 5], [10, 3, 59]],
}


def cml_table() -> pd.DataFrame:
    """Accuracy and class-wise recall/precision per seed, from the matrices."""
    rows = []
    for seed, cm in sorted(CML_CONFUSION.items()):
        cm = np.array(cm, dtype=float)
        tp = np.diag(cm)
        rec, prec = tp / cm.sum(1) * 100, tp / cm.sum(0) * 100
        r = {"seed": seed, "accuracy": tp.sum() / cm.sum() * 100,
             "n_test": int(cm.sum())}
        r.update({f"recall_{c}": v for c, v in zip(CLASSES, rec)})
        r.update({f"precision_{c}": v for c, v in zip(CLASSES, prec)})
        rows.append(r)
    return pd.DataFrame(rows)


def dl_table(which: str = "best", csv: Path = DEFAULT_CSV) -> tuple[pd.DataFrame, str]:
    """Per-seed soft-vote results of one combined-experiment configuration."""
    runs = load_runs(csv)
    ranked = ranked_soft_vote(runs)
    cfg = ranked.iloc[0] if which == "best" else ranked.iloc[len(ranked) // 2]
    sv = runs[runs.aggregation_method == "majority_vote"]
    sel = sv[(sv.model == cfg.model) & (sv.opt == cfg.opt) &
             (sv.tf == cfg.tf) & (sv.ch == cfg.ch)].sort_values("seed")
    name = f"{cfg.model} / {cfg.opt} / tf{cfg.tf} / {cfg.ch}"
    cols = {"recall_healthy": "recall_Healthy", "recall_copd": "recall_COPD",
            "recall_ild": "recall_ILD"}
    return sel.rename(columns=cols)[["seed", "accuracy"] +
                                    list(cols.values())], name


def compare(dl: np.ndarray, cml: np.ndarray) -> pd.DataFrame:
    """Paired and unpaired tests of the same accuracy difference."""
    d = dl - cml
    rows = [
        {"test": "Paired t-test", "pairing": "paired",
         "statistic": ttest_rel(dl, cml).statistic,
         "p": ttest_rel(dl, cml).pvalue},
        {"test": "Wilcoxon signed-rank", "pairing": "paired",
         "statistic": wilcoxon(dl, cml).statistic,
         "p": wilcoxon(dl, cml).pvalue},
        {"test": "Welch t-test", "pairing": "unpaired",
         "statistic": ttest_ind(dl, cml, equal_var=False).statistic,
         "p": ttest_ind(dl, cml, equal_var=False).pvalue},
        {"test": "Mann-Whitney U", "pairing": "unpaired",
         "statistic": mannwhitneyu(dl, cml, alternative="two-sided").statistic,
         "p": mannwhitneyu(dl, cml, alternative="two-sided").pvalue},
    ]
    t = pd.DataFrame(rows)
    # Floor of what each test could report here, so a large p can be read as
    # "no evidence" rather than "evidence of no difference".
    t["min_possible_p"] = [np.nan, 2 / 2 ** len(d), np.nan,
                           2 / math.comb(2 * len(d), len(d))]
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", choices=["best", "median"], default="best",
                    help="which combined-experiment configuration to compare")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="test_results.csv of a combined_exp run")
    args = ap.parse_args()

    cml = cml_table()
    dl, name = dl_table(args.reference, args.csv)
    print(f"deep configuration ({args.reference}): {name}\n")

    side = pd.DataFrame({
        "seed": cml.seed,
        "CML": cml.accuracy.to_numpy(),
        "DL": dl.accuracy.to_numpy(),
    })
    side["diff"] = side.DL - side.CML
    print("=== soft-vote accuracy per seed (%) ===")
    print(side.round(2).to_string(index=False))
    print(f"\nCML {cml.accuracy.mean():.2f} +- {cml.accuracy.std(ddof=1):.2f}"
          f"   DL {dl.accuracy.mean():.2f} +- {dl.accuracy.std(ddof=1):.2f}"
          f"   difference {side['diff'].mean():+.2f} pp")

    d = side["diff"].to_numpy()
    pooled = np.sqrt((cml.accuracy.var(ddof=1) + dl.accuracy.var(ddof=1)) / 2)
    print(f"Cohen's d (unpaired) {(dl.accuracy.mean() - cml.accuracy.mean()) / pooled:.2f}"
          f"   |   d_z (paired) {d.mean() / d.std(ddof=1):.2f}"
          f"   |  Shapiro-Wilk on the differences p={shapiro(d).pvalue:.3f}")

    print("\n=== tests ===")
    print(compare(dl.accuracy.to_numpy(), cml.accuracy.to_numpy())
          .round(4).to_string(index=False))

    print("\n=== class-wise recall, mean over five seeds (%) ===")
    cls = pd.DataFrame({
        "class": CLASSES,
        "CML": [cml[f"recall_{c}"].mean() for c in CLASSES],
        "DL": [dl[f"recall_{c}"].mean() for c in CLASSES],
    })
    cls["diff"] = cls.DL - cls.CML
    print(cls.round(2).to_string(index=False))
    for lab, col in [("CML", "CML"), ("DL", "DL")]:
        v = cls[col]
        print(f"  {lab}: spread {v.max() - v.min():.2f} pp, SD {v.std(ddof=1):.2f}, "
              f"weakest class {cls['class'][v.idxmin()]}")


if __name__ == "__main__":
    main()
