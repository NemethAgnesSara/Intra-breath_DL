"""Paired analysis of the temporal-encoding A/B.

Reads ``result/<date>/test_results.csv``, pairs the gradient-on and gradient-off
runs on (model, seed, aggregation_method), and reports the accuracy delta with
the same statistics the manuscript uses elsewhere (paired t-test + Wilcoxon),
plus the class-wise breakdown that R4 asks about.

Usage:
    python -m IB_DL.experiments.gradient.analysis <date>
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from IB_DL.utils import cprint

CLASS_LABELS = ["healthy", "COPD", "ILD"]
AGGREGATIONS = ["loop_wise", "majority_vote", "median_loop"]


def load(date: str) -> pd.DataFrame:
    path = Path(os.path.dirname(os.path.realpath(__file__))
                ) / "result" / date / "test_results.csv"
    if not path.exists():
        raise SystemExit(f"No test_results.csv at {path}")
    df = pd.read_csv(path)
    lab = df["model_label"]
    df["model"] = lab.str.extract(r"model_(.*?)_optimizer")
    df["gradient"] = lab.str.extract(r"_gradient_(True|False)_") == "True"
    df["seed"] = lab.str.extract(r"_seed_(\d+)_").astype(int)
    if df["gradient"].isna().any():
        raise SystemExit(
            "Some runs have no _gradient_ token in the prefix - they predate "
            "the A/B config and cannot be paired.")
    return df


def pair(df: pd.DataFrame) -> pd.DataFrame:
    key = ["model", "seed", "aggregation_method"]
    on = df[df.gradient].set_index(key)
    off = df[~df.gradient].set_index(key)
    m = on.join(off, lsuffix="_on", rsuffix="_off", how="inner").reset_index()
    m["delta"] = m["accuracy_on"] - m["accuracy_off"]
    return m


def _tests(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Paired t-test and Wilcoxon p-values; nan when the arms are identical."""
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), float("nan")
    t = stats.ttest_rel(a, b).pvalue
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = float("nan")
    return t, w


def _table(m: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, d in m.groupby(group):
        t, w = _tests(d["accuracy_on"], d["accuracy_off"])
        rows.append(dict(zip(group, keys if isinstance(keys, tuple) else (keys,)),
                         n=len(d), off=d["accuracy_off"].mean(),
                         on=d["accuracy_on"].mean(), delta=d["delta"].mean(),
                         t_p=t, wilcoxon_p=w))
    return pd.DataFrame(rows).round(3)


def _recalls(s: str) -> np.ndarray:
    cm = np.array(ast.literal_eval("(" + s.strip() + ")"), dtype=float)
    return cm.diagonal() / cm.sum(1)


def _routing(s: str, true_cls: int, pred_cls: int) -> float:
    cm = np.array(ast.literal_eval("(" + s.strip() + ")"), dtype=float)
    return cm[true_cls, pred_cls] / cm[true_cls].sum() * 100


def main(date: str):
    m = pair(load(date))
    cprint(f"Temporal encoding A/B - result/{date}", "highlighted_cyan")
    print(f"{len(m)} paired runs "
          f"({m['model'].nunique()} architectures x {m['seed'].nunique()} seeds "
          f"x {m['aggregation_method'].nunique()} aggregations)\n")

    cprint("Overall, by aggregation method", "underline_blue")
    print(_table(m, ["aggregation_method"]).to_string(index=False), "\n")

    cprint("By architecture", "underline_blue")
    print(_table(m, ["model", "aggregation_method"]).to_string(index=False), "\n")

    for agg in AGGREGATIONS:
        sub = m[m.aggregation_method == agg]
        if sub.empty:
            continue
        cprint(f"Class-wise recall - {agg}", "underline_blue")
        rows = []
        for i, name in enumerate(CLASS_LABELS):
            on = sub["confusion_matrix_str_on"].map(lambda s: _recalls(s)[i] * 100)
            off = sub["confusion_matrix_str_off"].map(lambda s: _recalls(s)[i] * 100)
            t, w = _tests(on, off)
            rows.append({"class": name, "off": off.mean(), "on": on.mean(),
                         "delta": (on - off).mean(), "t_p": t, "wilcoxon_p": w})
        print(pd.DataFrame(rows).round(3).to_string(index=False))

        # The two confusions the Conclusion names as unresolved.
        rows = []
        for pred, name in ((0, "ILD -> healthy"), (1, "ILD -> COPD")):
            on = sub["confusion_matrix_str_on"].map(lambda s: _routing(s, 2, pred))
            off = sub["confusion_matrix_str_off"].map(lambda s: _routing(s, 2, pred))
            t, w = _tests(on, off)
            rows.append({"confusion": name, "off": off.mean(), "on": on.mean(),
                         "delta": (on - off).mean(), "t_p": t, "wilcoxon_p": w})
        print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
