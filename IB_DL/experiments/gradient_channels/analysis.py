"""Does temporal encoding help only where the geometry hides direction?

Pairs the gradient-on and gradient-off runs *within each arm* and reports the
accuracy delta per arm. The prediction under test:

    Arm A  ['v;abs', 'v;rel']   volume only - clock, no phase label   -> should help
    Arm B  ['f;abs', 'f;rel']   flow only   - phase label, no clock   -> should help
    Arm C  ['f;abs', 'v;abs']   one of each - direction recoverable   -> should not

The headline is not any single arm's delta but the *contrast* between them, so
the arm-vs-arm comparison is reported explicitly rather than left to the reader.

Usage:
    python -m IB_DL.experiments.gradient_channels.analysis <date>
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

ARMS = {
    "['v;abs', 'v;rel']": "A  volume only  (clock, no phase label)",
    "['f;abs', 'f;rel']": "B  flow only    (phase label, no clock)",
    "['f;abs', 'v;abs']": "C  one of each  (direction recoverable)",
}
HELP_EXPECTED = {"A": True, "B": True, "C": False}


def load(date: str) -> pd.DataFrame:
    path = Path(os.path.dirname(os.path.realpath(__file__))
                ) / "result" / date / "test_results.csv"
    if not path.exists():
        raise SystemExit(f"No test_results.csv at {path}")
    df = pd.read_csv(path)
    lab = df["model_label"]
    df["model"] = lab.str.extract(r"model_(.*?)_optimizer")
    df["channels"] = lab.str.extract(r"img-channels_(\[.*?\])_gradient")
    df["gradient"] = lab.str.extract(r"_gradient_(True|False)_") == "True"
    df["seed"] = lab.str.extract(r"_seed_(\d+)_").astype(int)
    df["arm"] = df["channels"].map(lambda c: ARMS.get(c, c)[0])
    if df["channels"].isna().any() or df["gradient"].isna().any():
        raise SystemExit("Some runs lack a channel or gradient token in the prefix.")
    return df


def pair(df: pd.DataFrame) -> pd.DataFrame:
    """Match gradient-on to gradient-off within the same arm, model and seed."""
    key = ["arm", "channels", "model", "seed", "aggregation_method"]
    on = df[df.gradient].set_index(key)
    off = df[~df.gradient].set_index(key)
    m = on.join(off, lsuffix="_on", rsuffix="_off", how="inner").reset_index()
    m["delta"] = m["accuracy_on"] - m["accuracy_off"]
    return m


def _tests(a, b) -> tuple[float, float]:
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), float("nan")
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = float("nan")
    return stats.ttest_rel(a, b).pvalue, w


def _table(m: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, d in m.groupby(group):
        t, w = _tests(d["accuracy_on"], d["accuracy_off"])
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(group, keys), n=len(d),
                         off=d["accuracy_off"].mean(), on=d["accuracy_on"].mean(),
                         delta=d["delta"].mean(), t_p=t, wilcoxon_p=w,
                         improved=f"{int((d['delta'] > 0).sum())}/{len(d)}"))
    return pd.DataFrame(rows).round(3)


def main(date: str):
    m = pair(load(date))
    cprint(f"Channel-ablation x temporal encoding - result/{date}", "highlighted_cyan")
    print(f"{len(m)} paired runs "
          f"({m['arm'].nunique()} arms x {m['model'].nunique()} models x "
          f"{m['seed'].nunique()} seeds x {m['aggregation_method'].nunique()} aggregations)\n")

    for code, name in sorted((v[0], v) for v in ARMS.values()):
        print(f"  Arm {name}")
    print()

    cprint("Effect of temporal encoding, by arm", "underline_blue")
    print(_table(m, ["arm"]).to_string(index=False), "\n")

    cprint("By arm and aggregation method", "underline_blue")
    print(_table(m, ["arm", "aggregation_method"]).to_string(index=False), "\n")

    cprint("By arm and model", "underline_blue")
    print(_table(m, ["arm", "model"]).to_string(index=False), "\n")

    # The hypothesis lives in the contrast between arms, not in any one delta.
    cprint("Hypothesis check: is the gradient worth more where a cue is missing?",
           "underline_blue")
    rows = []
    per_arm = {a: d["delta"].to_numpy() for a, d in m.groupby("arm")}
    for a in ("A", "B"):
        if a not in per_arm or "C" not in per_arm:
            continue
        u = stats.mannwhitneyu(per_arm[a], per_arm["C"], alternative="greater")
        rows.append({"comparison": f"delta({a}) > delta(C)",
                     "mean delta (arm)": per_arm[a].mean(),
                     "mean delta (C)": per_arm["C"].mean(),
                     "difference": per_arm[a].mean() - per_arm["C"].mean(),
                     "mann-whitney p": u.pvalue})
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print("\n  Prediction: A and B benefit from the gradient, C does not.")
        print("  Support requires BOTH a positive difference and a small p.\n")

    for agg in AGGREGATIONS:
        sub = m[m.aggregation_method == agg]
        if sub.empty:
            continue
        cprint(f"Class-wise recall delta - {agg}", "underline_blue")
        rows = []
        for arm, d in sub.groupby("arm"):
            r = {"arm": arm}
            for i, name in enumerate(CLASS_LABELS):
                on = d["confusion_matrix_str_on"].map(lambda s: _recall(s, i))
                off = d["confusion_matrix_str_off"].map(lambda s: _recall(s, i))
                r[name] = (on - off).mean()
            rows.append(r)
        print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")


def _recall(s: str, i: int) -> float:
    cm = np.array(ast.literal_eval("(" + s.strip() + ")"), dtype=float)
    return cm[i, i] / cm[i].sum() * 100


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
