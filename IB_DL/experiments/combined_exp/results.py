"""Shared loading of the combined experiment's per-seed results.

The merged ``combined_and_4_ch/test_results.csv`` holds one row per run and
aggregation method. This turns it into tidy per-seed rows with the
configuration split into columns and the confusion-matrix metrics computed,
and averages them per configuration the same way the analysis notebook does.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "result" / "combined_and_4_ch" / "test_results.csv"
CONFIG = ["model", "opt", "tf", "ch"]
CLASSES = ["healthy", "copd", "ild"]        # confusion-matrix row order


def _metrics(cm_str: str) -> pd.Series:
    cm = np.array(ast.literal_eval(f"[{cm_str}]"), dtype=float)
    if cm.ndim == 3:
        cm = cm[0]
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) != 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) != 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp),
                   where=(prec + rec) != 0)
    out = {"macro_f1": f1.mean() * 100}
    out.update({f"recall_{c}": r * 100 for c, r in zip(CLASSES, rec)})
    return pd.Series(out)


def load_runs(csv: Path = DEFAULT_CSV) -> pd.DataFrame:
    """One row per (configuration, seed, aggregation method).

    Channel lists are normalised because the merged CSV stores some
    configurations under two spellings; exact duplicate runs are dropped.
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
    df = df.drop_duplicates(CONFIG + ["seed", "aggregation_method"])
    return pd.concat([df.reset_index(drop=True),
                      df.confusion_matrix_str.apply(_metrics).reset_index(drop=True)],
                     axis=1)


def config_means(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per configuration and aggregation method."""
    cols = ["accuracy", "macro_f1", "epoch"] + [f"recall_{c}" for c in CLASSES]
    return runs.groupby(CONFIG + ["aggregation_method"], as_index=False)[cols].mean()


def ranked_soft_vote(runs: pd.DataFrame) -> pd.DataFrame:
    """Configurations ordered by mean soft-vote accuracy, best first."""
    sv = config_means(runs)
    sv = sv[sv.aggregation_method == "majority_vote"]
    return sv.sort_values("accuracy", ascending=False).reset_index(drop=True)
