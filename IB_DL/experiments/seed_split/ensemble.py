"""Combine the seed runs of a fixed-split experiment into a probability ensemble.

Every member of a ``seed_split`` run shares one patient split, so the test set
is identical across members by construction. Each test loop is rendered once
and passed through all members; the ensemble averages their class
probabilities, then applies the same three aggregations as ``eval``:

    loop-wise    argmax of the averaged per-loop probabilities
    soft vote    per patient, mean of the averaged loop probabilities
    median loop  per patient, the averaged probabilities of the median loop

Individual members are scored in the same pass, so their numbers double as a
check against ``test_results.csv`` - they must agree, or something differs
between this path and ``eval``.

Usage:
    python -m IB_DL.experiments.seed_split.ensemble <date>
"""

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from IB_DL.datasets import init_dataset
from IB_DL.eval import get_probabilities, load_model
from IB_DL.transformations import MedianLoops
from IB_DL.utils import cprint, init_logreaders, set_seed

HERE = Path(os.path.dirname(os.path.realpath(__file__)))
AGGS = ["loop_wise", "majority_vote", "median_loop"]
AGG_NAME = {"loop_wise": "Loop-wise", "majority_vote": "Soft vote",
            "median_loop": "Median loop"}


def collect(readers, chunk: int = 256):
    """Per-member probabilities for every test loop and every median loop.

    Returns (loop_probs, loop_patient, median_probs, labels) where loop_probs is
    [members, loops, classes] and median_probs is [members, patients, classes].
    """
    ref = readers[0].params
    set_seed(ref["seed"])                 # as eval does; split_seed takes over inside
    *_, ds = init_dataset(ref)
    loops, labels = ds.get_test_patient_loops()
    idx = {c: i for i, c in enumerate(ds.class_labels)}
    labels = np.array([idx[l] for l in labels])

    models = [load_model(r) for r in readers]
    device = next(models[0].parameters()).device

    def render(batch):
        return torch.stack([ds.transform(x, {"augment": False})[0] for x in batch])

    loop_probs = [[] for _ in models]
    median_probs = [[] for _ in models]
    loop_patient = []
    median = MedianLoops()
    with torch.no_grad():
        for p, pl in enumerate(loops):
            loop_patient += [p] * len(pl)
            for s in range(0, len(pl), chunk):
                x = render(pl[s:s + chunk]).to(device)
                for k, m in enumerate(models):
                    loop_probs[k].append(get_probabilities(m(x)).cpu())
            x = render([median(pl)[0]]).to(device)
            for k, m in enumerate(models):
                median_probs[k].append(get_probabilities(m(x)).cpu())
            if (p + 1) % 25 == 0 or p + 1 == len(loops):
                print(f"  {p + 1}/{len(loops)} test patients")

    return (torch.stack([torch.cat(v) for v in loop_probs]).numpy(),
            np.array(loop_patient),
            torch.stack([torch.cat(v) for v in median_probs]).numpy(),
            labels, ds.class_labels)


def score(loop_p, loop_patient, median_p, labels, n_cls):
    """Accuracy and confusion matrix for one probability set, all aggregations."""
    out = {}
    loop_true = labels[loop_patient]
    loop_pred = loop_p.argmax(1)
    out["loop_wise"] = (loop_true, loop_pred)
    soft = np.stack([loop_p[loop_patient == p].mean(0)
                     for p in range(len(labels))])
    out["majority_vote"] = (labels, soft.argmax(1))
    out["median_loop"] = (labels, median_p.argmax(1))
    return {a: ((t == y).mean() * 100,
                confusion_matrix(t, y, labels=list(range(n_cls))))
            for a, (t, y) in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="result directory: a name under seed_split/result, "
                                "or any path")
    ap.add_argument("--model", default=None,
                    help="only use members of this architecture")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="force the split for runs that predate split_seed and "
                         "are known to have been trained on one fixed split")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV path (default: <run>/ensemble_results.csv)")
    args = ap.parse_args()

    src = Path(args.run) if Path(args.run).is_dir() else HERE / "result" / args.run
    logs = sorted(src.glob("*.log"))
    if not logs:
        raise SystemExit(f"No logs in {src}")
    readers = init_logreaders(log_files=[str(f) for f in logs])
    keep = [(r, l) for r, l in zip(readers, logs)
            if args.model is None or r.params["model"] == args.model]
    if not keep:
        raise SystemExit(f"No members with model {args.model}")
    readers, logs = [r for r, _ in keep], [l for _, l in keep]
    for r, log in zip(readers, logs):
        # The checkpoint sits next to its log; the logged best_model_filename
        # does not survive LogReader's comma re-joining for every prefix.
        r.params["best_model_filename"] = str(log.with_suffix(".pt"))
        if args.split_seed is not None:
            r.params["split_seed"] = args.split_seed

    splits = {r.params.get("split_seed") for r in readers}
    if len(splits) != 1 or None in splits:
        raise SystemExit(f"Members do not share one split_seed: {splits}")
    seeds = [r.params["seed"] for r in readers]
    cprint(f"{len(readers)} members, split_seed {splits.pop()}, "
           f"training seeds {seeds}", "highlighted_cyan")

    loop_p, loop_patient, median_p, labels, classes = collect(readers)
    n_cls = len(classes)

    rows = []

    def add(label, res):
        for a in AGGS:
            acc, cm = res[a]
            rows.append({"accuracy": acc, "aggregation_method": a,
                         "model_label": label,
                         "confusion_matrix_str": str(cm.tolist())[1:-1]})

    members = {}
    for k, s in enumerate(seeds):
        members[s] = score(loop_p[k], loop_patient, median_p[k], labels, n_cls)
        add(f"member_seed_{s}", members[s])
    ens = score(loop_p.mean(0), loop_patient, median_p.mean(0), labels, n_cls)
    add(f"ensemble_{len(seeds)}", ens)

    out = args.out or src / "ensemble_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\n{'':<13}{'members (mean ± sd)':>22}{'best':>8}{'ensemble':>10}"
          f"{'vs mean':>9}{'vs best':>9}")
    for a in AGGS:
        v = np.array([members[s][a][0] for s in seeds])
        e = ens[a][0]
        print(f"{AGG_NAME[a]:<13}{v.mean():>14.2f} ± {v.std(ddof=1):<5.2f}"
              f"{v.max():>8.2f}{e:>10.2f}{e - v.mean():>+9.2f}{e - v.max():>+9.2f}")

    # How much of the gain arrives with the first few members. Every subset of
    # each size is scored, so this is an average, not one lucky draw.
    print("\nEnsemble size vs accuracy (mean over all subsets of that size):")
    print(f"{'size':<6}" + "".join(f"{AGG_NAME[a]:>13}" for a in AGGS))
    for size in range(1, len(seeds) + 1):
        accs = {a: [] for a in AGGS}
        for sub in itertools.combinations(range(len(seeds)), size):
            sub = list(sub)
            r = score(loop_p[sub].mean(0), loop_patient,
                      median_p[sub].mean(0), labels, n_cls)
            for a in AGGS:
                accs[a].append(r[a][0])
        print(f"{size:<6}" + "".join(f"{np.mean(accs[a]):>13.2f}" for a in AGGS))

    cprint(f"\nwrote {out}", "highlighted_green")


if __name__ == "__main__":
    main()
