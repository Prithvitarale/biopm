#!/usr/bin/env python3
"""Run subject-aware nested-CV classification on BioPM features.

The protocol matches the paper:

  * LOSO outer CV when <= 10 subjects,
    GroupShuffleSplit(5, 80/20) otherwise.
  * Inner ``C`` selection via a single 12.5 %% held-out split on training
    subjects.
  * StandardScaler refit per outer fold.

Usage::

    python scripts/classify.py --features features/mhealth_50mr.npz

Produces a CSV with one row per fold plus an aggregate row, e.g.::

    fold,macro_f1,accuracy,C
    0,0.812,0.873,1.0
    1,0.795,0.851,1.0
    ...
    mean,0.804,0.866,
    std,0.013,0.011,
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopm import logreg_nested_cv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True,
                   help="Path to an .npz produced by extract_features.py")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Optional CSV path for per-fold + aggregate scores")
    p.add_argument("--val_fraction", type=float, default=0.125)
    p.add_argument("--n_outer_splits", type=int, default=5)
    p.add_argument("--outer_test_size", type=float, default=0.2)
    p.add_argument("--no_progress", action="store_true",
                   help="Suppress the tqdm progress bar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.features, allow_pickle=False)
    if "features" not in data or "labels" not in data:
        raise SystemExit(
            f"{args.features} must contain 'features' and 'labels' arrays.")

    X = data["features"]
    y = data["labels"].astype(int)
    sid_key = "subject_ids" if "subject_ids" in data.files else "pids"
    pids = data[sid_key]

    print("=" * 60)
    print("BioPM downstream classification")
    print("=" * 60)
    print(f"  features          : {X.shape}")
    print(f"  unique labels     : {sorted(np.unique(y).tolist())}")
    print(f"  unique subjects   : {len(np.unique(pids))}")
    print()

    res = logreg_nested_cv(
        X, y, pids,
        val_fraction=args.val_fraction,
        n_outer_splits=args.n_outer_splits,
        outer_test_size=args.outer_test_size,
        verbose=not args.no_progress,
    )

    print(f"  CV               : {res['cv_strategy']}")
    print(f"  n_folds          : {res['n_folds']}")
    print(f"  selected C (mode): {res['C_mode']}")
    print(f"  macro F1         : {res['macro_f1_mean']:.4f} \u00b1 {res['macro_f1_std']:.4f}")
    print(f"  accuracy         : {res['accuracy_mean']:.4f} \u00b1 {res['accuracy_std']:.4f}")

    if args.output_csv:
        rows = []
        for i, (f1, acc, C) in enumerate(zip(res["per_fold_f1"],
                                             res["per_fold_acc"],
                                             res["selected_C"])):
            rows.append({"fold": i, "macro_f1": f1, "accuracy": acc, "C": C})
        rows.append({"fold": "mean",
                     "macro_f1": res["macro_f1_mean"],
                     "accuracy": res["accuracy_mean"], "C": ""})
        rows.append({"fold": "std",
                     "macro_f1": res["macro_f1_std"],
                     "accuracy": res["accuracy_std"], "C": ""})
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".",
                    exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
