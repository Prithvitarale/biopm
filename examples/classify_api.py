"""Minimal API example for the downstream classifier.

Run after ``examples/extract_features_api.py`` (or ``scripts/extract_features.py``).
"""

import numpy as np
import biopm

# 1.  Load the features we saved earlier
data = np.load("features/mhealth_50mr.npz")
X, y, pids = data["features"], data["labels"], data["subject_ids"]

# 2.  Subject-aware nested CV.  Matches the paper's evaluation protocol.
res = biopm.logreg_nested_cv(X, y, pids)
print(f"CV = {res['cv_strategy']}  n_folds = {res['n_folds']}")
print(f"macro F1 = {res['macro_f1_mean']:.4f} \u00b1 {res['macro_f1_std']:.4f}")
print(f"accuracy = {res['accuracy_mean']:.4f} \u00b1 {res['accuracy_std']:.4f}")
