"""Minimal API example: end-to-end feature extraction in ~15 lines.

Assumes you already preprocessed your data into a directory of
``Data_MeLabel_*.h5`` files (e.g. with ``scripts/preprocess_mhealth.py``).
"""

import biopm

# Default checkpoint = 50% masking rate.  Other choices: 0.25 or 0.75.
X, y, subject_ids = biopm.extract_features(
    data_root="preprocessed/mhealth",
    masking_rate=0.5,
    per_axis=True,        # per-axis mean+std pooling (default)
    batch_size=32,
    device="cpu",         # or "cuda:0"
)

print(f"features={X.shape} labels={y.shape} subjects={len(set(subject_ids))}")

# Save for later (e.g. for scripts/classify.py)
import numpy as np
np.savez("features/mhealth_50mr.npz", features=X, labels=y, subject_ids=subject_ids)
