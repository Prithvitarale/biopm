# BioPM

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reference implementation, pretrained checkpoints, preprocessing pipeline
and downstream-classification protocol for **BioPM**, a movement-element
transformer for wearable 3-axis accelerometer data (ICML 2026).

> **Paper:** *coming soon*  &nbsp;·&nbsp;  **[Project page](https://prithvitarale.github.io/biopm-site/)**

The release is intentionally minimal: just the BioPM model, the
mHealth-style preprocessing it was trained for, a one-call feature
extractor, and a subject-aware nested-CV classifier.  All other
baselines / ablations from the paper live in a separate research repo.

---

## What's in the box

```
biopm/
├── biopm/                            ← Python package
│   ├── model.py                      ← BioPM (acc encoder + optional gravity CNN)
│   ├── preprocessing.py              ← resample / filter / zero-crossing ME
│   ├── features.py                   ← per-axis + total mean+std pooling, gravity fuse
│   ├── data.py                       ← MovementElementDataset
│   ├── evaluation.py                 ← subject-aware nested-CV LogReg
│   ├── pretraining.py                ← Decode_cnn + PretrainModel + masking utils
│   └── inference.py                  ← load_pretrained + extract_features
├── scripts/
│   ├── preprocess_mhealth.py         ← reference preprocessing pipeline
│   ├── extract_features.py           ← CLI wrapper around biopm.extract_features
│   ├── classify.py                   ← CLI wrapper around biopm.logreg_nested_cv
│   └── pretrain.py                   ← reference pretraining recipe (single-GPU + DDP)
├── checkpoints/                      ← three pretrained variants
│   ├── biopm_25mr.pt                 ← 25 %% masking rate
│   ├── biopm_50mr.pt                 ← 50 %% masking rate (default)
│   └── biopm_75mr.pt                 ← 75 %% masking rate
├── examples/                         ← short API snippets
└── requirements.txt
```

---

## Install

```bash
git clone <this-repo>
cd biopm
pip install -r requirements.txt
# (optional) install as an editable package so `import biopm` works
# from anywhere
pip install -e .
```

The three pretrained checkpoints are already in `checkpoints/`.  No
download step is required.

---

## Quick start

```python
import biopm

# 50 %% masking rate variant (default).  Other choices: 0.25 or 0.75.
features, labels, subject_ids = biopm.extract_features(
    data_root="preprocessed/mhealth",
    masking_rate=0.5,
    per_axis=True,        # the per-axis mean+std pooler (default)
    batch_size=32,
    device="cpu",
)

# features.shape == (N, 1023)  — per-axis acc(384) + gravity(639)
# labels.shape   == (N,)
# subject_ids    == (N,) subject ids per window

# Subject-aware nested CV exactly as in the paper
result = biopm.logreg_nested_cv(features, labels, subject_ids)
print(result["cv_strategy"], result["macro_f1_mean"], result["macro_f1_std"])
```

For the legacy *total* mean+std pooling (1028-d, matches the paper's
main table), call

```python
features, *_ = biopm.extract_features(data_root="...", per_axis=False)
# features.shape == (N, 1028)  — total acc(128) + gravity(900)
```

---

## Full pipeline

### 1. Preprocess raw data

`scripts/preprocess_mhealth.py` is a worked example for the public
[mHealth dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset).
Copy it and adapt the loader for your own dataset.

```bash
python scripts/preprocess_mhealth.py \
    --raw_data_dir  /path/to/MHEALTHDATASET \
    --output_dir    preprocessed/mhealth \
    --window_sec    10 \
    --slide_sec     5 \
    --ori_fs        50 \
    --target_fs     30
```

The pipeline does:

| Step | What it does                                                                 |
| ---- | ---------------------------------------------------------------------------- |
| 1    | Resample to 30 Hz (configurable)                                             |
| 2    | 0.5-12 Hz Butterworth bandpass → body-acceleration signal                    |
| 3    | 0.5 Hz Butterworth lowpass → gravity signal                                  |
| 4    | Sliding 10-second windows with 5-second overlap                              |
| 5    | Per-axis spline-based zero-crossing detection → movement elements (MEs)      |
| 6    | Resample each ME to 32 samples and tag it with `[axis, len, min, max, dirct]`|
| 7    | Save one HDF5 per subject (`Data_MeLabel_{subject}.h5`)                      |

### 2. Extract features

```bash
python scripts/extract_features.py \
    --data_dir     preprocessed/mhealth \
    --masking_rate 0.5 \
    --output       features/mhealth_50mr.npz \
    --device       cpu        # or cuda:0
```

By default this uses per-axis mean+std pooling, producing 1023-d
features.  Pass `--no_per_axis` for the legacy joint mean+std pooling
(1028-d).

### 3. Downstream classification (nested CV)

```bash
python scripts/classify.py --features features/mhealth_50mr.npz \
                           --output_csv reports/mhealth_50mr_folds.csv
```

The classifier follows the same protocol used in the paper:

- **outer CV**: LOSO when there are ≤ 10 subjects, else
  `GroupShuffleSplit(n_splits=5, test_size=0.2)`;
- **inner C selection**: one `GroupShuffleSplit` with 12.5 %% of the
  training subjects held out, grid search over
  `C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}`;
- `StandardScaler` refit per outer fold;
- multinomial `LogisticRegression(solver="lbfgs", max_iter=1000)`.

---

## Model details

BioPM has two streams:

1. **Acc encoder** (`biopm.model.TimeSeriesTransformer`)
   - 1-D CNN per ME patch (32 samples → 60-d).
   - Concatenate a 3-d learnable axis embedding and a scalar duration → 64-d token.
   - Add a positional embedding computed from the ME's fractional position
     in the window (MLP `1 → 64 → 64`).
   - Five layers of relative-position multi-head attention (4 heads,
     `max_rel_pos=15`).
   - Output: one 64-d token per ME.

2. **Gravity stream** (used during feature extraction, not learned)
   - The low-pass gravity window is linearly interpolated to a fixed
     length per axis (213 by default for per-axis pooling, 300 for
     total pooling) and flattened.

The final feature is the concatenation of pooled acc tokens and the
flattened gravity stream:

|                 | acc dim | gravity dim | total |
| --------------- | ------- | ----------- | ----- |
| `per_axis=True` | 384     | 639         | 1023  |
| `per_axis=False`| 128     | 900         | 1028  |

`per_axis=True` is the new default — empirically it gives stronger
downstream classification at the same total feature width.  Pass
`per_axis=False` if you want to reproduce the paper's headline table.

### Pretraining from scratch

You don't need this for downstream classification — the three checkpoints
in `checkpoints/` are ready to use.  It's here so the recipe used to
produce them is fully reproducible.

```bash
# Single GPU
python scripts/pretrain.py \
    --data_dir   /path/to/preprocessed_data_bins \
    --output_dir runs/biopm_50mr \
    --mask_rate  0.50

# All visible GPUs via DDP
python scripts/pretrain.py \
    --data_dir   /path/to/preprocessed_data_bins \
    --output_dir runs/biopm_50mr \
    --mask_rate  0.50 \
    --ddp
```

Set `--mask_rate 0.25` or `--mask_rate 0.75` to reproduce the other two
released checkpoints.

Pretraining expects one *file per window* (UKBB-scale streaming layout):

```
<data_dir>/<subject_id>/
    merge_acc_filt_<subject_id>_<idx>.npy             # (pad_size, 38)
    window_acc_filt_<subject_id>_<idx>.npy            # used only for file discovery
    me_normalizeInfo_padding_acc_filt_<subject_id>_<idx>.pkl
```

`merge_*.npy` is the same dense ME row produced by
`biopm.preprocessing.pack_window` (32 + 1 pos + 5 add). See the docstring
at the top of `scripts/pretrain.py` for the full spec.

### Three pretrained checkpoints

| File             | Masking rate | When to use                            |
| ---------------- | ------------ | -------------------------------------- |
| `biopm_25mr.pt`  | 25%          | sparser masking; smoother fine-tuning  |
| `biopm_50mr.pt`  | 50%          | **paper default** — best on benchmarks |
| `biopm_75mr.pt`  | 75%          | aggressive masking variant             |

All three share the *exact same* architecture; only the pretraining
masking ratio differs.  Load via:

```python
biopm.load_pretrained(masking_rate=0.50)   # or 0.25 / 0.75
```

---

## Citation

```bibtex
@inproceedings{biopm2026,
  title     = {BioPM: Biological Primitives Model for Wearable Accelerometer Data},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
}
```

(The bibtex entry will be replaced by the official one once the
proceedings are published.)

---

## License

MIT.  See `LICENSE`.
