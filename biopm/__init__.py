"""BioPM: a movement-element transformer for wearable accelerometer data.

Quick start
-----------
>>> from biopm import load_pretrained, extract_features
>>> # End-to-end feature extraction from a directory of preprocessed HDF5s:
>>> X, y, pids = extract_features("preprocessed/", masking_rate=0.5)

Available top-level objects
---------------------------
- ``BioPM``                full model (acc + gravity encoder)
- ``TimeSeriesTransformer`` the acc-stream encoder used standalone
- ``load_pretrained``       load 25 / 50 / 75 MR variants
- ``extract_features``      directory -> (features, labels, pids)
- ``encode_window``         encode a single batch of windows
- ``fuse_window_feature``   pooling + gravity concat (per-axis default)
- ``per_axis_mean_std``     just the per-axis pooler
- ``total_mean_std``        legacy joint mean+std pooler
- ``logreg_nested_cv``      subject-aware nested-CV LogReg evaluator
- ``MovementElementDataset``a Dataset wrapping the preprocessed arrays
- ``load_preprocessed_h5``  read all ``Data_MeLabel_*.h5`` in a directory
- ``PreprocessConfig``      knobs for the preprocessing pipeline
"""

from .model import BioPM, TimeSeriesTransformer, GravityCNNEncoder
from .features import (
    fuse_window_feature, per_axis_mean_std, total_mean_std,
    interpolate_gravity, expected_feature_dim, default_gravity_len_per_ch,
)
from .data import MovementElementDataset
from .preprocessing import (
    PreprocessConfig, load_preprocessed_h5, resample_to_target_fs,
    bandpass_filter, lowpass_filter, detect_zero_crossings,
    pack_window, windowize_and_extract, save_subject_h5,
)
from .evaluation import logreg_nested_cv, DEFAULT_C_GRID, LOSO_THRESHOLD
from .inference import (
    load_pretrained, extract_features, encode_window,
    list_available_checkpoints,
)
from .pretraining import (
    Decode_cnn, PretrainModel, EarlyStopping,
    masked_recon_loss, apply_time_based_masking, apply_uniform_masking,
)

__version__ = "0.1.0"

__all__ = [
    "BioPM",
    "TimeSeriesTransformer",
    "GravityCNNEncoder",
    "MovementElementDataset",
    "PreprocessConfig",
    "load_preprocessed_h5",
    "resample_to_target_fs",
    "bandpass_filter",
    "lowpass_filter",
    "detect_zero_crossings",
    "pack_window",
    "windowize_and_extract",
    "save_subject_h5",
    "fuse_window_feature",
    "per_axis_mean_std",
    "total_mean_std",
    "interpolate_gravity",
    "expected_feature_dim",
    "default_gravity_len_per_ch",
    "logreg_nested_cv",
    "DEFAULT_C_GRID",
    "LOSO_THRESHOLD",
    "load_pretrained",
    "extract_features",
    "encode_window",
    "list_available_checkpoints",
    "Decode_cnn",
    "PretrainModel",
    "EarlyStopping",
    "masked_recon_loss",
    "apply_time_based_masking",
    "apply_uniform_masking",
]
