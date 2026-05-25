"""Signal-processing utilities for BioPM.

Pipeline (run by ``scripts/preprocess_mhealth.py`` as a reference):

  1. Resample raw 3-axis accelerometer to ``target_fs`` Hz.
  2. 0.5-12 Hz bandpass filter         -> body-acceleration signal.
  3. 0.5 Hz lowpass filter            -> gravity signal.
  4. Slide a window of ``window_sec`` seconds with ``slide_sec`` overlap.
  5. Per window and per axis, detect zero crossings of the body signal
     and extract movement-element (ME) segments.
  6. Resample each ME to a fixed length (``normalize_size``, default 32)
     and pack it into a row of shape
     ``[norm_me(32) | pos(1) | axis(1) | len(1) | min(1) | max(1) | dirct(1)]``.
  7. Pad to ``pad_size`` rows per window with NaNs.
  8. Save one HDF5 per subject with keys ``window_acc_raw``, ``x_acc_filt``,
     ``x_gravity`` and ``window_label``.

The HDF5 layout is what ``biopm.data.load_preprocessed_h5`` consumes.
"""

from __future__ import annotations

import os
import re
import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import brentq
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# Default configuration (matches the paper's mHealth pipeline)
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    high_f1: float = 12.0          # bandpass upper cutoff
    low_f1: float = 0.5            # bandpass lower cutoff (= gravity cutoff)
    order: int = 6                 # Butterworth filter order
    ori_fs: int = 50               # raw sample rate (override per dataset)
    target_fs: int = 30            # resample target
    window_sec: int = 10           # window length in seconds
    slide_sec: int = 5             # slide / hop in seconds
    normalize_size: int = 32       # samples per ME after resampling
    pad_size: int = 192            # max MEs per window (length-dependent)

    @property
    def ws_samples(self) -> int:
        return int(self.window_sec * self.target_fs)

    @property
    def step_samples(self) -> int:
        return int(self.slide_sec * self.target_fs)

    @property
    def feature_columns(self) -> int:
        """Number of columns of `x_acc_filt[w, l, :]`."""
        # norm_me + pos + (axis, len, min, max, dirct)
        return self.normalize_size + 1 + 5


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def resample_to_target_fs(time_array: np.ndarray,
                          acc_raw: np.ndarray,
                          label: np.ndarray,
                          target_fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample 3-axis accelerometer + nearest-neighbour labels."""
    t_new = np.linspace(time_array[0], time_array[-1],
                        int(time_array[-1] * target_fs))
    cols = [interp1d(time_array, acc_raw[:, i], kind="linear")(t_new).reshape(-1, 1)
            for i in range(acc_raw.shape[1])]
    acc_new = np.concatenate(cols, axis=1)
    label_new = interp1d(time_array, label, kind="nearest")(t_new)
    return acc_new, t_new, label_new


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def bandpass_filter(data: np.ndarray, low_hz: float, high_hz: float,
                    fs: int, order: int = 6) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def lowpass_filter(data: np.ndarray, cutoff_hz: float,
                   fs: int, order: int = 6) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


# ---------------------------------------------------------------------------
# Zero-crossing ME extraction
# ---------------------------------------------------------------------------
def detect_zero_crossings(
    vel: np.ndarray, time_index: np.ndarray, cfg: PreprocessConfig,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, List[np.ndarray], List[List[float]]]:
    """Detect per-axis zero crossings of `vel` and extract normalised MEs.

    Returns
        me_normalize     : (n_me, normalize_size)
        me_info          : DataFrame with one row per ME (axis, start, end,
                           len, min, max, dirct, peaks)
        pos_info         : (n_me,) fractional centre position in [0, 1]
        zc_idx_list      : per-axis list of integer crossing indices (used
                           to align the gravity stream)
        zc_time_list     : per-axis list of floating-point crossing times
    """
    nsize = cfg.normalize_size
    me_normalize = np.empty((0, nsize))
    me_info = pd.DataFrame([])
    pos_info: List[float] = []
    zc_idx_list: List[np.ndarray] = []
    zc_time_list: List[List[float]] = []

    for axis_id in range(vel.shape[1]):
        spline = UnivariateSpline(time_index, vel[:, axis_id], s=0)

        zc_time: List[float] = []
        for ii in range(len(time_index) - 1):
            if spline(time_index[ii]) * spline(time_index[ii + 1]) < 0:
                zc_time.append(brentq(spline, time_index[ii], time_index[ii + 1]))
        zc_idx = [np.searchsorted(time_index, t) - 1 for t in zc_time]
        zc_idx = [i for i in zc_idx if 0 <= i < len(time_index) - 1]
        zc_idx = np.array(zc_idx)
        if zc_idx.size == 0:
            zc_idx_list.append(np.array([], dtype=int))
            zc_time_list.append([])
            continue

        # drop crossings that are too close together (<50 ms apart)
        min_gap = round(cfg.target_fs * 0.05)
        filt_idx = [zc_idx[0]]
        filt_time = [zc_time[0]]
        for aa in range(1, len(zc_idx)):
            if zc_idx[aa] - filt_idx[-1] > min_gap:
                filt_idx.append(zc_idx[aa])
                filt_time.append(zc_time[aa])

        # merge MEs whose peak amplitude is below threshold
        amp = [np.max(vel[filt_idx[ii]:filt_idx[ii + 1] + 2, axis_id])
               for ii in range(len(filt_idx) - 1)]
        amp = np.array(amp) if amp else np.array([])
        if amp.size > 0:
            low_mask = amp < 0.01
            small = np.where(low_mask)[0]
            if small.size > 0:
                groups = np.split(small, np.where(np.diff(small) > 1)[0] + 1)
                drops = np.concatenate([g[1:] for g in groups if len(g) > 1]) \
                    if any(len(g) > 1 for g in groups) else np.array([], dtype=int)
                if drops.size > 0:
                    filt_idx = np.delete(filt_idx, drops)
                    filt_time = np.delete(filt_time, drops)
        zc_idx_arr = np.asarray(filt_idx)
        zc_time_arr = np.asarray(filt_time)
        zc_idx_list.append(zc_idx_arr)
        zc_time_list.append(list(zc_time_arr))

        for ii in range(len(zc_idx_arr) - 1):
            t0, t1 = zc_time_arr[ii], zc_time_arr[ii + 1]
            t_new = np.linspace(t0, t1, nsize)
            c_vel = spline(t_new)
            direct = 1 if np.mean(c_vel) > 0 else -1
            normed = c_vel * direct
            mn, mx = normed.min(), normed.max()
            raw_len = zc_idx_arr[ii + 1] - zc_idx_arr[ii] + 2
            peaks, _ = find_peaks((normed - mn) / (mx - mn + 1e-10),
                                  height=0, prominence=0.3)
            pos = ((zc_idx_arr[ii] + zc_idx_arr[ii + 1]) / 2) / cfg.ws_samples
            pos_info.append(pos)
            me_normalize = np.concatenate((me_normalize, c_vel.reshape(1, -1)))
            row = pd.DataFrame([{
                "axis": axis_id, "start_point": zc_idx_arr[ii],
                "end_point": zc_idx_arr[ii + 1], "len": raw_len,
                "min": mn, "max": mx, "dirct": direct, "peaks": len(peaks),
            }])
            me_info = pd.concat([me_info, row], axis=0)

    # truncate to pad_size
    if len(me_normalize) > cfg.pad_size:
        me_normalize = me_normalize[: cfg.pad_size]
        me_info = me_info.iloc[: cfg.pad_size]
        pos_info = pos_info[: cfg.pad_size]

    return me_normalize, me_info.reset_index(drop=True), np.array(pos_info), \
        zc_idx_list, zc_time_list


def pack_window(me_normalize: np.ndarray,
                me_info: pd.DataFrame,
                pos_info: np.ndarray,
                cfg: PreprocessConfig) -> np.ndarray:
    """Pack one window's MEs into the dense `(pad_size, feature_columns)` layout."""
    if len(me_normalize) == 0:
        return np.full((cfg.pad_size, cfg.feature_columns), np.nan, dtype=np.float32)
    x = np.concatenate([
        me_normalize,
        pos_info.reshape(-1, 1),
        me_info[["axis", "len", "min", "max", "dirct"]].values,
    ], axis=1).astype(np.float32)
    if x.shape[0] < cfg.pad_size:
        pad = np.full((cfg.pad_size - x.shape[0], x.shape[1]),
                      np.nan, dtype=np.float32)
        x = np.vstack([x, pad])
    return x[: cfg.pad_size]


# ---------------------------------------------------------------------------
# Window loop
# ---------------------------------------------------------------------------
def windowize_and_extract(
    acc_resampled: np.ndarray,
    acc_filtered: np.ndarray,
    acc_gravity: np.ndarray,
    time_resampled: np.ndarray,
    labels: np.ndarray,
    cfg: PreprocessConfig,
    skip_labels: set = frozenset(),
    label_remap: Dict[int, int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterate sliding windows and produce four arrays:

        win_acc_raw : (W, T, 3) raw acceleration
        win_x_acc   : (W, pad_size, 37) ME features
        win_x_grav  : (W, T, 3) gravity signal
        win_labels  : (W,) integer labels
    """
    import statistics as _stat
    ws, step = cfg.ws_samples, cfg.step_samples
    out_raw, out_x_acc, out_x_grav, out_labels = [], [], [], []

    start = 0
    while start + ws < acc_filtered.shape[0]:
        win_label = labels[start:start + ws].astype(int)
        try:
            mode_label = _stat.mode(win_label.tolist())
        except _stat.StatisticsError:
            start += step
            continue
        if mode_label in skip_labels:
            start += step
            continue

        w_filt = acc_filtered[start:start + ws]
        w_grav = acc_gravity[start:start + ws]
        w_raw = acc_resampled[start:start + ws]
        w_time = time_resampled[start:start + ws]
        try:
            me_norm, me_info, pos, _, _ = detect_zero_crossings(w_filt, w_time, cfg)
        except Exception:
            start += step
            continue

        if len(me_norm) == 0:
            start += step
            continue

        x_acc = pack_window(me_norm, me_info, pos, cfg)
        mapped = label_remap[mode_label] if label_remap is not None else mode_label
        out_raw.append(w_raw.astype(np.float32))
        out_x_acc.append(x_acc.astype(np.float32))
        out_x_grav.append(w_grav.astype(np.float32))
        out_labels.append(float(mapped))
        start += step

    return (np.asarray(out_raw, dtype=np.float32),
            np.asarray(out_x_acc, dtype=np.float32),
            np.asarray(out_x_grav, dtype=np.float32),
            np.asarray(out_labels, dtype=np.float32))


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_subject_h5(out_dir: str, subject_id, win_acc_raw: np.ndarray,
                    win_x_acc: np.ndarray, win_x_grav: np.ndarray,
                    win_labels: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"Data_MeLabel_{subject_id}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("window_acc_raw", data=win_acc_raw)
        f.create_dataset("x_acc_filt", data=win_x_acc)
        f.create_dataset("x_gravity", data=win_x_grav)
        f.create_dataset("window_label", data=win_labels)
    return path


def load_preprocessed_h5(data_root: str):
    """Load all ``Data_MeLabel_*.h5`` files in a directory.

    Returns a tuple ready to feed into ``biopm.data.MovementElementDataset``:
        ``(acc_patches, pos_info, additional_embedding, labels,
           subject_ids, gravity_windows, raw_acc_windows)``.
    """
    pattern = re.compile(r"Data_MeLabel_.*\.h5")
    matched = []
    for root, _, fnames in os.walk(data_root):
        for f in fnames:
            if pattern.match(f):
                matched.append(os.path.join(root, f))
    if not matched:
        raise FileNotFoundError(f"No Data_MeLabel_*.h5 files in {data_root}")

    all_acc, all_lab, all_pid, all_grav, all_raw = [], [], [], [], []
    for p in sorted(matched):
        parts = p.replace(".h5", "").split("_")
        try:
            subject_id = int(parts[-1])
        except ValueError:
            subject_id = parts[-1]
        with h5py.File(p, "r") as hf:
            all_acc.append(np.array(hf["x_acc_filt"]))
            all_raw.append(np.array(hf["window_acc_raw"]))
            all_lab.append(np.array(hf["window_label"]))
            if "x_gravity" in hf:
                all_grav.append(np.array(hf["x_gravity"]))
            elif "gravity_window_40hz" in hf:
                all_grav.append(np.array(hf["gravity_window_40hz"]))
        n = len(all_lab[-1])
        all_pid.append(np.full(n, subject_id))

    acc = np.concatenate(all_acc)
    raw = np.concatenate(all_raw)
    lab = np.concatenate(all_lab)
    pid = np.concatenate(all_pid)
    grav = np.concatenate(all_grav) if all_grav else None

    NORM = 32
    me_patches = acc[:, :, :NORM]
    pos_info = acc[:, :, NORM]
    additional_embedding = acc[:, :, NORM + 1:]
    return me_patches, pos_info, additional_embedding, lab, pid, grav, raw
