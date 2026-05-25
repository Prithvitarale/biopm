#!/usr/bin/env python3
"""Preprocess raw mHealth data into BioPM-ready HDF5 files.

This script is intentionally tied to the public mHealth dataset:
https://archive.ics.uci.edu/dataset/319/mhealth+dataset

Treat it as the *reference* preprocessing example.  For your own data:

  1. Copy this file.
  2. Replace ``load_mhealth_subject`` with a loader that returns
     ``(acc_raw_in_g, labels, timestamps_in_seconds)``.
  3. Replace ``label_remap``/``skip_labels`` with the right mapping.

Everything else (filtering / windowing / ME extraction / HDF5 write) is
shared and lives in :mod:`biopm.preprocessing`.

Usage::

    python scripts/preprocess_mhealth.py \\
        --raw_data_dir  /path/to/mhealth/MHEALTHDATASET \\
        --output_dir    preprocessed/mhealth \\
        --window_sec    10 \\
        --slide_sec     5
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# Make this script work without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopm.preprocessing import (
    PreprocessConfig, bandpass_filter, lowpass_filter,
    resample_to_target_fs, save_subject_h5, windowize_and_extract,
)


# mHealth label codebook.  See the dataset README for the meaning.
MHEALTH_RAW_TO_NAME = {
    0:  "Null",
    1:  "Standing still",
    2:  "Sitting and relaxing",
    3:  "Lying down",
    4:  "Walking",
    5:  "Climbing stairs",
    6:  "Waist bends forward",
    7:  "Frontal elevation of arms",
    8:  "Knees bending (crouching)",
    9:  "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back",
}

# Skip "Null" (0) and "Jump front & back" (12) (rare / very short class)
MHEALTH_SKIP = {0, 12}

# Remap remaining labels to a contiguous 0..K-1 range
_KEPT_RAW = sorted(set(MHEALTH_RAW_TO_NAME.keys()) - MHEALTH_SKIP)
MHEALTH_REMAP = {raw: new for new, raw in enumerate(_KEPT_RAW)}


def load_mhealth_subject(file_path: str, ori_fs: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one ``mHealth_subjectN.log`` file.

    Returns:
        acc_g       : (N, 3) right-lower-arm accelerometer in *g* units.
        labels      : (N,)   integer activity labels.
        time_array  : (N,)   synthetic timestamps at ``ori_fs`` Hz.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    # mHealth columns: 14,15,16 = right-lower-arm accel (m/s^2),  23 = label
    acc = df.iloc[:, [14, 15, 16]].apply(pd.to_numeric, errors="coerce").interpolate().values
    labels = df.iloc[:, 23].apply(pd.to_numeric, errors="coerce").interpolate().values
    acc_g = acc / 9.80665  # m/s^2 -> g
    time_array = np.arange(len(acc_g)) / ori_fs
    return acc_g, labels.astype(int), time_array


def preprocess_one_subject(file_path: str, subject_id, cfg: PreprocessConfig,
                           output_dir: str) -> None:
    acc_g, labels, time_array = load_mhealth_subject(file_path, cfg.ori_fs)

    # Resample to target_fs
    acc_rs, time_rs, lab_rs = resample_to_target_fs(
        time_array, acc_g, labels, cfg.target_fs)

    # Bandpass = body acceleration;  lowpass = gravity
    acc_filt = bandpass_filter(acc_rs, cfg.low_f1, cfg.high_f1,
                               cfg.target_fs, order=cfg.order)
    acc_grav = lowpass_filter(acc_rs, cfg.low_f1, cfg.target_fs,
                              order=cfg.order)

    win_acc_raw, win_x_acc, win_x_grav, win_labels = windowize_and_extract(
        acc_resampled=acc_rs,
        acc_filtered=acc_filt,
        acc_gravity=acc_grav,
        time_resampled=time_rs,
        labels=lab_rs,
        cfg=cfg,
        skip_labels=MHEALTH_SKIP,
        label_remap=MHEALTH_REMAP,
    )

    if len(win_labels) == 0:
        print(f"  WARNING: no valid windows for subject {subject_id}")
        return

    path = save_subject_h5(output_dir, subject_id,
                           win_acc_raw, win_x_acc, win_x_grav, win_labels)
    print(f"  saved {len(win_labels):4d} windows -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_data_dir", required=True,
                   help="Directory containing mHealth_subject*.log files")
    p.add_argument("--output_dir", required=True,
                   help="Directory for the BioPM HDF5 outputs")
    p.add_argument("--window_sec", type=int, default=10)
    p.add_argument("--slide_sec", type=int, default=5)
    p.add_argument("--ori_fs", type=int, default=50)
    p.add_argument("--target_fs", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        ori_fs=args.ori_fs,
        target_fs=args.target_fs,
        window_sec=args.window_sec,
        slide_sec=args.slide_sec,
        pad_size=int(args.window_sec * 192 / 10),
    )
    print(f"Preprocessing config: {cfg}")

    raw_dir = args.raw_data_dir
    files = sorted(f for f in os.listdir(raw_dir)
                   if f.startswith("mHealth_subject") and f.endswith(".log"))
    if not files:
        raise SystemExit(f"No mHealth_subject*.log files in {raw_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in files:
        sid = fname.split("subject")[-1].split(".")[0]
        try:
            sid = int(sid)
        except ValueError:
            pass
        print(f"Processing subject {sid} ({fname}) ...")
        try:
            preprocess_one_subject(os.path.join(raw_dir, fname), sid, cfg,
                                   args.output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone.  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
