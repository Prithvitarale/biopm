#!/usr/bin/env python3
"""Extract BioPM features from preprocessed HDF5 files.

Usage::

    python scripts/extract_features.py \\
        --data_dir     preprocessed/mhealth \\
        --masking_rate 0.5 \\
        --output       features/mhealth_50mr.npz

The output ``.npz`` contains three arrays:

    features      : (N, F) np.float32
    labels        : (N,)   np.int64
    subject_ids   : (N,)   subject identifiers

``F`` is 1023 when ``--per_axis`` is set (the default) and 1028 when
``--no_per_axis`` is passed.  Use ``--gravity_len_per_ch`` to override
the gravity stream length per axis.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopm import extract_features, list_available_checkpoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Directory containing Data_MeLabel_*.h5 files")
    p.add_argument("--masking_rate", type=float, default=0.5,
                   choices=list_available_checkpoints(),
                   help="Pretrained checkpoint to use (default 0.5)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path (overrides --masking_rate)")
    p.add_argument("--output", type=str, default="features/biopm_features.npz",
                   help="Output .npz path (default features/biopm_features.npz)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu or cuda[:idx]")
    pool = p.add_mutually_exclusive_group()
    pool.add_argument("--per_axis", dest="per_axis", action="store_true",
                      default=True,
                      help="Per-axis mean+std pooling (default)")
    pool.add_argument("--no_per_axis", dest="per_axis", action="store_false",
                      help="Use joint mean+std pooling (paper's legacy total)")
    p.add_argument("--gravity_len_per_ch", type=int, default=None,
                   help="Override gravity interpolation length per axis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("BioPM feature extraction")
    print("=" * 60)
    print(f"  data_dir          : {args.data_dir}")
    print(f"  masking_rate      : {args.masking_rate}")
    print(f"  checkpoint        : {args.checkpoint or '(use --masking_rate)'}")
    print(f"  output            : {args.output}")
    print(f"  device            : {args.device}")
    print(f"  per_axis pooling  : {args.per_axis}")
    print(f"  gravity_len/ch    : {args.gravity_len_per_ch or '(auto)'}")
    print()

    features, labels, pids = extract_features(
        data_root=args.data_dir,
        masking_rate=args.masking_rate,
        checkpoint_path=args.checkpoint,
        per_axis=args.per_axis,
        gravity_len_per_ch=args.gravity_len_per_ch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    np.savez(args.output, features=features, labels=labels, subject_ids=pids)
    print(f"\nSaved {features.shape} -> {args.output}")
    print(f"  unique labels   : {sorted(np.unique(labels).tolist())}")
    print(f"  unique subjects : {sorted(np.unique(pids).tolist())}")


if __name__ == "__main__":
    main()
