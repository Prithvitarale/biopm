#!/usr/bin/env python3
"""Verify shared wall-clock pos alignment for a paired wrist/hip window."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biopm.cross_joint_data import load_window, WindowRef


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else (
        "/scratch4/workspace/ptarale_umass_edu-light2/figshare_imu_layouts/preprocessed"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else (
        "/scratch4/workspace/ptarale_umass_edu-light2/figshare_imu_layouts/logs/pos_align_wrist_hip.png"
    )
    sid, idx = 1, 0
    rw = WindowRef("wrist", sid, idx,
                   f"{root}/wrist/{sid}/merge_acc_filt_{sid}_{idx}.npy",
                   f"{root}/wrist/{sid}/me_normalizeInfo_padding_acc_filt_{sid}_{idx}.pkl")
    rh = WindowRef("hip", sid, idx,
                   f"{root}/hip/{sid}/merge_acc_filt_{sid}_{idx}.npy",
                   f"{root}/hip/{sid}/me_normalizeInfo_padding_acc_filt_{sid}_{idx}.pkl")
    _, pos_w, _, sew = load_window(rw)
    _, pos_h, _, seh = load_window(rh)
    vw = ~np.isnan(pos_w)
    vh = ~np.isnan(pos_h)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(pos_w[vw], np.zeros(vw.sum()), s=20, label="wrist ME mid pos", alpha=0.7)
    ax.scatter(pos_h[vh], np.ones(vh.sum()) * 0.2, s=20, label="hip ME mid pos", alpha=0.7)
    ax.set_xlabel("shared window-normalized position (ME midpoint / 10s)")
    ax.set_yticks([0, 0.2])
    ax.set_yticklabels(["wrist", "hip"])
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title(f"Paired window sid={sid} idx={idx}: same clock → same pos scale")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print("wrote", out)
    # assert both in [0,1]
    assert np.all((pos_w[vw] >= 0) & (pos_w[vw] <= 1.01))
    assert np.all((pos_h[vh] >= 0) & (pos_h[vh] <= 1.01))
    print("pos range OK")


if __name__ == "__main__":
    main()
