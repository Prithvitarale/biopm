"""Paired / single-joint datasets for cross-joint BioPM pretraining."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .adapters import JOINTS, NON_WRIST


@dataclass
class WindowRef:
    site: str
    subject_id: int
    index: int
    path_merge: str
    path_info: str


def _subject_dirs(data_root: str, site: str) -> Dict[int, str]:
    sroot = os.path.join(data_root, site)
    out = {}
    if not os.path.isdir(sroot):
        return out
    for name in os.listdir(sroot):
        if name.isdigit():
            out[int(name)] = os.path.join(sroot, name)
    return out


def _count_merges(subject_dir: str) -> int:
    return sum(
        1
        for f in os.listdir(subject_dir)
        if f.startswith("merge_acc_filt_") and f.endswith(".npy")
    )


def assert_subject_disjoint(
    train_subjects: Sequence[int],
    eval_subjects: Sequence[int],
    label: str = "figshare",
) -> None:
    inter = set(train_subjects) & set(eval_subjects)
    if inter:
        raise RuntimeError(
            f"Subject-disjointness violated for {label}: "
            f"overlap={sorted(inter)}"
        )


def load_eval_subjects(path: Optional[str]) -> Set[int]:
    if not path:
        return set()
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        ids = obj.get("figshare", obj.get("eval_subjects", []))
    else:
        ids = obj
    return {int(x) for x in ids}


def build_pair_index(
    data_root: str,
    train_subjects: Sequence[int],
) -> List[Tuple[WindowRef, WindowRef]]:
    """Build (wrist, other) pairs aligned by absolute window time.

    Layout assumption (verified on Figshare preprocess):
      bilateral sites store [Left windows | Right windows] with equal counts;
      hip stores only LowerBack with count == left count.
    So wrist[k] aligns with hip[k] and thigh[k] for k < n_hip;
    wrist[k + n_hip] aligns with thigh[k + n_hip] (right side).
    """
    pairs: List[Tuple[WindowRef, WindowRef]] = []
    wrist_dirs = _subject_dirs(data_root, "wrist")
    for sid in train_subjects:
        if sid not in wrist_dirs:
            continue
        n_w = _count_merges(wrist_dirs[sid])
        n_h = _count_merges(os.path.join(data_root, "hip", str(sid)))
        n_t = _count_merges(os.path.join(data_root, "thigh", str(sid)))
        n_a = _count_merges(os.path.join(data_root, "ankle", str(sid)))
        if not (n_w == n_t == n_a and n_w == 2 * n_h and n_h > 0):
            raise RuntimeError(
                f"Unexpected window counts sid={sid}: "
                f"wrist={n_w} hip={n_h} thigh={n_t} ankle={n_a}"
            )
        half = n_h

        def ref(site: str, idx: int) -> WindowRef:
            d = os.path.join(data_root, site, str(sid))
            return WindowRef(
                site=site,
                subject_id=sid,
                index=idx,
                path_merge=os.path.join(d, f"merge_acc_filt_{sid}_{idx}.npy"),
                path_info=os.path.join(
                    d, f"me_normalizeInfo_padding_acc_filt_{sid}_{idx}.pkl"
                ),
            )

        for other in NON_WRIST:
            n_o = {"hip": n_h, "thigh": n_t, "ankle": n_a}[other]
            for i in range(n_o):
                # For hip (half), also pair with right-wrist at i+half.
                pairs.append((ref("wrist", i), ref(other, i)))
                if other == "hip":
                    pairs.append((ref("wrist", i + half), ref(other, i)))
        # bilateral others already cover left+right via i over full n_o
    return pairs


def build_single_index(
    data_root: str,
    train_subjects: Sequence[int],
) -> List[WindowRef]:
    refs: List[WindowRef] = []
    for site in JOINTS:
        for sid in train_subjects:
            d = os.path.join(data_root, site, str(sid))
            if not os.path.isdir(d):
                continue
            n = _count_merges(d)
            for i in range(n):
                refs.append(
                    WindowRef(
                        site=site,
                        subject_id=sid,
                        index=i,
                        path_merge=os.path.join(
                            d, f"merge_acc_filt_{sid}_{i}.npy"
                        ),
                        path_info=os.path.join(
                            d,
                            f"me_normalizeInfo_padding_acc_filt_{sid}_{i}.pkl",
                        ),
                    )
                )
    return refs


def load_window(ref: WindowRef, max_len: int = 32, add_num: int = 5):
    merge = np.load(ref.path_merge)  # (pad, 32+1+5)
    me = merge[:, :max_len].astype(np.float32)
    pos = merge[:, max_len].astype(np.float32)
    add = merge[:, max_len + 1 : max_len + 1 + add_num].astype(np.float32)
    info = pd.read_pickle(ref.path_info)
    se = info[["start_point", "end_point"]].to_numpy(dtype=np.float32)
    # Absolute-time position: ME midpoint in seconds within the shared 10s
    # window, then normalize to [0,1] with the SAME wall-clock alignment
    # (divide by ws=10). Using sample midpoint / (fs*ws) == seconds/ws.
    # Prefer recomputing from start/end so both streams share the same clock.
    fs = 40.0
    ws = 10.0
    mid_samples = (se[:, 0] + se[:, 1]) * 0.5
    valid_row = ~np.isnan(me).any(axis=1)
    pos_abs = mid_samples / (fs * ws)  # in [0,1] if within window
    pos_out = pos.copy()
    pos_out[valid_row] = pos_abs[valid_row]
    # pad rows stay nan
    return me, pos_out, add, se


def contiguous_span_mask(
    se: np.ndarray,
    valid: np.ndarray,
    fs: float = 40.0,
    ws: float = 10.0,
    span_sec: float = 5.0,
) -> np.ndarray:
    """Mask MEs overlapping one contiguous ``span_sec`` span (one side only)."""
    mask = np.zeros(len(valid), dtype=np.float32)
    if not valid.any():
        return mask
    span = span_sec
    # start of span uniform so span fits in window
    t0 = random.uniform(0.0, max(ws - span, 1e-3))
    t1 = t0 + span
    # se is in samples @ fs
    s0, s1 = t0 * fs, t1 * fs
    starts, ends = se[:, 0], se[:, 1]
    for i in np.where(valid)[0]:
        if starts[i] < 0 or starts[i] == -100:
            continue
        if min(ends[i], s1) > max(starts[i], s0):
            mask[i] = 1.0
    # fallback if nothing hit
    if mask.sum() == 0 and valid.any():
        idx = np.where(valid)[0]
        k = max(1, int(0.5 * len(idx)))
        mask[np.random.choice(idx, size=k, replace=False)] = 1.0
    return mask


def within_joint_mask(
    se: np.ndarray,
    valid: np.ndarray,
    fs: float = 40.0,
    ws: float = 10.0,
    mask_rate: float = 0.5,
) -> np.ndarray:
    """BioPM-style mix: 50% uniform ME mask, 50% time-bin mask."""
    from .pretraining import apply_time_based_masking, apply_uniform_masking

    v = torch.as_tensor(valid)
    if random.random() < 0.5:
        m = apply_uniform_masking(v, mask_rate).numpy()
    else:
        m = torch.zeros(len(valid), dtype=torch.float32)
        se_t = torch.as_tensor(se, dtype=torch.float32)
        m = apply_time_based_masking(m, se_t, fs, ws, mask_rate).numpy()
    m = m.astype(np.float32)
    m[~valid] = 0.0
    return m


class CrossJointDataset(Dataset):
    """Yields paired (60%) or single-joint (40%) training items."""

    def __init__(
        self,
        data_root: str,
        train_subjects: Sequence[int],
        paired_frac: float = 0.60,
        samples_per_epoch: int = 1000,
        fs: float = 40.0,
        ws: float = 10.0,
        cross_span_sec: float = 5.0,
        eval_subjects: Optional[Sequence[int]] = None,
    ):
        self.data_root = data_root
        self.paired_frac = float(paired_frac)
        self.samples_per_epoch = int(samples_per_epoch)
        self.fs = fs
        self.ws = ws
        self.cross_span_sec = cross_span_sec

        eval_set = set(eval_subjects or [])
        assert_subject_disjoint(train_subjects, sorted(eval_set))
        self.train_subjects = list(train_subjects)

        self.pairs = build_pair_index(data_root, self.train_subjects)
        self.singles = build_single_index(data_root, self.train_subjects)
        self.pairs_by_other: Dict[str, List[Tuple[WindowRef, WindowRef]]] = {
            j: [] for j in NON_WRIST
        }
        for wa, wb in self.pairs:
            self.pairs_by_other[wb.site].append((wa, wb))
        self.singles_by_joint: Dict[str, List[WindowRef]] = {j: [] for j in JOINTS}
        for r in self.singles:
            self.singles_by_joint[r.site].append(r)
        if not self.pairs or not self.singles:
            raise RuntimeError(
                f"Empty index under {data_root} for subjects {self.train_subjects}"
            )
        for j in NON_WRIST:
            if not self.pairs_by_other[j]:
                raise RuntimeError(f"No pairs for other={j}")
        for j in JOINTS:
            if not self.singles_by_joint[j]:
                raise RuntimeError(f"No singles for joint={j}")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        if random.random() < self.paired_frac:
            return self._get_paired()
        return self._get_single()

    def _pack_stream(self, ref: WindowRef, mask_mode: str):
        me, pos, add, se = load_window(ref)
        valid = ~np.isnan(me).any(axis=1)
        if mask_mode == "within":
            mask = within_joint_mask(se, valid, self.fs, self.ws, 0.5)
        elif mask_mode == "cross":
            mask = contiguous_span_mask(
                se, valid, self.fs, self.ws, self.cross_span_sec
            )
        elif mask_mode == "none":
            mask = np.zeros(len(valid), dtype=np.float32)
        else:
            raise ValueError(mask_mode)
        return {
            "patches": torch.as_tensor(me),
            "pos": torch.as_tensor(pos),
            "mask": torch.as_tensor(mask),
            "add": torch.as_tensor(add),
            "joint": ref.site,
            "subject_id": ref.subject_id,
            "n_valid": int(valid.sum()),
        }

    def _get_paired(self):
        other = random.choice(NON_WRIST)
        wa, wb = random.choice(self.pairs_by_other[other])
        # Cross-joint: mask ONE side only
        if random.random() < 0.5:
            a = self._pack_stream(wa, "cross")
            b = self._pack_stream(wb, "none")
            masked_side = 0
        else:
            a = self._pack_stream(wa, "none")
            b = self._pack_stream(wb, "cross")
            masked_side = 1
        a_w = self._pack_stream(wa, "within")
        b_w = self._pack_stream(wb, "within")
        return {
            "mode": "paired",
            "a": a,
            "b": b,
            "a_within": a_w,
            "b_within": b_w,
            "masked_side": masked_side,
        }

    def _get_single(self):
        joint = random.choice(JOINTS)
        ref = random.choice(self.singles_by_joint[joint])
        s = self._pack_stream(ref, "within")
        return {"mode": "single", "a": s, "b": None, "masked_side": -1}


def collate_cross_joint(batch: List[dict]) -> dict:
    """Pad variable-length isn't needed — all windows already pad_size=192."""
    modes = [b["mode"] for b in batch]
    return {"items": batch, "modes": modes}
