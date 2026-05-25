"""Pooling and feature-construction utilities for BioPM.

For each window we produce a fixed-length feature vector of the form

    feature = pool(acc_tokens) || flatten(gravity_resampled)

Two pooling modes are supported:

  * ``per_axis=True``  (default; recommended)
        Mean and std of the per-axis subsets of tokens, concatenated in
        the order [mean_X, std_X, mean_Y, std_Y, mean_Z, std_Z].
        Acc dim = 6 * D = 384  (with D=64).

  * ``per_axis=False`` (legacy / total)
        Joint mean and std over all valid tokens.
        Acc dim = 2 * D = 128  (with D=64).

The gravity stream is the raw low-pass window, interpolated to a fixed
length per axis and flattened.  The default ``gravity_len_per_ch`` is
chosen so that the total feature width is close to 1024:

    per_axis=True  : 384 + 3*213 = 1023
    per_axis=False : 128 + 3*300 = 1028   (matches the paper's main table)

You can override ``gravity_len_per_ch`` to obtain whatever total width
you want.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


NUM_AXES: int = 3


# ---------------------------------------------------------------------------
# Acc pooling
# ---------------------------------------------------------------------------
def total_mean_std(tokens: Tensor, valid_mask: Tensor) -> Tensor:
    """Concatenate mean and std over the valid tokens in each window.

    Args:
        tokens     : (B, L, D)
        valid_mask : (B, L) bool tensor (True = valid token)

    Returns:
        (B, 2 * D)
    """
    m = valid_mask.unsqueeze(-1).float()
    count = m.sum(dim=1).clamp(min=1.0)
    mean = (tokens * m).sum(dim=1) / count
    diff = (tokens - mean.unsqueeze(1)) * m
    var = (diff ** 2).sum(dim=1) / count.clamp(min=2.0)
    std = var.sqrt()
    return torch.cat([mean, std], dim=-1)


def per_axis_mean_std(tokens: Tensor, axis_ids: Tensor,
                      valid_mask: Tensor, num_axes: int = NUM_AXES) -> Tensor:
    """Pool tokens by axis id, concatenating ``[mean_a, std_a]`` for each axis.

    Args:
        tokens     : (B, L, D)
        axis_ids   : (B, L) integer axis ids (NaN-safe; padding entries are
                     ignored by ``valid_mask``).
        valid_mask : (B, L) bool tensor.
        num_axes   : usually 3

    Returns:
        (B, 2 * num_axes * D), ordered [mean_0, std_0, mean_1, std_1, ...]
    """
    feats = []
    axis_ids = torch.nan_to_num(axis_ids, nan=-1.0).long()
    for a in range(num_axes):
        m = ((axis_ids == a) & valid_mask).unsqueeze(-1).float()
        count = m.sum(dim=1).clamp(min=1.0)
        mean = (tokens * m).sum(dim=1) / count
        diff = (tokens - mean.unsqueeze(1)) * m
        var = (diff ** 2).sum(dim=1) / count.clamp(min=2.0)
        std = var.sqrt()
        feats.append(mean)
        feats.append(std)
    return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Gravity stream
# ---------------------------------------------------------------------------
def interpolate_gravity(gravity: Tensor, target_len_per_ch: int) -> Tensor:
    """Linearly interpolate gravity from (B, T, 3) to (B, 3, target_len)
    and flatten to (B, 3 * target_len)."""
    g = gravity.transpose(1, 2)                                   # (B, 3, T)
    g = torch.where(torch.isnan(g), torch.zeros_like(g), g)
    if g.shape[-1] != target_len_per_ch:
        g = F.interpolate(g, size=target_len_per_ch,
                          mode="linear", align_corners=False)
    return g.reshape(g.shape[0], -1)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Per-channel gravity length that the paper uses for each pooling mode.
#
#   * Total mean+std  -> 300 samples/axis = 900 dims of gravity.
#     Combined with the 128-d acc this gives the headline 1028-d feature
#     (matches data_efficiency_4models.py).
#
#   * Per-axis mean+std -> 213 samples/axis = 639 dims of gravity.
#     Combined with the 384-d acc this gives the 1023-d feature used in
#     data_efficiency_per_axis_meanstd.py.
#
# Both choices keep the total feature width close to 1024 so the
# downstream LogReg sees the same input scale regardless of pooling mode.
DEFAULT_GRAV_LEN_TOTAL: int = 300
DEFAULT_GRAV_LEN_PER_AXIS: int = 213


def default_gravity_len_per_ch(per_axis: bool, d_model: int = 64) -> int:
    """Recommended per-channel gravity length for the requested pooling mode."""
    return DEFAULT_GRAV_LEN_PER_AXIS if per_axis else DEFAULT_GRAV_LEN_TOTAL


# ---------------------------------------------------------------------------
# One-shot fused feature
# ---------------------------------------------------------------------------
def fuse_window_feature(
    tokens: Tensor,
    additional_embedding: Tensor,
    raw_patches: Tensor,
    gravity: Tensor,
    *,
    per_axis: bool = True,
    gravity_len_per_ch: Optional[int] = None,
) -> Tensor:
    """Build the BioPM feature vector for a batch of windows.

    Args:
        tokens               : (B, L, D) output of ``encoder_acc``.
        additional_embedding : (B, L, >=1) where channel 0 is the integer
                               axis id (NaN where the row is padding).
        raw_patches          : (B, L, 32) so we can recover the valid mask.
        gravity              : (B, T, 3) raw low-pass gravity window.
        per_axis             : True for the recommended per-axis pooling.
        gravity_len_per_ch   : Length to interpolate gravity to per axis.
                               If None, picks a value that keeps the total
                               feature width close to 1024.

    Returns:
        (B, F) fused feature vector.  F is 1023 if ``per_axis`` else 1028
        (with the default ``gravity_len_per_ch``).
    """
    valid_mask = ~torch.isnan(raw_patches).any(dim=-1)
    if per_axis:
        axis_ids = additional_embedding[:, :, 0]
        acc_feat = per_axis_mean_std(tokens, axis_ids, valid_mask)
    else:
        acc_feat = total_mean_std(tokens, valid_mask)
    if gravity_len_per_ch is None:
        gravity_len_per_ch = default_gravity_len_per_ch(per_axis)
    grav_feat = interpolate_gravity(gravity, gravity_len_per_ch)
    return torch.cat([acc_feat, grav_feat], dim=-1)


# ---------------------------------------------------------------------------
# Convenience: numpy version that mirrors the torch path
# ---------------------------------------------------------------------------
def expected_feature_dim(per_axis: bool, d_model: int = 64,
                         gravity_len_per_ch: Optional[int] = None) -> int:
    """Predict the output dimension of :func:`fuse_window_feature`."""
    acc_dim = (2 * NUM_AXES if per_axis else 2) * d_model
    if gravity_len_per_ch is None:
        gravity_len_per_ch = default_gravity_len_per_ch(per_axis, d_model)
    return acc_dim + NUM_AXES * gravity_len_per_ch
