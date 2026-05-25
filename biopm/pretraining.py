"""Pretraining-time pieces for BioPM.

The inference-only ``biopm.model.TimeSeriesTransformer`` ships without
the decoder used during masked-patch pretraining.  This module supplies:

  * ``Decode_cnn``               -- 1-D ConvTranspose decoder back to
                                   ``patch_len`` samples.
  * ``PretrainModel``            -- wraps the released encoder + decoder
                                   and applies the BERT-style noisy /
                                   random-patch replacement used during
                                   pretraining.
  * ``masked_recon_loss``        -- weighted masked / unmasked L1 loss.
  * ``apply_time_based_masking`` -- time-grid masking strategy.
  * ``EarlyStopping``            -- tiny inline replacement for the
                                   `pytorchtools` helper used in the
                                   original training script.

The pretraining checkpoints shipped in ``checkpoints/`` were produced by
loading exactly these modules with ``mask_rate=0.25 / 0.50 / 0.75``;
they save the same state dict layout the inference-time
``BioPM.load_checkpoint`` already understands (which silently drops the
decoder weights on load).
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import D_MODEL, PATCH_LEN, TimeSeriesTransformer


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decode_cnn(nn.Module):
    """ConvTranspose decoder: (B, L, D=64) -> (B, L, PATCH_LEN=32).

    Architecture matches the one used to pretrain the released
    checkpoints.  Each ME token is upsampled independently.
    """

    def __init__(self, embed_dim: int = D_MODEL, patch_len: int = PATCH_LEN):
        super().__init__()
        self.patch_len = patch_len
        self.decode_cnn = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 64, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=False),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=False),
            nn.BatchNorm1d(8), nn.GELU(),
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2,
                               padding=1, output_padding=0),
        )

    def forward(self, h: Tensor) -> Tensor:
        B, L, D = h.shape
        out = self.decode_cnn(h.reshape(-1, D, 1))
        return out.reshape(B, L, -1)[:, :, : self.patch_len]


# ---------------------------------------------------------------------------
# Pretraining wrapper
# ---------------------------------------------------------------------------
class PretrainModel(nn.Module):
    """Encoder + decoder used during masked-patch pretraining.

    During training (``self.training == True``) it also applies the
    BERT-style perturbations the paper used:

      * patches in ``mask_info`` are replaced with Gaussian noise *before*
        the encoder sees them (the encoder then overwrites them with the
        learnable mask token, as in standard MAE);
      * an additional 20 %% of *unmasked* patches are swapped with random
        unmasked patches from elsewhere in the batch.

    Output shape: ``(B, L, PATCH_LEN)`` -- the reconstructed patches.
    """

    def __init__(self, encoder: Optional[TimeSeriesTransformer] = None,
                 patch_len: int = PATCH_LEN, bert_swap_prob: float = 0.20,
                 bert_log_every: float = 0.001):
        super().__init__()
        self.encoder_acc = encoder if encoder is not None else TimeSeriesTransformer()
        self.decoder_cnn = Decode_cnn(D_MODEL, patch_len)
        self.bert_swap_prob = float(bert_swap_prob)
        self.bert_log_every = float(bert_log_every)

    def forward(self, patches: Tensor, patch_indices: Tensor,
                mask_info: Tensor, additional_embedding: Tensor) -> Tensor:
        if self.training:
            patches = _bert_perturb(
                patches, mask_info,
                swap_prob=self.bert_swap_prob,
                log_every=self.bert_log_every,
            )
        h = self.encoder_acc(patches, patch_indices, mask_info, additional_embedding)
        return self.decoder_cnn(h)


def _bert_perturb(patches: Tensor, mask_info: Tensor,
                  swap_prob: float = 0.20,
                  log_every: float = 0.001) -> Tensor:
    """Apply the BERT-style noisy / swap perturbation used in the paper."""
    if log_every > 0 and random.random() < log_every:
        print("bert masking")
    B, T, D = patches.shape
    mask_flag = mask_info.clone().bool()

    # 1. fill masked positions with noise (encoder still overwrites these
    #    with the learnable mask token; this only matters because mask_info
    #    masking is applied to the *encoded* patches, not the raw ones,
    #    further down).
    out = patches.clone()
    mask_exp = mask_flag.unsqueeze(-1).expand_as(out)
    out = torch.where(mask_exp, torch.randn_like(out), out)

    # 2. with prob `swap_prob`, replace an unmasked patch with a random
    #    other patch from the same window (per-sample reshuffle).
    bert_mask_flag = (torch.rand_like(mask_flag.float()) < swap_prob) & ~mask_flag
    bert_exp = bert_mask_flag.unsqueeze(-1).expand_as(out)
    rand_src = torch.randint(T, (B, T), device=patches.device)
    random_patches = out.gather(1, rand_src.unsqueeze(-1).expand(-1, -1, D))
    return torch.where(bert_exp, random_patches, patches)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def masked_recon_loss(pred: Tensor, target: Tensor, mask_info: Tensor,
                      weight_for_masked: float = 100.0
                      ) -> Tuple[Tensor, float, float]:
    """Weighted L1 reconstruction loss.

    Returns
        loss            : scalar tensor (for backward)
        masked_l1       : weight * mean L1 over masked positions   (logging)
        unmasked_l1     :          mean L1 over unmasked positions (logging)
    """
    valid = ~torch.isnan(target)
    mask = mask_info.unsqueeze(-1).expand_as(target).bool()
    err_m = (pred[valid & mask] - target[valid & mask]).abs()
    err_u = (pred[valid & ~mask] - target[valid & ~mask]).abs()
    loss = (
        weight_for_masked * err_m.sum() / max(err_m.numel(), 1)
        + err_u.sum() / max(err_u.numel(), 1)
    )
    with torch.no_grad():
        masked_l1 = float(weight_for_masked * err_m.mean().item()) if err_m.numel() else 0.0
        unmasked_l1 = float(err_u.mean().item()) if err_u.numel() else 0.0
    return loss, masked_l1, unmasked_l1


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------
def apply_time_based_masking(mask_info: Tensor, se_time_index: Tensor,
                             fs: float, ws: float, mask_rate: float) -> Tensor:
    """Mask all MEs that overlap a random set of 1-second time bins.

    Mirrors the time-grid strategy from the original training script:
    pick ``mask_rate * (FS * WS / FS) = mask_rate * WS`` 1-second bins
    inside the window, then mask every ME that intersects any of them.
    """
    out = mask_info.clone()
    out[out == 1] = 0
    step = int(fs)
    candidate = torch.arange(0, int(fs * ws), step=step)
    num_select = max(1, int(len(candidate) * mask_rate))
    selected = candidate[torch.randperm(len(candidate))[:num_select]]
    starts, ends = se_time_index[:, 0], se_time_index[:, 1]
    for t in selected:
        intersect = torch.minimum(ends, t + step) > torch.maximum(starts, t)
        out[intersect] = 1
    return out


def apply_uniform_masking(valid_mask: Tensor, mask_rate: float) -> Tensor:
    """Uniformly sample ``mask_rate`` of the *valid* MEs and flag them."""
    out = torch.zeros_like(valid_mask, dtype=torch.float32)
    valid_idx = torch.where(valid_mask)[0]
    num_to_mask = int(len(valid_idx) * mask_rate)
    if num_to_mask > 0 and len(valid_idx) > 0:
        mi = valid_idx[torch.randperm(len(valid_idx))[:num_to_mask]]
        out[torch.sort(mi).values] = 1.0
    return out


# ---------------------------------------------------------------------------
# Early stopping (inline replacement for pytorchtools.EarlyStopping)
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Minimal early-stopping helper.

    Saves the model state dict to ``path`` whenever the monitored metric
    (lower is better) improves by at least ``delta``.  Sets
    ``self.early_stop = True`` after ``patience`` non-improving epochs.
    """

    def __init__(self, patience: int = 50, path: str = "checkpoint.pt",
                 delta: float = 0.0, verbose: bool = False):
        self.patience = int(patience)
        self.path = path
        self.delta = float(delta)
        self.verbose = bool(verbose)
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_loss: float = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -float(val_loss)
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            print(f"Val loss {self.best_loss:.6f} -> {val_loss:.6f}, saving {self.path}")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        state = (model.module.state_dict()
                 if hasattr(model, "module") else model.state_dict())
        torch.save(state, self.path)
        self.best_loss = float(val_loss)
