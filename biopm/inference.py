"""High-level inference API for BioPM.

This is the module most users will touch.  The two main entry points are

  * :func:`load_pretrained`
        Load one of the three shipped checkpoints (25% / 50% / 75%
        masking rate) into a fresh :class:`biopm.model.BioPM` instance.
  * :func:`extract_features`
        End-to-end feature extraction from a directory of preprocessed
        HDF5 files.  Produces a ``(N, F)`` array where F is 1023 (default,
        per-axis pooling) or 1028 (legacy total pooling).

There is also :func:`encode_window` for batched inference on a single
window tensor — useful if you have already preprocessed data into
PyTorch tensors and just want to run the encoder.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it

from .model import BioPM
from .features import fuse_window_feature, expected_feature_dim
from .data import MovementElementDataset
from .preprocessing import load_preprocessed_h5


# Path of the shipped checkpoint directory.  Used by load_pretrained when
# the caller passes a masking rate instead of an explicit path.
_DEFAULT_CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints",
)
_CKPT_FOR_MASK_RATE = {
    0.25: "biopm_25mr.pt",
    0.50: "biopm_50mr.pt",
    0.75: "biopm_75mr.pt",
}


def list_available_checkpoints() -> Tuple[str, ...]:
    """Return the masking-rate variants shipped with the package."""
    return tuple(sorted(_CKPT_FOR_MASK_RATE.keys()))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pretrained(masking_rate: float = 0.5,
                    checkpoint_path: Optional[str] = None,
                    device: str = "cpu") -> BioPM:
    """Load a pretrained BioPM model.

    Args:
        masking_rate    : 0.25, 0.50 (default), or 0.75 — picks one of the
                          three shipped checkpoints.
        checkpoint_path : explicit checkpoint path (overrides masking_rate).
        device          : "cpu" or e.g. "cuda:0".

    Returns:
        a ``BioPM`` model in eval mode on the given device.
    """
    if checkpoint_path is None:
        if masking_rate not in _CKPT_FOR_MASK_RATE:
            raise ValueError(
                f"masking_rate must be one of {sorted(_CKPT_FOR_MASK_RATE.keys())}, "
                f"got {masking_rate}"
            )
        checkpoint_path = os.path.join(
            _DEFAULT_CKPT_DIR, _CKPT_FOR_MASK_RATE[masking_rate])
        checkpoint_path = os.path.normpath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"BioPM checkpoint not found at {checkpoint_path}.  Make sure "
            "the file shipped with the repo is present.")

    model = BioPM().to(device, dtype=torch.float)
    model.load_checkpoint(checkpoint_path, map_location=device)
    return model.eval()


# ---------------------------------------------------------------------------
# Single-batch encode
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_window(model: BioPM,
                  patches: torch.Tensor,
                  pos_info: torch.Tensor,
                  additional_embedding: torch.Tensor,
                  gravity: torch.Tensor,
                  *,
                  per_axis: bool = True,
                  gravity_len_per_ch: Optional[int] = None,
                  device: Optional[str] = None) -> torch.Tensor:
    """Encode a single batch of windows into BioPM features.

    All inputs should be torch tensors with shapes:

      patches              : (B, L, 32) — NaN-padded
      pos_info             : (B, L)
      additional_embedding : (B, L, K>=2)  -- channel 0 = axis id, 1 = duration
      gravity              : (B, T, 3) -- the low-pass / raw gravity window

    Returns a ``(B, F)`` float tensor on CPU.
    """
    if device is None:
        device = next(model.parameters()).device
    patches = patches.to(device, dtype=torch.float)
    pos_info = pos_info.to(device, dtype=torch.float)
    additional_embedding = additional_embedding.to(device, dtype=torch.float)
    gravity = gravity.to(device, dtype=torch.float)

    B, L, _ = patches.shape
    mask_info = torch.zeros(B, L, device=device)
    tokens = model.encoder_acc(patches, pos_info, mask_info,
                               additional_embedding)
    feats = fuse_window_feature(
        tokens=tokens,
        additional_embedding=additional_embedding,
        raw_patches=patches,
        gravity=gravity,
        per_axis=per_axis,
        gravity_len_per_ch=gravity_len_per_ch,
    )
    return feats.detach().cpu()


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------
def extract_features(
    data_root: str,
    *,
    masking_rate: float = 0.5,
    checkpoint_path: Optional[str] = None,
    per_axis: bool = True,
    gravity_len_per_ch: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cpu",
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end feature extraction from preprocessed HDF5 files.

    Args:
        data_root          : directory containing ``Data_MeLabel_*.h5``.
        masking_rate       : which shipped checkpoint to use (0.25 / 0.50 / 0.75).
        checkpoint_path    : explicit checkpoint override.
        per_axis           : True for per-axis mean+std (default; 1023-d
                             total).  False for joint mean+std (1028-d).
        gravity_len_per_ch : override the per-channel gravity interpolation
                             length (default keeps total dim ~ 1024).
        batch_size         : DataLoader batch size.
        num_workers        : DataLoader workers.
        device             : "cpu" or e.g. "cuda:0".
        show_progress      : whether to print a tqdm bar.

    Returns:
        features    : (N, F) np.float32
        labels      : (N,)   np.int64
        subject_ids : (N,)   same dtype as your subject ids.
    """
    me_patches, pos_info, add_emb, labels, pids, gravity, raw_acc = \
        load_preprocessed_h5(data_root)

    if gravity is None:
        raise ValueError(
            "Preprocessed HDF5 files did not contain a gravity stream "
            "(neither 'x_gravity' nor 'gravity_window_40hz').  Re-run "
            "preprocessing with the BioPM preprocess script.")

    dataset = MovementElementDataset(
        X=me_patches, X_grav=gravity, y=labels,
        pos_info=pos_info, additional_embedding=add_emb, pid=pids,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    model = load_pretrained(masking_rate=masking_rate,
                            checkpoint_path=checkpoint_path,
                            device=device)

    feat_dim = expected_feature_dim(per_axis=per_axis,
                                    gravity_len_per_ch=gravity_len_per_ch)
    print(f"BioPM (mr={masking_rate}, per_axis={per_axis}): feature dim = {feat_dim}")

    feats_list, labels_list, pids_list = [], [], []
    iterator = loader if not show_progress else tqdm(loader, desc="Extract")
    with torch.no_grad():
        for batch in iterator:
            patches, lbl, pid, grav, pos, addemb = batch
            feats = encode_window(
                model, patches, pos, addemb, grav,
                per_axis=per_axis,
                gravity_len_per_ch=gravity_len_per_ch,
                device=device,
            )
            feats_list.append(feats.numpy())
            labels_list.append(
                lbl.numpy() if torch.is_tensor(lbl) else np.asarray(lbl))
            pids_list.append(
                pid.numpy() if torch.is_tensor(pid) else np.asarray(pid))

    features = np.concatenate(feats_list, axis=0).astype(np.float32)
    labels_arr = np.concatenate(labels_list, axis=0).astype(np.int64)
    pids_arr = np.concatenate(pids_list, axis=0)
    print(f"Extracted {features.shape[0]} windows, feature dim {features.shape[1]}")
    return features, labels_arr, pids_arr
