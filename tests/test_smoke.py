"""Smoke tests for the BioPM release package.

Run from the repo root with::

    pytest -q biopm/tests/test_smoke.py

These tests do not exercise correctness against the original paper
checkpoints, but they do verify that:

  * the package imports cleanly,
  * the three shipped checkpoints load without errors,
  * a forward pass on synthetic data produces the expected shapes,
  * per-axis and total pooling produce the documented dimensions,
  * the nested-CV evaluator returns a well-formed dict.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Add the parent directory to sys.path so `import biopm` works without
# `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import biopm  # noqa: E402
from biopm.features import (
    expected_feature_dim, per_axis_mean_std, total_mean_std,
)


def _fake_batch(B: int = 4, L: int = 192, T_grav: int = 300):
    patches = torch.randn(B, L, 32)
    patches[:, 50:, :] = float("nan")
    pos_info = torch.linspace(0, 1, L).expand(B, L).clone()
    add_emb = torch.zeros(B, L, 2)
    add_emb[:, :, 0] = torch.randint(0, 3, (B, L)).float()
    add_emb[:, 50:, :] = float("nan")
    gravity = torch.randn(B, T_grav, 3)
    return patches, pos_info, add_emb, gravity


def test_imports_top_level():
    assert hasattr(biopm, "load_pretrained")
    assert hasattr(biopm, "extract_features")
    assert hasattr(biopm, "encode_window")
    assert hasattr(biopm, "logreg_nested_cv")
    assert hasattr(biopm, "MovementElementDataset")
    assert hasattr(biopm, "fuse_window_feature")


def test_expected_feature_dim_defaults():
    assert expected_feature_dim(per_axis=True) == 1023
    assert expected_feature_dim(per_axis=False) == 1028


def test_per_axis_vs_total_shapes_synthetic():
    B, L, D = 3, 8, 64
    tokens = torch.randn(B, L, D)
    axis_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]] * B).float()
    valid = torch.ones(B, L, dtype=torch.bool)
    total = total_mean_std(tokens, valid)
    per_ax = per_axis_mean_std(tokens, axis_ids, valid, num_axes=3)
    assert total.shape == (B, 2 * D)
    assert per_ax.shape == (B, 2 * 3 * D)


def test_available_checkpoints():
    assert biopm.list_available_checkpoints() == (0.25, 0.50, 0.75)


def test_load_each_checkpoint_and_forward():
    patches, pos, add_emb, gravity = _fake_batch()
    for mr in biopm.list_available_checkpoints():
        model = biopm.load_pretrained(masking_rate=mr, device="cpu")

        feats_perax = biopm.encode_window(
            model, patches, pos, add_emb, gravity, per_axis=True,
        )
        assert feats_perax.shape == (patches.shape[0], 1023), \
            f"MR {mr}: expected 1023, got {feats_perax.shape}"
        assert torch.isfinite(feats_perax).all()

        feats_total = biopm.encode_window(
            model, patches, pos, add_emb, gravity, per_axis=False,
        )
        assert feats_total.shape == (patches.shape[0], 1028)
        assert torch.isfinite(feats_total).all()


def test_pretrain_forward_and_loss():
    """PretrainModel runs end-to-end and the loss is finite + differentiable."""
    from biopm.pretraining import PretrainModel, masked_recon_loss
    B, L = 2, 16
    patches = torch.randn(B, L, 32)
    patches[:, 8:, :] = float("nan")  # half the positions are padding
    pos_info = torch.linspace(0, 1, L).expand(B, L).clone()
    add_emb = torch.zeros(B, L, 2)
    add_emb[:, :, 0] = torch.randint(0, 3, (B, L)).float()
    add_emb[:, 8:, :] = float("nan")
    valid = ~torch.isnan(patches).any(dim=-1)
    mask_info = (torch.rand(B, L) < 0.5).float() * valid.float()

    m = PretrainModel().train()
    pred = m(patches, pos_info, mask_info, add_emb)
    assert pred.shape == (B, L, 32)
    loss, ml1, ul1 = masked_recon_loss(pred, patches, mask_info)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_logreg_nested_cv_runs():
    rng = np.random.default_rng(42)
    N, D, K, S = 200, 32, 4, 6
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.integers(0, K, N)
    pids = rng.integers(0, S, N)
    # add weak signal so the classifier learns something
    for c in range(K):
        X[y == c, c] += 1.5

    res = biopm.logreg_nested_cv(X, y, pids, verbose=False)
    assert res["cv_strategy"] == "LOSO"
    assert res["n_folds"] == S
    assert 0.0 < res["macro_f1_mean"] <= 1.0


if __name__ == "__main__":
    # Allow direct execution: `python biopm/tests/test_smoke.py`
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"=> {name}")
            fn()
    print("ALL OK")
