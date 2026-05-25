"""BioPM model definition.

BioPM is a two-stream encoder for 3-axis accelerometer data:

  1. ``TimeSeriesTransformer`` ("acc encoder")
       Encodes a sequence of normalised movement-element (ME) patches
       extracted by zero-crossing segmentation.  Produces one 64-d
       contextual token per ME.

  2. ``GravityCNNEncoder`` ("gravity encoder")
       A small 1-D CNN over the low-pass (gravity) component of the
       window.  Produces a single 64-d embedding per window.

For most downstream tasks you do not actually call the gravity encoder
during feature extraction: ``biopm.features.extract_window_feature``
concatenates the pooled acc tokens with the raw (interpolated) gravity
window, matching the protocol used in the paper.

Only what is needed for feature extraction lives in this file.  The
classifier head used during pretraining is intentionally omitted.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


D_MODEL: int = 64           # token width throughout the transformer
PATCH_LEN: int = 32         # length of one normalised ME patch
N_HEADS: int = 4            # multi-head attention
N_LAYERS: int = 5           # encoder depth
MAX_REL_POS: int = 15       # clamp for relative position bias
DROPOUT: float = 0.02       # inference dropout (matches the pretrained ckpts)


# ---------------------------------------------------------------------------
# Polyfills so old PyTorch builds still work
# ---------------------------------------------------------------------------
def _nanmedian(x: Tensor, dim: int):
    """Median ignoring NaNs.  Used for the time-relative-position bias."""
    mask = ~torch.isnan(x)
    filled = torch.where(mask, x, torch.full_like(x, float("inf")))
    sorted_vals, sorted_idx = torch.sort(filled, dim=dim, descending=False)
    counts = mask.sum(dim=dim)
    k = torch.clamp(counts - 1, min=0) // 2
    index_shape = list(sorted_vals.shape)
    index_shape[dim] = 1
    k_exp = k.reshape(index_shape).long()
    vals = torch.gather(sorted_vals, dim, k_exp).squeeze(dim)
    idxs = torch.gather(sorted_idx, dim, k_exp).squeeze(dim)
    zero_mask = counts == 0
    if zero_mask.any():
        vals = vals.clone()
        vals[zero_mask] = float("nan")
        idxs = idxs.clone()
        idxs[zero_mask] = 0
    return vals, idxs


# ---------------------------------------------------------------------------
# Relative-position multi-head attention
# ---------------------------------------------------------------------------
class RelPosMultiheadAttention(nn.Module):
    """Multi-head attention with two relative-position biases.

    The bias table ``rel_bias`` of shape ``(num_heads, 2 * max_rel_pos + 1)``
    is shared between two paths:

      * an index-based path that uses the sequence index difference
        ``i - j`` (constant across the batch)
      * a time-based path that uses the difference of patch indices
        ``patch_indices[b, i] - patch_indices[b, j]`` in units of the
        per-window median delta (variable across the batch)
    """

    def __init__(self, embed_dim: int = D_MODEL, num_heads: int = N_HEADS,
                 max_rel_pos: int = MAX_REL_POS, dropout: float = DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_rel_pos = max_rel_pos

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rel_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_rel_pos + 1))
        nn.init.trunc_normal_(self.rel_bias, std=0.1)

        self.dropout = nn.Dropout(dropout)
        self.use_seq_rel_pos = True

    def forward(self,
                query: Tensor, key: Tensor, value: Tensor,
                patch_indices: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        B, L, D = query.shape
        q, k, v = self.in_proj(query).split(D, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * self.scaling
        attn = torch.einsum("bhld,bhmd->bhlm", q, k)

        if self.use_seq_rel_pos:
            idx = torch.arange(L, device=query.device)
            rel = (idx[None, :] - idx[:, None]).clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            attn = attn + self.rel_bias[:, rel]

        deltas = patch_indices[:, 1:] - patch_indices[:, :-1]
        dt, _ = _nanmedian(deltas, dim=1)
        dt = torch.where(torch.isfinite(dt) & (dt > 0), dt, torch.tensor(1.0, device=dt.device))
        dt = dt.view(-1, 1, 1)
        rel_time = patch_indices.unsqueeze(2) - patch_indices.unsqueeze(1)
        invalid = torch.isnan(rel_time) | (rel_time.abs() > self.max_rel_pos)
        rel_time = torch.where(invalid, torch.zeros_like(rel_time), rel_time)
        rel_steps = (rel_time / dt).round().clamp(-self.max_rel_pos, self.max_rel_pos).long() + self.max_rel_pos
        M = self.max_rel_pos
        bias_table = self.rel_bias.unsqueeze(0).unsqueeze(2).expand(B, self.num_heads, L, 2 * M + 1)
        bias = torch.gather(bias_table, dim=3, index=rel_steps.unsqueeze(1).expand(B, self.num_heads, L, L))
        attn = attn + bias

        if is_causal:
            causal = torch.triu(torch.ones(L, L, device=query.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal[None, None], float("-inf"))

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :].bool(), -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.einsum("bhlm,bhmd->bhld", attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------
class RelPosTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, nhead: int = N_HEADS,
                 dim_feedforward: int = 2048, dropout: float = DROPOUT,
                 max_rel_pos: int = MAX_REL_POS):
        super().__init__()
        self.self_attn = RelPosMultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            max_rel_pos=max_rel_pos, dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, patch_indices: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:
        sa = self.self_attn(src, src, src, patch_indices,
                            key_padding_mask=src_key_padding_mask,
                            is_causal=is_causal)
        x = self.norm1(src + self.drop1(sa))
        ff = self.linear2(self.drop2(F.gelu(self.linear1(x))))
        return self.norm2(x + ff)


# ---------------------------------------------------------------------------
# Patch conv encoder: (B, L, 32) -> (B, L, 60)
# ---------------------------------------------------------------------------
class ConvEncode(nn.Module):
    """1-D CNN that turns one ME patch of 32 samples into a 60-d vector."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, L, P = x.shape
        return self.conv(x.reshape(-1, 1, P)).reshape(B, L, -1)


# ---------------------------------------------------------------------------
# Acc encoder: movement-element transformer
# ---------------------------------------------------------------------------
class TimeSeriesTransformer(nn.Module):
    """Acc-side encoder of BioPM.

    Input
        patches            : (B, L, 32) normalised ME patches (NaN = padding)
        patch_indices      : (B, L) fractional position of each ME in [0, 1]
        mask_info          : (B, L) 1.0 where the patch should be replaced
                              with the learnable mask token, 0.0 otherwise.
                              Pass zeros at inference time.
        additional_embedding : (B, L, >=2) where channel 0 is the integer
                              axis id (0=X, 1=Y, 2=Z) and channel 1 is the
                              ME duration.

    Output
        (B, L, 64) contextual tokens.  Padding rows are still produced and
        the consumer is expected to ignore them using the same NaN-mask
        that came in on ``patches``.
    """

    def __init__(self):
        super().__init__()
        D = D_MODEL
        self.conv_encode = ConvEncode()
        # 4 axis ids: 0/1/2 plus the "padding" sentinel that NaN values map to
        self.axis_embedding = nn.Embedding(num_embeddings=4, embedding_dim=3)
        self.mask_token = nn.Parameter(torch.randn(D))
        nn.init.trunc_normal_(self.mask_token, std=0.2)

        self.pos_emb_net = nn.Sequential(
            nn.Linear(1, D), nn.GELU(), nn.Linear(D, D),
        )
        self.post_pos_ln = nn.LayerNorm(D)
        self.transformer_encoder_points_within_segment = nn.ModuleList([
            RelPosTransformerEncoderLayer(d_model=D, nhead=N_HEADS,
                                          max_rel_pos=MAX_REL_POS,
                                          dropout=DROPOUT)
            for _ in range(N_LAYERS)
        ])

    def forward(self, patches: Tensor, patch_indices: Tensor,
                mask_info: Tensor, additional_embedding: Tensor,
                is_causal: bool = False) -> Tensor:
        padding_mask = ~torch.isnan(patches).any(dim=-1)
        patches = torch.nan_to_num(patches, nan=25.0)
        additional_embedding = torch.nan_to_num(additional_embedding, nan=3.0)[:, :, :2]

        h = self.conv_encode(patches)                                          # (B, L, 60)
        axis_emb = self.axis_embedding(additional_embedding[:, :, 0].long())   # (B, L, 3)
        duration = additional_embedding[:, :, 1].unsqueeze(-1)                 # (B, L, 1)
        h = torch.cat([h, axis_emb, duration], dim=-1)                         # (B, L, 64)

        mask_exp = mask_info.unsqueeze(-1).expand_as(h).bool()
        h = torch.where(mask_exp, self.mask_token, h)

        patch_indices = torch.nan_to_num(patch_indices, nan=0.0)
        h = self.post_pos_ln(h + self.pos_emb_net(patch_indices.unsqueeze(-1)))

        key_padding = ~padding_mask
        for layer in self.transformer_encoder_points_within_segment:
            h = layer(h, patch_indices,
                      src_key_padding_mask=key_padding,
                      is_causal=is_causal)
        return h


# ---------------------------------------------------------------------------
# Gravity encoder (optional)
# ---------------------------------------------------------------------------
class _AvgMaxPool1d(nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(K)
        self.max = nn.AdaptiveMaxPool1d(K)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.avg(x), self.max(x)], dim=1)


class GravityCNNEncoder(nn.Module):
    """1-D CNN that maps a gravity window (B, T, 3) to (B, 64).

    Not required for the default ``extract_features`` API (which uses the
    raw gravity stream directly).  Provided for users who want a learnable
    gravity embedding instead.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64,
                 dropout_p: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2,
                      bias=False, padding_mode="circular"),
            nn.GroupNorm(4, 16), nn.GELU(), nn.Dropout2d(p=DROPOUT),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1,
                      bias=False, padding_mode="circular"),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Dropout2d(p=DROPOUT),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1,
                      bias=False, padding_mode="circular"),
            nn.GroupNorm(16, 64), nn.GELU(),
        )
        K = 12
        self.pool = _AvgMaxPool1d(K)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * K, embed_dim), nn.GELU(),
        )
        self.out_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        # (B, T, 3) -> (B, 3, T)
        x = x.transpose(1, 2)
        return self.head(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# BioPM container
# ---------------------------------------------------------------------------
class BioPM(nn.Module):
    """Wraps the acc and (optional) gravity encoders.

    The pretrained checkpoints shipped with this repo only contain weights
    for ``encoder_acc``; calling ``load_checkpoint`` will load those and
    leave ``encoder_gravity`` randomly initialised.  That is intentional:
    in the paper the gravity stream is the raw window flattened, not a
    learned CNN.
    """

    def __init__(self):
        super().__init__()
        self.encoder_acc = TimeSeriesTransformer()
        self.encoder_gravity = GravityCNNEncoder()

    def forward(self, patches: Tensor, patch_indices: Tensor,
                mask_info: Tensor, additional_embedding: Tensor,
                is_causal: bool = False) -> Tensor:
        return self.encoder_acc(patches, patch_indices, mask_info,
                                additional_embedding, is_causal=is_causal)

    @torch.no_grad()
    def load_checkpoint(self, ckpt_path: str, strict: bool = False,
                        map_location: str = "cpu", verbose: bool = False) -> None:
        """Load pretrained acc-encoder weights from a state dict file.

        Pretraining checkpoints contain decoder/classifier weights that
        are not needed at inference; those are silently dropped.  Set
        ``verbose=True`` to see the full diff against the encoder
        state dict.
        """
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(sd, dict):
            for k in ("state_dict", "model", "model_state_dict"):
                if k in sd:
                    sd = sd[k]
                    break
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # Strip pretraining-only heads that are not in this encoder
        encoder_keys = set(self.encoder_acc.state_dict().keys())
        filtered = {k: v for k, v in sd.items() if k in encoder_keys}
        dropped = sorted(set(sd) - encoder_keys)
        missing, unexpected = self.encoder_acc.load_state_dict(
            filtered, strict=strict)
        if verbose:
            if missing:
                print(f"[BioPM] missing keys ({len(missing)}): {list(missing)[:6]}...")
            if unexpected:
                print(f"[BioPM] unexpected keys ({len(unexpected)}): {list(unexpected)[:6]}...")
            if dropped:
                print(f"[BioPM] dropped pretraining-only keys ({len(dropped)}): {dropped[:6]}...")
        else:
            if missing:
                print(f"[BioPM] WARNING: {len(missing)} keys missing from "
                      f"checkpoint; first few: {list(missing)[:4]}")
