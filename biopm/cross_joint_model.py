"""Cross-joint BioPM: frozen body + per-joint adapters + shared decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .adapters import (
    JOINTS,
    NON_WRIST,
    build_joint_modules,
    count_trainable,
)
from .model import D_MODEL, N_LAYERS, PATCH_LEN, TimeSeriesTransformer
from .pretraining import Decode_cnn, masked_recon_loss


@dataclass
class CrossJointConfig:
    """Trainable-module ladder for configs A–E / D-wide."""

    bottleneck: int = 64
    shared_dim: int = 32  # private_dim = D_MODEL - shared_dim
    use_joint_embedding: bool = True
    use_input_adapter: bool = True
    use_layer_adapters: bool = True
    use_cross_joint_objective: bool = True
    paired_step_frac: float = 0.60
    input_bottleneck: Optional[int] = None  # D-wide override
    weight_for_masked: float = 100.0
    cross_span_sec: float = 5.0
    fs: float = 40.0
    ws: float = 10.0

    @staticmethod
    def from_name(name: str, bottleneck: int = 64) -> "CrossJointConfig":
        name = name.strip().lower()
        if name == "a":
            return CrossJointConfig(
                bottleneck=bottleneck,
                use_joint_embedding=False,
                use_input_adapter=False,
                use_layer_adapters=False,
                use_cross_joint_objective=False,
            )
        if name == "b":
            return CrossJointConfig(
                bottleneck=bottleneck,
                use_joint_embedding=True,
                use_input_adapter=False,
                use_layer_adapters=False,
                use_cross_joint_objective=False,
            )
        if name == "c":
            return CrossJointConfig(
                bottleneck=bottleneck,
                use_joint_embedding=True,
                use_input_adapter=True,
                use_layer_adapters=False,
                use_cross_joint_objective=False,
            )
        if name == "d":
            return CrossJointConfig(
                bottleneck=bottleneck,
                use_joint_embedding=True,
                use_input_adapter=True,
                use_layer_adapters=True,
                use_cross_joint_objective=False,
            )
        if name == "e":
            return CrossJointConfig(
                bottleneck=bottleneck,
                use_joint_embedding=True,
                use_input_adapter=True,
                use_layer_adapters=True,
                use_cross_joint_objective=True,
            )
        if name in ("d-wide", "dwide", "d_wide"):
            # Match D's trainable params by widening C's input adapter only.
            # D ≈ joint_emb + input(d) + n_layers*adapter(d)
            # C-wide ≈ joint_emb + input(d_wide)
            # => d_wide ≈ d * (1 + N_LAYERS) for linear-dominated adapters.
            d_wide = int(bottleneck * (1 + N_LAYERS))
            return CrossJointConfig(
                bottleneck=bottleneck,
                input_bottleneck=d_wide,
                use_joint_embedding=True,
                use_input_adapter=True,
                use_layer_adapters=False,
                use_cross_joint_objective=False,
            )
        raise ValueError(f"Unknown config name: {name}")


class CrossJointBioPM(nn.Module):
    """Frozen BioPM encoder body + per-joint adapters + one shared decoder."""

    def __init__(self, cfg: CrossJointConfig):
        super().__init__()
        self.cfg = cfg
        if not (0 < cfg.shared_dim < D_MODEL):
            raise ValueError(f"shared_dim must be in (0, {D_MODEL}), got {cfg.shared_dim}")
        self.shared_dim = int(cfg.shared_dim)
        self.private_dim = D_MODEL - self.shared_dim

        self.encoder = TimeSeriesTransformer()
        self.decoder = Decode_cnn(D_MODEL, PATCH_LEN)
        self.joints = build_joint_modules(
            joints=JOINTS,
            d_model=D_MODEL,
            bottleneck=cfg.bottleneck,
            n_layers=N_LAYERS,
            use_input_adapter=cfg.use_input_adapter,
            use_layer_adapters=cfg.use_layer_adapters,
            use_joint_embedding=cfg.use_joint_embedding,
            input_bottleneck=cfg.input_bottleneck,
        )
        self.freeze_body()

    # ------------------------------------------------------------------ freeze
    def freeze_body(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()  # BN/dropout frozen behaviour for body

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep encoder (frozen body) in eval so BN stats stay put.
        self.encoder.eval()
        return self

    def load_paper_checkpoint(self, path: str, map_location="cpu") -> None:
        sd = torch.load(path, map_location=map_location)
        if isinstance(sd, dict):
            for k in ("state_dict", "model", "model_state_dict"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
        enc_sd = {}
        dec_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder_acc."):
                enc_sd[k[len("encoder_acc.") :]] = v
            elif k.startswith("decoder_cnn."):
                dec_sd[k] = v
            elif k.startswith("module.encoder_acc."):
                enc_sd[k[len("module.encoder_acc.") :]] = v
            elif k.startswith("module.decoder_cnn."):
                dec_sd[k[len("module.") :]] = v
            elif k.startswith("decoder_cnn.") or k.startswith("decode_cnn."):
                dec_sd[k if k.startswith("decoder_cnn.") else "decoder_cnn." + k[len("decode_cnn.") :]] = v
            else:
                # flat encoder keys from shipped ckpt
                enc_sd[k] = v
        missing_e, unexp_e = self.encoder.load_state_dict(enc_sd, strict=False)
        # Decoder may only partially load if we slimmed it; tolerate missing.
        missing_d, unexp_d = self.decoder.load_state_dict(dec_sd, strict=False)
        self.freeze_body()
        return {
            "encoder_missing": list(missing_e),
            "encoder_unexpected": list(unexp_e),
            "decoder_missing": list(missing_d),
            "decoder_unexpected": list(unexp_d),
        }

    def param_report(self) -> Dict[str, int]:
        body = sum(p.numel() for p in self.encoder.parameters())
        dec = count_trainable(self.decoder)
        adapt = count_trainable(self.joints)
        return {
            "body_frozen": body,
            "decoder_trainable": dec,
            "adapters_trainable": adapt,
            "total_trainable": dec + adapt,
        }

    def maybe_shrink_decoder(self, max_ratio: float = 2.0) -> Optional[str]:
        """Log a warning if the shared decoder dwarfs adapters.

        We keep the stock Decode_cnn (must emit PATCH_LEN=32) so paper
        decoder weights load cleanly and recon shapes stay correct. Shrinking
        is intentionally *not* done automatically — configs like B have only
        joint embeddings (~256 params) so any decoder looks huge.
        """
        rep = self.param_report()
        if rep["adapters_trainable"] <= 0:
            return (
                f"decoder_trainable={rep['decoder_trainable']} "
                f"(no adapters; decoder is the only trainable head)"
            )
        ratio = rep["decoder_trainable"] / max(rep["adapters_trainable"], 1)
        msg = (
            f"decoder/adapters param ratio={ratio:.1f} "
            f"(decoder={rep['decoder_trainable']}, "
            f"adapters={rep['adapters_trainable']})"
        )
        if ratio > max_ratio:
            msg += (
                " — decoder dominates adapters; consider a hand-tuned slim "
                "decoder if ladder ablations look decoder-driven."
            )
        return msg

    # --------------------------------------------------------------- encode
    def encode_stream(
        self,
        patches: Tensor,
        patch_indices: Tensor,
        mask_info: Tensor,
        additional_embedding: Tensor,
        joint: str,
        bypass_adapters: bool = False,
        return_attn: bool = False,
    ):
        """Encode one joint stream with optional adapters.

        Returns tokens (B, L, 64) and optionally list of attn maps per layer.
        """
        if joint not in self.joints:
            raise KeyError(joint)
        jm = self.joints[joint]
        enc = self.encoder

        padding_mask = ~torch.isnan(patches).any(dim=-1)
        patches_n = torch.nan_to_num(patches, nan=25.0)
        add = torch.nan_to_num(additional_embedding, nan=3.0)[:, :, :2]

        h = enc.conv_encode(patches_n)
        axis_emb = enc.axis_embedding(add[:, :, 0].long())
        duration = add[:, :, 1].unsqueeze(-1)
        h = torch.cat([h, axis_emb, duration], dim=-1)  # (B, L, 64)

        # Adapters + joint emb AFTER fusion, BEFORE transformer (and before mask token)
        h = jm.apply_input(h, bypass=bypass_adapters)
        h = jm.apply_joint_emb(h, bypass=bypass_adapters)

        mask_exp = mask_info.unsqueeze(-1).expand_as(h).bool()
        h = torch.where(mask_exp, enc.mask_token, h)

        pos = torch.nan_to_num(patch_indices, nan=0.0)
        h = enc.post_pos_ln(h + enc.pos_emb_net(pos.unsqueeze(-1)))

        key_padding = ~padding_mask
        attns = []
        for i, layer in enumerate(enc.transformer_encoder_points_within_segment):
            if return_attn:
                h, attn_w = layer(
                    h, pos, src_key_padding_mask=key_padding, return_attn=True
                )
                attns.append(attn_w)
            else:
                h = layer(h, pos, src_key_padding_mask=key_padding)
            h = jm.apply_layer(h, i, bypass=bypass_adapters)
        if return_attn:
            return h, padding_mask, attns
        return h, padding_mask, None

    def encode_paired(
        self,
        patches_a: Tensor,
        pos_a: Tensor,
        mask_a: Tensor,
        add_a: Tensor,
        joint_a: str,
        patches_b: Tensor,
        pos_b: Tensor,
        mask_b: Tensor,
        add_b: Tensor,
        joint_b: str,
        bypass_adapters: bool = False,
        return_attn: bool = False,
    ):
        """Lexically embed each stream, concat tokens, run shared transformer."""
        # Build pre-transformer tokens per stream, then concat and run layers once.
        # Simpler approach matching "concatenate token streams": encode each
        # through conv/adapters separately, concat, then shared transformer layers
        # with per-token layer adapters selected by joint id.
        ha, pad_a = self._pre_transformer(
            patches_a, pos_a, mask_a, add_a, joint_a, bypass_adapters
        )
        hb, pad_b = self._pre_transformer(
            patches_b, pos_b, mask_b, add_b, joint_b, bypass_adapters
        )
        h = torch.cat([ha, hb], dim=1)
        pos = torch.cat([pos_a, pos_b], dim=1)
        pad = torch.cat([pad_a, pad_b], dim=1)
        joint_ids = torch.cat(
            [
                torch.full(pad_a.shape, JOINTS.index(joint_a), device=h.device, dtype=torch.long),
                torch.full(pad_b.shape, JOINTS.index(joint_b), device=h.device, dtype=torch.long),
            ],
            dim=1,
        )
        key_padding = ~pad
        attns = []
        enc = self.encoder
        La = ha.shape[1]
        for i, layer in enumerate(enc.transformer_encoder_points_within_segment):
            if return_attn:
                h, attn_w = layer(
                    h, pos, src_key_padding_mask=key_padding, return_attn=True
                )
                attns.append(attn_w)
            else:
                h = layer(h, pos, src_key_padding_mask=key_padding)
            if not bypass_adapters and self.cfg.use_layer_adapters:
                h = self._apply_layer_adapters_mixed(h, joint_ids, i)
        if return_attn:
            return h, pad, La, attns
        return h, pad, La, None

    def _pre_transformer(
        self, patches, patch_indices, mask_info, additional_embedding, joint, bypass
    ):
        jm = self.joints[joint]
        enc = self.encoder
        padding_mask = ~torch.isnan(patches).any(dim=-1)
        patches_n = torch.nan_to_num(patches, nan=25.0)
        add = torch.nan_to_num(additional_embedding, nan=3.0)[:, :, :2]
        h = enc.conv_encode(patches_n)
        axis_emb = enc.axis_embedding(add[:, :, 0].long())
        duration = add[:, :, 1].unsqueeze(-1)
        h = torch.cat([h, axis_emb, duration], dim=-1)
        h = jm.apply_input(h, bypass=bypass)
        h = jm.apply_joint_emb(h, bypass=bypass)
        mask_exp = mask_info.unsqueeze(-1).expand_as(h).bool()
        h = torch.where(mask_exp, enc.mask_token, h)
        pos = torch.nan_to_num(patch_indices, nan=0.0)
        h = enc.post_pos_ln(h + enc.pos_emb_net(pos.unsqueeze(-1)))
        return h, padding_mask

    def _apply_layer_adapters_mixed(self, h: Tensor, joint_ids: Tensor, layer_idx: int) -> Tensor:
        out = h
        for jname, jid in ((j, i) for i, j in enumerate(JOINTS)):
            jm = self.joints[jname]
            if jm.layer_adapters is None:
                continue
            sel = joint_ids == jid
            if not sel.any():
                continue
            # Apply adapter to full tensor then blend (simple, correct).
            adapted = jm.layer_adapters[layer_idx](h)
            out = torch.where(sel.unsqueeze(-1), adapted, out)
        return out

    # -------------------------------------------------------- shared/private
    def split_tokens(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        return h[..., : self.shared_dim], h[..., self.shared_dim :]

    def for_within_joint_decode(self, h: Tensor) -> Tensor:
        return h  # full [shared||private]

    def for_cross_joint_decode(self, h: Tensor) -> Tensor:
        shared, private = self.split_tokens(h)
        return torch.cat([shared, torch.zeros_like(private)], dim=-1)

    def pool_split(self, h: Tensor, valid: Tensor) -> Dict[str, Tensor]:
        """Mean-pool valid tokens; return full / shared-only / private-only."""
        # valid: (B, L)
        mask = valid.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask).sum(dim=1) / denom
        shared, private = self.split_tokens(pooled.unsqueeze(1))
        shared, private = shared.squeeze(1), private.squeeze(1)
        return {
            "full": pooled,
            "shared": shared,
            "private": private,
        }

    # -------------------------------------------------------------- decode
    def decode(self, h: Tensor) -> Tensor:
        return self.decoder(h)

    # --------------------------------------------------------------- losses
    def within_joint_forward(
        self,
        patches: Tensor,
        pos: Tensor,
        mask: Tensor,
        add: Tensor,
        joint: str,
        bypass_adapters: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        h, valid, _ = self.encode_stream(
            patches, pos, mask, add, joint, bypass_adapters=bypass_adapters
        )
        pred = self.decode(self.for_within_joint_decode(h))
        return pred, valid

    @staticmethod
    def partner_attn_mass(
        attns: List[Tensor],
        len_a: int,
        valid: Tensor,
    ) -> float:
        """Fraction of attention mass from side-A queries onto side-B keys.

        attns: list of (B, H, L, L) over layers. Uses mean over layers/heads.
        """
        if not attns:
            return 0.0
        # Average layers
        A = torch.stack(attns, dim=0).mean(dim=0)  # (B, H, L, L)
        A = A.mean(dim=1)  # (B, L, L)
        Bsz, L, _ = A.shape
        # Queries on A (0:len_a), keys on B (len_a:)
        q_valid = valid[:, :len_a]
        mass = []
        for b in range(Bsz):
            qv = q_valid[b]
            if not qv.any():
                continue
            # attention from A queries to B keys
            row = A[b, :len_a, len_a:]  # (La, Lb)
            row = row[qv]
            # normalize by row sum over all keys that are valid
            total = A[b, :len_a, :][qv].sum(dim=-1).clamp(min=1e-8)
            partner = row.sum(dim=-1) / total
            mass.append(partner.mean())
        if not mass:
            return 0.0
        return float(torch.stack(mass).mean().item())


class _SlimDecode(nn.Module):
    """Smaller shared decoder when stock Decode_cnn dwarfs adapters."""

    def __init__(self, embed_dim: int = D_MODEL, patch_len: int = PATCH_LEN):
        super().__init__()
        self.patch_len = patch_len
        self.decode_cnn = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 32, kernel_size=4, stride=2,
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


def assert_body_frozen(model: CrossJointBioPM) -> None:
    bad = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert not bad, f"Body params require_grad=True: {bad[:8]}"


def assert_zero_body_grads(model: CrossJointBioPM) -> None:
    bad = []
    for n, p in model.encoder.named_parameters():
        if p.grad is not None and float(p.grad.abs().sum()) > 0:
            bad.append(n)
    assert not bad, f"Non-zero grads on frozen body: {bad[:8]}"
