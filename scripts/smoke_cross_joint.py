#!/usr/bin/env python3
"""Smoke test for cross-joint BioPM (configs B and E).

Pass criteria are asserts (not loss):
  1) adapters-bypassed wrist encode == stock BioPM (bit-identical)
  2) zero gradients on frozen body
  3) partner-stream attention mass >> 0 on cross-joint steps (config E)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopm.adapters import JOINTS
from biopm.cross_joint_data import CrossJointDataset, load_eval_subjects
from biopm.cross_joint_model import (
    CrossJointBioPM,
    CrossJointConfig,
    assert_body_frozen,
    assert_zero_body_grads,
)
from biopm.model import TimeSeriesTransformer
from biopm.pretraining import masked_recon_loss


def _stack_stream(s, device):
    return (
        s["patches"].unsqueeze(0).to(device),
        s["pos"].unsqueeze(0).to(device),
        s["mask"].unsqueeze(0).to(device),
        s["add"].unsqueeze(0).to(device),
        s["joint"],
    )


def assert_bit_identical(paper_ckpt: str, device: torch.device) -> None:
    stock = TimeSeriesTransformer().to(device).eval()
    sd = torch.load(paper_ckpt, map_location=device)
    if isinstance(sd, dict) and not any(k.startswith("conv_encode") for k in sd):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    enc_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder_acc."):
            enc_sd[k[len("encoder_acc.") :]] = v
        elif k.startswith("decoder_cnn.") or k.startswith("module."):
            continue
        else:
            enc_sd[k] = v
    stock.load_state_dict(enc_sd, strict=False)

    cfg = CrossJointConfig.from_name("e", bottleneck=64)
    model = CrossJointBioPM(cfg).to(device)
    model.load_paper_checkpoint(paper_ckpt, map_location=device)
    model.eval()

    B, L, P = 2, 192, 32
    patches = torch.randn(B, L, P, device=device)
    # put some nan padding
    patches[:, 50:, :] = float("nan")
    pos = torch.rand(B, L, device=device)
    pos[:, 50:] = float("nan")
    mask = torch.zeros(B, L, device=device)
    mask[:, :10] = 1.0
    add = torch.zeros(B, L, 5, device=device)
    add[:, :, 0] = torch.randint(0, 3, (B, L))
    add[:, :, 1] = torch.rand(B, L) * 20
    add[:, 50:, :] = float("nan")

    with torch.no_grad():
        y_stock = stock(patches, pos, mask, add)
        y_bypass, _, _ = model.encode_stream(
            patches, pos, mask, add, joint="wrist", bypass_adapters=True
        )
    if not torch.equal(y_stock, y_bypass):
        max_diff = (y_stock - y_bypass).abs().max().item()
        raise AssertionError(
            f"Bypass != stock BioPM (max_diff={max_diff}). Not bit-identical."
        )
    print("[assert] bypass wrist == stock BioPM: PASS (bit-identical)")


def run_smoke(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    assert_bit_identical(args.paper_ckpt, device)

    eval_subjects = sorted(load_eval_subjects(args.eval_subjects_json))
    all_subj = list(range(args.subject_min, args.subject_max + 1))
    train_subjects = [s for s in all_subj if s not in eval_subjects]
    train_subjects = train_subjects[: args.n_subjects]
    print(f"train_subjects={train_subjects} eval_held_out={eval_subjects}")

    ds = CrossJointDataset(
        args.data_root,
        train_subjects,
        paired_frac=args.paired_frac,
        samples_per_epoch=args.steps * 2,
        eval_subjects=eval_subjects,
    )

    for cfg_name in args.configs:
        print(f"\n===== smoke config {cfg_name} =====")
        cfg = CrossJointConfig.from_name(cfg_name, bottleneck=args.bottleneck)
        cfg.paired_step_frac = args.paired_frac
        model = CrossJointBioPM(cfg).to(device)
        info = model.load_paper_checkpoint(args.paper_ckpt, map_location=device)
        shrink_msg = model.maybe_shrink_decoder(max_ratio=2.0)
        if shrink_msg:
            print("[decoder]", shrink_msg)
        assert_body_frozen(model)
        rep = model.param_report()
        print("[params]", rep)
        assert rep["decoder_trainable"] > 0
        if cfg_name != "a":
            assert rep["adapters_trainable"] > 0 or cfg.use_joint_embedding

        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        partner_masses = []
        token_logs = []

        model.train()
        for step in range(args.steps):
            item = ds[step]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)

            if item["mode"] == "single":
                patches, pos, mask, add, joint = _stack_stream(item["a"], device)
                token_logs.append((joint, int(item["a"]["n_valid"])))
                pred, _ = model.within_joint_forward(patches, pos, mask, add, joint)
                l, _, _ = masked_recon_loss(
                    pred, patches, mask, cfg.weight_for_masked
                )
                loss = loss + l
            else:
                # within-joint both sides
                for key in ("a_within", "b_within"):
                    patches, pos, mask, add, joint = _stack_stream(item[key], device)
                    token_logs.append((joint, int(item[key]["n_valid"])))
                    pred, _ = model.within_joint_forward(
                        patches, pos, mask, add, joint
                    )
                    l, _, _ = masked_recon_loss(
                        pred, patches, mask, cfg.weight_for_masked
                    )
                    loss = loss + l

                if cfg.use_cross_joint_objective:
                    pa, posa, maska, adda, ja = _stack_stream(item["a"], device)
                    pb, posb, maskb, addb, jb = _stack_stream(item["b"], device)
                    h, valid, La, attns = model.encode_paired(
                        pa, posa, maska, adda, ja,
                        pb, posb, maskb, addb, jb,
                        return_attn=True,
                    )
                    h_dec = model.for_cross_joint_decode(h)
                    pred = model.decode(h_dec)
                    # targets / masks concatenated
                    tgt = torch.cat([pa, pb], dim=1)
                    msk = torch.cat([maska, maskb], dim=1)
                    l, _, _ = masked_recon_loss(
                        pred, tgt, msk, cfg.weight_for_masked
                    )
                    loss = loss + l
                    mass = CrossJointBioPM.partner_attn_mass(attns, La, valid)
                    partner_masses.append(mass)
                    if step % 10 == 0:
                        print(
                            f"  step {step}: loss={float(loss):.4f} "
                            f"partner_attn_mass={mass:.4f} "
                            f"tokens=({ja}:{int(item['a']['n_valid'])},"
                            f"{jb}:{int(item['b']['n_valid'])})"
                        )

            loss.backward()
            assert_zero_body_grads(model)
            opt.step()

        print("[assert] zero body grads throughout: PASS")
        if cfg.use_cross_joint_objective:
            if not partner_masses:
                raise AssertionError("No cross-joint steps logged attention mass")
            mean_mass = float(np.mean(partner_masses))
            print(f"[metric] mean partner attn mass={mean_mass:.4f}")
            if mean_mass < 0.02:
                raise AssertionError(
                    f"Partner attn mass ~0 ({mean_mass:.4g}). "
                    "Cross-joint objective is not driving cross attention — "
                    "fix masking before a full sweep."
                )
            print("[assert] partner attn mass materially > 0: PASS")
        else:
            print("[skip] partner attn mass (no cross-joint objective)")

        # token count log summary
        from collections import Counter
        c = Counter(j for j, _ in token_logs)
        print("[tokens/joint steps]", dict(c))

    print("\nSMOKE_CROSS_JOINT_OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        default="/scratch4/workspace/ptarale_umass_edu-light2/"
        "figshare_imu_layouts/preprocessed",
    )
    p.add_argument(
        "--paper_ckpt",
        default="/scratch4/workspace/ptarale_umass_edu-light2/"
        "resnet_paper_github/biopm/checkpoints/biopm_50mr.pt",
    )
    p.add_argument("--eval_subjects_json", default=None)
    p.add_argument("--subject_min", type=int, default=1)
    p.add_argument("--subject_max", type=int, default=30)
    p.add_argument("--n_subjects", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--configs", nargs="+", default=["b", "e"])
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--paired_frac", type=float, default=0.60)
    args = p.parse_args()

    # default held-out subjects for figshare transfer eval
    if args.eval_subjects_json is None:
        default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "figshare_eval_subjects.json",
        )
        if os.path.isfile(default):
            args.eval_subjects_json = default
        else:
            # inline default
            os.makedirs(os.path.dirname(default), exist_ok=True)
            with open(default, "w") as f:
                json.dump({"figshare": [28, 29, 30]}, f)
            args.eval_subjects_json = default
            print(f"wrote default eval subjects -> {default}")

    run_smoke(args)


if __name__ == "__main__":
    main()
