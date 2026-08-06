#!/usr/bin/env python3
"""Full cross-joint BioPM pretraining (configs A–E, d-sweep, D-wide)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopm.cross_joint_data import (
    CrossJointDataset,
    assert_subject_disjoint,
    collate_cross_joint,
    load_eval_subjects,
)
from biopm.cross_joint_model import (
    CrossJointBioPM,
    CrossJointConfig,
    assert_body_frozen,
    assert_zero_body_grads,
)
from biopm.pretraining import EarlyStopping, masked_recon_loss


def _stack(s, device):
    return (
        s["patches"].unsqueeze(0).to(device),
        s["pos"].unsqueeze(0).to(device),
        s["mask"].unsqueeze(0).to(device),
        s["add"].unsqueeze(0).to(device),
        s["joint"],
    )


def run_item(model, item, cfg, device, log_attn: bool):
    loss = torch.zeros((), device=device)
    metrics = {}
    if item["mode"] == "single":
        patches, pos, mask, add, joint = _stack(item["a"], device)
        pred, _ = model.within_joint_forward(patches, pos, mask, add, joint)
        l, ml, ul = masked_recon_loss(pred, patches, mask, cfg.weight_for_masked)
        loss = loss + l
        metrics["tokens"] = {joint: int(item["a"]["n_valid"])}
        metrics["within_masked_l1"] = ml
        return loss, metrics

    for key in ("a_within", "b_within"):
        patches, pos, mask, add, joint = _stack(item[key], device)
        pred, _ = model.within_joint_forward(patches, pos, mask, add, joint)
        l, ml, ul = masked_recon_loss(pred, patches, mask, cfg.weight_for_masked)
        loss = loss + 0.5 * l

    metrics["tokens"] = {
        item["a"]["joint"]: int(item["a"]["n_valid"]),
        item["b"]["joint"]: int(item["b"]["n_valid"]),
    }

    if cfg.use_cross_joint_objective:
        pa, posa, maska, adda, ja = _stack(item["a"], device)
        pb, posb, maskb, addb, jb = _stack(item["b"], device)
        h, valid, La, attns = model.encode_paired(
            pa, posa, maska, adda, ja,
            pb, posb, maskb, addb, jb,
            return_attn=log_attn,
        )
        pred = model.decode(model.for_cross_joint_decode(h))
        tgt = torch.cat([pa, pb], dim=1)
        msk = torch.cat([maska, maskb], dim=1)
        l, ml, ul = masked_recon_loss(pred, tgt, msk, cfg.weight_for_masked)
        loss = loss + l
        metrics["cross_masked_l1"] = ml
        if log_attn and attns is not None:
            metrics["partner_attn_mass"] = CrossJointBioPM.partner_attn_mass(
                attns, La, valid
            )
    return loss, metrics


def train_one(args, cfg_name: str, bottleneck: int) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CrossJointConfig.from_name(cfg_name, bottleneck=bottleneck)
    cfg.paired_step_frac = args.paired_frac

    eval_subjects = sorted(load_eval_subjects(args.eval_subjects_json))
    all_subj = list(range(args.subject_min, args.subject_max + 1))
    train_subjects = [s for s in all_subj if s not in set(eval_subjects)]
    assert_subject_disjoint(train_subjects, eval_subjects)

    out_dir = os.path.join(
        args.output_root, f"cfg_{cfg_name}_d{bottleneck}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Config A: baseline, no training
    if cfg_name == "a":
        model = CrossJointBioPM(cfg).to(device)
        model.load_paper_checkpoint(args.paper_ckpt, map_location=device)
        path = os.path.join(out_dir, "checkpoint.pt")
        torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, path)
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump({"note": "baseline frozen paper body, no training",
                       "params": model.param_report()}, f, indent=2)
        print(f"[A] wrote {path}")
        return path

    ds = CrossJointDataset(
        args.data_root,
        train_subjects,
        paired_frac=args.paired_frac,
        samples_per_epoch=args.steps_per_epoch,
        eval_subjects=eval_subjects,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cross_joint,
    )

    model = CrossJointBioPM(cfg).to(device)
    model.load_paper_checkpoint(args.paper_ckpt, map_location=device)
    msg = model.maybe_shrink_decoder(max_ratio=2.0)
    if msg:
        print(msg)
    assert_body_frozen(model)
    rep = model.param_report()
    print(f"[{cfg_name} d={bottleneck}] params={rep}")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    early = EarlyStopping(
        patience=args.patience,
        path=os.path.join(out_dir, "checkpoint.pt"),
        verbose=True,
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        partner = []
        token_counts = defaultdict(int)
        t0 = time.time()
        for bi, batch in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for item in batch["items"]:
                log_attn = cfg.use_cross_joint_objective and (bi % 20 == 0)
                l, metrics = run_item(model, item, cfg, device, log_attn)
                batch_loss = batch_loss + l
                for j, n in metrics.get("tokens", {}).items():
                    token_counts[j] += n
                if "partner_attn_mass" in metrics:
                    partner.append(metrics["partner_attn_mass"])
            batch_loss = batch_loss / max(len(batch["items"]), 1)
            batch_loss.backward()
            assert_zero_body_grads(model)
            opt.step()
            losses.append(float(batch_loss))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_partner = float(np.mean(partner)) if partner else None
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "partner_attn_mass": mean_partner,
            "tokens": dict(token_counts),
            "sec": time.time() - t0,
        }
        history.append(row)
        print(
            f"[{cfg_name} d={bottleneck}] epoch {epoch}: loss={mean_loss:.4f} "
            f"partner_mass={mean_partner} tokens={dict(token_counts)} "
            f"({row['sec']:.1f}s)"
        )
        # EarlyStopping expects a model state dict path — save wrapper
        # Use mean_loss as monitor; store full payload ourselves on improve.
        score = -mean_loss
        if early.best_score is None or score > early.best_score + early.delta:
            early.best_score = score
            early.counter = 0
            torch.save(
                {
                    "config": cfg.__dict__,
                    "cfg_name": cfg_name,
                    "bottleneck": bottleneck,
                    "state_dict": model.state_dict(),
                    "param_report": rep,
                    "epoch": epoch,
                },
                early.path,
            )
            early.best_loss = mean_loss
        else:
            early.counter += 1
            if early.counter >= early.patience:
                print("early stop")
                break

        if cfg.use_cross_joint_objective and mean_partner is not None:
            if epoch >= 1 and mean_partner < 0.02:
                raise RuntimeError(
                    f"partner attn mass collapsed to {mean_partner:.4g}; "
                    "refusing to continue full run"
                )

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"params": rep, "eval_subjects": eval_subjects,
                   "train_subjects": train_subjects}, f, indent=2)
    return early.path


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
    p.add_argument(
        "--output_root",
        default="/scratch4/workspace/ptarale_umass_edu-light2/"
        "figshare_imu_layouts/runs/cross_joint",
    )
    p.add_argument(
        "--eval_subjects_json",
        default="/scratch4/workspace/ptarale_umass_edu-light2/"
        "resnet_paper_github/biopm/configs/figshare_eval_subjects.json",
    )
    p.add_argument("--configs", nargs="+", default=["a", "b", "c", "d", "e", "d-wide"])
    p.add_argument("--bottlenecks", nargs="+", type=int, default=[64])
    p.add_argument("--sweep_cd_bottlenecks", nargs="+", type=int, default=[16, 64, 256],
                   help="Extra d sweep for configs C and D")
    p.add_argument("--subject_min", type=int, default=1)
    p.add_argument("--subject_max", type=int, default=30)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--paired_frac", type=float, default=0.60)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    jobs = []
    for name in args.configs:
        if name in ("c", "d"):
            for d in args.sweep_cd_bottlenecks:
                jobs.append((name, d))
        else:
            for d in args.bottlenecks:
                jobs.append((name, d))
    # dedupe
    seen = set()
    uniq = []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            uniq.append(j)

    print("jobs:", uniq)
    for name, d in uniq:
        train_one(args, name, d)


if __name__ == "__main__":
    main()
