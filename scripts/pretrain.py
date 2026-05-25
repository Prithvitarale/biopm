"""Reference pretraining script for BioPM.

This is the *exact* training recipe used to produce the three pretrained
checkpoints in ``checkpoints/``.  Default ``--mask_rate`` is 0.50, which
corresponds to ``biopm_50mr.pt``.  Set it to ``0.25`` or ``0.75`` to
reproduce the other two variants.

Expected data layout
--------------------
Pretraining streams from many subjects, each of which can have tens of
thousands of windows, so the on-disk format keeps every window in its
own small file.  For each subject we expect a directory containing the
following files for every window ``idx``:

    <data_dir>/<subject_id>/
        merge_acc_filt_<subject_id>_<idx>.npy        # (pad_size, 32 + 1 + 5)
        window_acc_filt_<subject_id>_<idx>.npy       # (window_samples, 3) -- index file
        me_normalizeInfo_padding_acc_filt_<subject_id>_<idx>.pkl  # ME meta

The ``merge_*.npy`` array is the same dense ME representation produced
by ``biopm.preprocessing.pack_window`` (``[norm_me(32) | pos(1) | axis,
len, min, max, dirct]``).  ``window_*.npy`` is used only for file
discovery / subject counting.  ``me_normalizeInfo_*.pkl`` is a pandas
``DataFrame`` saved with ``to_pickle``; column 1 and column 2 must be
the per-ME start / end sample indices used by the time-based masking
strategy.

If you only have the per-subject HDF5 files produced by
``scripts/preprocess_mhealth.py``, you'll first need to split each
window out into the three-file layout above (one trivial script
forthcoming -- the layout exists because of UKBB-scale training, not
because we like .pkl files).

Usage
-----
Single GPU::

    python scripts/pretrain.py \\
        --data_dir /path/to/preprocessed_data_bins \\
        --output_dir runs/biopm_50mr \\
        --mask_rate 0.50

Multi-GPU (DDP, all visible GPUs)::

    python scripts/pretrain.py \\
        --data_dir /path/to/preprocessed_data_bins \\
        --output_dir runs/biopm_50mr \\
        --mask_rate 0.50 \\
        --ddp

To reproduce the other two released checkpoints, simply re-run with
``--mask_rate 0.25`` or ``--mask_rate 0.75``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import os
import random
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Add biopm package to path so this script works both from the repo
# root and as an installed entry point.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopm.pretraining import (
    EarlyStopping, PretrainModel, apply_time_based_masking,
    apply_uniform_masking, masked_recon_loss,
)

warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool")


# ===================== DDP setup / teardown =====================
def _ddp_setup(rank: int, world_size: int, master_port: str = "12355") -> None:
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def _ddp_cleanup() -> None:
    dist.destroy_process_group()


# ===================== Streaming dataset with uniform weights =====================
class _SubjectReservoirCache:
    """Bounded reservoir sample of window-file paths per subject."""

    def __init__(self, max_subjects: int = 128, reservoir_size: int = 1200):
        self.max_subjects = int(max_subjects)
        self.reservoir_size = int(reservoir_size)
        self._cache: Dict[int, List[str]] = {}
        self._lru: List[int] = []

    def _touch(self, sid: int) -> None:
        if sid in self._lru:
            self._lru.remove(sid)
        self._lru.append(sid)
        while len(self._lru) > self.max_subjects:
            self._cache.pop(self._lru.pop(0), None)

    def get_or_build(self, sid: int, subject_dir: str, input_type: str) -> List[str]:
        if sid in self._cache:
            self._touch(sid)
            return self._cache[sid]
        reservoir: List[str] = []
        k = 0
        prefix = f"window_{input_type}_"
        try:
            with os.scandir(subject_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not (name.startswith(prefix) and name.endswith(".npy")):
                        continue
                    k += 1
                    p = os.path.join(subject_dir, name)
                    if len(reservoir) < self.reservoir_size:
                        reservoir.append(p)
                    else:
                        j = random.randint(1, k)
                        if j <= self.reservoir_size:
                            reservoir[j - 1] = p
        except FileNotFoundError:
            reservoir = []
        if not reservoir:
            sample = glob.glob(os.path.join(subject_dir, f"window_{input_type}_*.npy"))
            reservoir = sample[: self.reservoir_size]
        self._cache[sid] = reservoir
        self._touch(sid)
        return reservoir


class UniformSubjectDataset(Dataset):
    """Stream windows uniformly across subjects.

    Each ``__getitem__`` picks a random subject, then a random window
    from that subject's bounded reservoir.  Returns the same 4-tuple the
    pretraining model expects:
        (me_patches, pos_info, mask_info, additional_embedding)
    """

    def __init__(self, subject_ids: List[int], subject_dirs: Dict[int, str],
                 max_length: int, mask_rate: float, fs: float, ws: float,
                 input_type: str = "acc_filt",
                 per_subject_samples: int = 1000,
                 reservoir_subject_cache: int = 128,
                 reservoir_size_per_subject: int = 1200,
                 additional_embedding_num: int = 3):
        self.subject_ids = list(subject_ids)
        self.subject_dirs = dict(subject_dirs)
        self.max_length = int(max_length)
        self.mask_rate = float(mask_rate)
        self.fs = float(fs)
        self.ws = float(ws)
        self.input_type = str(input_type)
        self.per_subject_samples = int(per_subject_samples)
        self.additional_embedding_num = int(additional_embedding_num)
        self.cache = _SubjectReservoirCache(
            max_subjects=reservoir_subject_cache,
            reservoir_size=reservoir_size_per_subject,
        )

    def __len__(self) -> int:
        return len(self.subject_ids) * self.per_subject_samples

    def _pick(self, sid: int) -> Optional[str]:
        files = self.cache.get_or_build(sid, self.subject_dirs[sid], self.input_type)
        return random.choice(files) if files else None

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx // self.per_subject_samples]
        window_path = self._pick(sid)
        if window_path is None:
            empty = torch.full((1, self.max_length), float("nan"))
            return (empty, torch.full((1,), float("nan")),
                    torch.zeros((1,), dtype=torch.float32),
                    torch.full((1, self.additional_embedding_num), float("nan")))

        parts = window_path.replace(".npy", "").split("_")
        sid_str, index_str = parts[-2], parts[-1]
        subject_dir = self.subject_dirs[int(sid_str)]
        merge_path = (f"{subject_dir}/merge_{self.input_type}_"
                      f"{sid_str}_{index_str}.npy")
        info_path = (f"{subject_dir}/me_normalizeInfo_padding_"
                     f"{self.input_type}_{sid_str}_{index_str}.pkl")

        merge_padded = np.load(merge_path)
        me_normalize = merge_padded[:, : self.max_length]
        pos_info = merge_padded[:, self.max_length]
        add_emb = merge_padded[:, self.max_length + 1:]
        me_info_df = pd.read_pickle(info_path).astype(np.float32)

        me = torch.as_tensor(me_normalize, dtype=torch.float32).clone()
        pos = torch.as_tensor(pos_info, dtype=torch.float32).clone()
        add = torch.as_tensor(add_emb, dtype=torch.float32).clone()
        # Drop wrap-around (positions encoded > 1 are concatenated windows)
        pos[torch.where(pos > 1)] -= 1
        pos[torch.where(pos > 1)] -= 1

        valid = ~torch.isnan(me).any(dim=-1)
        me_info = torch.as_tensor(np.asarray(me_info_df, dtype=np.float32))
        se_time_index = me_info[:, 1:3]

        if torch.rand(1).item() < 0.5:
            mask = apply_uniform_masking(valid, self.mask_rate)
        else:
            mask = torch.zeros_like(valid, dtype=torch.float32)
            mask = apply_time_based_masking(
                mask, se_time_index, self.fs, self.ws, self.mask_rate,
            )

        return me, pos, mask, add


# ===================== Subject discovery / split =====================
def _discover_subjects(data_dir: str, input_type: str,
                       subject_min: int, subject_max: int):
    pat = f"window_{input_type}_"
    subject_ids: List[int] = []
    subject_dirs: Dict[int, str] = {}
    for sid in range(int(subject_min), int(subject_max)):
        sd = os.path.join(data_dir, str(sid))
        if not os.path.isdir(sd):
            continue
        has_any = False
        try:
            with os.scandir(sd) as it:
                for entry in it:
                    if (entry.is_file() and entry.name.startswith(pat)
                            and entry.name.endswith(".npy")):
                        has_any = True
                        break
        except FileNotFoundError:
            continue
        if has_any:
            subject_ids.append(sid)
            subject_dirs[sid] = sd
    return subject_ids, subject_dirs


def _split_subjects(subject_ids, seed: int):
    from sklearn.model_selection import train_test_split
    arr = np.array(subject_ids)
    train, test = train_test_split(arr, test_size=0.05, random_state=seed)
    sub_train, sub_val = train_test_split(train, test_size=0.05, random_state=seed)
    return list(sub_train), list(sub_val), list(test)


# ===================== Train / Validate loops =====================
def _train_one_epoch(model, loader, optimizer, device, epoch, rank, world_size,
                     weight_for_masked: float, iter_ckpt_every: int,
                     iter_ckpt_dir: str, keep_only_last: bool):
    model.train()
    losses, masked_log, unmasked_log = [], [], []
    t0 = time.time()
    it = 0
    for me, pos, mask, add in loader:
        me = me.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        add = add.to(device, non_blocking=True)

        pred = model(me, pos, mask, add)
        loss, m_log, u_log = masked_recon_loss(pred, me, mask, weight_for_masked)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        masked_log.append(m_log)
        unmasked_log.append(u_log)

        if (rank == 0 or world_size == 1) and iter_ckpt_every and it % iter_ckpt_every == 0:
            state = (model.module.state_dict()
                     if isinstance(model, DDP) else model.state_dict())
            path = os.path.join(iter_ckpt_dir, f"rank{rank}_e{epoch}_it{it}.pt")
            torch.save(state, path)
            if keep_only_last:
                for f in os.listdir(iter_ckpt_dir):
                    if f.startswith(f"rank{rank}_e{epoch}_it") and f != os.path.basename(path):
                        try:
                            os.remove(os.path.join(iter_ckpt_dir, f))
                        except OSError:
                            pass
        it += 1

    if rank == 0 or world_size == 1:
        m = sum(masked_log) / max(len(masked_log), 1)
        u = sum(unmasked_log) / max(len(unmasked_log), 1)
        print(f"[train] epoch {epoch}  it {it}  masked_l1={m:.6f}  unmasked_l1={u:.6f}")
    return losses, time.time() - t0


@torch.no_grad()
def _validate(model, loader, device, weight_for_masked: float, tag: str,
              rank: int, world_size: int):
    model.eval()
    losses, masked_log, unmasked_log = [], [], []
    for me, pos, mask, add in loader:
        me = me.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        add = add.to(device, non_blocking=True)
        pred = model(me, pos, mask, add)
        loss, m_log, u_log = masked_recon_loss(pred, me, mask, weight_for_masked)
        losses.append(loss.item())
        masked_log.append(m_log)
        unmasked_log.append(u_log)
    if rank == 0 or world_size == 1:
        m = sum(masked_log) / max(len(masked_log), 1)
        u = sum(unmasked_log) / max(len(unmasked_log), 1)
        print(f"[{tag}] masked_l1={m:.6f}  unmasked_l1={u:.6f}")
    return losses


# ===================== Loss-curve plot (optional) =====================
def _plot_losses(train, val, test, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if train:
        ax.plot(train, label="train")
    if val:
        ax.plot(val, label="val")
    if test:
        ax.plot(test, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("BioPM pretraining")
    fig.tight_layout()
    out = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


# ===================== Single-rank training driver =====================
def _train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if world_size > 1:
        _ddp_setup(rank, world_size, master_port=args.ddp_port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0 or world_size == 1:
        print(f"=> data_dir       : {args.data_dir}")
        print(f"=> output_dir     : {args.output_dir}")
        print(f"=> mask_rate      : {args.mask_rate}")
        print(f"=> world_size     : {world_size}")

    sids, sdirs = _discover_subjects(
        args.data_dir, args.input_type, args.subject_min, args.subject_max,
    )
    if rank == 0 or world_size == 1:
        print(f"found {len(sids)} subjects")
    if not sids:
        raise RuntimeError(f"No subjects found under {args.data_dir}")

    train_sids, val_sids, test_sids = _split_subjects(sids, args.seed)
    if args.edit_mode:
        train_sids, val_sids, test_sids = train_sids[:5], val_sids[:5], test_sids[:5]
    if rank == 0 or world_size == 1:
        print(f"train={len(train_sids)}  val={len(val_sids)}  test={len(test_sids)}")

    common = dict(
        max_length=args.max_length, mask_rate=args.mask_rate,
        fs=args.fs, ws=args.ws, input_type=args.input_type,
        reservoir_subject_cache=args.reservoir_subject_cache,
        reservoir_size_per_subject=args.reservoir_size_per_subject,
        additional_embedding_num=args.additional_embedding_num,
    )
    train_ds = UniformSubjectDataset(train_sids, sdirs,
                                     per_subject_samples=args.per_subject_samples,
                                     **common)
    val_ds = UniformSubjectDataset(val_sids, sdirs,
                                   per_subject_samples=args.per_subject_eval_samples,
                                   **common)
    test_ds = UniformSubjectDataset(test_sids, sdirs,
                                    per_subject_samples=args.per_subject_eval_samples,
                                    **common)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, world_size, rank,
                                         shuffle=False, drop_last=True)
        test_sampler = DistributedSampler(test_ds, world_size, rank,
                                          shuffle=False, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=args.num_workers,
                                  pin_memory=args.pin_memory,
                                  prefetch_factor=args.prefetch_factor or 2,
                                  persistent_workers=args.persistent_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                sampler=val_sampler,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=args.pin_memory)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=args.pin_memory,
                                  prefetch_factor=args.prefetch_factor or 2,
                                  persistent_workers=args.persistent_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=args.pin_memory)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = PretrainModel(patch_len=args.max_length).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                  patience=250)

    os.makedirs(args.output_dir, exist_ok=True)
    iter_ckpt_dir = os.path.join(args.output_dir, "iter_ckpt")
    epoch_ckpt_dir = os.path.join(args.output_dir, "epoch_ckpt")
    os.makedirs(iter_ckpt_dir, exist_ok=True)
    os.makedirs(epoch_ckpt_dir, exist_ok=True)

    early_path = os.path.join(args.output_dir, "checkpoint.pt")
    early = EarlyStopping(patience=args.patience, path=early_path, verbose=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {n_params:,}")

    train_losses, val_losses, test_losses = [], [], []
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0 or world_size == 1:
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            print(f"epoch {epoch}  GPU mem ~ {mem_mb:.1f} MB")

        tr, elapsed = _train_one_epoch(
            model, train_loader, optimizer, device, epoch, rank, world_size,
            args.weight_for_masked, args.iter_checkpoint_every,
            iter_ckpt_dir, args.keep_only_last,
        )
        va = _validate(model, val_loader, device, args.weight_for_masked,
                       tag="val ", rank=rank, world_size=world_size)
        te = _validate(model, test_loader, device, args.weight_for_masked,
                       tag="test", rank=rank, world_size=world_size)

        if world_size > 1:
            dist.barrier()
        train_losses.append(float(np.mean(tr)) if tr else float("nan"))
        val_losses.append(float(np.mean(va)) if va else float("nan"))
        test_losses.append(float(np.mean(te)) if te else float("nan"))

        if rank == 0 or world_size == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  EPOCH {epoch + 1}/{args.epochs}"
                  f"  {elapsed:.1f}s"
                  f"  train={train_losses[-1]:.6f}"
                  f"  val={val_losses[-1]:.6f}"
                  f"  test={test_losses[-1]:.6f}"
                  f"  lr={lr:.2e}")

        # Early stopping on rank 0; broadcast decision
        should_stop_t = torch.tensor(0, device=device)
        if rank == 0:
            early(val_losses[-1], model)
            should_stop_t.fill_(int(early.early_stop))
        if world_size > 1:
            dist.broadcast(should_stop_t, src=0)
        if bool(should_stop_t.item()):
            if rank == 0:
                print("Early stopping triggered.")
            break

        if (rank == 0 or world_size == 1) and args.epoch_checkpoint_every:
            if (epoch + 1) % args.epoch_checkpoint_every == 0:
                state = (model.module.state_dict()
                         if isinstance(model, DDP) else model.state_dict())
                path = os.path.join(epoch_ckpt_dir, f"rank{rank}_epoch{epoch + 1}.pt")
                torch.save(state, path)
                if args.keep_only_last:
                    for f in os.listdir(epoch_ckpt_dir):
                        if f.startswith(f"rank{rank}_epoch") and f != os.path.basename(path):
                            try:
                                os.remove(os.path.join(epoch_ckpt_dir, f))
                            except OSError:
                                pass

    # Reload best
    if rank == 0 or world_size == 1:
        if os.path.exists(early_path):
            sd = torch.load(early_path, map_location=device, weights_only=False)
            target = model.module if isinstance(model, DDP) else model
            target.load_state_dict(sd, strict=False)
            print(f"Reloaded best checkpoint from {early_path}")

        _plot_losses(train_losses, val_losses, test_losses, args.output_dir)

    if world_size > 1:
        dist.barrier()
        _ddp_cleanup()


def _ddp_entry(rank: int, world_size: int, args: argparse.Namespace) -> None:
    args = copy.deepcopy(args)
    torch.cuda.set_device(rank)
    _train(rank, world_size, args)


# ===================== CLI =====================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Directory containing per-subject subdirectories "
                        "with the merge_/window_/me_normalizeInfo_ files.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write checkpoints and the loss-curve plot.")
    p.add_argument("--mask_rate", type=float, default=0.50,
                   help="Fraction of valid MEs to mask. Default 0.50.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--weight_for_masked", type=float, default=100.0,
                   help="Weight on the masked-position L1 term.")
    p.add_argument("--patience", type=int, default=1000,
                   help="EarlyStopping patience (in epochs).")
    p.add_argument("--max_length", type=int, default=32,
                   help="Patch length (samples per normalised ME).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fs", type=float, default=40.0,
                   help="Sampling rate of the source acceleration signal.")
    p.add_argument("--ws", type=float, default=10.0,
                   help="Window length in seconds.")
    p.add_argument("--input_type", default="acc_filt")
    p.add_argument("--subject_min", type=int, default=62161,
                   help="Inclusive lower bound of integer subject ids to scan.")
    p.add_argument("--subject_max", type=int, default=83461,
                   help="Exclusive upper bound of integer subject ids to scan.")
    p.add_argument("--per_subject_samples", type=int, default=1000)
    p.add_argument("--per_subject_eval_samples", type=int, default=128)
    p.add_argument("--reservoir_subject_cache", type=int, default=128)
    p.add_argument("--reservoir_size_per_subject", type=int, default=1200)
    p.add_argument("--additional_embedding_num", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=1)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--iter_checkpoint_every", type=int, default=10,
                   help="Save model every N iterations (0 disables).")
    p.add_argument("--epoch_checkpoint_every", type=int, default=1,
                   help="Save model every N epochs (0 disables).")
    p.add_argument("--keep_only_last", action="store_true",
                   help="Keep only the most recent iter / epoch checkpoint.")
    p.add_argument("--edit_mode", action="store_true",
                   help="Use only 5 subjects per split (smoke test).")
    p.add_argument("--ddp", action="store_true",
                   help="Use all visible GPUs via torch.distributed.")
    p.add_argument("--ddp_port", default="12355",
                   help="TCP port for the DDP rendezvous master.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA available: {torch.cuda.is_available()}  "
          f"GPUs visible: {n_gpus}")

    if args.ddp and n_gpus > 1:
        print(f"Running DDP on {n_gpus} GPUs")
        mp.spawn(_ddp_entry, args=(n_gpus, args), nprocs=n_gpus, join=True)
    else:
        if args.ddp and n_gpus <= 1:
            print("--ddp requested but only one GPU visible; falling back to single-GPU.")
        _train(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Script completed in {time.time() - start:.1f} s")
