#!/usr/bin/env python3
"""
Diagnostic: does the warm-started checkpoint already produce NaN/exploding predictions
on real data at a single forward pass, BEFORE any training step?

Context: 04_train_warm_start.sh at WINDOW=32 hit loss=463525017.6 at step 10, then the
eval pass crashed with sklearn's "Input contains NaN" -- confirming the model's
predictions actually diverged to NaN within the first 10 steps. A max-abs scan of the
real training data came back at 10.16 (totally normal), so the blow-up is in the model's
predictions, not the data. This script isolates whether that's already true at
initialization (pointing at the warm start/transplant itself -- old_13M's weights were
trained with num_patches=10 per region, this checkpoint runs at num_patches=32) or only
develops after a few gradient updates (pointing at optimizer/LR settings instead).

No backward pass, no gradient checkpointing needed (torch.no_grad()) -- deliberately
lightweight, safe to run directly on CPU without touching the GPU queue at all.

Usage:
    python debug_warmstart_forward.py \
        --checkpoint_dir /share/data1/mossf/data/brainlm/pretrain_1tr/checkpoints/warm_start_init_w32 \
        --train_dataset_path /share/data1/mossf/data/brainlm/pretrain_1tr/arrow/train \
        --coords_dataset_path /share/data1/mossf/data/brainlm/pretrain_1tr/arrow/coords \
        --window 32
"""
import argparse
import os
import sys

import torch
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brainlm_mae.modeling_brainlm import BrainLMForPretraining


def describe(name, t):
    t = t.detach()
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        print(f"[{name}] shape={tuple(t.shape)} ALL non-finite (nan={has_nan} inf={has_inf})")
        return
    print(
        f"[{name}] shape={tuple(t.shape)} min={finite.min().item():.4g} "
        f"max={finite.max().item():.4g} mean={finite.mean().item():.4g} "
        f"std={finite.std().item():.4g} has_nan={has_nan} has_inf={has_inf}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--train_dataset_path", required=True)
    ap.add_argument("--coords_dataset_path", required=True)
    ap.add_argument("--recording_col_name", default="Voxelwise_RobustScaler_Normalized_Recording")
    ap.add_argument("--window", type=int, required=True, help="num_timepoints_per_voxel, must match the checkpoint")
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    print(f"[load] model from {args.checkpoint_dir} ...")
    model = BrainLMForPretraining.from_pretrained(args.checkpoint_dir)
    model.eval()
    print(f"[config] num_brain_voxels={model.config.num_brain_voxels} "
          f"timepoint_patching_size={model.config.timepoint_patching_size} "
          f"num_timepoints_per_voxel={model.config.num_timepoints_per_voxel} "
          f"mask_ratio={model.config.mask_ratio}")

    print(f"[load] train dataset from {args.train_dataset_path} ...")
    train_ds = load_from_disk(args.train_dataset_path)
    coords_ds = load_from_disk(args.coords_dataset_path)

    xyz = torch.tensor(
        [[coords_ds[i]["X"], coords_ds[i]["Y"], coords_ds[i]["Z"]] for i in range(len(coords_ds))],
        dtype=torch.float32,
    )  # [num_regions, 3]

    signal_batch = []
    for i in range(args.batch_size):
        row = train_ds[i]
        full_signal = torch.tensor(row[args.recording_col_name], dtype=torch.float32)  # [T, num_regions]
        window = full_signal[0:args.window, :]  # deterministic start, for reproducibility
        window = torch.movedim(window, 0, 1)  # [num_regions, window]
        signal_batch.append(window)
    signal_vectors = torch.stack(signal_batch)  # [batch, num_regions, window]
    xyz_vectors = xyz.unsqueeze(0).repeat(args.batch_size, 1, 1)  # [batch, num_regions, 3]

    describe("input signal_vectors", signal_vectors)

    print("[forward] running single forward pass (no grad, no training) ...")
    with torch.no_grad():
        out = model(signal_vectors=signal_vectors, xyz_vectors=xyz_vectors)

    pred_logits, encoder_latents = out.logits
    print(f"[output] loss = {out.loss.item()}")
    describe("pred_logits", pred_logits)
    describe("encoder_latents (incl. CLS)", encoder_latents)
    describe("cls_token (encoder_latents[:, 0, :])", encoder_latents[:, 0, :])


if __name__ == "__main__":
    main()
