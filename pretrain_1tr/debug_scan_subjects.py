#!/usr/bin/env python3
"""
Diagnostic: scan every training subject through the (untrained, warm-started) model with
a single forward pass each, random window per subject (same random-windowing the real
training run uses), and report any subject whose loss/predictions are anomalous.

Context: debug_warmstart_forward.py showed the model is completely healthy at init on
2 subjects with a deterministic window (loss=2.11, no NaN/inf anywhere). The real
training run diverged to loss=463525017.6 and then NaN within its first 10 optimizer
steps (160 micro-batches, real random windows across many subjects). This scans ALL
training subjects with random windows through the SAME untrained model (no weight
updates at all) to check whether a specific subject/window combination alone already
produces a bad loss -- which would point at a particular pathological data sample --
versus every subject looking clean here too, which would point at something that only
emerges through the optimizer/backward dynamics instead.

CPU-only, forward pass only (torch.no_grad()) -- no GPU/queue needed.

Usage:
    python debug_scan_subjects.py \
        --checkpoint_dir /share/data1/mossf/data/brainlm/pretrain_1tr/checkpoints/warm_start_init_w32 \
        --train_dataset_path /share/data1/mossf/data/brainlm/pretrain_1tr/arrow/train \
        --coords_dataset_path /share/data1/mossf/data/brainlm/pretrain_1tr/arrow/coords \
        --window 32 --loss_threshold 10 --seed 0
"""
import argparse
import os
import sys
from random import Random

import torch
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brainlm_mae.modeling_brainlm import BrainLMForPretraining


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--train_dataset_path", required=True)
    ap.add_argument("--coords_dataset_path", required=True)
    ap.add_argument("--recording_col_name", default="Voxelwise_RobustScaler_Normalized_Recording")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--loss_threshold", type=float, default=10.0, help="flag any subject whose loss exceeds this")
    ap.add_argument("--max_subjects", type=int, default=None, help="default: scan all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = Random(args.seed)

    print(f"[load] model from {args.checkpoint_dir} ...")
    model = BrainLMForPretraining.from_pretrained(args.checkpoint_dir)
    model.eval()

    print(f"[load] train dataset from {args.train_dataset_path} ...")
    train_ds = load_from_disk(args.train_dataset_path)
    coords_ds = load_from_disk(args.coords_dataset_path)
    n_subjects = len(train_ds) if args.max_subjects is None else min(args.max_subjects, len(train_ds))

    xyz = torch.tensor(
        [[coords_ds[i]["X"], coords_ds[i]["Y"], coords_ds[i]["Z"]] for i in range(len(coords_ds))],
        dtype=torch.float32,
    ).unsqueeze(0)  # [1, num_regions, 3]

    losses = []
    flagged = []
    for i in range(n_subjects):
        row = train_ds[i]
        full_signal = torch.tensor(row[args.recording_col_name], dtype=torch.float32)  # [T, num_regions]
        start_idx = rng.randint(0, full_signal.shape[0] - args.window)
        window = full_signal[start_idx:start_idx + args.window, :]
        window = torch.movedim(window, 0, 1).unsqueeze(0)  # [1, num_regions, window]

        with torch.no_grad():
            out = model(signal_vectors=window, xyz_vectors=xyz)
        loss = out.loss.item()
        losses.append(loss)

        pred_logits = out.logits[0]
        bad = (
            loss > args.loss_threshold
            or torch.isnan(pred_logits).any().item()
            or torch.isinf(pred_logits).any().item()
        )
        if bad:
            flagged.append((i, row.get("Subject_ID", "?"), start_idx, loss))
            print(f"[FLAGGED] subject_idx={i} id={row.get('Subject_ID', '?')} "
                  f"start_idx={start_idx} loss={loss:.6g} "
                  f"has_nan={torch.isnan(pred_logits).any().item()} "
                  f"has_inf={torch.isinf(pred_logits).any().item()}")

        if (i + 1) % 100 == 0:
            print(f"[progress] {i + 1}/{n_subjects} scanned, {len(flagged)} flagged so far")

    losses_t = torch.tensor(losses)
    print()
    print(f"[summary] scanned {n_subjects} subjects")
    print(f"[summary] loss: min={losses_t.min().item():.4g} max={losses_t.max().item():.4g} "
          f"mean={losses_t.mean().item():.4g} median={losses_t.median().item():.4g}")
    print(f"[summary] {len(flagged)} subjects flagged (loss > {args.loss_threshold} or NaN/Inf)")
    if flagged:
        print("[summary] flagged subjects:")
        for i, sid, start_idx, loss in flagged:
            print(f"    subject_idx={i} id={sid} start_idx={start_idx} loss={loss:.6g}")


if __name__ == "__main__":
    main()
