"""
Benchmark one forward+backward step for a candidate (num_brain_voxels,
num_timepoints_per_voxel) config with timepoint_patching_size=1, on synthetic data --
run this on the actual cluster GPU BEFORE committing to a full training job, to check
memory and step time scale the way you expect for the window length you're about to
commit to.

Sequence length = num_brain_voxels * num_timepoints_per_voxel + 1 (CLS). The encoder
is Nystromformer (linear attention via landmarks), so this should scale much better
than quadratic full attention, but the embedding/FFN/decoder cost is still
O(seq_len x hidden_size) per layer and is worth measuring directly rather than assuming.

Usage:
    python bench_seq_len.py --num_timepoints_per_voxel 64 --batch_size 4 --device cuda
    python bench_seq_len.py --num_timepoints_per_voxel 32 64 128 --batch_size 4 --device cuda
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMForPretraining

# old_13M's architecture (hidden_size, layer counts, etc.) -- see transplant_checkpoint.py.
# Kept identical here so the benchmark reflects the model you'll actually warm-start.
BASE_CONFIG = dict(
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    decoder_hidden_size=512,
    decoder_num_hidden_layers=2,
    decoder_num_attention_heads=4,
    decoder_intermediate_size=1024,
    mask_ratio=0.2,
    num_landmarks=64,
    segment_means_seq_len=64,
    conv_kernel_size=65,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.1,
    loss_fn="mse",
)


def bench_one(num_brain_voxels, num_timepoints_per_voxel, batch_size, device, gradient_checkpointing):
    cfg = BrainLMConfig(
        **BASE_CONFIG,
        num_brain_voxels=num_brain_voxels,
        num_timepoints_per_voxel=num_timepoints_per_voxel,
        timepoint_patching_size=1,
    )
    model = BrainLMForPretraining(cfg).to(device)
    if gradient_checkpointing:
        model.vit.encoder.gradient_checkpointing = True
        model.decoder.gradient_checkpointing = True
    model.train()

    signal_vectors = torch.randn(batch_size, num_brain_voxels, num_timepoints_per_voxel, device=device)
    xyz_vectors = torch.randn(batch_size, num_brain_voxels, 3, device=device)

    seq_len = num_brain_voxels * num_timepoints_per_voxel + 1
    n_params = sum(p.numel() for p in model.parameters())

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    out = model(signal_vectors=signal_vectors, xyz_vectors=xyz_vectors)
    out.loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    step_time = time.time() - t0

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    print(
        f"voxels={num_brain_voxels:4d} window={num_timepoints_per_voxel:4d} "
        f"seq_len={seq_len:6d} batch={batch_size:3d} params={n_params/1e6:.1f}M "
        f"grad_ckpt={gradient_checkpointing} -> step_time={step_time:.2f}s "
        f"peak_mem={peak_mem_gb:.2f}GB loss={out.loss.item():.4f}"
    )
    del model, out
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_brain_voxels", type=int, default=424)
    ap.add_argument("--num_timepoints_per_voxel", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    print(f"[device] {args.device}")
    for window in args.num_timepoints_per_voxel:
        try:
            bench_one(args.num_brain_voxels, window, args.batch_size, args.device, args.gradient_checkpointing)
        except RuntimeError as e:
            print(f"window={window} -> FAILED: {e}")


if __name__ == "__main__":
    main()
