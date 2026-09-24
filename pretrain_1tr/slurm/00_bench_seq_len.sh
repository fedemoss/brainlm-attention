#!/bin/bash
#SBATCH -J brainlm_1tr_bench
#SBATCH -N 1
#SBATCH -o bench_seq_len_%j.out
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=c4
#SBATCH --cpus-per-task=3
#SBATCH --mem=0
# --nodelist=c4 (2x A100): the strongest candidate for window>=32 -- T4 (c2/c3) confirmed
# OOMs there (window=16 fits at 12.94GB peak, window=32 needs ~11.8GB/layer and doesn't).
# NOTE: --nodelist with multiple comma-separated nodes does NOT mean "try either" -- per
# `man sbatch`, the job must span ALL listed nodes, so `--nodelist=c4,c6` with `-N 1`
# produces "invalid number of nodes (-N 2-1)" (needs >=2 nodes for the list, capped at 1).
# To also try c6 (2x L4, 24GB, unbenched), submit a SEPARATE job overriding this at the
# command line instead: `sbatch --nodelist=c6 00_bench_seq_len.sh`.
# --mem=0 is Slurm's "use all memory on the node" special value -- a fixed --mem=32gb
# was failing with "Memory specification can not be satisfied" / "Requested node
# configuration is not available" on every partition, and `sinfo` reports MEMORY=1 for
# every node here, which looks like RealMemory is misconfigured cluster-side. If --mem=0
# also fails, this is a cluster config issue outside this script -- check
# `scontrol show node <nodename>` for the real configured RealMemory/AllocMem.

# Run this FIRST, before committing to a real training job. Tries a few window
# lengths and reports step time + peak GPU memory so you can pick
# num_timepoints_per_voxel for 02/03/04 with actual numbers instead of a guess.

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-/share/data1/mossf/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
cd "${PROJ_DIR}/pretrain_1tr"

# expandable_segments avoids PyTorch pre-reserving memory in fixed-size chunks it can't
# hand back for a differently-shaped request -- a confirmed T4 run at window=32 (batch=2)
# was short by under 1GB (needed ~15.3GB total vs 14.56GB available) with "3.21 GiB
# reserved by PyTorch but unallocated" at the moment of failure -- exactly the
# fragmentation pattern this setting targets. Cheap to try before assuming a window needs
# a bigger GPU.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# One python process PER window, not bench_seq_len.py's own multi-window loop in a
# single process: a confirmed run showed ~18-30GB still "in use" right before the
# window=32 attempt, immediately after window=16 (peak 12.94GB) supposedly finished and
# called `del model, out; torch.cuda.empty_cache()` -- torch/dynamo checkpoint-machinery
# caching that empty_cache() doesn't reclaim, contaminating later windows' numbers with
# earlier ones' leftover allocations. A fresh process per window has a clean CUDA context,
# so each number reflects only that window.
#
# --batch_size 2, not 4: 03/04's actual training config uses
# --per_device_train_batch_size 2 -- benching at batch=4 tests 2x the memory a real
# training step needs. Attention memory scales linearly with batch, so match it here.
BATCH_SIZE="${BENCH_BATCH_SIZE:-2}"
for WINDOW in 16 32 64 128; do
    echo "=== window=${WINDOW} (fresh process, batch_size=${BATCH_SIZE}) ==="
    python bench_seq_len.py \
        --num_brain_voxels 424 \
        --num_timepoints_per_voxel "${WINDOW}" \
        --batch_size "${BATCH_SIZE}" \
        --device cuda \
        --gradient_checkpointing \
        || echo "window=${WINDOW} -> process failed (see error above)"
done

# NOTE: num_landmarks == segment_means_seq_len (64 == 64) in every config this project
# uses, which makes NystromformerSelfAttention take its EXACT full-softmax branch
# (transformers/models/nystromformer/modeling_nystromformer.py, `if self.num_landmarks
# == self.seq_len`, where self.seq_len is config.segment_means_seq_len -- a config
# check, not the real runtime sequence length). So attention here is always
# O((num_brain_voxels * window)^2) per layer, never the sub-quadratic Nystrom
# approximation. --gradient_checkpointing avoids holding all 6 layers' (4 encoder + 2
# decoder) attention tensors live at once, matching what 03/04's training runs actually
# use, but it does NOT shrink any single layer's tensor.
#
# Confirmed so far (batch=4, contaminated multi-window-per-process numbers -- treat as
# upper bounds, not the real per-window cost): window=16 fits everywhere (T4/L4/A100,
# 12.94GB peak); window=32/64/128 OOM'd on T4 (c2/c3, 14.56GiB) AND L4 (c6, 22.04GiB) AND
# A100 (c4, 39.49GiB) -- but A100's OOM at window=32 happened with 30.60GB already
# "in use" before the attempt, i.e. not a clean measurement. Re-bench with this script's
# per-window isolation + batch=2 before trusting any window>=32 number.
