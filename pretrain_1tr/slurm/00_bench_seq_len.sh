#!/bin/bash
#SBATCH -J brainlm_1tr_bench
#SBATCH -N 1
#SBATCH -o bench_seq_len_%j.out
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32gb

# Run this FIRST, before committing to a real training job. Tries a few window
# lengths and reports step time + peak GPU memory so you can pick
# num_timepoints_per_voxel for 02/03/04 with actual numbers instead of a guess.

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
cd "${PROJ_DIR}/pretrain_1tr"

python bench_seq_len.py \
    --num_brain_voxels 424 \
    --num_timepoints_per_voxel 16 32 64 128 \
    --batch_size 4 \
    --device cuda
