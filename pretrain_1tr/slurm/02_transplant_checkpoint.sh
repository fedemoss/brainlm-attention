#!/bin/bash
#SBATCH -J brainlm_1tr_transplant
#SBATCH -N 1
#SBATCH -o transplant_checkpoint_%j.out
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb

# Warm-starts a timepoint_patching_size=1 model from old_13M (copies every
# shape-matching weight; reinitializes only the two patch-size-dependent layers).
# CPU-only, cheap. Needs network access to fetch old_13M from the HF Hub the first
# time (or point --hf_repo at a local snapshot dir if the cluster has no internet
# on compute nodes -- see ../README.md).
#
# WINDOW must match what 03/04 will train with (num_timepoints_per_voxel).

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
DATA_DIR="${DATA_DIR:-$HOME/data/brainlm}"
WINDOW="${WINDOW:-64}"

cd "${PROJ_DIR}/pretrain_1tr"

python transplant_checkpoint.py \
    --hf_repo vandijklab/brainlm \
    --subfolder old_13M \
    --num_timepoints_per_voxel "${WINDOW}" \
    --out_dir "${DATA_DIR}/pretrain_1tr/checkpoints/warm_start_init"
