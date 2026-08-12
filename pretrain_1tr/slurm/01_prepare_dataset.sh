#!/bin/bash
#SBATCH -J brainlm_1tr_prep
#SBATCH -N 1
#SBATCH -o prepare_dataset_%j.out
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb

# Builds the 424-region arrow datasets (train/val/test + coords) from
# $DATA_DIR/input/fmri_timeseries.h5. CPU-only, no GPU needed.
#
# Before running: copy fmri_timeseries.h5 and A424_Coordinates.dat to
# $DATA_DIR/input/ on the cluster yourself (scp/rsync/Globus) -- these are real
# subject data and must never go through git. See ../README.md.

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
DATA_DIR="${DATA_DIR:-$HOME/data/brainlm}"

cd "${PROJ_DIR}/pretrain_1tr"

python prepare_dataset.py \
    --input_h5 "${DATA_DIR}/input/fmri_timeseries.h5" \
    --coords   "${DATA_DIR}/input/toolkit/atlases/A424_Coordinates.dat" \
    --out_dir  "${DATA_DIR}/pretrain_1tr/arrow"
