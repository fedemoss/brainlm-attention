#!/bin/bash
#SBATCH -J brainlm_1tr_ftau_fdc
#SBATCH -N 1
#SBATCH -o run_ftau_fdc_%j.out
#SBATCH -t 06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
# --mem=0 = "use all memory on the node" (Slurm special value) -- see 00_bench_seq_len.sh
# for why a fixed --mem value fails on every partition on this cluster.

# Runs validation/run_model_1tr.py over every subject in a cohort h5: extracts the
# last-layer region x region attention matrix + F_tau(lag) matrix per subject, at the
# checkpoint's real training/eval scale (window_size must match the checkpoint's
# num_timepoints_per_voxel, 32 here -- see ../README.md). CPU-only, no GPU requested:
# a synthetic single-window forward pass at this scale (seq_len=13,568) measured ~19s on
# CPU, so even the full 1082-subject cohort (3 windows/subject, default --num_windows)
# is ~1082*3*19s =~ 17hrs worst case serial, comfortably inside a re-submittable job, and
# far less for the 100-subject subset this defaults to (~85min). Bump -t if you switch
# to the full cohort.
#
# Deliberately NOT run on a laptop: NystromformerSelfAttention's exact O(seq_len^2)
# attention fallback (num_landmarks == segment_means_seq_len, see README) means every
# subject's forward pass peaks ~15GB RSS just for the attention tensors -- confirmed to
# OOM-kill (exit 137) a laptop even at --limit 1 (one subject). --mem=0 on this node
# sidesteps that entirely.
#
# One-time prerequisite: pretrain_1tr/validation/run_model_1tr.py is not committed to
# git yet (its checkpoint/ subdirectory holds a 71MB weights file that must never go
# into the repo -- see ../../README.md's data policy -- so don't `git add` that
# directory wholesale). Copy just the script over by hand before the first submission:
#   scp pretrain_1tr/validation/run_model_1tr.py \
#       udesa-cluster:"${PROJ_DIR:-\$HOME/projects/brainlm-attention}"/pretrain_1tr/validation/run_model_1tr.py
# postprocess.py / visualize_attention.py (repo root) it imports are already tracked and
# present on the cluster.

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-/share/data1/mossf/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
DATA_DIR="${DATA_DIR:-/share/data1/mossf/data/brainlm}"

# Same 100-subject subset the existing FC/FDC validation notebooks compare against
# (old_13M's attention-vs-FC/FDC correlation, see ../README.md "Motivation") -- override
# with the full cohort only if you deliberately want numbers over the larger dataset
# instead of a like-for-like comparison against those already-established results.
INPUT_H5="${INPUT_H5:-${DATA_DIR}/input/fmri_timeseries_subset_100.h5}"

# The checkpoint slurm/04_train_warm_start.sh produced (final save sits at this run
# directory's top level, alongside the per-step checkpoint-N/ subdirs -- see
# trainer_state.json's best_model_checkpoint for provenance). Override to point at a
# different RUN_NAME if you've trained since.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DATA_DIR}/pretrain_1tr/checkpoints/warm_start/2026-08-19-00_59_41}"

WINDOW="${WINDOW:-32}"     # must match the checkpoint's num_timepoints_per_voxel (see README)
FTAU_LAG="${FTAU_LAG:-1}"
OUT_DIR="${DATA_DIR}/pretrain_1tr/validation/attention_matrices"
FIG_DIR="${DATA_DIR}/pretrain_1tr/validation/figures"

mkdir -p "${OUT_DIR}" "${FIG_DIR}"

cd "${PROJ_DIR}/pretrain_1tr/validation"

python run_model_1tr.py \
    --input_h5       "${INPUT_H5}" \
    --coords         "${DATA_DIR}/input/toolkit/atlases/A424_Coordinates.dat" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir     "${OUT_DIR}" \
    --figures_dir    "${FIG_DIR}" \
    --window_size    "${WINDOW}" \
    --ftau_lag       "${FTAU_LAG}"

echo "Done. Pull results back with, e.g.:"
echo "  rsync -avz udesa-cluster:${OUT_DIR}/ pretrain_1tr/validation/attention_matrices/"
