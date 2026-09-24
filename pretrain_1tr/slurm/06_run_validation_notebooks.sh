#!/bin/bash
#SBATCH -J brainlm_1tr_validation_nb
#SBATCH -N 1
#SBATCH -o run_validation_nb_%j.out
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
# --mem=0 = "use all memory on the node" (Slurm special value) -- see 00_bench_seq_len.sh
# for why a fixed --mem value fails on every partition on this cluster.

# Executes, in place, the two model-forward-pass validation notebooks that are unsafe to
# run on a laptop:
#   - fc_attention_correlation.ipynb   (Part 2 only, "single fixed-window" section --
#     Part 1 already ran locally against precomputed outputs/attention_matrices/*.npy)
#   - ftau_fdc_asymmetry_check.ipynb   (all of it -- computes its own forward passes)
#
# Both do a per-subject forward pass at this checkpoint's real training/eval scale
# (window_size=32 -> seq_len=13,568), same computation as run_model_1tr.py
# (see slurm/05_run_ftau_fdc_validation.sh's header) -- confirmed to OOM-kill (exit 137)
# a laptop even at a single subject, because NystromformerSelfAttention's exact
# O(seq_len^2) attention fallback peaks ~15GB RSS per forward pass just for the attention
# tensors. This node's --mem=0 sidesteps that entirely (192GB RAM, see ../README.md
# "Cluster hardware").
#
# fc_attention_correlation.ipynb re-runs ALL cells (nbconvert doesn't skip already-executed
# ones), including the already-local-verified Part 1 -- harmless, it's fast (no forward
# passes) and keeps both parts consistent within one execution.
#
# One-time prerequisites (already done for this submission, re-run by hand if paths move):
#   - pretrain_1tr/validation/checkpoint -> symlinked to the warm_start run directory
#     both notebooks hardcode CHECKPOINT_DIR = "checkpoint" for (see their own headers).
#   - pretrain_1tr/validation/outputs/attention_matrices -> symlinked to the run_model_1tr.py
#     output dir (fc_attention_correlation.ipynb Part 1 reads *_attention.npy from there).
#   - data/input/{fmri_timeseries_subset_100.h5,toolkit/atlases/A424_Coordinates.dat} ->
#     symlinked into the repo checkout (both notebooks use "../../data/input/..." relative
#     paths, matching the local/laptop repo layout).
# None of these are committed to git (see ../../README.md's data policy).

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-/share/data1/mossf/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
cd "${PROJ_DIR}/pretrain_1tr/validation"

echo "=== fc_attention_correlation.ipynb ==="
jupyter nbconvert --to notebook --execute --inplace fc_attention_correlation.ipynb \
    --ExecutePreprocessor.timeout=-1

echo "=== ftau_fdc_asymmetry_check.ipynb ==="
jupyter nbconvert --to notebook --execute --inplace ftau_fdc_asymmetry_check.ipynb \
    --ExecutePreprocessor.timeout=-1

echo "Done. Pull results back with, e.g.:"
echo "  rsync -avz udesa-cluster:${PROJ_DIR}/pretrain_1tr/validation/fc_attention_correlation.ipynb pretrain_1tr/validation/"
echo "  rsync -avz udesa-cluster:${PROJ_DIR}/pretrain_1tr/validation/ftau_fdc_asymmetry_check.ipynb pretrain_1tr/validation/"
echo "  rsync -avz udesa-cluster:${PROJ_DIR}/pretrain_1tr/validation/outputs/ pretrain_1tr/validation/outputs/"
echo "  rsync -avz udesa-cluster:${PROJ_DIR}/pretrain_1tr/validation/figures/ pretrain_1tr/validation/figures/"
