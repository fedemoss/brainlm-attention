#!/bin/bash
#SBATCH -J brainlm_1tr_warmstart
#SBATCH -N 1
#SBATCH -o train_warm_start_%j.out
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb

# Continues MAE pretraining from the transplanted old_13M checkpoint
# (02_transplant_checkpoint.sh). model_name_or_path takes the config (incl.
# timepoint_patching_size=1, hidden sizes) from the saved transplant dir, so the
# --hidden_size etc. flags below are NOT read in this branch of train.py -- only
# mask_ratio / attention_probs_dropout_prob are (train.py applies them
# unconditionally after loading config, see `config.update(...)`).
#
# WINDOW must match what 02_transplant_checkpoint.sh was run with.

set -euo pipefail
date; hostname; pwd

source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm-attention}"
DATA_DIR="${DATA_DIR:-$HOME/data/brainlm}"
WINDOW="${WINDOW:-64}"
RUN_NAME="${RUN_NAME:-$(date +%Y-%m-%d-%H_%M_%S)}"

cd "${PROJ_DIR}/pretrain_1tr"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

python train.py \
    --model_name_or_path "${DATA_DIR}/pretrain_1tr/checkpoints/warm_start_init" \
    --output_dir "${DATA_DIR}/pretrain_1tr/checkpoints/warm_start/${RUN_NAME}" \
    --train_dataset_path  "${DATA_DIR}/pretrain_1tr/arrow/train" \
    --val_dataset_path    "${DATA_DIR}/pretrain_1tr/arrow/val" \
    --coords_dataset_path "${DATA_DIR}/pretrain_1tr/arrow/coords" \
    --recording_col_name "Voxelwise_RobustScaler_Normalized_Recording" \
    --num_timepoints_per_voxel "${WINDOW}" \
    --timepoint_patching_size 1 \
    --mask_ratio 0.2 \
    --attention_probs_dropout_prob 0.1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 100 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --wandb_logging True \
    --base_learning_rate 2e-4
