#!/bin/bash
#SBATCH -J brainlm_1tr_scratch
#SBATCH -N 1
#SBATCH -o train_from_scratch_%j.out
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb

# Random-init control run: same architecture as old_13M (hidden_size=512, 4 encoder /
# 2 decoder layers) but timepoint_patching_size=1, trained from scratch on the new
# 424-region / 1-TR-token dataset. Compare against 04_train_warm_start.sh to check
# whether transplanting old_13M's weights actually helps.
#
# WINDOW must match what 01/02 were run with.

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
    --output_dir "${DATA_DIR}/pretrain_1tr/checkpoints/from_scratch/${RUN_NAME}" \
    --train_dataset_path  "${DATA_DIR}/pretrain_1tr/arrow/train" \
    --val_dataset_path    "${DATA_DIR}/pretrain_1tr/arrow/val" \
    --coords_dataset_path "${DATA_DIR}/pretrain_1tr/arrow/coords" \
    --recording_col_name "Voxelwise_RobustScaler_Normalized_Recording" \
    --num_timepoints_per_voxel "${WINDOW}" \
    --timepoint_patching_size 1 \
    --hidden_size 512 \
    --num_hidden_layers 4 \
    --num_attention_heads 4 \
    --intermediate_size 1024 \
    --decoder_hidden_size 512 \
    --decoder_num_hidden_layers 2 \
    --decoder_num_attention_heads 4 \
    --decoder_intermediate_size 1024 \
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
    --wandb_logging True
