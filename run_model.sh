#!/bin/bash

#SBATCH -J brainlm_inference
#SBATCH -N 1
#SBATCH -o brainlm_output_%j.out
#SBATCH -t 24:00:00


# Prepare the cluster's underlying library paths
ml python/3.11.3

# Source your custom fast-storage Conda manager
source /share/data1/mossf/miniconda3/etc/profile.d/conda.sh

# Activate your specific data science environment
conda activate brainlm

# =====================================================================
# PATH CONFIGURATION (Update these to match your exact directory layout)
# =====================================================================
PROJ_DIR="/home/mossf/projects/brainlm"
DATA_DIR="/share/data1/mossf/data/brainlm"

SCRIPT_PATH="${PROJ_DIR}/run_model.py"
INPUT_H5="${DATA_DIR}/input/fmri_timeseries_subset_100.h5"
# Changed from a single .h5 file to a directory path
OUTPUT_DIR="${DATA_DIR}/outputs/attention_matrices"
COORDS="${DATA_DIR}/input/toolkit/atlases/A424_Coordinates.dat"
OUT_DIR="${PROJ_DIR}/logs"
HF_LOCAL_REPO="${DATA_DIR}/input/brainlm_hf_model"

# =====================================================================
# EXECUTION
# =====================================================================
# Runs the script safely with the built-in CPU thread-capping protections
python "$SCRIPT_PATH" \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --coords "$COORDS" \
    --hf_repo "$HF_LOCAL_REPO"