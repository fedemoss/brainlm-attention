#!/bin/bash

#SBATCH -J brainlm_inference
#SBATCH -N 1
#SBATCH -o brainlm_output_%j.out
#SBATCH -t 24:00:00


# Prepare the cluster's underlying library paths (adjust to your cluster's module system)
ml python/3.11.3

# Source your conda installation, then activate the env created per the README
source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate brainlm

# =====================================================================
# PATH CONFIGURATION (Update these to match your exact directory layout)
# =====================================================================
PROJ_DIR="${PROJ_DIR:-$HOME/projects/brainlm}"
DATA_DIR="${DATA_DIR:-$HOME/data/brainlm}"

SCRIPT_PATH="${PROJ_DIR}/run_model.py"
INPUT_H5="${DATA_DIR}/input/fmri_timeseries_subset_100.h5"
OUTPUT_DIR="${DATA_DIR}/outputs/attention_matrices"
COORDS="${DATA_DIR}/input/toolkit/atlases/A424_Coordinates.dat"
# Hub repo id (weights download and cache automatically) or a local snapshot path.
HF_REPO="${HF_REPO:-vandijklab/brainlm}"

# =====================================================================
# EXECUTION
# =====================================================================
# --subfolder must stay old_13M: it's the only checkpoint whose weights match
# the BrainLMForPretraining class this script loads (see README "Correctness
# guarantees"). Runs with the built-in CPU thread-capping protections.
python "$SCRIPT_PATH" \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --coords "$COORDS" \
    --hf_repo "$HF_REPO" \
    --subfolder old_13M