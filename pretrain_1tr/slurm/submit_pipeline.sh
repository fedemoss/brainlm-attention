#!/bin/bash
# Submits the warm-start branch of the pretrain_1tr pipeline as chained SLURM jobs:
#   00 (bench, informational)  -\
#   01 (prepare_dataset)        +--> 04 (train_warm_start), via --dependency=afterok
#   02 (transplant_checkpoint) -/
#
# 03 (train_from_scratch) is intentionally NOT submitted here -- at the current dataset
# scale (~1082 subjects, see ../README.md "Data") it's not expected to converge to
# anything useful and isn't needed to run the warm-start branch. Submit it yourself
# (`sbatch 03_train_from_scratch.sh`) if you want the from-scratch control run later.
#
# 00 doesn't block anything -- it's informational (peak-memory/step-time numbers to help
# pick WINDOW) and 01/02 don't depend on its result. Read its output and re-run this
# script with a different WINDOW by hand if it suggests one; there's no way to automate
# "read the bench numbers and pick a window" here.
#
# Usage:
#   ./submit_pipeline.sh                       # WINDOW=64
#   WINDOW=32 RUN_NAME=my_run ./submit_pipeline.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export WINDOW="${WINDOW:-64}"

echo "[submit] 00_bench_seq_len.sh (informational, does not block 01/02/04)"
JOB00=$(sbatch --parsable 00_bench_seq_len.sh)
echo "  -> job ${JOB00}"

echo "[submit] 01_prepare_dataset.sh"
JOB01=$(sbatch --parsable 01_prepare_dataset.sh)
echo "  -> job ${JOB01}"

echo "[submit] 02_transplant_checkpoint.sh (WINDOW=${WINDOW})"
JOB02=$(sbatch --parsable 02_transplant_checkpoint.sh)
echo "  -> job ${JOB02}"

echo "[submit] 04_train_warm_start.sh (WINDOW=${WINDOW}), waiting on ${JOB01} (dataset) + ${JOB02} (checkpoint)"
JOB04=$(sbatch --parsable --dependency="afterok:${JOB01}:${JOB02}" 04_train_warm_start.sh)
echo "  -> job ${JOB04}"

cat <<EOF

[summary]
  00 bench_seq_len         ${JOB00}   (informational only)
  01 prepare_dataset       ${JOB01}
  02 transplant_checkpoint ${JOB02}   (WINDOW=${WINDOW})
  04 train_warm_start      ${JOB04}   (WINDOW=${WINDOW}, waits on ${JOB01} + ${JOB02})

03_train_from_scratch.sh was NOT submitted (data-scale control, skipped -- see ../README.md).
Check status with: squeue -u \$USER
If 01 or 02 fails, 04 will sit as "DependencyNeverSatisfied" and never run -- cancel it
(scancel ${JOB04}) and resubmit once the failure is fixed.
EOF
