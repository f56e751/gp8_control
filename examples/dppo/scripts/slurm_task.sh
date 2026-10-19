#!/bin/bash
#SBATCH --job-name=gp8_dppo
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=gp8_control/examples/dppo/runs/slurm/slurm-%A_%a.out
# Runs line $SLURM_ARRAY_TASK_ID of the task file ($1) as a shell command, from the
# parent directory of gp8_control. One GPU, 10 CPUs, 60 GB per task (cluster rule).
# The log directory above must exist before submitting (make_table_tasks.sh creates it).
#
#     sbatch --array=1-N gp8_control/examples/dppo/scripts/slurm_task.sh <task file>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# Single-threaded CPU math, as in the reported runs: the thread count can change the
# last bits of the CPU reductions that set the preference margin.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
[ -n "$CMD" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $1" >&2; exit 1; }
echo "host=$(hostname) $(date) task=$SLURM_ARRAY_TASK_ID: $CMD"
bash -c "$CMD"
echo "DONE $(date)"
