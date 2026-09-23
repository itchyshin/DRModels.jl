#!/bin/bash
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%5
#SBATCH --job-name=drm-fw
#SBATCH --output=logs/drm-fw-%A_%a.out
#SBATCH --error=logs/drm-fw-%A_%a.err
#
# Phase B first-wave DRM (10 cells). Julia abs only.
set -euo pipefail
module load StdEnv/2023 julia/1.11.3 || module load julia/1.12.5 || module load julia/1.10.10
export JULIA_DEPOT_PATH="/scratch/snakagaw/julia_depot_speed_phaseB"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs bench/results docs/dev-log/evidence/2026-09-23-speed-firstwave-drac
OUT="docs/dev-log/evidence/2026-09-23-speed-firstwave-drac/drm_task${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.tsv"
export SPEED_ARRAY_INDEX="$SLURM_ARRAY_TASK_ID"
export DRM_FIRSTWAVE_OUT="$OUT"
export DRM_FIRSTWAVE_REPS=3
julia --project=. bench/speed_board_firstwave_drm.jl
