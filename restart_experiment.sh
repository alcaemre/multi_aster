#!/bin/bash
#SBATCH --job-name=pdisk
#SBATCH --partition=ccb
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128        # one proposal-evaluating worker per core
#SBATCH --mem=200G

# --- Environment setup ---
module load python/3.11
module load ffmpeg

# parallelism comes from the worker processes; stop numpy's BLAS from also starting a
# thread per core inside each of them
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Run code ---
python restart_experiment.py