#!/bin/bash
#SBATCH --job-name=progress
#SBATCH --partition=ccb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # single core per job
#SBATCH --mem=128G                 # the whole trajectory is loaded into memory

# usage: sbatch plot_progress.sh <experiment_dir>
#        sbatch plot_progress.sh <experiment_dir> --window 100 --force both    (optional settings, shown at their defaults)

# --- Environment setup ---
module load python/3.11

# --- Run code ---
python plot_progress.py "$@"
