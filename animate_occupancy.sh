#!/bin/bash
#SBATCH --job-name=animate
#SBATCH --partition=ccb
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # single core per job
#SBATCH --mem=128G

# --- Environment setup ---
module load python/3.11
module load ffmpeg

# --- Run code ---
python animate_occupancy.py