#!/bin/bash
#SBATCH --job-name=asym1pull
#SBATCH --partition=ccb
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # single core per job
#SBATCH --mem=64G

# --- Environment setup ---
module load python/3.11
module load ffmpeg

# --- Run code ---
python asymmetric_motor_lattice.py