#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
# Run calibrate_model.py with locked revisions, --dtype bfloat16, --target_style llama,
# --simulate_row_parallel_calibration --num_partitions 8, and the manifest protocol.
