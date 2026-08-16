#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
# Array index maps to CONFIGS in submit_three_model_range_sweep.sh; invoke eval_lm_harness.py
# with arc_easy,boolq --limit 2 --batch_size 1 and a distinct smoke JSON per configuration.
