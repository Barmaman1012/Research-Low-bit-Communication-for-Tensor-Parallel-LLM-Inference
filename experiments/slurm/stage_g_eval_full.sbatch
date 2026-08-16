#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
# Same array mapping; full five-task evaluation, no --limit, distinct JSON/log per configuration.
