#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00
set -euo pipefail
MODEL=$1; LOCK=$2
# Resolve MODEL_ID/MODEL_REVISION/TOKENIZER_REVISION from $LOCK before submission.
# Gemma stops here if peak allocation exceeds 85% of H100 80GB.
echo "Use scripts/model_loading_smoke.py with the locked revisions for $MODEL"
