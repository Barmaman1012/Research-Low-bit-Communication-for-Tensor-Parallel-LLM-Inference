#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail

MODEL_KEY=${1:?usage: sbatch stage_d_validate_analyze.sbatch MODEL_KEY REVISION_LOCK}
LOCK=${2:?usage: sbatch stage_d_validate_analyze.sbatch MODEL_KEY REVISION_LOCK}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source .venv-gpu310/bin/activate
LOG_DIR="outputs/three_model_range_sweep/${MODEL_KEY}/logs/validation"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/${SLURM_JOB_ID:?}.out") 2> >(tee "$LOG_DIR/${SLURM_JOB_ID}.err" >&2)
python scripts/three_model_campaign.py run-stage d --model-key "$MODEL_KEY" --lock "$LOCK" --job-id "$SLURM_JOB_ID"
