#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00
set -euo pipefail

MODEL_KEY=${1:?usage: sbatch stage_b_load_smoke.sbatch MODEL_KEY REVISION_LOCK}
LOCK=${2:?usage: sbatch stage_b_load_smoke.sbatch MODEL_KEY REVISION_LOCK}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN must be exported to this Slurm job." >&2; exit 2; }
source .venv-gpu310/bin/activate
LOG_DIR="outputs/three_model_range_sweep/${MODEL_KEY}/logs/load_smoke"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/${SLURM_JOB_ID:?}.out") 2> >(tee "$LOG_DIR/${SLURM_JOB_ID}.err" >&2)
python scripts/three_model_campaign.py run-stage b --model-key "$MODEL_KEY" --lock "$LOCK" --job-id "$SLURM_JOB_ID"
