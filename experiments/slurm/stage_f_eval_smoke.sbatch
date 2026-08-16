#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
set -euo pipefail

MODEL_KEY=${1:?usage: sbatch --array=0-12%1 stage_f_eval_smoke.sbatch MODEL_KEY REVISION_LOCK}
LOCK=${2:?usage: sbatch --array=0-12%1 stage_f_eval_smoke.sbatch MODEL_KEY REVISION_LOCK}
: "${SLURM_ARRAY_TASK_ID:?SLURM array task ID is required}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN must be exported to this Slurm job." >&2; exit 2; }
source .venv-gpu310/bin/activate
LOG_DIR="outputs/three_model_range_sweep/${MODEL_KEY}/logs/smoke"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") 2> >(tee "$LOG_DIR/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)
python scripts/three_model_campaign.py run-stage f --model-key "$MODEL_KEY" --lock "$LOCK" --array-index "$SLURM_ARRAY_TASK_ID" --job-id "$SLURM_JOB_ID"
