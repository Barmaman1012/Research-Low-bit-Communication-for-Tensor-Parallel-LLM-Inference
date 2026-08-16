#!/usr/bin/env bash
#SBATCH --account=columbia
#SBATCH --partition=columbia
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
set -euo pipefail

MODEL_KEY=${1:?usage: sbatch stage_c_calibrate.sbatch MODEL_KEY REVISION_LOCK}
LOCK_INPUT=${2:?usage: sbatch stage_c_calibrate.sbatch MODEL_KEY REVISION_LOCK}
ROOT=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
if [[ ! -f "$ROOT/experiments/three_model_range_sweep.yaml" || ! -f "$ROOT/experiments/three_model_range_sweep.revisions.json" || ! -f "$ROOT/scripts/three_model_campaign.py" || ! -f "$ROOT/.venv-gpu310/bin/activate" ]]; then
  echo "Invalid campaign repository ROOT=$ROOT; required manifest, revision lock, campaign helper, and activation $ROOT/.venv-gpu310/bin/activate." >&2; exit 2
fi
cd "$ROOT"
LOCK=$LOCK_INPUT; [[ "$LOCK" = /* ]] || LOCK="$ROOT/$LOCK"
[[ -f "$LOCK" ]] || { echo "Revision lock is missing at ROOT=$ROOT: $LOCK" >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN must be exported to this Slurm job." >&2; exit 2; }
source .venv-gpu310/bin/activate
LOG_DIR="outputs/three_model_range_sweep/${MODEL_KEY}/logs/calibration"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/${SLURM_JOB_ID:?}.out") 2> >(tee "$LOG_DIR/${SLURM_JOB_ID}.err" >&2)
python scripts/three_model_campaign.py run-stage c --model-key "$MODEL_KEY" --lock "$LOCK" --job-id "$SLURM_JOB_ID"
sha256sum "$(.venv-gpu310/bin/python scripts/three_model_campaign.py lookup --model-key "$MODEL_KEY" --lock "$LOCK" | .venv-gpu310/bin/python -c 'import json,sys; print(json.load(sys.stdin)["calibration_path"])')" > "outputs/three_model_range_sweep/${MODEL_KEY}/SHA256SUMS"
