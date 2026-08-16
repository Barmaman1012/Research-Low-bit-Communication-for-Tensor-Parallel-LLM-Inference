#!/usr/bin/env bash
# Review/edit only. This script is never invoked by repository tests.
set -euo pipefail
MANIFEST=experiments/three_model_range_sweep.yaml
LOCK=experiments/three_model_range_sweep.revisions.json
ROOT=outputs/three_model_range_sweep
MODELS=(gemma2_27b llama2_13b mistral_nemo_12b)
CONFIGS=(full int4 random_bf16 selected_bf16 range_threshold_bf16-t1.0 range_threshold_bf16-t1.2 range_threshold_bf16-t1.4 range_threshold_bf16-t1.6 range_threshold_bf16-t1.8 range_threshold_bf16-t1.9 range_threshold_bf16-t2.0 range_threshold_bf16-t2.1 range_threshold_bf16-t2.3)
# Stage A is deliberately separate: set HF_TOKEN after accepting Gemma/Llama licenses.
python scripts/three_model_campaign.py resolve-revisions --manifest "$MANIFEST" --output "$LOCK"
for model in "${MODELS[@]}"; do
  smoke=$(sbatch --parsable experiments/slurm/stage_b_load_smoke.sbatch "$model" "$LOCK")
  calibration=$(sbatch --parsable --dependency=afterok:"$smoke" experiments/slurm/stage_c_calibrate.sbatch "$model" "$LOCK")
  validate=$(sbatch --parsable --dependency=afterok:"$calibration" experiments/slurm/stage_d_validate_analyze.sbatch "$model" "$LOCK")
  smoke_array=$(sbatch --parsable --dependency=afterok:"$validate" --array=0-12%1 experiments/slurm/stage_f_eval_smoke.sbatch "$model" "$LOCK")
  sbatch --dependency=afterok:"$smoke_array" --array=0-12%1 experiments/slurm/stage_g_eval_full.sbatch "$model" "$LOCK"
done
# Submit Stage H only after all three Stage-G arrays are known successful.
