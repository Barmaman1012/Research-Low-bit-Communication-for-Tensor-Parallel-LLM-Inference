#!/usr/bin/env bash
# Submit only after reviewing this file and accepting the gated-model licenses.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
MANIFEST=experiments/three_model_range_sweep.yaml
LOCK="$REPO_ROOT/experiments/three_model_range_sweep.revisions.json"
MODELS=(gemma2_27b llama2_13b mistral_nemo_12b)
PYTHON_BIN=${PYTHON_BIN:-.venv-gpu310/bin/python}
DRY_RUN=false
LAST_STAGE=""

usage() {
  cat <<'EOF'
Usage: experiments/slurm/submit_three_model_range_sweep.sh [--dry-run] (--stage-only b | --through-stage c | --through-stage f | --all)

  --stage-only b    submit only locked model-loading/discovery smoke jobs
  --through-stage c submit B then calibration C after B succeeds
  --through-stage f submit B -> C -> D -> smoke-array F after successful dependencies
  --all             submit B -> C -> D -> F -> full-array G after successful dependencies
  --dry-run         print the exact sbatch commands; do not call sbatch or write a receipt

HF_TOKEN must already be exported in the submitting shell.  sbatch receives it
through --export=ALL; the token is never placed in a command line, file, or receipt.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --stage-only) [[ ${2:-} == b ]] || { usage >&2; exit 2; }; LAST_STAGE=b; shift ;;
    --through-stage) case ${2:-} in c|f) LAST_STAGE=$2 ;; *) usage >&2; exit 2 ;; esac; shift ;;
    --all) LAST_STAGE=g ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$LAST_STAGE" ]] || { usage >&2; exit 2; }
[[ -f "$MANIFEST" && -f "$LOCK" ]] || { echo "Missing manifest or immutable revision lock." >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN must be exported before submission." >&2; exit 2; }

# Strictly validate every requested model/lock pair before any sbatch call.
for model in "${MODELS[@]}"; do
  "$PYTHON_BIN" scripts/three_model_campaign.py lookup --model-key "$model" --lock "$LOCK" >/dev/null
done

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
receipt="outputs/three_model_range_sweep/submission_receipts/${timestamp}.tsv"
submit() {
  local model=$1 stage=$2 dependency=$3; shift 3
  local -a command=(sbatch --parsable --export=ALL)
  [[ -n "$dependency" ]] && command+=("--dependency=afterok:${dependency}")
  # Slurm command-line options override the conservative defaults in wrappers.
  if [[ "$stage" != d ]]; then
    case "$model" in
      gemma2_27b) command+=(--mem=192G) ;;
      llama2_13b|mistral_nemo_12b) command+=(--mem=128G) ;;
    esac
  fi
  command+=("$@")
  if "$DRY_RUN"; then
    printf 'DRY-RUN model=%s stage=%s dependency=%s: ' "$model" "$stage" "${dependency:-none}"
    printf '%q ' "${command[@]}"; printf '\n'
    REPLY="DRYRUN-${model}-${stage}"
    return
  fi
  local job_id
  job_id=$("${command[@]}")
  printf '%s\t%s\t%s\t%s\n' "$job_id" "$model" "$stage" "${dependency:-none}" | tee -a "$receipt"
  REPLY=$job_id
}

if ! "$DRY_RUN"; then mkdir -p "$(dirname "$receipt")"; printf 'job_id\tmodel\tstage\tdependency\n' > "$receipt"; fi
for model in "${MODELS[@]}"; do
  submit "$model" b "" experiments/slurm/stage_b_load_smoke.sbatch "$model" "$LOCK"; b=$REPLY
  [[ "$LAST_STAGE" == b ]] && continue
  submit "$model" c "$b" experiments/slurm/stage_c_calibrate.sbatch "$model" "$LOCK"; c=$REPLY
  [[ "$LAST_STAGE" == c ]] && continue
  submit "$model" d "$c" experiments/slurm/stage_d_validate_analyze.sbatch "$model" "$LOCK"; d=$REPLY
  submit "$model" f "$d" --array=0-12%1 experiments/slurm/stage_f_eval_smoke.sbatch "$model" "$LOCK"; f=$REPLY
  [[ "$LAST_STAGE" == f ]] && continue
  submit "$model" g "$f" --array=0-12%1 experiments/slurm/stage_g_eval_full.sbatch "$model" "$LOCK"
done
[[ "$DRY_RUN" == true ]] || echo "Submission receipt: $receipt"
