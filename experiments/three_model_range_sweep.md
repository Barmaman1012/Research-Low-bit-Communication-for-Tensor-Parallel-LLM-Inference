# Three-model range sweep

## Executable workflow

The immutable revisions are in `three_model_range_sweep.revisions.json`; do
not replace them with `main`.  Gemma 2 and Llama 2 are gated on Hugging Face:
accept their licenses and export `HF_TOKEN` in the shell that invokes `sbatch`.
The submitter uses `--export=ALL`, so the scheduler inherits the token without
placing it in command lines, receipts, results, or repository files.

Review commands without submission:

```bash
export HF_TOKEN  # set securely in your shell; do not save it in a script
experiments/slurm/submit_three_model_range_sweep.sh --dry-run --all
```

Choose exactly one submission scope (receipts are saved under
`outputs/three_model_range_sweep/submission_receipts/`):

```bash
experiments/slurm/submit_three_model_range_sweep.sh --stage-only b
experiments/slurm/submit_three_model_range_sweep.sh --through-stage c
experiments/slurm/submit_three_model_range_sweep.sh --through-stage f
experiments/slurm/submit_three_model_range_sweep.sh --all
```

Each invocation is intentionally independent. Do not invoke a later command
after jobs from an earlier invocation are already queued: it creates a new
chain. Use `--all` once for the complete dependency graph. Stage B rejects a
non-H100 GPU, missing `o_proj`/`down_proj` boundaries, or peak allocated memory
above 85% of total device memory.  Gemma therefore cannot advance to
calibration until this safety smoke succeeds.

Models: Gemma 2 27B, Llama 2 13B, and Mistral-Nemo Base 2407; all use llama targets, BF16, and TP8 numerical simulation. Gemma and Llama require an authenticated Hugging Face account with accepted model licenses; export `HF_TOKEN` before Stage A.

Run Stage A with `python scripts/three_model_campaign.py resolve-revisions`. It writes immutable commit IDs to `experiments/three_model_range_sweep.revisions.json`; do not submit later stages until that lock exists and is reviewed. The campaign never uses floating `main` revisions.

The submission script documents A–H dependencies and `%1` evaluation arrays. It is intentionally not submitted by this repository. Outputs are new under `outputs/three_model_range_sweep/`; old results remain untouched. This is a single-GPU numerical simulation, not NCCL/packed transport measurement.
