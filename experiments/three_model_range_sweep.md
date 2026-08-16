# Three-model range sweep

Models: Gemma 2 27B, Llama 2 13B, and Mistral-Nemo Base 2407; all use llama targets, BF16, and TP8 numerical simulation. Gemma and Llama require an authenticated Hugging Face account with accepted model licenses; export `HF_TOKEN` before Stage A.

Run Stage A with `python scripts/three_model_campaign.py resolve-revisions`. It writes immutable commit IDs to `experiments/three_model_range_sweep.revisions.json`; do not submit later stages until that lock exists and is reviewed. The campaign never uses floating `main` revisions.

The submission script documents A–H dependencies and `%1` evaluation arrays. It is intentionally not submitted by this repository. Outputs are new under `outputs/three_model_range_sweep/`; old results remain untouched. This is a single-GPU numerical simulation, not NCCL/packed transport measurement.
