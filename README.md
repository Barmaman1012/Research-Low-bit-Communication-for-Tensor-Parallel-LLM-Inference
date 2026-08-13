# lowbit_tp_comm

Research scaffold for reproducing ideas from *Towards Low-bit Communication for Tensor Parallel LLM Inference* (arXiv:2411.07942).

## Project goal

The paper studies how to reduce the communication cost of tensor-parallel LLM inference by compressing activations exchanged between tensor-parallel shards. The objective is to preserve model quality while lowering synchronization bandwidth.

## What tensor-parallel synchronization is

In tensor parallelism, a layer is split across multiple devices. Each device computes only part of the layer output, and the full result requires synchronization such as `all-gather`, `reduce-scatter`, or `all-reduce`. Those communication steps can become the bottleneck during inference.

## What this repository simulates first

This repository starts with a single-process simulation rather than true distributed multi-GPU execution. The code models:

- tensor-parallel row-split partial outputs
- Int4 communication quantization
- selected BF16 outlier features
- calibration from real model activations
- module replacement inside Hugging Face causal LMs

The focus is correctness and clarity before system-level optimization.

## Current reproduction levels

### Level 1: Toy linear layer

- Proves tensor-parallel split and summation behavior.
- Proves that preserving selected BF16 outlier features can reduce quantization error.

### Level 2: Tiny GPT-2

- Proves the hook, calibration, replacement, and perplexity-evaluation code path.
- Not scientifically meaningful because the hidden dimensions are tiny, so `k = floor(E / 64)` often becomes `1`, which inflates the effective average bits per value.

### Level 3: Larger causal LM

Recommended next tests:

- `gpt2`
- `distilgpt2`
- `facebook/opt-125m`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` if hardware allows

As feature dimension `E` grows, `k = floor(E / 64)` becomes closer to the paper's intended low selected-feature fraction and the effective average bits per value approaches the paper's target regime.

## Next recommended models

CPU / laptop:

- `distilgpt2`
- `gpt2`
- `facebook/opt-125m`

Closer to the paper's target architecture:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `HuggingFaceTB/SmolLM2-135M`
- `google/gemma-2-2b` if hardware and access allow
- `mistralai/Mistral-7B-v0.1` if GPU access allows

Llama, Gemma, and Mistral style models typically expose `self_attn.o_proj` and `mlp.down_proj`, which are closer to the paper's target synchronization points than GPT-2 style `c_proj` modules.

## Layout

```text
lowbit_tp_comm/
  README.md
  requirements.txt
  src/
    lowbit_tp_comm/
      __init__.py
      calibration.py
      config.py
      evaluation.py
      hooks.py
      quantization.py
      tp_linear.py
  scripts/
    calibrate_model.py
    eval_model.py
    eval_ppl_simulated.py
    inspect_calibration.py
    run_toy_demo.py
  tests/
    test_calibration.py
    test_hooks.py
    test_quantization.py
    test_replacements.py
    test_tp_linear.py
```

## Experiment workflow

Install and verify:

```bash
PYTHONPATH=src ./.venv/bin/pytest
```

Toy simulation:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_toy_demo.py
```

Calibrate tiny GPT-2:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/calibrate_model.py --model_name sshleifer/tiny-gpt2 --target_style gpt2 --num_sequences 8 --sequence_length 64 --output_path calibration.pt
```

Inspect calibration:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/inspect_calibration.py --calibration_path calibration.pt
```

Evaluate all modes:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/eval_ppl_simulated.py --model_name sshleifer/tiny-gpt2 --calibration_path calibration.pt --target_style gpt2 --modes full,int4,random_bf16,selected_bf16 --num_sequences 8 --sequence_length 64 --num_partitions 2
```

Calibrate a small Llama-style model:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/calibrate_model.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --target_style llama \
  --num_sequences 32 \
  --sequence_length 128 \
  --output_path calibration-smollm2.pt
```

Inspect:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/inspect_calibration.py \
  --calibration_path calibration-smollm2.pt
```

Evaluate:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/eval_ppl_simulated.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --calibration_path calibration-smollm2.pt \
  --target_style llama \
  --modes full,int4,random_bf16,selected_bf16 \
  --num_sequences 32 \
  --sequence_length 128 \
  --num_partitions 2
```

Calibrate `distilgpt2` with simulated row-parallel calibration:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/calibrate_model.py \
  --model_name distilgpt2 \
  --target_style gpt2 \
  --simulate_row_parallel_calibration \
  --num_partitions 2 \
  --num_sequences 128 \
  --sequence_length 128 \
  --output_path calibration-distilgpt2-tp2.pt
```

## Explicit execution dtype

`calibrate_model.py`, `eval_ppl_simulated.py`, and `eval_lm_harness.py` all
accept `--dtype auto|float32|float16|bfloat16`. Explicit values are passed to
the installed Transformers API through its supported `dtype` loading argument;
they are not implemented by loading FP32 then silently treating it as BF16.

For BF16 runs, model parameters, target-module inputs/partials, preserved
selected communication values, Int4-dequantized values, and replacement
outputs are BF16. Calibration EMA min/max values and Int4 scales intentionally
remain FP32 for numerical stability, but are computed from BF16 partials.
The calibration artifact records both facts and evaluation rejects an explicit
dtype mismatch. Analytical selected-feature bit reports use 16 bits only for a
BF16/FP16 path; FP32 selected values are reported as 32-bit values.

CUDA BF16 requests fail clearly on GPUs without native BF16 support rather than
falling back to FP32. CPU dtype-routing tests remain supported.

Empire AI smoke calibration (adjust the model name and output path as needed):

```bash
PYTHONPATH=src python scripts/calibrate_model.py \
  --model_name google/gemma-2-27b \
  --target_style llama \
  --dtype bfloat16 --device cuda --num_partitions 8 \
  --simulate_row_parallel_calibration \
  --num_sequences 8 --sequence_length 128 \
  --output_path calibration-gemma2-27b-tp8-bf16.pt
```

Empire AI smoke evaluation:

```bash
PYTHONPATH=src python scripts/eval_lm_harness.py \
  --model_name google/gemma-2-27b \
  --calibration_path calibration-gemma2-27b-tp8-bf16.pt \
  --target_style llama --mode selected_bf16 \
  --dtype bfloat16 --device cuda --num_partitions 8 \
  --tasks arc_easy --limit 10 --batch_size 1 \
  --output_path results-gemma2-27b-tp8-bf16-smoke.json
```

Inspect:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/inspect_calibration.py \
  --calibration_path calibration-distilgpt2-tp2.pt
```

Evaluate:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/eval_ppl_simulated.py \
  --model_name distilgpt2 \
  --calibration_path calibration-distilgpt2-tp2.pt \
  --target_style gpt2 \
  --modes full,tp_uncompressed,all_bf16,int4,random_bf16,selected_bf16 \
  --num_sequences 128 \
  --sequence_length 128 \
  --num_partitions 2 \
  --verbose_bits
```

## Paper-style benchmark evaluation

Install `lm-eval` first:

```bash
./.venv/bin/pip install lm-eval
```

Calibrate first with partition-aware calibration, then run a limited debug benchmark:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/eval_lm_harness.py \
  --model_name distilgpt2 \
  --calibration_path calibration-distilgpt2-tp2.pt \
  --target_style gpt2 \
  --mode selected_bf16 \
  --tasks arc_easy,boolq \
  --limit 10 \
  --batch_size 1
```

Run all modes on the same debug slice:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/eval_lm_harness.py \
  --model_name distilgpt2 \
  --calibration_path calibration-distilgpt2-tp2.pt \
  --target_style gpt2 \
  --modes full,int4,random_bf16,selected_bf16 \
  --tasks arc_easy,boolq \
  --limit 10 \
  --batch_size 1
```

`eval_lm_harness.py` applies replacements using the exact module names stored
in the calibration artifact. Its `--target_style` option is retained as
calibration provenance in result metadata; it does not retarget modules during
evaluation.

The original paper evaluates much larger models, including Gemma 2 27B, Llama 2 13B, and Mistral NeMo 12B across 8 devices. This repository remains a smaller single-process simulated reproduction of the communication idea rather than a direct systems-scale reproduction.

## Notes

## Experimental three-tier extension

This repository also contains an experimental extension, not part of the
original paper: selected BF16 features, a second Int8 tier, and Int4 for the
remaining features. It reuses an existing calibration artifact; it does not
change the original two-tier modes.

Original two-tier results belong under `results/mistral_nemo_tp8_random_chunks/`:

```bash
PYTHONPATH=src python scripts/eval_lm_harness.py --model_revision a4477a2f977929a969745b69bbd62e03043551a5 --tokenizer_revision a4477a2f977929a969745b69bbd62e03043551a5 --modes full,int4,random_bf16,selected_bf16 --output_path results/mistral_nemo_tp8_random_chunks/two_tier.json
```

Three-tier experimental results belong under `results/mistral_nemo_tp8_three_tier/`:

```bash
PYTHONPATH=src python scripts/eval_lm_harness.py --model_revision a4477a2f977929a969745b69bbd62e03043551a5 --tokenizer_revision a4477a2f977929a969745b69bbd62e03043551a5 --modes selected_bf16_int8,selected_bf16_random_int8 --int8_fraction 0.015625 --output_path results/mistral_nemo_tp8_three_tier/three_tier.json
```

- This is still a numerical communication simulation. Int4 values are stored in `torch.int8`; true bandwidth savings would require bit-packing.
- Calibration collected with `num_partitions=1` is currently repeated across simulated partitions during replacement. Real tensor-parallel calibration would need partition-specific statistics.
- The Hugging Face and `datasets` stacks may emit a Python 3.12 `resource_tracker` shutdown warning after successful runs. The current scripts still complete and save results correctly.
- Perplexity on small evaluation slices can behave strangely, especially when the slice is only tens or hundreds of sequences.
- The paper reports zero-shot task accuracy, not only perplexity, so perplexity alone is not enough to judge whether selected BF16 is working as intended.
- We should not conclude that `selected_bf16` fails until we run larger evaluations and benchmark tasks.

## Experimental global-threshold BF16

`threshold_bf16` is an equal-budget numerical simulation for comparing global
allocation against the fixed per-module selection. `selected_bf16` preserves
`E/64` features independently in each module; `threshold_bf16` preserves the
same total count across all calibrated modules, ranking each feature by its
aggregated range divided by that module's median range. It uses BF16 for the
global winners and the existing Int4 path for every other feature—there is no
Int8 tier. This models equal theoretical communication budgets, not packed
communication.

```bash
PYTHONPATH=src ./.venv/bin/python scripts/eval_lm_harness.py \
  --model_name mistralai/Mistral-Nemo-Instruct-2407 \
  --calibration_path calibration-mistral-nemo-tp8.pt \
  --target_style llama --mode threshold_bf16 --num_partitions 8
```

## Range-threshold experiments

`selected_bf16` keeps the fixed top `E/64` features independently in every module. Existing `threshold_bf16` instead uses the same total selected count globally, ranked by range divided by module median. The new `range_threshold_bf16` is different: it selects every feature whose range is at least `--bf16_range_threshold` times its own module median, so its BF16 count and theoretical bits/value emerge from the semantic threshold. `matched_low_range_bf16` is an equal-count negative control that protects the globally smallest normalized ranges.

These are numerical single-GPU simulations: they do not pack payloads, perform NCCL communication, or measure transport speed. Average bits/value is theoretical communicated payload size excluding metadata and packing overhead.

```bash
# Run separately for thresholds 1.0, 1.2, 1.3, 2.0, or 4.0; use distinct outputs.
PYTHONPATH=src python scripts/eval_lm_harness.py --modes range_threshold_bf16 \
  --bf16_range_threshold 2.0 --calibration_path calibration-mistral-nemo-tp8.pt \
  --output_path results/range-threshold-2.0.json

PYTHONPATH=src python scripts/eval_lm_harness.py --modes matched_low_range_bf16 \
  --bf16_range_threshold 2.0 --calibration_path calibration-mistral-nemo-tp8.pt \
  --output_path results/matched-low-2.0.json
```
