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

## Notes

- This is still a numerical communication simulation. Int4 values are stored in `torch.int8`; true bandwidth savings would require bit-packing.
- Calibration collected with `num_partitions=1` is currently repeated across simulated partitions during replacement. Real tensor-parallel calibration would need partition-specific statistics.
- The Hugging Face and `datasets` stacks may emit a Python 3.12 `resource_tracker` shutdown warning after successful runs. The current scripts still complete and save results correctly.
