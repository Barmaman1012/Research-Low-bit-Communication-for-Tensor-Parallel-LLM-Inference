# lowbit_tp_comm

Research scaffold for reproducing ideas from *Towards Low-bit Communication for Tensor Parallel LLM Inference* (arXiv:2411.07942).

## Project goal

The paper studies how to reduce the communication cost of tensor-parallel large language model inference by compressing activations exchanged between tensor-parallel shards. The core objective is to preserve model quality while lowering synchronization bandwidth.

## What tensor-parallel synchronization is

In tensor parallelism, a layer is split across multiple devices. Each device computes only part of the layer output, and the full result usually requires collective communication such as `all-gather`, `reduce-scatter`, or `all-reduce`. Those synchronization steps can become a bottleneck during inference, especially for large models and high-throughput serving.

## What this repository simulates first

This repository starts with a single-machine simulation rather than real multi-GPU distributed execution. The first step is to model the interfaces around:

- virtual tensor-parallel linear layers
- communication hooks
- calibration data structures
- quantization containers
- evaluation configuration

The immediate goal is correctness and clarity. We want a clean place to reason about compression behavior before introducing actual distributed systems concerns.

## Planned roadmap

The intended next phases are:

1. Calibration passes to collect activation statistics.
2. Selection of features that should remain in BF16.
3. Int4 communication quantization for the remaining synchronized values.
4. Evaluation on toy examples first, then real language-model workloads.

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
    run_toy_demo.py
  tests/
    test_quantization.py
    test_tp_linear.py
```

## Quick start

Create an environment with Python 3.10+ and install the requirements:

```bash
pip install -r requirements.txt
```

For local execution without packaging metadata, set `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/run_toy_demo.py
pytest
```

## Status

Only the initial skeleton is implemented. The current code provides module boundaries, dataclasses, placeholder abstractions, and a tiny demo path. It does not yet implement the paper's compression algorithm.
# Research-Low-bit-Communication-for-Tensor-Parallel-LLM-Inference
