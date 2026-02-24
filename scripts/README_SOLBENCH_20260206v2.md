# SOL-bench-20260206v2 Pipeline Guide

Complete guide for running the SOLAR pipeline on the SOL-bench-20260206v2 dataset to measure speed-of-light (SOL) execution limits for neural network models.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Expected Results](#expected-results)
6. [Output Structure](#output-structure)
7. [Troubleshooting](#troubleshooting)
8. [Technical Details](#technical-details)

---

## Overview

### What is SOLAR?

SOLAR (Speed-Of-Light Analysis and Roofline) is a 5-phase pipeline that analyzes PyTorch models to predict their theoretical execution limits on GPU architectures:

1. **Phase 1 - Graph Generation**: Extract compute graph from PyTorch model
2. **Phase 2 - Einsum Conversion**: Convert graph to einsum representation
3. **Phase 3 - Analysis**: Compute operation counts and memory statistics
4. **Phase 4 - Performance Prediction**: Calculate SOL metrics using roofline model
5. **Phase 5 - Timeloop Export**: Convert to Timeloop format for further analysis

### Dataset: SOL-bench-20260206v2

- **Source**: Collection of PyTorch model implementations from various benchmarks
- **Levels**:
  - **L1**: 125 simpler models (transformer blocks, attention mechanisms, etc.)
  - **L2**: 110 complex models (full transformers, MoE, Mamba, multimodal)
- **Format**: Converted from SolBench format to KernelBench format for torchview compatibility

---

## Prerequisites

### 1. Environment Setup

```bash
# Ensure you're in the SOLAR project directory
cd /path/to/llm4arch

# Install dependencies
uv pip install -r requirements.txt

# Install SOLAR package in editable mode (REQUIRED)
uv pip install -e ./solar
```

**Why install SOLAR package?**
The pipeline uses Python module imports like `solar.cli.process_model`, which requires the SOLAR package to be installed. The `-e` flag (editable mode) allows you to make changes to the code without reinstalling.

**Key dependencies**:
- `torch` - PyTorch framework
- `torchview` - Compute graph visualization
- `networkx` - Graph analysis
- `openai>=1.0.0` - LLM-based conversion (optional, for dataset preparation)

### 2. Dataset Location

The pipeline expects the converted dataset at:
```
llm4arch/solbench-20260206v2-postprocess-llm/
├── L1/
│   ├── 0000_model_name/
│   │   └── model.py
│   ├── 0001_model_name/
│   └── ...
└── L2/
    ├── 0000_model_name/
    └── ...
```

**Note**: If your dataset is in a different location, create a symbolic link:
```bash
cd llm4arch
ln -s /path/to/solbench-20260206v2-postprocess-llm .
```

---

## Dataset Preparation

### Optional: LLM-Based Conversion

If you need to convert raw SolBench files to KernelBench format:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Convert all levels
./scripts/convert_solbench_llm.sh

# Or convert specific level
./scripts/convert_solbench_llm.sh --level L1

# Test with a few files first
./scripts/convert_solbench_llm.sh --max-files 5 --dry-run
```

**Important**: The LLM converter includes critical fixes for:
- Preserving `torch.set_default_dtype()` settings (bfloat16/float16)
- Matching forward() signature with get_inputs() return values
- Handling global constants vs. input parameters

---

## Running the Pipeline

### Basic Usage

```bash
# Run complete pipeline on L1 subset
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --output-dir solar/output_solbench-20260206v2

# Run complete pipeline on L2 subset
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L2 \
  --output-dir solar/output_solbench-20260206v2

# Run both L1 and L2
uv run bash scripts/run_solbench-20260206v2.sh \
  --output-dir solar/output_solbench-20260206v2
```

### Advanced Options

```bash
# Custom architecture (default: B200)
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --arch H100_PCIe \
  --output-dir solar/output_custom

# Custom precision (default: BF16)
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --precision FP16 \
  --output-dir solar/output_fp16

# Increased timeout for large models (default: 300s)
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L2 \
  --timeout 600 \
  --output-dir solar/output_long

# Skip completed models (resume interrupted run)
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --skip-completed \
  --output-dir solar/output_solbench-20260206v2
```

### Monitoring Progress

The script provides real-time progress:
```
========================================
Processing: 0009_briaai_Bria-3.2_02_diffusion_timestep_embedding
========================================
  Step 1/5: Generating PyTorch graph...
  ✓ Step 1 completed
  Step 2/5: Converting to einsum graph...
  ✓ Step 2 completed
  Step 3/5: Analyzing einsum graph...
  ✓ Step 3 completed
  Step 4/5: Predicting performance...
  ✓ Step 4 completed
  Step 5/5: Converting to Timeloop format...
  ✓ Step 5 completed
✓ Model completed successfully (5/5 phases)
```

---

## Expected Results

### Success Rates (as of 2026-02-09)

| Level | Total Models | Successful | Success Rate | Notes |
|-------|-------------|------------|--------------|-------|
| **L1** | 125 | 122 | **97.6%** | High success rate for simpler models |
| **L2** | 110 | 93 | **84.5%** | Lower due to complex architectures |
| **Overall** | 235 | 215 | **91.5%** | Excellent coverage |

### L1 Expected Failures (3 models)

1. **0012** - `google_switch-xxl_58_moe_routing_top1_gating_top2_selection_dispatch_aggregate_mlp`
   - Reason: 160+ experts, very large MoE architecture
   - Expected: Timeout or memory issues

2. **0027** - `state-spaces_mamba-2.8b-slimpj_93_mamba_full`
   - Reason: Complex Mamba SSM architecture
   - Expected: Graph generation challenges

3. **0075** - `Qwen_QwQ-32B_76_grouped_query_cross_attention`
   - Reason: Large-scale attention mechanism
   - Expected: Timeout on analysis phase

### L2 Expected Failures (17 models)

**MoE Models** (8 models):
- Very large expert counts (>160 experts)
- Complex routing and dispatching logic
- Examples: DBRX, Switch Transformers, Arctic

**Mamba/SSM Models** (4 models):
- State-space model complexity
- Selective scan operations
- Examples: Mamba variants, Jamba hybrid

**Multimodal Models** (3 models):
- Vision-language integration
- Cross-modal attention
- Examples: Qwen2-VL, Phi-3-vision

**Other Complex Architectures** (2 models):
- Hybrid transformer variants
- Custom attention patterns

---

## Output Structure

For each successfully processed model, the output directory contains:

```
output_solbench-20260206v2/
└── L1/
    └── 0009_briaai_Bria-3.2_02_diffusion_timestep_embedding/
        ├── graph/
        │   └── pytorch_graph.yaml              # Phase 1: PyTorch compute graph
        ├── einsum/
        │   ├── einsum_graph.yaml               # Phase 2: Initial einsum graph
        │   └── einsum_graph_renamed.yaml       # Phase 2: Renamed dimensions
        ├── analysis/
        │   └── analysis.yaml                   # Phase 3: Op counts, memory stats
        ├── perf/
        │   └── perf_B200.yaml                  # Phase 4: SOL metrics (KEY OUTPUT)
        └── timeloop/
            └── timeloop_*.yaml                 # Phase 5: Timeloop format
```

### Key Output: SOL Metrics

The most important file is `perf/perf_B200.yaml`:

```yaml
model_name: "0009_briaai_Bria-3.2_02_diffusion_timestep_embedding"
total_layers: 9
total_macs: 293002240
total_memory_bytes: 4718592

# Speed-of-Light Metrics
unfused_runtime_ms: 0.1532        # Individual op execution
fused_runtime_ms: 0.0847          # Optimized fusion
speedup: 1.81                     # Fusion benefit

# Roofline Analysis
compute_bound_layers: 6
memory_bound_layers: 3
arithmetic_intensity: 62.1        # MACs/byte
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'solar'

**Symptom**: Error when running pipeline
```
ModuleNotFoundError: No module named 'solar'
```

**Solution**: Install SOLAR package in editable mode
```bash
cd llm4arch
uv pip install -e ./solar
```

#### 2. ImportError: circular import 'torch.utils'

**Symptom**: Error during graph generation phase
```
ImportError: cannot import name 'backcompat' from partially initialized module 'torch.utils'
```

**Solution**: Use `uv run` for correct environment activation
```bash
# ✓ Correct
uv run bash scripts/run_solbench-20260206v2.sh --level L1

# ✗ Wrong
bash scripts/run_solbench-20260206v2.sh --level L1
```

#### 2. Dtype Mismatch Errors

**Symptom**:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**Cause**: Model uses `torch.set_default_dtype()` but get_inputs() doesn't preserve it

**Solution**: Reconvert the model using the LLM converter (includes dtype fix)
```bash
export OPENAI_API_KEY="your-key"
./scripts/convert_solbench_llm.sh --level L1
```

#### 3. Timeout on Large Models

**Symptom**: Model killed after 300 seconds

**Solution**: Increase timeout for complex models
```bash
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L2 \
  --timeout 600
```

#### 4. Self-Loop Hang (Fixed)

**Symptom**: Einsum conversion hangs indefinitely (this was a bug, now fixed)

**Cause**: Bias_add layers created self-referencing cycles in graph

**Status**: ✅ Fixed in `solar/einsum/pytorch_to_einsum.py` (line 1126)

#### 5. Missing Input Files

**Symptom**:
```
Error: Input directory not found: llm4arch/solbench-20260206v2-postprocess-llm
```

**Solution**: Create symbolic link to dataset
```bash
cd llm4arch
ln -s /path/to/solbench-20260206v2-postprocess-llm .
```

---

## Technical Details

### Recent Fixes (2026-02-09)

#### 1. Self-Loop Bug Fix

**File**: `solar/einsum/pytorch_to_einsum.py` (line 1126)

**Problem**: When linear layers with bias were split into matmul + bias_add operations, the `_fix_split_connections()` function created self-loops where bias_add referenced itself as input.

**Fix**: Added condition to prevent self-referencing:
```python
# Before (buggy):
if inp in node_id_remap:
    new_inputs.append(node_id_remap[inp])

# After (fixed):
if inp in node_id_remap and node_id_remap[inp] != layer_id:
    new_inputs.append(node_id_remap[inp])
```

**Impact**:
- Pipeline success improved from 80% → 91.5%
- Eliminates infinite hangs in BFS renaming algorithm
- Graph structure now accurately represents model computation

**Accuracy**: ✅ Fix IMPROVES accuracy by correcting graph topology to match reality. SOL calculation methodology unchanged.

#### 2. Environment Fix

**File**: `scripts/run_solbench-20260206v2.sh` (5 locations)

**Change**: `python3` → `uv run python` for all 5 phases

**Reason**: Prevents circular import issues in torch.utils

#### 3. LLM Converter Enhancement

**File**: `solar/benchmark/solbench/llm_converter.py`

**Enhancement**: Added dtype preservation requirement to LLM prompt

**Benefit**: Ensures bfloat16/float16 models are converted correctly

### Pipeline Performance

**Typical Runtimes** (per model on CPU):
- Simple models (L1): 30-120 seconds
- Complex models (L2): 120-300 seconds
- Very large MoE: May timeout at 300s (use --timeout 600)

**Bottlenecks**:
- Phase 1 (Graph): torchview trace (compute-bound)
- Phase 2 (Einsum): Graph transformations (memory-bound for large graphs)
- Phase 3 (Analysis): Operation counting (fast)
- Phase 4 (SOL): Roofline calculations (fast)
- Phase 5 (Timeloop): Format conversion (fast)

### Architecture Specifications

**Default: B200**
```yaml
# configs/arch/B200.yaml
compute_throughput_tflops: 2500    # BF16 peak
memory_bandwidth_gbps: 8000         # HBM3e bandwidth
```

**Alternative: H100_PCIe**
```yaml
# configs/arch/H100_PCIe.yaml
compute_throughput_tflops: 1600    # BF16 peak
memory_bandwidth_gbps: 2000         # PCIe bandwidth
```

---

## Validation

### Quick Validation Test

```bash
# Test a known-good model
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --max-models 1 \
  --output-dir solar/output_validation

# Check for SOL metrics
ls solar/output_validation/L1/*/perf/perf_B200.yaml
```

**Expected output**: File exists with valid SOL metrics

### Full Validation

```bash
# Run complete L1 pipeline
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --output-dir solar/output_solbench-20260206v2

# Count successes
find solar/output_solbench-20260206v2/L1/*/perf/perf_B200.yaml 2>/dev/null | wc -l
```

**Expected**: ~122 files (97.6% of 125 models)

---

## Summary

**Quick Start**:
```bash
# 1. Setup
cd llm4arch
uv pip install -r requirements.txt
uv pip install -e ./solar
export OPENAI_API_KEY="your-key"  # Only if converting dataset

# 2. Run pipeline
uv run bash scripts/run_solbench-20260206v2.sh \
  --level L1 \
  --output-dir solar/output_solbench-20260206v2

# 3. Check results
ls solar/output_solbench-20260206v2/L1/*/perf/perf_B200.yaml
```

**Expected Success Rates**:
- L1: 97.6% (122/125)
- L2: 84.5% (93/110)

**Key Features**:
- ✅ Self-loop bug fixed (accurate graph structure)
- ✅ Dtype preservation (bfloat16/float16 support)
- ✅ UV environment (no circular imports)
- ✅ 300s timeout (handles complex models)

---

**Last Updated**: 2026-02-09
**Pipeline Version**: solbench-20260206v2
**Maintainer**: SOLAR Team
