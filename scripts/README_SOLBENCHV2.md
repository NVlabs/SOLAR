# SolBench v2 for Solar Analysis

This document describes how to run Solar analysis on SolBench v2 kernels from `sol-bench/data/benchmark`.

## Overview

SolBench v2 is a collection of fused kernel opportunities organized by complexity level:
- **L1**: Level 1 kernels (simpler operations like projections, norms)
- **L2**: Level 2 kernels (more complex fused operations, full layers)
- **Quant**: Quantized kernels (FP8, NVFP4, etc.)

The Solar pipeline analyzes these kernels to:
1. Convert to einsum representation
2. Calculate compute and memory requirements
3. Predict performance on different architectures (default: B200)

## Prerequisites

1. **sol-bench directory**: Clone or download SolBench to `<repo_root>/sol-bench/`
   ```
   sol-bench/
   └── data/
       └── benchmark/
           ├── L1/       (116 kernels)
           ├── L2/       (109 kernels)
           └── Quant/    (51 kernels)
   ```

2. **Python 3.10+**: Required for type hint compatibility
3. **Solar installation**: Ensure Solar is properly installed with all dependencies

## Directory Structure

```
solar/
├── scripts/
│   ├── check_solbenchv2_kernels.py  # Check kernel validity
│   ├── run_solbenchv2.sh            # Run Solar analysis pipeline
│   └── README_SOLBENCHV2.md         # This file
├── solar/benchmark/solbenchv2/
│   ├── postprocess.py               # Postprocess kernels for compatibility
├── solar/
│   └── benchmark/
│       └── solbenchv2/              # SolBench v2 support modules
│           ├── __init__.py
│           ├── parser.py            # Parse kernel files
│           └── converter.py         # Convert to Solar format
└── output_solbenchv2/               # Output directory (created automatically)
    ├── L1/
    │   └── <kernel_name>/
    │       ├── source_<kernel_name>.py
    │       ├── metadata.yaml
    │       ├── graph/
    │       ├── einsum/
    │       ├── analysis/
    │       ├── perf/
    │       └── timeloop/
    ├── L2/
    │   └── ...
    └── Quant/
        └── ...
```

## Quick Start

### Step 1: Check Kernel Validity (Optional)

Before running the full pipeline, you can check which kernels have all required methods:

```bash
cd solar/scripts

# Check all kernels and output to CSV
python3 check_solbenchv2_kernels.py

# Check specific level
python3 check_solbenchv2_kernels.py --level L1
```

This outputs `sol-bench-v2.csv` with the status of each kernel:
- `has_get_inputs`: Whether `get_inputs()` function exists
- `has_ReferenceModel`: Whether `ReferenceModel` class exists
- `has_reference_backward`: Whether `reference_backward()` function exists
- `has_launch_reference_implementation`: Whether `launch_reference_implementation()` exists
- `is_valid`: Whether kernel has all required components

### Step 2: Run Solar Analysis

```bash
cd solar/scripts

# Process all levels with B200 (default)
./run_solbenchv2.sh

# Process only L1 kernels
./run_solbenchv2.sh --level L1

# Process only L2 kernels
./run_solbenchv2.sh --level L2

# Process only Quant kernels
./run_solbenchv2.sh --level Quant

# Process with different architecture
./run_solbenchv2.sh --level L1 --arch H100_PCIe
```

## Command Options

### check_solbenchv2_kernels.py

```
Options:
  --benchmark-dir DIR   Path to sol-bench/data/benchmark (auto-detected)
  --output FILE         Output CSV file (default: sol-bench-v2.csv)
  --level LEVEL         Check specific level: L1, L2, Quant, or all
```

### run_solbenchv2.sh

```
Options:
  --level LEVEL      Process specific level: L1, L2, Quant (default: all)
  --name NAME        Process kernels matching name pattern
  --max-models N     Maximum number of models to process
  --phase PHASES     Comma-separated phases: graph,einsum,analysis,perf,timeloop
  --skip-existing    Skip models that already have output
  --timeout SEC      Timeout per model in seconds (default: 300)
  --precision PRE    Precision for analysis: fp32, fp16, bf16 (default: bf16)
  --arch ARCH        Architecture config name (default: B200)
  --force            Force regeneration even if output exists
  --debug            Enable debug output
  -h, --help         Show help message
```

## Examples

```bash
# Process all kernels with B200
./run_solbenchv2.sh

# Process L1 kernels only
./run_solbenchv2.sh --level L1

# Process L2 kernels with H100
./run_solbenchv2.sh --level L2 --arch H100_PCIe

# Process Quant kernels with FP16 precision
./run_solbenchv2.sh --level Quant --precision fp16

# Process first 10 kernels for testing
./run_solbenchv2.sh --max-models 10

# Process kernels with 'attention' in name
./run_solbenchv2.sh --name attention

# Process only L1 mamba kernels
./run_solbenchv2.sh --level L1 --name mamba

# Re-run perf phase with different architecture
./run_solbenchv2.sh --level L1 --phase perf --arch H100_SXM
```

## Automatic Postprocessing

The pipeline automatically postprocesses kernel files to ensure Solar compatibility:

### Case 1: Remove `device=` Specifications
Removes hardcoded device specifications like `device="cuda"` from tensor/model creation, allowing Solar to run on CPU for graph extraction.

### Case 2: Replace Triton `_fused_fma` with PyTorch
Replaces Triton-based `_fused_fma` kernels with pure PyTorch equivalent:
```python
def _fused_fma(y, x, s, BLOCK=128):
    """Fused multiply-add: y = y + x * s (in-place)"""
    y.add_(x * s)
    return y
```

### Output Files
When postprocessing makes changes:
- `orig_<kernel_name>.py` - Original unmodified file
- `src_<kernel_name>.py` - Postprocessed file (used for analysis)

When no changes needed:
- `src_<kernel_name>.py` - Copy of original (used for analysis)

### Standalone Postprocessing
You can run postprocessing separately:
```bash
# Process single file
python3.10 solar/solar/benchmark/solbenchv2/postprocess.py --file <input.py> --output-dir <output_dir>

# Process entire benchmark directory
python3.10 solar/solar/benchmark/solbenchv2/postprocess.py --input-dir sol-bench/data/benchmark --output-dir output/ --level Quant

# Generate CSV summary
python3.10 solar/solar/benchmark/solbenchv2/postprocess.py --input-dir sol-bench/data/benchmark --output-dir output/ --summary results.csv
```

## Kernel Requirements

For a kernel to be processed by Solar, it must have:

1. **`get_inputs()` function**: Returns input tensors for the model
2. **Reference implementation**: Either:
   - `ReferenceModel` class with a `forward()` method, OR
   - `reference_backward()` function for backward pass kernels
3. **`launch_reference_implementation()` function**: Driver function

The converter automatically wraps `ReferenceModel` or `reference_backward` into a Solar-compatible `Model` class.

## Output Files

For each processed kernel, the following files are generated:

| Directory | Contents |
|-----------|----------|
| `graph/` | `pytorch_graph.yaml` - PyTorch computation graph |
| `einsum/` | `einsum_graph.yaml`, `einsum_graph_renamed.yaml`, `einsum_graph.pdf` |
| `analysis/` | `analysis.yaml` - Compute/memory statistics |
| `perf/` | `perf_<arch>.yaml` - Performance predictions |
| `timeloop/` | `timeloop_graph.yaml` - Timeloop-compatible format |

## Supported Architectures

The default architecture is **B200**. Other supported architectures include:
- `H100_PCIe`
- `H100_SXM`
- `A100_PCIe`
- `A100_SXM`

## Troubleshooting

### Kernel Conversion Failed
If a kernel is skipped with "kernel conversion failed", it's missing required methods.
Run `check_solbenchv2_kernels.py` to see which methods are missing.

### Timeout Errors
For complex kernels, increase the timeout:
```bash
./run_solbenchv2.sh --timeout 600
```

### Memory Issues
For large models, try processing one level at a time:
```bash
./run_solbenchv2.sh --level L1
./run_solbenchv2.sh --level L2
./run_solbenchv2.sh --level Quant
```
