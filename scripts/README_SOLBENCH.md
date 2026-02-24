# SolBench for Solar Analysis

This document describes how to run Solar analysis on SolBench models.

## Overview

SolBench is a collection of fused kernel opportunities from real LLM models. Each model represents a subgraph that could benefit from kernel fusion. The Solar pipeline analyzes these models to:

1. Convert to einsum representation
2. Calculate compute and memory requirements
3. Predict performance on different architectures

## Prerequisites

1. **sol-bench directory**: Clone or download SolBench to `<repo_root>/sol-bench/`
2. **Solar installation**: Ensure Solar is properly installed with all dependencies

## Directory Structure

```
sol-bench/
├── data/
│   └── sample/
│       ├── 0000_inclusionAI-Ling-flash-2_0_19_mtp_embedding_hidden_concat_project.py
│       ├── 0001_baidu-ERNIE-4_5-300B-A47B-PT_15_attention_qk_matmul_with_scaling_backward.py
│       └── ... (200 models)

solbench-postprocess/                # Converted models (KernelBench format)
├── 0000_inclusionAI-Ling-flash-2_0_19_mtp_embedding_hidden_concat_project.py
├── 0001_baidu-ERNIE-4_5-300B-A47B-PT_15_attention_qk_matmul_with_scaling_backward.py
└── ...

solar/
├── scripts/
│   ├── convert_solbench.sh   # Step 1: Convert to KernelBench format
│   └── run_solbench.sh       # Step 2: Run Solar analysis
├── solar/
│   └── benchmark/
│       └── solbench/         # SolBench support modules
│           ├── __init__.py
│           ├── parser.py     # Parse SolBench model files
│           ├── generator.py  # Generate Solar-compatible files
│           ├── converter.py  # Convert to KernelBench format
│           └── cli.py        # CLI for model preparation
└── output_solbench/          # Output directory (created automatically)
    └── <model_name>/
        ├── source_<model_name>.py  # Converted model file
        ├── metadata.yaml           # Model metadata
        ├── graph/                  # PyTorch graph
        ├── einsum/                 # Einsum representation
        ├── analysis/               # Analysis results
        ├── perf/                   # Performance predictions
        └── timeloop/               # Timeloop format
```

## Quick Start

### Two-Step Process

```bash
cd solar/scripts

# Step 1: Convert SolBench models to KernelBench format
./convert_solbench.sh

# Step 2: Run Solar analysis
./run_solbench.sh
```

## Step 1: Convert SolBench Models

The converter transforms SolBench model files into KernelBench-compatible format with:
- `class Model(nn.Module)` wrapper
- `def get_inputs()` function with random tensors
- `def get_init_inputs()` function

### Conversion Options

```bash
# Convert all models (default)
./convert_solbench.sh

# Convert with debug output
./convert_solbench.sh --debug

# Convert first N models
./convert_solbench.sh --max-files 10

# Custom configuration
./convert_solbench.sh --batch 64 --seq-len 128 --hidden-size 8192
```

### Default Configuration

The converter uses the following default configuration:
```yaml
hidden_size: 6144
num_attention_heads: 64
num_key_value_heads: 8
head_dim: 96
intermediate_size: 19648
rms_norm_eps: 1e-05
dtype: bfloat16
batch: 32
seq_len: 32
```

## Step 2: Run Solar Analysis

### Basic Usage (Default: B200 architecture)

```bash
# Process all converted models with B200 architecture (default)
./run_solbench.sh

# Process a specific model by index
./run_solbench.sh --index 0000

# Process a specific model by name pattern
./run_solbench.sh --name mtp_embedding

# Process first N models
./run_solbench.sh --max-models 10
```

### Architecture Options

```bash
# Use B200 (default)
./run_solbench.sh --arch B200

# Use H100 PCIe
./run_solbench.sh --arch H100_PCIe

# Rerun only perf phase with different architecture
./run_solbench.sh --phase perf --arch H100_PCIe
```

### Phase Control

Available phases: `graph`, `einsum`, `analysis`, `perf`, `timeloop`

```bash
# Run only specific phases
./run_solbench.sh --phase graph,einsum
./run_solbench.sh --phase analysis,perf

# Skip existing results
./run_solbench.sh --skip-existing
```

### Preparation Only

```bash
# Only prepare model files without running Solar pipeline
./run_solbench.sh --generate-only

# Force regeneration of model files
./run_solbench.sh --force
```

### Debug Mode

```bash
# Enable verbose output
./run_solbench.sh --debug
```

## Example: Process Single Model

Process the first model (MTP embedding fusion):

```bash
# First convert (if not already done)
./convert_solbench.sh

# Then analyze
./run_solbench.sh --index 0000
```

This will:
1. Prepare the model file in `output_solbench/0000_inclusionAI-Ling-flash-2_0_19_mtp_embedding_hidden_concat_project/`
2. Generate PyTorch graph
3. Convert to einsum representation
4. Analyze compute/memory requirements
5. Predict performance on B200

## Output Files

For each processed model:

```
output_solbench/<model_name>/
├── source_<model_name>.py     # Converted Python model (KernelBench format)
├── metadata.yaml              # Model metadata
├── process.log                # Processing log
├── graph/
│   └── pytorch_graph.yaml     # PyTorch operation graph
├── einsum/
│   ├── einsum_graph.yaml      # Raw einsum graph
│   ├── einsum_graph_renamed.yaml  # Renamed graph
│   └── einsum_graph.pdf       # Graph visualization
├── analysis/
│   └── analysis.yaml          # Compute/memory analysis
├── perf/
│   ├── perf_B200.yaml         # B200 performance prediction
│   └── analysis.yaml          # Copy of analysis
└── timeloop/
    └── ...                    # Timeloop format files
```

## Model Metadata

Each SolBench model includes:

- **name**: Full model name
- **index**: Model index (0000-0199)
- **op_type**: Operation type (e.g., "fused_op")
- **priority**: Optimization priority (high/medium/low)
- **description**: What the model does
- **config**: Configuration constants
- **module_class**: PyTorch nn.Module class name
- **optimization_notes**: Fusion opportunities

## Troubleshooting

### Conversion errors
```
Error converting <file>: ...
```
- Check that the model file is valid Python
- Some models may have complex type hints - the converter adds `from __future__ import annotations` to handle these

### Model not found
```
Error: solbench-postprocess directory not found
```
Solution: Run `./convert_solbench.sh` first to convert the models.

### Graph generation fails
- Check that the model file is valid Python
- Check `process.log` for detailed errors
- Some models may have unsupported operations

### Performance prediction fails
- Ensure the analysis phase completed successfully
- Check that the architecture config exists in `configs/arch/`

## Default Configuration

- **Architecture**: B200
- **Precision**: bf16 (matching most LLM workloads)
- **Timeout**: 300 seconds per model

## Batch Processing

For large-scale analysis:

```bash
# Process all models, skip existing, with timeout
./run_solbench.sh --skip-existing --timeout 600

# Process in parallel (run multiple instances)
./run_solbench.sh --index 0000 &
./run_solbench.sh --index 0001 &
./run_solbench.sh --index 0002 &
wait
```

## Integration with Solar CLI

The SolBench modules can also be used directly:

```python
from solar.benchmark.solbench import SolBenchConverter

# Convert models programmatically
converter = SolBenchConverter(config={
    "batch": 32,
    "seq_len": 32,
    "hidden_size": 6144,
})

# Convert single file
converter.convert_file(
    input_path=Path("sol-bench/data/sample/0000_xxx.py"),
    output_path=Path("solbench-postprocess/0000_xxx.py")
)

# Convert directory
success, failed = converter.convert_directory(
    input_dir=Path("sol-bench/data/sample"),
    output_dir=Path("solbench-postprocess"),
    max_files=10
)
```

## Command Reference

### convert_solbench.sh

| Option | Description | Default |
|--------|-------------|---------|
| `--batch N` | Batch size for input tensors | 32 |
| `--seq-len N` | Sequence length | 32 |
| `--hidden-size N` | Hidden dimension size | 6144 |
| `--max-files N` | Maximum files to convert | all |
| `--debug` | Enable debug output | off |

### run_solbench.sh

| Option | Description | Default |
|--------|-------------|---------|
| `--index INDEX` | Process model by index | all |
| `--name NAME` | Process model by name pattern | all |
| `--max-models N` | Maximum models to process | all |
| `--arch ARCH` | Architecture config | B200 |
| `--precision PRE` | Precision (fp32, fp16, bf16) | bf16 |
| `--phase PHASES` | Phases to run (comma-separated) | all |
| `--skip-existing` | Skip processed models | off |
| `--timeout SEC` | Timeout per model | 300 |
| `--generate-only` | Only prepare files | off |
| `--force` | Force regeneration | off |
| `--debug` | Enable debug output | off |
