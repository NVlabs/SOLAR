# SolBench V1 Pipeline

This document describes how to run the Solar analysis pipeline on **SolBench V1** benchmark files.

## Overview

SolBench V1 files are located in `benchmark/L1/`, `benchmark/L2/`, etc. They contain:
- `get_inputs()` - Function that returns input tensors
- `reference_backward()` or `ReferenceModel` - Reference implementation
- `launch_reference_implementation()` - Wrapper function

The pipeline converts these files to Solar-compatible format and runs the full analysis.

## Quick Start

```bash
cd solar/scripts

# Step 1: Convert benchmark files to Solar format
./convert_solbenchv1.sh

# Step 2: Run Solar analysis on converted models (default: B200 architecture)
./run_solbenchv1.sh
```

## Scripts

### 1. convert_solbenchv1.sh

Converts SolBench V1 benchmark files to Solar-compatible format.

**Usage:**
```bash
./convert_solbenchv1.sh [OPTIONS]

Options:
  --level LEVEL     Only convert specific level (L1, L2, L1-Quant, L2-Quant)
  --max-files N     Maximum number of files to convert per level
  --debug           Enable debug output
```

**Example:**
```bash
# Convert all files from all levels
./convert_solbenchv1.sh

# Convert only L1 files
./convert_solbenchv1.sh --level L1

# Convert first 10 files from each level
./convert_solbenchv1.sh --max-files 10
```

**Output:** `solbenchv1-postprocess/<level>/<filename>.py`

### 2. run_solbenchv1.sh

Run Solar CLI pipeline on converted SolBench V1 models.

**Default Configuration:**
- Architecture: **B200** (NVIDIA Blackwell)
- Precision: **bf16**

**Usage:**
```bash
./run_solbenchv1.sh [OPTIONS]

Options:
  --level LEVEL        Process only specific level (L1, L2, L1-Quant, L2-Quant)
  --index INDEX        Process model by index (e.g., 0130)
  --name NAME          Process model matching name pattern
  --max-models N       Maximum number of models to process
  --phase PHASES       Comma-separated phases: graph,einsum,analysis,perf,timeloop
  --arch ARCH          Architecture config (default: B200)
  --precision PRE      Precision (fp32, fp16, bf16) (default: bf16)
  --skip-existing      Skip already processed models
  --debug              Enable debug output
```

**Examples:**
```bash
# Process all models with B200 architecture
./run_solbenchv1.sh

# Process specific model by index (e.g., 0130_black-forest-labs_FLUX.1-dev_08_*)
./run_solbenchv1.sh --level L1 --index 0130

# Process models matching pattern
./run_solbenchv1.sh --name flux_context

# Process first 5 models
./run_solbenchv1.sh --max-models 5

# Run only performance prediction with H100 architecture
./run_solbenchv1.sh --phase perf --arch H100_PCIe
```

**Output:** `solar/output_solbenchv1/<level>/<model_name>/`

## Pipeline Phases

1. **graph** - Generate PyTorch graph from model
2. **einsum** - Convert to einsum graph
3. **analysis** - Analyze einsum graph (compute/memory stats)
4. **perf** - Predict performance using roofline model
5. **timeloop** - Convert to Timeloop format

## Example: Running the FLUX Context Embedding Model

```bash
# Convert and run the specific model mentioned in the plan
cd solar/scripts

# Convert all L1 files (or just filter later)
./convert_solbenchv1.sh --level L1

# Run analysis on the FLUX context embedding projection backward model
./run_solbenchv1.sh --level L1 --index 0130

# Output will be in:
# solar/output_solbenchv1/L1/0130_black-forest-labs_FLUX.1-dev_08_flux_context_embedding_projection_backward/
```

## Output Structure

```
solar/output_solbenchv1/
└── L1/
    └── 0130_black-forest-labs_FLUX.1-dev_08_flux_context_embedding_projection_backward/
        ├── source_*.py           # Converted source file
        ├── metadata.yaml         # Model metadata
        ├── graph/                 # PyTorch graph
        │   └── pytorch_graph.yaml
        ├── einsum/                # Einsum graph
        │   ├── einsum_graph.yaml
        │   └── einsum_graph_renamed.yaml
        ├── analysis/              # Analysis results
        │   └── analysis.yaml
        ├── perf/                  # Performance predictions
        │   └── perf_B200.yaml
        └── timeloop/              # Timeloop format
            └── timeloop_graph.yaml
```

## Supported Architectures

- **B200** (default) - NVIDIA Blackwell
- **H100_PCIe** - NVIDIA H100 PCIe
- **H100_SXM5** - NVIDIA H100 SXM5
- **A100_SXM4** - NVIDIA A100 SXM4
- Custom configs in `solar/configs/arch/`

## Notes

- The default architecture is **B200** (NVIDIA Blackwell) for SolBench V1
- Files must have `get_inputs()` and either `reference_backward()` or `ReferenceModel` to be converted
- The converter wraps `reference_backward()` in a `Model` class for Solar compatibility
