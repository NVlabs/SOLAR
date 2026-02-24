# FlashInfer Benchmark for Solar Analysis

This document describes how to run Solar analysis on FlashInfer benchmark models.

## Overview

FlashInfer benchmark provides traced workloads from real LLM inference scenarios including:
- **GQA (Grouped Query Attention)**: Paged and ragged attention kernels
- **MLA (Multi-head Latent Attention)**: Paged attention for DeepSeek-style models
- **MoE (Mixture of Experts)**: FP8 block-scaled MoE operations
- **GEMM**: Various matrix multiplication patterns
- **RMSNorm**: Normalization operations

The Solar pipeline analyzes these workloads to:
1. Convert to einsum representation
2. Calculate compute and memory requirements
3. Predict performance on different architectures

## Prerequisites

0. **Install Solar**: Follow the installation instructions in the [main README](../../README.md)
   ```bash
   # Clone the repository
   git clone ssh://git@gitlab-master.nvidia.com:12051/jennyhuang/llm4arch.git
   cd llm4arch

   # Install Solar in development mode
   cd solar
   pip install -e .
   ```

   Dependencies:
   ```bash
   # Core dependencies are in requirements.txt
   pip install -r requirements.txt

   # For graph visualization (optional)
   pip install graphviz matplotlib
   ```

1. **flashinfer-trace directory**: Unzip the flashinfer-trace.zip file in the llm4arch root
   ```bash
   cd /path/to/llm4arch
   unzip flashinfer-trace.zip
   ```

2. **flashinfer-trace-postprocess**: Generated model files (created by the scripts)

3. **Solar installation**: Ensure Solar is properly installed with all dependencies

## Directory Structure

```
flashinfer-trace/                    # Original trace data (from HuggingFace)
├── traces/
│   ├── gemm/
│   ├── gqa_paged/
│   ├── mla_paged/
│   └── ...

flashinfer-trace-postprocess/        # Generated model files
├── gemm/
│   └── gemm_n128_k2048/
│       ├── 0/                       # Workload row 0
│       │   ├── gemm_n128_k2048.py
│       │   └── uuid.yaml
│       ├── 1/                       # Workload row 1
│       └── ...
├── gqa_paged/
│   └── gqa_paged_decode_h32_kv4_d128_ps1/
│       └── ...
└── ...

solar/
├── scripts/
│   ├── run_flashinferbench.sh       # Main runner script
│   ├── run_flashinferbench_all.sh   # Run all benchmarks
│   └── parse_flashinferbench_results.py  # Parse and compare results
└── output_flashinferbench/          # Analysis output
    └── <op_type>/<definition>/<row_id>/
        ├── uuid.yaml
        ├── graph/
        ├── einsum/
        ├── analysis/
        ├── perf/
        └── timeloop/
```

## Quick Start

```bash
cd solar/scripts

# Step 1: Generate model files from traces
./run_flashinferbench.sh --generate-only

# Step 2: Run Solar analysis on all models
./run_flashinferbench.sh

# Or process specific operations
./run_flashinferbench.sh gemm
./run_flashinferbench.sh gqa_paged gqa_paged_decode_h32_kv4_d128_ps1
```

## Usage

### Basic Usage

```bash
# Process all operation types
./run_flashinferbench.sh

# Process specific op type
./run_flashinferbench.sh gemm
./run_flashinferbench.sh gqa_paged
./run_flashinferbench.sh mla_paged
./run_flashinferbench.sh moe

# Process specific definition
./run_flashinferbench.sh gqa_paged gqa_paged_decode_h32_kv4_d128_ps1

# Process specific workload rows (0 to 5)
./run_flashinferbench.sh gemm gemm_n128_k2048 0 5
```

### Architecture Options

```bash
# Use H100 PCIe (default)
./run_flashinferbench.sh --arch H100_PCIe

# Use B200
./run_flashinferbench.sh --arch B200

# Rerun only perf phase with different architecture
./run_flashinferbench.sh --phase perf --arch B200 gemm
```

### Phase Control

Available phases: `graph`, `einsum`, `analysis`, `perf`, `timeloop`

```bash
# Run only specific phases
./run_flashinferbench.sh --phase graph,einsum gemm
./run_flashinferbench.sh --phase analysis,perf gemm

# Skip existing results
./run_flashinferbench.sh --skip-existing
```

### Other Options

```bash
# Limit workloads per definition
./run_flashinferbench.sh --max-workloads 10

# Set timeout (seconds)
./run_flashinferbench.sh --timeout 600

# Set precision (fp32, fp16, bf16)
./run_flashinferbench.sh --precision fp16

# Enable debug output
./run_flashinferbench.sh --debug
```

## Parsing Results

After running the benchmark, compare Solar predictions with actual trace solutions:

```bash
# Parse results and create comparison CSV
python scripts/parse_flashinferbench_results.py --arch B200

# Specify custom paths
python scripts/parse_flashinferbench_results.py \
    --arch B200 \
    --output results.csv \
    --output-dir solar/output_flashinferbench \
    --traces-dir flashinfer-trace/traces
```

The output CSV contains:
- Solar predictions (unfused, fused, fused_prefetched runtime in ms)
- Actual measured latencies from trace solutions
- Ratio comparison (actual / predicted)

## Operation Types

### GQA Paged (`gqa_paged`)
Grouped Query Attention with paged KV cache. Supports decode and prefill modes.

Example: `gqa_paged_decode_h32_kv4_d128_ps1`
- 32 query heads, 4 KV heads
- 128 head dimension
- Page size 1

### GQA Ragged (`gqa_ragged`)
Grouped Query Attention with ragged (variable-length) sequences.

### MLA Paged (`mla_paged`)
Multi-head Latent Attention (DeepSeek-style) with paged KV cache.

Example: `mla_paged_decode_h16_ckv512_kpe64_ps1`
- 16 heads
- 512 compressed KV dimension
- 64 KPE dimension

### MoE (`moe`)
Mixture of Experts operations with FP8 quantization.

Example: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- FP8 block-scale quantization
- Top-8 routing
- 32 local experts

### GEMM (`gemm`)
General Matrix Multiplication with various shapes.

### RMSNorm (`rmsnorm`)
Root Mean Square Layer Normalization.

## Output Files

For each processed workload:

```
output_flashinferbench/<op_type>/<definition>/<row_id>/
├── uuid.yaml              # Workload UUID and axes
├── process.log            # Processing log
├── graph/
│   └── pytorch_graph.yaml
├── einsum/
│   ├── einsum_graph.yaml
│   └── einsum_graph_renamed.yaml
├── analysis/
│   └── analysis.yaml
├── perf/
│   └── perf_<arch>.yaml   # e.g., perf_B200.yaml
└── timeloop/
    └── ...
```

## Troubleshooting

### Missing math import error
If you see `NameError: name 'math' is not defined`, the model file is missing `import math`. This has been fixed in the generated files.

### Model generation fails
Ensure `flashinfer-trace` is properly cloned with git-lfs:
```bash
cd flashinfer-trace
git lfs pull
```

### Performance prediction fails
- Ensure the analysis phase completed successfully
- Check that the architecture config exists in `configs/arch/`

## Default Configuration

- **Architecture**: H100_PCIe
- **Precision**: fp16
- **Timeout**: 300 seconds per model

## Batch Processing

For large-scale analysis:

```bash
# Process all with skip existing
./run_flashinferbench.sh --skip-existing

# Process in parallel (different op types)
./run_flashinferbench.sh gemm &
./run_flashinferbench.sh gqa_paged &
./run_flashinferbench.sh mla_paged &
wait
```

## Command Reference

### run_flashinferbench.sh

| Option | Description | Default |
|--------|-------------|---------|
| `--generate-only` | Only generate model files | off |
| `--phase PHASES` | Phases to run (comma-separated) | all |
| `--arch ARCH` | Architecture config | H100_PCIe |
| `--precision PRE` | Precision (fp32, fp16, bf16) | fp16 |
| `--skip-existing` | Skip processed models | off |
| `--timeout SEC` | Timeout per model | 300 |
| `--max-workloads N` | Max workloads per definition | all |
| `--debug` | Enable debug output | off |

### parse_flashinferbench_results.py

| Option | Description | Default |
|--------|-------------|---------|
| `--arch ARCH` | Architecture config | B200 |
| `--output PATH` | Output CSV path | flashinferbench_comparison_<arch>.csv |
| `--output-dir PATH` | Solar output directory | solar/output_flashinferbench |
| `--traces-dir PATH` | FlashInfer traces directory | flashinfer-trace/traces |
