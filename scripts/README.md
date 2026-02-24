# Solar Scripts

This directory contains utility scripts for running and analyzing Solar pipeline results.

## Scripts

### `run_kernelbench.sh`

Run the Solar CLI pipeline on kernelbench models.

**Usage:**
```bash
# Process all levels
./run_kernelbench.sh

# Process only level1
./run_kernelbench.sh level1

# Process level3 kernels 1-10
./run_kernelbench.sh level3 1 10

# Process level3 kernel 5 only
./run_kernelbench.sh level3 5

# Process single file by name
./run_kernelbench.sh level3/1_MLP.py

# Skip already processed models
./run_kernelbench.sh level1 --skip-existing

# Show help
./run_kernelbench.sh --help
```

**Options:**
- `--skip-existing` - Skip models that already have output
- `--timeout SEC` - Timeout per model in seconds (default: 300)
- `--precision PRE` - Precision for analysis: fp32, fp16, bf16, int8 (default: fp32)
- `--arch ARCH` - Architecture config name (default: H100_PCIe)
- `--debug` - Enable debug output

**Output Structure:**
```
output_kernelbench/
└── level1/
    └── 1_Square_matrix_multiplication_/
        ├── graph/
        │   ├── pytorch_graph.yaml
        │   └── torchview_graph.pdf
        ├── einsum/
        │   ├── einsum_graph.yaml
        │   ├── einsum_graph_renamed.yaml
        │   └── einsum_graph.pdf
        ├── analysis/
        │   └── analysis.yaml
        ├── perf/
        │   └── perf_H100_PCIe.yaml
        └── timeloop/
            └── timeloop_graph.yaml
```

---

### `run_kernelbench_perf.sh`

Re-run specific pipeline phases on existing kernelbench results. Useful when you want to re-run certain steps without re-running the entire pipeline, or to change precision/architecture settings.

**Prerequisites:** Required input files must exist from previous runs (see Phases table below).

**Usage:**
```bash
# Process all levels with default settings (perf phase only)
./run_kernelbench_perf.sh

# Process only level1 (perf phase only)
./run_kernelbench_perf.sh level1

# Process level3 kernels 1-10
./run_kernelbench_perf.sh level3 1 10

# Run specific phases
./run_kernelbench_perf.sh --phase einsum level1           # Just einsum conversion
./run_kernelbench_perf.sh --phase analysis level1         # Just analysis
./run_kernelbench_perf.sh --phase perf level1             # Just perf prediction
./run_kernelbench_perf.sh --phase timeloop level1         # Just timeloop conversion

# Run multiple phases (comma-separated)
./run_kernelbench_perf.sh --phase einsum,analysis level1  # Einsum + analysis
./run_kernelbench_perf.sh --phase analysis,perf level1    # Analysis + perf

# Run all phases
./run_kernelbench_perf.sh --phase all level1              # All 5 phases

# Combine with other options
./run_kernelbench_perf.sh --phase perf --precision fp16 --arch A100 level1

# Run on specific kernel range
./run_kernelbench_perf.sh --phase einsum,analysis level1 1 10
```

**Options:**
- `--phase PHASES` - Comma-separated list of phases to run (default: perf)
- `--precision PRE` - Precision for analysis: fp32, fp16, bf16, int8 (default: fp32)
- `--arch ARCH` - Architecture config name (default: H100_PCIe)
- `--timeout SEC` - Timeout per step in seconds (default: 60)
- `--debug` - Enable debug output

**Available Phases:**

| Phase | Input Required | Output |
|-------|----------------|--------|
| `graph` | `kernelbench/*.py` | `graph/pytorch_graph.yaml` |
| `einsum` | `graph/pytorch_graph.yaml` | `einsum/einsum_graph_renamed.yaml` |
| `analysis` | `einsum/einsum_graph_renamed.yaml` | `analysis/analysis.yaml` |
| `perf` | `analysis/analysis.yaml` | `perf/perf_<arch>.yaml` |
| `timeloop` | `einsum/einsum_graph_renamed.yaml` | `timeloop/timeloop_graph.yaml` |
| `all` | Runs all phases in order | All outputs |

**Notes:**
- Phases will be skipped (with a warning) if their prerequisite files don't exist
- The `graph` phase requires the original model file in `kernelbench/`
- Multiple phases are executed in dependency order regardless of how they're specified

---

### `run_tests.sh`

Run tests for the Solar package.

**Usage:**
```bash
# Run all tests
./run_tests.sh

# Run specific test type
./run_tests.sh unit
./run_tests.sh integration
./run_tests.sh examples

# Verbose output
./run_tests.sh all -v
```

---

### `collect_perf_results.py`

Collect performance results from kernelbench runs into a CSV summary.

**Usage:**
```bash
# Default output: results_H100_PCIe.csv
python3 collect_perf_results.py

# Filter by level
python3 collect_perf_results.py --level level1

# Full output with all metrics
python3 collect_perf_results.py --full

# Different architecture (output: results_A100.csv)
python3 collect_perf_results.py --arch A100

# Custom output filename
python3 collect_perf_results.py --output custom_results.csv

# Custom output directory
python3 collect_perf_results.py --output-dir /path/to/output_kernelbench
```

**Options:**
- `--arch ARCH` - Architecture name (default: H100_PCIe)
- `--output FILE` - Output CSV file (default: results_<arch>.csv)
- `--level LEVEL` - Only collect from specific level (e.g., level1)
- `--output-dir DIR` - Output directory for kernelbench results
- `--simple` - Output simplified CSV (default)
- `--full` - Output full CSV with all metrics

**Simple Output Columns:**
| Column | Description |
|--------|-------------|
| `level` | Kernelbench level (level1, level2, etc.) |
| `kernel_id` | Kernel ID number |
| `kernel_name` | Kernel name |
| `sol_time_ms` | Unfused SOL runtime in milliseconds |
| `fused_sol_time_ms` | Fused SOL runtime in milliseconds |

**Full Output Columns (with `--full`):**
| Column | Description |
|--------|-------------|
| `level` | Kernelbench level |
| `kernel_id` | Kernel ID number |
| `kernel_name` | Kernel name |
| `total_macs` | Total multiply-accumulate operations |
| `total_flops` | Total floating-point operations |
| `unfused_memory_bytes` | Memory bytes for unfused execution |
| `unfused_runtime_ms` | Unfused SOL runtime (ms) |
| `unfused_bottleneck` | Bottleneck type (compute/memory) |
| `unfused_ai` | Arithmetic intensity (unfused) |
| `fused_memory_bytes` | Memory bytes for fused execution |
| `fused_runtime_ms` | Fused SOL runtime (ms) |
| `fused_bottleneck` | Bottleneck type (compute/memory) |
| `fused_ai` | Arithmetic intensity (fused) |
| `fused_prefetched_memory_bytes` | Memory bytes for fused+prefetched |
| `fused_prefetched_runtime_ms` | Fused+prefetched SOL runtime (ms) |
| `fused_prefetched_bottleneck` | Bottleneck type (compute/memory) |
| `fused_prefetched_ai` | Arithmetic intensity (fused+prefetched) |
| `speedup_fused_vs_unfused` | Speedup of fused over unfused |
| `speedup_fused_prefetched_vs_unfused` | Speedup of fused+prefetched over unfused |

---

## SOL Performance Models

The scripts use three SOL (Speed-of-Light) roofline models:

1. **Unfused (`sol_time_ms`)**: Each operation runs in isolation, all tensors accessed from DRAM
2. **Fused (`fused_sol_time_ms`)**: Intermediate tensors excluded from memory cost (assumed cached)
3. **Fused+Prefetched**: Single roofline for entire graph with perfect overlap

See [SOL_Guide.md](../SOL_Guide.md) for detailed explanation.

---

## Examples

### Run all level1 kernels and collect results
```bash
# Run the pipeline
./run_kernelbench.sh level1 --skip-existing

# Collect results (outputs results_H100_PCIe.csv)
python3 collect_perf_results.py --level level1
```

### Compare performance across levels
```bash
# Run all levels
./run_kernelbench.sh --skip-existing

# Collect all results with full metrics
python3 collect_perf_results.py --full
```

### Analyze specific kernel range
```bash
# Run kernels 1-20 in level1
./run_kernelbench.sh level1 1 20

# Collect results for different architectures
python3 collect_perf_results.py --arch H100_PCIe  # outputs results_H100_PCIe.csv
python3 collect_perf_results.py --arch A100       # outputs results_A100.csv
```

### Re-run specific phases
```bash
# Re-run just the analysis phase on level1
./run_kernelbench_perf.sh --phase analysis level1

# Re-run einsum conversion and analysis for a specific kernel
./run_kernelbench_perf.sh --phase einsum,analysis level1 19

# Re-run perf with different precision
./run_kernelbench_perf.sh --phase perf --precision fp16 level1
./run_kernelbench_perf.sh --phase analysis,perf --precision fp16 level1 19 
# Re-run all phases for kernels that failed
./run_kernelbench_perf.sh --phase all level1 1 10
```
