<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Solar: PyTorch Model Analysis Toolkit

Solar is a toolkit for analyzing PyTorch model graphs, converting them to einsum representations, and performing hardware-aware SOL performance predictions.

## Features

- **5-Stage Analysis Pipeline**: Seamless conversion from PyTorch models to performance predictions
- **Graph Extraction**: Extract structured computation graphs from PyTorch models (torchview-based)
- **Einsum Conversion**: Convert PyTorch operations to einsum notation with automatic rank renaming
- **Graph Visualization**: Generate PDF visualizations of einsum graphs
- **Hardware-Independent Analysis**: Compute MACs, FLOPs, and memory footprints
- **Performance Prediction**: Architecture-aware roofline modeling (H100, A6000, etc.)
- **Timeloop / Orojenesis Export**: Convert to Timeloop workload format for architectural exploration
- **Benchmark Support**: Native support for kernelbench benchmark suites
- **Human-Readable YAML**: All outputs use clean YAML without anchors/aliases

## Installation

```bash
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

## The 5-Stage Pipeline

Solar processes models through five distinct stages:

```
Stage 1: PyTorch Graph Extraction
  └─> pytorch_graph.yaml

Stage 2: Einsum Conversion + Rank Renaming
  └─> einsum_graph.yaml
  └─> einsum_graph_renamed.yaml
  └─> einsum_graph.pdf (optional)

Stage 3: Hardware-Independent Analysis
  └─> analysis.yaml

Stage 4: Performance Prediction
  └─> perf_<arch>.yaml

Stage 5: Timeloop Export (optional)
  └─> timeloop_graph.yaml
```

## Examples

Solar includes several example models demonstrating different attention patterns:

### Available Examples

| Example | Description | Based On |
|---------|-------------|----------|
| `Attention/` | Multi-head self-attention | Standard Transformer |
| `BERT/` | BERT-like encoder model | BERT architecture |

### Running an Example

```bash
# Run the complete pipeline for any example
cd solar/examples/Attention
bash run_solar.sh

# Outputs:
#   - output/graph/pytorch_graph.yaml           (Stage 1)
#   - output/einsum/einsum_graph.yaml           (Stage 2)
#   - output/einsum/einsum_graph_renamed.yaml   (Stage 2 - with BFS rank renaming)
#   - output/einsum/einsum_graph.pdf            (Stage 2 - visualization)
#   - output/analysis/analysis.yaml             (Stage 3)
#   - output/perf/perf_H100_PCIe.yaml           (Stage 4)
#   - output/timeloop/timeloop_graph.yaml       (Stage 5)
```

### Benchmark Suite (Kernelbench)

Process benchmark models:

```bash
# Process and analyze kernelbench models
solar-toeinsum --level level1 --kernel-ids 1 2 3

# Use different architecture
solar-toeinsum --level level1 --kernel-ids 1 --arch-config B200
```

## CLI Commands

### Single Model Processing

```bash
# Stage 1: Extract PyTorch graph
solar-process-model --model-file model.py --output-dir output/graph

# Stage 2: Convert to einsum (with optional PDF visualization)
solar-toeinsum-model --graph-path output/graph/pytorch_graph.yaml \
                     --output-dir output/einsum --no-copy-graph \
                     --save-graph

# Stage 3: Analyze (hardware-independent)
solar-analyze-model --einsum-graph-path output/einsum/einsum_graph_renamed.yaml \
                    --output-dir output/analysis

# Stage 4: Predict performance
solar-predict-perf-model --analysis-path output/analysis/analysis.yaml \
                         --output-dir output/perf --arch-config H100_PCIe

# Stage 5: Convert to Timeloop format
solar-totimeloop --einsum-graph-path output/einsum/einsum_graph_renamed.yaml \
                 --output-dir output/timeloop
```

## Output File Formats

All output files use **human-readable YAML** without anchors/aliases:

- **pytorch_graph.yaml**: Structured graph with layers, shapes, weights, connections
- **einsum_graph.yaml**: Einsum equations + shapes for each layer
- **einsum_graph_renamed.yaml**: Einsum graph with consistent dimension labels (BFS-based)
- **einsum_graph.pdf**: Visual representation of the computation graph
- **analysis.yaml**: Hardware-independent metrics (MACs, FLOPs, bytes)
- **perf_<arch>.yaml**: Architecture-specific performance predictions
- **timeloop_graph.yaml**: Timeloop workload format for architectural exploration

## Testing

```bash
# Run all tests
bash run_tests.sh

# Quick smoke tests
bash run_tests.sh quick

# Run specific test categories
bash run_tests.sh graph      # Graph processing tests
bash run_tests.sh einsum     # Einsum analyzer tests
bash run_tests.sh unit       # All unit tests
bash run_tests.sh integration # Integration tests

# Test examples
bash run_tests.sh examples   # Run all example scripts

# Test benchmark compatibility
bash run_tests.sh kernelbench

# Verbose output
bash run_tests.sh all -v
```

See `TESTING_GUIDE.md` for detailed testing documentation.

## Python API

### Graph Processing

```python
from solar.graph import PyTorchProcessor

processor = PyTorchProcessor()
processor.process_model_file("model.py", output_dir="outputs/my_model")
```

### Einsum Conversion

```python
from solar.einsum import PyTorchToEinsum

converter = PyTorchToEinsum()
einsum_graph = converter.convert(
    "outputs/my_model/pytorch_graph.yaml",
    "outputs/my_model",
    copy_graph=False  # Don't duplicate input graph
)
# Produces both einsum_graph.yaml and einsum_graph_renamed.yaml
```

### Graph Visualization

```python
from solar.einsum import EinsumGraphVisualizer

visualizer = EinsumGraphVisualizer()
visualizer.save_graph_pdf(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/einsum_graph.pdf"
)
```

### Analysis

```python
from solar.analysis import EinsumGraphAnalyzer

analyzer = EinsumGraphAnalyzer()
analysis = analyzer.analyze_graph(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model"
)
```

### Performance Prediction

```python
from solar.perf import EinsumGraphPerfModel

perf_model = EinsumGraphPerfModel()
perf = perf_model.predict(
    "outputs/my_model/analysis.yaml",
    "outputs/my_model",
    arch_config="H100_PCIe"
)
```

### Timeloop Export

```python
from solar.einsum import EinsumToTimeloop

converter = EinsumToTimeloop()
result = converter.convert(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/timeloop_graph.yaml"
)
```

## Architecture

```
solar/
├── solar/
│   ├── common/        # Shared types, constants, utilities (NoAliasDumper)
│   ├── graph/         # Stage 1: PyTorch graph extraction
│   ├── einsum/        # Stage 2: Einsum conversion, visualization, Timeloop export
│   ├── analysis/      # Stage 3: Hardware-independent analysis
│   ├── perf/          # Stage 4: Performance prediction
│   └── cli/           # Command-line interfaces
├── tests/             # Comprehensive test suite
├── examples/          # Example models (Attention, BERT, sparse attention variants)
└── configs/           # Architecture configs (H100_PCIe.yaml, A6000.yaml)
```

## Key Components

- **NoAliasDumper**: Custom YAML dumper for human-readable output (no `&id001` references)
- **EinsumRankRenamer**: BFS-based dimension label renaming for consistent einsum equations
- **EinsumGraphVisualizer**: PDF visualization of computation graphs
- **EinsumToTimeloop**: Export to Timeloop workload format
- **Node Registry**: Extensible registry for operation handlers
- **LLM Agent** (optional): Dynamic handler generation for unknown operations
- **Benchmark Processors**: Specialized handling for kernelbench file structures

## Active Contributors

- [hqjennynv](https://github.com/hqjennynv)
- [sdamani-nvidia](https://github.com/sdamani-nvidia)
- [askiad](https://github.com/askiad)
- [LemonAndRabbit](https://github.com/LemonAndRabbit)

## Contributing

- Follow Google's Python Style Guide
- Add tests for new features
- Update documentation for API changes
- Run `bash run_tests.sh` before submitting PRs

## Documentation

- `TESTING_GUIDE.md`: Comprehensive testing documentation
- `REFACTORING_SUMMARY.md`: Design decisions and refactoring history
- `MIGRATION_COMPLETE.md`: Migration guide from legacy JSON format

## License

MIT License

## Reproducing the SOL-ExecBench results

The leaderboard SOL table (`sol_latency_ms` in SOL-ExecBench's `sol_latencies.csv`) is produced by the
scripts under `scripts/` with the setup in `configs/execbench/leaderboard_b200.yaml`. Everything below
is recorded in each result's `sol_summary.json` (setup name, sha256, effective precision, arch), so a
number can be traced back to the policy that produced it.

**Setup.** Install Solar (`install.sh` also applies `patches/torchview-parameter-tensors.patch` to
torchview; the F.linear keyword-argument handling needs it) and check out SOL-ExecBench next to this
repo with its data downloaded (`SOL-ExecBench/data/benchmark/<subset>/<problem>/{definition.json,workload.jsonl}`).
Solar traces on the meta device / CPU; no GPU is needed.

**Pricing policy (`configs/execbench/leaderboard_b200.yaml`).** B200 arch config; memory bytes at each
tensor's own dtype (`bool` 1 B, fp8 1 B, bf16 2 B, fp32 4 B); the MAC rate class is inferred from the
problem's declared input dtypes (narrowest floating class wins, fp8/nvfp4 from the quant metadata), and
problems whose inputs are all float32 are priced at the **16-bit tensor-core rate** (`fp32_as: fp16`,
also the runner default): the optimized kernels of the fp32 attention/decoder problems run bf16 math,
and a SOL must stay below every implementation. Memory stays 4 B per fp32 element. Use `fp32_as: tf32`
or `fp32` for the TF32 or CUDA-core rate; per-problem overrides (`problem_overrides`, each with a
`reason`) are supported but none are used.

```bash
# One problem, one workload shape (writes out/execbench/<problem>/<uuid>/sol_summary.json)
python scripts/run_execbench_problem.py ../SOL-ExecBench/data/benchmark/L1/044_moe_expert_computation \
    --workload-index 0 --setup-config configs/execbench/leaderboard_b200.yaml

# Every problem, every shape, one at a time (a 60 GB host; RLIMIT_AS cap per run, GPU hidden)
python scripts/sweep_execbench.py --all-workloads --jobs 1 --mem-gb 55 --timeout 3600 \
    --setup-config configs/execbench/leaderboard_b200.yaml --out-root out/execbench_full

# Re-analyse with a changed converter/analyzer but the same traces (stage 1 is the expensive, policy-
# independent part): --reuse-from reuses <root>/<problem>/<uuid>/graph/pytorch_graph.yaml;
# --reuse-einsum also reuses the einsum graph; --in-process skips ~6 s of torch import per stage.
python scripts/sweep_execbench.py --all-workloads --jobs 1 --mem-gb 55 --timeout 3600 \
    --setup-config configs/execbench/leaderboard_b200.yaml --out-root out/execbench_v2 \
    --runner-args "--reuse-from out/execbench_full --in-process"

# Change only the pricing policy or the arch: re-run the perf stage alone (seconds per shape)
python scripts/reprice_execbench.py --from-root out/execbench_v2 --to-root out/execbench_v3 \
    --setup-config configs/execbench/leaderboard_b200.yaml

# Compare with / export the leaderboard table (later roots override earlier ones for the same shape)
python scripts/export_execbench_csv.py --ref ../SOL-ExecBench/.../sol_latencies.csv \
    --results-root out/execbench_full --results-root out/execbench_v3 -o out/execbench_v3/sol_latencies
```

`export_execbench_csv.py` writes `<prefix>_compare.csv` (reference columns plus Solar's fused /
unfused SOL, MACs, bytes, bottleneck, precision, setup, ratio and status) and `<prefix>_solar.csv`, a
drop-in replacement for `sol_latencies.csv` whose `sol_latency_ms` is Solar's fused SOL. The `fused`
SOL is the fully fused lower bound (each external byte read once, intermediates on chip); `unfused`
charges every op's inputs and outputs.

Reproducibility notes:
* Run sweeps sequentially (`--jobs 1`); a dozen parallel traces exhausted a 60 GB host.
* Traces of large shapes are slow (minutes to ~40 min for the biggest MLA prefill shape) but are
  reused by every later re-analysis; never re-trace to change a policy.
* One shape does not convert on a 60 GB host: FlashInfer-Bench/014 #11 (18 sequences, 291k pages;
  the trace is 1.5 GB / 1.4 M nodes). `export_execbench_csv.py` keeps the reference value for it and
  flags it in `sol_source`.
* `scripts/crosscheck_execbench_ref.py`, `judge_execbench_deviations.py` and
  `summarize_execbench_sweep.py` produce the per-shape comparison, verdicts and sweep summaries.
