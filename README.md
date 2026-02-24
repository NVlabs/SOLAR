# Solar: PyTorch Model Analysis Toolkit

Solar is a toolkit for analyzing PyTorch model graphs, converting them to einsum representations, and performing hardware-aware SOL performance predictions.

## Features

- **5-Stage Analysis Pipeline**: Seamless conversion from PyTorch models to performance predictions
- **Graph Extraction**: Extract structured computation graphs from PyTorch models (torchview-based)
- **Einsum Conversion**: Convert PyTorch operations to einsum notation with automatic rank renaming
- **Graph Visualization**: Generate PDF visualizations of einsum graphs
- **Hardware-Independent Analysis**: Compute MACs, FLOPs, and memory footprints
- **Performance Prediction**: Architecture-aware roofline modeling (H100, A6000, etc.)
- **Timeloop Export**: Convert to Timeloop workload format for architectural exploration
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
| `DenseAttention/` | Full attention matrix | 
| `SlidingWindowAttention/` | Local window attention | 
| `RandomAttention/` | Random sparse attention |
| `BlockSparseAttention/` | Block-sparse attention |

### Running an Example

```bash
# Run the complete pipeline for any example
cd solar/examples/DenseAttention
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
solar-toeinsum --level level1 --kernel-ids 1 --arch-config A6000
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

### Benchmark Processing

```bash
# All-in-one: process + analyze
solar-toeinsum --level level1 --kernel-ids 1 2 3

# Show kernel status
solar-toeinsum --kernel-status --level level1

# Collect operation statistics
solar-toeinsum --level level1 --collect-stats

# List available analyses
solar-toeinsum --list-analyses
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
