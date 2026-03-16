# SOLAR Usage

This page contains the more detailed “how to use SOLAR” material that’s intentionally kept out of the top-level [`README.md`](../README.md).

## Installation

```bash
# From repo root
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

For PDF graph rendering, install Graphviz (the `dot` binary) on your system.

## The 5-Stage Pipeline

SOLAR processes models through five distinct stages:

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

Related guides:

- [`SOL_GUIDE.md`](SOL_GUIDE.md)
- [`EINSUM_GUIDE.md`](EINSUM_GUIDE.md)

## Examples

| Example | What it covers |
|---------|----------------|
| `examples/Attention/` | Multi-head self-attention |
| `examples/BERT/` | BERT-style encoder block |
| `examples/Conv2d/` | Convolution layers |
| `examples/Matmul/` | GEMM / matmul patterns |

Run any example end-to-end:

```bash
cd examples/Attention
bash run_solar.sh
```

## CLI reference (end-to-end)

### Single model processing

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

### Benchmark processing (Kernelbench)

```bash
# Process and analyze kernelbench models
solar-toeinsum --level level1 --kernel-ids 1 2 3

# Use different architecture
solar-toeinsum --level level1 --kernel-ids 1 --arch-config B200
```

## Output file formats

All outputs use **human-readable YAML** (no anchors/aliases).

- **pytorch_graph.yaml**: Structured graph with layers, shapes, weights, connections
- **einsum_graph.yaml**: Einsum equations + shapes for each layer
- **einsum_graph_renamed.yaml**: Einsum graph with consistent dimension labels (BFS-based)
- **einsum_graph.pdf**: Visual representation of the computation graph
- **analysis.yaml**: Hardware-independent metrics (MACs, FLOPs, bytes)
- **perf_<arch>.yaml**: Architecture-specific performance predictions
- **timeloop_graph.yaml**: Timeloop workload format for architectural exploration

## Testing

```bash
# Quick smoke tests
bash scripts/run_tests.sh quick

# Run all tests
bash scripts/run_tests.sh
```

See [`TESTING_GUIDE.md`](TESTING_GUIDE.md) for details.

## Python API (snippets)

Graph extraction:

```python
from solar.graph import PyTorchProcessor

processor = PyTorchProcessor()
processor.process_model_file("model.py", output_dir="outputs/my_model")
```

Einsum conversion:

```python
from solar.einsum import PyTorchToEinsum

converter = PyTorchToEinsum()
converter.convert(
    "outputs/my_model/pytorch_graph.yaml",
    "outputs/my_model",
    copy_graph=False,
)
```

Analysis:

```python
from solar.analysis import EinsumGraphAnalyzer

analyzer = EinsumGraphAnalyzer()
analyzer.analyze_graph(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model",
)
```

Performance prediction:

```python
from solar.perf import EinsumGraphPerfModel

perf_model = EinsumGraphPerfModel()
perf_model.predict(
    "outputs/my_model/analysis.yaml",
    "outputs/my_model",
    arch_config="H100_PCIe",
)
```

Timeloop export:

```python
from solar.einsum import EinsumToTimeloop

converter = EinsumToTimeloop()
converter.convert(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/timeloop_graph.yaml",
)
```

## Repo layout (high level)

```
.
├── solar/             # Core library (graph/einsum/analysis/perf/cli/benchmark)
├── examples/          # Example models
├── tests/             # Test suite
├── configs/           # Architecture configs (see configs/arch/)
├── scripts/           # Utilities + benchmark runners
└── docs/              # Guides and verification docs
```

