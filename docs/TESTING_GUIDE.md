# Solar Testing Guide

## Overview

The Solar package includes comprehensive tests that validate the entire **5-stage analysis pipeline** and support **kernelbench** benchmark models. All test outputs use **human-readable YAML** without anchors or aliases.

## Solar 5-Stage Pipeline

```
Stage 1: PyTorch Graph Extraction
  Input:  model.py
  Output: pytorch_graph.yaml
  
Stage 2: Einsum Conversion + Rank Renaming
  Input:  pytorch_graph.yaml
  Output: einsum_graph.yaml
          einsum_graph_renamed.yaml
          einsum_graph.pdf (optional)
  
Stage 3: Hardware-Independent Analysis
  Input:  einsum_graph_renamed.yaml
  Output: analysis.yaml
  
Stage 4: Performance Prediction
  Input:  analysis.yaml + arch config
  Output: perf_<arch>.yaml

Stage 5: Timeloop Export (optional)
  Input:  einsum_graph_renamed.yaml
  Output: timeloop_graph.yaml
```

## Test Structure

```
tests/
├── conftest.py                # Pytest fixtures and configuration
├── test_graph_processing.py   # Stage 1: PyTorch graph extraction
├── test_einsum_analyzer.py    # Stage 2: Einsum conversion
├── test_model_analyzer.py     # Stages 3-4: Analysis + performance
├── test_llm_agent.py          # LLM agent and node registry
├── test_standalone_bert.py    # Full pipeline on BERT example
└── test_integration.py        # End-to-end benchmark tests
```

## Running Tests

### Quick Start

```bash
# Run all tests (~4-5 minutes)
bash run_tests.sh

# Quick smoke tests (~1 minute)
bash run_tests.sh quick

# Run unit tests only (no integration)
bash run_tests.sh unit

# Run integration tests only
bash run_tests.sh integration

# Run example scripts
bash run_tests.sh examples
```

### Pipeline Stage Tests

```bash
# Stage 1: Graph extraction (pytorch_graph.yaml)
bash run_tests.sh graph

# Stage 2: Einsum conversion (einsum_graph.yaml, einsum_graph_renamed.yaml)
bash run_tests.sh einsum

# Stages 3-4: Analysis + performance (analysis.yaml, perf_*.yaml)
bash run_tests.sh model
```

### Component Tests

```bash
# LLM agent and node registry
bash run_tests.sh llm

# Standalone BERT example (full 5-stage pipeline)
bash run_tests.sh bert
```

### Benchmark Compatibility

```bash
# Test kernelbench models
bash run_tests.sh kernelbench

# Verbose output
bash run_tests.sh all -v
```

### Using Pytest Directly

```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/test_einsum_analyzer.py -v

# Run tests matching pattern
python3 -m pytest tests/ -k "kernelbench"
python3 -m pytest tests/ -k "Integration"

# With coverage
python3 -m pytest tests/ --cov=solar --cov-report=html
```

## Test Categories

### 1. Graph Processing Tests (Stage 1)
Tests PyTorch graph extraction to `pytorch_graph.yaml`:
- **TorchviewProcessor**: Core graph extraction using torchview
- **PyTorchProcessor**: Single-model processing with explicit paths
- **BenchmarkProcessor**: Batch processing for kernelbench
- RNN model handling with device fallback (meta → cpu)
- Parameter extraction (weights, biases, module args)

**Key Tests:**
- `test_process_graph`: End-to-end graph generation
- `test_generate_torchview_graph`: Torchview integration
- `test_is_rnn_model`: RNN detection and special handling

### 2. Einsum Analyzer Tests (Stage 2)
Tests einsum equation generation and conversion to `einsum_graph.yaml` and `einsum_graph_renamed.yaml`:
- **Dynamic einsum generation**: matmul, linear, conv (1D/2D/3D)
- **Reduction operations**: sum, mean, max, min, prod
- **Element-wise operations**: relu, sigmoid, add, mul
- **Attention operations**: scaled_dot_product_attention
- **Rank renaming**: BFS-based dimension label propagation
- **Compute cost**: MAC calculation for all operation types
- **Memory cost**: Element counting for orojenesis/fusion analysis

**Key Tests:**
- `test_matmul`: Dynamic matmul einsum (1D-4D)
- `test_conv2d`: Convolution einsum with stride/padding
- `test_torch_prod`: Product reduction support
- `test_full_model_analysis`: Complete model conversion

### 3. Model Analyzer Tests (Stages 3-4)
Tests hardware-independent analysis (`analysis.yaml`) and performance prediction (`perf_<arch>.yaml`):
- **EinsumGraphAnalyzer**: Compute MACs, FLOPs, orojenesis_bytes, fused_bytes
- **EinsumGraphPerfModel**: SOL roofline predictions
- **Architecture configs**: H100_PCIe, A6000, H100_fp32
- **LLM agent integration**: Dynamic handler generation for unknown ops
- **Node registry**: Extensible operation handler system

**Key Tests:**
- `test_analyze_graph`: Hardware-independent metrics
- `test_predict_performance`: Roofline modeling
- `test_unknown_node_handling`: LLM agent fallback

### 4. LLM Agent Tests
Tests dynamic operation handler generation:
- Agent configuration and initialization
- Code generation for unknown operations
- Handler validation and safety checks
- Caching mechanisms for generated handlers
- Node type registry operations

**Key Tests:**
- `test_agent_initialization`: Setup and config
- `test_generate_handler`: Dynamic code generation
- `test_handler_caching`: Cache persistence

### 5. Standalone BERT Tests
Tests the complete 5-stage pipeline on a real model:
- Full pipeline: model.py → pytorch_graph.yaml → einsum_graph.yaml → einsum_graph_renamed.yaml → analysis.yaml
- Multi-head attention handling
- Feed-forward network processing
- Embedding layer support

**Key Tests:**
- `test_bert_full_pipeline`: End-to-end BERT processing

### 6. Integration Tests
End-to-end tests with benchmark suites:
- **Kernelbench pipeline**: Full directory processing
- **Batch processing**: Multiple kernels at once

**Key Tests:**
- `test_full_kernelbench_pipeline`: Kernelbench end-to-end

### 7. Example Tests
Tests that all example scripts run successfully:
- **DenseAttention**: Full attention matrix computation
- **SlidingWindowAttention**: Local window attention
- **RandomAttention**: Random sparse attention
- **BlockSparseAttention**: Block-sparse attention patterns
- **Attention**: Multi-head self-attention
- **BERT**: Complete BERT-like model

## Model Compatibility

### Kernelbench Models
- **File Format**: `{kernel_id}_{name}.py` (e.g., `1_ResNet50.py`)
- **Directory Structure**: `kernelbench/level{N}/`
- **Output Structure**: `kernelbench_outputs/level{N}/{kernel_id}/`
- **Node Types**: PascalCase (e.g., `Conv2d`, `Linear`, `ReLU`)

### Compatibility Features
- Automatic name normalization (PascalCase ↔ lowercase)
- Flexible ID parsing (numeric and string)
- Mixed naming convention support
- Unified analysis pipeline for both benchmark types

### Output Files (All Stages)

Each kernel output directory contains:
```
level{N}/{kernel_id}/
├── pytorch_graph.yaml         # Stage 1 output
├── einsum_graph.yaml          # Stage 2 output
├── einsum_graph_renamed.yaml  # Stage 2 output (with BFS rank renaming)
├── einsum_graph.pdf           # Stage 2 output (optional visualization)
├── analysis.yaml              # Stage 3 output
├── perf_<arch>.yaml           # Stage 4 output
└── timeloop_graph.yaml        # Stage 5 output (optional)
```

All YAML files use **NoAliasDumper** for human readability (no `&id001` references).

## Test Data

### Sample Models
Tests create sample models dynamically following benchmark conventions:

```python
# Kernelbench-style model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.fc = nn.Linear(64 * 112 * 112, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_inputs():
    """Required function for Solar processing."""
    return [torch.randn(1, 3, 224, 224)]
```

### Fixtures
Common fixtures are defined in `conftest.py`:
- `sample_node_data`: Sample node information
- `sample_torchview_nodes`: Sample graph nodes
- `kernelbench_sample_path`: Path to test kernelbench model
- `tmp_path`: Pytest's built-in temporary directory

### Expected Output Formats

**pytorch_graph.yaml** (Stage 1):
```yaml
model_name: BERT
layers:
  Model.linear:
    type: linear
    node_class: FunctionNode
    input_shapes:
    - - 2
      - 16
      - 64
    output_shapes:
    - - 2
      - 16
      - 64
    weight_nodes:
    - weight
    - bias
    weight_shapes:
    - - 64
      - 64
    - - 64
    module_args:
      in_features: 64
      out_features: 64
    connections:
      inputs: []
      outputs: []
```

**einsum_graph_renamed.yaml** (Stage 2 - with BFS rank renaming):
```yaml
model_name: BERT
layers:
  start:
    type: start
    einsum_equation: ->ABC
    is_real_einsum: false
    is_einsum_supportable: false
    shapes:
      Output:
      - 2
      - 16
      - 64
    connections:
      inputs: []
      outputs:
      - Model.linear
  Model.linear:
    type: linear
    einsum_equation: ABC,DC->ABD
    is_real_einsum: true
    is_einsum_supportable: true
    shapes:
      Input:
      - 2
      - 16
      - 64
      Weight:
      - 64
      - 64
      Output:
      - 2
      - 16
      - 64
    connections:
      inputs:
      - start
      outputs: []
```

**analysis.yaml** (Stage 3):
```yaml
model_name: BERT
total:
  macs: 131072
  flops: 262144
  orojenesis_bytes: 24640
  fused_bytes: 16448
layers:
  Model.linear:
    macs: 131072
    flops: 262144
    orojenesis_bytes: 24640
    fused_bytes: 16448
```

**perf_H100_PCIe.yaml** (Stage 4):
```yaml
arch: H100_PCIe
precision: fp32
total:
  unfused_runtime_ms: 0.0012
  fused_runtime_ms: 0.0008
layers:
  Model.linear:
    unfused_runtime_ms: 0.0012
    fused_runtime_ms: 0.0008
```

## Adding New Tests

### Test Template

When adding new operation support or features:

```python
import pytest
from pathlib import Path
import yaml

class TestNewFeature:
    """Tests for new feature."""
    
    def test_stage1_graph_extraction(self, tmp_path):
        """Test graph extraction produces valid pytorch_graph.yaml."""
        from solar.graph import PyTorchProcessor
        
        # Create test model
        model_file = tmp_path / "model.py"
        model_file.write_text("...")
        
        # Process
        processor = PyTorchProcessor()
        success = processor.process_model_file(str(model_file), str(tmp_path))
        
        # Verify pytorch_graph.yaml exists and is valid
        graph_path = tmp_path / "pytorch_graph.yaml"
        assert graph_path.exists()
        
        with open(graph_path) as f:
            graph = yaml.safe_load(f)
            assert "layers" in graph
            assert "model_name" in graph
    
    def test_stage2_einsum_conversion(self, tmp_path):
        """Test einsum conversion produces valid einsum_graph.yaml."""
        from solar.einsum import PyTorchToEinsum
        
        # Create test pytorch_graph.yaml
        # ... convert it ...
        
        # Verify einsum_graph.yaml and einsum_graph_renamed.yaml format
        einsum_path = tmp_path / "einsum_graph.yaml"
        renamed_path = tmp_path / "einsum_graph_renamed.yaml"
        assert einsum_path.exists()
        assert renamed_path.exists()
        
        with open(renamed_path) as f:
            einsum_graph = yaml.safe_load(f)
            # Verify no YAML anchors/aliases
            content = renamed_path.read_text()
            assert "&id" not in content
            assert "*id" not in content
    
    def test_kernelbench_support(self):
        """Test feature with kernelbench models (PascalCase)."""
        # Test implementation
```

### Best Practices

1. **Test all pipeline stages** when adding new operations
2. **Verify YAML format**: No anchors/aliases (`&id001`, `*id001`)
3. **Test both model types** (kernelbench PascalCase)
4. **Use fixtures** for common test data and temporary directories
5. **Mock external dependencies** (e.g., LLM API calls) for unit tests
6. **Include integration tests** for end-to-end validation
7. **Check file outputs**: Verify expected files are created with correct structure
8. **Test error handling**: Include tests for invalid inputs and edge cases

### Testing New Operation Types

When adding support for a new PyTorch operation:

```python
def test_new_operation_einsum(self):
    """Test new operation einsum generation."""
    from solar.einsum import EinsumAnalyzer
    
    analyzer = EinsumAnalyzer()
    shapes = {"Input": [32, 64], "Weight": [128, 64]}
    
    # Test einsum generation
    einsum_op = analyzer.get_linear_einsum_op(shapes)
    assert einsum_op.equation == "BMK,NK->BMN"
    
    # Test compute cost
    cost = analyzer.get_compute_cost("Linear", shapes)
    assert cost == 32 * 128 * 64  # Expected MACs

def test_new_operation_full_pipeline(self, tmp_path):
    """Test new operation through full pipeline."""
    # Create model with new operation
    # Run through all 5 stages
    # Verify outputs at each stage
    pass
```

## Continuous Integration

### Recommended CI Strategy

For CI pipelines, use a tiered approach:

```bash
# Tier 1: Quick validation (on every commit)
bash run_tests.sh quick

# Tier 2: Unit tests (on PR)
bash run_tests.sh unit

# Tier 3: Full suite (on merge to main)
bash run_tests.sh all
```

### GitHub Actions Workflow (example)

```yaml
name: Solar Tests

on: [push, pull_request]

jobs:
  quick-tests:
    name: Quick Smoke Tests
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.8
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run quick tests
      run: |
        bash run_tests.sh quick

  full-tests:
    name: Full Test Suite
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run all tests with coverage
      run: |
        bash run_tests.sh all
        python3 -m pytest tests/ --cov=solar --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## YAML Output Format

Solar uses a custom `NoAliasDumper` to ensure all YAML outputs are human-readable:

```python
# Standard PyYAML with anchors (hard to read)
input_shapes: &id001
- - 16
output_shapes: *id001

# Solar with NoAliasDumper (easy to read)
input_shapes:
- - 16
output_shapes:
- - 16
```

The `NoAliasDumper` is automatically used for all YAML outputs:
- `pytorch_graph.yaml`
- `einsum_graph.yaml`
- `einsum_graph_renamed.yaml`
- `analysis.yaml`
- `perf_<arch>.yaml`
- `timeloop_graph.yaml`

This makes outputs easier to inspect, diff, and debug, at a small cost of slightly larger file sizes.

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install package in development mode
   pip install -e .
   ```

2. **Missing Dependencies**
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   
   # For development/testing
   pip install pytest pytest-cov
   
   # For graph visualization
   pip install graphviz matplotlib
   ```

3. **Test Discovery Issues**
   ```bash
   # Run from solar/ root directory
   cd /path/to/solar
   python3 -m pytest tests/
   ```

4. **Model Loading Failures**
   ```bash
   # Check PyTorch and torchview versions
   pip install torch>=2.0.0 torchview>=0.2.6
   
   # For RNN models, ensure CPU fallback is working
   # (Solar automatically handles meta → cpu fallback)
   ```

5. **YAML Anchor/Alias Issues**
   ```bash
   # If you see &id001 or *id001 in outputs, ensure NoAliasDumper is used
   # All Solar components should automatically use NoAliasDumper from solar.common.utils
   ```

6. **Graph Visualization Errors**
   ```bash
   # Install graphviz system package
   # Ubuntu/Debian:
   sudo apt-get install graphviz
   
   # macOS:
   brew install graphviz
   
   # Then install Python package:
   pip install graphviz
   ```

## Coverage Reports

Generate coverage reports:
```bash
# HTML report
python -m pytest tests/ --cov=solar --cov-report=html
open htmlcov/index.html

# Terminal report
python -m pytest tests/ --cov=solar --cov-report=term-missing
```

## Running Examples

Solar includes several example models in `examples/`:

```bash
# Run all examples
bash run_tests.sh examples

# Or run individual examples
cd examples/DenseAttention && bash run_solar.sh
cd examples/SlidingWindowAttention && bash run_solar.sh
cd examples/RandomAttention && bash run_solar.sh
cd examples/BlockSparseAttention && bash run_solar.sh
cd examples/Attention && bash run_solar.sh
cd examples/BERT && bash run_solar.sh
```

Each example demonstrates:
- Complete 5-stage pipeline
- PDF graph visualization
- Performance prediction on H100

## Performance Testing

For performance-sensitive components:
```python
import pytest
import time

@pytest.mark.benchmark
def test_einsum_performance():
    """Benchmark einsum generation."""
    from solar.einsum import EinsumAnalyzer
    
    analyzer = EinsumAnalyzer()
    
    start = time.time()
    for _ in range(1000):
        analyzer.generate_matmul_einsum([100, 200], [200, 300])
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in under 1 second
```

## Test Execution Times

Approximate execution times on a typical development machine:

| Test Category | Tests | Time | Command |
|--------------|-------|------|---------|
| Quick smoke tests | 2 | ~1 min | `bash run_tests.sh quick` |
| Graph processing | 10 | ~50 sec | `bash run_tests.sh graph` |
| Einsum analyzer | 15 | ~37 sec | `bash run_tests.sh einsum` |
| Model analyzer | 15 | ~39 sec | `bash run_tests.sh model` |
| LLM agent | 20 | ~39 sec | `bash run_tests.sh llm` |
| BERT example | 1 | ~44 sec | `bash run_tests.sh bert` |
| Integration | 6 | ~48 sec | `bash run_tests.sh integration` |
| Examples | 6 | ~3 min | `bash run_tests.sh examples` |
| **All tests** | **~70** | **~5-6 min** | `bash run_tests.sh` |

## Next Steps

1. Add property-based testing with hypothesis for einsum equation validation
2. Implement performance regression tests
3. Add mutation testing for critical paths
4. Create test data generators for complex transformer models
5. Add CI/CD pipeline configuration for automated testing
