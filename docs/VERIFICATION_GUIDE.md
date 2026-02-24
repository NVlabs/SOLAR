# Einsum Verification Guide

This guide explains how to verify that generated einsum expressions produce correct results when compared to the original PyTorch implementations.

## Overview

The verification system validates that the einsum transformations in Solar correctly represent the semantics of the original PyTorch operations. It does this by:

1. **Parsing** the generated `einsum_graph.yaml` to extract the taco expression
2. **Generating** test input data with appropriate shapes
3. **Executing** both the einsum expression and PyTorch reference
4. **Comparing** the outputs within specified tolerances

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Einsum Verification Flow                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  einsum_graph    │     │  PyTorch Source  │     │  Test Data       │
│    .yaml         │     │    (source.py)   │     │  (generated)     │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        │
┌──────────────────┐     ┌──────────────────┐              │
│ Parse Expression │     │ Load Reference   │              │
│ & Extract Shapes │     │ Model            │              │
└────────┬─────────┘     └────────┬─────────┘              │
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Einsum Executor  │◄────┤ Random Inputs    │────►│ PyTorch Model    │
│ (AST-based)      │     │ (scaled shapes)  │     │ (eval mode)      │
└────────┬─────────┘     └──────────────────┘     └────────┬─────────┘
         │                                                  │
         ▼                                                  ▼
┌──────────────────┐                              ┌──────────────────┐
│ Einsum Output    │                              │ PyTorch Output   │
└────────┬─────────┘                              └────────┬─────────┘
         │                                                  │
         └────────────────────┬─────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Tensor Comparator│
                    │ (atol, rtol)     │
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Verification     │
                    │ Result (YAML)    │
                    └──────────────────┘
```

## Components

### 1. Einsum Executor (`solar_verifier/src/einsum_executor.py`)

The einsum executor is an AST-based Python interpreter that:

- **Parses** einsum expressions with subscript notation (e.g., `C[i,j] = A[i,k] * B[k,j]`)
- **Infers** dimensions from input tensor shapes
- **Generates** executable Python loop code
- **Supports** element-wise operations, reductions, and activation functions

Supported operations:
- **Element-wise**: `+`, `-`, `*`, `/`, `**`
- **Reductions**: `sum`, `max`, `min` over index dimensions
- **Activations**: `relu`, `gelu`, `sigmoid`, `tanh`, `leaky_relu`, `elu`, `selu`, `hardsigmoid`, `softplus`, `hardtanh`

### 2. Tensor Comparator (`solar_verifier/src/comparator.py`)

Compares two tensors with configurable tolerances:

- **Absolute tolerance** (`atol`): Maximum allowed absolute difference
- **Relative tolerance** (`rtol`): Maximum allowed relative difference

Default tolerances: `atol=1e-5`, `rtol=1e-4`

### 3. CLI Tool (`solar/solar/cli/verify_einsum.py`)

Command-line interface for running verification:

```bash
# Verify all kernels in level1
python -m solar.cli.verify_einsum --level level1

# Verify specific kernels
python -m solar.cli.verify_einsum --level level1 --kernel-ids 19 20 21

# Verbose output with custom tolerances
python -m solar.cli.verify_einsum --kernel-ids 19 --verbose --atol 1e-4 --rtol 1e-3
```

### 4. Bash Script (`solar/scripts/run_kernelbench_einsum_verification.sh`)

Batch verification with CSV summary:

```bash
# Run all level1 verifications
./scripts/run_kernelbench_einsum_verification.sh

# Run with verbose output
./scripts/run_kernelbench_einsum_verification.sh --verbose

# Specify kernel IDs
./scripts/run_kernelbench_einsum_verification.sh --kernel-ids 19 20 21
```

## Output Format

### Per-Benchmark Result (`einsum_verification/einsum_verification.yaml`)

Each verified benchmark gets an `einsum_verification.yaml` file and an `emulated_code.py` file:

**Directory structure:**
```
output_kernelbench/level1/19_ReLU/
├── einsum/
│   └── einsum_graph.yaml      # Generated einsum expression
├── einsum_verification/
│   ├── einsum_verification.yaml  # Verification result
│   └── emulated_code.py          # Generated Python loop code
└── graph/
    └── source_19_ReLU.py      # PyTorch reference
```

**einsum_verification.yaml:**

**Passed:**
```yaml
status: passed
benchmark_name: 19_ReLU
timestamp: 2026-01-25T10:30:00.123456
expression: "O0(b,c,h,w) = relu(In0(b,c,h,w))"
shapes:
  Input: [8, 64, 32, 32]
  Output: [8, 64, 32, 32]
verification_stats:
  max_absolute_error: 0.0
  max_relative_error: 0.0
```

**Failed:**
```yaml
status: failed
benchmark_name: 50_conv_standard_2D
timestamp: 2026-01-25T10:30:00.123456
expression: "O0(b,c,h,w) = In0(b,ci,h+kh,w+kw) * In1(c,ci,kh,kw)"
shapes:
  Input: [8, 3, 32, 32]
  Weight: [64, 3, 3, 3]
  Output: [8, 64, 30, 30]
error:
  type: output_mismatch
  message: Einsum output does not match PyTorch reference
  details:
    max_absolute_error: 0.125
    max_relative_error: 0.05
    tolerance_atol: 0.00001
    tolerance_rtol: 0.0001
    einsum_output_shape: [8, 64, 30, 30]
    pytorch_output_shape: [8, 64, 30, 30]
```

### Summary CSV (`einsum_verification_results.csv`)

```csv
benchmark_name,status,error_type,error_message
"19_ReLU","passed","",""
"20_LeakyReLU","passed","",""
"50_conv_standard_2D","failed","output_mismatch","Einsum output does not match PyTorch reference"
```

## Error Types

| Error Type | Description | Common Causes |
|------------|-------------|---------------|
| `missing_file` | `einsum_graph.yaml` not found | Einsum conversion was not run or failed |
| `no_taco_expression` | No taco_expression in YAML | Operation not supported for einsum conversion |
| `yaml_parse_error` | Failed to parse YAML | Malformed YAML syntax |
| `missing_reference` | No PyTorch source file | Graph processing incomplete |
| `missing_shape` | Shape not defined for tensor | Incomplete shape inference |
| `einsum_execution_error` | Einsum execution failed | Expression parsing or execution bug |
| `pytorch_execution_error` | PyTorch execution failed | Model initialization or input issues |
| `output_mismatch` | Results don't match | Incorrect einsum expression or numerical precision |
| `comparison_error` | Failed to compare tensors | Shape mismatch or NaN values |

## Troubleshooting

### Common Issues

#### 1. "No taco_expression found"
The operation may not be supported for einsum conversion. Check if:
- The operation is in the supported operations list
- The einsum conversion completed successfully

```bash
# Check if einsum conversion was successful
cat output_kernelbench/level1/19_ReLU/einsum/einsum_graph.yaml | grep taco_expression
```

#### 2. "Missing shape for input tensor"
The shape mapping may be incomplete. Check:
- The shapes section in einsum_graph.yaml
- Input/Output naming conventions match

#### 3. "Output mismatch"
The einsum expression may be incorrect. Debug with:
```bash
python -m solar.cli.verify_einsum --kernel-ids 19 --verbose
```

This shows:
- Parsed expression details
- Generated Python loop code
- Side-by-side output comparison

#### 4. Numerical precision issues
For operations with accumulation or complex functions:
```bash
# Increase tolerances
python -m solar.cli.verify_einsum --kernel-ids 50 --atol 1e-4 --rtol 1e-3
```

### Debugging Steps

1. **Check einsum_graph.yaml exists and has expression:**
   ```bash
   ls output_kernelbench/level1/19_ReLU/einsum/
   ```

2. **Verify expression manually:**
   ```python
   import yaml
   with open('output_kernelbench/level1/19_ReLU/einsum/einsum_graph.yaml') as f:
       data = yaml.safe_load(f)
   print(data['layers'])
   ```

3. **Run verbose verification:**
   ```bash
   python -m solar.cli.verify_einsum --kernel-ids 19 --verbose
   ```

4. **Check the verification result:**
   ```bash
   cat output_kernelbench/level1/19_ReLU/einsum_verification/einsum_verification.yaml
   ```

5. **Review the emulated Python code:**
   ```bash
   cat output_kernelbench/level1/19_ReLU/einsum_verification/emulated_code.py
   ```

## Scaling for Performance

Verification uses scaled tensor dimensions for faster testing. The default scale factor is 0.01 (1% of original size).

```bash
# Use smaller tensors for quick testing
python -m solar.cli.verify_einsum --level level1 --scale 0.001

# Use full-size tensors for thorough testing
python -m solar.cli.verify_einsum --level level1 --scale 1.0
```

## Supported Operations

The verification system supports:

| Category | Operations |
|----------|------------|
| **Matrix Multiply** | `matmul`, `bmm`, batched matmul |
| **Activations** | `relu`, `leaky_relu`, `gelu`, `sigmoid`, `tanh`, `elu`, `selu`, `hardsigmoid`, `softplus`, `hardtanh` |
| **Reductions** | `sum`, `max`, `min`, `mean` (over dimensions) |
| **Element-wise** | `add`, `sub`, `mul`, `div`, arithmetic |
| **Normalization** | Partial support (batchnorm, layernorm shapes) |

**Not yet supported:**
- Convolution operations (complex index arithmetic)
- Attention mechanisms
- Cumulative operations (cumsum, cumprod)
- Loss functions

## Integration with Solar Pipeline

The verification step fits into the Solar pipeline as follows:

```
PyTorch Model → Graph Extraction → Einsum Conversion → Verification → Performance Analysis
                                                              ↑
                                                        (This guide)
```

Run the complete pipeline:

```bash
# 1. Generate einsum graphs
python -m solar.cli.toeinsum --level level1

# 2. Verify the generated einsums
python -m solar.cli.verify_einsum --level level1

# 3. Or use the batch script
./scripts/run_kernelbench_einsum_verification.sh
```

## API Reference

### Python API

```python
from solar.cli.verify_einsum import EinsumVerifier, VerificationResult

# Create verifier
verifier = EinsumVerifier(
    debug=False,
    scale_factor=0.01,
    atol=1e-5,
    rtol=1e-4
)

# Verify single benchmark
result = verifier.verify_benchmark(Path("output_kernelbench/level1/19_ReLU"))

# Check result
if result.passed:
    print(f"✅ Passed with max error: {result.max_abs_error}")
else:
    print(f"❌ Failed: {result.error_type} - {result.error_message}")

# Get all benchmark directories
benchmark_dirs = verifier.get_benchmark_directories(
    base_dir=Path("output_kernelbench"),
    level="level1",
    kernel_ids=[19, 20, 21]
)
```

### CLI Options

```
usage: verify_einsum.py [-h] [--level LEVEL] [--kernel-ids IDS...] 
                        [--output-dir DIR] [--verbose] [--scale SCALE]
                        [--atol ATOL] [--rtol RTOL]

Options:
  --level LEVEL       Kernel level (level1, level2, etc.)
  --kernel-ids IDS    Specific kernel IDs to verify
  --output-dir DIR    Output directory with einsum graphs
  --verbose, -v       Enable verbose output
  --scale SCALE       Tensor dimension scale factor (default: 0.01)
  --atol ATOL         Absolute tolerance (default: 1e-5)
  --rtol RTOL         Relative tolerance (default: 1e-4)
```

## Contributing

To add support for new operations:

1. Add the function to `solar_verifier/src/einsum_executor.py`:
   ```python
   def new_activation_numpy(x):
       return np.your_implementation(x)
   
   BUILTIN_FUNCTIONS['new_activation'] = new_activation_numpy
   ```

2. Add test case in `solar_verifier/tests/`:
   ```python
   def test_new_activation():
       expr = "O0[i] = new_activation(In0[i])"
       # Test implementation
   ```

3. Run verification on relevant benchmarks:
   ```bash
   python -m solar.cli.verify_einsum --kernel-ids <id> --verbose
   ```
