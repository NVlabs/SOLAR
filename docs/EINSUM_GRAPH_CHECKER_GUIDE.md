# Einsum Graph Checker Guide

This guide explains how to use the `einsum_graph_check.py` utility to validate einsum graph consistency.

## Overview

The einsum graph checker validates that an einsum graph is internally consistent by checking:

1. **Connection Consistency**: Bidirectional connection validity
2. **Tensor Name Consistency**: Tensor names match across connected nodes  
3. **Shape Consistency**: Tensor shapes are compatible (optional warnings)

## Quick Start

### Command Line Usage

```bash
# Basic validation
python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml

# With debug output
python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml --debug

# Treat warnings as errors
python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml --strict
```

### Python API

```python
from solar.common.einsum_graph_check import (
    EinsumGraphChecker,
    check_einsum_graph,
    check_einsum_graph_file,
)

# Check a graph dictionary
result = check_einsum_graph(graph_dict)
print(result.summary())

# Check a file
result = check_einsum_graph_file("path/to/einsum_graph.yaml")
if not result.is_valid:
    for error in result.errors:
        print(error)
```

## Validation Rules

### 1. Connection Consistency

The checker ensures bidirectional consistency in the `connections` field:

#### Rule 1a: Output → Input Consistency
If layer A lists layer B in `connections.outputs`, then layer B must list layer A in `connections.inputs`.

```yaml
# VALID: Bidirectional connection
layer_A:
  connections:
    outputs: [layer_B]
layer_B:
  connections:
    inputs: [layer_A]

# INVALID: A lists B as output, but B doesn't list A as input
layer_A:
  connections:
    outputs: [layer_B]
layer_B:
  connections:
    inputs: []  # ERROR: Missing layer_A
```

#### Rule 1b: Input → Output Consistency
If layer B lists layer A in `connections.inputs`, then layer A must list layer B in `connections.outputs`.

```yaml
# VALID
layer_A:
  connections:
    outputs: [layer_B]
layer_B:
  connections:
    inputs: [layer_A]

# INVALID: B lists A as input, but A doesn't list B as output
layer_A:
  connections:
    outputs: []  # ERROR: Missing layer_B
layer_B:
  connections:
    inputs: [layer_A]
```

### 2. Tensor Name Consistency

The checker validates that `tensor_names` fields properly reference each other:

#### Rule 2a: Input Tensor References
Each entry in `tensor_names.inputs` should exist in the predecessor's `tensor_names.outputs`.

```yaml
# VALID: Input tensor comes from predecessor's output
start:
  tensor_names:
    outputs: [start.Output]
Model.linear:
  tensor_names:
    inputs: [start.Output]  # References start's output

# WARNING: Input tensor not found in predecessor
start:
  tensor_names:
    outputs: [start.Output]
Model.linear:
  tensor_names:
    inputs: [wrong_name.Output]  # Warning: Not in start's outputs
```

#### Rule 2b: Output Tensor Usage
Each entry in `tensor_names.outputs` should be used in at least one successor's `tensor_names.inputs`.

```yaml
# VALID: Output is used by successor
Model.linear:
  tensor_names:
    outputs: [Model.linear.Output]
  connections:
    outputs: [Model.relu]
Model.relu:
  tensor_names:
    inputs: [Model.linear.Output]  # Uses the output

# WARNING: Output not used by any successor
Model.linear:
  tensor_names:
    outputs: [Model.linear.Output]
  connections:
    outputs: [Model.relu]
Model.relu:
  tensor_names:
    inputs: [other.Output]  # Warning: Doesn't use Model.linear.Output
```

### 3. Shape Consistency (Warnings Only)

The checker can optionally validate that tensor shapes are compatible across connections. This generates warnings rather than errors since some shape mismatches may be intentional (e.g., broadcasting).

## Error Types

| Error Type | Severity | Description |
|-----------|----------|-------------|
| `missing_layers` | error | Graph is missing the 'layers' key |
| `missing_successor` | error | Output connection references non-existent layer |
| `missing_predecessor` | error | Input connection references non-existent layer |
| `connection_mismatch` | error | Bidirectional connection not satisfied |
| `tensor_name_mismatch` | warning | Input tensor not found in source outputs |
| `unused_output` | warning | Output tensor not used by any successor |
| `file_not_found` | error | Graph file does not exist |
| `yaml_parse_error` | error | YAML syntax error in graph file |
| `load_error` | error | Unexpected error loading graph file |

## Error Message Format

Each error message provides detailed diagnostic information to help quickly identify and fix issues:

1. **Layer identification**: Shows the layer ID and type where the issue occurs
2. **Related layers**: Shows information about connected layers (predecessors/successors)
3. **Current state**: Displays the actual values in the graph (connections, tensor names)
4. **Expected state**: Indicates what the graph checker expected to find
5. **Fix suggestion**: Provides actionable steps to resolve the issue

Example error structure:
```
[Error N] Layer: {layer_id}
  Type: {error_type}
  Details:
    {problem_description}
        Context: {relevant_info_about_current_layer}
        Expected: {what_should_be_there}
        Found: {what_was_actually_found}
        FIX: {suggested_fix_action}
```

## Example Output

### Valid Graph
```
======================================================================
Einsum Graph Validator
======================================================================
File: path/to/einsum_graph.yaml
======================================================================

✅ Graph validation passed

======================================================================
✅ Validation PASSED - Graph is consistent
======================================================================
```

### Graph with Errors
```
======================================================================
Einsum Graph Validator
======================================================================
File: path/to/einsum_graph.yaml
======================================================================

❌ 2 error(s) | ⚠️ 1 warning(s)

----------------------------------------------------------------------
ERRORS (2):
----------------------------------------------------------------------

[Error 1] Layer: Model.linear
  Type: connection_mismatch
  Details:
    Bidirectional connection broken: output 'Model.linear' -> 'Model.relu'
        Layer 'Model.linear' (type=linear) lists 'Model.relu' in connections.outputs
        BUT layer 'Model.relu' (type=relu) does NOT list 'Model.linear' in connections.inputs
        'Model.relu'.connections.inputs = []
        FIX: Add 'Model.linear' to 'Model.relu'.connections.inputs, or remove 'Model.relu' from 'Model.linear'.connections.outputs

[Error 2] Layer: Model.relu
  Type: missing_predecessor
  Details:
    Input connection 'Model.conv' does not exist in graph.
        Layer 'Model.relu' (type=relu) lists 'Model.conv' in connections.inputs
        but 'Model.conv' is not a valid layer ID.
        Available layers: ['start', 'Model.linear', 'Model.relu', 'Model.add', 'output']
        FIX: Update connections.inputs to reference an existing layer, or add the missing layer.

----------------------------------------------------------------------
WARNINGS (1):
----------------------------------------------------------------------

[Warning 1] Layer: Model.add
  Type: tensor_name_mismatch
  Details:
    Input tensor 'Model.unknown.Output' not found in source layer's outputs.
        Current layer: 'Model.add' (type=add)
        tensor_names.inputs: ['Model.unknown.Output', 'Model.bias.Output']
        Extracted source layer: 'Model.unknown' (MISSING, type=N/A)
        Source layer's tensor_names.outputs: []
        FIX: Either update 'Model.add'.tensor_names.inputs to match 'Model.unknown'.tensor_names.outputs,
             or update 'Model.unknown'.tensor_names.outputs to include 'Model.unknown.Output'

======================================================================
❌ Validation FAILED - Fix errors before proceeding
======================================================================
```

## Integration with Solar Pipeline

The graph checker can be integrated into the Solar pipeline:

```python
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.common.einsum_graph_check import check_einsum_graph

# Convert to einsum
converter = PyTorchToEinsum()
graph = converter.convert(pytorch_graph_path, output_dir)

# Validate result
result = check_einsum_graph(graph)
if not result.is_valid:
    print("Graph validation failed!")
    for error in result.errors:
        print(f"  {error}")
```

## Common Issues and Fixes

### Issue 1: Expanded Operations Have Stale References

When operations like `scaled_dot_product_attention` are expanded into subgraphs, predecessors may still reference the original node ID.

**Fix**: The `_fix_split_connections` method in `pytorch_to_einsum.py` should update predecessor outputs to point to the correct subgraph entry nodes.

### Issue 2: Tensor Names Don't Match Connection IDs

Tensor names should follow the format `{layer_id}.Output` or `{layer_id}.Output_{n}`.

**Fix**: Ensure tensor names are generated consistently with layer IDs.

### Issue 3: Missing Connections After Graph Transformation

When graphs are transformed (split, expanded, renamed), connections need to be updated accordingly.

**Fix**: Use the graph checker after each transformation step to catch issues early.

## API Reference

### Classes

#### `EinsumGraphChecker`
Main checker class.

```python
checker = EinsumGraphChecker(debug=False)
result = checker.check_graph(graph_dict)
result = checker.check_file(Path("graph.yaml"))
```

#### `ValidationError`
Represents a single validation error.

```python
@dataclass
class ValidationError:
    layer_id: str      # Layer that has the issue
    error_type: str    # Type of error (see table above)
    message: str       # Human-readable description
    severity: str      # "error" or "warning"
```

#### `ValidationResult`
Container for validation results.

```python
result.is_valid      # True if no errors
result.has_warnings  # True if warnings exist
result.errors        # List of ValidationError
result.warnings      # List of ValidationError (severity="warning")
result.summary()     # Human-readable summary string
```

### Functions

#### `check_einsum_graph(graph, debug=False)`
Validate a graph dictionary.

#### `check_einsum_graph_file(path, debug=False)`
Validate a graph from a YAML file.

## See Also

- `solar/solar/einsum/pytorch_to_einsum.py` - Graph generation
- `solar/EINSUM_GUIDE.md` - Einsum equation conventions
- `solar/SOL_Guide.md` - Performance analysis models
