# Einsum Equation Convention Guide

This guide explains the einsum equation conventions used in the Solar codebase for representing tensor operations.

## Table of Contents

1. [Rank Variable Naming Convention](#rank-variable-naming-convention)
2. [Einsum Equation Format](#einsum-equation-format)
3. [Extended Einsum Operations](#extended-einsum-operations)
4. [Examples](#examples)

---

## Rank Variable Naming Convention

In Solar's einsum representation, **rank variables (dimension names) follow a strict naming convention**:

### Format: Capital Letter + Optional Number

- Single capital letters: `A`, `B`, `C`, ..., `Z`
- Capital letter followed by an integer: `A0`, `A1`, `A10`, `A100`, ..., `Z99`, etc.

**Note:** Multi-letter prefixes (e.g., `AA0`, `AB1`) are **NOT allowed**. Only a single capital letter followed by an optional integer is valid.

### Label Generation Order

Labels are generated in the following sequence:
1. `A`, `B`, `C`, ..., `Z` (26 labels)
2. `A0`, `B0`, `C0`, ..., `Z0` (26 labels)
3. `A1`, `B1`, `C1`, ..., `Z1` (26 labels)
4. ... continuing with incrementing integers ...
5. `A10`, `B10`, ..., `Z10`, `A11`, ..., etc.

This provides **unlimited unique dimension labels** by using incrementing integers after the single capital letter.

### Examples

```yaml
# Valid rank variables
einsum_equation: B0B1K,NK->B0B1N      # B0, B1, K, N are valid
einsum_equation: ABC,CD->ABD          # A, B, C, D are valid
einsum_equation: A0B0C0,C0D0->A0B0D0  # A0, B0, C0, D0 are valid

# Invalid rank variables (NOT allowed)
einsum_equation: batch,seq,hidden     # lowercase not allowed
einsum_equation: b0b1k,nk->b0b1n      # lowercase not allowed
```

---

## Einsum Equation Format

### Basic Format

```
<input_operands> -> <output_operand>
```

Where:
- **Input operands** are comma-separated dimension strings
- **Output operand** is the result dimension string
- Dimensions that appear in inputs but not in output are **contracted (reduced)**

### Examples

```yaml
# Matrix multiplication: (M,K) @ (K,N) -> (M,N)
einsum_equation: MK,KN->MN

# Batched matrix multiplication: (B,M,K) @ (B,K,N) -> (B,M,N)
einsum_equation: BMK,BKN->BMN

# Multi-head attention style: (B0,B1,K) @ (N,K) -> (B0,B1,N)
einsum_equation: B0B1K,NK->B0B1N

# Elementwise operation (copy): (A,B,C) -> (A,B,C)
einsum_equation: ABC->ABC

# Reduction (sum over last dim): (A,B,C) -> (A,B)
einsum_equation: ABC->AB
```

### Sliding Window Format for Convolutions

Convolutions use a special **sliding window format** that explicitly shows the relationship between output positions and kernel positions. This format can be directly flattened into nested loops.

#### 2D Convolution: `BC(P+R)(Q+S),OCRS->BOPQ`

```yaml
# Conv2d sliding window format
einsum_equation: BC(P+R)(Q+S),OCRS->BOPQ
```

Where:
- `B` = batch dimension
- `C` = input channels (contracted)
- `O` = output channels
- `P`, `Q` = output spatial positions (height, width)
- `R`, `S` = kernel positions (height, width, contracted)
- `(P+R)`, `(Q+S)` = input spatial positions as function of output + kernel

This maps directly to the nested loop structure:
```python
for b in B:
    for o in O:
        for p in P:
            for q in Q:
                for c in C:
                    for r in R:
                        for s in S:
                            out[b,o,p,q] += inp[b,c,p+r,q+s] * weight[o,c,r,s]
```

#### 1D Convolution: `BC(P+R),OCR->BOP`

```yaml
einsum_equation: BC(P+R),OCR->BOP
```

#### 3D Convolution: `BC(P+T)(Q+R)(U+S),OCTRS->BOPQU`

```yaml
einsum_equation: BC(P+T)(Q+R)(U+S),OCTRS->BOPQU
```

---

## Extended Einsum Operations

Solar extends the standard einsum notation with explicit **elementwise** and **reduction** operators, allowing representation of operations beyond traditional matrix multiplication.

### Default Operators

By default, einsum uses:
- `elementwise_op: mul` (multiplication)
- `reduction_op: add` (summation)

This corresponds to the standard einsum semantics:
```
Output[i,j] = Σ_k Input1[i,k] * Input2[k,j]
```

### Custom Operators

You can specify different operators for non-standard operations:

#### Elementwise Operations
- `mul` - multiplication (default)
- `add` - addition
- `sub` - subtraction
- `div` - division
- `max` - element-wise maximum
- `min` - element-wise minimum

#### Reduction Operations
- `add` - summation (default)
- `mul` - product
- `max` - maximum
- `min` - minimum
- `mean` - average

### Examples

```yaml
# Standard matrix multiplication
einsum_equation: MK,KN->MN
elementwise_op: mul
reduction_op: add

# Element-wise addition (no reduction)
einsum_equation: ABC,ABC->ABC
elementwise_op: add
reduction_op: null

# Max pooling style reduction
einsum_equation: ABCD->AB
elementwise_op: null
reduction_op: max

# Softmax numerator (exp then sum)
einsum_equation: ABC->ABC
elementwise_op: exp
reduction_op: null

# Attention scores normalization
einsum_equation: ABC->AB
elementwise_op: null
reduction_op: add
```

---

## Examples

### Complete Layer Examples

#### Linear Layer with Bias (Split into Matmul + Add)

When a linear layer has bias, Solar automatically splits it into two operations:

```yaml
# Step 1: Matrix multiplication (matmul)
linear_0:
  type: matmul
  einsum_equation: B0B1K,NK->B0B1N
  elementwise_op: mul
  reduction_op: add
  is_real_einsum: true
  is_einsum_supportable: true
  shapes:
    Input: [2, 32, 64]
    Weight: [128, 64]
    Output: [2, 32, 128]
  connections:
    inputs: [start]
    outputs: [linear_0.bias_add]

# Step 2: Bias addition
linear_0.bias_add:
  type: add
  einsum_equation: ABC->ABC
  elementwise_op: add
  reduction_op: none
  is_real_einsum: false
  is_einsum_supportable: true
  shapes:
    Input: [2, 32, 128]
    Bias: [128]
    Output: [2, 32, 128]
  connections:
    inputs: [linear_0]
    outputs: [relu_0]
```

#### Linear Layer without Bias

```yaml
linear_0:
  type: linear
  einsum_equation: B0B1K,NK->B0B1N
  elementwise_op: mul
  reduction_op: add
  is_real_einsum: true
  is_einsum_supportable: true
  shapes:
    Input: [2, 32, 64]
    Weight: [128, 64]
    Output: [2, 32, 128]
```

#### Elementwise ReLU

```yaml
relu_0:
  type: relu
  einsum_equation: ABC->ABC
  elementwise_op: relu
  reduction_op: null
  is_real_einsum: false
  is_einsum_supportable: true
  shapes:
    Input: [2, 32, 128]
    Output: [2, 32, 128]
```

#### Softmax (Multi-step)

```yaml
# Step 1: Find max for numerical stability
softmax_max:
  type: reduce_max
  einsum_equation: ABC->AB
  elementwise_op: null
  reduction_op: max
  is_real_einsum: false
  is_einsum_supportable: true

# Step 2: Subtract max and exp
softmax_exp:
  type: exp
  einsum_equation: ABC->ABC
  elementwise_op: exp
  reduction_op: null
  is_real_einsum: false
  is_einsum_supportable: true

# Step 3: Sum for normalization
softmax_sum:
  type: reduce_sum
  einsum_equation: ABC->AB
  elementwise_op: null
  reduction_op: add
  is_real_einsum: false
  is_einsum_supportable: true
```

#### Flatten Operation

Flatten reshapes a tensor by collapsing multiple dimensions into one. In einsum notation, this is represented as a dimension mapping:

```yaml
flatten_0:
  type: flatten
  einsum_equation: ABCD->AE
  elementwise_op: null
  reduction_op: null
  is_real_einsum: false
  is_einsum_supportable: false
  shapes:
    Input: [2, 8, 16, 32]    # B, C, H, W
    Output: [2, 4096]         # B, C*H*W
  module_args:
    start_dim: 1
    end_dim: -1
```

Note: Flatten is marked `is_einsum_supportable: false` because it involves dimension reshaping rather than computation and requires special handling for index transformations.

---

## Parsing Dimension Tokens

The `parse_dim_tokens` function in `solar/common/utils.py` handles parsing dimension strings:

```python
from solar.common.utils import parse_dim_tokens

# Single letters
parse_dim_tokens("ABC")      # -> ["A", "B", "C"]

# Letter + number (single letter prefix only!)
parse_dim_tokens("A0B1C2")   # -> ["A0", "B1", "C2"]

# Mixed single letters and letter+number
parse_dim_tokens("B0B1K")    # -> ["B0", "B1", "K"]

# Multi-digit integers
parse_dim_tokens("A12B34")   # -> ["A12", "B34"]
```

### Important Constraints

1. **Single-letter prefix only**: Multi-letter prefixes are NOT allowed.
   - `"AA0"` is parsed as `["A", "A0"]` (two separate tokens), NOT as `["AA0"]`
   
2. **No repeated ranks in the same tensor**: Each dimension in a tensor must be unique.
   - `"AA"` → `["A", "A"]` is **invalid** (repeated rank)
   - Use `"AB"` or `"A0A1"` instead
```

---

## See Also

- `solar/solar/einsum/pytorch_to_einsum.py` - PyTorch to einsum conversion
- `solar/solar/einsum/einsum_rank_renamer.py` - Dimension rank renaming
- `solar/solar/einsum/einsum_to_taco.py` - TACO expression generation
- `solar/solar/einsum/einsum_to_timeloop.py` - Timeloop format conversion
- `solar/solar/common/utils.py` - Utility functions including `parse_dim_tokens`

