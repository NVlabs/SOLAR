<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Einsum Equation Convention Guide

This guide explains the einsum equation conventions used in the Solar codebase for representing tensor operations.

## Table of Contents

1. [Rank Variable Naming Convention](#rank-variable-naming-convention)
2. [Einsum Equation Format](#einsum-equation-format)
3. [Extended Einsum Operations](#extended-einsum-operations)
4. [TACO Expression Format](#taco-expression-format)
5. [Examples](#examples)
6. [Alias, View and Copy Operations](#alias-view-and-copy-operations)

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

#### Determinism guarantee

Rank labelling is a pure function of the einsum graph. The BFS renamer
(`solar/einsum/einsum_rank_renamer.py`, now a placeholder that raises on use) was replaced by
`solar/einsum/af_graph_builder.py`, which assigns `R0`, `R1`, ... to axis equivalence classes in topological order
of first appearance (a union-find pass over `(layer, role, position)` keys); `--enable-rename` keeps writing
`einsum_graph_renamed.yaml` as a copy of `einsum_graph.yaml`. The same `pytorch_graph.yaml` therefore yields
byte-identical `einsum_graph.yaml`, `einsum_graph_renamed.yaml` and `af_einsum_graph.yaml` in every Python process,
for every `PYTHONHASHSEED` (issue #43; checked by `tests/test_rank_label_determinism.py` in fresh subprocesses).
Labels are a presentation of the contraction only: MACs, traffic and SOL are computed from operand shapes and
never depend on which labels were chosen.

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

#### Transposed Convolutions

Transposed convolutions use subtraction in the sliding window:

```yaml
# ConvTranspose2d: BC(P-R)(Q-S),CKRS->BKPQ
einsum_equation: BC(P-R)(Q-S),CKRS->BKPQ
```

A transposed convolution is a scatter: every input element is multiplied
with all `(C_out / groups) * prod(kernel)` taps, so its MACs are counted
from the **input** shape,

```
MACs = N * C_in * prod(input_spatial) * (C_out / groups) * prod(kernel)
```

independent of stride, padding, output_padding and dilation (the same
formula `torch.utils.flop_counter` uses for transposed convolutions). This
is the **dense** convention and is what the `macs` field reports.

Three quantities exist for a transposed convolution. Per spatial dim an
input index `i` and tap `r` write output index `o = i*stride - padding +
r*dilation`:

| quantity | formula | field |
|---|---|---|
| old output-based count | `N * (C_out/groups) * prod(output_spatial) * C_in * prod(kernel)` (every output position times all taps; the generic equation product) | not reported since #31 |
| dense input-based count | `N * C_in * prod(input_spatial) * (C_out/groups) * prod(kernel)` (every input element times all taps, cropped ones included) | `macs` |
| minimum work | `N * C_in * (C_out/groups) * prod_dims #{(i, r): 0 <= o < out}` (only taps that land inside the output) | `min_work_macs` |

`min_work <= dense` and `min_work <= old` always hold; `dense == min_work`
iff no tap is cropped, i.e. padding 0 in every dim (whatever the stride,
dilation or output_padding). The bound needs stride/padding/dilation, which
the converter exports as `additional_info.conv_args` from the captured call
attributes; a layer without them reports `min_work: not_representable`
(and so does `total`), never a number equal to `macs`.

Relative to the dense count, the old/new ratio is
`prod(output_spatial) / prod(input_spatial)` for the same channel and
kernel dimensions. It equals `prod(stride)` only when the output is exactly
`stride` times the input, as in KernelBench L2/20 (`ConvTranspose3d`,
stride 2, padding 1, output_padding 1: 8x, an over-count); other padding,
dilation or output_padding settings give other ratios (17/7 for a length-7
input with `k=4, stride=2, padding=1, dilation=2`), and the ratio can be
below 1 for cropped configurations (`N=1, C_in=C_out=4, in=7, k=3,
stride=1, padding=2`: old 240, dense 336, ratio 5/7, where the old count
happens to equal the minimum work 240). For L2/20 the three quantities are
115,964,116,992 / 14,495,514,624 / 13,899,366,400 (min/dense =
424175/442368: one `(i=0, r=0)` tap per dim is cropped by the padding).
The equation itself is unchanged; it still drives traffic and exports.

### Costs derived from an equation

* **MACs** are a **cost convention**, not a count of emitted GPU instructions. A MAC is
  the accounting unit of one multiply-accumulate and equals **two** FLOPs
  (`flops = 2 × macs`); it is not universally one hardware instruction. The arch configs
  express throughput in MACs per cycle on the same 2 FLOP/MAC conversion (datasheet dense
  FLOP/s ÷ 2 ÷ the declared clock), see the units box in `SOL_GUIDE.md`.
  * **Primary result (`macs`).** The handler in charge of the layer decides it: operand-
    specific handlers (the transposed convolution above, convolutions, linear) use the
    operation's own count; a **reducing two-operand contraction** (one output, at least one
    index summed away: `matmul`, `bmm`, `ik,kj->ij`) costs the product of the extents of
    all its unique indices, exactly the multiply-accumulates of that single contraction; a
    two-operand equation that reduces **nothing** is an elementwise product and costs
    **0 MACs** plus `numel(output)` scalar multiplies (`i,j->ij` with sizes 3 and 4:
    0 MACs + 12 scalar multiplies, reported as `scalar_multiplies` and billed in
    `other_ops`); an explicit multi-operand `torch.einsum` is priced by its own handler as
    the **left-to-right chain of pairwise contractions**, reducing steps in `macs`,
    non-reducing steps in `scalar_multiplies` (see "Explicit `torch.einsum` (multi-operand)
    MAC accounting" below; `ops/einsum_ops.py` L181–L275 at an earlier revision).
  * **Two labelled estimates beside it, for multi-operand equations.** They qualify the
    primary result; neither replaces it, and the operand-evidence and
    contraction-convention gates still decide whether the equation is priced at all.
    `macs_as_written_all_index_dense` is the **all-index product** of every rank extent in
    the equation as written (the generic single-contraction formula applied to the whole
    equation; for `AB,BC,CD->AD` with `[2,3]`, `[3,5]`, `[5,7]`: `2×3×5×7 = 210`).
    `macs_dense_binary_schedule`, `scalar_multiplies_dense_binary_schedule` and
    `binary_schedule` come from `optimal_dense_binary_schedule`
    (`ops/base.py` L360–L465): an exhaustive search over the **ordinary dense binary
    contraction trees** of the operands, objective `2 × MACs + scalar multiplies`, each
    step priced from its own operand/intermediate broadcast extents (so pairing the two
    singletons of `[1],[1],[3]` first costs `1 + 3`, not `3 + 3`, scalar multiplies; for the
    `AB,BC,CD->AD` example the schedule `(AB,BC->AC)` then `(AC,CD->AD)` costs
    `30 + 70 = 100` MACs). The analyzer emits the four fields only when such a schedule
    exists (three to eight operands, no compound rank, every index bound;
    `graph_analyzer.py` L2099–L2104): not every equation admits one. Neither estimate is a
    universal lower bound, and the schedule is an optimized-work estimate within its
    stated class, not a statement about any implementation.
* **Element counts and traffic** come from the operand shapes, not from the label
  spellings: a consistent alpha-renaming of the labels that preserves the operand/output
  relationships never changes an accounting number (a renaming that merges, splits or
  re-pairs labels is a different equation). Per-layer reads and writes feed
  the unfused total; the fused total counts each unique external storage region once and
  prices every tensor at its recorded dtype width. The conventions, their limits and the
  difference between the fused-traffic principle and its implementation are documented in
  [`SOL_GUIDE.md`](SOL_GUIDE.md) ("Fused SOL", "Physical byte widths", "Known
  Implementation Gaps"); this guide only defines the equation.

---

### `torch._scaled_mm` (FP8 / NVFP4 block-scaled matmul)

`torch._scaled_mm(A, B, scale_a, scale_b, bias=None, out_dtype=...)` is a 2-D
matmul and is billed as one: equation `MK,KN->MN` (or `MK,NK->MN` when `B` is
recorded `[N, K]`), a real einsum whose handler owns the MAC count.

* **MACs = M x K_logical x N.** `K_logical = K_stored x pack_factor`, the pack
  factor taken from the RECORDED dtype of the matrix operands:
  `torch.float4_e2m1fn_x2` stores two E2M1 values per byte along the reduction
  dimension (route 195 records `A [1024, 4096]` for a logical `[1024, 8192]`
  operand; its NVFP4 block scales have `numel = M x K_logical / 16`), fp8 and
  wider dtypes store one.  The stored-shape rank product (`M x K_stored x N`) is
  NOT the MAC count for packed operands.
* **Traffic** stays at the recorded (stored) shapes and the recorded dtype
  widths: 1 B per packed FP4 pair, 1 B per fp8 element, the scale tensors read
  at their own width as additional operands, the output written at its dtype.
  The scales are never MAC-bearing (`scaled_mm.scales_mac_bearing: false`).
* **Evidence or refusal.** The handler binds only with BOTH matrix operand
  dtypes recorded and consistent (same packing) and with the recorded output
  agreeing with `[M, N]`; otherwise it refuses and the converter records the
  refusal on the layer (`scaled_mm_rejection`, `is_einsum_supportable: false`,
  a rank-preserving placeholder equation that bills nothing) and the analyzer
  reports the layer `status: incomplete`, `incomplete_reason:
  scaled_mm_evidence_unresolved` with 0 MACs — never the stored-shape rank
  product and never an assumed packing (evidence-1).  The perf model
  classifies that reason as *zeroed work*: its `prediction_qualification` is
  the `partial: ... contribute 0 MACs` text naming the reason and saying the
  layer's MACs are excluded from the compute cycles (runtimes are lower bounds),
  not the `unqualified: ... accounted from their known shapes` text that
  applies to reasons which keep the layer's costs (wording-1).
* **Rate.** The layer records `compute_dtype` (`nvfp4` / `fp8`) — for the
  scaled-mm contract only; an ordinary matmul with fp8-recorded inputs keeps the
  workload precision's rate (scope-1) — and the analysis total
  `macs_by_compute_dtype`.  The perf model prices those MACs at
  `MAC_per_cycle_<dtype>_tc` when the architecture declares it (B200 declares
  `MAC_per_cycle_nvfp4_tc`); when it does not (H100 has no FP4 rate) the layers
  are reported `rate_unavailable:<dtype>` with an incomplete qualification and
  contribute no compute cycles — no rate is invented and the workload
  precision's rate is not inherited.  A layer that is already incomplete for
  another reason keeps both causes, `'+'`-joined in `incomplete_layers`
  (e.g. `einsum_operands_unresolved+rate_unavailable:nvfp4`), and the perf
  `prediction_qualification` is derived once from that combined record
  (qualification-1).

### Gather, index_select and embedding traffic

`torch.gather`, `index_select`, `take`/`take_along_dim`, `embedding` and advanced
indexing with a tensor index (`x[idx]`) are re-indexings. SOLAR bills them as:

- every index tensor is read fully;
- the source is read only at the gathered elements: `numel(output)` elements,
  as written (a row selected twice is charged twice; no unique-element view);
- the output is written fully;
- `other_ops = numel(output)` (one address computation per output element,
  never one per source element); MACs are zero.

Embedding lookups follow the same rule (the table is the source). Basic slicing
(`x[:, 1:]`, a `__getitem__` with only slice/int indices) remains a zero-copy
view: an output-sized read, no write, no ops.

### Explicit `torch.einsum` (multi-operand) MAC accounting

Units: a chain step is a multiply-accumulate only when it sums at least one index away; a step that drops no index (e.g. a broadcast mask or decay factor, `bclhs,bhcls->bclhs`) is an elementwise multiply and is reported as `scalar_multiplies` (billed in `other_ops`), never in `macs`. The hand values of the SOL self-audit use the same units (Mamba2 L3/48: 15,183,380,480 reduction MACs + 301,989,888 scalar multiplies under the left-to-right schedule).

An explicit `torch.einsum` keeps its own equation.  Its MACs are the
**reduction multiply-accumulates of the left-to-right chain of pairwise contractions** `((op0 * op1) * op2) * ...`,
torch's documented order without `opt_einsum`: each step performs the product of
the extents of the union of the two operands' indices scalar multiplications, and
the intermediate keeps the indices still needed by the output or a later operand.
That work is MACs only for a reducing step; a two-operand einsum that reduces
therefore costs the product of all unique indices in MACs, exactly like
`matmul`/`bmm` and DSL kernels, while a two-operand outer product (`i,j->ij`,
3 x 4) is 0 MACs + 12 scalar multiplies where the generic single-contraction
formula would report 12.  A one-operand einsum keeps the previous convention
(product of its unique indices as reduction work).  Operands that the graph capture did not
record as inputs (parameter-derived tensors) are restored from the node's
`raw_attributes` so every operand's shape counts for both MACs and traffic.

## Tensor Types: `weight` vs `input`

Every layer of the einsum graph carries `tensor_types` aligned with
`tensor_names` / `tensor_shapes`.  Inputs are typed by **provenance**, not by
shape or role:

- `weight` — the tensor originates in the module's state: an `nn.Parameter`
  passed as an operand (torchview records it as a `parameter-tensor` node, the
  graph processor types the connection `weight`), or a normalisation parameter
  or running-statistics buffer attached from the recorded call attributes.
  The converter names such inputs `<layer>.Weight[_k]`, emits no activation
  connection for them, and records WHICH parameter feeds each slot in
  `parameter_sources.inputs[i]` (the capture's `parameter-tensor` node id,
  `null` for every other slot).  That recorded identity is the provenance; a
  tensor's name is never provenance.
- `unresolved` — an einsum operand the capture did not trace, restored from
  `raw_attributes` for its shape (issue #46).  Restoration establishes the
  shape, not the origin: such an operand is an external read of unknown
  provenance, never a parameter.  Graphs saved by a converter before the operand-evidence fix
  typed these operands `weight` (INTEG-2); because they carry no
  `parameter_sources`, the analyzer treats a `weight`-typed operand of an
  explicit `einsum` in such a graph as provenance unknown as well.
- `input` — an activation: a model input or a tensor produced by another layer.

The analyzer reports a layer's parameter footprint under `weight_elements`:
the numel of each DISTINCT parameter it consumes (by `parameter_sources`
identity), once per consuming op — a parameter occupying two slots of one
call is one footprint, a parameter consumed by two layers is counted by each
(the per-consumer sum, which can exceed the model's unique parameter
inventory).  Its DRAM reads are `weight_read_elements`, under the op's read
model.  The two quantities can differ: gathers read rows (fewer or more
elements than the table, an earlier decision), repeated operands are read twice.  A
`weight`-typed slot without a recorded parameter source is provenance
unknown: it is excluded from the footprint, listed in
`weight_provenance_unresolved` with its numel in `weight_elements_unresolved`,
its read is model I/O, and the layer is `status: incomplete`
(`incomplete_reason: weight_provenance_unresolved`, STATUS_CONTRACT rule 1);
graphs saved without `parameter_sources` keep the label rule for every
other op type (the only other source of the label is a capture parameter).
`input` tensors that another layer produces are intermediates, the others are
model inputs under `model_io_elements`.  An input beyond the typed list has
no provenance and is model I/O, never a weight.  Per layer
`weight_read_elements + model_io_elements + intermediate_elements == unfused_elements`;
`total.weight_elements`, `total.weight_read_elements` and `total.model_io_elements`
are layer sums, while the unfused/fused totals also carry graph-level model
I/O no layer owns and `total.fused_elements` is the deduplicated external
footprint.
A tensor the capture did not record at all (a registered buffer multiplied in
`forward`, a parameter reached through a view such as `self.W.t()`, a
parameter passed as a keyword argument) has no type because it has no entry;
that is a capture omission, not a typing choice, and it is absent from traffic
until a dedicated repair attaches or restores it.

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

## TACO Expression Format

Solar can convert einsum equations to [TACO](http://tensor-compiler.org/) (Tensor Algebra Compiler) expression format for compatibility with sparse tensor compilers.

### TACO Expression Syntax

```
Output(indices) = Output(indices) <reduction_op> Input0(indices) <elementwise_op> Input1(indices)
```

Where:
- Indices are lowercase versions of the rank variables
- `O0`, `In0`, `In1`, etc. are tensor names
- Operations are explicit

### Conversion Rules

| Einsum | TACO |
|--------|------|
| `MK,KN->MN` | `O0(m,n) = O0(m,n) + In0(m,k) * In1(k,n)` |
| `ABC->ABC` | `O0(a,b,c) = In0(a,b,c)` |
| `ABC->AB` | `O0(a,b) = O0(a,b) + In0(a,b,c)` |

### Examples

```yaml
# Matrix multiplication
einsum_equation: B0B1K,NK->B0B1N
elementwise_op: mul
reduction_op: add
taco_expression: O0(b0,b1,n) = O0(b0,b1,n) + In0(b0,b1,k) * In1(n,k)

# Batched matmul
einsum_equation: BMK,BKN->BMN
elementwise_op: mul
reduction_op: add
taco_expression: O0(b,m,n) = O0(b,m,n) + In0(b,m,k) * In1(b,k,n)

# Elementwise copy
einsum_equation: ABC->ABC
elementwise_op: null
reduction_op: null
taco_expression: O0(a,b,c) = In0(a,b,c)

# Reduction (sum)
einsum_equation: ABC->AB
elementwise_op: null
reduction_op: add
taco_expression: O0(a,b) = O0(a,b) + In0(a,b,c)
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
  taco_expression: O0(b0,b1,n) = O0(b0,b1,n) + In0(b0,b1,k) * In1(n,k)
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
  taco_expression: O0(b0,b1,n) = O0(b0,b1,n) + In0(b0,b1,k) * In1(n,k)
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
  taco_expression: O0(a,b,c) = relu(In0(a,b,c))
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
  taco_expression: O0(a,b) = max(O0(a,b), In0(a,b,c))
  is_real_einsum: false
  is_einsum_supportable: true

# Step 2: Subtract max and exp
softmax_exp:
  type: exp
  einsum_equation: ABC->ABC
  elementwise_op: exp
  reduction_op: null
  taco_expression: O0(a,b,c) = exp(In0(a,b,c))
  is_real_einsum: false
  is_einsum_supportable: true

# Step 3: Sum for normalization
softmax_sum:
  type: reduce_sum
  einsum_equation: ABC->AB
  elementwise_op: null
  reduction_op: add
  taco_expression: O0(a,b) = O0(a,b) + In0(a,b,c)
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
  taco_expression: O0(a,e) = In0(a,b,c,d)  # where e = b*C*D + c*D + d
  is_real_einsum: false
  is_einsum_supportable: false
  shapes:
    Input: [2, 8, 16, 32]    # B, C, H, W
    Output: [2, 4096]         # B, C*H*W
  module_args:
    start_dim: 1
    end_dim: -1
```

Note: Flatten is marked `is_einsum_supportable: false` because it involves dimension reshaping rather than computation. The TACO expression shows the logical mapping but requires special handling for the index transformation.

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

- `solar/einsum/pytorch_to_einsum.py` - PyTorch to einsum conversion
- `solar/einsum/af_graph_builder.py` - AccelForge graph and rank labelling (`einsum_rank_renamer.py` is a placeholder that raises)
- `solar/einsum/einsum_to_taco.py` - TACO expression generation
- `solar/einsum/einsum_to_timeloop.py` - Timeloop format conversion
- `solar/common/utils.py` - Utility functions including `parse_dim_tokens`

---

## Alias, View and Copy Operations

The analyzer (`solar/analysis/graph_analyzer.py`) bills every layer of the
einsum graph for compute (`macs` / `other_ops`) and for DRAM traffic in two
modes: **unfused** (every layer reads its inputs and writes its outputs) and
**fused** (only model inputs, parameters and model outputs reach DRAM; graph
intermediates stay on chip).  Operations that only re-describe an existing
storage must not be billed as if they moved or computed anything.  The
convention is explicit and tested (`tests/test_alias_view_accounting.py`,
`tests/test_graph_analyzer_regression.py`):

| Class | Captured ops | other_ops | Unfused traffic | Fused traffic |
|---|---|---|---|---|
| **Pure alias** (`ALIAS_OPS`) | `detach`, `detach_`, `alias`, `view_as`, `swapaxes`, `swapdims`, `movedim`, `moveaxis`, `as_strided`, `unbind` | 0 | 0 | 0 for an intermediate; for the tested `detach` single-output cases a model input or output that passes through the alias is counted once at the boundary (see the limitations below for the other members) |
| **Layout-dependent alias** (`LAYOUT_DEPENDENT_ALIAS_OPS`) | `reshape_as` (PyTorch: a view *or* a new tensor) | 0 | 0 only when the source's contiguous layout is **established from recorded evidence**: contiguity recorded at capture for a model input or parameter, a fresh row-major result (`matmul`/`mm`/`bmm`/`linear`/`addmm`/`embedding`), a recorded `memory_format` argument that guarantees it (`contiguous()` without one, `clone(memory_format=torch.contiguous_format)`), or a layout-preserving view / layout-inheriting op whose data inputs are all established. Otherwise, including every **unknown** layout, the **copy is charged**: source read + output write. Operation names alone never prove a layout: `clone` preserves a dense transposed layout, `einsum('ij->ji')` stays transposed, `contiguous`/`to` with `memory_format=channels_last` are channels-last. The shape-reference operand is never a data read. The decision is recorded per layer in `alias_qualification` (`alias (layout established: ...)` or `copy_charged (layout unknown: charged as copy)`) | as for the alias / copy case respectively |
| **Zero-copy view** (pre-existing) | `view`, `reshape`, `expand`, `expand_as`, `transpose`, `permute`, `t`, `squeeze`, `unsqueeze`, `flatten`, `unflatten`, `unfold`, `chunk`, `split`, `tensor_split` | 0 | 0 | 0 (the consumer accounts for the read) |
| **Slice view** | `__getitem__`, `narrow`, `slice`, `select` | 0 | output-sized read | boundary rules as for any layer |
| **Real copy** | `clone`, `copy_`, `to` / `type` casts, `cat`, `stack`, `repeat`, `pad`, `roll`, `flip` | 0 (memory-bound, no ALU work) | full read + write | boundary rules as for any layer |
| **`contiguous`** | `contiguous` | 0 | 0 | 0 |

Notes:

* `tensor.detach()` returns a view of the same storage; `x.clone().detach()`
  (KernelBench level2/20, level2/40) is therefore one real copy, not two.
  Before this convention the `detach` layer was billed one op per element and
  a full read and write in unfused mode (issue #35, an earlier decision).
* `contiguous()` copies only when its input is not contiguous, which the
  captured graph cannot prove.  It keeps the pre-existing zero-copy-view
  treatment; this is an optimistic modelling choice, not a proof, and it is
  listed here so the reviewer can see it.
* Graph connectivity is never changed by the convention: an alias layer stays
  in the graph and its producers and consumers resolve through it. For the
  **tested cases** (`detach` with a single output, returned directly or
  consumed by one op; Reformer returns detached tensors) the fused model
  keeps the DRAM write of a model output that is returned through the alias.
* MACs are never affected: aliases and copies have none.
* `reshape_as` is billed as an alias only when the source layout is
  established from recorded evidence (`alias_qualification: alias (layout
  established: ...)`); `x.t().reshape_as(other)` materialises a copy and is
  charged 24 + 24 elements for a 24-element tensor, and so is every source
  whose layout is unknown (`alias_qualification: copy_charged (layout unknown:
  charged as copy)`). A model input or parameter counts as established only
  because the capture now records its contiguity (`module_args.layout` on the
  tensor nodes of `pytorch_graph.yaml`, `tensor_layouts` in
  `einsum_graph.yaml`), **bound to the identity of the traced tensor** (the
  recorder tensor torchview creates for every input, nested lists/tuples/dicts
  included) or of the `nn.Parameter` object, never matched by shape; an
  unrecorded or non-contiguous input or parameter is charged. A transposed
  model input or `nn.Parameter` (stride `[1, 6]`) is therefore charged, not
  reported as a free alias, and a same-shape contiguous input cannot donate
  its record. `matmul`/`mm`/`bmm`/`linear`/`addmm` count as fresh row-major
  allocations only when the recorded call has no `out=` destination.
  The same layout dependence exists for the pre-existing `reshape`,
  `flatten` and `contiguous` conventions, which still assume a view; they are
  not changed here and remain an open correction.

#### Boundary classification: current limitation and historical observations

* **Current limitation — shape-reference operands counted as data in fused
  mode.** `view_as(other)` reads `other` as data in **fused** mode (72 instead
  of 48 fused elements; in unfused mode `view_as` is a pure alias with zero
  traffic): every captured boundary operand is treated as a data read.
  `reshape_as` zeroes its shape operand under this convention; `view_as` and
  other shape-reference ops do not yet. Reproduced on the lineage base
  (reviewer boundary probe `test_view_as_shape_operand_is_not_a_data_read`).
* **Historical observations at the earlier pin (the earlier baseline and before), resolved
  by the lineage's boundary classification, kept for the record:**
  a model output returned through a pre-existing zero-copy view
  (`(x * 2).view(2, 12)`, `.contiguous()` on contiguous data) lost its DRAM
  write in fused mode (24 instead of 48 elements); and for `unbind` (any
  multi-output layer) one `any(...)` decision over all outputs assigned the
  layer's aggregate output size to each terminal output name (48 instead of
  24 fused elements when all three outputs of `unbind(x, 0)` on a `[3, 4]`
  input are returned; 16 instead of 24 for `a * 2, b, c`). The unchanged
  reviewer probes for these four cases pass on the lineage base composition; they
  are not current limitations of this tree.
* **Receiver provenance for `reshape_as`.** A positive layout proof requires
  the receiver to be a connected graph operand bound by tensor identity: the
  capture records, per call, whether the first positional tensor argument IS
  one of the node's operands (`call_operands.receiver_bound`, keyword form
  included; counts and shapes of the recorded arguments are not evidence). A
  receiver that never becomes a graph operand (a registered buffer or a plain
  tensor attribute, positional or `other=` keyword spelling) is charged as a
  copy at the output's size, marked `status: incomplete` /
  `incomplete_reason: reshape_as_receiver_unbound` (`capture_incomplete`
  carries the detail), its read is accounted as an external read in the
  fused totals, the total and the perf workload carry `status: incomplete`,
  and its output layout is unknown for every downstream proof (a chained
  second `reshape_as` is charged). Before this rule (heads up to the earlier rule) a
  same-shaped shape-reference tensor could lend its layout in the keyword and
  chained forms; since it, the shape-reference tensor's layout is never
  borrowed. Layout records themselves are bound to the traced tensor's
  identity (nested inputs included) and to the parameter object, never
  matched by shape. Any recorded `out=` destination disables both the
  fresh-allocation and the layout-inheritance proofs (`add(x, x, out=dst)`
  keeps `dst`'s strides); the destination is torch's aliasing output, not an
  operand, so the capture binds the remaining operands and records it as
  `call_operands.destination` instead of refusing the call.

The current limitation above is a documented accounting gate for affected
routes; the historical items are retained so earlier receipts stay readable.

