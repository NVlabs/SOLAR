<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shape Primitives — Design and Motivation

This document describes the "shape primitives" added to `solar.einsum_lib`:
`slice_at`, `stack`, and `concat`. It covers why they exist, how they work,
and how sequential recurrence (RNNs, LSTMs, etc.) is handled on top of them.

---

## 1. The Problem: Broken Graphs from NumPy Escapes

### Background

The `einsumIR` (`solar.einsum_lib`) represents programs as:
- `@kernel` functions: perfectly nested loops with named dimensions (map directly to einsums)
- `model()` function: call graph connecting kernels

The LLM translates source programs into this IR. The **tracer** records `Tensor`
creation and `@kernel` calls to build a dataflow graph (`.yaml`), which downstream
tools analyze for MACs, memory, and performance.

### What went wrong with RNNs (LSTM, BiLSTM)

When translating recurrent models, the LLM originally generated code like:

```python
def model(x, h0, c0, W_ih, W_hh, ...):
    outputs = []
    h, c = h0, c0
    for t in range(seq_len):
        x_t = x.data[:, t, :]                              # <-- NumPy escape!
        h, c = lstm_cell(Tensor.from_numpy(x_t, ...), h, c, ...)
        outputs.append(h)
    result = Tensor.from_numpy(np.stack([o.data for o in outputs]), ...)  # <-- escape!
    return result
```

This is **numerically correct** (verification passes), but structurally broken because:

1. **`.data` access** escapes the `Tensor` tracking system. The tracer doesn't see
   the relationship between `x` and `x_t` — so `x_t` becomes a phantom input
   (`__const_XXXXX`).

2. **`Tensor.from_numpy(...)`** on intermediate results creates a brand-new traced
   tensor with no producer in the graph — another phantom input.

3. **Broken edges**: because each timestep's `h` is re-wrapped via `Tensor.from_numpy`,
   the graph shows no data dependency between consecutive timesteps — making the
   graph useless for analysis.

The bug was never the Python `for` loop itself — a plain `for t in range(T)`
with proper `Tensor` objects threaded through it would have traced fine. The bug
was the NumPy escapes that silently left the tracer's view of the dataflow.

### Why softmax didn't have this problem

Softmax is decomposed into multiple `@kernel` loop nests (max, exp, sum,
normalize). It has no loop-carried dependencies across iterations and all
intermediate tensors stay within the `Tensor` system, so no escapes are needed.

Recurrence is different: it requires state carried across iterations of a
Python-level loop that surrounds multiple kernel calls. The `@kernel`
decorator forbids loop-carried dependencies within its loop nests (because
each nest must be perfectly nested and parallelizable). So the sequential
iteration MUST happen at the `model()` level — and the model level previously
had no traced vocabulary for slicing / stacking without falling back to NumPy.

---

## 2. The Solution: Four Traced Primitives

We added four first-class operations that handle "shape plumbing" without
escaping the `Tensor`/tracer system. All are defined in `solar/einsum_lib/core.py`.

### `slice_at(tensor, axis, index) -> Tensor`

Extracts one slice along a named axis, dropping that dimension.

```python
h0_layer0 = slice_at(h0, axis="L", index=0)  # "L,B,H" -> "B,H"
x_t       = slice_at(x, axis="T", index=t)   # "B,T,H" -> "B,H"
```

**Implementation**: `np.take(tensor.data, index, axis=axis_pos)` plus a
`"builtin"` trace entry with proper input->output edges.

### `stack(tensors, new_dim, size=None) -> Tensor`

Stacks a list of tensors along a new leading dimension.

```python
h_all = stack(h_per_step, new_dim="T")  # list of "B,H" -> "T,B,H"
```

**Implementation**: `np.stack([t.data for t in tensors], axis=0)` plus a
`"builtin"` trace entry.

### `concat(tensors, along_dim) -> Tensor`

Concatenates along an existing named dimension.

```python
h_bidir = concat([h_fwd, h_bwd], along_dim="H")  # doubles H
```

**Implementation**: `np.concatenate(...)` plus a `"builtin"` trace entry.

### `gather(table, index, axis) -> Tensor`

Gathers slices from `table` along a named axis using integer indices from
`index`. The axis dimension is replaced by the dimensions of `index` in
the output. Covers `nn.Embedding` lookups and `table[index_tensor]` patterns
that would otherwise require data-dependent indexing (`A[B[i]]`) inside
kernels — which the DSL forbids.

```python
tok = gather(word_emb, input_ids, axis="V")  # "V,H" + "B,S" -> "B,S,H"
rpb = gather(bias_table, rel_pos_idx, axis="T")  # "T,NH" + "NN" -> "NN,NH"
```

**Implementation**: `np.take(table.data, index.data.astype(int), axis=axis_pos)`
plus a `"builtin"` trace entry. Without this primitive, the LLM must either
violate the `A[B[i]]` rule inside a kernel (incorrect analysis) or construct
a dense one-hot selector matrix and use a matmul workaround (inflated MACs).

---

## 3. Recurrence: Unrolled Python Loops

With the three primitives in place, recurrence is written as a **plain Python
`for` loop inside `model()`**, threading carried state through ordinary
Python variables. Per-step inputs come from `slice_at`; per-step outputs are
appended to a list and assembled with `stack` at the end.

```python
def model(x, h0, W_ih, b_ih, W_hh, b_hh):
    seq_len = x.shape[1]
    ih_all = input_proj(x, W_ih, b_ih)           # precompute for all t
    state = slice_at(h0, axis="L", index=0)
    outputs = []
    for t in range(seq_len):
        x_t = slice_at(ih_all, axis="T", index=t)
        g_hh = hh_proj(state, W_hh, b_hh)
        state = add_tanh(x_t, g_hh)
        outputs.append(state)
    all_h = stack(outputs, new_dim="T")
    return transpose_tb(all_h)
```

Every intermediate is a `Tensor`; the tracer sees every edge. No phantom inputs,
no broken dependencies — the graph is correct by construction.

### Why not a dedicated `scan` primitive?

An earlier version of `einsum_lib` provided a `scan(body_fn, carried, ...)`
primitive that compiled recurrence to a single IR node with a body sub-graph.
It was removed in favor of plain unrolling for three reasons:

1. **The phantom-input problem never needed scan.** `slice_at`/`stack`/`concat`
   alone are sufficient to eliminate NumPy escapes. `scan` was a
   *compactness* optimization on top — not a correctness fix.

2. **Numba acceleration works natively for unrolled loops.** The fast path
   in `numba_accel.py` replaces `@kernel` functions with Numba-JIT'd versions
   in a new namespace, then runs `model()` there. Unrolled kernel calls in
   `model()` resolve directly to the Numba versions. A `scan` body function
   lives at module scope, so its kernel references resolve through the
   *original* (slow) wrappers, defeating the acceleration. Removing scan
   eliminates that corner case.

3. **The downstream pipeline handles flat kernel chains natively.** Unrolling
   produces `T` copies of each body kernel in the graph; every stage after
   `einsum_lib` (assembler, analyzer, perf model) already handles ordinary
   kernel calls, so no special-casing is needed.

### Accepted trade-offs

- **Graph size**: for a BiLSTM with `T=512` the YAML grows to a few thousand
  kernel nodes. This is linear in `T`, acceptable for downstream tools, and
  the alternative (scan) required new hierarchical-graph support throughout
  the pipeline.
- **LLM output tokens**: the LLM emits a few extra lines of loop body per
  model. At realistic sequence lengths this adds hundreds of tokens, not
  thousands, and is worth it for the simpler IR and Numba wins.
- **`seq_len` baked into trace**: the traced YAML is specialized to one
  concrete sequence length. Re-running the trace at a different length is a
  single-digit-seconds operation.

---

## 4. Summary Table

| Primitive | Why needed | Replaces | MACs |
|-----------|-----------|----------|------|
| `slice_at` | Extract per-layer / per-step tensor without `.data` escape | `tensor.data[idx]`, `np.take` | 0 |
| `stack` | Reassemble per-step outputs without `np.stack` escape | `np.stack([t.data for t in ...])` | 0 |
| `concat` | Combine fwd/bwd hidden states without `np.concatenate` escape | `np.concatenate(...)` | 0 |
| `gather` | Embedding / index-select without `A[B[i]]` inside kernels | `nn.Embedding`, `table[index]`, one-hot matmul workaround | 0 |

All four are zero-compute data-movement operations. They are recorded as
`type: builtin` nodes in the graph, distinct from `type: kernel` compute
nodes, so the analyzer can skip them when accumulating FLOPs.

---

## 5. Open Items

- **`scatter` / `pad`**: other structural operations that can cause phantom
  inputs if the LLM reaches for NumPy. Potential future additions.
- **Downstream handling of `type: builtin` nodes**: `slice_at` / `stack` /
  `concat` / `gather` emit zero-MAC nodes. The analyzer identifies them via
  `_EINSUM_LIB_BUILTINS` and sets `ops_cost = 0`; the assembler routes them
  as pass-through edges rather than compute kernels.
