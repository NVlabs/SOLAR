# solar.einsum_lib — Kernel Translation Reference

Translate the source program into Python functions using `solar.einsum_lib`. Each kernel becomes one `@kernel` function with explicitly named dimensions on all tensors. A `model()` function defines the call graph.

## Rules

- One `@kernel` function per kernel. Each function = one operation.
- Annotate every input/output with named dims: `A: Tensor("B,M,K", dtype="fp16")`.
- Mark weight tensors: `Tensor("K,N", role="weight", dtype="fp16")`.
- If a tensor has a known sparsity (density), set `sparsity=<value>` on the `Tensor.from_numpy()` call. The value is the fraction of non-zero elements (0-1). This metadata is used for FLOPs/bytes analysis.
- When model buffers are listed (e.g. running mean, running variance, or other pre-computed statistics), use them as-is via `Tensor.from_numpy()` with `role="weight"`. Do NOT recompute values that buffers already provide. For example, if `bn_running_mean` and `bn_running_var` are provided as buffers, use them directly instead of computing mean/variance from the input.
- Every tensor MUST have `dtype=` set. Infer from the source program. Default `"fp32"` if unspecified.
- Supported dtypes: `fp64`, `fp32`, `tf32`, `fp16`, `bf16`, `fp8`, `int8`, `nvfp4`.
- Create the output tensor inside the function with sizes computed from input `.shape`: `C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype="fp16")`. Do NOT hardcode literal dimension sizes.
- Tensors are zero-initialized by default. Use `fill=` for reductions that need a different identity element: `fill=float("-inf")` for max-reduction, `fill=float("inf")` for min-reduction.
- **Never hardcode architectural constants** (num_heads, groups, window_size, etc.) as integer literals inside kernels. Instead, pass them as **integer entries in the `inputs` dict** and thread them through `run_model()`, `model()`, and kernels as explicit `int` parameters. This allows the code to be verified at small sizes and analyzed at full sizes. For derived dimensions, compute them from input shapes and these parameters: `head_dim = X.shape[2] // num_heads`.
- One `for` loop per dimension, using `Dim("name")`.
- Mark the outermost loop that has no loop-carried dependency with `parallel=True`: `for b in Dim("B", parallel=True)`. A loop has a loop-carried dependency when one iteration reads a value written by a different iteration of that same loop (e.g. reductions via `+=` or `max=`). Mark only one loop per nest — the outermost dependency-free loop.
- When a loop bound is derived from tensor shapes rather than being a tensor dimension itself (e.g. number of groups, channels per group, tile size), use `Dim("name", size=expr)` to specify the bound explicitly: `for g in Dim("G", size=groups)`. Use computed index expressions to map these loop variables back to tensor dimensions: `X[b, g * channels_per_group + cg, ...]`.
- Index tensors with loop variables only. Affine expressions allowed: `X[b, c, p+r, q+s]`.
- No library calls inside loops — only arithmetic (`+`, `-`, `*`, `/`) and scalar builtins: `max`, `min`, `abs`, `exp`, `log`, `sqrt`. These are available as bare names inside `@kernel` bodies — do NOT import them (they are injected by the framework).
- Each loop nest must be perfectly nested — computation only in the innermost body. A kernel may contain multiple sequential loop nests over intermediate tensors.
- For softmax: decompose into perfectly nested loop nests (max-reduction, exp-shift, sum-reduction, normalize). See the softmax example below.
- No data-dependent indexing (`A[B[i]]`) inside `@kernel` bodies — for embedding/lookup operations, use the `gather()` builtin instead. No dynamic control flow inside kernels.
- Reduction dims = dims that appear in inputs but NOT in the output.
- `model()` function calls kernels in order to define the dataflow graph.

## Dimension naming

| Name | Meaning | Typical use |
|------|---------|-------------|
| B | batch | all ops |
| M, N | spatial / sequence | matmul, elementwise |
| K | reduction (contracted) | matmul |
| C | input channels | conv |
| O | output channels | conv |
| H, W | input height, width | conv |
| P, Q | output height, width | conv (P=H-R+1) |
| R, S | filter height, width | conv |

## Examples

### Matmul (reduction over K)

```python
@kernel
def matmul(A: Tensor("B,M,K", dtype="fp16"), W: Tensor("K,N", role="weight", dtype="fp16")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype="fp16")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                for k in Dim("K"):
                    C[b,m,n] += A[b,m,k] * W[k,n]
    return C
```

### Elementwise ReLU (no reduction)

```python
@kernel
def relu(A: Tensor("B,M,N", dtype="fp16")) -> Tensor("B,M,N"):
    C = Tensor(B=A.shape[0], M=A.shape[1], N=A.shape[2], dtype="fp16")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                C[b,m,n] = max(0, A[b,m,n])
    return C
```

### Transpose (permuted output dims)

```python
@kernel
def transpose(A: Tensor("B,M,N", dtype="fp16")) -> Tensor("B,N,M"):
    C = Tensor(B=A.shape[0], N=A.shape[2], M=A.shape[1], dtype="fp16")
    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            for n in Dim("N"):
                C[b,n,m] = A[b,m,n]
    return C
```

### Conv2d (sliding window)

```python
@kernel
def conv2d(X: Tensor("B,C,H,W", dtype="fp16"), W: Tensor("O,C,R,S", role="weight", dtype="fp16")) -> Tensor("B,O,P,Q"):
    Y = Tensor(B=X.shape[0], O=W.shape[0], P=X.shape[2]-W.shape[2]+1, Q=X.shape[3]-W.shape[3]+1, dtype="fp16")
    for b in Dim("B", parallel=True):
        for o in Dim("O"):
            for p in Dim("P"):
                for q in Dim("Q"):
                    for c in Dim("C"):
                        for r in Dim("R"):
                            for s in Dim("S"):
                                Y[b,o,p,q] += X[b,c,p+r,q+s] * W[o,c,r,s]
    return Y
```

### Scatter pattern (strided writes to a larger output)

When an operation writes each input element to strided or offset positions in a larger output (e.g. transposed convolution, upsampling, pixel shuffle), use the **scatter** pattern: allocate a full-sized output and use affine index expressions to place values at the correct positions.

```python
@kernel
def scatter(X: Tensor("B,C,H,W", dtype="fp32")) -> Tensor("B,C,FH,FW"):
    stride = 2
    full_h = (X.shape[2] - 1) * stride + 1
    full_w = (X.shape[3] - 1) * stride + 1
    Z = Tensor(B=X.shape[0], C=X.shape[1], FH=full_h, FW=full_w, dtype="fp32")
    for b in Dim("B", parallel=True):
        for c in Dim("C"):
            for h in Dim("H"):
                for w in Dim("W"):
                    Z[b, c, h * stride, w * stride] = X[b, c, h, w]
    return Z
```

If the result needs trimming (e.g. removing padding), follow with a **crop** kernel:

```python
@kernel
def crop(Z: Tensor("B,C,FH,FW", dtype="fp32")) -> Tensor("B,C,P,Q"):
    pad = 1
    Y = Tensor(B=Z.shape[0], C=Z.shape[1], P=Z.shape[2] - 2 * pad, Q=Z.shape[3] - 2 * pad, dtype="fp32")
    for b in Dim("B", parallel=True):
        for c in Dim("C"):
            for p in Dim("P"):
                for q in Dim("Q"):
                    Y[b, c, p, q] = Z[b, c, p + pad, q + pad]
    return Y
```

These two patterns compose: scatter into a large intermediate, then crop to the final size. Adapt to any number of spatial dimensions.

### Grouped channels

When an operation partitions channels into independent groups (e.g. grouped convolution), the group index and channel-within-group index are not tensor dimensions — they are derived from the channel dimension. Pass `groups` as an `int` parameter and use `Dim("G", size=groups)` and `Dim("CG", size=channels_per_group)` with computed tensor indices:

```python
# groups is passed as an int parameter, not hardcoded
cpg = X.shape[1] // groups
for g in Dim("G", size=groups):
    for cg in Dim("CG", size=cpg):
        # map back to the channel dimension
        ... X[b, g * cpg + cg, h, w] ...
```

### Softmax decomposition pattern

Whenever softmax appears in the source program, decompose it into 4 perfectly nested loop nests: (1) max-reduction, (2) exp with shift, (3) sum-reduction, (4) normalize. The softmax dimension is the one being reduced over. All other dimensions are outer loops. Use `fill=float("-inf")` to initialize the max tensor.

Example — softmax over dimension N with outer dimension B:

```python
@kernel
def my_softmax(X: Tensor("B,N", dtype="fp32")) -> Tensor("B,N"):
    MaxX = Tensor(B=X.shape[0], dtype="fp32", fill=float("-inf"))
    E = Tensor(B=X.shape[0], N=X.shape[1], dtype="fp32")
    SumE = Tensor(B=X.shape[0], dtype="fp32")
    P = Tensor(B=X.shape[0], N=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            MaxX[b] = max(MaxX[b], X[b,n])
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            E[b,n] = exp(X[b,n] - MaxX[b])
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            SumE[b] += E[b,n]
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            P[b,n] = E[b,n] / SumE[b]
    return P
```

Apply this pattern to any number of outer dimensions. For example, attention softmax over N with outer dims B,H,M would have `MaxS = Tensor(B=..., H=..., M=..., fill=float("-inf"))` and 4 loop nests each with `for b / for h / for m / for n`.

Do NOT use a `softmax()` function call — always expand it inline.

### Multi-head reshape (derived dimensions)

When splitting a dimension into heads (e.g. D -> H x A), pass the architectural constant as an `int` parameter and compute derived sizes from input shapes:

```python
@kernel
def reshape_to_heads(X: Tensor("B,S,D", dtype="fp32"), num_heads: int) -> Tensor("B,H,S,A"):
    head_dim = X.shape[2] // num_heads
    Y = Tensor(B=X.shape[0], H=num_heads, S=X.shape[1], A=head_dim, dtype="fp32")
    for b in Dim("B", parallel=True):
        for s in Dim("S"):
            for h in Dim("H"):
                for a in Dim("A"):
                    Y[b,h,s,a] = X[b, s, h * head_dim + a]
    return Y
```

Here `num_heads` is passed in from the `inputs` dict (see below). `head_dim` is derived from `X.shape[2]`. `B` and `S` come from the input tensor directly.

### Scalar inputs (0-d tensors)

Scalar inputs have shape `()` and no dimension names. Use `Tensor("", dtype="fp32")` for the annotation and index with `S[()]` in the kernel body:

```python
@kernel
def scale(X: Tensor("B,M,N", dtype="fp32"), S: Tensor("", dtype="fp32")) -> Tensor("B,M,N"):
    Y = Tensor(B=X.shape[0], M=X.shape[1], N=X.shape[2], dtype="fp32")
    for b in Dim("B"):
        for m in Dim("M"):
            for n in Dim("N"):
                Y[b,m,n] = X[b,m,n] * S[()]
    return Y
```

In `run_model`, load with: `s = Tensor.from_numpy(inputs["x_1"], "", dtype="fp32")`

### Masked operations (captured masks)

If the prompt lists **captured masks** (keys starting with `captured_mask_`), these are boolean/binary tensors captured from the original model during an instrumented run. They represent masking patterns (causal masks, random attention masks, block-sparse masks, etc.).

**Convention**: masks use `1 = keep`, `0 = masked out`. Each mask listing includes a `masked_value` — use it as the `masked_value=` argument in `apply_mask`.

Call the **builtin** `apply_mask`. Do NOT define `apply_mask` as a `@kernel` — it is a predefined function provided by the library. Import it:

```python
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask, slice_at, stack, concat, scan, gather
```

`apply_mask(tensor, mask, masked_value=...)` replaces masked positions (where mask is `0`) with `masked_value`.

Example — attention with a captured mask (where the prompt lists `masked_value: -inf`):

```python
def model(q, k, v, mask):
    scores = attention_scores(q, k)
    scaled = scale_scores(scores)
    masked = apply_mask(scaled, mask, masked_value=-1e9)  # builtin, use listed masked_value
    probs  = attention_softmax(masked)
    output = attention_output(probs, v)
    return output
```

Load captured masks in `run_model` via `Tensor.from_numpy(inputs["captured_mask_0"], "M,N", dtype="fp32", sparsity=<value>)`. The builtin broadcasts the mask across batch dimensions and propagates sparsity metadata automatically.

### Embedding / index-select lookup (`gather` builtin)

When the source program uses `nn.Embedding`, `torch.index_select`, or any `table[indices]` pattern, use the `gather` builtin. Do NOT attempt to implement embedding as a matmul with one-hot vectors or as data-dependent indexing inside a `@kernel`.

`gather(table, index, axis)` gathers slices from `table` along a named `axis` using an integer `index` tensor. The `axis` dimension of `table` is replaced by all dimensions of `index` in the output.

Parameters:
- `table`: Tensor containing the lookup table (e.g. embedding weights), with `axis` as one of its named dimensions
- `index`: Tensor of integer indices (e.g. token IDs), with `dtype="int64"`
- `axis`: name of the dimension in `table` to index into (e.g. `"V"` for vocab)

```python
# nn.Embedding(vocab_size=50000, embed_dim=768) with input token_ids [batch, seq]
# embed_weight has dims "V,D" (vocab_size x embed_dim), role="weight"
# token_ids has dims "B,S" (batch x sequence), dtype="int64"
embedded = gather(embed_weight, token_ids, axis="V")  # -> dims "B,S,D"
```

For models with both token and position embeddings:

```python
def model(token_ids, pos_ids, tok_embed_w, pos_embed_w, ...):
    tok_emb = gather(tok_embed_w, token_ids, axis="V")   # "V,D" + "B,S" -> "B,S,D"
    pos_emb = gather(pos_embed_w, pos_ids, axis="P")     # "P,D" + "B,S" -> "B,S,D"
    x = add_embeddings(tok_emb, pos_emb)                  # @kernel: elementwise add
    # ... transformer layers ...
```

In `run_model`, load integer index tensors with their original dtype:

```python
token_ids = Tensor.from_numpy(inputs["x_0"], "B,S", dtype="int64")
embed_w = Tensor.from_numpy(inputs["model_embed_tokens_weight"], "V,D", role="weight", dtype="fp32")
embedded = gather(embed_w, token_ids, axis="V")
```

### Model composition + run_model

```python
def model(x, w):
    return matmul(x, w)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K", dtype="fp16")
    w = Tensor.from_numpy(inputs["weight"], "K,N", role="weight", dtype="fp16")
    output = model(x, w)
    return {"output": output.data}
```

All arrays (activation inputs AND model parameters) are passed in the `inputs` dict.
Use `Tensor.from_numpy()` for every one of them. Never create weights with
`Tensor.from_random()` — always read them from `inputs`.
The `dtype` must match the source program's precision for each tensor.

### Integer hyperparameters in `inputs`

Architectural constants that affect tensor shapes (num_heads, groups, window_size, etc.) are passed as plain `int` entries in the `inputs` dict. Read them directly (no `Tensor.from_numpy`) and thread them through `model()` and kernels:

```python
def model(x, q_proj_w, k_proj_w, v_proj_w, out_proj_w, num_heads):
    q = linear_q_proj(x, q_proj_w)
    q_heads = reshape_to_heads(q, num_heads)
    ...

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,S,H", dtype="fp32")
    q_proj_w = Tensor.from_numpy(inputs["q_proj_weight"], "H2,H", role="weight", dtype="fp32")
    num_heads = inputs["num_heads"]  # plain int, not a Tensor
    output = model(x, q_proj_w, ..., num_heads)
    return {"output": output.data}
```

This allows the same generated code to be verified at small sizes (e.g. `num_heads=2`) and analyzed at full sizes (e.g. `num_heads=32`).

## Do NOT

- Use numpy, torch, or any library calls inside kernel bodies.
- Use data-dependent indexing inside `@kernel` bodies — for embedding/lookup operations, use the `gather()` builtin instead.
- Put multiple operations in one kernel — split into separate `@kernel` functions.
- Use a `softmax()` function call — expand it into 4 perfectly nested loop nests (max, exp-shift, sum, normalize).
- Put computation between loop levels — each loop nest must be perfectly nested with work only in the innermost body.
- Omit dimension annotations or `dtype=` on inputs/outputs.
- Optimize for performance — no loop unrolling, constant folding, tiling, or precomputation.
- Introduce dead code
- Make the code input-specific — capture the general operation structure, not particular values.
- Inline literal values from the source — treat all inputs as symbolic.
- Hardcode literal dimension sizes in output tensors — always compute sizes from input `.shape` so the code works at any shape.
- Hardcode architectural constants (num_heads, groups, window_size, etc.) as integer literals inside kernels — pass them as `int` entries in the `inputs` dict instead.
