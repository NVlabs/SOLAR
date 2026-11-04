# solar.einsum_lib — JAX/Flax Kernel Translation Reference

Translate the JAX/Flax source program into Python functions using `solar.einsum_lib`. Each kernel becomes one `@kernel` function with explicitly named dimensions on all tensors. A `model()` function defines the call graph.

## Rules

- One `@kernel` function per kernel. Each function = one operation.
- Annotate every input/output with named dims: `A: Tensor("B,M,K", dtype="fp16")`.
- Mark weight tensors: `Tensor("K,N", role="weight", dtype="fp16")`.
- If a tensor has a known sparsity (density), set `sparsity=<value>` on the `Tensor.from_numpy()` call. The value is the fraction of non-zero elements (0-1). This metadata is used for FLOPs/bytes analysis.
- When model buffers are listed (e.g. batch statistics from BatchNorm), use them as-is via `Tensor.from_numpy()` with `role="weight"`. Do NOT recompute values that buffers already provide. For example, if `batch_stats_BatchNorm_0_mean` and `batch_stats_BatchNorm_0_var` are provided as buffers, use them directly instead of computing mean/variance from the input.
- Every tensor MUST have `dtype=` set. Infer from the source program. Default `"fp32"` if unspecified.
- Supported dtypes: `fp64`, `fp32`, `tf32`, `fp16`, `bf16`, `fp8`, `int8`, `nvfp4`.
- Create the output tensor inside the function with sizes computed from input `.shape`: `C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype="fp16")`. Do NOT hardcode literal dimension sizes.
- Tensors are zero-initialized by default. Use `fill=` for reductions that need a different identity element: `fill=float("-inf")` for max-reduction, `fill=float("inf")` for min-reduction.
- For derived dimensions (e.g. num_heads, head_dim), compute them from input shapes: `head_dim = X.shape[2] // num_heads`.
- One `for` loop per dimension, using `Dim("name")`.
- Mark the outermost loop that has no loop-carried dependency with `parallel=True`: `for b in Dim("B", parallel=True)`. A loop has a loop-carried dependency when one iteration reads a value written by a different iteration of that same loop (e.g. reductions via `+=` or `max=`). Mark only one loop per nest — the outermost dependency-free loop.
- When a loop bound is derived from tensor shapes rather than being a tensor dimension itself (e.g. number of groups, channels per group, tile size), use `Dim("name", size=expr)` to specify the bound explicitly: `for g in Dim("G", size=groups)`. Use computed index expressions to map these loop variables back to tensor dimensions: `X[b, g * channels_per_group + cg, ...]`.
- Index tensors with loop variables only. Affine expressions allowed: `X[b, c, p+r, q+s]`.
- No library calls inside loops — only arithmetic (`+`, `-`, `*`, `/`) and scalar builtins: `max`, `min`, `abs`, `exp`, `log`, `sqrt`.
- Each loop nest must be perfectly nested — computation only in the innermost body. A kernel may contain multiple sequential loop nests over intermediate tensors.
- For softmax (including `jax.nn.softmax`): decompose into perfectly nested loop nests (max-reduction, exp-shift, sum-reduction, normalize). See the softmax example below.
- No data-dependent indexing (`A[B[i]]`), no dynamic control flow.
- Reduction dims = dims that appear in inputs but NOT in the output.
- `model()` function calls kernels in order to define the dataflow graph.

## JAX/Flax-Specific Rules

### Flax parameter naming

Flax parameters are flattened from the variable tree into the inputs dict. The naming convention is:
- `nn.Dense` parameters: `Dense_0_kernel`, `Dense_0_bias`
- `nn.Conv` parameters: `Conv_0_kernel`
- Nested modules: `SelfAttention_0_Dense_0_kernel`

Use the exact names as listed in the "Model parameters" section of the prompt.

### Flax layer mapping

| Flax layer | einsum_lib equivalent |
|---|---|
| `nn.Dense(features)` | Matmul: `Y[b,m,n] += X[b,m,k] * W[k,n]` then add bias |
| `nn.Conv(features, kernel_size)` | Sliding-window conv loops |
| `nn.BatchNorm()` | Use provided buffers (mean/var) directly |
| `nn.relu` / `jax.nn.relu` | `max(0, X[b,m,n])` |
| `nn.softmax` / `jax.nn.softmax` | 4-loop decomposition (see softmax example) |
| `nn.LayerNorm()` | Compute mean, variance, normalize in separate loop nests |

### `jnp.einsum` expansion

When the source contains `jnp.einsum("ij,jk->ik", A, B)`, expand it into explicit `Dim` loops. Do NOT copy the einsum string.

Example: `jnp.einsum("bhsd,bhsd->bhs", Q, K)` becomes:
```python
@kernel
def dot_product(Q: Tensor("B,H,S,D", dtype="fp32"), K: Tensor("B,H,S,D", dtype="fp32")) -> Tensor("B,H,S"):
    R = Tensor(B=Q.shape[0], H=Q.shape[1], S=Q.shape[2], dtype="fp32")
    for b in Dim("B", parallel=True):
        for h in Dim("H"):
            for s in Dim("S"):
                for d in Dim("D"):
                    R[b,h,s] += Q[b,h,s,d] * K[b,h,s,d]
    return R
```

### `jax.nn.one_hot` expansion

`jax.nn.one_hot(indices, num_classes)` becomes a loop with conditional assignment. Since we cannot use conditionals inside loops, express it as: the target class gets value 1.0, all others stay at the default 0.0. This requires a scatter-style write using the index value.

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

### Flax Dense (matmul + bias)

Translating `nn.Dense(features=64)` which has parameters `Dense_0_kernel` (shape K,N) and `Dense_0_bias` (shape N):

```python
@kernel
def dense_matmul(X: Tensor("B,K", dtype="fp32"), W: Tensor("K,N", role="weight", dtype="fp32")) -> Tensor("B,N"):
    Y = Tensor(B=X.shape[0], N=W.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            for k in Dim("K"):
                Y[b,n] += X[b,k] * W[k,n]
    return Y

@kernel
def add_bias(Y: Tensor("B,N", dtype="fp32"), Bias: Tensor("N", role="weight", dtype="fp32")) -> Tensor("B,N"):
    Z = Tensor(B=Y.shape[0], N=Y.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for n in Dim("N"):
            Z[b,n] = Y[b,n] + Bias[n]
    return Z

def model(x, kernel, bias):
    y = dense_matmul(x, kernel)
    z = add_bias(y, bias)
    return z

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,K", dtype="fp32")
    kernel = Tensor.from_numpy(inputs["Dense_0_kernel"], "K,N", role="weight", dtype="fp32")
    bias = Tensor.from_numpy(inputs["Dense_0_bias"], "N", role="weight", dtype="fp32")
    output = model(x, kernel, bias)
    return {"output": output.data}
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

### Softmax decomposition pattern

Whenever softmax appears in the source program (including `jax.nn.softmax`, `nn.softmax`, or `jax.nn.log_softmax`), decompose it into 4 perfectly nested loop nests: (1) max-reduction, (2) exp with shift, (3) sum-reduction, (4) normalize.

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

Do NOT use a `softmax()` function call — always expand it inline.

### Multi-head reshape (derived dimensions)

```python
@kernel
def reshape_to_heads(X: Tensor("B,S,D", dtype="fp32")) -> Tensor("B,H,S,A"):
    num_heads = 4
    head_dim = X.shape[2] // num_heads
    Y = Tensor(B=X.shape[0], H=num_heads, S=X.shape[1], A=head_dim, dtype="fp32")
    for b in Dim("B", parallel=True):
        for s in Dim("S"):
            for h in Dim("H"):
                for a in Dim("A"):
                    Y[b,h,s,a] = X[b, s, h * head_dim + a]
    return Y
```

### Model composition + run_model

```python
def model(x, w):
    return matmul(x, w)

def run_model(inputs: dict) -> dict:
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,M,K", dtype="fp16")
    w = Tensor.from_numpy(inputs["Dense_0_kernel"], "K,N", role="weight", dtype="fp16")
    output = model(x, w)
    return {"output": output.data}
```

All arrays (activation inputs AND model parameters) are passed in the `inputs` dict.
Use `Tensor.from_numpy()` for every one of them. Never create weights with
`Tensor.from_random()` — always read them from `inputs`.
The `dtype` must match the source program's precision for each tensor.

## Do NOT

- Use numpy, jax, jnp, torch, or any library calls inside kernel bodies.
- Use data-dependent indexing or conditionals (e.g., `A[B[i]]`, `if A[i] > 0`).
- Put multiple operations in one kernel — split into separate `@kernel` functions.
- Use a `softmax()` function call — expand it into 4 perfectly nested loop nests (max, exp-shift, sum, normalize).
- Copy `jnp.einsum` strings — expand them into explicit `Dim` loops.
- Put computation between loop levels — each loop nest must be perfectly nested with work only in the innermost body.
- Omit dimension annotations or `dtype=` on inputs/outputs.
- Optimize for performance — no loop unrolling, constant folding, tiling, or precomputation.
- Introduce dead code.
- Make the code input-specific — capture the general operation structure, not particular values.
- Inline literal values from the source — treat all inputs as symbolic.
- Hardcode literal dimension sizes in output tensors — always compute sizes from input `.shape` so the code works at any shape.
