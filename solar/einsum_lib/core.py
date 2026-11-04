# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core classes for the einsum_lib kernel representation library.

Provides Tensor, Dim, and kernel utilities for writing executable loop-nest
programs with named dimensions that can be mechanically compiled to einsums.
"""

import sys
import threading
import functools
import inspect
import re
from contextlib import contextmanager

import numpy as np


VALID_DTYPES = frozenset({
    "fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int32", "int64", "nvfp4",
})

_BYTES_PER_ELEMENT = {
    "fp64": 8,
    "fp32": 4,
    "tf32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "nvfp4": 0.5,
}


_INTEGER_DTYPES = frozenset({"int8", "int32", "int64"})

def _numpy_dtype(dtype):
    """Return the numpy dtype to use for internal storage."""
    if dtype in _INTEGER_DTYPES:
        return {"int8": np.int8, "int32": np.int32, "int64": np.int64}[dtype]
    return np.float64


_NARROW_FLOAT_DTYPES = frozenset({"bf16", "fp16"})


def _truncate_to_dtype(arr: np.ndarray, dtype_str: str) -> np.ndarray:
    """Round fp64 values to the precision of a narrower float dtype.

    Used after each kernel so the IR has the same intermediate rounding
    as the reference model.  Only bf16 and fp16 are truncated; fp32/fp64
    are left untouched.
    """
    if dtype_str == "bf16":
        import torch
        return (torch.from_numpy(arr.astype(np.float32))
                .to(torch.bfloat16)
                .to(torch.float64)
                .numpy())
    if dtype_str == "fp16":
        return arr.astype(np.float16).astype(np.float64)
    return arr


def fp32(x):
    """Round a scalar (or array) to the nearest float32 value and hand it back
    as float64 — the DSL's explicit fp32 rounding-boundary emulation
    (DSL-SIGMOID-FP32-BOUNDARY-1 / snap-mac-1).

    Semantically a cast, not arithmetic: the converter classifies it as a
    zero-cost elementwise cast (no MACs), unlike the inline grid idiom
    ``round(x / 2**k) * 2**k`` whose ``*`` / ``**`` make a reduction loop look
    like a multiply-accumulate.  Round-to-nearest-even including subnormals;
    values beyond the float32 range become ``inf`` as they do in torch's fp32
    arithmetic.  The Numba lowering maps a call to ``np.float64(np.float32(x))``.
    """
    with np.errstate(over="ignore"):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32).astype(np.float64)
        return float(np.float32(x))


def _shape_only_zeros(shape, np_dt):
    """Zero-memory placeholder with the correct shape for shape-only tracing.

    A zero-stride view over a single element has the right ``.shape`` (all the
    trace pass needs to record the einsum graph for SOL analysis) without
    allocating the full buffer.  This is critical for large intermediates —
    e.g. an attention score matrix ``(B, H, S, S)`` at full seq_len — which
    would otherwise OOM the trace pass *after* verification has already
    succeeded on the capped shapes.  Unlike ``np.broadcast_to`` the view is
    WRITABLE: kernels that update tensors in place (e.g. double-write RoPE,
    cache updates) all write into the same single cell, which is harmless in a
    shape-only trace where data values are never read for correctness.
    """
    base = np.zeros(1, np_dt)
    return np.lib.stride_tricks.as_strided(
        base, shape=tuple(shape), strides=(0,) * len(shape))


def is_shape_only_placeholder(arr) -> bool:
    """True for the zero-storage placeholder :func:`_shape_only_zeros` hands out in a
    shape-only trace (an array with dimensions whose strides are all zero)."""
    return (isinstance(arr, np.ndarray) and arr.ndim > 0
            and all(s == 0 for s in arr.strides))


def _validate_dtype(dtype):
    if dtype not in VALID_DTYPES:
        raise ValueError(
            f"Unknown dtype '{dtype}'. "
            f"Supported: {sorted(VALID_DTYPES)}"
        )


class _DimRegistry:
    """Thread-local registry mapping dimension names to integer sizes."""

    def __init__(self):
        self._local = threading.local()

    def _get_dims(self):
        if not hasattr(self._local, "dims"):
            self._local.dims = {}
        return self._local.dims

    def register(self, name: str, size: int):
        dims = self._get_dims()
        dims[name] = size

    def get_size(self, name: str) -> int:
        dims = self._get_dims()
        if name not in dims:
            raise KeyError(
                f"Dimension '{name}' is not registered. "
                f"Create a Tensor with this dimension first."
            )
        return dims[name]

    def clear(self):
        self._get_dims().clear()

    @contextmanager
    def scope(self):
        """Save current dims, clear, yield, then restore on exit.

        Each kernel starts with a clean registry so that dimension names
        can be reused across kernels with different sizes (e.g. input W=32
        vs output W=64 in a transposed convolution).  The ``@kernel``
        wrapper re-registers dims from the kernel's own input tensors.
        """
        saved = dict(self._get_dims())
        self._get_dims().clear()
        try:
            yield
        finally:
            self._local.dims = saved


_registry = _DimRegistry()


# ---------------------------------------------------------------------------
# Runtime tracer — records tensor creation and kernel/builtin calls
# ---------------------------------------------------------------------------

class _Tracer:
    """Thread-local tracer for capturing the dataflow graph at runtime."""

    def __init__(self):
        self._local = threading.local()

    @property
    def active(self):
        return getattr(self._local, "active", False)

    @property
    def entries(self):
        return getattr(self._local, "entries", [])

    def start(self, shape_only: bool = False):
        from .output_provenance import OutputBindings
        self._local.active = True
        self._local.shape_only = shape_only
        self._local.entries = []
        self._local.tensor_counter = 0
        self._local.input_ids = {}
        # Return provenance (issue #59, NONRETURNED-OUTPUT-BILLING-1): which
        # traced tensors run_model() actually returns, bound by live array
        # identity inside THIS trace (an owner token per trace).
        self._local.output_bindings = OutputBindings()
        self._local.owner = object()

    @property
    def shape_only(self):
        return getattr(self._local, "shape_only", False)

    def stop(self):
        self._local.active = False

    def next_id(self):
        self._local.tensor_counter += 1
        return self._local.tensor_counter

    def register_inputs(self, inputs_dict):
        """Map ``id(array)`` to input name for each numpy array in *inputs_dict*.

        Non-array entries (e.g. integer hyperparameters) are skipped.
        """
        self._local.input_ids = {
            id(arr): name for name, arr in inputs_dict.items()
            if isinstance(arr, np.ndarray)
        }
        self._local.output_bindings.register_inputs(inputs_dict)

    def lookup_input_name(self, array_id):
        return getattr(self._local, "input_ids", {}).get(array_id)

    def record_tensor(self, tensor_id, dim_names, shape, dtype, role, sparsity,
                      input_name=None, sparsity_source=None, data=None,
                      input_alias=False):
        if not self.active:
            return
        entry = {
            "type": "tensor",
            "id": tensor_id,
            "dim_names": list(dim_names),
            "shape": tuple(shape),
            "dtype": dtype,
            "role": role,
            "sparsity": sparsity,
            "input_name": input_name,
            "input_alias": bool(input_alias),
        }
        if sparsity_source is not None and sparsity is not None:
            entry["sparsity_source"] = sparsity_source
        self._local.entries.append(entry)
        if data is not None:
            self.bind_tensor_data(tensor_id, data)

    @property
    def owner(self):
        return getattr(self._local, "owner", None)

    def tensor_id(self, tensor):
        if not self.active:
            return None
        if getattr(tensor, "_trace_owner", None) is not self.owner:
            raise ValueError("output provenance: tensor belongs to another trace")
        return getattr(tensor, "_trace_id", None)

    def bind_tensor_data(self, tensor_id, data):
        if self.active and tensor_id is not None:
            rows = [e for e in self.entries if e.get("type") == "tensor" and e.get("id") == tensor_id]
            if len(rows) != 1 or tuple(rows[0]["shape"]) != tuple(data.shape):
                raise ValueError("output provenance: data has missing/conflicting traced shape")
            self._local.output_bindings.bind(tensor_id, data)

    def record_model_outputs(self, outputs):
        """Record which traced tensors (or registered inputs) run_model()
        returned -- the ``dsl-return-v1`` record the metadata builder turns
        into ``model_outputs``.  When a returned array cannot be bound to a
        traced identity (a derived view, a copy, a retired version) the trace
        carries ``model_outputs_unrecorded`` instead and the analyzer fails
        closed on the external-output footprint (never a guess)."""
        if not self.active:
            raise ValueError("output provenance: return record requires active owning trace")
        try:
            leaves = self._local.output_bindings.returns(outputs)
        except ValueError as exc:
            self._local.entries.append({"type": "model_outputs_unrecorded", "reason": str(exc)})
            return
        self._local.entries.append({"type": "model_outputs", "schema": "dsl-return-v1",
                                   "outputs": leaves})

    def mark_trace_only_unrounded(self, tensor_id, dtype):
        """Record on the tensor's trace entry that its narrow-dtype rounding was
        skipped because the run is trace-only (issue #67): the recorded data is
        the zero-storage placeholder, rounding is deferred to execution.  Trace
        entries only -- the metadata builder copies named fields and this one
        is deliberately not among them, so the sidecar is unchanged."""
        if not self.active or tensor_id is None:
            return
        for entry in reversed(getattr(self._local, "entries", [])):
            if entry.get("type") == "tensor" and entry.get("id") == tensor_id:
                entry["trace_only_unrounded"] = dtype
                return

    def update_tensor_sparsity(self, tensor_id, sparsity, source=None):
        """Back-patch a recorded tensor's sparsity (measured post-execution).

        Kernel outputs are recorded at construction (data still zeros); the
        density profiler measures them after the kernel body ran and updates
        the entry here.
        """
        if not self.active or tensor_id is None:
            return
        for entry in reversed(getattr(self._local, "entries", [])):
            if entry.get("type") == "tensor" and entry.get("id") == tensor_id:
                entry["sparsity"] = sparsity
                if source is not None and sparsity is not None:
                    entry["sparsity_source"] = source
                else:
                    entry.pop("sparsity_source", None)
                return

    def record_call(self, name, call_type, input_ids, output_id,
                    call_site_line=None, input_order=None, registration_order=None, unsupported_return=None,
                    scalar_args=None):
        if not self.active:
            return
        entry = {
            "type": "call",
            "name": name,
            "call_type": call_type,
            "input_ids": list(input_ids),
            "output_id": output_id,
            "call_site_line": call_site_line,
        }
        if input_order is not None:
            # "signature": tensor inputs are in the kernel's parameter order
            # (what the converter's param_names mapping assumes).  Absent in
            # artifacts traced before this marker existed.
            entry["input_order"] = input_order
        if registration_order is not None:
            # TARGET-DSL-DIM-KWORDER-7: indices into input_ids in run-time registration (call) order
            entry["registration_order"] = list(registration_order)
        if unsupported_return is not None:
            # TARGET-DSL-TUPLE-13: the kernel returned no Tensor (a tuple / another value); the node has no output
            entry["unsupported_return"] = dict(unsupported_return)
        if scalar_args:
            # TARGET-KB-MAXPOOL: int arguments by parameter name (sizes such as Dim("R", size=kernel_size))
            entry["scalar_args"] = dict(scalar_args)
        self._local.entries.append(entry)


_tracer = _Tracer()
_trace_only = threading.local()
_density_profiling = threading.local()

# Density measurement guards: skip huge tensors (O(n) count is cheap but not
# free) and don't tag effectively-dense tensors.
_DENSITY_MEASURE_MAX_ELEMS = int(2e8)
_DENSITY_DENSE_CUTOFF = 0.999


def set_density_profiling(on: bool, source: str = "measured", *,
                          measure_attention: bool = False):
    """Enable measured-density profiling.

    When ON (and execution is real, i.e. not trace-only/shape-only), every
    ``from_numpy`` input, ``@kernel`` output, and masking builtin measures its
    EXACT density (``count_nonzero / size``) and records it as ``sparsity`` —
    feeding the analyzer's zeros-are-free compute/memory discounts.  Intended
    for an instrumented full-value run at verification (capped) shapes; see
    ``scripts/trace_saved_ir.py --measure-density``. Attention probabilities
    identified by name or role stay dense unless ``measure_attention=True``.
    Explicit ``sparsity=`` annotations are not changed by this policy.
    """
    _density_profiling.active = bool(on)
    _density_profiling.source = str(source)
    _density_profiling.measure_attention = bool(measure_attention)
    _density_profiling.reported = set()


def density_profiling_enabled():
    return getattr(_density_profiling, "active", False)


def density_profiling_source():
    """Label recorded as ``sparsity_source`` on measured densities (e.g.
    "measured" for a same-scale run, "measured_capped" when the profiled run
    executes at capped verification shapes while the yaml records original
    sizes — downstream consumers must then check scale-freeness)."""
    return getattr(_density_profiling, "source", "measured")


def _measure_density(arr, *, name=None, role=None):
    """Exact density (fraction nonzero) or None when not worth tagging."""
    try:
        size = int(arr.size)
    except Exception:
        return None
    if size == 0 or size > _DENSITY_MEASURE_MAX_ELEMS:
        return None
    d = float(np.count_nonzero(arr)) / float(size)
    if d < _DENSITY_DENSE_CUTOFF and not getattr(
            _density_profiling, "measure_attention", False):
        # A conservative naming guard, not a semantic classifier. Authors of
        # opaque/fused kernels can mark the returned Tensor with this role.
        normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", name or "").lower()
        protected = (role == "attention_probability" or re.search(
            r"(?:^|_)(?:softmax|probs?|probabilities)(?:_|$)|"
            r"(?:attn|attention)_weights(?:_|$)", normalized))
        if protected:
            label = name or role
            reported = getattr(_density_profiling, "reported", set())
            if label not in reported:
                print(f"[density] GUARDED {label}: stays dense; opt in with "
                      "--measure-density --measure-attention-density or use "
                      "an explicit sparsity annotation / --density-override",
                      file=sys.stderr)
                reported.add(label)
            return None
    return d if d < _DENSITY_DENSE_CUTOFF else None


def _is_trace_only():
    return getattr(_trace_only, "active", False)


def _profiling_real_run():
    """Profiling is on AND tensors carry real values (not placeholders)."""
    return (density_profiling_enabled()
            and not _is_trace_only()
            and not (_tracer.active and _tracer.shape_only))


def start_trace(trace_only=False):
    """Enable runtime tracing (clears previous trace).

    When *trace_only* is ``True``, ``Dim`` iterators yield nothing so
    loop bodies are skipped.  Tensor constructors and ``@kernel`` /
    ``apply_mask`` hooks still fire, providing the full dataflow graph
    without executing any computation.
    """
    _tracer.start()
    _trace_only.active = trace_only


def stop_trace():
    """Disable tracing (entries remain readable via ``get_trace()``)."""
    _tracer.stop()
    _trace_only.active = False


def get_trace():
    """Return the list of trace entries from the most recent trace run."""
    return list(_tracer.entries)


class Tensor:
    """Named-dimension tensor — executable with numpy, parseable by the compiler.

    Spec mode (for function annotations — no data allocated):
        def fn(A: Tensor("B,M,K"), W: Tensor("K,N", role="weight")): ...

    Data mode (for computation — backed by a numpy array):
        C = Tensor(B=2, M=32, N=128)             # zero-init, explicit sizes
        C = Tensor(B=A.shape[0], M=A.shape[1])   # zero-init, sizes from inputs
        A = Tensor.from_numpy(arr, "B,M,K")      # wrap existing array
        X = Tensor.from_random(B=2, M=32, K=64)  # random data for testing
    """

    def __init__(self, dims=None, *, role=None, dtype="fp32", sparsity=None,
                 fill=0.0, **dim_sizes):
        if dims is not None and dim_sizes:
            raise TypeError("Provide either a dims string or keyword dim sizes, not both")
        _validate_dtype(dtype)
        self.role = role
        self.dtype = dtype
        self.sparsity = sparsity

        if dims is not None:
            if not isinstance(dims, str):
                raise TypeError(f"dims must be a string, got {type(dims).__name__}")
            self.dim_names = [d.strip() for d in dims.split(",") if d.strip()] if dims.strip() else []
            self.data = None
            self.shape = None
        elif dim_sizes:
            self.dim_names = list(dim_sizes.keys())
            self.shape = tuple(dim_sizes.values())
            np_dt = _numpy_dtype(dtype)
            if _is_trace_only():
                # Shape-only trace: avoid materializing the full buffer (loop
                # bodies are skipped, so .data is never indexed).
                self.data = _shape_only_zeros(self.shape, np_dt)
            else:
                self.data = np.full(self.shape, fill, dtype=np_dt)
            for name, size in dim_sizes.items():
                _registry.register(name, size)
            if _tracer.active:
                self._trace_id = _tracer.next_id()
                self._trace_owner = _tracer.owner
                _tracer.record_tensor(
                    self._trace_id, self.dim_names, self.shape,
                    self.dtype, self.role, self.sparsity,
                    data=self.data,
                )
        else:
            # Scalar tensor (0-d) — no dimensions
            self.dim_names = []
            self.shape = ()
            np_dt = _numpy_dtype(dtype)
            self.data = np.full((), fill, dtype=np_dt)
            if _tracer.active:
                self._trace_id = _tracer.next_id()
                self._trace_owner = _tracer.owner
                _tracer.record_tensor(
                    self._trace_id, self.dim_names, self.shape,
                    self.dtype, self.role, self.sparsity,
                    data=self.data,
                )

    @classmethod
    def from_numpy(cls, array, dims, *, role=None, dtype="fp32", sparsity=None):
        """Wrap an existing numpy array with named dimensions."""
        _validate_dtype(dtype)
        orig_array_id = id(array)
        t = cls.__new__(cls)
        np_dt = _numpy_dtype(dtype)
        if _is_trace_only():
            # Shape-only trace: never materialize input data. np.asarray on a
            # zero-memory broadcast placeholder would COPY it (tens of GB for
            # full-scale MoE weights); a placeholder is all the trace needs.
            t.data = _shape_only_zeros(np.shape(array), np_dt)
        else:
            t.data = np.asarray(array, dtype=np_dt)
            if dtype in _NARROW_FLOAT_DTYPES:
                t.data = _truncate_to_dtype(t.data, dtype)
        t.dim_names = [d.strip() for d in dims.split(",") if d.strip()] if dims.strip() else []
        t.shape = t.data.shape
        t.role = role
        t.dtype = dtype
        measured_here = False
        if sparsity is None and _profiling_real_run():
            sparsity = _measure_density(
                t.data, name=_tracer.lookup_input_name(orig_array_id), role=role)
            measured_here = sparsity is not None
        t.sparsity = sparsity
        t._density_measured = measured_here
        if len(t.dim_names) != len(t.shape):
            raise ValueError(
                f"Number of dim names ({len(t.dim_names)}) doesn't match "
                f"array dimensions ({len(t.shape)})"
            )
        if _tracer.active:
            t._trace_id = _tracer.next_id()
            t._trace_owner = _tracer.owner
            input_name = _tracer.lookup_input_name(orig_array_id)
            _tracer.record_tensor(
                t._trace_id, t.dim_names, t.shape,
                t.dtype, t.role, t.sparsity, input_name=input_name,
                sparsity_source=(density_profiling_source() if measured_here else None),
                data=t.data,
                # the wrapped data IS the registered input (no cast copy): a
                # returned input can then be recognised as the input itself
                input_alias=(t.data is array or (_is_trace_only() and isinstance(array, np.ndarray)
                                                 and dtype not in _NARROW_FLOAT_DTYPES
                                                 and np.dtype(array.dtype) == np.dtype(np_dt))),
            )
        return t

    @classmethod
    def from_random(cls, *, role=None, dtype="fp32", sparsity=None, **dim_sizes):
        """Create a tensor with random data for testing."""
        _validate_dtype(dtype)
        t = cls.__new__(cls)
        t.dim_names = list(dim_sizes.keys())
        t.shape = tuple(dim_sizes.values())
        np_dt = _numpy_dtype(dtype)
        if dtype in _INTEGER_DTYPES:
            t.data = np.random.randint(0, 100, size=t.shape, dtype=np_dt)
        else:
            t.data = np.random.randn(*t.shape).astype(np_dt)
        t.role = role
        t.dtype = dtype
        t.sparsity = sparsity
        for name, size in dim_sizes.items():
            _registry.register(name, size)
        if _tracer.active:
            t._trace_id = _tracer.next_id()
            t._trace_owner = _tracer.owner
            _tracer.record_tensor(
                t._trace_id, t.dim_names, t.shape,
                t.dtype, t.role, t.sparsity,
                data=t.data,
            )
        return t

    @property
    def bytes_per_element(self):
        """Number of bytes per element for this tensor's dtype."""
        return _BYTES_PER_ELEMENT[self.dtype]

    def __getitem__(self, key):
        if self.data is None:
            raise RuntimeError("Cannot index a Tensor spec (no data)")
        return self.data[key]

    def __setitem__(self, key, value):
        if self.data is None:
            raise RuntimeError("Cannot index a Tensor spec (no data)")
        self.data[key] = value

    def __repr__(self):
        dtype_str = f", dtype='{self.dtype}'"
        sparsity_str = f", sparsity={self.sparsity}" if self.sparsity is not None else ""
        if self.data is not None:
            sizes = ", ".join(f"{n}={s}" for n, s in zip(self.dim_names, self.shape))
            role_str = f", role='{self.role}'" if self.role else ""
            return f"Tensor({sizes}{role_str}{dtype_str}{sparsity_str})"
        dims = ",".join(self.dim_names)
        role_str = f", role='{self.role}'" if self.role else ""
        return f'Tensor("{dims}"{role_str}{dtype_str}{sparsity_str})'


class Dim:
    """Iterable dimension for kernel loops.

    for b in Dim("B", parallel=True):
        for m in Dim("M"):
            C[b, m] = A[b, m] + 1

    When the loop bound is not a tensor dimension, pass ``size`` to
    register it explicitly::

        groups = 4
        for g in Dim("G", size=groups):
            ...

    The ``parallel`` flag is advisory: the slow Python path ignores it,
    but the Numba lowerer emits ``numba.prange`` only for dims marked
    ``parallel=True``.
    """

    def __init__(self, name: str, parallel: bool = False, size: int = None):
        self.name = name
        self.parallel = parallel
        if size is not None:
            _registry.register(name, size)

    def __iter__(self):
        if _is_trace_only() or (_tracer.active and _tracer.shape_only):
            return iter([])  # skip loop body during trace-only or shape-only trace
        size = _registry.get_size(self.name)
        return iter(range(size))

    def __repr__(self):
        parts = [f'"{self.name}"']
        if self.parallel:
            parts.append("parallel=True")
        return f'Dim({", ".join(parts)})'


def clear_dims():
    """Clear all registered dimension sizes."""
    _registry.clear()


def apply_mask(tensor, mask, masked_value=-1e9):
    """Conditionally replace masked positions (builtin, not a kernel).

    Where ``mask > 0.5`` (True / 1.0), keeps the original value from
    *tensor*.  Where ``mask <= 0.5`` (False / 0.0), replaces with
    *masked_value*.

    Common ``masked_value`` choices:

    * ``-1e9`` — before softmax (attention masking, default)
    * ``0.0``  — multiplicative / gating masking

    Propagates ``sparsity`` metadata from *mask* to the result so
    downstream FLOPs/bytes analysis accounts for the effective density.

    The einsum compiler should skip this function by name — it is
    infrastructure, not a computational kernel.
    """
    caller_line = None
    if _tracer.active:
        frame = sys._getframe(1)
        caller_line = frame.f_lineno
    if _is_trace_only():
        result_data = _shape_only_zeros(tensor.shape, _numpy_dtype(tensor.dtype))
        result_sparsity = mask.sparsity
    else:
        result_data = np.where(mask.data > 0.5, tensor.data, masked_value)
        result_sparsity = mask.sparsity
        if _profiling_real_run():
            # Measured beats propagated (exact for 0.0 fills; a -1e9 fill is
            # dense and correctly measures as such).
            measured = _measure_density(result_data)
            if measured is not None:
                result_sparsity = measured
    result = Tensor.from_numpy(
        result_data,
        ",".join(tensor.dim_names),
        dtype=tensor.dtype,
        sparsity=result_sparsity,
    )
    if _tracer.active:
        _tracer.record_call(
            "apply_mask", "builtin",
            [getattr(tensor, "_trace_id", None),
             getattr(mask, "_trace_id", None)],
            getattr(result, "_trace_id", None),
            call_site_line=caller_line,
        )
    return result


def top_k_mask(tensor, k, axis=-1):
    """Binary mask of the top-*k* values along *axis* (builtin, not a kernel).

    Returns a Tensor the SAME shape (and dims) as *tensor* whose entries are
    ``1.0`` at the positions holding the ``k`` largest values along *axis* and
    ``0.0`` everywhere else (ties broken by index order).  Typical use is MoE /
    top-k expert routing, where the mask then gates expert contributions.

    Like :func:`apply_mask`, this is infrastructure — the einsum compiler skips
    it by name — and it is dim-preserving.  ``sparsity`` is set to the fraction
    of positions kept (``k / size_along_axis``) so downstream FLOPs/bytes
    analysis accounts for the routed density.
    """
    caller_line = None
    if _tracer.active:
        frame = sys._getframe(1)
        caller_line = frame.f_lineno
    if _is_trace_only():
        result_data = _shape_only_zeros(tensor.shape, _numpy_dtype(tensor.dtype))
        # ANALYTIC density: k/n depends only on the axis extent, so it is
        # exact at whatever scale this trace runs at (an original-shape
        # trace gets k/E_original with no measurement needed).
        ax = axis if axis >= 0 else len(tensor.shape) + axis
        n = tensor.shape[ax]
        density = (max(0, min(int(k), n)) / n) if n else None
        if density is not None and density >= _DENSITY_DENSE_CUTOFF:
            density = None
    else:
        data = tensor.data
        ax = axis if axis >= 0 else data.ndim + axis
        n = data.shape[ax]
        kk = max(0, min(int(k), n))
        result_data = np.zeros_like(data)
        if kk == n:
            result_data[...] = 1.0
        elif kk > 0:
            # Indices of the kk largest values along `ax` (unordered top-kk).
            part = np.argpartition(data, n - kk, axis=ax)
            topk_idx = np.take(part, np.arange(n - kk, n), axis=ax)
            np.put_along_axis(result_data, topk_idx, 1.0, axis=ax)
        density = (kk / n) if n else None
    result = Tensor.from_numpy(
        result_data,
        ",".join(tensor.dim_names),
        dtype=tensor.dtype,
        sparsity=density,
    )
    if _tracer.active:
        _tracer.record_call(
            "top_k_mask", "builtin",
            [getattr(tensor, "_trace_id", None)],
            getattr(result, "_trace_id", None),
            call_site_line=caller_line,
        )
    return result


def slice_at(tensor, axis, index):
    """Slice a tensor at a fixed index along a named axis, dropping that dim.

    Returns a new Tensor with the sliced dimension removed.
    Records itself as a ``builtin`` call in the trace so the graph
    has a proper producer for the output (no phantom inputs).
    """
    caller_line = None
    if _tracer.active:
        caller_line = sys._getframe(1).f_lineno
    axis_pos = tensor.dim_names.index(axis)
    if _is_trace_only():
        new_shape = list(tensor.shape)
        del new_shape[axis_pos]
        result_data = _shape_only_zeros(new_shape, _numpy_dtype(tensor.dtype))
    else:
        result_data = np.take(tensor.data, index, axis=axis_pos)
    new_dims = [d for d in tensor.dim_names if d != axis]
    result = Tensor.from_numpy(
        result_data,
        ",".join(new_dims) if new_dims else "",
        dtype=tensor.dtype,
    )
    if _tracer.active:
        _tracer.record_call(
            "slice_at", "builtin",
            [getattr(tensor, "_trace_id", None)],
            getattr(result, "_trace_id", None),
            call_site_line=caller_line,
        )
    return result


def stack(tensors, new_dim, size=None):
    """Stack a list of tensors along a new leading dimension.

    Introduces a new named dimension. All input tensors must share
    the same dim_names and shape. Records itself as a ``builtin``
    call in the trace.
    """
    caller_line = None
    if _tracer.active:
        caller_line = sys._getframe(1).f_lineno
    n = len(tensors)
    if _is_trace_only():
        inner_shape = list(tensors[0].shape) if tensors else []
        dt = _numpy_dtype(tensors[0].dtype) if tensors else np.float64
        result_data = _shape_only_zeros([n] + inner_shape, dt)
    else:
        result_data = np.stack([t.data for t in tensors], axis=0)
    inner_dims = tensors[0].dim_names if tensors else []
    dim_str = f"{new_dim}," + ",".join(inner_dims) if inner_dims else new_dim
    result = Tensor.from_numpy(
        result_data,
        dim_str,
        dtype=tensors[0].dtype if tensors else "fp32",
    )
    if _tracer.active:
        input_ids = [getattr(t, "_trace_id", None) for t in tensors]
        _tracer.record_call(
            "stack", "builtin", input_ids,
            getattr(result, "_trace_id", None),
            call_site_line=caller_line,
        )
    return result


def concat(tensors, along_dim):
    """Concatenate tensors along an existing named dimension.

    The along_dim size in the output equals the sum of that dim across
    all inputs. All other dimensions must match. Records itself as a
    ``builtin`` call in the trace.
    """
    caller_line = None
    if _tracer.active:
        caller_line = sys._getframe(1).f_lineno
    axis_pos = tensors[0].dim_names.index(along_dim)
    if _is_trace_only():
        shapes = [list(t.shape) for t in tensors]
        new_shape = list(shapes[0])
        new_shape[axis_pos] = sum(s[axis_pos] for s in shapes)
        result_data = _shape_only_zeros(new_shape, _numpy_dtype(tensors[0].dtype))
    else:
        result_data = np.concatenate([t.data for t in tensors], axis=axis_pos)
    result = Tensor.from_numpy(
        result_data,
        ",".join(tensors[0].dim_names),
        dtype=tensors[0].dtype,
    )
    if _tracer.active:
        input_ids = [getattr(t, "_trace_id", None) for t in tensors]
        _tracer.record_call(
            "concat", "builtin", input_ids,
            getattr(result, "_trace_id", None),
            call_site_line=caller_line,
        )
    return result


def gather(table, index, axis):
    """Gather slices from *table* along *axis* using integer *index* tensor.

    Semantics: ``out[...] = table[index[...]]`` along the named *axis*.
    The *axis* dimension of *table* is replaced by all dimensions of *index*
    in the output shape.

    This is the traced equivalent of ``np.take`` / ``table[index]`` and
    covers ``nn.Embedding`` lookups and index-select operations.
    Records itself as a ``builtin`` call in the trace (zero compute).
    """
    caller_line = None
    if _tracer.active:
        caller_line = sys._getframe(1).f_lineno
    axis_pos = table.dim_names.index(axis)

    table_dims = list(table.dim_names)
    index_dims = list(index.dim_names)
    out_dims = table_dims[:axis_pos] + index_dims + table_dims[axis_pos + 1:]

    if _is_trace_only():
        out_shape = (list(table.shape)[:axis_pos]
                     + list(index.shape)
                     + list(table.shape)[axis_pos + 1:])
        result_data = _shape_only_zeros(out_shape, _numpy_dtype(table.dtype))
    else:
        raw = np.take(table.data, index.data.astype(int), axis=axis_pos)
        result_data = raw

    result = Tensor.from_numpy(
        result_data,
        ",".join(out_dims) if out_dims else "",
        dtype=table.dtype,
    )
    if _tracer.active:
        _tracer.record_call(
            "gather", "builtin",
            [getattr(table, "_trace_id", None),
             getattr(index, "_trace_id", None)],
            getattr(result, "_trace_id", None),
            call_site_line=caller_line,
        )
    return result


def kernel(func):
    """Decorator that isolates dimension scope for a kernel function.

    Pushes a new scope on entry, registers dims from input Tensor arguments,
    and restores the previous scope on exit.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        caller_line = None
        if _tracer.active:
            frame = sys._getframe(1)
            caller_line = frame.f_lineno
        with _registry.scope():
            for arg in args:
                if isinstance(arg, Tensor) and arg.data is not None:
                    for name, size in zip(arg.dim_names, arg.shape):
                        _registry.register(name, size)
            for arg in kwargs.values():
                if isinstance(arg, Tensor) and arg.data is not None:
                    for name, size in zip(arg.dim_names, arg.shape):
                        _registry.register(name, size)
            result = func(*args, **kwargs)
            if (isinstance(result, Tensor)
                    and result.dtype in _NARROW_FLOAT_DTYPES
                    and result.data is not None):
                if _is_trace_only():
                    # Shape-only trace (issue #67): no kernel body ran, ``result.data`` is
                    # the zero-storage placeholder and nothing reads its values, so
                    # rounding zeros is the identity.  Rounding here would ``astype``
                    # / torch-cast the placeholder into full-size fp32, bf16 and fp64
                    # arrays at the ORIGINAL logical dims (an attention-score tensor
                    # of 1-11 GiB per kernel) and the provenance binding would keep
                    # them alive.  Keep the placeholder -- the object bound at
                    # construction stays the traced output -- and say so on the entry.
                    _tracer.mark_trace_only_unrounded(
                        getattr(result, "_trace_id", None), result.dtype)
                else:
                    result.data = _truncate_to_dtype(result.data, result.dtype)
            if (isinstance(result, Tensor) and result.data is not None
                    and (result.sparsity is None
                         or getattr(result, "_density_measured", False))
                    and _profiling_real_run()):
                # Measured density of the kernel output (the trace recorded the
                # tensor at construction, before the body ran — back-patch it).
                d = _measure_density(result.data, name=func.__name__,
                                     role=result.role)
                result.sparsity = d
                result._density_measured = d is not None
                _tracer.update_tensor_sparsity(
                    getattr(result, "_trace_id", None), d,
                    source=density_profiling_source())
            if _tracer.active:
                # Record tensor inputs in the kernel's SIGNATURE order, not
                # call order: the converter maps recorded inputs onto
                # param_names positionally, and a keyword call such as
                # f(Z=z, X=x) must not swap them.
                ordered = list(args) + list(kwargs.values())
                input_order = "call"
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    ordered = []
                    for pname, value in bound.arguments.items():
                        kind = sig.parameters[pname].kind
                        if kind is inspect.Parameter.VAR_POSITIONAL:
                            ordered.extend(value)             # *args tuple
                        elif kind is inspect.Parameter.VAR_KEYWORD:
                            ordered.extend(value.values())    # **kwargs dict
                        else:
                            ordered.append(value)             # one argument, whatever its type
                    input_order = "signature"
                except (TypeError, ValueError):
                    ordered = list(args) + list(kwargs.values())
                input_ids = [
                    getattr(a, "_trace_id", None)
                    for a in ordered
                    if isinstance(a, Tensor)
                ]
                # TARGET-DSL-DIM-KWORDER-7: the registry above was filled in CALL
                # order (positional arguments, then keywords in call order), and
                # the last registration of a name wins.  Record that order as
                # indices into the signature-ordered input list so the converter
                # can replay the registrations exactly (f(Z=z, X=x) registers Z
                # before X although the inputs are listed X, Z).
                sig_tensors = [a for a in ordered if isinstance(a, Tensor)]
                registration_order = []
                for a in list(args) + list(kwargs.values()):
                    if isinstance(a, Tensor):
                        idx = next((i for i, t in enumerate(sig_tensors) if t is a and i not in registration_order), None)
                        if idx is not None:
                            registration_order.append(idx)
                # TARGET-KB-MAXPOOL (an earlier change round 4): the kernel's int arguments by parameter name, so that a
                # size written as ``Dim("R", size=kernel_size)`` is resolvable from the trace at this call.
                scalar_args = None
                if input_order == "signature":
                    # scalar-extent-18 (round 5): an omitted ``n: int = 3`` IS the call's value -- the defaults
                    # are applied (on a separate binding, so the tensor input order above is untouched); a 0 is a
                    # value like any other and is recorded.
                    with_defaults = sig.bind(*args, **kwargs)
                    with_defaults.apply_defaults()
                    scalar_args = {pname: int(value) for pname, value in with_defaults.arguments.items()
                                   if isinstance(value, int) and not isinstance(value, bool)
                                   and sig.parameters[pname].kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)} or None
                output_id = getattr(result, "_trace_id", None) if isinstance(result, Tensor) else None
                # TARGET-DSL-TUPLE-13: a kernel that returns a tuple (or any non-Tensor) has no output node; the
                # trace says so, and the assembler refuses with a typed error instead of a KeyError on the node name.
                unsupported_return = None
                if not isinstance(result, Tensor):
                    if isinstance(result, tuple):
                        unsupported_return = {"kind": "tuple", "length": len(result), "tensors": sum(isinstance(r, Tensor) for r in result)}
                    else:
                        unsupported_return = {"kind": type(result).__name__}
                _tracer.record_call(
                    func.__name__, "kernel", input_ids, output_id,
                    call_site_line=caller_line, input_order=input_order,
                    registration_order=registration_order if len(registration_order) == len(input_ids) else None,
                    unsupported_return=unsupported_return, scalar_args=scalar_args,
                )
                if output_id is not None and getattr(result, "data", None) is not None:
                    # the kernel body may have rebound the output's data array
                    _tracer.bind_tensor_data(output_id, result.data)
            return result

    return wrapper
