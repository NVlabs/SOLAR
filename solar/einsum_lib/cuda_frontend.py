# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-referencing frontend for CUDA C++ kernels.

Mirrors the Triton frontend: the pipeline never compiles or launches anything
itself.  A Python adapter sits beside the ``.cu`` file and exposes:

    get_inputs() -> {name: np.ndarray}
        Concrete inputs (seeded random is fine).  Shapes and dtypes are
        derived from these arrays, never declared separately.

    kernel_fn(inputs: dict) -> {name: np.ndarray}
        Compiles the ``.cu`` (cached), launches the real kernel on CUDA, and
        returns the golden output.  This is the only code that touches nvcc or
        the GPU, and it lives in the example directory, not the pipeline.

    get_source() -> str
        Return the CUDA C++ source the LLM should translate.  Unlike Triton --
        where the kernel and the adapter are the same Python file -- a CUDA
        example keeps the kernel in a separate ``.cu``, so this is effectively
        mandatory here: without it the LLM would receive the ctypes glue
        instead of the kernel.

Unlike Triton, CUDA can express fp64 natively.  Where a kernel has an fp64
instantiation, verification therefore runs the same kernel the benchmark runs at
double precision and matches the fp64 einsum_lib IR exactly -- no IEEE-mode env
flags or tolerance widening.

Vendor tensor-core kernels are the exception: their operand and output types are
template parameters with no fp64 or even fp32 instantiation, so a kernel that
stores its output in fp16 cannot agree with an fp64 IR more closely than fp16
rounding.  For those, the tolerance floor this module derives from the dtypes of
the returned reference is what makes the comparison meaningful rather than
guaranteed to fail.  Adapters should therefore return the output in the kernel's
own precision, since upcasting it here would hide the constraint.
"""

from pathlib import Path

import numpy as np
import logging

logger = logging.getLogger(__name__)

_NUMPY_TO_SOLAR = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "bfloat16": "bf16",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
    "bool": "int8",
}


def _np_dtype_to_solar(dtype) -> str:
    return _NUMPY_TO_SOLAR.get(np.dtype(dtype).name, "fp32")


def _load_module(file_path: str):
    """Dynamically import a .py file (framework-neutral).

    Loaded from the CURRENT source bytes (bytecode caches bypassed) so the
    verification runner can bind the executed adapter code to the hashed
    source — see solar.einsum_lib.source_load.
    """
    from solar.einsum_lib.source_load import load_source_module

    return load_source_module(file_path, "_cuda_module")


_CAP_DIM = 64


def _cap_shapes(input_shapes: dict, cap_dim: int = _CAP_DIM) -> dict:
    """Reduce every dimension to at most *cap_dim*."""
    return {name: tuple(min(s, cap_dim) for s in shape)
            for name, shape in input_shapes.items()}


def _slice_inputs(inputs: dict, verify_shapes: dict) -> dict:
    """Slice each array down to *verify_shapes*."""
    out = {}
    for name, arr in inputs.items():
        slices = tuple(slice(0, s) for s in verify_shapes[name])
        out[name] = np.ascontiguousarray(arr[slices])
    return out


def _cast_inputs(inputs: dict, use_fp64: bool) -> dict:
    """Return a copy of *inputs* with float arrays cast to the verify dtype.

    Integer arrays keep their dtype so index tensors stay valid.
    """
    float_dtype = np.float64 if use_fp64 else np.float32
    out: dict = {}
    for name, arr in inputs.items():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            out[name] = arr.astype(float_dtype)
        else:
            out[name] = arr
    return out


# Half an ULP is the least a value stored in a given precision can differ from
# the exact result, i.e. eps/2 relative.  The factor of 4 below allows 8 of those
# rather than 1, covering the intermediates that pass through the same precision
# on the way to the output -- for FMHA, the probabilities handed to the second
# GEMM and a fast-math exp.  For fp32 and fp64 references the result is far
# smaller than the tolerance the pipeline already uses, so the floor is inert and
# only a genuinely narrow output relaxes anything.
_ULP_HEADROOM = 4.0


def _output_precision_floor(reference_outputs: dict) -> float:
    """Return the closest agreement the reference's own dtypes permit.

    A kernel that stores its result in fp16 cannot agree with an fp64 IR more
    closely than fp16 rounding, however correct the translation is.  Deriving the
    figure from the dtype rather than letting each example name a number keeps
    this from becoming a place to hide a real mismatch: the only way to loosen the
    comparison is for the kernel to actually have a narrower output, which is
    visible in the reference it returns.
    """
    floor = 0.0
    for arr in reference_outputs.values():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            floor = max(floor, _ULP_HEADROOM * float(np.finfo(arr.dtype).eps))
    return floor


def prepare_cuda(file_path: str, verify: bool,
                 use_fp64: bool = False, debug: bool = False,
                 cap_input_size: bool = False):
    """Load a CUDA adapter, derive shapes, and capture the GPU reference.

    Returns a tuple matching the Triton/JAX frontend contract:

      verify=True:
        (source_code, input_shapes, input_dtypes, reference_outputs,
         all_inputs, input_sparsity, mask_fill_values, buffer_keys,
         verify_shapes, compare_min_tol)
      verify=False:
        (source_code, input_shapes, input_dtypes, reference_outputs,
         all_inputs, input_sparsity, mask_fill_values, buffer_keys)

    The trailing ``compare_min_tol`` is the tolerance floor implied by the dtypes
    of the reference outputs; see ``_output_precision_floor``.

    The kernel is a pure function of its arguments, so there are no model
    parameters, buffers, masks, or int constants to extract.
    """
    module = _load_module(file_path)

    if hasattr(module, "get_source"):
        source_code = module.get_source()
    else:
        # Falling back to the adapter would hand the LLM ctypes plumbing rather
        # than the kernel, so this is a hard error for CUDA rather than the
        # silent fallback the Triton frontend allows.
        raise RuntimeError(
            f"{Path(file_path).name} must define get_source() returning the "
            "CUDA C++ source to translate (typically the sibling .cu file)."
        )

    if not hasattr(module, "get_inputs"):
        raise RuntimeError(
            f"{Path(file_path).name} must define get_inputs() returning "
            "{name: np.ndarray}"
        )

    raw_inputs = module.get_inputs()
    if not isinstance(raw_inputs, dict):
        raise RuntimeError(
            "get_inputs() must return a dict {name: np.ndarray} for a "
            "self-referencing CUDA source (names become einsum_lib inputs)."
        )
    inputs = {k: np.asarray(v) for k, v in raw_inputs.items()}

    input_shapes = {k: tuple(v.shape) for k, v in inputs.items()}
    input_dtypes = {k: _np_dtype_to_solar(v.dtype) for k, v in inputs.items()}
    if debug:
        logger.debug(f"Derived input shapes: {input_shapes}")
        logger.debug(f"Derived input dtypes: {input_dtypes}")

    input_sparsity: dict = {}
    mask_fill_values = None
    buffer_keys: set = set()

    if not verify:
        return (source_code, input_shapes, input_dtypes, None,
                None, input_sparsity, mask_fill_values, buffer_keys)

    if not hasattr(module, "kernel_fn"):
        raise RuntimeError(
            f"{Path(file_path).name} must define kernel_fn(inputs) returning "
            '{"output": np.ndarray} to produce the golden reference.'
        )

    if cap_input_size:
        verify_shapes = _cap_shapes(input_shapes)
        verify_inputs_raw = _slice_inputs(inputs, verify_shapes)
        if debug:
            logger.debug(f"Capped verify shapes: {verify_shapes}")
    else:
        verify_shapes = input_shapes
        verify_inputs_raw = inputs

    # The einsum_lib core stores every float tensor as float64 (see
    # core._numpy_dtype), so the generated IR always computes in fp64.  CUDA has
    # a native fp64 path, so the kernel receives that precision directly and the
    # comparison is exact -- verification and benchmarking still run one kernel,
    # differing only in which template instantiation the dtype selects.
    verify_inputs = _cast_inputs(verify_inputs_raw, use_fp64)

    if debug:
        prec = "fp64" if use_fp64 else "fp32"
        logger.debug(f"Running kernel_fn on GPU for reference ({prec})...")
    reference_outputs = module.kernel_fn(verify_inputs)

    if not isinstance(reference_outputs, dict):
        raise RuntimeError(
            'kernel_fn must return a dict, e.g. {"output": array}.'
        )
    reference_outputs = {
        k: np.asarray(v) for k, v in reference_outputs.items()
    }
    if debug:
        shapes = {k: v.shape for k, v in reference_outputs.items()}
        logger.debug(f"Got reference output(s): {shapes}")

    all_inputs = verify_inputs

    compare_min_tol = _output_precision_floor(reference_outputs)
    if debug and compare_min_tol:
        dtypes = {k: str(v.dtype) for k, v in reference_outputs.items()}
        logger.debug(f"Reference output dtypes {dtypes} permit agreement no "
              f"closer than {compare_min_tol:.3e}; using that as a tolerance floor")

    return (source_code, input_shapes, input_dtypes, reference_outputs,
            all_inputs, input_sparsity, mask_fill_values, buffer_keys,
            verify_shapes, compare_min_tol)
