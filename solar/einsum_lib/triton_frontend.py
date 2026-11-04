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

"""Self-referencing frontend for standalone Triton kernels.

Unlike the PyTorch frontend, the pipeline never imports torch/triton here.
The source file brings its own reference by exposing two functions:

    get_inputs() -> {name: np.ndarray}
        Concrete inputs (seeded random is fine).  Shapes and dtypes are
        derived from these arrays, never declared separately.

    kernel_fn(inputs: dict) -> {"output": np.ndarray}
        Moves the inputs to CUDA, launches the real ``@triton.jit`` kernel,
        and returns the golden output.  This is the only code that touches
        Triton/CUDA, and it lives in the example file, not the pipeline.

The same input arrays are handed to both the GPU kernel (reference) and the
generated einsum_lib code (CPU), so the two sides see identical numbers.

The contract is deliberately framework-neutral: any file that implements
``get_inputs``/``kernel_fn`` works, so this path extends to other kernel
languages without pipeline changes.

Optionally a file may also define:

    get_source() -> str
        Return the source code that the LLM should translate.  When a thin
        adapter imports a kernel from an installed package (e.g. Liger-Kernel),
        ``get_source()`` can use ``inspect.getsource()`` to extract the real
        kernel source so the LLM sees the actual implementation rather than
        just the adapter glue.  If absent, the adapter file itself is used.
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

    return load_source_module(file_path, "_triton_module")


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


def prepare_triton(file_path: str, verify: bool,
                   use_fp64: bool = False, debug: bool = False,
                   cap_input_size: bool = False):
    """Load a Triton kernel file, derive shapes, and capture the GPU reference.

    Returns a tuple matching the JAX frontend contract:

      verify=True:
        (source_code, input_shapes, input_dtypes, reference_outputs,
         all_inputs, input_sparsity, mask_fill_values, buffer_keys,
         verify_shapes)
      verify=False:
        (source_code, input_shapes, input_dtypes, reference_outputs,
         all_inputs, input_sparsity, mask_fill_values, buffer_keys)

    The kernel is a pure function of its arguments, so there are no model
    parameters, buffers, masks, or int constants to extract.
    """
    module = _load_module(file_path)
    if hasattr(module, "get_source"):
        source_code = module.get_source()
    else:
        source_code = Path(file_path).read_text(encoding="utf-8")

    if not hasattr(module, "get_inputs"):
        raise RuntimeError(
            f"{Path(file_path).name} must define get_inputs() returning "
            "{name: np.ndarray}"
        )

    raw_inputs = module.get_inputs()
    if not isinstance(raw_inputs, dict):
        raise RuntimeError(
            "get_inputs() must return a dict {name: np.ndarray} for a "
            "self-referencing Triton source (names become einsum_lib inputs)."
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
    # core._numpy_dtype), so the generated IR always computes in fp64.  The
    # kernel receives that same precision and adapts internally -- verification
    # and benchmarking must exercise one code path, otherwise the translation is
    # not being tested against the real kernel.
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

    return (source_code, input_shapes, input_dtypes, reference_outputs,
            all_inputs, input_sparsity, mask_fill_values, buffer_keys,
            verify_shapes)
