# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX/Flax front-end for the SOLAR einsum_lib translation pipeline.

All JAX and Flax imports are confined to this module so the rest of the
pipeline can run without JAX installed.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Must be set before any JAX array creation -- JAX silently downcasts
# float64 to float32 by default, breaking verification tolerances.
import jax
import logging

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_JAX_DTYPE_TO_SOLAR = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "bfloat16": "bf16",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
}

_INTEGER_DTYPES = frozenset({"int8", "int32", "int64"})

_VERIFY_MAX_DIM = 8
_CAPTURE_SEED = 42


def _jax_dtype_str(dtype) -> str:
    return _JAX_DTYPE_TO_SOLAR.get(str(dtype), "fp32")


def _is_integer_dtype(dtype: str) -> bool:
    return dtype in _INTEGER_DTYPES


def _scale_shape_for_verify(shape: tuple, max_dim: int = _VERIFY_MAX_DIM) -> tuple:
    return tuple(min(d, max_dim) for d in shape)


def _to_numpy(x) -> np.ndarray:
    """Convert a JAX array or Python scalar to a numpy array.

    Integer arrays keep their dtype; floating-point arrays are promoted to
    float64 for verification.
    """
    if isinstance(x, (int, float)):
        return np.array(x, dtype=np.float64)
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    return arr.astype(np.float64)


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def load_jax_module(file_path: str):
    """Dynamically import a .py file containing a JAX/Flax model.

    Loaded from the CURRENT source bytes (bytecode caches bypassed) so the
    verification runner can bind the executed model code to the hashed
    source — see solar.einsum_lib.source_load.
    """
    from solar.einsum_lib.source_load import load_source_module

    return load_source_module(file_path, "_jax_model_module", register=True)


def get_jax_model_and_inputs(module):
    """Extract the Flax model, variables, and numpy inputs from a module.

    Convention:
      - ``get_model()`` returns a Flax ``nn.Module`` instance (or None for
        raw JAX with a ``forward()`` function).
      - ``get_inputs()`` returns a list of JAX arrays.

    Returns ``(model, variables, inputs)`` where:
      - *model* is the Flax module (or None for raw JAX)
      - *variables* is the Flax variable dict (``{"params": ..., ...}``)
        or None for raw JAX
      - *inputs* is a list of numpy arrays
    """
    if not hasattr(module, "get_inputs"):
        raise RuntimeError("Module must define get_inputs()")

    raw_inputs = module.get_inputs()
    np_inputs = [_to_numpy(t) for t in raw_inputs]

    if hasattr(module, "get_model"):
        model = module.get_model()
    else:
        model = None

    variables = None
    if model is not None:
        jax_inputs = [jnp.array(a) for a in np_inputs]
        variables = model.init(jax.random.PRNGKey(0), *jax_inputs)

    return model, variables, np_inputs


# ---------------------------------------------------------------------------
# Variable flattening
# ---------------------------------------------------------------------------

def _flatten_collection(tree: dict, prefix: str = "") -> dict:
    """Flatten a nested Flax variable collection into ``{safe_name: np.ndarray}``."""
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_collection(value, prefix=f"{name}_"))
        else:
            flat[name] = _to_numpy(value)
    return flat


def _flatten_variables(variables: dict):
    """Split and flatten a Flax variable dict into params and buffers.

    Returns ``(param_dict, buffer_dict)`` where both are
    ``{safe_name: np.ndarray}`` dicts.
    """
    param_dict = {}
    buffer_dict = {}

    for collection_name, collection in variables.items():
        if collection_name == "params":
            param_dict = _flatten_collection(collection)
        else:
            buffer_dict.update(
                _flatten_collection(collection, prefix=f"{collection_name}_")
            )

    return param_dict, buffer_dict


# ---------------------------------------------------------------------------
# Shape / dtype extraction
# ---------------------------------------------------------------------------

def build_jax_shapes_and_dtypes(param_dict, buffer_dict, inputs, input_dtypes_raw):
    """Build unified shapes/dtypes dicts matching the PyTorch pipeline contract.

    Returns ``(shapes, dtypes, fixed_inputs, buffer_keys)``.
    """
    shapes = {}
    dtypes = {}
    fixed_inputs: dict = {}
    buffer_keys: set = set()

    # Activations
    for i, arr in enumerate(inputs):
        key = "x" if len(inputs) == 1 else f"x_{i}"
        shapes[key] = arr.shape
        dtypes[key] = input_dtypes_raw[i]
        fixed_inputs[key] = arr

    # Parameters (shape/dtype only -- random values during verification)
    for name, arr in param_dict.items():
        shapes[name] = arr.shape
        dtypes[name] = "fp64" if np.issubdtype(arr.dtype, np.floating) else _jax_dtype_str(arr.dtype)

    # Buffers (fixed values)
    for name, arr in buffer_dict.items():
        shapes[name] = arr.shape
        dtypes[name] = "fp64" if np.issubdtype(arr.dtype, np.floating) else _jax_dtype_str(arr.dtype)
        if arr.size > 0:
            fixed_inputs[name] = arr
        buffer_keys.add(name)

    return shapes, dtypes, fixed_inputs, buffer_keys


# ---------------------------------------------------------------------------
# Reference execution
# ---------------------------------------------------------------------------

def _run_jax_reference(model, variables, np_inputs):
    """Run the Flax model and return reference outputs as numpy float64.

    Handles both stateless models (params only) and stateful models
    (params + batch_stats or other mutable collections).
    """
    jax_inputs = [jnp.array(a) for a in np_inputs]

    mutable_collections = [k for k in variables if k != "params"]

    if mutable_collections:
        output, _updates = model.apply(
            variables, *jax_inputs, mutable=mutable_collections
        )
    else:
        output = model.apply(variables, *jax_inputs)

    if isinstance(output, (tuple, list)):
        out_arr = np.asarray(output[0]).astype(np.float64)
    else:
        out_arr = np.asarray(output).astype(np.float64)

    return {"output": out_arr}


def _run_raw_jax_reference(module, np_inputs):
    """Run a raw JAX forward() function and return reference outputs."""
    if not hasattr(module, "forward"):
        raise RuntimeError(
            "Raw JAX module (get_model() returned None) must define forward()"
        )
    jax_inputs = [jnp.array(a) for a in np_inputs]
    output = module.forward(*jax_inputs)

    if isinstance(output, (tuple, list)):
        out_arr = np.asarray(output[0]).astype(np.float64)
    else:
        out_arr = np.asarray(output).astype(np.float64)

    return {"output": out_arr}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_jax(file_path: str, verify: bool, cap_input_size: bool = False,
                use_fp64: bool = False, debug: bool = False,
                mem_budget_bytes: Optional[int] = None):
    """Load a JAX/Flax model, build shapes/dtypes, optionally run reference.

    Returns the same 8-tuple (no verify) or 9-tuple (verify) as
    ``prepare_pytorch`` in ``pipeline.py``.

    When *cap_input_size* is True the verification shapes are proposed by the
    LLM (via :func:`scale_shapes_with_llm`) instead of naively clamping every
    dimension to a fixed maximum.  This keeps dimensions mutually consistent so
    the reference ``workload``/model does not crash on reshapes to hardcoded
    constants, and respects *mem_budget_bytes* to avoid OOM.  If the proposed
    shapes still crash the reference, the error is fed back and the LLM
    re-proposes (up to ``_MAX_CAP_RETRIES`` rounds).
    """
    from solar.einsum_lib.verify import compute_sparsity

    module = load_jax_module(file_path)
    model, variables, np_inputs = get_jax_model_and_inputs(module)
    source_code = Path(file_path).read_text(encoding="utf-8")

    # Capture original dtypes before any conversion
    input_dtypes_raw = []
    raw_inputs = module.get_inputs()
    for t in raw_inputs:
        if isinstance(t, (int, float)):
            input_dtypes_raw.append("fp64")
        elif hasattr(t, "dtype"):
            input_dtypes_raw.append(_jax_dtype_str(t.dtype))
        else:
            input_dtypes_raw.append("fp32")

    # Flatten variables
    if variables is not None:
        param_dict, buffer_dict = _flatten_variables(variables)
    else:
        param_dict, buffer_dict = {}, {}

    input_shapes, input_dtypes, fixed_inputs, buffer_keys = build_jax_shapes_and_dtypes(
        param_dict, buffer_dict, np_inputs, input_dtypes_raw
    )

    input_sparsity: dict = {}
    for key, arr in fixed_inputs.items():
        density = compute_sparsity(arr)
        if density < 1.0:
            input_sparsity[key] = round(density, 4)

    if not verify:
        return (source_code, input_shapes, input_dtypes, None, None,
                input_sparsity, None, buffer_keys)

    # Build numpy_inputs for *verify_shapes* and run the reference model.
    # Raises whatever the reference raises (reshape crash / OOM) so the caller
    # can feed the error back to the capping LLM and re-propose.
    def _build_and_run(vshapes):
        rng = np.random.RandomState(_CAPTURE_SEED)
        ninputs: dict = {}
        for name, shape in vshapes.items():
            if name in fixed_inputs:
                arr = fixed_inputs[name]
                slices = tuple(slice(0, s) for s in shape)
                ninputs[name] = arr[slices]
            elif _is_integer_dtype(input_dtypes.get(name, "")):
                high = max(shape) if shape else 100
                ninputs[name] = rng.randint(0, high, size=shape).astype(np.int64)
            else:
                ninputs[name] = rng.uniform(-0.1, 0.1, size=shape).astype(np.float64)

        ref_np_inputs = [
            ninputs[("x" if len(np_inputs) == 1 else f"x_{i}")]
            for i in range(len(np_inputs))
        ]
        if model is not None and variables is not None:
            ref_variables = _rebuild_variables_for_verify(variables, ninputs, param_dict)
            ref_out = _run_jax_reference(model, ref_variables, ref_np_inputs)
        else:
            ref_out = _run_raw_jax_reference(module, ref_np_inputs)
        return ninputs, ref_out

    if cap_input_size:
        # LLM-proposed capping with a crash/OOM feedback loop, mirroring
        # prepare_pytorch.  Shapes are clamped to shrink-only for consistency.
        from solar.einsum_lib.pipeline import (
            scale_shapes_with_llm, _check_memory, _MAX_CAP_RETRIES,
        )

        def _clamp(proposed):
            return {n: tuple(min(p, o) for p, o in zip(s, input_shapes[n]))
                    for n, s in proposed.items()}

        cap_error = None
        numpy_inputs = None
        reference_outputs = None
        for cap_try in range(_MAX_CAP_RETRIES):
            verify_shapes = _clamp(scale_shapes_with_llm(
                source_code, input_shapes, debug=debug,
                error_context=cap_error, mem_budget_bytes=mem_budget_bytes,
                use_fp64=use_fp64,
            ))
            if debug:
                changed = {k: v for k, v in verify_shapes.items()
                           if v != input_shapes[k]}
                if changed:
                    logger.debug(f"JAX capped shapes: {changed}")
            try:
                # Convert an over-budget proposal into a catchable error so the
                # LLM re-proposes smaller instead of being OOM-killed (SIGKILL).
                _check_memory(verify_shapes, input_dtypes, use_fp64,
                              mem_budget_bytes=mem_budget_bytes)
                numpy_inputs, reference_outputs = _build_and_run(verify_shapes)
                break
            except Exception as exc:
                cap_error = str(exc)
                if cap_try < _MAX_CAP_RETRIES - 1:
                    print(f"  [WARN] Capped shapes crashed the JAX model "
                          f"({str(exc)[:120]}), asking LLM to fix "
                          f"(attempt {cap_try + 2}/{_MAX_CAP_RETRIES})...")
                else:
                    raise RuntimeError(
                        f"Capped shapes crashed the JAX model after "
                        f"{_MAX_CAP_RETRIES} attempts: {exc}"
                    ) from exc
    else:
        verify_shapes = input_shapes
        numpy_inputs, reference_outputs = _build_and_run(verify_shapes)

    return (source_code, input_shapes, input_dtypes,
            reference_outputs, numpy_inputs, input_sparsity,
            None, buffer_keys, verify_shapes)


def _rebuild_variables_for_verify(original_variables, numpy_inputs, param_dict):
    """Rebuild a Flax variable dict using verification numpy values.

    Parameters use values from ``numpy_inputs`` (random during verification).
    Other collections (batch_stats, etc.) also use ``numpy_inputs`` values
    so that the reference run uses identical data to the LLM-generated code.
    """
    new_vars = {}

    for collection_name, collection in original_variables.items():
        if collection_name == "params":
            new_vars["params"] = _unflatten_into(
                collection, numpy_inputs, prefix=""
            )
        else:
            new_vars[collection_name] = _unflatten_into(
                collection, numpy_inputs, prefix=f"{collection_name}_"
            )

    return new_vars


def _unflatten_into(tree: dict, numpy_inputs: dict, prefix: str) -> dict:
    """Rebuild a nested Flax param tree using values from numpy_inputs."""
    result = {}
    for key, value in tree.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            result[key] = _unflatten_into(value, numpy_inputs, prefix=f"{name}_")
        else:
            if name in numpy_inputs:
                result[key] = jnp.array(numpy_inputs[name])
            else:
                result[key] = jnp.array(_to_numpy(value))
    return result
