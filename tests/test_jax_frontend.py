# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JAX/Flax front-end (solar.einsum_lib.jax_frontend).

All tests are guarded with pytest.importorskip so they skip cleanly when
JAX is not installed, never breaking the existing test suite.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
flax_linen = pytest.importorskip("flax.linen")

from solar.einsum_lib.jax_frontend import (
    _flatten_variables,
    _run_jax_reference,
    _run_raw_jax_reference,
    _to_numpy,
    build_jax_shapes_and_dtypes,
    get_jax_model_and_inputs,
    load_jax_module,
    prepare_jax,
)


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples_jax"


# ---------------------------------------------------------------------------
# test_load_jax_module
# ---------------------------------------------------------------------------

class TestLoadJaxModule:
    def test_loads_flax_matmul(self):
        module = load_jax_module(str(EXAMPLES_DIR / "FlaxMatmul" / "FlaxMatmul.py"))
        assert hasattr(module, "get_model")
        assert hasattr(module, "get_inputs")

    def test_loads_raw_jax_softmax(self):
        module = load_jax_module(str(EXAMPLES_DIR / "JaxSoftmax" / "JaxSoftmax.py"))
        assert hasattr(module, "get_model")
        assert hasattr(module, "forward")


# ---------------------------------------------------------------------------
# test_get_jax_model_and_inputs
# ---------------------------------------------------------------------------

class TestGetJaxModelAndInputs:
    def test_flax_model_returns_variables(self):
        module = load_jax_module(str(EXAMPLES_DIR / "FlaxMatmul" / "FlaxMatmul.py"))
        model, variables, inputs = get_jax_model_and_inputs(module)
        assert model is not None
        assert variables is not None
        assert "params" in variables
        assert len(inputs) == 1
        assert isinstance(inputs[0], np.ndarray)

    def test_raw_jax_returns_none_model(self):
        module = load_jax_module(str(EXAMPLES_DIR / "JaxSoftmax" / "JaxSoftmax.py"))
        model, variables, inputs = get_jax_model_and_inputs(module)
        assert model is None
        assert variables is None
        assert len(inputs) == 1


# ---------------------------------------------------------------------------
# test_flatten_variables
# ---------------------------------------------------------------------------

class TestFlattenVariables:
    def test_params_flattened(self):
        module = load_jax_module(str(EXAMPLES_DIR / "FlaxMatmul" / "FlaxMatmul.py"))
        model, variables, _ = get_jax_model_and_inputs(module)
        param_dict, buffer_dict = _flatten_variables(variables)
        assert any("kernel" in k for k in param_dict), f"Expected kernel in {list(param_dict.keys())}"
        assert any("bias" in k for k in param_dict), f"Expected bias in {list(param_dict.keys())}"
        for v in param_dict.values():
            assert isinstance(v, np.ndarray)

    def test_batch_stats_in_buffer_dict(self):
        module = load_jax_module(str(EXAMPLES_DIR / "FlaxBatchNorm" / "FlaxBatchNorm.py"))
        model, variables, inputs = get_jax_model_and_inputs(module)
        param_dict, buffer_dict = _flatten_variables(variables)
        assert len(buffer_dict) > 0, "Expected batch_stats in buffer_dict"
        assert any("batch_stats" in k for k in buffer_dict), (
            f"Expected batch_stats key, got {list(buffer_dict.keys())}"
        )


# ---------------------------------------------------------------------------
# test_build_jax_shapes_and_dtypes
# ---------------------------------------------------------------------------

class TestBuildJaxShapesAndDtypes:
    def test_shapes_dtypes_match(self):
        module = load_jax_module(str(EXAMPLES_DIR / "FlaxMatmul" / "FlaxMatmul.py"))
        model, variables, inputs = get_jax_model_and_inputs(module)
        param_dict, buffer_dict = _flatten_variables(variables)
        dtypes_raw = ["fp64"]
        shapes, dtypes, fixed_inputs, buffer_keys = build_jax_shapes_and_dtypes(
            param_dict, buffer_dict, inputs, dtypes_raw
        )
        assert "x" in shapes
        assert "x" in dtypes
        assert "x" in fixed_inputs
        for name in param_dict:
            assert name in shapes
            assert name in dtypes

    def test_batch_stats_as_buffers(self):
        module = load_jax_module(str(EXAMPLES_DIR / "FlaxBatchNorm" / "FlaxBatchNorm.py"))
        model, variables, inputs = get_jax_model_and_inputs(module)
        param_dict, buffer_dict = _flatten_variables(variables)
        dtypes_raw = ["fp64"]
        shapes, dtypes, fixed_inputs, buffer_keys = build_jax_shapes_and_dtypes(
            param_dict, buffer_dict, inputs, dtypes_raw
        )
        assert len(buffer_keys) > 0
        for bk in buffer_keys:
            assert bk in fixed_inputs or bk in shapes


# ---------------------------------------------------------------------------
# test_prepare_jax
# ---------------------------------------------------------------------------

class TestPrepareJax:
    def test_no_verify_returns_8_tuple(self):
        result = prepare_jax(
            str(EXAMPLES_DIR / "FlaxMatmul" / "FlaxMatmul.py"),
            verify=False,
        )
        assert len(result) == 8
        source_code, input_shapes, input_dtypes, ref_out, all_in, sparsity, mask_fv, buf_keys = result
        assert ref_out is None
        assert all_in is None
        assert isinstance(input_shapes, dict)
        assert isinstance(buf_keys, set)

    def test_verify_returns_9_tuple(self):
        result = prepare_jax(
            str(EXAMPLES_DIR / "FlaxMatmul" / "FlaxMatmul.py"),
            verify=True,
        )
        assert len(result) == 9
        (source_code, input_shapes, input_dtypes, ref_out,
         all_in, sparsity, mask_fv, buf_keys, verify_shapes) = result
        assert ref_out is not None
        assert "output" in ref_out
        assert all_in is not None
        assert isinstance(verify_shapes, dict)


# ---------------------------------------------------------------------------
# test_float64_enabled
# ---------------------------------------------------------------------------

class TestFloat64:
    def test_jax_float64_active(self):
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        assert x.dtype == jnp.float64

    def test_to_numpy_preserves_float64(self):
        x = jnp.array([1.0, 2.0], dtype=jnp.float64)
        arr = _to_numpy(x)
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# test_bfloat16_conversion
# ---------------------------------------------------------------------------

class TestBfloat16:
    def test_bf16_to_numpy(self):
        x = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)
        arr = _to_numpy(x)
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, [1.0, 2.0], atol=0.01)
