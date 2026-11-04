# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-bearing containers in keyword arguments are never classified scalar (nested-1).

The scalar-keyword retry of an earlier change (ACC-140-SDPA-KWARGS) binds the tensor-bearing keywords alone and records the
others as ``unbound_scalar_kwargs``.  ``_has_tensor`` used ``_flatten_args``, which descends lists/tuples but treats
mappings as leaves, so ``payload={'extra': v}`` was classified scalar, dropped from the bind and the capture emitted a
graph with fewer operand edges than the call consumed.  Tensor-bearing-ness is now recognised recursively through
dict values and nested lists/tuples/dicts BEFORE any retry.  A mapping that holds a tensor cannot be placed as ordered
operand leaves by the capture contract (``_flatten_args`` / ``operand_kinds`` do not descend mappings), so such a
keyword is refused with the typed diagnostic naming it: never scalar, never silently dropped, no invented ordering.
"""
import functools

import pytest
import torch

from solar.graph import _torchview_param_shim as shim
from test_sdpa_scalar_kwargs_capture import _capture, _source

x, y, z = (torch.ones(2, 2) for _ in range(3))


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    {"extra": z}, {"a": {"b": z}}, {"a": {"b": [z]}}, [{"t": z}], ({"t": (z,)},), {"k": [0, (1, {"t": z})]},
], ids=["dict", "dict2", "dict-list", "list-dict", "tuple-dict-tuple", "deep-mixed"])
def test_has_tensor_sees_through_mappings(value):
    assert shim._has_tensor(value)


@pytest.mark.parametrize("value", [{"a": 1}, {"a": {"b": [0.5, "s"]}}, [], {}, 0.5, None, [{"a": (1, 2)}]],
                         ids=["scalar-dict", "nested-scalar-dict", "empty-list", "empty-dict", "float", "none", "list-scalar-dict"])
def test_has_tensor_scalar_containers_stay_scalar(value):
    assert not shim._has_tensor(value)


# ---------------------------------------------------------------------------
# binder: the reviewer's ordering matrix (all four wraps x catch-all), kept
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wrap", [lambda t: t, lambda t: [t], lambda t: (0, t), lambda t: {"nested": [t]}],
                         ids=["direct", "list", "tuple", "dict"])
@pytest.mark.parametrize("catchall", [False, True])
def test_unplaceable_tensor_cannot_drop_with_scalars(wrap, catchall):
    def adapter(input, **kwargs):
        return input
    fn = adapter if catchall else torch.mm
    leaves, info = shim.bind_operand_leaves(fn, (x,), {"mat3": wrap(x), "unknown_scalar": .125})
    assert leaves is None and "mat3" in info["refusal"]


# ---------------------------------------------------------------------------
# binder: mixed direct tensor + nested mapping keywords
# ---------------------------------------------------------------------------

def _wrapped_adapter():
    def legacy(input, mat2):
        return input + mat2

    @functools.wraps(legacy)
    def adapter(input, mat2, **kw):
        return input + mat2 + kw["payload"]["extra"]
    return adapter


def test_mixed_direct_tensor_and_nested_dict_keyword_refuses_naming_the_keyword():
    leaves, info = shim.bind_operand_leaves(_wrapped_adapter(), (x,), {"mat2": y, "payload": {"extra": z}, "unknown_scalar": .5})
    assert leaves is None
    assert "payload" in info["refusal"] and "mat2" not in info["refusal"]
    assert info["unbound_scalar_kwargs"] == []          # nothing was classified scalar on a refused bind


def test_dict_with_only_scalars_is_still_a_scalar_keyword():
    leaves, info = shim.bind_operand_leaves(_wrapped_adapter(), (x,), {"mat2": y, "opts": {"alpha": 0.5, "mode": "x"}, "unknown_scalar": .5})
    assert leaves is not None and len(leaves) == 2 and leaves[0] is x and leaves[1] is y
    assert info == {"unbound_scalar_kwargs": ["opts", "unknown_scalar"], "refusal": None}


def test_tensor_two_levels_deep_in_a_dict_keyword_refuses():
    leaves, info = shim.bind_operand_leaves(_wrapped_adapter(), (x,), {"mat2": y, "payload": {"outer": {"inner": [z]}}})
    assert leaves is None and "payload" in info["refusal"]


def test_tensor_bearing_dict_bound_to_a_known_parameter_refuses_not_drops():
    """The signature knows ``opts`` but the contract cannot order the tensors inside a mapping."""
    def f(input, opts):
        return input + opts["w"]
    leaves, info = shim.bind_operand_leaves(f, (x,), {"opts": {"w": z}})
    assert leaves is None and "opts" in info["refusal"] and "mapping" in info["refusal"]


def test_tensor_bearing_dict_in_a_catch_all_refuses_naming_the_keyword():
    def adapter(input, **kwargs):
        return input
    leaves, info = shim.bind_operand_leaves(adapter, (x,), {"payload": {"extra": z}, "flag": True})
    assert leaves is None and "payload" in info["refusal"]


def test_direct_tensor_keywords_and_scalar_retry_unchanged():
    """The SDPA control: scalar keywords unknown to the stub are recorded, tensors bind in semantic order."""
    sdpa = torch.nn.functional.scaled_dot_product_attention
    leaves, info = shim.bind_operand_leaves(sdpa, (x, x, x), {"attn_mask": y, "scale": 0.5, "is_causal": False})
    assert leaves is not None and len(leaves) == 4 and leaves[3] is y
    assert info == {"unbound_scalar_kwargs": ["is_causal", "scale"], "refusal": None}


def test_positional_only_calls_are_untouched():
    assert shim.bind_operand_leaves(torch.mm, (x, y), {}) == ([x, y], {"unbound_scalar_kwargs": [], "refusal": None})
    # scalar-only keywords (a scalar mapping included) never enter a bind, so nothing is recorded as dropped
    assert shim.bind_operand_leaves(torch.mm, (x, y), {"opts": {"a": 1}}) == ([x, y], {"unbound_scalar_kwargs": [], "refusal": None})


# ---------------------------------------------------------------------------
# capture: both device paths refuse with the typed diagnostic
# ---------------------------------------------------------------------------

def _nested_preamble(access):
    """A functools.wraps adapter (wrapped signature ``(input, mat2)``) that also consumes a tensor from ``payload``."""
    return f'''import functools
def legacy(input, mat2):
    return input + mat2
@functools.wraps(legacy)
def adapter(input, mat2, **kwargs):
    extra = kwargs['payload']{access}
    if torch.overrides.has_torch_function_variadic(input, mat2, extra):
        return torch.overrides.handle_torch_function(adapter, (input, mat2, extra), input, mat2=mat2, **kwargs)
    return input + mat2 + extra
'''


NESTED_PREAMBLE = _nested_preamble("['extra']")


@pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "forced-cpu"])
@pytest.mark.parametrize("call, access", [
    ("adapter(q, mat2=k, payload={'extra': v}, unknown_scalar=.5)", "['extra']"),
    ("adapter(q, mat2=k, payload={'outer': {'inner': [v]}})", "['outer']['inner'][0]"),
], ids=["dict", "two-levels"])
def test_capture_refuses_a_tensor_hidden_in_a_mapping_keyword(tmp_path, force_cpu, call, access):
    ok, out, graph = _capture(tmp_path, _source(call, force_cpu, _nested_preamble(access)))
    assert ok is False and graph is None
    # the typed diagnostic names the callee (functools.wraps gives the adapter the wrapped name) and the keyword
    assert "UnsupportedFrontendOperand (accounting-incomplete)" in out and "'legacy'" in out
    assert "payload" in out and "mapping" in out


def test_capture_with_a_scalar_only_dict_keyword_still_succeeds(tmp_path):
    preamble = '''import functools
def legacy(input, mat2):
    return input + mat2
@functools.wraps(legacy)
def adapter(input, mat2, **kwargs):
    if torch.overrides.has_torch_function_variadic(input, mat2):
        return torch.overrides.handle_torch_function(adapter, (input, mat2), input, mat2=mat2, **kwargs)
    return (input + mat2) * kwargs['opts']['alpha']
'''
    ok, out, graph = _capture(tmp_path, _source("adapter(q, mat2=k, opts={'alpha': 0.5})", preamble=preamble))
    assert ok, out
    nodes = [v for v in graph["layers"].values() if v.get("type") == "legacy"]
    assert len(nodes) == 1
    assert nodes[0]["input_shapes"] == [[1, 2, 4, 8], [1, 2, 4, 8]]
    assert nodes[0]["module_args"]["unbound_scalar_kwargs"] == ["opts"]


def test_ci_unit_lane_selects_these_modules():
    import pathlib
    ci = pathlib.Path(__file__).resolve().parents[1] / ".gitlab-ci.yml"
    text = ci.read_text()
    assert "tests/test_nested_tensor_keywords.py" in text and "tests/test_sdpa_nested_delta.py" in text
