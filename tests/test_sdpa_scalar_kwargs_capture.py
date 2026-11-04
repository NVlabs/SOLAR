# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scalar keywords unknown to a builtin's testing-override signature never refuse a capture (ACC-140-SDPA-KWARGS).

torch's testing-override stub for ``scaled_dot_product_attention`` is ``(query, key, value, attn_mask=None,
dropout_p=0.0)`` and predates ``is_causal`` / ``scale`` / ``enable_gqa``.  The shim's operand binder used to fail the
whole bind on such a scalar keyword and refuse the call with a "tensor keyword arguments" message, although every
tensor operand's semantic slot was known.  The binder now retries with the tensor-bearing keywords only, records the
dropped scalar keywords on the node (``unbound_scalar_kwargs``), and keeps failing closed, with an accurate message, when
a TENSOR keyword cannot be placed.
"""
import contextlib
import io

import pytest
import torch
import torch.nn.functional as F
import yaml

from solar.common.types import ProcessingConfig
from solar.graph import _torchview_param_shim as shim
from solar.graph.pytorch_processor import PyTorchProcessor

INPUTS = "torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), torch.zeros(1, 2, 4, 4)"


def _source(call, force_cpu=False, preamble=""):
    guard = "        if q.device.type == 'meta':\n            raise RuntimeError('force the CPU fallback')\n" if force_cpu else ""
    return ("import torch, torch.nn.functional as F\nfrom torch import nn\n" + preamble
            + "class Model(nn.Module):\n    def forward(self, q, k, v, mask):\n" + guard + f"        return {call}\n"
            + f"def get_inputs():\n    return [{INPUTS}]\ndef get_init_inputs():\n    return []\n")


def _capture(tmp_path, code):
    (tmp_path / "model.py").write_text(code)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        ok = PyTorchProcessor(ProcessingConfig(force_rerun=True, safe_mode=False)).process_model_file(
            str(tmp_path / "model.py"), str(tmp_path / "graph"))
    graph = yaml.safe_load((tmp_path / "graph" / "pytorch_graph.yaml").read_text()) if ok else None
    return ok, stream.getvalue(), graph


def _sdpa_node(graph):
    nodes = [v for v in graph["layers"].values() if v.get("type") == "scaled_dot_product_attention"]
    assert len(nodes) == 1, [v.get("type") for v in graph["layers"].values()]
    return nodes[0]


@pytest.mark.parametrize("force_cpu", [False, True], ids=["meta", "cpu-fallback"])
def test_route_140_spelling_captures_with_scale_keyword(tmp_path, force_cpu):
    ok, out, graph = _capture(tmp_path, _source("F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.125)", force_cpu))
    assert ok, out
    node = _sdpa_node(graph)
    assert node["input_shapes"] == [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 4]]   # q, k, v, attn_mask in semantic order
    assert node["module_args"]["unbound_scalar_kwargs"] == ["scale"]
    assert "scale: 0.125" in str(node["module_args"]["raw_attributes"])                     # the value is still recorded


def test_is_causal_and_enable_gqa_keywords_capture(tmp_path):
    """Scalar-only keywords never enter a bind (no tensor keyword to place): captured, values in raw_attributes."""
    ok, out, graph = _capture(tmp_path, _source("F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False, scale=0.5)"))
    assert ok, out
    node = _sdpa_node(graph)
    assert node["input_shapes"] == [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]]
    assert "unbound_scalar_kwargs" not in node["module_args"]            # nothing was dropped from a bind
    raw = str(node["module_args"]["raw_attributes"])
    assert "is_causal: True" in raw and "enable_gqa: False" in raw and "scale: 0.5" in raw


def test_is_causal_with_a_tensor_mask_records_the_dropped_scalars(tmp_path):
    ok, out, graph = _capture(tmp_path, _source("F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, enable_gqa=False, scale=0.5)"))
    assert ok, out
    node = _sdpa_node(graph)
    assert node["input_shapes"] == [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 4]]
    assert node["module_args"]["unbound_scalar_kwargs"] == ["enable_gqa", "is_causal", "scale"]


@pytest.mark.parametrize("call", ["F.scaled_dot_product_attention(q, k, v, attn_mask=mask)",
                                  "F.scaled_dot_product_attention(q, key=k, value=v, attn_mask=mask)",
                                  "F.scaled_dot_product_attention(q, k, v, mask)"], ids=["mask_kw", "kv_mask_kw", "positional"])
def test_existing_spellings_unchanged(tmp_path, call):
    ok, out, graph = _capture(tmp_path, _source(call))
    assert ok, out
    node = _sdpa_node(graph)
    assert node["input_shapes"] == [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 4]]
    assert "unbound_scalar_kwargs" not in node["module_args"]


def test_tensor_keyword_the_stub_cannot_place_still_refuses_with_an_accurate_message(tmp_path):
    preamble = ("def adapter(input, **kwargs):\n    m = kwargs['mat3']; dst = kwargs['out']\n"
                "    if torch.overrides.has_torch_function_variadic(input, m, dst):\n"
                "        return torch.overrides.handle_torch_function(adapter, (input, m, dst), input, **kwargs)\n"
                "    return torch.mm(input, m, out=dst)\n")
    ok, out, _ = _capture(tmp_path, _source("adapter(q[0, 0], **{'mat3': k[0, 0].t(), 'out': mask[0, 0]})", preamble=preamble))
    assert ok is False
    assert "UnsupportedFrontendOperand (accounting-incomplete)" in out and "'adapter'" in out
    assert "mat3" in out and "catch-all" in out                       # the message names the unplaceable tensor keyword


def test_binder_unit_contract():
    x = torch.ones(2, 2)
    sdpa = F.scaled_dot_product_attention
    leaves, info = shim.bind_operand_leaves(sdpa, (x, x, x), {"attn_mask": x, "scale": 0.5, "is_causal": False})
    assert leaves is not None and len(leaves) == 4 and info["unbound_scalar_kwargs"] == ["is_causal", "scale"] and info["refusal"] is None
    leaves, info = shim.bind_operand_leaves(torch.mm, (x,), {"mat3": x})           # unknown TENSOR keyword on a stub-only builtin
    assert leaves is None and "mat3" in info["refusal"] and "cannot be placed" in info["refusal"]

    def adapter(input, **kwargs):
        return input
    leaves, info = shim.bind_operand_leaves(adapter, (x,), {"mat3": x})
    assert leaves is None and "catch-all" in info["refusal"]
    assert shim._semantic_operand_leaves(sdpa, (x, x, x), {}) == [x, x, x]
    # torch's aliasing destination is split off before the bind (an earlier change / INTEG-10): ``mm(out=)`` binds its two
    # operands and records no dropped scalar keyword
    leaves, info = shim.bind_operand_leaves(torch.mm, (x, x), {"out": torch.empty(2, 2)})
    assert leaves == [x, x] and info == {"unbound_scalar_kwargs": [], "refusal": None}


def test_ci_unit_lane_selects_this_module():
    import pathlib
    ci = (pathlib.Path(__file__).resolve().parents[1] / ".gitlab-ci.yml").read_text()
    assert "tests/test_sdpa_scalar_kwargs_capture.py" in ci
