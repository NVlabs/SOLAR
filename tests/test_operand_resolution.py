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

"""Operands of expanded ops are found by signature position, not by label.

``scaled_dot_product_attention``, ``multi_head_attention_forward``,
``torch.lstm`` and ``torch.gru`` must count a weight, bias, mask or initial
state the same whether it is an ``nn.Parameter``, a plain input tensor, a
keyword argument or a tensor computed in ``forward``.  Each table below runs
the full pipeline on one op and checks

* the fused traffic equals the analytic sum of real operands and outputs,
* the MACs equal the analytic count,
* every activation input of every layer is produced by some layer (no
  phantom names), and the resolved operands reference their producers.
"""
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.operands import SIGNATURES, Operand, bind_operands
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor

HEADER = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
"""


def _run(tmp_path: Path, init: str, args: str, body: str, inputs: str) -> dict:
    """Trace, convert and analyze one forward; returns a bundle of results."""
    src = HEADER + dedent(f"""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                {init}

            def forward(self, {args}):
                return {body}

        def get_inputs():
            return [{inputs}]

        def get_init_inputs():
            return []
        """)
    (tmp_path / "model.py").write_text(src)
    for d in ("graph", "einsum", "analysis"):
        (tmp_path / d).mkdir()
    processor = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert processor.process_model_file(str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert PyTorchToEinsum().convert(str(tmp_path / "graph" / "pytorch_graph.yaml"), str(tmp_path / "einsum"))
    analysis = EinsumGraphAnalyzer().analyze_graph(
        str(tmp_path / "einsum" / "einsum_graph_renamed.yaml"), str(tmp_path / "analysis"),
        precision="fp32", copy_graph=False,
    )
    layers = yaml.safe_load((tmp_path / "einsum" / "einsum_graph.yaml").read_text())["layers"]
    af = yaml.safe_load((tmp_path / "einsum" / "af_einsum_graph.yaml").read_text())
    reads = {e["name"]: [t["name"] for t in e["tensor_accesses"] if not t.get("output")]
             for e in af["workload"]["einsums"]}
    return {"analysis": analysis, "layers": layers, "reads": reads}


def _check_wiring(layers: dict) -> None:
    """Every activation input names a tensor some layer produces.

    Tensors torchview could not trace to a producer (``hidden-tensor``
    placeholders such as an RNN's zero initial state) are the one
    legitimate exception; the AF builder accepts the same names.
    """
    produced = {n for l in layers.values() for n in l["tensor_names"]["outputs"]}
    for lid, l in layers.items():
        names, types = l["tensor_names"]["inputs"], l["tensor_types"]["inputs"]
        activations = {n for n, t in zip(names, types)
                       if t != "weight" and "hidden-tensor" not in n and ".Input_" not in n}
        assert activations <= produced, f"{lid}: unproduced inputs {activations - produced}"


def _elems(*shapes):
    total = 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        total += n
    return total


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------
B, H, Q, K, D = 2, 4, 8, 8, 16
QKV = "torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16)"


class TestSdpaOperands:
    # name: (forward args, body, extra inputs, mask shape or None, expected mask reader)
    CASES = {
        "no_mask": ("q, k, v", "F.scaled_dot_product_attention(q, k, v)", "", None, None),
        "mask_kw": ("q, k, v, m", "F.scaled_dot_product_attention(q, k, v, attn_mask=m)",
                    ", torch.randn(8, 8)", (8, 8), "start_3"),
        "mask_pos": ("q, k, v, m", "F.scaled_dot_product_attention(q, k, v, m)",
                     ", torch.randn(8, 8)", (8, 8), "start_3"),
        "mask_bool": ("q, k, v, m", "F.scaled_dot_product_attention(q, k, v, attn_mask=m)",
                      ", torch.ones(2, 1, 8, 8, dtype=torch.bool)", (2, 1, 8, 8), "start_3"),
        "mask_computed": ("q, k, v, m", "F.scaled_dot_product_attention(q, k, v, attn_mask=m * 2)",
                          ", torch.randn(2, 4, 8, 8)", (2, 4, 8, 8), "Model_mul"),
        "mask_sorts_first": ("q, k, v, bias",
                             "F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), "
                             "v.transpose(1, 2), attn_mask=bias.to(q.dtype))",
                             ", torch.randn(2, 4, 8, 8)", (2, 4, 8, 8), "Model_to"),
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_sdpa(self, tmp_path, name):
        args, body, extra, mask_shape, mask_reader = self.CASES[name]
        inputs = QKV + extra
        if name == "mask_sorts_first":
            # [B, S, H, D] inputs, transposed inside forward
            inputs = "torch.randn(2, 8, 4, 16), torch.randn(2, 8, 4, 16), torch.randn(2, 8, 4, 16)" + extra
        r = _run(tmp_path, "pass", args, body, inputs)
        _check_wiring(r["layers"])
        expected = _elems((B, H, Q, D), (B, H, K, D), (B, H, K, D), (B, H, Q, D))
        if mask_shape:
            expected += _elems(mask_shape)
        assert r["analysis"]["total"]["fused_elements"] == expected
        assert r["analysis"]["total"]["macs"] == 2 * B * H * Q * K * D
        sub = {k: v for k, v in r["reads"].items() if "scaled_dot_product_attention" in k}
        names = [k.rsplit("_", 1)[-1] for k in sub]
        assert ("mask" in names) == bool(mask_shape)
        if mask_shape:
            assert sub["Model_scaled_dot_product_attention_mask"][1] == mask_reader

    def test_sdpa_5d(self, tmp_path):
        """Blocked attention [B, M, Nh, C, D] keeps every leading dim."""
        r = _run(tmp_path, "pass", "q, k, v", "F.scaled_dot_product_attention(q, k, v)",
                 "torch.randn(2, 3, 4, 16, 8), torch.randn(2, 3, 4, 16, 8), torch.randn(2, 3, 4, 16, 8)")
        _check_wiring(r["layers"])
        assert r["analysis"]["total"]["fused_elements"] == 4 * 2 * 3 * 4 * 16 * 8
        assert r["analysis"]["total"]["macs"] == 2 * (2 * 3 * 4) * 16 * 16 * 8


# ---------------------------------------------------------------------------
# multi_head_attention_forward
# ---------------------------------------------------------------------------
L, NB, E, NH = 8, 2, 16, 2
MHA_W = "torch.randn(48, 16), torch.randn(48), torch.randn(16, 16), torch.randn(16)"
MHA_FUNC = ("F.multi_head_attention_forward(q, q, q, 16, 2, {w}, {b}, None, None, False, 0.0, "
            "{ow}, {ob}, need_weights=False{extra})[0]")


def _mha_expected(mask_shape=None, bias=True):
    total = _elems((L, NB, E), (3 * E, E), (E, E), (L, NB, E))
    if bias:
        total += _elems((3 * E,), (E,))
    if mask_shape:
        total += _elems(mask_shape)
    return total


MHA_MACS = L * NB * E * 3 * E + 2 * NB * NH * L * L * (E // NH) + L * NB * E * E


class TestMhaOperands:
    # name: (init, forward args, body, inputs, mask shape, expected AF reads of in_proj)
    CASES = {
        "module": ("self.mha = nn.MultiheadAttention(16, 2)", "q", "self.mha(q, q, q)[0]",
                   "torch.randn(8, 2, 16)", None, ["start", "W1"]),
        "plain": ("pass", "q, w, b, ow, ob", MHA_FUNC.format(w="w", b="b", ow="ow", ob="ob", extra=""),
                  "torch.randn(8, 2, 16), " + MHA_W, None, ["start", "start_1"]),
        "mask_kw": ("pass", "q, w, b, ow, ob, m",
                    MHA_FUNC.format(w="w", b="b", ow="ow", ob="ob", extra=", attn_mask=m"),
                    "torch.randn(8, 2, 16), " + MHA_W + ", torch.randn(8, 8)", (8, 8), ["start", "start_1"]),
        "computed_weight": ("pass", "q, w, b, ow, ob",
                            MHA_FUNC.format(w="w * 2", b="b", ow="ow", ob="ob", extra=""),
                            "torch.randn(8, 2, 16), " + MHA_W, None, ["start", "Model_mul"]),
        "mixed": ("self.mha = nn.MultiheadAttention(16, 2)", "q, ow",
                  MHA_FUNC.format(w="self.mha.in_proj_weight", b="self.mha.in_proj_bias",
                                  ow="ow", ob="self.mha.out_proj.bias", extra=""),
                  "torch.randn(8, 2, 16), torch.randn(16, 16)", None, ["start", "W1"]),
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_mha(self, tmp_path, name):
        init, args, body, inputs, mask_shape, in_proj_reads = self.CASES[name]
        r = _run(tmp_path, init, args, body, inputs)
        _check_wiring(r["layers"])
        assert r["analysis"]["total"]["fused_elements"] == _mha_expected(mask_shape)
        assert r["analysis"]["total"]["macs"] == MHA_MACS
        assert r["reads"]["Model_multi_head_attention_forward_in_proj"] == in_proj_reads
        assert ("Model_multi_head_attention_forward_attn_mask" in r["reads"]) == bool(mask_shape)

    def test_cross_attention_projects_each_source(self, tmp_path):
        """key/value differ from query: three projections, weight read once."""
        r = _run(tmp_path, "self.mha = nn.MultiheadAttention(16, 2)", "q, kv", "self.mha(q, kv, kv)[0]",
                 "torch.randn(8, 2, 16), torch.randn(4, 2, 16)")
        _check_wiring(r["layers"])
        expected = _elems((8, 2, 16), (4, 2, 16), (48, 16), (48,), (16, 16), (16,), (8, 2, 16))
        assert r["analysis"]["total"]["fused_elements"] == expected
        assert r["analysis"]["total"]["macs"] == (8 + 4 + 4) * 2 * 16 * 16 + 2 * 2 * 2 * 8 * 4 * 8 + 8 * 2 * 16 * 16
        assert {k[len("Model_multi_head_attention_forward_"):] for k in r["reads"] if k.endswith("_proj")} == {
            "q_proj", "k_proj", "v_proj", "out_proj"}


# ---------------------------------------------------------------------------
# lstm / gru
# ---------------------------------------------------------------------------
S, RB, I, RH = 8, 2, 16, 32


def _rnn_expected(gates, layers=1, dirs=1, biases=True, states=0, batch_first=False):
    """x + every weight/bias + ``states`` traced initial states + output.

    The zero state ``nn.LSTM``/``nn.GRU`` create inside forward is a
    creation op and is not read from memory.
    """
    G = gates * RH
    total = _elems((S, RB, I), (S, RB, dirs * RH))
    for layer in range(layers):
        inp = I if layer == 0 else dirs * RH
        for _ in range(dirs):
            total += _elems((G, inp), (G, RH)) + (2 * G if biases else 0)
    total += _elems((layers * dirs, RB, RH)) * states
    return total


def _rnn_macs(gates, layers=1, dirs=1):
    G = gates * RH
    macs = 0
    for layer in range(layers):
        inp = I if layer == 0 else dirs * RH
        macs += dirs * S * RB * G * (inp + RH)
    return macs


class TestRnnOperands:
    # name: (init, forward args, body, inputs, kind, kwargs for expected)
    CASES = {
        "lstm_module": ("self.rnn = nn.LSTM(16, 32)", "x", "self.rnn(x)[0]", "torch.randn(8, 2, 16)",
                        "lstm", {}),
        "gru_module": ("self.rnn = nn.GRU(16, 32)", "x", "self.rnn(x)[0]", "torch.randn(8, 2, 16)",
                       "gru", {}),
        "lstm_plain": ("pass", "x, h0, c0, wih, whh, bih, bhh",
                       "torch.lstm(x, (h0, c0), [wih, whh, bih, bhh], True, 1, 0.0, False, False, False)[0]",
                       "torch.randn(8, 2, 16), torch.randn(1, 2, 32), torch.randn(1, 2, 32), "
                       "torch.randn(128, 16), torch.randn(128, 32), torch.randn(128), torch.randn(128)",
                       "lstm", {"states": 2}),
        "lstm_plain_no_bias": ("pass", "x, h0, c0, wih, whh",
                               "torch.lstm(x, (h0, c0), [wih, whh], False, 1, 0.0, False, False, False)[0]",
                               "torch.randn(8, 2, 16), torch.randn(1, 2, 32), torch.randn(1, 2, 32), "
                               "torch.randn(128, 16), torch.randn(128, 32)", "lstm", {"biases": False, "states": 2}),
        "gru_plain": ("pass", "x, h0, wih, whh, bih, bhh",
                      "torch.gru(x, h0, [wih, whh, bih, bhh], True, 1, 0.0, False, False, False)[0]",
                      "torch.randn(8, 2, 16), torch.randn(1, 2, 32), torch.randn(96, 16), torch.randn(96, 32), "
                      "torch.randn(96), torch.randn(96)", "gru", {"states": 1}),
        "gru_computed_weight": ("pass", "x, h0, wih, whh, bih, bhh",
                                "torch.gru(x, h0, [wih * 2, whh, bih, bhh], True, 1, 0.0, False, False, False)[0]",
                                "torch.randn(8, 2, 16), torch.randn(1, 2, 32), torch.randn(96, 16), torch.randn(96, 32), "
                                "torch.randn(96), torch.randn(96)", "gru", {"states": 1}),
        "lstm_given_state": ("self.rnn = nn.LSTM(16, 32)", "x, h0, c0", "self.rnn(x, (h0, c0))[0]",
                             "torch.randn(8, 2, 16), torch.randn(1, 2, 32), torch.randn(1, 2, 32)", "lstm", {"states": 2}),
        "lstm_2layer_bidir": ("self.rnn = nn.LSTM(16, 32, num_layers=2, bidirectional=True)", "x",
                              "self.rnn(x)[0]", "torch.randn(8, 2, 16)", "lstm", {"layers": 2, "dirs": 2}),
        "gru_batch_first": ("self.rnn = nn.GRU(16, 32, batch_first=True)", "x", "self.rnn(x)[0]",
                            "torch.randn(2, 8, 16)", "gru", {"batch_first": True}),
    }

    @pytest.mark.parametrize("name", list(CASES))
    def test_rnn(self, tmp_path, name):
        init, args, body, inputs, kind, kw = self.CASES[name]
        gates = {"lstm": 4, "gru": 3}[kind]
        r = _run(tmp_path, init, args, body, inputs)
        _check_wiring(r["layers"])
        assert r["analysis"]["total"]["fused_elements"] == _rnn_expected(gates, **kw)
        assert r["analysis"]["total"]["macs"] == _rnn_macs(gates, kw.get("layers", 1), kw.get("dirs", 1))
        if name == "gru_computed_weight":
            assert r["reads"]["Model_gru_l0_ih_linear"] == ["start", "Model_mul"]


# ---------------------------------------------------------------------------
# bind_operands on its own
# ---------------------------------------------------------------------------
def _op(i):
    return Operand(i, f"t{i}", [1], None, "input", None)


class TestBindOperands:
    def test_keyword_tensor_after_positionals(self):
        node = {"module_args": {"raw_attributes":
                "[[Tensor(shape=(2, 4), dtype=torch.float32), Tensor(shape=(2, 4), dtype=torch.float32), "
                "Tensor(shape=(2, 4), dtype=torch.float32)], {attn_mask: Tensor(shape=(4, 4), dtype=torch.float32), "
                "is_causal: False}]"}}
        roles = bind_operands(node, [_op(0), _op(1), _op(2), _op(3)], SIGNATURES["scaled_dot_product_attention"])
        assert [roles[k].slot for k in ("query", "key", "value", "attn_mask")] == [0, 1, 2, 3]
        assert roles["is_causal"] is False

    def test_unrecorded_keyword_tensor_is_none(self):
        node = {"module_args": {"raw_attributes":
                "[[Tensor(shape=(2, 4), dtype=torch.float32), Tensor(shape=(2, 4), dtype=torch.float32), "
                "Tensor(shape=(2, 4), dtype=torch.float32)], {attn_mask: Tensor(shape=(4, 4), dtype=torch.float32)}]"}}
        roles = bind_operands(node, [_op(0), _op(1), _op(2)], SIGNATURES["scaled_dot_product_attention"])
        assert roles["attn_mask"] is None

    def test_none_and_scalar_arguments_consume_no_slot(self):
        node = {"module_args": {"raw_attributes":
                "[[Tensor(shape=(8, 2, 16), dtype=torch.float32), Tensor(shape=(8, 2, 16), dtype=torch.float32), "
                "Tensor(shape=(8, 2, 16), dtype=torch.float32), 16, 2, Tensor(shape=(48, 16), dtype=torch.float32), "
                "None, None, None, False, 0.0, Tensor(shape=(16, 16), dtype=torch.float32), None], {}]"}}
        roles = bind_operands(node, [_op(i) for i in range(5)], SIGNATURES["multi_head_attention_forward"])
        assert roles["num_heads"] == 2
        assert roles["in_proj_bias"] is None
        assert roles["in_proj_weight"].slot == 3 and roles["out_proj_weight"].slot == 4

    def test_list_argument_and_zero_dim_tensor(self):
        node = {"module_args": {"raw_attributes":
                "[[Tensor(shape=(8, 2, 16), dtype=torch.float32), Tensor(shape=(), dtype=torch.float32), "
                "[Tensor(shape=(96, 16), dtype=torch.float32), Tensor(shape=(96, 32), dtype=torch.float32)], "
                "False, 1, 0.0, False, False, True], {}]"}}
        roles = bind_operands(node, [_op(0), _op(1), _op(2)], SIGNATURES["gru"])
        assert roles["hx"] is None
        assert [o.slot for o in roles["params"]] == [1, 2]
        assert roles["batch_first"] is True

    def test_without_call_record_binds_by_position(self):
        roles = bind_operands({}, [_op(0), _op(1), _op(2)], SIGNATURES["scaled_dot_product_attention"])
        assert [roles[k].slot for k in ("query", "key", "value")] == [0, 1, 2]
        assert roles["attn_mask"] is None
