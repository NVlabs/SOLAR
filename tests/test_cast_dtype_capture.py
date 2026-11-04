# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual-cast output dtype capture at the trace recorder (review follow-up).

The recorder shim executes the real function and HOLDS the result when the
FunctionNode is created, yet no output dtype was recorded: torchview nodes
carried no dtype, so stage 1 filled the model-default width and a
``.to(torch.bfloat16)`` result was recorded — and billed — as FP32 (the
``test_sparse_update_of_zero_buffer_writes_touched_rows_only`` gate: 32768
touched BF16 elements billed 131072 bytes instead of 65536). The repair
records the ACTUAL ordered output dtypes from the executed tensor results
in the existing capture path and consumes that evidence before the
model-default fallback. Nothing is inferred from an op name: the converter's
cast-repair refusal (``_repair_torchview_quirks``) is unchanged — evidence
is now recorded upstream, so there is nothing to guess.
"""

from pathlib import Path
from textwrap import dedent

import yaml

import test_graph_analyzer_memory as mem
from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor


def _stage1(tmp_path, model_source):
    (tmp_path / "model.py").write_text(dedent(model_source))
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    processor = PyTorchProcessor(ProcessingConfig(
        save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    ok = processor.process_model_file(
        str(tmp_path / "model.py"), str(graph_dir))
    assert ok, "PyTorchProcessor.process_model_file failed"
    return yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())


def _model(body, inputs="torch.randn(8, 16)"):
    return f"""
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def forward(self, x):
{body}

        def get_inputs():
            return [{inputs}]

        def get_init_inputs():
            return []
    """


class TestActualCastDtypeCapture:

    def test_to_dtype_records_actual_output_dtype(self, tmp_path):
        g = _stage1(tmp_path, _model(
            "                y = x + 1.0\n"
            "                return y.to(torch.bfloat16)"))
        assert g["layers"]["Model.to"]["output_dtypes"] == (
            ["torch.bfloat16"]), (
            "the executed result IS bfloat16; the recorder holds it when "
            "the node is created — record it, never the model default"
        )

    def test_dtype_keyword_records_actual_output_dtype(self, tmp_path):
        g = _stage1(tmp_path, _model(
            "                return x.sum(dim=1, dtype=torch.bfloat16)"))
        assert g["layers"]["Model.sum"]["output_dtypes"] == (
            ["torch.bfloat16"])

    def test_convenience_cast_records_actual_output_dtype(self, tmp_path):
        g = _stage1(tmp_path, _model(
            "                return (x * 2.0).half()"))
        assert g["layers"]["Model.half"]["output_dtypes"] == (
            ["torch.float16"])

    def test_non_cast_control_keeps_trace_dtype(self, tmp_path):
        g = _stage1(tmp_path, _model(
            "                return x + 1.0"))
        assert g["layers"]["Model.add"]["output_dtypes"] == (
            ["torch.float32"])

    def test_mixed_output_dtypes_stay_ordered_per_slot(self, tmp_path):
        g = _stage1(tmp_path, _model(
            "                values, indices = torch.max(x, dim=1)\n"
            "                return values, indices"))
        assert g["layers"]["Model.max"]["output_dtypes"] == (
            ["torch.float32", "torch.int64"]), (
            "each output slot records ITS OWN executed dtype, in order"
        )

    def test_cast_dtype_propagates_through_converter(self, tmp_path):
        mem._run_full_pipeline(tmp_path, _model(
            "                y = x + 1.0\n"
            "                return y.to(torch.bfloat16)"))
        g = yaml.safe_load(
            (tmp_path / "einsum/einsum_graph_renamed.yaml").read_text())
        cast = g["layers"]["Model.to"]
        assert cast["tensor_dtypes"]["outputs"] == ["torch.bfloat16"], (
            "stage-1 recorded cast evidence must survive the einsum "
            "converter so boundary billing prices the ACTUAL stored width"
        )
