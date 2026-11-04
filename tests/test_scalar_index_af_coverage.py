# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A scalar (rank-0) tensor index selects a basic view and converts (R6, gather-integration-1).

``x = arange(60).reshape(6, 10); x[torch.tensor(1)]`` converted and billed on
the earlier baseline ((other_ops, output_elements, unfused_elements) = (0, 0, 10)).  The
lineage's post-emit AF coverage check (af_graph_builder._validate_af_coverage)
refused it: the 0-dim index is a model-input predecessor of ``__getitem__``
with no AF tensor access ("non-weight preds ['start_1'] missing ...; reads=
['start']"), because a rank-0 tensor has no rank the emit could project.  The
check waives only an evidenced INTEGRAL rank-0 index ARGUMENT (slot >= 1, int
dtype) of a basic-view op; a rank-0 operand produced by a real op (Reformer's
``where(cond, x, x.max())``, surfaced once R8 restored the call attributes) is
REPRESENTED as an empty-projection read of its producer; a Boolean scalar mask
(advanced indexing, distinct storage), a rank-0 data scalar in an index slot, or
a rank-0 model input used as data is refused (P2-3 of that review).
"""
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

pytest.importorskip("torch")

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.af_graph_builder import _validate_af_coverage, build_af_graph_from_dict
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import PyTorchProcessor

SCALAR_INDEX_MODEL = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, idx):
            return x[idx]

    def get_inputs():
        return [torch.arange(60).reshape(6, 10), torch.tensor(1)]

    def get_init_inputs():
        return []
"""


def _pipeline(tmp_path: Path):
    (tmp_path / "model.py").write_text(dedent(SCALAR_INDEX_MODEL))
    for n in ("graph", "einsum", "analysis"):
        (tmp_path / n).mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(
        str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
    analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"), str(tmp_path / "analysis"),
                                                   precision="fp32", copy_graph=False)
    assert analysis is not None
    einsum = yaml.safe_load((tmp_path / "einsum/einsum_graph.yaml").read_text())
    return einsum, analysis


def test_scalar_long_tensor_index_is_a_basic_view(tmp_path: Path):
    """The reviewer's unchanged case (test_boundaries.py)."""
    import torch
    x = torch.arange(60).reshape(6, 10)
    assert x[torch.tensor(1)].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    einsum, analysis = _pipeline(tmp_path)
    name, layer = next((n, l) for n, l in analysis["layers"].items() if l["type"] == "__getitem__")
    assert (layer["other_ops"], layer["output_elements"], layer["unfused_elements"]) == (0, 0, 10), layer
    # the scalar index is the second, rank-0 operand of the view op
    assert einsum["layers"][name]["tensor_shapes"]["inputs"] == [[6, 10], []]
    af = build_af_graph_from_dict(einsum)
    notes = [d for d in af.get("_diagnostics", []) if "classified as a scalar index" in str(d)]
    assert len(notes) == 1 and "'Model.__getitem__'" in notes[0], af.get("_diagnostics")
    assert (tmp_path / "einsum/af_einsum_graph.yaml").exists()


def _af_with_single_read(einsum_name: str, read: str) -> dict:
    return {"workload": {"einsums": [{"name": einsum_name, "tensor_accesses": [
        {"name": read, "output": False}, {"name": einsum_name, "output": True}]}]}}


def _layer(op_type: str, second_shape, second_dtype="torch.int64"):
    return {
        "type": op_type,
        "connections": {"inputs": ["start", "start_1"], "outputs": []},
        "tensor_types": {"inputs": ["input", "input"], "outputs": ["output"]},
        "tensor_names": {"inputs": ["start.Output", "start_1.Output"], "outputs": ["Model.op.Output"]},
        "tensor_shapes": {"inputs": [[6, 10], second_shape], "outputs": [[10]]},
        "tensor_dtypes": {"inputs": ["torch.int64", second_dtype], "outputs": ["torch.int64"]},
    }


def test_coverage_exempts_only_rank0_operands_of_indexing_ops():
    layers = {"Model.op": _layer("__getitem__", [])}
    notes = _validate_af_coverage(_af_with_single_read("Model_op", "start"), layers)  # no error
    assert len(notes) == 1 and "scalar index" in notes[0] and "'Model.op'" in notes[0], notes


def test_coverage_stays_fail_closed_outside_the_integral_index_contract():
    refused = [
        _layer("__getitem__", [3]),                              # non-scalar index pred
        _layer("where", [1, 1]),                                 # non-scalar operand of a non-indexing op
        _layer("__getitem__", [], second_dtype="torch.bool"),    # Boolean scalar mask: advanced indexing, not a view
        _layer("__getitem__", [], second_dtype="torch.uint8"),   # byte mask: x[torch.tensor(1, dtype=uint8)] -> [1, 2, 3], distinct storage
        _layer("__getitem__", [], second_dtype="torch.float32"), # a rank-0 data scalar in an index slot
        _layer("where", []),                                     # rank-0 model input used as data (no producer to represent)
    ]
    for layer in refused:
        with pytest.raises(RuntimeError, match=r"non-weight preds \['start_1'\] missing"):
            _validate_af_coverage(_af_with_single_read("Model_op", "start"), {"Model.op": layer})
    # the data tensor in slot 0 is never an index argument, even when rank-0 and integral
    slot0 = {**_layer("__getitem__", [4]), "tensor_shapes": {"inputs": [[], [4]], "outputs": [[1]]}}
    with pytest.raises(RuntimeError, match=r"non-weight preds \['start'\] missing"):
        _validate_af_coverage(_af_with_single_read("Model_op", "start_1"), {"Model.op": slot0})


def test_integral_scalar_index_is_waived_with_its_dtype_in_the_note():
    for dtype in ("torch.int64", "torch.int32"):
        notes = _validate_af_coverage(_af_with_single_read("Model_op", "start"), {"Model.op": _layer("__getitem__", [], second_dtype=dtype)})
        assert len(notes) == 1 and dtype in notes[0] and "scalar index of a basic view" in notes[0], notes


REFORMER_WHERE_MODEL = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            return torch.where(x > 0, x, x.max())

    def get_inputs():
        return [torch.randn(4, 8)]

    def get_init_inputs():
        return []
"""


def test_where_with_a_rank0_tensor_operand_converts(tmp_path: Path):
    (tmp_path / "model.py").write_text(dedent(REFORMER_WHERE_MODEL))
    for n in ("graph", "einsum", "analysis"):
        (tmp_path / n).mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(
        str(tmp_path / "model.py"), str(tmp_path / "graph"))
    assert PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum")) is not None
    einsum = yaml.safe_load((tmp_path / "einsum/einsum_graph.yaml").read_text())
    where = next(l for l in einsum["layers"].values() if l["type"] == "where")
    assert len(where["tensor_shapes"]["inputs"]) == 3 and where["tensor_shapes"]["inputs"][2] == [], where["tensor_shapes"]
    af = build_af_graph_from_dict(einsum)
    # the scalar is REPRESENTED: an empty-projection read of Model.max in the where einsum
    assert any("represented as an empty-projection read" in str(d) for d in af["_diagnostics"]), af["_diagnostics"]
    where_e = next(e for e in af["workload"]["einsums"] if e["name"].endswith("where"))
    reads = {ta["name"]: ta["projection"] for ta in where_e["tensor_accesses"] if not ta.get("output")}
    assert any(n.endswith("max") and list(pr) == [] for n, pr in reads.items()), reads
    analysis = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / "einsum/einsum_graph_renamed.yaml"), str(tmp_path / "analysis"), precision="fp32", copy_graph=False)
    assert analysis is not None


BOOL_SCALAR_INDEX_MODEL = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x, idx):
            return x[idx]

    def get_inputs():
        return [torch.arange(6).reshape(2, 3), torch.tensor(True)]

    def get_init_inputs():
        return []
"""


def test_boolean_scalar_index_is_refused_not_waived(tmp_path: Path):
    """``x[torch.tensor(True)]`` is Boolean advanced indexing (output [1, 2, 3], distinct storage): no waiver."""
    import torch
    x = torch.arange(6).reshape(2, 3)
    y = x[torch.tensor(True)]
    assert list(y.shape) == [1, 2, 3] and y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    (tmp_path / "model.py").write_text(dedent(BOOL_SCALAR_INDEX_MODEL))
    for n in ("graph", "einsum"):
        (tmp_path / n).mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(
        str(tmp_path / "model.py"), str(tmp_path / "graph"))
    with pytest.raises(RuntimeError, match="AF graph coverage check failed"):
        PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum"))


@pytest.mark.parametrize("value,expected_shape", [(1, [1, 2, 3]), (0, [0, 2, 3])])
def test_uint8_scalar_mask_is_refused_not_waived(tmp_path: Path, value, expected_shape):
    """``x[torch.tensor(v, dtype=torch.uint8)]`` is byte-mask advanced indexing (round-2 P2-3)."""
    import torch
    x = torch.arange(6).reshape(2, 3)
    y = x[torch.tensor(value, dtype=torch.uint8)]
    assert list(y.shape) == expected_shape and y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    src = dedent(BOOL_SCALAR_INDEX_MODEL).replace("torch.tensor(True)", f"torch.tensor({value}, dtype=torch.uint8)")
    (tmp_path / "model.py").write_text(src)
    for n in ("graph", "einsum"):
        (tmp_path / n).mkdir()
    assert PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False)).process_model_file(
        str(tmp_path / "model.py"), str(tmp_path / "graph"))
    with pytest.raises(RuntimeError, match="AF graph coverage check failed"):
        PyTorchToEinsum().convert(str(tmp_path / "graph/pytorch_graph.yaml"), str(tmp_path / "einsum"))
