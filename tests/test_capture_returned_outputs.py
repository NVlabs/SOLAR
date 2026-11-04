# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issue #53 (SOL58-OUTPUT-BOUNDARY): every returned output of forward() is
bound as an external output; an output the recorder never traced (allocated
in forward and filled in place) is charged at its declared footprint AND the
accounting fails closed (``status: incomplete``, reason
``returned_output_unbound``).  Patterns: the row-58 shape (zeros + slice
assign + tuple return), a tuple with one allocated output, a dict return.
"""
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph.pytorch_processor import PyTorchProcessor

ROW58_PATTERN = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, topk_idx):
            num_experts = 4
            flat = topk_idx.reshape(-1)
            _, sorted_token_indices = flat.sort(stable=True)
            expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
            expert_offsets[1:] = torch.bincount(flat.long(), minlength=num_experts).cumsum(0).to(torch.int32)
            return sorted_token_indices.to(torch.int32), expert_offsets

    def get_inputs():
        return [torch.randint(0, 4, (2, 8, 2), dtype=torch.int32)]

    def get_init_inputs():
        return []
"""

TUPLE_ALLOCATED = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            y = x * 2.0
            buf = torch.empty(3, 7, dtype=torch.float32)
            buf.fill_(1.0)
            return y, buf

    def get_inputs():
        return [torch.randn(5, 6)]

    def get_init_inputs():
        return []
"""

DICT_RETURN = """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            return {"a": x * 2.0, "b": (x + 1.0).sum(dim=-1)}

    def get_inputs():
        return [torch.randn(5, 6)]

    def get_init_inputs():
        return []
"""


def _pipeline(tmp_path: Path, src: str):
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(src))
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    proc = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(graph_dir))
    pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    eg = yaml.safe_load((einsum_dir / "einsum_graph.yaml").read_text())
    an_dir = tmp_path / "analysis"
    an_dir.mkdir()
    EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph.yaml"), str(an_dir), copy_graph=False)
    analysis = yaml.safe_load((an_dir / "analysis.yaml").read_text())
    return pg, eg, analysis


def _tensor_leaves(pg):
    return [r for r in pg.get("returned_outputs") or [] if r.get("kind") == "tensor"]


def test_row58_pattern_second_output_bound_and_fail_closed(tmp_path):
    pg, eg, an = _pipeline(tmp_path, ROW58_PATTERN)
    leaves = _tensor_leaves(pg)
    assert pg.get("model_output_provenance") == "pytorch-return-v1"
    assert [l["path"] for l in leaves] == ["return[0]", "return[1]"]
    assert leaves[0]["shape"] == [32] and leaves[0]["dtype"] == "torch.int32"
    assert leaves[1]["shape"] == [5] and leaves[1]["dtype"] == "torch.int32"
    # the allocate-then-fill output is returned but never traced
    assert leaves[1]["bound"] is False and leaves[1]["bound_node"] is None
    assert pg["returned_outputs_unbound"] == 1
    # converter declares it as a model output ...
    declared = [mo for mo in eg["model_outputs"] if mo.get("unbound_returned")]
    assert len(declared) == 1 and declared[0]["shape"] == [5]
    # ... the analyzer charges both outputs: 32*4 + 5*4 bytes ...
    total = an["total"]
    assert total["external_output_bytes"] == 32 * 4 + 5 * 4
    # ... and fails closed instead of silently pricing only the bound one
    assert total["status"] == "incomplete"
    assert "returned_output_unbound" in total["incomplete_layers"].values()
    assert total["returned_outputs_unbound"][0]["shape"] == [5]


def test_tuple_return_with_allocated_output(tmp_path):
    """``torch.empty`` + ``fill_`` IS traced by the recorder (a producer-less
    hidden tensor feeding a ``fill_`` node), so both tuple leaves bind and
    the footprint sums both outputs with no marker.  Only an untraced
    producer (``out[1:] = ...`` on an allocated tensor, the row-58 pattern)
    is unbound."""
    pg, eg, an = _pipeline(tmp_path, TUPLE_ALLOCATED)
    leaves = _tensor_leaves(pg)
    assert [l["path"] for l in leaves] == ["return[0]", "return[1]"]
    assert [l["shape"] for l in leaves] == [[5, 6], [3, 7]]
    assert all(l["bound"] and l["bound_node"] for l in leaves)
    assert pg["returned_outputs_unbound"] == 0
    assert an["total"]["external_output_bytes"] == 5 * 6 * 4 + 3 * 7 * 4
    assert an["total"].get("status") != "incomplete"


def test_dict_return_all_bound_no_marker(tmp_path):
    pg, eg, an = _pipeline(tmp_path, DICT_RETURN)
    leaves = _tensor_leaves(pg)
    assert sorted(l["path"] for l in leaves) == ["return['a']", "return['b']"]
    assert all(l["bound"] for l in leaves)
    assert pg["returned_outputs_unbound"] == 0
    assert not any(mo.get("unbound_returned") for mo in eg.get("model_outputs") or [])
    assert an["total"]["external_output_bytes"] == 5 * 6 * 4 + 5 * 4
    assert an["total"].get("status") != "incomplete"


def test_legacy_graph_without_returned_outputs_is_unchanged(tmp_path):
    """A pre-#53 pytorch_graph.yaml (no returned_outputs key) converts and
    analyses exactly as before: no declared unbound outputs, no marker."""
    pg, eg, an = _pipeline(tmp_path, DICT_RETURN)
    graph_dir = tmp_path / "graph"
    for k in ("returned_outputs", "returned_outputs_unbound", "model_output_provenance"):
        pg.pop(k, None)
    (graph_dir / "pytorch_graph.yaml").write_text(yaml.safe_dump(pg, sort_keys=False))
    legacy_dir = tmp_path / "einsum_legacy"
    legacy_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(legacy_dir)) is not None
    eg2 = yaml.safe_load((legacy_dir / "einsum_graph.yaml").read_text())
    assert "returned_outputs" not in eg2
    an2_dir = tmp_path / "analysis_legacy"
    an2_dir.mkdir()
    EinsumGraphAnalyzer().analyze_graph(str(legacy_dir / "einsum_graph.yaml"), str(an2_dir), copy_graph=False)
    an2 = yaml.safe_load((an2_dir / "analysis.yaml").read_text())
    assert an2["total"]["external_output_bytes"] == an["total"]["external_output_bytes"]
    assert an2["total"].get("status") != "incomplete"


# ---------------------------------------------------------------------------
# Independent reviewer round-1 controls (an earlier fix / issue #53), model bodies VERBATIM
# from the reviewer's independent_controls.py and fault_injection.py.
# ---------------------------------------------------------------------------
import textwrap

from solar.perf import EinsumGraphPerfModel

_REVIEWER_CASES = [
    # (name, forward body, expected unique external-output bytes, expect unbound marker)
    ("same_bound", "y=x*2\nreturn y,y", 120, False),
    ("same_unbound", "b=torch.zeros(5, dtype=x.dtype, device=x.device)\nb[1:]=x.reshape(-1)[:4]\nreturn b,b", 20, True),
    ("distinct_unbound", "a=torch.zeros(5, dtype=x.dtype, device=x.device)\nb=torch.zeros(5, dtype=x.dtype, device=x.device)\na[1:]=x.reshape(-1)[:4]\nb[1:]=x.reshape(-1)[:4]\nreturn a,b", 40, True),
]


def _reviewer_model(body: str) -> str:
    return ("import torch\nfrom torch import nn\nclass Model(nn.Module):\n    def forward(self,x):\n"
            + textwrap.indent(body, "        ")
            + "\ndef get_inputs():\n    return [torch.arange(30,dtype=torch.float32).reshape(5,6)]\ndef get_init_inputs():\n    return []\n")


def _run_control(tmp_path: Path, body: str):
    model_file = tmp_path / "model.py"
    model_file.write_text(_reviewer_model(body))
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    proc = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(graph_dir))
    pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    eg = yaml.safe_load((einsum_dir / "einsum_graph.yaml").read_text())
    an_dir = tmp_path / "analysis"; an_dir.mkdir()
    EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph.yaml"), str(an_dir), copy_graph=False)
    an = yaml.safe_load((an_dir / "analysis.yaml").read_text())
    perf = EinsumGraphPerfModel().predict(an_dir / "analysis.yaml", tmp_path / "perf", arch_config="B200")
    return pg, eg, an, perf


def _qualification(perf) -> str:
    # the field lives under perf["workload"] (reviewer's corrected interpretation)
    return str((perf.get("workload") or {}).get("prediction_qualification") or perf.get("prediction_qualification") or "")


@pytest.mark.parametrize("name,body,expected_bytes,expect_unbound", _REVIEWER_CASES, ids=[c[0] for c in _REVIEWER_CASES])
def test_reviewer_alias_controls(tmp_path, name, body, expected_bytes, expect_unbound):
    """return-alias-1: the same unbound tensor returned twice is ONE 20-byte
    footprint; two distinct unbound buffers are 40 bytes; bound repeats dedup to 120."""
    pg, eg, an, perf = _run_control(tmp_path, body)
    leaves = _tensor_leaves(pg)
    assert len(leaves) == 2
    assert an["total"]["external_output_bytes"] == expected_bytes
    if name == "same_unbound":
        assert leaves[0]["tensor_identity"]["storage_key"] == leaves[1]["tensor_identity"]["storage_key"]
        assert leaves[1]["alias_of_path"] == "return[0]"
        assert sum(1 for mo in eg["model_outputs"] if mo.get("unbound_returned")) == 1
    if name == "distinct_unbound":
        assert leaves[0]["tensor_identity"]["storage_key"] != leaves[1]["tensor_identity"]["storage_key"]
        assert sum(1 for mo in eg["model_outputs"] if mo.get("unbound_returned")) == 2
    if expect_unbound:
        assert an["total"]["status"] == "incomplete"
        assert "returned_output_unbound" in an["total"]["incomplete_layers"].values()
        assert _qualification(perf).startswith("unqualified")
    else:
        assert an["total"].get("status") != "incomplete"


def test_reviewer_fault_injection_extractor_failure_fails_closed(tmp_path):
    """return-hook-error-2 (fault_injection.py verbatim model): when the
    returned-output extractor raises, the capture records `unrecorded`, the converter
    propagates it, the analysis total is incomplete (`returned_outputs_unrecorded`) and
    the B200 prediction is unqualified -- never a silently priced legacy-looking result."""
    model_file = tmp_path / "model.py"
    model_file.write_text("import torch\nclass Model(torch.nn.Module):\n    def forward(self,x):\n        b=torch.zeros(5,dtype=x.dtype,device=x.device)\n        b[1:]=x.reshape(-1)[:4]\n        return x*2,b\ndef get_inputs():return [torch.arange(30,dtype=torch.float32).reshape(5,6)]\ndef get_init_inputs():return []\n")
    proc = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))

    def injected_failure(out):
        raise RuntimeError("reviewer-injected output-metadata extraction failure")

    proc._flatten_returned = injected_failure
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    assert proc.process_model_file(str(model_file), str(graph_dir))
    pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
    assert pg["returned_outputs_unrecorded"] is True
    assert any(r.get("kind") == "unrecorded" for r in pg["returned_outputs"])
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    eg = yaml.safe_load((einsum_dir / "einsum_graph.yaml").read_text())
    assert eg["returned_outputs_unrecorded"] is True
    an_dir = tmp_path / "analysis"; an_dir.mkdir()
    EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph.yaml"), str(an_dir), copy_graph=False)
    an = yaml.safe_load((an_dir / "analysis.yaml").read_text())
    assert an["total"]["status"] == "incomplete"
    assert an["total"]["incomplete_layers"].get("returned_outputs") == "returned_outputs_unrecorded"
    perf = EinsumGraphPerfModel().predict(an_dir / "analysis.yaml", tmp_path / "perf", arch_config="B200")
    assert _qualification(perf).startswith("unqualified")


# ---------------------------------------------------------------------------
# Independent reviewer round-2 controls (return-stride-4 / return-region-5):
# __init__ / forward bodies and the byte-address oracle VERBATIM from
# the reviewer's additional_controls_v2.py.  Expected unique
# returned bytes: overlapping 32, disjoint 32, stride collision 24, exact
# views 16, direct + dict alias 20.  Each unbound result stays incomplete /
# unqualified; the footprint is the per-storage union of byte regions.
# ---------------------------------------------------------------------------
import importlib.util
import itertools


def _reviewer_oracle(v):
    leaves = []

    def walk(x):
        if isinstance(x, torch.Tensor):
            leaves.append(x)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)
        elif isinstance(x, (tuple, list)):
            for y in x:
                walk(y)
    walk(v)
    regions = {}
    description = []
    for x in leaves:
        sid = str(x.untyped_storage()._cdata); width = x.element_size(); offset = x.storage_offset(); addresses = set()
        for idx in itertools.product(*[range(int(n)) for n in x.shape]):
            element = offset + sum(i * s for i, s in zip(idx, x.stride()))
            addresses.update(range(element * width, (element + 1) * width))
        regions.setdefault(sid, set()).update(addresses)
        description.append({'shape': list(x.shape), 'stride': list(x.stride()), 'offset': offset, 'numel': x.numel(), 'element_size': width, 'storage': sid, 'bytes': len(addresses)})
    return {'unique_bytes': sum(len(v) for v in regions.values()), 'leaves': description}


_REVIEWER_V2_CASES = [
    # (name, __init__ body, forward body, expected unique returned bytes, expected declared unbound outputs)
    ('cached_overlap', 'self.b=torch.zeros(8)\nself.a=self.b[:5]\nself.c=self.b[3:]', 'self.b[:]=x.reshape(-1)[:8]\nreturn self.a,self.c', 32, 1),
    ('cached_disjoint', 'self.b=torch.zeros(8)\nself.a=self.b[:4]\nself.c=self.b[4:]', 'self.b[:]=x.reshape(-1)[:8]\nreturn self.a,self.c', 32, 1),
    ('cached_stride_collision', 'self.b=torch.zeros(8)\nself.a=self.b[:4]\nself.c=self.b[::2]', 'self.b[:]=x.reshape(-1)[:8]\nreturn self.a,self.c', 24, 1),
    ('cached_exact_views', 'self.b=torch.zeros(8)\nself.a=self.b[:4]\nself.c=self.b[:4]', 'self.b[:]=x.reshape(-1)[:8]\nreturn self.a,self.c', 16, 1),
    ('direct_and_dict', 'pass', 'b=torch.zeros(5,dtype=x.dtype,device=x.device)\nb[1:]=x.reshape(-1)[:4]\nreturn b,{"again":b}', 20, 1),
]


def _reviewer_v2_source(init: str, body: str) -> str:
    return ('import torch\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n'
            + textwrap.indent(init, '        ') + '\n    def forward(self,x):\n' + textwrap.indent(body, '        ')
            + '\ndef get_inputs():return [torch.arange(30,dtype=torch.float32).reshape(5,6)]\ndef get_init_inputs():return []\n')


import torch  # noqa: E402  (oracle + source loading)


@pytest.mark.parametrize("name,init,body,expected_bytes,expected_declared", _REVIEWER_V2_CASES, ids=[c[0] for c in _REVIEWER_V2_CASES])
def test_reviewer_v2_region_union_controls(tmp_path, name, init, body, expected_bytes, expected_declared):
    model_file = tmp_path / "model.py"
    model_file.write_text(_reviewer_v2_source(init, body))
    spec = importlib.util.spec_from_file_location("reviewer_v2_" + name, model_file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    oracle = _reviewer_oracle(m.Model()(*m.get_inputs()))
    assert oracle["unique_bytes"] == expected_bytes  # the reviewer's expected value, reproduced by the oracle
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    proc = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(graph_dir))
    pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
    leaves = _tensor_leaves(pg)
    assert len(leaves) == 2 and all(l["bound"] is False for l in leaves)
    idents = [l["tensor_identity"] for l in leaves]
    assert all("stride" in i and "shape" in i for i in idents)
    if name in ("cached_exact_views", "direct_and_dict"):
        # exact alias: storage, offset, shape, stride and width all equal
        assert idents[0]["storage_key"] == idents[1]["storage_key"]
        assert leaves[1]["alias_of_path"] == "return[0]"
    else:
        # same storage, different bytes: distinct leaves (never an exact alias)
        assert idents[0]["storage_id"] == idents[1]["storage_id"]
        assert idents[0]["storage_key"] != idents[1]["storage_key"]
        assert leaves[1]["alias_of_path"] is None
    if name == "cached_stride_collision":
        assert idents[0]["storage_offset"] == idents[1]["storage_offset"] and idents[0]["numel"] == idents[1]["numel"]
        assert idents[0]["stride"] != idents[1]["stride"]
    einsum_dir = tmp_path / "einsum"; einsum_dir.mkdir()
    assert PyTorchToEinsum().convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir)) is not None
    eg = yaml.safe_load((einsum_dir / "einsum_graph.yaml").read_text())
    declared = [mo for mo in eg["model_outputs"] if mo.get("unbound_returned")]
    assert len(declared) == expected_declared
    assert declared[0]["footprint_bytes"] == expected_bytes
    assert declared[0]["footprint_method"] == "byte_region_union"
    assert sorted(declared[0]["returned_paths"]) == sorted(l["path"] for l in leaves)   # provenance of every leaf kept
    assert len(eg["returned_outputs"]) == 2
    an_dir = tmp_path / "analysis"; an_dir.mkdir()
    EinsumGraphAnalyzer().analyze_graph(str(einsum_dir / "einsum_graph.yaml"), str(an_dir), copy_graph=False)
    an = yaml.safe_load((an_dir / "analysis.yaml").read_text())
    assert an["total"]["external_output_bytes"] == expected_bytes == oracle["unique_bytes"]
    assert an["total"]["status"] == "incomplete"
    assert "returned_output_unbound" in an["total"]["incomplete_layers"].values()
    assert an["total"]["returned_outputs_unbound"][0]["footprint_bytes"] == expected_bytes
    perf = EinsumGraphPerfModel().predict(an_dir / "analysis.yaml", tmp_path / "perf", arch_config="B200")
    assert _qualification(perf).startswith("unqualified")


def test_returned_region_helpers_unit():
    """Interval helper edge cases: non-unit innermost stride, 2-D strided view,
    clipping to the storage size, legacy identity without shape/stride."""
    from solar.einsum.pytorch_to_einsum import _merge_byte_intervals, _returned_leaf_byte_intervals, _interval_total
    e = 4
    iv, m = _returned_leaf_byte_intervals({"storage_offset": 0, "shape": [4], "stride": [2], "element_size": e}, {})
    assert m == "byte_region_union" and _interval_total(iv) == 16 and iv == [(0, 4), (8, 12), (16, 20), (24, 28)]
    iv, m = _returned_leaf_byte_intervals({"storage_offset": 1, "shape": [2, 3], "stride": [6, 1], "element_size": e}, {})
    assert m == "byte_region_union" and iv == [(4, 16), (28, 40)]
    merged = _merge_byte_intervals([(0, 20), (12, 32), (40, 48)], storage_nbytes=44)
    assert merged == [(0, 32), (40, 44)] and _interval_total(merged) == 36
    iv, m = _returned_leaf_byte_intervals({"storage_offset": 2, "numel": 5, "element_size": e}, {"shape": None})
    assert m == "contiguous_assumed_legacy_identity" and iv == [(8, 28)]


# ---------------------------------------------------------------------------
# the review round-3 controls (negative-stride-7): ``oracle`` / ``observed``
# helpers and ``test_signed_stride_metadata_exact_or_rejected`` VERBATIM from
# checks-4f8de95a/test_review2_regions.py (sha256 3ee197df…); the small
# positive/zero-stride oracle sweep is the same control with its
# ``REVIEW_RUN_OUT`` receipt written under ``tmp_path`` instead.
# ---------------------------------------------------------------------------
import json as _json
from solar.einsum.pytorch_to_einsum import _returned_leaf_byte_intervals as intervals, _merge_byte_intervals as merge, _interval_total as total


def oracle(shape,stride,offset,width,size):
 b=set()
 for index in itertools.product(*(range(n) for n in shape)):
  start=(offset+sum(i*s for i,s in zip(index,stride)))*width
  b.update(x for x in range(start,start+width) if 0<=x<size)
 return b
def observed(shape,stride,offset,width,size):
 iv,m=intervals({'shape':shape,'stride':stride,'storage_offset':offset,'element_size':width,'storage_nbytes':size},{})
 return merge(iv,size),m


@pytest.mark.parametrize('shape,stride,offset,size',[([4],[-1],3,16),([4],[-2],6,32)])
def test_signed_stride_metadata_exact_or_rejected(shape,stride,offset,size):
 iv,m=observed(shape,stride,offset,4,size)
 actual=set(x for a,b in iv for x in range(a,b));expected=oracle(shape,stride,offset,4,size)
 assert actual==expected,{'intervals':iv,'method':m,'actual_bytes':len(actual),'expected_bytes':len(expected)}


def test_positive_and_zero_stride_small_oracle(tmp_path):
 OUT=tmp_path
 count=0
 for shape in ([0],[1],[4],[2,3],[3,2,2]):
  for stride in itertools.product(range(4),repeat=len(shape)):
   for offset in [0,2]:
    iv,m=observed(shape,stride,offset,4,80)
    expected=oracle(shape,stride,offset,4,80)
    actual=set(x for a,b in iv for x in range(a,b))
    assert actual==expected,(shape,stride,offset,iv,expected)
    assert m=='byte_region_union';count+=1
 (OUT/'small-oracle.json').write_text(_json.dumps({'cases':count,'all_exact':True}))


def test_signed_stride_multi_dim_and_mixed_sign_oracle():
    """Author addition: negative strides in a 2-D view, mixed signs, and a
    negative stride on a size-1 dimension (no reflection) all match the
    signed-address oracle and stay labelled exact."""
    for shape, stride, offset, size in [([2, 3], [-3, 1], 3, 24), ([2, 3], [3, -1], 2, 24), ([3, 2], [-2, -1], 5, 24), ([1, 4], [-7, 1], 0, 16), ([4], [-1], 3, 12)]:
        iv, m = observed(shape, stride, offset, 4, size)
        actual = set(x for a, b in iv for x in range(a, b)); expected = oracle(shape, stride, offset, 4, size)
        assert actual == expected and m == 'byte_region_union', (shape, stride, offset, iv)
