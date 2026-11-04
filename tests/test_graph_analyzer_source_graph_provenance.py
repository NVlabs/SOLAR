# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""issue #63 -- ``analysis.metadata.source_graph`` must be the exact selected input graph path.

On 90840700 ``EinsumGraphAnalyzer.analyze_graph`` bound the input path to a local named ``src`` and later reused that name
for (i) the returned-output loop over ``metadata.model_outputs`` (``src = op`` / ``_trace_source_through_views(op)``) and (ii)
the gather / embedding read-slot index (``src = len(input_sizes) - 1 if op_type == "embedding" else 0``), so the provenance
field emitted the last returned operator's name (``prefix_offsets#0``, ``Model.einsum``) or a slot index instead of the path.
The strict ``ANALYSIS_SOURCE_GRAPH_PATH_NOT_EQUAL`` guard downstream refused correctly.  The repair is a distinct local for the
path and nothing else; these regressions pin (a) the returned-output case, (b) the embedding / gather case, (c) that every
other analysis field is byte-identical to the 90840700 analyzer's output on the in-repo fixture graphs, and (d) that the
reordered-graph selection still reports the reordered path.
"""
import json
import pathlib

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

FX = pathlib.Path(__file__).parent / "fixtures"
GOLD = FX / "graph_analyzer_source_graph"
INDEX = json.loads((GOLD / "INDEX.json").read_text())
CASES = INDEX["cases"]


def _analyze(graph, out):
    a = EinsumGraphAnalyzer().analyze_graph(str(graph), str(out), precision="fp32", copy_graph=False)
    assert isinstance(a, dict), graph
    return a


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c["has_model_outputs"]))
def test_returned_output_loop_does_not_overwrite_source_graph(tmp_path, name):
    """(a) a graph whose metadata carries ``model_outputs`` (dsl-return-v1): on 90840700 the provenance was the last returned
    operator (recorded in INDEX.json as ``base_source_graph_value``); here it is the exact input path."""
    c = CASES[name]; graph = FX / c["graph"]
    assert not c["base_source_graph_is_path"], c                       # the fixture reproduces the defect on the base
    a = _analyze(graph, tmp_path / name)
    assert a["metadata"]["source_graph"] == str(graph), (a["metadata"]["source_graph"], c["base_source_graph_value"])


def test_embedding_read_slot_reuse_does_not_overwrite_source_graph(tmp_path):
    """(b) an ``embedding`` layer makes the analyzer reuse the name for the gathered read slot (``len(input_sizes) - 1``):
    on 90840700 the provenance became that slot index as a string; here it is the input path."""
    layers = {
        "start": {"type": "start", "einsum_equation": "->VD", "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": False,
                  "tensor_names": {"inputs": [], "outputs": ["start.W", "start.I"]}, "tensor_types": {"inputs": [], "outputs": ["weight", "input"]},
                  "tensor_shapes": {"inputs": [], "outputs": [[50, 8], [2, 7]]}, "connections": {"inputs": [], "outputs": ["Model.emb"]}},
        "Model.emb": {"type": "embedding", "einsum_equation": "VD,BL->BLD", "elementwise_op": "copy", "reduction_op": "none", "is_real_einsum": False, "is_einsum_supportable": True,
                      "tensor_names": {"inputs": ["start.W", "start.I"], "outputs": ["Model.emb.Output"]}, "tensor_types": {"inputs": ["weight", "input"], "outputs": ["output"]},
                      "tensor_shapes": {"inputs": [[50, 8], [2, 7]], "outputs": [[2, 7, 8]]}, "connections": {"inputs": ["start"], "outputs": []}},
    }
    p = tmp_path / "einsum_graph_embedding.yaml"
    p.write_text(yaml.safe_dump({"model_name": "emb", "layers": layers}, sort_keys=False))
    a = _analyze(p, tmp_path / "analysis")
    assert "Model.emb" in a["layers"], list(a["layers"])
    assert a["metadata"]["source_graph"] == str(p), a["metadata"]["source_graph"]
    assert a["metadata"]["source_graph"] not in ("0", "1")


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_non_provenance_field_is_byte_identical_to_90840700(tmp_path, name):
    """(c) the fix changes the provenance field only: the analysis of each in-repo fixture graph, with ``metadata.source_graph``
    removed, is byte-identical (sorted-key JSON) to the golden stored next to INDEX.json (first produced by the 90840700
    analyzer; regenerated additively by issue #58 -- see ``golden_pin`` in INDEX.json)."""
    c = CASES[name]; graph = FX / c["graph"]
    a = _analyze(graph, tmp_path / name)
    assert a["metadata"].pop("source_graph") == str(graph)
    got = json.dumps(a, indent=1, sort_keys=True, default=str) + "\n"
    assert got == (GOLD / f"{name}.golden.analysis.json").read_text(), name


def test_reordered_graph_selection_still_reports_the_reordered_path(tmp_path):
    """(d) unchanged behaviour: when ``einsum_graph.yaml`` sits next to ``einsum_graph_reordered.yaml`` the reordered graph is
    selected and ``source_graph`` names THAT file (the selected input), exactly as before."""
    c = CASES[sorted(CASES)[0]]; src = (FX / c["graph"]).read_text()
    (tmp_path / "einsum_graph.yaml").write_text(src); (tmp_path / "einsum_graph_reordered.yaml").write_text(src)
    a = _analyze(tmp_path / "einsum_graph.yaml", tmp_path / "analysis")
    assert a["metadata"]["source_graph"] == str(tmp_path / "einsum_graph_reordered.yaml")
