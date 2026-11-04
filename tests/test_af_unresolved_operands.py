# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AF graph: untraced (``unresolved``) einsum operands do not consume a graph predecessor (R7).

an earlier change (the operand-evidence fix, issue #46 / INTEG-2) types an einsum operand the capture did not
trace as ``unresolved`` and gives every operand a role of its own.  The
lineage's af_graph_builder paired input roles with ``connections.inputs``
positionally, skipping only ``weight`` slots: with two unresolved operands in
front, the Mamba2 SSD einsum's traced operands (L from exp_1, X from reshape)
were unioned against the wrong roles and the emit named the wrong tensors, so
the one-rank-tuple invariant refused the graph ("tensor 'Model_exp_1' has
inconsistent rank tuples across accesses"; INTEG-5b).  ``unresolved`` slots
are now predecessor-less like weights in the union, the emit and the coverage
check; their reads are emitted under their recorded ``*.Unresolved[_k]`` name
as declared untraced sources, and the invariant keeps rejecting genuinely
inconsistent accesses.

Fixture: the Mamba2 v2 capture converted on the operand-evidence fix working tree
(d03-multioperand-einsum-fix/repair-integ2/evidence/
mamba2_v2capture_einsum_graph.repaired-working-tree.yaml, gzip-copied).
"""

import pytest

from solar.einsum.af_graph_builder import _ranks_of_projection, build_af_graph_from_dict


def _accesses(af):
    return {e["name"]: {(ta["name"], bool(ta.get("output"))): _ranks_of_projection(ta["projection"])
                        for ta in e["tensor_accesses"]} for e in af["workload"]["einsums"]}


def _start(name, shape):
    labels = [f"{c}0" for c in "ABCDEFG"[: len(shape)]]
    return {"type": "start", "einsum_equation": "->" + "".join(labels), "is_real_einsum": False,
            "operands": {"Output": labels}, "tensor_shapes": {"inputs": [], "outputs": [shape]},
            "tensor_types": {"inputs": [], "outputs": ["input"]},
            "tensor_names": {"inputs": [], "outputs": [f"{name}.Output"]}, "connections": {"inputs": [], "outputs": []}}


def _graph_with_unresolved_first_operand(consumer_types=("unresolved", "input", "input")):
    layers = {"start": _start("start", [4, 5]), "start_1": _start("start_1", [5, 6])}
    layers["start"]["connections"]["outputs"] = ["Model.e"]
    layers["start_1"]["connections"]["outputs"] = ["Model.e"]
    layers["Model.e"] = {
        "type": "einsum", "einsum_equation": "IJ,JK,KL->IL", "is_real_einsum": True,
        "elementwise_op": "mul", "reduction_op": "add",
        "operands": {"Input": ["I", "J"], "Output": ["I", "L"], "Input_1": ["J", "K"], "Input_2": ["K", "L"]},
        "tensor_shapes": {"inputs": [[3, 4], [4, 5], [5, 6]], "outputs": [[3, 6]]},
        "tensor_types": {"inputs": list(consumer_types), "outputs": ["output"]},
        "tensor_names": {"inputs": ["Model.e.Unresolved", "start.Output", "start_1.Output"], "outputs": ["Model.e.Output"]},
        "connections": {"inputs": ["start", "start_1"], "outputs": []},
    }
    return {"model_name": "t", "layers": layers}


def test_unresolved_slot_does_not_consume_a_predecessor():
    af = build_af_graph_from_dict(_graph_with_unresolved_first_operand())
    acc = _accesses(af)
    e = acc["Model_e"]
    assert e[("start", False)] == acc["start"][("start", True)]
    assert e[("start_1", False)] == acc["start_1"][("start_1", True)]
    assert ("Model_e_Unresolved", False) in e
    rank_sizes = af["workload"]["rank_sizes"]
    assert sorted(rank_sizes[r] for r in e[("Model_e_Unresolved", False)]) == [3, 4]


def test_genuinely_inconsistent_accesses_are_still_refused():
    graph = _graph_with_unresolved_first_operand()
    # the producer's recorded output is [5, 4] while the consumer reads it as [4, 5]: no position
    # can be unioned, so the read mints its own ranks and the one-rank-tuple gate must refuse
    graph["layers"]["start"]["tensor_shapes"]["outputs"] = [[5, 4]]
    with pytest.raises(RuntimeError, match="inconsistent rank tuples"):
        build_af_graph_from_dict(graph)
