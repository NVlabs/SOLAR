# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model input that is also returned must keep its external read traffic.

an earlier repair exports returned inputs as ``model_outputs`` with ``is_input``.  The
analyzer must not treat such a producer-less declared output as an in-model
buffer (on-chip read) when real kernels consume it.  Hand-authored metadata
only: no model, DSL or capture execution.
"""
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.einsum_dsl.assembler import build_dsl_graph
from solar.einsum_dsl.einsum_converter import annotate_einsum_equations
from solar.einsum_dsl.exporter import dsl_graph_to_dict
from solar.einsum_lib.metadata import build_metadata

N = 257


def _tensor(tid, name=None, input_alias=False):
    return {'type': 'tensor', 'id': tid, 'dim_names': ['D0'], 'shape': (N,),
            'dtype': 'int32', 'role': None, 'sparsity': None,
            'input_name': name, 'input_alias': input_alias}


def _analyze(returned, tmp_path):
    entries = [
        _tensor(1, 'x', True), _tensor(2),
        {'type': 'call', 'name': 'produce', 'call_type': 'kernel',
         'input_ids': [1], 'output_id': 2, 'call_site_line': None},
        {'type': 'model_outputs', 'schema': 'dsl-return-v1', 'outputs': [
            {'path': ['result', i], 'tensor_id': tid}
            for i, tid in enumerate(returned)]},
    ]
    md = build_metadata(entries)
    graph = build_dsl_graph(md, {'produce': 'O[d] = X[d] + 1'}, 'returned_input')
    graph = annotate_einsum_equations(graph, dsl_graph_to_dict(graph))
    path = tmp_path / 'graph.yaml'
    path.write_text(yaml.safe_dump(graph, sort_keys=False))
    return md, EinsumGraphAnalyzer().analyze_graph(
        path, tmp_path / 'analysis', precision='fp32', copy_graph=False)


def test_returned_input_keeps_consumer_read_external(tmp_path):
    md, result = _analyze(returned=(1, 2), tmp_path=tmp_path)
    assert any(o.get('is_input') for o in md['model_outputs'])
    total = result['total']
    assert total['external_input_bytes'] == N * 4      # produce reads x from DRAM
    assert total['external_output_bytes'] == N * 4     # only produce's output is written
    assert total['fused_bytes'] == 2 * N * 4
    assert total['unfused_bytes'] == 2 * N * 4


def test_not_returning_the_input_gives_the_same_traffic(tmp_path):
    _, result = _analyze(returned=(2,), tmp_path=tmp_path)
    total = result['total']
    assert total['external_input_bytes'] == N * 4
    assert total['external_output_bytes'] == N * 4
