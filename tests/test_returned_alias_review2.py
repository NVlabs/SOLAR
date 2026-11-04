# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""real-alias-1 controls.  The parametrised pair below is the reviewer's test_live_alias_review2.py VERBATIM
(checks-d9bc78e2, sha256 af34a7b537a228198107b2eb6441f59090c4478dea8abc268dd83f79e195a841): a bound output plus a
plain exact alias of it must bill 16 B once; a distinct allocation of the same shape must bill 32 B.
The author controls after it cover partial overlap and the storage-identity marking."""
import json,os,textwrap
from pathlib import Path as P
import pytest,torch,yaml
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
@pytest.mark.parametrize('alias',[True,False])
def test_bound_and_plain_return_share_storage_write_once(tmp_path,alias):
    # Public PyTorch API only; no graph, hook or recorder monkeypatching.
    # Different allocations of identical shape must remain separate outputs.
    expression='y.as_subclass(torch.Tensor)' if alias else 'y.clone().as_subclass(torch.Tensor)'
    src='import torch\nclass Model(torch.nn.Module):\n    def forward(self,x):\n        y=x*2\n        p='+expression+'\n        return y,p\ndef get_inputs():return [torch.arange(4,dtype=torch.float32)]\ndef get_init_inputs():return []\n'
    (tmp_path/'model.py').write_text(src)
    ns={};exec(src,ns);vals=ns['Model']()(*ns['get_inputs']());ids=[v.untyped_storage()._cdata for v in vals];expected=16 if alias else 32
    assert (ids[0]==ids[1])==alias
    p=PyTorchProcessor(ProcessingConfig(save_graph=False,force_rerun=True,debug=False,safe_mode=False))
    assert p.process_model_file(str(tmp_path/'model.py'),str(tmp_path/'graph'))
    graph_path=tmp_path/'graph/pytorch_graph.yaml';pg=yaml.safe_load(graph_path.read_text())
    assert PyTorchToEinsum().convert(str(graph_path),str(tmp_path/'einsum')) is not None
    eg=yaml.safe_load((tmp_path/'einsum/einsum_graph_renamed.yaml').read_text())
    a=EinsumGraphAnalyzer().analyze_graph(str(tmp_path/'einsum/einsum_graph_renamed.yaml'),str(tmp_path/'analysis'),precision='fp32')
    result={'alias':alias,'expected_output_bytes':expected,'eager_storage_ids':ids,'eager_equal':torch.equal(*vals),'hook':getattr(p,'_returned_outputs',None),'id_map':p.torchview_processor._original_to_clean_id,'returned_outputs':pg.get('returned_outputs'),'output_nodes':{k:v for k,v in pg['layers'].items() if v['type']=='output-tensor'},'model_outputs':eg.get('model_outputs'),'total':a['total']}
    (tmp_path/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    assert a['total']['external_output_bytes']==expected,result


# --------------------------------------------------------------------------- author controls
def _run(tmp_path, body):
    src = 'import torch\nclass Model(torch.nn.Module):\n    def forward(self,x):\n' + ''.join('        ' + l + '\n' for l in body.split('\n')) + 'def get_inputs():return [torch.arange(4,dtype=torch.float32)]\ndef get_init_inputs():return []\n'
    (tmp_path / 'model.py').write_text(src)
    p = PyTorchProcessor(ProcessingConfig(save_graph=False, force_rerun=True, debug=False, safe_mode=False))
    assert p.process_model_file(str(tmp_path / 'model.py'), str(tmp_path / 'graph'))
    pg = yaml.safe_load((tmp_path / 'graph/pytorch_graph.yaml').read_text())
    assert PyTorchToEinsum().convert(str(tmp_path / 'graph/pytorch_graph.yaml'), str(tmp_path / 'einsum')) is not None
    eg = yaml.safe_load((tmp_path / 'einsum/einsum_graph_renamed.yaml').read_text())
    a = EinsumGraphAnalyzer().analyze_graph(str(tmp_path / 'einsum/einsum_graph_renamed.yaml'), str(tmp_path / 'analysis'), precision='fp32')
    return pg, eg, a['total']


def test_bound_leaf_records_real_storage_identity(tmp_path):
    pg, eg, total = _run(tmp_path, 'y=x*2\nreturn y, y.as_subclass(torch.Tensor)')
    leaves = [r for r in pg['returned_outputs'] if r['kind'] == 'tensor']
    assert leaves[0]['bound'] is True and leaves[1]['bound'] is False
    ids = [l['tensor_identity'] for l in leaves]
    assert ids[0]['storage_identity'] == 'storage_handle' and ids[1]['storage_identity'] == 'storage_handle'
    assert ids[0]['storage_id'] == ids[1]['storage_id'] and not ids[0]['storage_id'].startswith('obj:')
    assert ids[0]['storage_key'] == ids[1]['storage_key']          # exact alias: same storage, offset, shape, stride, width
    assert leaves[1]['alias_of_path'] == 'return[0]'
    # the plain alias is fully covered by the bound output: no second declaration, charged once
    assert not any(mo.get('unbound_returned') for mo in eg['model_outputs'])
    fp = next(iter(eg['returned_footprint'].values()))
    assert fp['method'] == 'alias_of_bound_output' and fp['union_bytes'] == 0 and fp['covered_by_bound_bytes'] == 16
    assert total['external_output_bytes'] == 16 and total.get('status') != 'incomplete'


def test_partial_unbound_view_of_bound_output_bills_remainder_once(tmp_path):
    """y[:2] is traced (bound, bytes 0-8); y[1:] returned as a plain alias (bytes 4-16): the shared bytes 4-8 are charged
    once, the uncovered remainder 8-16 is declared as an unbound output -> 16 B total, not 8 + 12 = 20 B; fail-closed stays."""
    pg, eg, total = _run(tmp_path, 'y=x*2\nreturn y[:2], y[1:].as_subclass(torch.Tensor)')
    declared = [mo for mo in eg['model_outputs'] if mo.get('unbound_returned')]
    assert len(declared) == 1 and declared[0]['footprint_bytes'] == 8
    fp = next(v for v in eg['returned_footprint'].values() if v.get('leaves'))
    assert fp['covered_by_bound_bytes'] == 4 and fp['union_bytes'] == 8
    assert total['external_output_bytes'] == 16
    assert total['status'] == 'incomplete' and 'returned_output_unbound' in total['incomplete_layers'].values()


def test_unbound_view_of_bound_output_fully_covered(tmp_path):
    pg, eg, total = _run(tmp_path, 'y=x*2\nreturn y, y[1:3].as_subclass(torch.Tensor)')
    assert not any(mo.get('unbound_returned') for mo in eg['model_outputs'])
    assert total['external_output_bytes'] == 16 and total.get('status') != 'incomplete'


def test_distinct_storage_unbound_still_declared(tmp_path):
    pg, eg, total = _run(tmp_path, 'y=x*2\nb=torch.zeros(4,dtype=x.dtype,device=x.device)\nb[1:]=x[:3]\nreturn y, b')
    declared = [mo for mo in eg['model_outputs'] if mo.get('unbound_returned')]
    assert len(declared) == 1 and declared[0]['footprint_bytes'] == 16
    assert total['external_output_bytes'] == 32 and total['status'] == 'incomplete'
