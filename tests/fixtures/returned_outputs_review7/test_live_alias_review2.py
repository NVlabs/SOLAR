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
