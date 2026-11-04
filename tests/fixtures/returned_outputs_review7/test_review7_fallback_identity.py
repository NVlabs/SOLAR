import json
import torch,yaml
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

def test_live_unavailable_identity_alias_of_bound_stays_unqualified(tmp_path):
 src='''import torch
class HiddenStorage(torch.Tensor):
    @staticmethod
    def __new__(cls,t):
        return torch.Tensor._make_subclass(cls,t,False)
    def as_subclass(self,typ):
        raise RuntimeError("plain storage view unavailable")
    def untyped_storage(self):
        raise RuntimeError("storage accessor unavailable")
class Model(torch.nn.Module):
    def forward(self,x):
        y=x*2
        return y,HiddenStorage(y.as_subclass(torch.Tensor))
def get_inputs():return [torch.arange(8,dtype=torch.float32)]
def get_init_inputs():return []
'''
 f=tmp_path/'model.py';f.write_text(src)
 ns={};exec(src,ns);y,p=ns['Model']()(*ns['get_inputs']())
 # Ground truth using PyTorch base methods; the subclass's public accessors genuinely raise.
 assert y.untyped_storage()._cdata==torch.Tensor.untyped_storage(p)._cdata
 processor=PyTorchProcessor(ProcessingConfig(save_graph=False,force_rerun=True,safe_mode=False))
 assert processor.process_model_file(str(f),str(tmp_path/'graph'))
 pg=yaml.safe_load((tmp_path/'graph/pytorch_graph.yaml').read_text())
 assert pg['returned_outputs'][0]['bound'] and not pg['returned_outputs'][1]['bound']
 assert pg['returned_outputs'][1]['tensor_identity']['storage_identity']=='object_fallback'
 assert PyTorchToEinsum().convert(str(tmp_path/'graph/pytorch_graph.yaml'),str(tmp_path/'einsum')) is not None
 eg=yaml.safe_load((tmp_path/'einsum/einsum_graph_renamed.yaml').read_text())
 a=EinsumGraphAnalyzer().analyze_graph(str(tmp_path/'einsum/einsum_graph_renamed.yaml'),str(tmp_path/'analysis'),precision='fp32')
 (tmp_path/'RESULT.json').write_text(json.dumps({'returned':pg['returned_outputs'],'model_outputs':eg['model_outputs'],'total':a['total']},indent=2))
 assert a['total']['external_output_bytes']==64 # conservative: unknown-storage leaf must not dedupe
 assert a['total']['status']=='incomplete'

class PointerOnly(torch.Tensor):
 @staticmethod
 def __new__(cls,t):return torch.Tensor._make_subclass(cls,t,False)
 def as_subclass(self,typ):raise RuntimeError('plain view unavailable')
 def untyped_storage(self):
  real=torch.Tensor.untyped_storage(self)
  class Proxy:
   def data_ptr(self):return real.data_ptr()
   def nbytes(self):return real.nbytes()
  return Proxy()

def test_data_pointer_fallback_distinguishes_real_allocations(tmp_path):
 base=torch.arange(4,dtype=torch.float32)
 values=(PointerOnly(base),PointerOnly(base),PointerOnly(base.clone()))
 leaves=PyTorchProcessor._flatten_returned(values)
 (tmp_path/'flatten.json').write_text(json.dumps(leaves,indent=2))
 ids=[l['tensor_identity'] for l in leaves]
 assert all(i['storage_identity']=='data_ptr' for i in ids)
 assert ids[0]['storage_id']==ids[1]['storage_id']
 assert ids[0]['storage_id']!=ids[2]['storage_id']

