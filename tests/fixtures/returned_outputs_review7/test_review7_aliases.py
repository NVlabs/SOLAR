import copy,itertools,json,os
from pathlib import Path
import pytest,torch,yaml
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

def capture(tmp_path,expression):
 src='import torch\nclass Model(torch.nn.Module):\n    def forward(self,x):\n        y=x*2\n        return '+expression+'\ndef get_inputs():return [torch.arange(8,dtype=torch.float32)]\ndef get_init_inputs():return []\n'
 f=tmp_path/'model.py';f.write_text(src)
 ns={};exec(src,ns)
 vals=ns['Model']()(*ns['get_inputs']());vals=vals if isinstance(vals,tuple) else (vals,)
 oracle=set();meta=[]
 for t in vals:
  st=t.untyped_storage()
  for index in itertools.product(*(range(d) for d in t.shape)):
   offset=(t.storage_offset()+sum(i*s for i,s in zip(index,t.stride())))*t.element_size()
   oracle.update((st._cdata,b) for b in range(offset,offset+t.element_size()))
  meta.append({'storage_offset':t.storage_offset(),'stride':list(t.stride()),'shape':list(t.shape),'element_size':t.element_size(),'storage_nbytes':st.nbytes()})
 p=PyTorchProcessor(ProcessingConfig(save_graph=False,force_rerun=True,safe_mode=False))
 assert p.process_model_file(str(f),str(tmp_path/'graph'))
 pg=yaml.safe_load((tmp_path/'graph/pytorch_graph.yaml').read_text())
 return pg,len(oracle),meta

def convert(tmp_path,pg):
 path=tmp_path/'probe.yaml';path.write_text(yaml.safe_dump(pg,sort_keys=False))
 assert PyTorchToEinsum().convert(str(path),str(tmp_path/'einsum')) is not None
 eg=yaml.safe_load((tmp_path/'einsum/einsum_graph_renamed.yaml').read_text())
 a=EinsumGraphAnalyzer().analyze_graph(str(tmp_path/'einsum/einsum_graph_renamed.yaml'),str(tmp_path/'analysis'),precision='fp32')
 (tmp_path/'RESULT.json').write_text(json.dumps({'returned':pg.get('returned_outputs'),'footprint':eg.get('returned_footprint'),'model_outputs':eg.get('model_outputs'),'total':a['total']},indent=2))
 return eg,a['total']

@pytest.mark.parametrize('expression',[
 'y, y[:5].as_subclass(torch.Tensor), y[3:].as_subclass(torch.Tensor)',
 'y[2:6], y[:4].as_subclass(torch.Tensor), y[4:].as_subclass(torch.Tensor)',
 'y[4:], y[:4].as_subclass(torch.Tensor)',
 'y[2:6], y[:4].as_subclass(torch.Tensor)',
 'y[::2], y[1::2].as_subclass(torch.Tensor)',
 'y[::2], y[2:6].as_subclass(torch.Tensor)',
 'y[1:6:2], y[3:8:2].as_subclass(torch.Tensor)',
 'y, y.clone().as_subclass(torch.Tensor), y.clone().as_subclass(torch.Tensor)',
])
def test_live_region_oracle(tmp_path,expression):
 pg,expected,meta=capture(tmp_path,expression)
 eg,total=convert(tmp_path,pg)
 (tmp_path/'oracle.json').write_text(json.dumps({'expected_output_bytes':expected,'eager_metadata':meta,'expression':expression},indent=2))
 assert total['external_output_bytes']==expected
 for leaf,truth in zip(pg['returned_outputs'],meta):
  ident=leaf['tensor_identity']
  assert all(ident[k]==v for k,v in truth.items())
 if any(m.get('unbound_returned') for m in eg['model_outputs']):
  assert total['status']=='incomplete'

def test_negative_stride_alias_of_bound_metadata(tmp_path):
 # Torch cannot construct negative strides; exercise supported capture metadata, explicitly not a live torch view.
 pg,_,_=capture(tmp_path,'y[4:], y[:4].as_subclass(torch.Tensor)')
 leaf=pg['returned_outputs'][1];ident=leaf['tensor_identity']
 ident.update(storage_offset=7,stride=[-1],shape=[4],numel=4)
 leaf['shape']=[4];leaf['nbytes']=16
 ident['storage_key']=ident['storage_id']+':7:[4]:[-1]:4'
 eg,total=convert(tmp_path,pg)
 assert total['external_output_bytes']==16
 fp=next(v for v in eg['returned_footprint'].values() if v.get('leaves'))
 assert fp['covered_by_bound_bytes']==16 and fp['method']=='alias_of_bound_output'

class StorageUnavailable(torch.Tensor):
 @staticmethod
 def __new__(cls,t):
  return torch.Tensor._make_subclass(cls,t,False)
 def as_subclass(self,typ):
  raise RuntimeError('storage/plain view unavailable for this tensor subclass')
 def untyped_storage(self):
  raise RuntimeError('storage extraction unavailable')

def test_genuine_storage_failure_marked_never_deduped(tmp_path):
 base=torch.arange(4,dtype=torch.float32)
 alias=StorageUnavailable(base)
 leaves=PyTorchProcessor._flatten_returned((alias,alias,StorageUnavailable(base.clone())))
 assert all(l['tensor_identity']['storage_identity']=='object_fallback' for l in leaves)
 assert all(l['tensor_identity']['storage_id'].startswith('obj:') for l in leaves)
 # Same unavailable-storage object repeated must not alias according to the requested fail-closed contract.
 (tmp_path/'flatten.json').write_text(json.dumps(leaves,indent=2))
 pg,_,_=capture(tmp_path,'y')
 for l in leaves:l['bound_node']=None
 pg['returned_outputs']+=leaves
 eg,total=convert(tmp_path,pg)
 declared=[m for m in eg['model_outputs'] if m.get('unbound_returned')]
 assert len(declared)==3 and sum(m['footprint_bytes'] for m in declared)==48
 assert total['external_output_bytes']==80 and total['status']=='incomplete'

def test_object_fallback_same_literal_key_never_cross_dedupes(tmp_path):
 pg,_,_=capture(tmp_path,'y, y.clone().as_subclass(torch.Tensor)')
 for leaf in pg['returned_outputs']:
  leaf['tensor_identity']['storage_id']='obj:sentinel'
  leaf['tensor_identity']['storage_identity']='object_fallback'
 eg,total=convert(tmp_path,pg)
 assert total['external_output_bytes']==64 and total['status']=='incomplete'

