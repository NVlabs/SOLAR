import itertools,json,os
from pathlib import Path
import pytest,torch,yaml
from solar.einsum.pytorch_to_einsum import _returned_leaf_byte_intervals as intervals,_merge_byte_intervals as merge,_interval_total as total,PyTorchToEinsum
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.common.types import ProcessingConfig
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel
OUT=Path(os.environ['REVIEW_RUN_OUT'])
def oracle(shape,stride,offset,width,size):
 b=set()
 for index in itertools.product(*(range(n) for n in shape)):
  start=(offset+sum(i*s for i,s in zip(index,stride)))*width
  b.update(x for x in range(start,start+width) if 0<=x<size)
 return b
def observed(shape,stride,offset,width,size):
 iv,m=intervals({'shape':shape,'stride':stride,'storage_offset':offset,'element_size':width,'storage_nbytes':size},{})
 return merge(iv,size),m
def test_positive_and_zero_stride_small_oracle():
 count=0
 for shape in ([0],[1],[4],[2,3],[3,2,2]):
  for stride in itertools.product(range(4),repeat=len(shape)):
   for offset in [0,2]:
    iv,m=observed(shape,stride,offset,4,80)
    expected=oracle(shape,stride,offset,4,80)
    actual=set(x for a,b in iv for x in range(a,b))
    assert actual==expected,(shape,stride,offset,iv,expected)
    assert m=='byte_region_union';count+=1
 (OUT/'small-oracle.json').write_text(json.dumps({'cases':count,'all_exact':True}))
@pytest.mark.parametrize('shape,stride,offset,size',[([4],[-1],3,16),([4],[-2],6,32)])
def test_signed_stride_metadata_exact_or_rejected(shape,stride,offset,size):
 iv,m=observed(shape,stride,offset,4,size)
 actual=set(x for a,b in iv for x in range(a,b));expected=oracle(shape,stride,offset,4,size)
 assert actual==expected,{'intervals':iv,'method':m,'actual_bytes':len(actual),'expected_bytes':len(expected)}
def test_torch_negative_stride_is_not_constructible():
 errors={}
 for key,f in {'as_strided':lambda:torch.as_strided(torch.zeros(8),(4,),(-1,),3),'slice':lambda:torch.zeros(8)[::-1]}.items():
  try:f()
  except (ValueError,RuntimeError) as exc:errors[key]=str(exc)
 assert len(errors)==2
 (OUT/'torch-negative-stride.json').write_text(json.dumps(errors,indent=2))
def test_large_noncontiguous_fallback_and_end_clipping():
 n=(1<<20)+1
 iv,m=observed([n],[2],2,4,8*n)
 assert m=='bounding_interval'
 assert iv==[(8,8*n)]
 assert total(iv)==8*n-8
def test_legacy_identity_is_labelled_and_clipped():
 iv,m=intervals({'storage_offset':3,'numel':4,'element_size':4},{'shape':[4]})
 assert m=='contiguous_assumed_legacy_identity'
 assert merge(iv,20)==[(12,20)]
 assert merge([(-8,8),(12,28),(99,100)],20)==[(0,8),(12,20)]
@pytest.mark.parametrize('kind,init,expected',[('zero_stride','self.b=torch.zeros(1)\nself.v=self.b.expand(9)',4),('empty','self.b=torch.zeros(1)\nself.v=self.b[:0]',0)])
def test_live_unbound_zero_stride_and_empty(tmp_path,kind,init,expected):
 import textwrap
 src='import torch\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n'+textwrap.indent(init,'        ')+'\n    def forward(self,x):\n        self.b[:]=x[:1]\n        return self.v\ndef get_inputs():return [torch.arange(4,dtype=torch.float32)]\ndef get_init_inputs():return []\n'
 p=tmp_path/'model.py';p.write_text(src)
 assert PyTorchProcessor(ProcessingConfig(save_graph=False,force_rerun=True,safe_mode=False)).process_model_file(str(p),str(tmp_path/'graph'))
 pg=yaml.safe_load((tmp_path/'graph/pytorch_graph.yaml').read_text());assert pg['returned_outputs'][0]['bound_node'] is None
 assert PyTorchToEinsum().convert(str(tmp_path/'graph/pytorch_graph.yaml'),str(tmp_path/'einsum')) is not None
 eg=yaml.safe_load((tmp_path/'einsum/einsum_graph.yaml').read_text());a=EinsumGraphAnalyzer().analyze_graph(tmp_path/'einsum/einsum_graph.yaml',tmp_path/'analysis',copy_graph=False);perf=EinsumGraphPerfModel().predict(tmp_path/'analysis/analysis.yaml',tmp_path/'perf',arch_config='B200')
 rec={'case':kind,'returned':pg['returned_outputs'],'model_outputs':eg['model_outputs'],'footprint':eg['returned_footprint'],'total':a['total'],'qualification':perf['workload']['prediction_qualification']}
 (OUT/(kind+'.json')).write_text(json.dumps(rec,indent=2))
 assert a['total']['external_output_bytes']==expected
 assert a['total']['status']=='incomplete'
 assert rec['qualification'].startswith('unqualified')
 assert len(eg['returned_footprint'])==1
 assert next(iter(eg['returned_footprint'].values()))['union_bytes']==expected

