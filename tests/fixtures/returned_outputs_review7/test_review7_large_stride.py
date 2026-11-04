import json
import torch,yaml
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel

def test_large_bound_stride_does_not_cover_hole(tmp_path):
 # Real PyTorch capture, no injected graph/storage metadata. The bound stride-2
 # output crosses the helper's enumeration cap. Its bounding interval includes
 # byte holes that have never been charged as bound output elements.
 n=(1<<20)+1
 src=f'import torch\nclass Model(torch.nn.Module):\n    def forward(self,x):\n        y=x*2\n        return y[::2], y[1:2].as_subclass(torch.Tensor)\ndef get_inputs():return [torch.arange({2*n},dtype=torch.float32)]\ndef get_init_inputs():return []\n'
 f=tmp_path/'model.py';f.write_text(src)
 ns={};exec(src,ns);a,b=ns['Model']()(*ns['get_inputs']())
 assert a.untyped_storage()._cdata==b.untyped_storage()._cdata
 assert a.stride()==(2,) and a.storage_offset()==0 and b.storage_offset()==1
 expected=(a.numel()+b.numel())*4
 assert expected==4194312
 p=PyTorchProcessor(ProcessingConfig(save_graph=False,force_rerun=True,safe_mode=False))
 assert p.process_model_file(str(f),str(tmp_path/'graph'))
 pg=yaml.safe_load((tmp_path/'graph/pytorch_graph.yaml').read_text())
 assert PyTorchToEinsum().convert(str(tmp_path/'graph/pytorch_graph.yaml'),str(tmp_path/'einsum')) is not None
 egpath=tmp_path/'einsum/einsum_graph_renamed.yaml'
 eg=yaml.safe_load(egpath.read_text())
 result=EinsumGraphAnalyzer().analyze_graph(str(egpath),str(tmp_path/'analysis'),precision='fp32')
 perf=EinsumGraphPerfModel().predict(tmp_path/'analysis/analysis.yaml',tmp_path/'perf',arch_config='B200')
 rec={'expected_output_bytes':expected,'observed_output_bytes':result['total']['external_output_bytes'],'returned':pg['returned_outputs'],'footprint':eg.get('returned_footprint'),'model_outputs':eg['model_outputs'],'total':result['total'],'qualification':perf['workload'].get('prediction_qualification')}
 (tmp_path/'RESULT.json').write_text(json.dumps(rec,indent=2))
 assert result['total']['external_output_bytes']==expected,rec
 assert result['total'].get('status')=='incomplete',rec
 assert rec['qualification'].startswith('unqualified'),rec

