import ast,collections,json,os,pathlib,sys
import numpy as np
import pytest,yaml
from solar.einsum_lib import clear_dims,stop_trace,get_trace,build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
R=pathlib.Path(__file__).parent
H='from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\n'
K='@kernel\ndef k(X:Tensor("T"),Z:Tensor("U"),n:int=3)->Tensor("O"):\n'
RUN='\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"U")\n    return {"out":k(x,z).data}\n'
def prog(body,call='k(x,z)'):return H+K+body+RUN.replace('k(x,z)',call)
CASES={
 'same_line_allocation':prog('    Y=Tensor(O=1)\n    for t in Dim("T"): Tmp=Tensor(U=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),
 'multiline_allocation_control':prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Tmp=Tensor(U=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),
 'explicit_dim_registers':prog('    Y=Tensor(O=1)\n    for u in Dim("U",size=3):\n        Y[0]+=X[0]\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),
 'explicit_dim_before_loop':prog('    Y=Tensor(O=1)\n    D=Dim("U",size=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),
 'scalar_default_missing':prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n'),
 'scalar_explicit_control':prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n','k(x,z,n=3)'),
 'scalar_zero':prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n','k(x,z,n=0)'),
 'scalar_legacy_missing':prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n','k(x,z,n=3)'),
 'loop_alloc_instances':prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y=Tensor(R=1+2*t)\n        for r in Dim("R"):\n            Y[0]+=X[0]\n    return Y\n'),
}
def observe(name):
 code=CASES[name];folder=R/'runs'/('round4-'+os.environ['REVIEW_ROLE'])/name;dsl=folder/'dsl';dsl.mkdir(parents=True,exist_ok=True)
 (dsl/'probe_einsum_lib.py').write_text(code)
 fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='k')
 weighted={n.lineno for loop in ast.walk(fn) if isinstance(loop,ast.For) for n in ast.walk(loop) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))}
 counts=collections.Counter()
 def trace(frame,event,arg):
  if frame.f_code.co_name=='k' and event=='line' and frame.f_lineno in weighted:counts[frame.f_lineno]+=1
  return trace
 clear_dims();stop_trace();sys.settrace(trace)
 try:out,_=execute_translated(code,{},all_inputs={'x':np.ones(2),'z':np.ones(1)},fast=False,trace=True,shape_only=False)
 finally:sys.settrace(None)
 md=build_metadata(get_trace());write_metadata_yaml(md,str(dsl/'probe_einsum_lib.yaml'))
 if name=='scalar_legacy_missing':
  (folder/'fresh-metadata.json').write_text(json.dumps(md,indent=2)+'\n')
  md['graph']['k#0'].pop('scalar_args',None);write_metadata_yaml(md,str(dsl/'probe_einsum_lib.yaml'))
 # A one-line for's trace event also fires once to terminate iteration; count actual allocation calls instead.
 if name=='same_line_allocation':
  actual=2+3
 else:actual=sum(counts.values())
 rec={'case':name,'role':os.environ['REVIEW_ROLE'],'runtime_line_counts':dict(counts),'runtime_count':actual,'runtime_count_basis':'two Tensor allocations plus three += executions' if name=='same_line_allocation' else 'in-loop assignment line events','outputs':{k:np.asarray(v).tolist() for k,v in out.items()},'metadata':md}
 try:
  p=translate_with_einsum(dsl,folder/'einsum');rec['graph']=yaml.safe_load(p.read_text());rec['analysis']=EinsumGraphAnalyzer().analyze_graph(str(p),str(folder/'analysis'),precision='fp32',copy_graph=False)
 except Exception as e:rec['conversion_error']={'type':type(e).__name__,'code':getattr(e,'code',None),'message':str(e)}
 (folder/'record.json').write_text(json.dumps(rec,indent=2)+'\n');return rec
@pytest.mark.parametrize('name',CASES)
def test_runtime_or_refusal(name):
 r=observe(name)
 if 'conversion_error' in r:
  assert r['conversion_error']['type'] in ['ValueError','LoopDimExtentMismatch'],r
  return
 L=r['graph']['layers']['k#0']
 if L.get('loop_iterations_complete') is False:
  assert 'loop_iterations' not in L and r['analysis']['layers']['k#0']['status']=='incomplete',r
 else:assert L.get('loop_iterations')==r['runtime_count'],r
