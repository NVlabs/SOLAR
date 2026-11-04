import ast,collections,dis,json,os,pathlib,sys
import numpy as np
import pytest,yaml
from solar.einsum_lib import Tensor,clear_dims,stop_trace,get_trace,build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
import solar.einsum_dsl.cli_einsum as cli
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
R=pathlib.Path(__file__).parent
H='from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\n'
K='@kernel\ndef k(X:Tensor("T"),Z:Tensor("U"),n:int=3)->Tensor("O"):\n'
RUN='\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"U")\n    return {"out":k(x,z).data}\n'
def prog(body,sig=K,call='k(x,z)',helper=''):return H+helper+sig+body+RUN.replace('k(x,z)',call)
COMMON='    Y=Tensor(O=1)\n    for t in Dim("T",size=n):\n        Y[0]+=X[0]\n    return Y\n'
SHAPE=H+'@kernel\ndef k(X:Tensor("T,H"),Z:Tensor("B"))->Tensor("O"):\n    nb=X.shape[1]\n    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for b in Dim("B",size=nb):\n            Y[0]+=X[0,0]\n    for b in Dim("B"):\n        Y[0]+=X[0,0]\n    return Y\n\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T,H")\n    z=Tensor.from_numpy(inputs["z"],"B")\n    return {"out":k(x,z).data}\n'
NESTED=prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"): Tmp=Tensor(V=3)\n        for v in Dim("V"):\n            Y[0]+=X[0]\n    return Y\n')
SEMICOLON=prog('    Y=Tensor(O=1)\n    for t in Dim("T"): A=Tensor(U=3); B=Tensor(U=5)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n')
ONCE=prog('    Y=Tensor(O=1)\n    for t in Dim("T",size=1):\n        Y=Tensor(R=choose())\n        for r in Dim("R"):\n            Y[0]+=X[0]\n    return Y\n',helper='def choose():\n    return 3\n')
CASES={
 'header_shape_fresh':{'code':SHAPE,'shape':True},
 'header_shape_legacy':{'code':SHAPE,'shape':True,'strip':['input_order']},
 'header_helper':{'code':prog('    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U",size=choose()):\n            Y[0]+=X[0]\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n',helper='def choose():\n    return 3\n')},
 'semicolon_allocations':{'code':SEMICOLON},
 'semicolon_allocations_legacy_columns':{'code':SEMICOLON,'legacy_columns':True},
 'nested_header_allocation':{'code':NESTED},
 'nested_header_allocation_legacy_columns':{'code':NESTED,'legacy_columns':True},
 'comment_shifted':{'code':NESTED.replace('@kernel\n','# a comment before the decorator\n@kernel\n').replace('        for u','        # shift the nested body line\n        for u')},
 'tab_indentation':{'code':NESTED.replace('            ','\t\t\t').replace('        ','\t\t').replace('    ','\t')},
 'returned_once':{'code':ONCE},
 'keyword_only_default':{'code':prog(COMMON,sig=K.replace(',n:int=3',',*,n:int=3'))},
 'keyword_only_supplied':{'code':prog(COMMON,sig=K.replace(',n:int=3',',*,n:int=3'),call='k(x,z,n=4)')},
 'bool_default_used':{'code':prog(COMMON,sig=K.replace('n:int=3','n:bool=True'))},
 'varargs_scalar':{'code':prog(COMMON.replace('size=n','size=sizes[0]'),sig=K.replace('n:int=3','*sizes'),call='k(x,z,3)')},
 'varkw_scalar':{'code':prog(COMMON.replace('size=n','size=sizes["n"]'),sig=K.replace('n:int=3','**sizes'),call='k(x,z,n=3)')},
}

def observe(name):
 case=CASES[name];code=case['code'];folder=R/'runs'/('round5-'+os.environ['REVIEW_ROLE'])/name;dsl=folder/'dsl';dsl.mkdir(parents=True,exist_ok=True);(dsl/'probe_einsum_lib.py').write_text(code)
 fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='k');spans=[(n.lineno,n.end_lineno) for n in ast.walk(fn) if isinstance(n,ast.For)]
 counts=collections.Counter();events=[]
 def trace(frame,event,arg):
  if frame.f_code.co_name=='k':
   frame.f_trace_opcodes=True
   if event=='opcode' and dis.opname[frame.f_code.co_code[frame.f_lasti]]=='STORE_SUBSCR':
    counts['in_loop_stores']+=1;events.append({'event':'STORE_SUBSCR','line':frame.f_lineno,'offset':frame.f_lasti})
  if event=='call' and frame.f_code.co_name=='__init__' and isinstance(frame.f_locals.get('self'),Tensor):
   parent=frame.f_back
   if parent and parent.f_code.co_name=='k' and any(lo<=parent.f_lineno<=hi for lo,hi in spans):
    counts['in_loop_tensor_allocations']+=1;events.append({'event':'Tensor.__init__','line':parent.f_lineno})
  return trace
 clear_dims();stop_trace();sys.settrace(trace)
 try:out,_=execute_translated(code,{},all_inputs={'x':np.ones((2,3)) if case.get('shape') else np.ones(2),'z':np.ones(1)},fast=False,trace=True,shape_only=False)
 finally:sys.settrace(None)
 md=build_metadata(get_trace());original=json.loads(json.dumps(md))
 for field in case.get('strip',[]):md['graph']['k#0'].pop(field,None)
 write_metadata_yaml(md,str(dsl/'probe_einsum_lib.yaml'))
 rec={'case':name,'role':os.environ['REVIEW_ROLE'],'runtime_count':sum(counts.values()),'runtime_event_counts':dict(counts),'runtime_events':events,'outputs':{k:np.asarray(v).tolist() for k,v in out.items()},'metadata':md,'fresh_metadata':original,'case_options':{k:v for k,v in case.items() if k!='code'}}
 saved=cli.extract_kernel_info
 def without_columns(path):
  result=saved(path)
  def walk(x):
   if isinstance(x,dict):
    x.pop('col',None);x.pop('end_col',None)
    for y in x.values():walk(y)
   elif isinstance(x,list):
    for y in x:walk(y)
  walk(result);return result
 try:
  if case.get('legacy_columns'):cli.extract_kernel_info=without_columns
  rec['parsed_kernel_info']=cli.extract_kernel_info(dsl/'probe_einsum_lib.py')
  p=cli.translate_with_einsum(dsl,folder/'einsum');rec['graph']=yaml.safe_load(p.read_text());rec['analysis']=EinsumGraphAnalyzer().analyze_graph(str(p),str(folder/'analysis'),precision='fp32',copy_graph=False)
 except Exception as e:rec['conversion_error']={'type':type(e).__name__,'code':getattr(e,'code',None),'message':str(e)}
 finally:cli.extract_kernel_info=saved
 (folder/'record.json').write_text(json.dumps(rec,indent=2)+'\n');return rec

@pytest.mark.parametrize('name',CASES)
def test_count_matches_runtime_or_typed_refusal(name):
 r=observe(name)
 assert 'conversion_error' not in r,r
 layer=r['graph']['layers']['k#0']
 if layer.get('loop_iterations_complete') is False:
  assert 'loop_iterations' not in layer and r['analysis']['layers']['k#0']['status']=='incomplete',r
  assert any(k.startswith('loop_') and ('unresolved' in k or 'mismatch' in k or 'unknown' in k) for k in layer),r
 else:
  assert layer.get('loop_iterations')==r['runtime_count'],r
