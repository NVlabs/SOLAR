import ast, collections, json, os, pathlib, sys
import numpy as np
import pytest
import yaml
from solar.einsum_lib import clear_dims, stop_trace, get_trace, build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

ROOT=pathlib.Path(__file__).parent
H='from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n'

def observe(name, body, run, inputs, helper=''):
    code=H+helper+'@kernel\ndef k(X:Tensor("T"), Z:Tensor("U")) -> Tensor("O"):\n'+body+'\ndef run_model(inputs):\n    clear_dims()\n'+run
    folder=ROOT/'runs'/('order-'+os.getenv('REVIEW_ROLE','head'))/name
    dsl=folder/'dsl';dsl.mkdir(parents=True,exist_ok=True)
    (dsl/'probe_einsum_lib.py').write_text(code)
    fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='k')
    weighted_lines=set()
    for loop in ast.walk(fn):
        if isinstance(loop,ast.For):
            for stmt in ast.walk(loop):
                if isinstance(stmt,(ast.Assign,ast.AugAssign,ast.AnnAssign)): weighted_lines.add(stmt.lineno)
    calls=[]
    def tracer(frame,event,arg):
        if frame.f_code.co_name=='k':
            if event=='call': calls.append(collections.Counter())
            if event=='line' and frame.f_lineno in weighted_lines: calls[-1][frame.f_lineno]+=1
        return tracer
    clear_dims();stop_trace();sys.settrace(tracer)
    try: out,_=execute_translated(code,{},all_inputs=inputs,fast=False,trace=True,shape_only=False)
    finally:sys.settrace(None)
    md=build_metadata(get_trace());write_metadata_yaml(md,str(dsl/'probe_einsum_lib.yaml'))
    path=translate_with_einsum(dsl,folder/'einsum');graph=yaml.safe_load(path.read_text())
    analysis=EinsumGraphAnalyzer().analyze_graph(str(path),str(folder/'analysis'),precision='fp32',copy_graph=False)
    record={'outputs':{k:np.asarray(v).tolist() for k,v in out.items()},'runtime_statement_counts':[sum(x.values()) for x in calls], 'runtime_line_counts':calls,'weighted_lines':sorted(weighted_lines),'graph':graph,'analysis':analysis}
    (folder/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

def safe(r):
    ls=[v for k,v in r['graph']['layers'].items() if k.startswith('k#')]
    assert len(ls)==len(r['runtime_statement_counts']),r
    for layer,count in zip(ls,r['runtime_statement_counts']):
        assert layer.get('loop_iterations_complete') is False or layer.get('loop_iterations')==count,r

DEFAULT='    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"U")\n    return {"out":k(x,z).data}\n'
INPUTS={'x':np.ones(2),'z':np.ones(1)}

def test_last_wins_and_swapped_calls():
    body='    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y[0]+=X[0]\n    return Y\n'
    run='    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"T")\n    return {"a":k(x,z).data,"b":k(z,x).data}\n'
    r=observe('last_wins_swapped',body,run,{'x':np.ones(2),'z':np.ones(5)})
    assert r['outputs']=={'a':[5.0],'b':[2.0]};safe(r)

def test_keyword_call_order():
    body='    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y[0]+=X[0]\n    return Y\n'
    run='    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"T")\n    return {"out":k(Z=z,X=x).data}\n'
    r=observe('keyword_order',body,run,{'x':np.ones(2),'z':np.ones(5)})
    assert r['outputs']=={'out':[2.0]};safe(r)

def test_allocation_after_loop():
    r=observe('after_loop','    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Y[0]+=X[0]\n    Tmp=Tensor(T=5)\n    return Y\n',DEFAULT,INPUTS)
    assert r['outputs']=={'out':[2.0]};safe(r)

def test_allocation_inside_outer_loop_before_inner():
    r=observe('inside_loop','    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Tmp=Tensor(U=3)\n        for u in Dim("U"):\n            Y[0]+=X[0]\n    return Y\n',DEFAULT,INPUTS)
    assert r['outputs']=={'out':[6.0]};safe(r)

def test_nested_stable_registry():
    r=observe('nested_stable','    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"):\n            Y[0]+=X[0]\n    return Y\n',DEFAULT,{'x':np.ones(2),'z':np.ones(3)})
    assert r['outputs']=={'out':[6.0]};safe(r)

def test_conditional_allocation_not_executed():
    r=observe('conditional_not_taken','    Y=Tensor(O=1)\n    if X[0]<0:\n        Tmp=Tensor(T=5)\n    for t in Dim("T"):\n        Y[0]+=X[0]\n    return Y\n',DEFAULT,INPUTS)
    assert r['outputs']=={'out':[2.0]};safe(r)

def test_returned_dynamic_allocation_control():
    r=observe('returned_dynamic','    Y=Tensor(R=choose())\n    for r in Dim("R"):\n        Y[r]=X[0]\n    return Y\n',DEFAULT,INPUTS,'def choose():\n    return 3\n')
    assert r['outputs']=={'out':[1.0]*3};safe(r)

def test_dim_names_do_not_identify_returned_allocation():
    r=observe('wrong_return_identity','    Y=Tensor(R=3)\n    Tmp=Tensor(R=choose())\n    for r in Dim("R"):\n        Y[0]+=X[0]\n    return Y\n',DEFAULT,INPUTS,'def choose():\n    return 5\n')
    assert r['outputs']=={'out':[5.0,0.0,0.0]};safe(r)

def test_unresolved_nonreturned_allocation_declines():
    r=observe('unresolved_local','    Y=Tensor(O=1)\n    Tmp=Tensor(R=choose())\n    for r in Dim("R"):\n        Y[0]+=X[0]\n    return Y\n',DEFAULT,INPUTS,'def choose():\n    return 3\n')
    assert r['outputs']=={'out':[3.0]};safe(r)
    layer=r['graph']['layers']['k#0'];assert layer.get('loop_iterations_complete') is False
    assert 'loop_dim_extent_unresolved' in layer and 'loop_iterations' not in layer
    assert r['analysis']['layers']['k#0']['incomplete_reason']=='loop_metadata_incomplete'
