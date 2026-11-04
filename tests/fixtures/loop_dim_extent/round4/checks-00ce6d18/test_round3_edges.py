import ast, collections, json, os, pathlib, sys
import numpy as np
import pytest
import yaml
from solar.einsum_lib import clear_dims, stop_trace, get_trace, build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer

ROOT = pathlib.Path(__file__).parent
H = 'from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n'
K = '@kernel\ndef k(X:Tensor("T"), Z:Tensor("U")) -> Tensor("O"):\n'
RUN = '\ndef run_model(inputs):\n    clear_dims()\n    x=Tensor.from_numpy(inputs["x"],"T")\n    z=Tensor.from_numpy(inputs["z"],"U")\n    return {"out":k(x,z).data}\n'
LOOP = '    for t in Dim("T"):\n        Y[0]+=X[0]\n    return Y\n'
RLOOP = '    for r in Dim("R"):\n        Y[0]+=X[0]\n'
CASES = {
    'same_extent_taken': '    Y=Tensor(O=1)\n    if X[0]>0:\n        Tmp=Tensor(T=2)\n'+LOOP,
    'same_extent_untaken': '    Y=Tensor(O=1)\n    if X[0]<0:\n        Tmp=Tensor(T=2)\n'+LOOP,
    'if_else_different': '    Y=Tensor(O=1)\n    if X[0]>0:\n        Tmp=Tensor(T=3)\n    else:\n        Tmp=Tensor(T=5)\n'+LOOP,
    'try_except': '    Y=Tensor(O=1)\n    try:\n        Tmp=Tensor(T=3)\n    except ValueError:\n        Tmp=Tensor(T=5)\n'+LOOP,
    'certain_reset': '    Y=Tensor(O=1)\n    if X[0]>0:\n        Tmp=Tensor(T=5)\n    Certain=Tensor(T=3)\n'+LOOP,
    'early_return_taken': '    Y=Tensor(O=1)\n    if X[0]>0:\n        return Y\n    Tmp=Tensor(T=5)\n'+LOOP,
    'early_return_untaken': '    Y=Tensor(O=1)\n    if X[0]<0:\n        return Y\n    Tmp=Tensor(T=5)\n'+LOOP,
    'return_alias': '    Y=Tensor(R=choose())\n'+RLOOP+'    Alias=Y\n    return Alias\n',
    'return_rebound_tensor': '    Y=Tensor(R=choose())\n    Y=Tensor(R=3)\n'+RLOOP+'    return Y\n',
    'return_rebound_nonallocation': '    Existing=Tensor(R=3)\n    Y=Tensor(R=choose())\n'+RLOOP+'    Y=Existing\n    return Y\n',
    'return_helper': '    Y=Tensor(R=choose())\n'+RLOOP+'    return identity(Y)\n',
    'return_tuple': '    Y=Tensor(R=choose())\n'+RLOOP+'    return Y,Z\n',
    'three_deep': '    Y=Tensor(O=1)\n    for t in Dim("T"):\n        Tmp=Tensor(U=3)\n        for u in Dim("U"):\n            Deep=Tensor(V=4)\n            for v in Dim("V"):\n                Y[0]+=X[0]\n    return Y\n',
    'between_inner_loops': '    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"):\n            Y[0]+=X[0]\n        Tmp=Tensor(U=3)\n        for u in Dim("U"):\n            Y[0]+=X[0]\n    return Y\n',
    'after_inner_loop': '    Y=Tensor(O=1)\n    for t in Dim("T"):\n        for u in Dim("U"):\n            Y[0]+=X[0]\n        Tmp=Tensor(U=3)\n    return Y\n',
    'conditional_inside_outer': '    Y=Tensor(O=1)\n    for t in Dim("T"):\n        if X[0]>0:\n            Tmp=Tensor(U=3)\n        for u in Dim("U"):\n            Y[0]+=X[0]\n    return Y\n',
}

def observe(name, body):
    code = H+'def choose():\n    return 5\ndef identity(x):\n    return x\n'+K+body+RUN
    if name == 'return_tuple':
        code = code.replace('return {"out":k(x,z).data}', 'a,b=k(x,z)\n    return {"out":a.data,"other":b.data}')
    folder = ROOT/'runs'/('edges-'+os.environ['REVIEW_ROLE'])/name
    dsl = folder/'dsl'; dsl.mkdir(parents=True,exist_ok=True)
    (dsl/'probe_einsum_lib.py').write_text(code)
    fn = next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='k')
    weighted = {n.lineno for loop in ast.walk(fn) if isinstance(loop,ast.For)
                for n in ast.walk(loop) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))}
    counts = collections.Counter()
    def tracer(frame,event,arg):
        if frame.f_code.co_name=='k' and event=='line' and frame.f_lineno in weighted:
            counts[frame.f_lineno]+=1
        return tracer
    clear_dims(); stop_trace(); sys.settrace(tracer)
    try:
        out,_=execute_translated(code,{},all_inputs={'x':np.ones(2),'z':np.ones(1)},fast=False,trace=True,shape_only=False)
    finally:
        sys.settrace(None)
    trace=get_trace();md=build_metadata(trace)
    write_metadata_yaml(md,str(dsl/'probe_einsum_lib.yaml'))
    record={'name':name,'runtime_weighted_assignments':sum(counts.values()),'runtime_line_counts':dict(counts),
            'outputs':{k:np.asarray(v).tolist() for k,v in out.items()},'metadata':md}
    try:
        path=translate_with_einsum(dsl,folder/'einsum')
        record['graph']=yaml.safe_load(path.read_text())
        record['analysis']=EinsumGraphAnalyzer().analyze_graph(str(path),str(folder/'analysis'),precision='fp32',copy_graph=False)
    except Exception as e:
        record['conversion_error']={'type':type(e).__name__,'message':str(e)}
    (folder/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

@pytest.mark.parametrize('name',CASES)
def test_runtime_count_or_explicit_refusal(name):
    r=observe(name,CASES[name])
    assert 'conversion_error' not in r,r
    layers={k:v for k,v in r['graph']['layers'].items() if k.startswith('k#')}
    assert len(layers)==1,r
    layer=next(iter(layers.values()))
    if name in ['same_extent_taken','same_extent_untaken','certain_reset','three_deep']:
        assert layer.get('loop_iterations_complete') is True,r
        assert layer.get('loop_iterations')==r['runtime_weighted_assignments'],r
    elif layer.get('loop_iterations_complete') is False:
        assert 'loop_iterations' not in layer,r
        assert r['analysis']['layers']['k#0']['status']=='incomplete',r
    else:
        assert layer.get('loop_iterations')==r['runtime_weighted_assignments'],r
