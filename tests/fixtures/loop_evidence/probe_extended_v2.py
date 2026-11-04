import json
from pathlib import Path
import sys
import traceback
import numpy as np
import yaml

sys.path.insert(0, str(Path.cwd()/'tests'))
from test_sparse_cost import DENSE_DSL, _traced_metadata, _analyze
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.perf.perf_model import EinsumGraphPerfModel

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
inputs = {'x':np.arange(1,9,dtype=np.float32).reshape(2,4),
          'w':np.arange(1,13,dtype=np.float32).reshape(3,4)}
mac = '                Y[n, f] += X[n, h] * W[f, h]'
inner = '            for h in Dim("H"):\n'+mac
cases = {
 'all_dim':(DENSE_DSL,inputs,24,False,1),
 'inner_range':(DENSE_DSL.replace('for h in Dim("H"):', 'for h in range(X.shape[1]):'),inputs,24,True,1),
 'all_range':(DENSE_DSL.replace('for n in Dim("N", parallel=True):','for n in range(X.shape[0]):').replace('for f in Dim("F"):','for f in range(W.shape[0]):').replace('for h in Dim("H"):','for h in range(X.shape[1]):'),inputs,24,True,1),
 'mixed_while':(DENSE_DSL.replace(mac,mac+'\n                repeat = 0\n                while repeat < 1:\n                    Y[n, f] += X[n, h] * W[f, h]\n                    repeat += 1'),inputs,48,True,2),
 'try_dim':(DENSE_DSL.replace(inner,'            try:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            finally:\n                pass'),inputs,24,True,1),
 'with_dim':('from contextlib import nullcontext\n'+DENSE_DSL.replace(inner,'            with nullcontext():\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]'),inputs,24,True,1),
 'for_else_dim':(DENSE_DSL.replace(mac,mac+'\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]'),inputs,48,True,2),
}
dot = '''
from solar.einsum_lib import Tensor, kernel, clear_dims
@kernel
def dot(X: Tensor("H", dtype="fp32"), W: Tensor("H", dtype="fp32")) -> Tensor("O"):
    Y = Tensor(O=1, dtype="fp32")
    for h in range(X.shape[0]):
        Y[0] += X[h] * W[h]
    return Y
def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "H", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "H", dtype="fp32")
    return {"output": dot(x,w).data}
'''
di = {'x':np.arange(1,5,dtype=np.float32),'w':np.arange(2,6,dtype=np.float32)}
cases['single_range']=(dot,di,4,True,1)
cases['single_while']=(dot.replace('    for h in range(X.shape[0]):','    h = 0\n    while h < X.shape[0]:').replace('        Y[0] += X[h] * W[h]','        Y[0] += X[h] * W[h]\n        h += 1'),di,4,True,1)
records=[]
for name,(code,ins,work,require_incomplete,factor) in cases.items():
    dest=out/name;dest.mkdir()
    record={'case':name,'independent_executed_macs':work,'requires_incomplete_for_unrepresented_structure':require_incomplete}
    try:
        values,_=execute_translated(code,{},all_inputs=ins,trace=False,fast=False)
        expected=ins['x']@ins['w'].T*factor
        np.testing.assert_allclose(values['output'],expected)
        md=_traced_metadata(code,ins,profile=False)
        analysis=_analyze(code,md,dest)
        graph=yaml.safe_load(next((dest/'einsum').glob('*.yaml')).read_text())
        lid=next(k for k in graph['layers'] if k!='start')
        raw=graph['layers'][lid];layer=analysis['layers'][lid]
        ap=dest/'analysis/analysis.yaml'
        perf=EinsumGraphPerfModel().predict(ap,dest/'perf',arch_config=str(Path.cwd()/'configs/arch/H100_PCIe.yaml'))
        assert perf is not None
        assert perf['workload']['total_macs'] == analysis['total']['macs'], 'Perf must consume analysis.yaml'
        record.update(numerical_output_correct=True,parser=extract_kernel_info(next((dest/'dsl').glob('*.py'))),
                      graph=raw,layer=layer,total=analysis['total'],perf_workload=perf['workload'])
        marked=layer.get('status')=='incomplete' and analysis['total'].get('status')=='incomplete' and perf['workload'].get('status')=='incomplete'
        record['all_three_incomplete']=marked
        record['qualification_check_passed']=marked if require_incomplete else layer['macs']==work and not marked
    except Exception as exc:
        record.update(error=repr(exc),traceback=traceback.format_exc(),qualification_check_passed=False)
    records.append(record)
    (out/'RESULT.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('parser','graph','layer','total','perf_workload')},
                     default=str),flush=True)
    if 'layer' in record:
        print(json.dumps({'case':name,'macs':record['layer']['macs'],'other_ops':record['layer']['other_ops'],
              'loop_iterations':record['graph'].get('loop_iterations'),
              'loop_iterations_complete':record['graph'].get('loop_iterations_complete'),
              'omitted':record['graph'].get('loop_iterations_omitted'),
              'equation':record['graph'].get('einsum_equation')}),flush=True)

failures=[r['case'] for r in records if not r['qualification_check_passed']]
print(json.dumps({'qualification_failures':failures,'passed':len(records)-len(failures),'failed':len(failures)}))
sys.exit(1 if failures else 0)
