import pathlib,json,sys,copy,traceback
import numpy as np,yaml
sys.path.insert(0,str(pathlib.Path.cwd()/'tests'))
from test_sparse_cost import DENSE_DSL,_traced_metadata,_analyze
from solar.einsum_lib.verify import execute_translated
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer,dsl_loop_count_evidence
from solar.perf import EinsumGraphPerfModel
R=pathlib.Path(__file__).parent
out=R/'extended';out.mkdir(exist_ok=True)
inner='            for h in Dim("H"):\n                Y[n, f] += X[n, h] * W[f, h]'
ins={'x':np.arange(1,9,dtype=np.float32).reshape(2,4),'w':np.arange(1,13,dtype=np.float32).reshape(3,4)}
bodies={
'if_true_both':'            if True:\n                for h in Dim("H"):\n                    Y[n,f] += X[n,h]*W[f,h]\n            else:\n                for h in Dim("H"):\n                    Y[n,f] += 999*X[n,h]*W[f,h]',
'if_elif_else':'            if False:\n                for h in Dim("H"):\n                    Y[n,f] += 999*X[n,h]*W[f,h]\n            elif X[n,0]>0:\n                for h in Dim("H"):\n                    Y[n,f] += X[n,h]*W[f,h]\n            else:\n                for h in Dim("H"):\n                    Y[n,f] += 999*X[n,h]*W[f,h]',
'else_only':'            if X[n,0]<0:\n                Y[n,f]=999\n            else:\n                for h in Dim("H"):\n                    Y[n,f] += X[n,h]*W[f,h]',
'data_dependent_reduction':'            if X[n,0]>0:\n                for h in Dim("H"):\n                    Y[n,f] += X[n,h]*W[f,h]',
}
records={}
for name,body in bodies.items():
 code=DENSE_DSL.replace(inner,body);assert code!=DENSE_DSL
 d=out/name;d.mkdir(exist_ok=True)
 values,_=execute_translated(code,{},all_inputs=ins,trace=False,fast=False)
 np.testing.assert_allclose(values['output'],ins['x']@ins['w'].T,rtol=1e-5,atol=1e-5)
 a=_analyze(code,_traced_metadata(code,ins,profile=False),d)
 g=yaml.safe_load(next((d/'einsum').glob('*.yaml')).read_text())
 lid=next(k for k in a['layers'] if k!='start');l=a['layers'][lid];raw=g['layers'][lid]
 perf=EinsumGraphPerfModel().predict(d/'analysis/analysis.yaml',d/'perf',arch_config='H100_PCIe')
 marked=all(x.get('status')=='incomplete' for x in [l,a['total'],perf['workload']])
 records[name]={'numerical_values_correct':True,'actual_macs':24,'graph':{k:raw.get(k) for k in ('loop_iterations','loop_statements','loop_outer_statements','loop_iterations_complete','loop_iterations_omitted','equation_faithful')},'billed':{k:l.get(k) for k in ('macs','other_ops','status','incomplete_reason')},'layer_total_perf_incomplete':marked,'passed':marked and raw.get('loop_iterations_complete') is False}
# Unequal no-loop branches: executed 8 assignments on positive inputs; 24 is a conservative maximum.
code='''from solar.einsum_lib import Tensor,Dim,kernel,clear_dims
@kernel
def nonloop(X:Tensor("N,H"))->Tensor("N,H"):
 Y=Tensor(N=X.shape[0],H=X.shape[1])
 for n in Dim("N"):
  for h in Dim("H"):
   if X[n,h]>0:
    Y[n,h]=X[n,h]+1
   else:
    a=X[n,h]+1
    b=a+1
    Y[n,h]=b+1
 return Y
def run_model(inputs):
 clear_dims()
 x=Tensor.from_numpy(inputs["x"],"N,H")
 return {"output":nonloop(x).data}
'''
d=out/'no_loop_branch';d.mkdir(exist_ok=True);vals,_=execute_translated(code,{},all_inputs={'x':ins['x']},trace=False,fast=False)
np.testing.assert_allclose(vals['output'],ins['x']+1)
a=_analyze(code,_traced_metadata(code,{'x':ins['x']},profile=False),d);g=yaml.safe_load(next((d/'einsum').glob('*.yaml')).read_text());l=a['layers']['nonloop#0'];raw=g['layers']['nonloop#0']
records['no_loop_branch']={'executed_assignments':8,'maximum_branch_assignments':24,'other_ops':l['other_ops'],'complete':raw.get('loop_iterations_complete'),'status':l.get('status'),'sound_assignment_upper_bound':l['other_ops']==24,'hardware_instruction_count_claim':False}
# Shared-gate schema cases and their actual analyzer/perf consequences.
d=out/'gate';d.mkdir(exist_ok=True)
base=_analyze(DENSE_DSL,_traced_metadata(DENSE_DSL,ins,profile=False),d/'base')
graph=yaml.safe_load(next((d/'base/einsum').glob('*.yaml')).read_text());lid=next(iter(base['layers']))
gate=[]
for name,value in [('int',24),('fractional_float',7.9),('integral_float',24.0),('numeric_string','24'),('bool',True),('zero',0),('negative',-1),('missing',None),('infinity',float('inf'))]:
 node=copy.deepcopy(graph['layers'][lid]);node.update(loop_iterations=value,loop_iterations_complete=True);node.pop('loop_iterations_omitted',None)
 rec={'case':name,'raw_repr':repr(value),'raw_type':type(value).__name__}
 try:
  ev=dsl_loop_count_evidence(node);rec['gate']=ev._asdict()
  gg=copy.deepcopy(graph);gg['layers'][lid]=node;dd=d/name;dd.mkdir(exist_ok=True);p=dd/'graph.yaml';p.write_text(yaml.safe_dump(gg))
  aa=EinsumGraphAnalyzer().analyze_graph(str(p),str(dd/'analysis'));ll=aa['layers'][lid]
  pp=EinsumGraphPerfModel().predict(dd/'analysis/analysis.yaml',dd/'perf',arch_config='H100_PCIe')
  rec['billed']={k:ll.get(k) for k in ('macs','other_ops','status','incomplete_reason')}
  rec['perf_status']=pp['workload'].get('status')
  rec['positive_integer_contract_pass']=(ev.trusted==(type(value) is int and value>0))
 except Exception as e:rec.update(exception=repr(e),traceback=traceback.format_exc(),positive_integer_contract_pass=False)
 gate.append(rec)
records['gate']=gate
(R/'evidence/extended-results.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2),flush=True)

