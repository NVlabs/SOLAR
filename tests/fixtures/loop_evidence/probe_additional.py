import json,pathlib,sys,traceback,ast
import numpy as np,yaml
sys.path.insert(0,str(pathlib.Path.cwd()/'tests'))
from test_sparse_cost import DENSE_DSL,_traced_metadata,_analyze
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.perf import EinsumGraphPerfModel
root=pathlib.Path(sys.argv[1]);label=sys.argv[2];out=root/'runs/additional'/label;out.mkdir(parents=True)
ins={'x':np.arange(1,9,dtype=np.float32).reshape(2,4),'w':np.arange(1,13,dtype=np.float32).reshape(3,4)}
inner='            for h in Dim("H"):\n                Y[n, f] += X[n, h] * W[f, h]'
nested=DENSE_DSL.replace(inner,'            def accumulate():\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]\n            accumulate()')
dead=DENSE_DSL.replace(inner,'            if False:\n                for h in Dim("H"):\n                    Y[n, f] += 999 * X[n, h] * W[f, h]\n            else:\n                for h in Dim("H"):\n                    Y[n, f] += X[n, h] * W[f, h]')
norm='''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
@kernel
def second_moment(X: Tensor("N,H",dtype="fp32")) -> Tensor("N,H"):
    Y=Tensor(N=X.shape[0],H=X.shape[1],dtype="fp32")
    for n in Dim("N",parallel=True):
        for h in Dim("H"):
            acc=0.0
            for j in Dim("H"):
                acc += X[n,j]*X[n,j]
            Y[n,h]=acc
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"N,H",dtype="fp32")
    return {"output":second_moment(x).data}
'''
cases=[('nested_function',nested,ins,ins['x']@ins['w'].T,'incomplete',24),('dead_if_else',dead,ins,ins['x']@ins['w'].T,'exact_or_incomplete',24),('recomputed_moment',norm,{'x':ins['x']},np.repeat((ins['x']**2).sum(axis=1)[:,None],4,axis=1),'scalar_classification',32)]
# Bind author's two-chain source as inert AST text, but supply an independent value/count oracle.
tree=ast.parse((root/'head/solar/tests/test_loop_basis_completeness.py').read_text())
two=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TWO_CHAIN_GATHER_DSL' for t in n.targets))
rng=np.random.default_rng(773);T,K,I,H,E=4,2,3,5,6
inp={'ggp':rng.normal(size=(T,K,I)).astype('float32'),'guo':rng.normal(size=(T,K,I)).astype('float32'),'se':rng.integers(E,size=(T,K),dtype=np.int64),'wg':rng.normal(size=(E,I,H)).astype('float32'),'wu':rng.normal(size=(E,I,H)).astype('float32')}
ref=np.zeros((T,H));count=0
for t in range(T):
 for k in range(K):
  for i in range(I):
   for h in range(H):
    ref[t,h]+=inp['ggp'][t,k,i]*inp['wg'][inp['se'][t,k],i,h]+inp['guo'][t,k,i]*inp['wu'][inp['se'][t,k],i,h];count+=2
cases.append(('two_chain_gather',two,inp,ref,'exact',count))
records=[]
for name,code,inputs,expected,rule,work in cases:
 d=out/name;d.mkdir();rec={'case':name,'source_tree':label,'hand_count':work,'rule':rule}
 try:
  values,_=execute_translated(code,{},all_inputs=inputs,trace=False,fast=False)
  np.testing.assert_allclose(values['output'],expected,rtol=1e-5,atol=1e-5)
  md=_traced_metadata(code,inputs,profile=False);a=_analyze(code,md,d)
  g=yaml.safe_load(next((d/'einsum').glob('*.yaml')).read_text());lid=next(k for k in a['layers'] if k!='start')
  layer=a['layers'][lid];raw=g['layers'][lid]
  p=EinsumGraphPerfModel().predict(d/'analysis/analysis.yaml',d/'perf',arch_config='H100_PCIe')
  marked=all(x.get('status')=='incomplete' for x in [layer,a['total'],p['workload']])
  satisfied={'incomplete':marked,'exact_or_incomplete':marked or layer['macs']==work,'scalar_classification':not raw.get('is_real_einsum'),'exact':layer['macs']==work and not marked}[rule]
  rec.update(numerical_output_correct=True,parser=extract_kernel_info(next((d/'dsl').glob('*.py'))),raw=raw,layer=layer,total=a['total'],perf=p,all_three_incomplete=marked,satisfied=satisfied)
 except Exception as e:rec.update(error=repr(e),traceback=traceback.format_exc(),satisfied=False)
 records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k not in ['parser','raw','layer','total','perf']}),flush=True)
 if 'layer' in rec:print({k:rec['layer'].get(k) for k in ['macs','other_ops','status','is_real_einsum']},'raw', {k:rec['raw'].get(k) for k in ['loop_iterations','loop_iterations_complete','loop_iterations_omitted','is_real_einsum']},flush=True)
(out/'RESULT.json').write_text(json.dumps(records,indent=2)+'\n')
sys.exit(0 if all(x['satisfied'] for x in records) else 1)
