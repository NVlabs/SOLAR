import json,pathlib,sys,numpy as np,yaml
sys.path.insert(0,str(pathlib.Path.cwd()/'tests'))
from test_sparse_cost import _traced_metadata,_analyze
from solar.einsum_lib.verify import execute_translated
from solar.perf import EinsumGraphPerfModel
root=pathlib.Path(sys.argv[1]);label=sys.argv[2];out=root/'runs/recompute2'/label;out.mkdir(parents=True)
code='''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
@kernel
def repeated_inner_product(X: Tensor("N,H",dtype="fp32"), W: Tensor("N,H",dtype="fp32")) -> Tensor("N,H"):
    Y=Tensor(N=X.shape[0],H=X.shape[1],dtype="fp32")
    for n in Dim("N"):
        for h in Dim("H"):
            acc=0.0
            for j in Dim("H"):
                acc += X[n,j]*W[n,j]
            Y[n,h]=acc
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"N,H",dtype="fp32")
    w=Tensor.from_numpy(inputs["w"],"N,H",dtype="fp32")
    return {"output":repeated_inner_product(x,w).data}
'''
inputs={'x':np.arange(8,dtype=np.float32).reshape(2,4),'w':np.arange(8,16,dtype=np.float32).reshape(2,4)}
expected=np.repeat((inputs['x']*inputs['w']).sum(axis=1)[:,None],4,axis=1)
values,_=execute_translated(code,{},all_inputs=inputs,trace=False,fast=False);np.testing.assert_allclose(values['output'],expected,rtol=1e-5,atol=1e-5)
md=_traced_metadata(code,inputs,profile=False);a=_analyze(code,md,out);g=yaml.safe_load(next((out/'einsum').glob('*.yaml')).read_text())
p=EinsumGraphPerfModel().predict(out/'analysis/analysis.yaml',out/'perf',arch_config='H100_PCIe')
r={'label':label,'numerical_output_correct':True,'hand_iterations':32,'union':8,'graph':g,'analysis':a,'perf':p}
(out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
print(label,{k:{x:v.get(x) for x in ['macs','other_ops','is_real_einsum']} for k,v in a['layers'].items()})

