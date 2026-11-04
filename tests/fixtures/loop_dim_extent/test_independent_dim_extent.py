import os,pathlib,json
import numpy as np,pytest,yaml
from solar.einsum_lib import clear_dims,stop_trace,get_trace,build_metadata
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_lib.verify import execute_translated
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
R=pathlib.Path(__file__).parent;ROLE=os.environ.get('REVIEW_ROLE','head')
OUT=R/'runs'/('direct-'+ROLE);OUT.mkdir(parents=True,exist_ok=True)
H='from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n'
def traced(name,code,inputs):
 d=OUT/name;d.mkdir(exist_ok=True);dsl=d/'dsl';dsl.mkdir(exist_ok=True)
 (dsl/'probe_einsum_lib.py').write_text(code)
 clear_dims();stop_trace()
 out,_=execute_translated(code,{},all_inputs=inputs,fast=False,trace=True,shape_only=False)
 md=build_metadata(get_trace());write_metadata_yaml(md,str(dsl/'probe_einsum_lib.yaml'))
 rec={'outputs':{k:np.asarray(v).tolist() for k,v in out.items()},'metadata':md}
 try:
  p=translate_with_einsum(dsl,d/'einsum');raw=yaml.safe_load(p.read_text())
  a=EinsumGraphAnalyzer().analyze_graph(str(p),str(d/'analysis'),precision='fp32',copy_graph=False)
  rec.update(graph=raw,analysis=a)
 except Exception as exc:rec.update(error=type(exc).__name__,message=str(exc))
 (d/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
 return rec
def layers(r,fn='k'):return [v for k,v in r['graph']['layers'].items() if k.startswith(fn+'#')]
def assert_counts(r,counts):
 assert 'error' not in r,r
 ls=layers(r);assert [v.get('loop_iterations') for v in ls]==counts,r
 assert all(v.get('loop_iterations_complete') is True for v in ls),r

REGISTRY_BINDING=H+'''@kernel
def k(X:Tensor("T"), Z:Tensor("U")) -> Tensor("O"):
    Y=Tensor(O=1)
    for t in Dim("T"):
        Y[0]+=X[0]
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"N")
    z=Tensor.from_numpy(inputs["z"],"T")
    w=Tensor.from_numpy(inputs["w"],"T")
    a=k(x,z)
    b=k(x,w)
    return {"a":a.data,"b":b.data}
'''
def test_annotations_are_not_runtime_dim_bindings():
 r=traced('annotation_binding',REGISTRY_BINDING,{'x':np.ones(8),'z':np.zeros(2),'w':np.zeros(3)})
 assert r['outputs']=={'a':[2.0],'b':[3.0]},r
 assert_counts(r,[2,3])

@pytest.mark.parametrize('annotated',[True,False])
def test_two_dims_and_partial_wider_access(annotated):
 sig='X:Tensor("N,H"), I:Tensor("I")' if annotated else 'X,I'
 ret=' -> Tensor("I,H")' if annotated else ''
 code=H+f'@kernel\ndef k({sig}){ret}:\n'+'''    Y=Tensor(I=I.shape[0],H=X.shape[1])
    for i in Dim("I"):
        for h in Dim("H"):
            Y[i,h]=X[i,h]*2.0
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"N,H")
    i=Tensor.from_numpy(inputs["i"],"I")
    return {"output":k(x,i).data}
'''
 r=traced('partial_'+str(annotated),code,{'x':np.ones((16,4)),'i':np.zeros(2)})
 assert r['outputs']['output']==[[2.0]*4]*2
 assert_counts(r,[8])

def test_shared_kernel_same_annotations_different_shapes():
 code=H+'''@kernel
def k(X:Tensor("T")) -> Tensor("T"):
    Y=Tensor(T=X.shape[0])
    for t in Dim("T"):
        Y[t]=X[t]*2
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"T")
    z=Tensor.from_numpy(inputs["z"],"T")
    return {"a":k(x).data,"b":k(z).data}
'''
 assert_counts(traced('shared',code,{'x':np.ones(2),'z':np.ones(5)}),[2,5])

def test_local_dim_alias_declines_or_counts_exactly():
 code=H+'''@kernel
def k(X:Tensor("T")) -> Tensor("T"):
    Y=Tensor(T=X.shape[0])
    axis=Dim("T")
    for t in axis:
        Y[t]=X[t]*2
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"T")
    return {"output":k(x).data}
'''
 r=traced('local_alias',code,{'x':np.ones(3)})
 if 'error' in r:return
 ls=layers(r)
 assert all((x.get('loop_iterations_complete') is False) or x.get('loop_iterations')==3 for x in ls),r
 if ls[0].get('loop_iterations_complete') is False:
  assert next(v for k,v in r['analysis']['layers'].items() if k.startswith('k#'))['incomplete_reason']=='loop_metadata_incomplete'

def test_registry_only_dim_with_local_tensor_registration():
 code=H+'''@kernel
def k(X:Tensor("N")) -> Tensor("O"):
    Tmp=Tensor(R=3)
    Y=Tensor(O=1)
    for r in Dim("R"):
        Y[0]+=X[r]
    return Y
def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"N")
    return {"output":k(x).data}
'''
 r=traced('registry_only',code,{'x':np.ones(8)})
 assert r['outputs']['output']==[3.0]
 assert_counts(r,[3])

def test_repository_window_crop_is_partial_traversal():
 source=pathlib.Path(os.environ['REVIEW_SOURCE'])/'tests/release_portable/fixtures/window_padding_crop_einsum_lib.py'
 r=traced('repository_crop',source.read_text(),{'x':np.ones((2,4,5,2)),'wt':np.ones((3,2)),'p':np.ones((8,9,2))})
 assert 'error' not in r,r
 crop=next(v for k,v in r['graph']['layers'].items() if k.startswith('window_unpartition_crop#'))
 assert crop['loop_iterations']==80 and crop['loop_iterations_complete'] is True,r
 assert np.asarray(r['outputs']['crop']).shape==(2,4,5,2)
 np.testing.assert_array_equal(r['outputs']['crop'],np.full((2,4,5,2),2.0))

