"""Independent round8 semantic controls; no production files changed."""
import os,pathlib,json,importlib.util
import numpy as np,pytest
from solar.einsum_lib import clear_dims,stop_trace
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
R=pathlib.Path(__file__).parent
OUT=R/'runs/round8';OUT.mkdir(parents=True,exist_ok=True)
H='from solar.einsum_lib import Tensor, kernel, clear_dims\nimport numpy as np\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)\n    return Y\n'
RUN="\ndef run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')\n    return {'output':k(a).data}\n"
CASES={}
CASES['caller_local_shadow']=H+'DT="bf16"\n'+K+RUN.replace('    clear_dims()','    DT="fp32"\n    clear_dims()')
CASES['helper_local_shadow']=H+'DT="bf16"\n'+K+'\ndef invoke(a):\n    DT="fp32"\n    return k(a).data\n'+RUN.replace('k(a).data','invoke(a)')
CASES['dtype_side_effect_before_fill']=H+'COUNT=0\ndef choose_dtype():\n    global COUNT\n    COUNT+=1\n    return "bf16"\n'+K.replace('dtype=DT,fill=1.003','dtype=choose_dtype(),fill=COUNT+1.003')+RUN
CASES['dtype_evaluation_count']=H+'COUNT=0\ndef choose_dtype():\n    global COUNT\n    COUNT+=1\n    return "bf16"\n'+K.replace('    Y=Tensor','    T=Tensor(M=A.shape[0],N=A.shape[1],dtype=choose_dtype(),fill=1)\n    Y=Tensor').replace('dtype=DT','dtype=choose_dtype()')+RUN.replace("return {'output':k(a).data}","v=k(a).data\n    return {'output':v,'count':np.array([COUNT])}")
CASES['body_mutates_module_between_calls']=H+'DT="bf16"\n'+K.replace('    return Y','    globals()["DT"]="fp32"\n    return Y')+RUN.replace("return {'output':k(a).data}","v=k(a).data\n    w=k(a).data\n    return {'output':v,'second':w}")
CASES['raising_dtype']=H+'def choose_dtype():\n    raise ValueError("dtype-sentinel")\n'+K.replace('dtype=DT','dtype=choose_dtype()')+RUN
CASES['nonstring_dtype']=H+'DT=object()\n'+K+RUN
CASES['forwarded_kernel_wrapper']=H+'real_kernel=kernel\ndef kernel(f):\n    return real_kernel(f)\nkernel.__dict__.update(real_kernel.__dict__)\nDT="bf16"\n'+K+RUN
CASES['dynamic_integer_dtype']=H+'DT="int32"\n'+K.replace('fill=1.003','fill=1.75')+RUN
for dt in ['bf16','fp16','fp32','fp64','int32']:
 CASES['literal_'+dt]=H+K.replace('dtype=DT',f'dtype="{dt}"')+RUN
CASES['literal_nested_call']=H+K.replace('dtype=DT','dtype="bf16"')+RUN.replace('k(a).data','k(k(a)).data')
CASES['literal_side_effect_default']=H+'COUNT=0\ndef effect():\n    global COUNT\n    COUNT+=1\n    return COUNT\ndef marker(x=effect()):\n    pass\n'+K.replace('dtype=DT','dtype="bf16"')+RUN
def serial(out):return {k:{'value':np.asarray(v).tolist(),'dtype':str(np.asarray(v).dtype)} for k,v in out.items()}
def call(fn):
 try:return {'out':serial(fn())}
 except Exception as e:return {'error':type(e).__name__,'message':str(e)}
@pytest.mark.parametrize('name',list(CASES))
@pytest.mark.parametrize('fp64',[True,False])
def test_runtime_semantics(name,fp64):
 code=CASES[name];x=np.array([[1.003,1.009]],dtype=np.float64)
 clear_dims();stop_trace();reasons=[];fallback=[]
 low=na.numba_lower(code,boundscheck=True,use_fp64=fp64,on_decline=reasons.append)
 slow=call(lambda:execute_translated(code,{},all_inputs={'a':x.copy()},fast=False)[0])
 raw=call(lambda:na.execute_fast(code,{'a':x.copy()},use_fp64=fp64,on_fallback=fallback.append))
 public=call(lambda:execute_translated(code,{},all_inputs={'a':x.copy()},fast=True,use_fp64=fp64)[0])
 rec=dict(name=name,fp64=fp64,slow=slow,raw=raw,public=public,static_declines=reasons,runtime_declines=fallback,lowered=low is not None)
 (OUT/(name+'.py')).write_text(code)
 if low:(OUT/(name+'-'+str(fp64)+'.lowered.py')).write_text(low)
 (OUT/(name+'-'+str(fp64)+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
 if 'error' in slow:
  assert 'error' in public,rec
  assert any(s.startswith('kernel_output_dtype_unestablished:k') for s in reasons+fallback),rec
  return
 assert 'out' in public,rec
 # Exact carrier-cast comparison, not a tolerance adjustment. Fallback stays fp64.
 expected={}
 for k,v in slow['out'].items():
  arr=np.array(v['value'],dtype=v['dtype'])
  if not fp64 and not fallback and arr.dtype.kind=='f':arr=arr.astype(np.float32)
  expected[k]=arr
 assert set(expected)==set(public['out']),rec
 for k,v in expected.items():np.testing.assert_array_equal(v,public['out'][k]['value'],err_msg=str(rec))
 if name=='forwarded_kernel_wrapper':assert any('the @kernel decorator' in s for s in fallback),rec

@pytest.mark.parametrize('binding',['parameter','module'])
def test_normalizer_namespace_binding(binding):
 spec=importlib.util.spec_from_file_location('r8normalizer',pathlib.Path(os.environ['REVIEW_SOURCE'])/'tests/test_lowering_final_rounding.py')
 t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 before='def f(np):\n    s=0\n    for m in numba.prange(3):\n        s+=m\n    return s\n'
 if binding=='module':before=before.replace('f(np)','f()')
 after=before.replace('for m in numba.prange(3):\n        s+=m','for _m_prange in numba.prange(3):\n        m=np.intp(_m_prange)\n        s+=m')
 import types,numba
 fake=types.SimpleNamespace(intp=lambda n:n+10);vals=[]
 for src in [before,after]:
  ns={'numba':numba,'np':fake};exec(src,ns);vals.append(ns['f'](fake) if binding=='parameter' else ns['f']())
 norms=[t._normalise_prange_rebind(s) for s in [before,after]]
 rec=dict(before=before,after=after,values=vals,normalizations=norms)
 (OUT/('normalizer-'+binding+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
 assert vals[0]!=vals[1]
 assert norms[0]!=norms[1],rec

