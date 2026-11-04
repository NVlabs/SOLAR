# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round10 allocation-shape and literal-boundary probes."""
import inspect,json,os,pathlib
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib import clear_dims,stop_trace
from solar.einsum_lib.verify import execute_translated
R=pathlib.Path(__file__).parent;OUT=R/'runs/round10';OUT.mkdir(parents=True,exist_ok=True)
H='from solar.einsum_lib import Tensor, kernel, clear_dims\nimport numpy as np\nCOUNT=0\nDT="bf16"\ndef choose_dtype():\n    global COUNT\n    COUNT+=1\n    return "bf16"\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n'
Y='    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype="bf16",fill=COUNT+1.003)\n    return Y\n'
RUN='\ndef run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs["a"],"M,N",dtype="fp32")\n    return {"output":k(a).data}\n'
ALLOC='Tensor(M=A.shape[0],N=A.shape[1],dtype=choose_dtype(),fill=0)'
CASES={
'nested_helper':H+K+'    def helper():\n        T='+ALLOC+'\n        return 0\n    helper()\n'+Y+RUN,
'loop':H+K+'    for i in range(3):\n        T='+ALLOC+'\n'+Y+RUN,
'positional_dtype':H+K+'    T=Tensor(None, choose_dtype(), M=A.shape[0],N=A.shape[1],fill=0)\n'+Y+RUN,
'positional_dtype_literal':H+K+'    T=Tensor(None, "bf16", M=A.shape[0],N=A.shape[1],fill=0)\n'+Y+RUN,
'kwargs_inline':H+K+'    T=Tensor(M=A.shape[0],N=A.shape[1],fill=0,**{"dtype":choose_dtype()})\n'+Y+RUN,
'kwargs_name':H+K+'    options={"dtype":choose_dtype()}\n    T=Tensor(M=A.shape[0],N=A.shape[1],fill=0,**options)\n'+Y+RUN,
'alias_local':H+K+'    Factory=Tensor\n    T=Factory(M=A.shape[0],N=A.shape[1],dtype=choose_dtype(),fill=0)\n'+Y+RUN,
'alias_import':H+'from solar.einsum_lib import Tensor as Factory\n'+K+'    T=Factory(M=A.shape[0],N=A.shape[1],dtype=choose_dtype(),fill=0)\n'+Y+RUN,
'nonallocation_spec_dynamic':H+K+'    spec=Tensor("M,N",dtype=choose_dtype())\n'+Y+RUN,
'nonallocation_spec_literal':H+K+'    spec=Tensor("M,N",dtype="fp32")\n'+Y+RUN,
'literal_nested_helper':H+K+'    def helper():\n        T=Tensor(M=A.shape[0],N=A.shape[1],dtype="fp32",fill=0)\n        return 0\n    helper()\n'+Y+RUN,
'literal_loop':H+K+'    for i in range(3):\n        T=Tensor(M=A.shape[0],N=A.shape[1],dtype="fp32",fill=0)\n'+Y+RUN,
}
def call(fn):
 try:
  r=fn()
  if isinstance(r,tuple):r=r[0]
  return {'value':{k:{'data':np.asarray(v).tolist(),'dtype':str(np.asarray(v).dtype)} for k,v in r.items()}} if r is not None else {'none':True}
 except Exception as e:return {'error':type(e).__name__,'message':str(e)}
def observe(name,fp64):
 code=CASES[name];clear_dims();stop_trace();reasons=[];fallback=[];inputs={'a':np.ones((1,2))}
 low=None
 try:low=na.numba_lower(code,use_fp64=fp64,**({'on_decline':reasons.append} if 'on_decline' in inspect.signature(na.numba_lower).parameters else {}));lowerror=None
 except Exception as e:lowerror={'error':type(e).__name__,'message':str(e)}
 slow=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=False));raw=call(lambda:na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append));public=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
 rec={'name':name,'fp64':fp64,'code':code,'lowered':isinstance(low,str),'lower_error':lowerror,'reasons':reasons,'fallback':fallback,'slow':slow,'raw':raw,'public':public}
 if isinstance(low,str):(OUT/(name+'-'+os.getenv('VARIANT','head')+'-'+str(fp64)+'.lowered.py')).write_text(low)
 (OUT/(name+'-'+os.getenv('VARIANT','head')+'-'+str(fp64)+'.json')).write_text(json.dumps(rec,indent=2)+'\n');return rec
@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_public_semantics(name,fp64):
 r=observe(name,fp64)
 if 'error' in r['slow']:
  assert 'error' in r['public'],r
  assert r['slow']['error']==r['public']['error'],r
 else:
  assert 'value' in r['public'],r
  for k,v in r['slow']['value'].items():
   expected=np.asarray(v['data'])
   if not fp64 and not r['fallback']:expected=expected.astype(np.float32)
   np.testing.assert_array_equal(expected,r['public']['value'][k]['data'],err_msg=str(r))
 if name in ['nested_helper','loop','nonallocation_spec_dynamic']:
  assert not r['lowered'] and any(s.startswith('kernel_output_dtype_unestablished:k') for s in r['reasons']),r
 if name in ['literal_nested_helper','literal_loop','nonallocation_spec_literal']:
  assert r['lowered'] and not r['reasons'],r
