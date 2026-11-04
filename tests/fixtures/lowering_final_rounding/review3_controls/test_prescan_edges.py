# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect,json,os,pathlib
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib import clear_dims,stop_trace
from solar.einsum_lib.verify import execute_translated
R=pathlib.Path(__file__).parent
H='from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\nimport numpy as np\n'
K='@kernel\ndef k(A:Tensor("M,N")) -> Tensor("M,N"):\n'
Y='    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype="bf16",fill=2.003)\n    return Y\n'
RUN='\ndef run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs["a"],"M,N")\n    return {"out":k(a).data}\n'
CASES={
'parameter_tensor':H+'def factory(M,N,dtype,fill):\n    return Tensor(M=M,N=N,dtype=dtype,fill=7)\n'+K.replace('A:Tensor("M,N")','A:Tensor("M,N"),Tensor')+Y+RUN.replace('k(a)','k(a,factory)'),
'import_T2_inside':H+K+'    from solar.einsum_lib import Tensor as T2\n    Tmp=T2(M=1,N=2,dtype="fp32")\n'+Y+RUN,
'getattr_tensor':H+'import solar.einsum_lib as mod\n'+K+'    Tmp=getattr(mod,"Tensor")(M=1,N=2,dtype="fp32")\n'+Y+RUN,
'helper_positional':H+'def helper(A):\n    Tmp=Tensor(M=A.shape[0],N=A.shape[1],dtype="fp32",fill=3)\n    return Tmp\n'+K+'    Tmp=helper(A)\n'+Y+RUN,
'helper_keywords':H+'def helper(A):\n    Tmp=Tensor(M=A.shape[0],N=A.shape[1],dtype="fp32",fill=3)\n    return Tmp\n'+K+'    Tmp=helper(A=A)\n'+Y+RUN,
'dim_kwargs':H+K+'    for m in Dim("M",**{"parallel":False}):\n        pass\n'+Y+RUN,
'dim_starargs':H+K+'    for m in Dim(*("M",)):\n        pass\n'+Y+RUN,
'dims_string_fill':H+K+'    spec=Tensor("M,N",fill=7,dtype="fp32")\n'+Y+RUN,
'nested_def_tensor':H+K+'    def Tensor(M,N,dtype,fill):\n        return A\n'+Y+RUN,
'import_wrong_symbol_as_tensor': 'from solar.einsum_lib import Dim as Tensor,kernel\n@kernel\ndef k(A):\n    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype="bf16",fill=7)\n    return Y\ndef run_model(inputs):\n    return {"out":k(inputs["a"]).data}\n',
'kernel_import_tensor':H+K+'    from solar.einsum_lib import Tensor\n'+Y+RUN,
}

def call(f):
    try:
        out=f()
        if isinstance(out,tuple):out=out[0]
        if out is None:return {'none':True}
        return {'value':{k:{'data':np.asarray(v).tolist(),'dtype':str(np.asarray(v).dtype)} for k,v in out.items()}}
    except Exception as e:return {'error':type(e).__name__,'message':str(e)}

def observe(name,fp64):
    code=CASES[name];role=os.environ['REVIEW_ROLE'];out=R/'runs'/('edges-'+role);out.mkdir(exist_ok=True)
    clear_dims();stop_trace();why=[];fallback=[]
    try:low=na.numba_lower(code,use_fp64=fp64,on_decline=why.append);low_error=None
    except Exception as e:low=None;low_error={'error':type(e).__name__,'message':str(e)}
    inputs={'a':np.ones((1,2))}
    slow=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=False))
    raw=call(lambda:na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append))
    public=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
    record={'name':name,'fp64':fp64,'role':role,'code':code,'lowered':isinstance(low,str),'lower_error':low_error,'declines':why,'fallback':fallback,'slow':slow,'raw':raw,'public':public}
    stem=name+'-'+str(fp64);(out/(stem+'.py')).write_text(code)
    if isinstance(low,str):(out/(stem+'.lowered.py')).write_text(low)
    (out/(stem+'.json')).write_text(json.dumps(record,indent=2)+'\n');return record

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_boundary(name,fp64):
    r=observe(name,fp64)
    if 'error' in r['slow']:
        assert r['public'].get('error')==r['slow']['error'],r
    else:
        assert 'value' in r['public'],r
        for k,v in r['slow']['value'].items():
            expected=np.asarray(v['data'])
            if not fp64 and not r['fallback'] and expected.dtype.kind=='f':expected=expected.astype(np.float32)
            np.testing.assert_array_equal(r['public']['value'][k]['data'],expected,err_msg=str(r))
    if name in ['dims_string_fill','helper_positional']:
        # These calls are intentionally within the old literal/helper boundary.
        assert r['lowered'] and not r['declines'],r
    else:
        assert not r['lowered'] and any(s.startswith('kernel_allocation_') for s in r['declines']),r
    assert all(s.startswith('kernel_allocation_') for s in r['fallback']),r
