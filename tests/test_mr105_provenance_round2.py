# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect,json
import numpy as np
import pytest
from solar.einsum_lib import clear_dims,stop_trace
from solar.einsum_lib.numba_accel import numba_lower,execute_fast
from solar.einsum_lib.verify import execute_translated

HEADER='from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nDT="bf16"\n'
RUN='''
def run_model(inputs):
    clear_dims()
    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')
    y=k(a)
    return {'output':y.data}
'''
CASES={
'for_break':'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    for m in Dim('M'):
        Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
        break
    return Y
''',
'for_continue':'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    for m in Dim('M'):
        Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
        continue
    return Y
''',
'while_break':'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    while A[0,0]>0:
        Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
        break
    return Y
''',
'four_step_loop':'''    Y0=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    Y1=Y0
    Y2=Y0
    Y3=Y0
    Y4=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
    for i in range(4):
        Y0=Y1
        Y1=Y2
        Y2=Y3
        Y3=Y4
    return Y0
''',
'local_dtype_shadows_global':'''    DT='fp32'
    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)
    return Y
''',
'finally_overrides_return':'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    try:
        return Y
    finally:
        return Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
''',
'match_untaken':'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    match 1:
        case 0:
            Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
    return Y
''',
'global_conditional_rebind':'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)
    return Y
''',
}
@pytest.fixture(autouse=True)
def cleanup():
 clear_dims();stop_trace();yield;clear_dims();stop_trace()

@pytest.mark.parametrize('case',list(CASES))
def test_return_site_boundaries_match_or_decline(tmp_path,case):
 header=HEADER+('if True:\n    DT="fp32"\n' if case=='global_conditional_rebind' else '')
 code=header+'@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n'+CASES[case]+RUN
 x=np.array([[1.003,1.009]],dtype=np.float64);seen=[];declines=[]
 slow,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=False)
 kwargs={'on_decline':declines.append} if 'on_decline' in inspect.signature(numba_lower).parameters else {}
 low=numba_lower(code,use_fp64=True,boundscheck=True,**kwargs)
 fast=execute_fast(code,{'a':x},use_fp64=True,on_fallback=seen.append)
 obs={'case':case,'code':code,'lowered':low,'lower_decline':declines,'fallback':seen,'slow':np.asarray(slow['output']).tolist(),'fast':None if fast is None else np.asarray(fast['output']).tolist()}
 (tmp_path/'observation.json').write_text(json.dumps(obs,indent=2)+'\n')
 if fast is not None:np.testing.assert_array_equal(fast['output'],slow['output'])
 elif low is None and kwargs:assert declines and declines[0].startswith('kernel_output_dtype_')
