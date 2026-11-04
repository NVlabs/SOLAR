# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast,json,pathlib
import numpy as np
import pytest
from solar.einsum_lib import clear_dims,stop_trace
from solar.einsum_lib.numba_accel import numba_lower,execute_fast
from solar.einsum_lib.verify import execute_translated
O=pathlib.Path(__file__).resolve().parent
HEADER='from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n'
CASES={
 'rebound_parameter': '''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16")
    Y = A
    return Y
''',
 'conditional_rebind_false': '''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)
    if A[0, 0] < 0:
        Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)
    return Y
''',
 'dead_rebind_after_return': '''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="bf16", fill=1.003)
    return Y
    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="fp32", fill=1.003)
''',
 'runtime_dtype_literal_global': '''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype=DT, fill=1.003)
    return Y
''',
}
RUN='''
def run_model(inputs):
    clear_dims()
    a = Tensor.from_numpy(inputs['a'], 'M,N', dtype='fp32')
    return {'output': k(a).data}
'''
@pytest.fixture(autouse=True)
def clean():
 clear_dims();stop_trace();yield;clear_dims();stop_trace()
@pytest.mark.parametrize('case',list(CASES))
def test_return_provenance_matches_interpreter_or_declines(case,tmp_path):
 code=HEADER+'DT="bf16"\n@kernel\ndef k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N"):\n'+CASES[case]+RUN
 x=np.array([[1.003,1.009]],dtype=np.float64);inputs={'a':x};seen=[]
 slow,_=execute_translated(code,{'a':x.shape},all_inputs=inputs,fast=False)
 fast=execute_fast(code,inputs,use_fp64=True,on_fallback=seen.append)
 low=numba_lower(code,use_fp64=True,boundscheck=True)
 variant='head' if 'on_decline' in __import__('inspect').signature(numba_lower).parameters else 'target'
 result={'case':case,'variant':variant,'code':code,'lowered':low,'slow':np.asarray(slow['output']).tolist(),'fast':None if fast is None else np.asarray(fast['output']).tolist(),'fallback':seen}
 evidence=tmp_path/'evidence';evidence.mkdir(exist_ok=True)   # diagnostic probe output goes to pytest's tmp_path (the repo's portable convention), not to a pre-existing tests/evidence/
 (evidence/('probe-'+variant+'-'+case+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 if fast is None:assert seen and seen[0].startswith('kernel_output_dtype_')
 else:np.testing.assert_array_equal(fast['output'],slow['output'])

@pytest.mark.parametrize('dtype',['bf16','fp16'])
@pytest.mark.parametrize('form',['plain','chain','annassign','subscript','list','arithmetic'])
def test_call_form_rounds_once(dtype,form):
 body='''    Y = Tensor(M=A.shape[0], N=A.shape[1], dtype="%s", fill=1.003)
    return Y
'''%dtype
 cases={'plain':"    y = k(a)\n    return {'output': y.data}",'chain':"    x = y = k(a)\n    return {'output': y.data}",'annassign':"    y: Tensor = k(a)\n    return {'output': y.data}",'subscript':"    box = {}\n    box['x'] = k(a)\n    return {'output': box['x'].data}",'list':"    box = [k(a)]\n    return {'output': box[0].data}",'arithmetic':"    return {'output': k(a).data + 0.0}"}
 code=HEADER+'@kernel\ndef k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N", dtype="'+dtype+'"):\n'+body+'\ndef run_model(inputs):\n    clear_dims()\n    a = Tensor.from_numpy(inputs["a"], "M,N", dtype="fp32")\n'+cases[form]+'\n'
 inp={'a':np.array([[1.003,1.009]])};seen=[];slow,_=execute_translated(code,{'a':(1,2)},all_inputs=inp,fast=False);fast=execute_fast(code,inp,use_fp64=True,on_fallback=seen.append)
 if form in ('subscript','list'):
  # an earlier fix round 10 (the review adjudication, the review to confirm): a container element is an INTENTIONAL supported-path refusal under the
  # the established-or-refuse rule -- typed decline + the public output exactly the interpreter's -- not a lowering.
  low=numba_lower(code,use_fp64=True,boundscheck=True,on_decline=seen.append);assert low is None and fast is None,(low,fast)
  assert seen and seen[0].startswith('kernel_function_statement_unresolved:k: subscript:'),seen
  public,_=execute_translated(code,{'a':(1,2)},all_inputs=inp,fast=True,use_fp64=True)
  expected={'bf16':[[1.0,1.0]],'fp16':[[1.0029296875,1.0029296875]]}[dtype]
  assert np.asarray(public['output']).tolist()==np.asarray(slow['output']).tolist()==expected,(public,slow,expected)
  return
 low=numba_lower(code,use_fp64=True,boundscheck=True);assert low is not None
 count=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_truncate_'+dtype for n in ast.walk(ast.parse(low)))
 assert count==1,(form,low)
 assert fast is not None,seen
 np.testing.assert_array_equal(fast['output'],slow['output'])
