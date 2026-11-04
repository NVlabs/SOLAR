# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round11 closure probes. Expected public/interpreter equality and conservative escape refusal frozen before execution."""
from pathlib import Path
import ast,json,os
import numpy as np
import pytest
from solar.einsum_lib.numba_accel import numba_lower,execute_fast
try:
    from solar.einsum_lib.numba_accel import is_typed_decline
except ImportError:
    def is_typed_decline(x): return False
from solar.einsum_lib.verify import execute_translated
from test_round9_provenance import H,END,outcome
R=Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE'];MODEL='def model(a):\n    return k(a)\ndef replacement(a):\n    return Box()\n';SIDE='SIDE=int(model(np.ones((1,2))).data)\n'
MUT={
 'class_global_alias':'def mutate():\n    model=lambda a:a\n    class C:\n        global model\n        alias=model\n        alias.__code__=replacement.__code__\nmutate()\n',
 'class_global_argument':'def change(f):\n    f.__code__=replacement.__code__\ndef mutate():\n    model=lambda a:a\n    class C:\n        global model\n        change(model)\nmutate()\n',
 'class_global_default':'def mutate():\n    model=lambda a:a\n    class C:\n        global model\n        def change(f=model):\n            f.__code__=replacement.__code__\n        change()\nmutate()\n',
 'class_global_type':'def mutate():\n    model=lambda a:a\n    class C:\n        global model\n        type(model).__setattr__(model,"__code__",replacement.__code__)\nmutate()\n',
 'function_global_alias':'def mutate():\n    global model\n    alias=model\n    alias.__code__=replacement.__code__\nmutate()\n',
 'nested_nonlocal_alias':'def mutate():\n    alias=model\n    def inner():\n        nonlocal alias\n        alias.__code__=replacement.__code__\n    inner()\nmutate()\n',
}
SAFE={
 'local_shadow':'def unused():\n    model=lambda a:a\n    alias=model\n',
 'parameter_shadow':'def unused(model):\n    alias=model\n',
 'loop_target_shadow':'def unused():\n    for model in range(2):\n        alias=model\n',
 'comprehension_shadow':'def unused():\n    return [model for model in range(2)]\n',
 'lambda_shadow':'unused=lambda model:model\n',
}
def record(name,code,fp64):
 compile(code,'<round11-review4>','exec');inputs={'a':np.array([[1.003,1.003]],dtype=np.float32)};reasons=[];fallback=[]
 low=numba_lower(code,use_fp64=fp64,boundscheck=True,on_decline=reasons.append)
 slow=outcome(lambda:execute_translated(code,{},all_inputs=inputs,fast=False)[0]);raw=outcome(lambda:execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append));public=outcome(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64)[0])
 row={'name':name,'fp64':fp64,'source':code,'lowered':low is not None,'reasons':reasons,'fallback':fallback,'slow':slow,'raw':raw,'public':public}
 from solar.einsum_lib import numba_accel as na
 row['escapes']=getattr(na._Program(ast.parse(code),{'k'}),'escapes',None) if hasattr(na,'_Program') else None
 d=R/'runs'/('round11-compatible-'+ROLE);d.mkdir(parents=True,exist_ok=True);p=d/(name+'-'+str(fp64));p.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');p.with_suffix('.py').write_text(code)
 if low is not None:p.with_suffix('.lowered.py').write_text(low)
 return row
@pytest.mark.parametrize('fp64',[False,True])
@pytest.mark.parametrize('name',MUT)
def test_module_identity_escapes_through_class_global(tmp_path,name,fp64):
 row=record(name,H+MODEL+MUT[name]+SIDE+END,fp64);assert row['slow']=={'value':{'output':5}},row;assert row['public']==row['slow'],row;assert not row['lowered'] and any(is_typed_decline(x) for x in row['reasons']),row
@pytest.mark.parametrize('fp64',[False,True])
@pytest.mark.parametrize('name',SAFE)
def test_nearer_local_load_is_not_module_escape(tmp_path,name,fp64):
 code=H+'def model(a):\n    return k(a)\n'+SAFE[name]+'def run_model(inputs):\n    clear_dims()\n    return {"output":model(Tensor.from_numpy(inputs["a"],"M,N",dtype="fp32")).data}\n';row=record(name,code,fp64);assert row['public']==row['slow'] and 'value' in row['slow'],row;assert row['lowered'] and not row['reasons'],row
@pytest.mark.parametrize('fp64',[False,True])
@pytest.mark.parametrize('dtype',['__invalid_dtype__','BF16'])
def test_invalid_literal_builtin_dtype_preserves_refusal(tmp_path,dtype,fp64):
 code=H+'def run_model(inputs):\n    clear_dims()\n    return {"output":Tensor.from_numpy(inputs["a"],"M,N",dtype='+repr(dtype)+').data}\n';row=record('invalid-'+dtype,code,fp64);assert 'error' in row['slow'],row;assert row['public']==row['slow'],row
