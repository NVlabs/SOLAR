# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent r10 probes; expected public equality frozen before execution."""
import json,os
from pathlib import Path
import numpy as np
import pytest
from solar.einsum_lib.numba_accel import numba_lower,execute_fast
from solar.einsum_lib.verify import execute_translated
from test_round9_provenance import H,END,outcome
R=Path(__file__).parent
MODEL='def model(a):\n    return k(a)\ndef replacement(a):\n    return Box()\n'
SIDE='SIDE=int(model(np.ones((1,2))).data)\n'
MUTATIONS={
 'alias_code':MODEL+'alias=model\nalias.__code__=replacement.__code__\n'+SIDE+END,
 'alias_chain_code':MODEL+'alias=model\nother=alias\nother.__code__=replacement.__code__\n'+SIDE+END,
 'parameter_code':MODEL+'def change(f):\n    f.__code__=replacement.__code__\nchange(model)\n'+SIDE+END,
 'type_setattr_code':MODEL+'type(model).__setattr__(model,"__code__",replacement.__code__)\n'+SIDE+END,
 'direct_defaults':MODEL+'model.__defaults__=(1,)\nSIDE=model(np.ones((1,2))).data\n'+END,
 'direct_wrapped':MODEL+'model.__wrapped__=replacement\nSIDE=model(np.ones((1,2))).data\n'+END,
 'direct_dict':MODEL+'model.__dict__["note"]=1\nSIDE=model(np.ones((1,2))).data\n'+END,
 'direct_setattr':MODEL+'setattr(model,"__code__",replacement.__code__)\n'+SIDE+END,
 'globals_write':MODEL+'globals()["model"]=replacement\n'+SIDE+END,
 'after_last_call':MODEL+'SIDE=model(np.ones((1,2))).data\nmodel.__code__=replacement.__code__\n'+END,
}
ALIASES={
 'local_kernel_alias':'def run_model(inputs):\n    clear_dims()\n    g=k\n    return {"output":g(Tensor.from_numpy(inputs["a"],"M,N",dtype="fp32")).data}\n',
 'module_kernel_alias':'g=k\ndef run_model(inputs):\n    clear_dims()\n    return {"output":g(Tensor.from_numpy(inputs["a"],"M,N",dtype="fp32")).data}\n',
}
READLESS={
 'local_shadow':'def run_model(inputs):\n    clear_dims()\n    k=lambda a:a\n    return {"output":k(inputs["a"])}\n',
 'parameter_shadow':'def model(a,k):\n    return k(a)\ndef run_model(inputs):\n    clear_dims()\n    return {"output":model(inputs["a"],lambda a:a)}\n',
 'default_shadow':'def model(a,k=lambda a:a):\n    return k(a)\ndef run_model(inputs):\n    clear_dims()\n    return {"output":model(inputs["a"])}\n',
}
BH=H.replace('clear_dims\n','clear_dims,apply_mask,top_k_mask,gather,concat,stack,slice_at\n')
BUILTINS={
 'direct_from_numpy_bf16':'def run_model(inputs):\n    clear_dims()\n    return {"output":Tensor.from_numpy(inputs["a"],"M,N",dtype="bf16").data}\n',
 'direct_from_numpy_fp16':'def run_model(inputs):\n    clear_dims()\n    return {"output":Tensor.from_numpy(inputs["a"],"M,N",dtype="fp16").data}\n',
 'named_from_numpy_bf16':'def run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs["a"],"M,N",dtype="bf16")\n    return {"output":a.data}\n',
 'nested_from_numpy_bf16':'def run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(np.ones((1,2)),"M,N",dtype="fp32")\n    return {"output":apply_mask(Tensor.from_numpy(inputs["a"],"M,N",dtype="bf16"),a,0).data}\n',
 'builtin_lambda_shadow':'def run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs["a"],"M,N",dtype="fp32")\n    return {"output":np.array((lambda apply_mask:int(apply_mask(a,a).data))(lambda *args:Box()))}\n',
}
def observe(name,code,fp64):
 compile(code,'<independent-r10>','exec');inputs={'a':np.array([[1.003,1.003]],dtype=np.float32)};reasons=[];fallback=[]
 low=numba_lower(code,use_fp64=fp64,boundscheck=True,on_decline=reasons.append)
 slow=outcome(lambda:execute_translated(code,{},all_inputs=inputs,fast=False)[0])
 raw=outcome(lambda:execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append))
 public=outcome(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64)[0])
 row=dict(name=name,fp64=fp64,code=code,lowered=low is not None,reasons=reasons,fallback=fallback,slow=slow,raw=raw,public=public)
 d=R/'runs'/('round10-'+os.environ['REVIEW_ROLE']);d.mkdir(parents=True,exist_ok=True);p=d/(name+'-'+str(fp64));p.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');p.with_suffix('.py').write_text(code)
 if low is not None:p.with_suffix('.lowered.py').write_text(low)
 assert 'value' in slow,row
 return row
@pytest.mark.parametrize('name',MUTATIONS)
@pytest.mark.parametrize('fp64',[False,True])
def test_visible_mutation(tmp_path,name,fp64):
 r=observe(name,H+MUTATIONS[name],fp64);assert r['public']==r['slow'],r
 assert not r['lowered'] and any(s.startswith('kernel_') for s in r['reasons']),r
@pytest.mark.parametrize('name',ALIASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_kernel_alias(tmp_path,name,fp64):
 r=observe(name,H+ALIASES[name],fp64);assert r['public']==r['slow'],r
 assert r['lowered'] or any(s.startswith('kernel_') for s in r['reasons']),r
@pytest.mark.parametrize('name',READLESS)
@pytest.mark.parametrize('fp64',[False,True])
def test_no_data_read_callee_binding(tmp_path,name,fp64):
 h=H.replace('dtype="fp32"','dtype="bf16"')
 r=observe(name,h+READLESS[name],fp64);assert r['public']==r['slow'],r
@pytest.mark.parametrize('name',BUILTINS)
@pytest.mark.parametrize('fp64',[False,True])
def test_builtin_result_semantics(tmp_path,name,fp64):
 r=observe(name,BH+BUILTINS[name],fp64);assert r['public']==r['slow'],r
