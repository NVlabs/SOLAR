# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util,json,os,pathlib,sys
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib import Tensor as ActualTensor,clear_dims,stop_trace
from solar.einsum_lib.verify import execute_translated

R=pathlib.Path(__file__).parent
MODS=R/'probe_modules'
MODS.mkdir(exist_ok=True)
MODULES={
 'review4_tensor_reexport':'from solar.einsum_lib import Tensor\n',
 'solar.einsum_lib._review4_tensor_reexport':'from solar.einsum_lib import Tensor\n',
 'solar.einsum_lib._review4_tensor_subclass':'from solar.einsum_lib import Tensor as BaseTensor\nclass Tensor(BaseTensor):\n    def __init__(self,*args,**kwargs):\n        if "fill" in kwargs: kwargs["fill"]=7\n        super().__init__(*args,**kwargs)\n',
}
module_records={}
for name,code in MODULES.items():
    p=MODS/(name+'.py');p.write_text(code)
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module_records[name]={'path':str(p),'tensor_is_DSL_class':module.Tensor is ActualTensor}
(R/'evidence'/('probe-module-identities-'+os.environ['REVIEW_ROLE']+'.json')).write_text(json.dumps(module_records,indent=2)+'\n')

H='from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\nimport numpy as np\n'
K='@kernel\ndef k(A):\n'
Y='    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype="bf16",fill=2.003)\n    return Y\n'
RUN='\ndef run_model(inputs):\n    clear_dims()\n    return {"out":k(inputs["a"]).data}\n'
def program(body='',prefix='',header=H):return header+prefix+K+body+Y+RUN

# Expected "lower" is reserved for explicitly requested positive controls.
CASES={
 'nested_assignment_only':(program('    def helper():\n        Tensor=1\n        return 2\n    unused=helper()\n'),'lower'),
 'nested_parameter_only':(program('    def helper(Tensor):\n        return 2\n    unused=helper(3)\n'),'lower'),
 'lambda_parameter_only':(program('    unused=lambda Tensor: 2\n'),'lower'),
 'lambda_binds_constructor':(program('    Tensor=lambda **kw: A\n'),'decline'),
 'match_capture':(program('    match int:\n        case Tensor:\n            pass\n'),'decline'),
 'delete_then_use':(program('    from solar.einsum_lib import Tensor\n    del Tensor\n'),'decline'),
 'class_member_only':(program(prefix='class Holder:\n    Tensor=int\n'),'lower'),
 'class_named_tensor':(program(prefix='class Tensor:\n    pass\n'),'decline'),
 'kernel_exec':(program('    exec("global Tensor; Tensor=int")\n'),'decline'),
 'kernel_globals':(program('    globals()["Tensor"]=int\n'),'decline'),
 'module_if_binding':(program(prefix='if True:\n    Tensor=int\n'),'decline'),
 'module_for_binding':(program(prefix='for Tensor in [int]:\n    pass\n'),'decline'),
 'module_match_binding':(program(prefix='match int:\n    case Tensor:\n        pass\n'),'decline'),
 'module_exec_binding':(program(prefix='exec("Tensor=int")\n'),'decline'),
 'module_globals_binding':(program(prefix='globals()["Tensor"]=int\n'),'decline'),
 'module_delete_binding':(program(prefix='del Tensor\n'),'decline'),
 'relative_import':(program(header='__package__="solar.einsum_lib"\nfrom .core import Tensor\nfrom solar.einsum_lib import kernel,clear_dims\n'),'decline'),
 'module_attribute_constructor':(program(prefix='from solar import einsum_lib as E\n').replace('Y=Tensor(','Y=E.Tensor('),'decline'),
 'computed_import':(program('    Tmp=__import__("solar.einsum_lib",fromlist=["Tensor"]).Tensor(M=1,N=2)\n'),'decline'),
 'reexport_in_allowed_namespace':(program(header='from solar.einsum_lib._review4_tensor_reexport import Tensor\nfrom solar.einsum_lib import kernel,clear_dims\n'),'lower'),
 'reexport_external_module':(program(header='from review4_tensor_reexport import Tensor\nfrom solar.einsum_lib import kernel,clear_dims\n'),'lower'),
 'subclass_in_allowed_namespace':(program(header='from solar.einsum_lib._review4_tensor_subclass import Tensor\nfrom solar.einsum_lib import kernel,clear_dims\n'),'decline'),
 'not_importable_module':(program(header='from solar.einsum_lib._review4_missing_module import Tensor\nfrom solar.einsum_lib import kernel,clear_dims\n'),'decline'),
}

def call(f):
    try:
        out=f()
        if isinstance(out,tuple):out=out[0]
        if out is None:return {'none':True}
        return {'value':{k:{'data':np.asarray(v).tolist(),'dtype':str(np.asarray(v).dtype)} for k,v in out.items()}}
    except Exception as e:return {'error':type(e).__name__,'message':str(e)}

def observe(name,fp64):
    code,expect=CASES[name];folder=R/'runs'/('binding-'+os.environ['REVIEW_ROLE']);folder.mkdir(exist_ok=True)
    clear_dims();stop_trace();why=[];fallback=[]
    try:low=na.numba_lower(code,use_fp64=fp64,on_decline=why.append);error=None
    except Exception as e:low=None;error={'type':type(e).__name__,'message':str(e)}
    inputs={'a':np.ones((1,2))}
    slow=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=False))
    raw=call(lambda:na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append))
    public=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
    rec={'case':name,'role':os.environ['REVIEW_ROLE'],'fp64':fp64,'expected':expect,'code':code,
         'lowered':isinstance(low,str),'lower_error':error,'declines':why,'fallback':fallback,'slow':slow,'raw':raw,'public':public}
    stem=name+'-'+str(fp64);(folder/(stem+'.py')).write_text(code)
    if isinstance(low,str):(folder/(stem+'.lowered.py')).write_text(low)
    (folder/(stem+'.json')).write_text(json.dumps(rec,indent=2)+'\n');return rec

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_binding_contract(name,fp64):
    r=observe(name,fp64)
    if 'error' in r['slow']:
        assert r['public'].get('error')==r['slow']['error'],r
    else:
        assert 'value' in r['public'],r
        for k,v in r['slow']['value'].items():
            expected=np.asarray(v['data'])
            if not fp64 and not r['fallback'] and expected.dtype.kind=='f':expected=expected.astype(np.float32)
            np.testing.assert_array_equal(r['public']['value'][k]['data'],expected,err_msg=str(r))
    if r['expected']=='lower':
        assert r['lowered'] and not r['declines'] and not r['fallback'],r
    else:
        assert not r['lowered'] and any(s.startswith('kernel_allocation_callee_unresolved:') for s in r['declines']),r
        assert all(s.startswith('kernel_allocation_callee_unresolved:') for s in r['fallback']),r
