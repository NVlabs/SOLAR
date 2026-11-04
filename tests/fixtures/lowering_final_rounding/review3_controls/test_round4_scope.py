# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round-4 requested scope/evaluation probes, both carriers."""
import ast,json,os,sys,types
from pathlib import Path
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
import test_round3_scope as old

R=Path(__file__).parent
H,K,Y,RUN=old.H,old.K,old.Y,old.RUN
FUTURE='from __future__ import annotations\n'
WRITER='def change():\n    global Tensor\n    Tensor=int\n'
BRIDGE='review4_runtime_bridge'
sys.modules[BRIDGE]=types.ModuleType(BRIDGE)
def program(body='',prefix='',header=H,tail=Y,run=RUN):return header+prefix+K+body+tail+run

# "prescan" allows a correctly reported numba compiler fallback for constructs
# outside numba's supported subset. "invalid" is judged by Python compile,
# not just ast.parse (which accepts some forbidden PEP 572 forms).
CASES={
 'annotation_return':(program(prefix='def unrelated() -> (Tensor := int):\n    pass\n'),'decline'),
 'annotation_kernel_nested':(program('    def unused(x: (Tensor := int)):\n        pass\n'),'decline'),
 'annotation_lambda_uninvoked':(program(prefix='def unused(x: (lambda: (Tensor := int))):\n    pass\n'),'lower'),
 'annotation_lambda_invoked':(program(prefix='def unused(x: (lambda: globals().__setitem__("Tensor",int))()):\n    pass\n'),'decline'),
 'annotation_in_class_local':(program(prefix='class Holder:\n    def unused(x: (Tensor := int)):\n        pass\n'),'lower'),
 'annotation_in_class_global':(program(prefix='class Holder:\n    global Tensor\n    def unused(x: (Tensor := int)):\n        pass\n'),'decline'),
 'future_plain':(program(prefix='def unused(x: Tensor) -> Tensor:\n    pass\n',header=FUTURE+H),'lower'),
 'future_namespace_call':(program(prefix='def unused(x: globals().__setitem__("Tensor",int)):\n    pass\n',header=FUTURE+H),'observe'),
 'future_walrus_invalid':(program(prefix='def unused(x: (Tensor := int)):\n    pass\n',header=FUTURE+H),'invalid'),
 'future_import_not_first':(program(header='x=1\n'+FUTURE+H),'invalid'),
 'writer_alias':(program(prefix=WRITER+'f2=change\nf2()\n'),'decline'),
 'writer_getattr_sysmodules':(program(prefix=WRITER+'import sys\n__name__="review4_runtime_bridge"\nsys.modules[__name__].change=change\ngetattr(sys.modules[__name__],"change")()\n'),'decline'),
 'writer_lambda_wrapper':(program(prefix=WRITER+'f=lambda: change()\nf()\n'),'decline'),
 'writer_class_method':(program(prefix='class Holder:\n    @staticmethod\n    def m():\n        global Tensor\n        Tensor=int\nHolder.m()\n'),'decline'),
 'writer_default':(program(prefix=WRITER+'def unused(x=change()):\n    pass\n'),'decline'),
 'writer_decorator':(program(prefix='def deco(f):\n    global Tensor\n    Tensor=int\n    return f\n@deco\ndef unused():\n    pass\n'),'decline'),
 'writer_inside_uncalled':(program(prefix=WRITER+'def unused():\n    change()\n'),'lower'),
 'writer_transitive_called':(program(prefix=WRITER+'def invoke():\n    change()\ninvoke()\n'),'decline'),
 'writer_mutual_recursion':(program(prefix='def first(n):\n    if n:\n        second(n-1)\ndef second(n):\n    global Tensor\n    Tensor=int\n    if n:\n        first(n-1)\nfirst(1)\n'),'decline'),
 'called_nonlocal_closure':(program(prefix='def outer():\n    Tensor=int\n    def inner():\n        nonlocal Tensor\n        Tensor=float\n    inner()\nouter()\n'),'lower'),
 'writer_class_namespace':(program(prefix='class Holder:\n    globals()["Tensor"]=int\n'),'decline'),
 'writer_nested_class_global':(program(prefix='class Outer:\n    class Inner:\n        global Tensor\n        Tensor=int\n'),'decline'),
 'writer_class_body_method_call':(program(prefix='class Holder:\n    def m():\n        global Tensor\n        Tensor=int\n    m()\n'),'decline'),
 'comprehension_nested_target':(program('    unused=[[Tensor for Tensor in range(3)] for Tensor in range(2)]\n'),'lower'),
 'comprehension_iterable_walrus_invalid':(program('    unused=[x for x in (Tensor := [1])]\n'),'invalid'),
 'comprehension_condition_walrus':(program('    unused=[x for x in [1] if (Tensor := int)]\n'),'decline'),
 'generator_walrus_unconsumed':(program('    unused=((Tensor := int) for x in [1])\n'),'decline'),
 'generator_walrus_consumed':(program('    unused=list((Tensor := int) for x in [1])\n'),'decline'),
 'dict_comprehension_target':(program('    unused={Tensor: 1 for Tensor in range(3)}\n'),'lower'),
 'class_comprehension_target':(program(prefix='class Holder:\n    unused=[Tensor for Tensor in range(3)]\n'),'lower'),
 'generator_local_unconsumed':(program('    unused=(Tensor for Tensor in range(3))\n'),'prescan'),
 'generator_local_consumed':(program('    unused=list(Tensor for Tensor in range(3))\n'),'prescan'),
 'lambda_calls_writer_uninvoked':(program(prefix=WRITER+'unused=lambda: change()\n'),'lower'),
 'lambda_calls_writer_invoked':(program(prefix=WRITER+'(lambda: change())()\n'),'decline'),
 'lambda_namespace_uninvoked':(program(prefix='unused=lambda: globals().__setitem__("Tensor",int)\n'),'lower'),
 'lambda_namespace_invoked':(program(prefix='(lambda: globals().__setitem__("Tensor",int))()\n'),'decline'),
 'lambda_namespace_named_invoked':(program(prefix='f=lambda: globals().__setitem__("Tensor",int)\nf()\n'),'decline'),
 'lambda_nested_uninvoked':(program('    unused=lambda: (lambda: (Tensor := int))\n'),'lower'),
 'lambda_nested_invoked':(program(prefix='f=lambda: (lambda: globals().__setitem__("Tensor",int))\nf()()\n'),'decline'),
 'lambda_constructor_binding':(program('    Tensor=lambda **kw: A\n'),'decline'),
 'lambda_nested_default':(program('    unused=lambda x=(lambda y=(Tensor := int): y): x\n'),'decline'),
}

SIDE='\ndef run_model(inputs):\n    clear_dims()\n    return {"out":k(inputs["a"]).data,"side":np.array(COUNT)}\n'
MARK='COUNT=0\ndef mark():\n    global COUNT\n    COUNT+=1\n    return int\ndef unused(x: mark()):\n    pass\n'
CASES['future_annotation_side_effect']=(program(prefix=MARK,header=FUTURE+H,run=SIDE),'observe')
CASES['eager_annotation_side_effect']=(program(prefix=MARK,run=SIDE),'observe')

def observe(name,fp64):
    code,expected=CASES[name];reasons=[];fallback=[]
    try:compile(code,'<round4-probe>','exec',dont_inherit=True);compile_error=None
    except Exception as exc:compile_error={'type':type(exc).__name__,'message':str(exc)}
    try:low=na.numba_lower(code,use_fp64=fp64,on_decline=reasons.append);lower_error=None
    except Exception as exc:low=None;lower_error={'type':type(exc).__name__,'message':str(exc)}
    inputs={'a':np.full((1,2),2.003,dtype=np.float64)}
    def call(f):
        try:return old.isolated_call(f)
        finally:vars(sys.modules[BRIDGE]).pop('change',None)
    slow=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=False))
    raw=call(lambda:na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append))
    public=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
    r=dict(name=name,fp64=fp64,expected=expected,code=code,compile_error=compile_error,lower_error=lower_error,lowered=isinstance(low,str),declines=reasons,fallback=fallback,slow=slow,raw=raw,public=public)
    dest=R/'runs'/('round4-'+os.environ['REVIEW_ROLE']);dest.mkdir(exist_ok=True);stem=name+'-'+str(fp64)
    (dest/(stem+'.json')).write_text(json.dumps(r,indent=2)+'\n');(dest/(stem+'.py')).write_text(code)
    if isinstance(low,str):(dest/(stem+'.lowered.py')).write_text(low)
    return r

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_round4_scope(name,fp64):
    r=observe(name,fp64)
    if 'error' in r['slow']:
        assert r['public'].get('error')==r['slow']['error'],r
    else:
        assert 'value' in r['public'],r
        for key,value in r['slow']['value'].items():
            want=np.asarray(value['data'])
            if not fp64 and not r['fallback']:want=want.astype(np.float32)
            np.testing.assert_array_equal(r['public']['value'][key]['data'],want,err_msg=str(r))
    if r['expected']=='decline':assert not r['lowered'] and any(x.startswith('kernel_') for x in r['declines']),r
    elif r['expected']=='lower':assert r['lowered'] and not r['declines'] and not r['fallback'],r
    elif r['expected']=='prescan':assert r['lowered'] and not r['declines'],r
    elif r['expected']=='invalid':assert r['compile_error'] and r['compile_error']['type']=='SyntaxError' and 'error' in r['public'],r
