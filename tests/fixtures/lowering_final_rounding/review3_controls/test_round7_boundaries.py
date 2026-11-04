# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round6 probes: entry-binding completeness, attribute paths, module statement semantics."""
import json,os,sys,types
from pathlib import Path
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
import test_round3_scope as old
import test_round4_scope as r4
import test_round5_effects as r5
R=Path(__file__).parent;P=r4.program;W=r4.WRITER
GOOD=r4.RUN.replace('def run_model','def good');BAD=r5.BADRUN
HOLDER='class Holder:\n    @staticmethod\n    def m():\n        global Tensor\n        Tensor=int\n'
BOX='class Box:\n    data=5\n    def __int__(self):\n        return 9\n'
SIDE='\ndef run_model(inputs):\n    clear_dims()\n    return {"out":k(inputs["a"]).data,"side":np.array(SIDE)}\n'
BRIDGE='review4_r6_bridge';sys.modules[BRIDGE]=types.ModuleType(BRIDGE)

def post(body,prefix=BOX,run=SIDE):
 return P(prefix=prefix,run='\n'+body+run)
CASES={
 'entry_same_safe_pair':(P(run=GOOD+'run_model=good; run_model=good\n'),'lower'),
 'entry_nested_if':(P(run=GOOD+BAD+'if True: run_model=good; run_model=(runner,)[0]\n'),'decline'),
 'entry_for_capture':(P(run=GOOD+BAD+'run_model=good\nfor run_model in (runner,): pass\n'),'decline'),
 'entry_match_capture':(P(run=GOOD+BAD+'run_model=good\nmatch runner:\n    case run_model: pass\n'),'decline'),
 'entry_default_walrus':(P(run=GOOD+BAD+'run_model=good\ndef holder(x=(run_model:=runner)): pass\n'),'decline'),
 'entry_annotation_walrus':(P(run=GOOD+BAD+'run_model=good\nvalue: (run_model:=runner)=0\n'),'decline'),
 'entry_class_global':(P(run=GOOD+BAD+'run_model=good\nclass Install:\n    global run_model\n    run_model=runner\n'),'decline'),
 'attr_deep_class':(P(prefix='class Outer:\n    class Inner:\n        @staticmethod\n        def m():\n            global Tensor\n            Tensor=int\nnp._review4_holder=Outer\nnp._review4_holder.Inner.m()\n'),'decline'),
 'attr_intermediate_setattr':(P(prefix=HOLDER+'setattr(np,"_review4_holder",Holder)\nnp._review4_holder.m()\n'),'decline'),
 'attr_class_body_store':(P(prefix=HOLDER+'class Install:\n    np._review4_holder=Holder\nnp._review4_holder.m()\n'),'decline'),
 'attr_deep_store':(P(prefix=HOLDER+'np._review4_holder=Holder\nHolder.next=Holder\nnp._review4_holder.next.m()\n'),'decline'),
 'attr_intermediate_function':(P(prefix=W+'np._review4_holder=change\nnp._review4_holder.m=change\nnp._review4_holder.m()\n'),'decline'),
 'attr_untouched':(P(prefix=HOLDER+'np._review4_holder=Holder\nnp.random.seed(0)\n'),'lower'),
 'data_reassign_plain':(post('X=k(np.ones((1,2)))\nX=Box()\nSIDE=int(X.data)\n'),'safe'),
 'data_reassign_tuple':(post('X=k(np.ones((1,2)))\nX,unused=Box(),0\nSIDE=int(X.data)\n'),'safe'),
 'data_reassign_ann':(post('X=k(np.ones((1,2)))\nX:object=Box()\nSIDE=int(X.data)\n'),'safe'),
 'data_reassign_walrus':(post('X=k(np.ones((1,2)))\n(X:=Box())\nSIDE=int(X.data)\n'),'safe'),
 'data_conditional_binding':(post('if False:\n    X=k(np.ones((1,2)))\nelse:\n    X=Box()\nSIDE=int(X.data)\n'),'safe'),
 'data_class_scope_leak':(post('X=Box()\nclass Snapshot:\n    X=k(np.ones((1,2)))\nSIDE=int(X.data)\n'),'safe'),
 'data_lambda_shadow':(post('X=k(np.ones((1,2)))\nread=lambda X:int(X.data)\nSIDE=read(Box())\n'),'safe'),
 'data_comprehension_shadow':(post('X=k(np.ones((1,2)))\nSIDE=[int(X.data) for X in (Box(),)][0]\n'),'safe'),
 'data_builtin_next':(post('SIDE=int(isinstance(next(iter([k(np.ones((1,2)))])).data,np.ndarray))\n',prefix=''),'safe'),
 'data_builtin_getattr':(post('class Store:\n    X=k(np.ones((1,2)))\nSIDE=int(isinstance(getattr(Store,"X").data,np.ndarray))\n',prefix=''),'safe'),
 'data_class_new_returns_dsl':(post('class Factory:\n    def __new__(cls):\n        return k(np.ones((1,2)))\nSIDE=int(isinstance(Factory().data,np.ndarray))\n',prefix=''),'safe'),
 'data_plain_container_mutation':(post('X=[]\nX.append(k(np.ones((1,2))))\nSIDE=int(isinstance(X[0].data,np.ndarray))\n',prefix=''),'safe'),
 'data_plain_object_field':(post('box=Box()\nbox.tensor=k(np.ones((1,2)))\nSIDE=int(isinstance(box.tensor.data,np.ndarray))\n'),'safe'),
 'data_function_body':(post('SIDE=read(Box())\n',prefix=BOX+'def read(x):\n    return int(x.data)\n'),'safe'),
 'data_method_body':(post('SIDE=Reader().read(Box())\n',prefix=BOX+'class Reader:\n    def read(self,x):\n        return int(x.data)\n'),'safe'),
 'data_known_dsl_alias':(post('X=k(np.ones((1,2)))\nY=X\nSIDE=int(isinstance(Y.data,np.ndarray))\n',prefix=''),'lower'),
 'data_plain_numeric':(post('SIDE=int(Box().data)+int(np.sum(np.array([2,3])))\n'),'lower'),
}

def observe(name,fp64):
 code,expected=CASES[name];reasons=[];fallback=[]
 try:compile(code,'<round7-probe>','exec',dont_inherit=True);compile_error=None
 except Exception as e:compile_error={'type':type(e).__name__,'message':str(e)}
 try:low=na.numba_lower(code,use_fp64=fp64,on_decline=reasons.append);lower_error=None
 except Exception as e:low=None;lower_error={'type':type(e).__name__,'message':str(e)}
 inputs={'a':np.full((1,2),2.003,dtype=np.float64)}
 def call(f):
  sentinel=object();saved=[(m,k,getattr(m,k,sentinel)) for m in [np,np.random,sys.modules[BRIDGE]] for k in ['_review4_change','_review4_holder','run_model']];rng=np.random.get_state()
  try:return old.isolated_call(f)
  finally:
   for m,k,v in saved:
    if v is sentinel:vars(m).pop(k,None)
    else:setattr(m,k,v)
   np.random.set_state(rng)
 slow=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=False));raw=call(lambda:na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append));public=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
 row=dict(name=name,fp64=fp64,expected=expected,code=code,compile_error=compile_error,lower_error=lower_error,lowered=isinstance(low,str),declines=reasons,fallback=fallback,slow=slow,raw=raw,public=public)
 out=R/'runs'/('round7-'+os.environ['REVIEW_ROLE']);out.mkdir(exist_ok=True);stem=name+'-'+str(fp64)
 (out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n');(out/(stem+'.py')).write_text(code)
 if isinstance(low,str):(out/(stem+'.lowered.py')).write_text(low)
 return row

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_round7_boundaries(name,fp64):
 r=observe(name,fp64);assert r['compile_error'] is None,r
 if 'error' in r['slow']:assert r['public'].get('error')==r['slow']['error'],r
 else:
  assert 'value' in r['public'],r
  for k,v in r['slow']['value'].items():
   want=np.asarray(v['data'])
   if not fp64 and not r['fallback']:want=want.astype(np.float32)
   np.testing.assert_array_equal(r['public']['value'][k]['data'],want,err_msg=str(r))
 if r['expected']=='decline':assert not r['lowered'] and any(s.startswith('kernel_') for s in r['declines']),r
 elif r['expected']=='lower':assert r['lowered'] and not r['declines'] and not r['fallback'],r
 # safe: a conservative refusal is acceptable; a changed public result is not.
