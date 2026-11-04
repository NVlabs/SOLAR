# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round5 effect reachability probes; no edits to inherited tests."""
import ast,json,os,math
from pathlib import Path
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
import test_round3_scope as old
import test_round4_scope as r4
R=Path(__file__).parent
P=r4.program;W=r4.WRITER
BADRUN='\ndef runner(inputs):\n    global Tensor\n    Tensor=int\n    clear_dims()\n    return {"out":k(inputs["a"]).data}\n'
CASES={
 'harness_alias':(P(run=BADRUN+'run_model=runner\n'),'decline'),
 'harness_alias_chain':(P(run=BADRUN+'second=runner\nrun_model=second\n'),'decline'),
 'harness_lambda':(P(prefix=W,run='\nrun_model=lambda inputs: (change(),clear_dims(),{"out":k(inputs["a"]).data})[-1]\n'),'decline'),
 'harness_rebound':(P(run=r4.RUN+BADRUN+'run_model=runner\n'),'decline'),
 'harness_class_static':(P(run='\nclass Runner:\n    @staticmethod\n    def run(inputs):\n        global Tensor\n        Tensor=int\n        clear_dims()\n        return {"out":k(inputs["a"]).data}\nrun_model=Runner.run\n'),'decline'),
 'import_attribute_writer':(P(prefix=W+'np._review4_change=change\nnp._review4_change()\n'),'decline'),
 'import_attribute_lambda':(P(prefix='np._review4_change=lambda: globals().__setitem__("Tensor",int)\nnp._review4_change()\n'),'decline'),
 'import_attribute_nested':(P(prefix=W+'np.random._review4_change=change\nnp.random._review4_change()\n'),'decline'),
 'import_attribute_via_alias':(P(prefix=W+'np._review4_change=change\nother=np\nother._review4_change()\n'),'decline'),
 'import_attribute_writer_in_run':(P(prefix=W+'np._review4_change=change\n',run=r4.RUN.replace('    clear_dims()','    np._review4_change()\n    clear_dims()')),'decline'),
 'ordinary_import_safe':(P(prefix=W+'unused=np.zeros((1,))\n'),'lower'),
 'uncalled_import_attribute_writer':(P(prefix=W+'np._review4_change=change\n'),'lower'),
 'callback_argument':(P(prefix=W+'def call(f):\n    f()\ncall(change)\n'),'decline'),
 'callback_lambda_argument':(P(prefix='def call(f):\n    f()\ncall(lambda: globals().__setitem__("Tensor",int))\n'),'decline'),
 'called_class_property':(P(prefix='class C:\n    @property\n    def x(self):\n        global Tensor\n        Tensor=int\nc=C()\nc.x\n'),'decline'),
 'called_class_base':(P(prefix='class Base:\n    def __init_subclass__(cls):\n        global Tensor\n        Tensor=int\nclass Child(Base):\n    pass\n'),'decline'),
 'comp_writer_callable':(P(prefix=W+'unused=[f() for f in (change,)]\n'),'decline'),
 'comp_shadow_known_writer':(P(prefix=W+'unused=[change for change in (1,2)]\nchange()\n'),'decline'),
 'comp_safe_with_capability':(P(prefix=W+'unused=[f() for f in (lambda: 0,)]\n'),'decline'),
 'comp_safe_without_capability':(P(prefix='unused=[f() for f in (lambda: 0,)]\n'),'lower'),
 'future_doc_multiple':(P(prefix=r4.MARK,header='"doc"\nfrom __future__ import division\n'+r4.FUTURE+r4.H,run=r4.SIDE),'lower'),
 'future_default_writer':(P(prefix=W+'def unused(x=change()):\n    pass\n',header=r4.FUTURE+r4.H),'decline'),
 'harness_safe_alias':(P(run=r4.RUN.replace('def run_model','def runner')+'run_model=runner\n'),'lower'),
 'harness_safe_lambda':(P(run='\nrun_model=lambda inputs: (clear_dims(),{"out":k(inputs["a"]).data})[-1]\n'),'lower'),
}

def observe(name,fp64):
 code,expected=CASES[name];reasons=[];fallback=[]
 try: compile(code,'<round5-effects>','exec',dont_inherit=True);compile_error=None
 except Exception as e: compile_error={'type':type(e).__name__,'message':str(e)}
 try: low=na.numba_lower(code,use_fp64=fp64,on_decline=reasons.append);lower_error=None
 except Exception as e: low=None;lower_error={'type':type(e).__name__,'message':str(e)}
 inputs={'a':np.full((1,2),2.003,dtype=np.float64)}
 def call(f):
  sentinel=object();saved=[(m,getattr(m,'_review4_change',sentinel)) for m in (np,np.random,math)]
  try:return old.isolated_call(f)
  finally:
   for m,val in saved:
    if val is sentinel: vars(m).pop('_review4_change',None)
    else: setattr(m,'_review4_change',val)
 slow=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=False))
 raw=call(lambda:na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append))
 public=call(lambda:execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
 row=dict(name=name,fp64=fp64,expected=expected,code=code,compile_error=compile_error,lower_error=lower_error,lowered=isinstance(low,str),declines=reasons,fallback=fallback,slow=slow,raw=raw,public=public)
 out=R/'runs'/('round5-'+os.environ['REVIEW_ROLE']);out.mkdir(exist_ok=True);stem=name+'-'+str(fp64)
 (out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n');(out/(stem+'.py')).write_text(code)
 if isinstance(low,str):(out/(stem+'.lowered.py')).write_text(low)
 return row

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_round5_effects(name,fp64):
 r=observe(name,fp64);assert r['compile_error'] is None,r
 if 'error' in r['slow']: assert r['public'].get('error')==r['slow']['error'],r
 else:
  assert 'value' in r['public'],r
  for key,v in r['slow']['value'].items():
   want=np.asarray(v['data'])
   if not fp64 and not r['fallback']:want=want.astype(np.float32)
   np.testing.assert_array_equal(r['public']['value'][key]['data'],want,err_msg=str(r))
 if r['expected']=='decline': assert not r['lowered'] and any(s.startswith('kernel_') for s in r['declines']),r
 elif r['expected']=='lower': assert r['lowered'] and not r['declines'] and not r['fallback'],r
