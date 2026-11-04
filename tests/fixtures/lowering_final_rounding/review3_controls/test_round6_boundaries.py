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
CASES={
 'entry_globals':(P(run=GOOD+BAD+'globals()["run_model"]=runner\n'),'decline'),
 'entry_setattr':(P(run=GOOD+BAD+'import sys\n__name__="review4_r6_bridge"\nsetattr(sys.modules[__name__],"run_model",runner)\nrun_model=good\n'),'decline'),
 'entry_conditional':(P(run=GOOD+BAD+'if True:\n    run_model=runner\nelse:\n    run_model=good\n'),'decline'),
 'entry_call_result':(P(run=GOOD+BAD+'run_model=(lambda: runner)()\n'),'decline'),
 'entry_same_line_subscript':(P(run=GOOD+BAD+'run_model=good; run_model=(runner,)[0]\n'),'decline'),
 'entry_same_line_annassign':(P(run=GOOD+BAD+'run_model=good; run_model: object=runner\n'),'decline'),
 'entry_same_line_tuple':(P(run=GOOD+BAD+'run_model=good; run_model,other=runner,0\n'),'decline'),
 'entry_same_line_walrus':(P(run=GOOD+BAD+'run_model=good; (run_model:=runner)\n'),'decline'),
 'entry_separate_subscript':(P(run=GOOD+BAD+'run_model=good\nrun_model=(runner,)[0]\n'),'decline'),
 'entry_alias_safe':(P(run=GOOD+'one=good\ntwo=one\nrun_model=two\n'),'lower'),
 'entry_alias_cycle':(P(run=GOOD+'one=good\ntwo=one\none=two\nrun_model=one\n'),'decline'),
 'entry_del':(P(run=r4.RUN+'del run_model\n'),'decline'),
 'entry_unbound':(P(run=''),'decline'),
 'attr_store_root_alias':(P(prefix=W+'other=np\nother._review4_change=change\nnp._review4_change()\n'),'decline'),
 'attr_setattr':(P(prefix=W+'setattr(np,"_review4_change",change)\nnp._review4_change()\n'),'decline'),
 'attr_install_in_run':(P(prefix=W+'def install():\n    np.random._review4_change=change\n',run=r4.RUN.replace('    clear_dims()','    install()\n    np.random._review4_change()\n    clear_dims()')),'decline'),
 'attr_intermediate':(P(prefix=HOLDER+'np._review4_holder=Holder\nnp._review4_holder.m()\n'),'decline'),
 'attr_intermediate_alias':(P(prefix=HOLDER+'other=np\nother._review4_holder=Holder\nnp._review4_holder.m()\n'),'decline'),
 'attr_intermediate_in_run':(P(prefix=HOLDER+'def install():\n    np._review4_holder=Holder\n',run=r4.RUN.replace('    clear_dims()','    install()\n    np._review4_holder.m()\n    clear_dims()')),'decline'),
 'attr_intermediate_uncalled':(P(prefix=HOLDER+'np._review4_holder=Holder\n'),'lower'),
 'attr_ordinary_import':(P(prefix=W+'np.random.seed(0)\n'),'lower'),
 'attr_same_name_overapprox':(P(prefix=W+'class C:\n    pass\nC.zeros=change\nnp.zeros((1,))\n'),'safe'),
 'module_object_data':(P(prefix=BOX+'SIDE=int(Box().data)\n',run=SIDE),'safe'),
 'module_alias_data':(P(prefix=BOX+'box=Box()\nSIDE=int(box.data)\n',run=SIDE),'safe'),
 'module_lambda_data':(P(prefix=BOX+'read=lambda x:int(x.data)\nSIDE=read(Box())\n',run=SIDE),'safe'),
 'module_class_body_data':(P(prefix=BOX+'class Snapshot:\n    SIDE=int(Box().data)\nSIDE=Snapshot.SIDE\n',run=SIDE),'safe'),
 'module_numpy_data':(P(prefix='SIDE=int(isinstance(np.arange(3).data,memoryview))\n',run=SIDE),'safe'),
 'module_ordinary_numeric':(P(prefix='SIDE=float(np.sum(np.array([1.25,2.5])))\n',run=SIDE),'lower'),
 'module_importtime_kernel':(P(run='\nFIRST=k(np.ones((1,2)))\n'+r4.RUN.replace('k(inputs["a"]).data','FIRST.data')),'lower'),
 'module_class_kernel':(P(run='\nclass Runner:\n    def go(self,inputs):\n        return {"out":k(inputs["a"]).data}\n'+r4.RUN.replace('return {"out":k(inputs["a"]).data}','return Runner().go(inputs)')),'lower'),
}

def observe(name,fp64):
 code,expected=CASES[name];reasons=[];fallback=[]
 try:compile(code,'<round6-probe>','exec',dont_inherit=True);compile_error=None
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
 out=R/'runs'/('round6-'+os.environ['REVIEW_ROLE']);out.mkdir(exist_ok=True);stem=name+'-'+str(fp64)
 (out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n');(out/(stem+'.py')).write_text(code)
 if isinstance(low,str):(out/(stem+'.lowered.py')).write_text(low)
 return row

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_round6_boundaries(name,fp64):
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
