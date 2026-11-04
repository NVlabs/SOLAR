import ast,json,pathlib,os,importlib.util
import numpy as np
import numba
import pytest
from solar.einsum_lib import clear_dims,stop_trace
from solar.einsum_lib.numba_accel import numba_lower,execute_fast,_module_string_constants,_module_dynamic_namespace
from solar.einsum_lib.verify import execute_translated
C=pathlib.Path(__file__).resolve().parent
OUT=C/'evidence/new-controls';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('rounding_tests',pathlib.Path(os.environ['REVIEW_SOURCE'])/'tests/test_lowering_final_rounding.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
H='from solar.einsum_lib import Tensor, kernel, clear_dims\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n'
ALLOC='    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)\n    return Y\n'
RUN="""\ndef run_model(inputs):
    clear_dims()
    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')
    y=k(a)
    return {'output':y.data}
"""
REFLECT={
'getattr_function_globals':'def marker():\n    pass\ngetattr(marker, "__globals__")["DT"]="fp32"\n',
'getattribute_function_globals':'def marker():\n    pass\nmarker.__getattribute__("__globals__")["DT"]="fp32"\n',
'object_getattribute_globals':'def marker():\n    pass\nobject.__getattribute__(marker, "__globals__")["DT"]="fp32"\n',
'computed_attribute_string':'def marker():\n    pass\ngetattr(marker, "__glo"+"bals__").update({"DT":"fp32"})\n',
'generator_frame_globals':'g=(x for x in [])\ng.gi_frame.f_globals["DT"]="fp32"\n',
'decorator_injected_globals':'def inject(f):\n    getattr(f,"__globals__")["DT"]="fp32"\n    return f\n@inject\ndef marker():\n    pass\n',
'class_body_reflection':'class C:\n    getattr(lambda:0,"__globals__")["DT"]="fp32"\n',
}
def observe(name,code,fp64):
 clear_dims();stop_trace();x=np.array([[1.003,1.009]],np.float64)
 (OUT/(name+'.py')).write_text(code)
 slow,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=False)
 reasons=[];fallback=[];low=numba_lower(code,boundscheck=True,use_fp64=fp64,on_decline=reasons.append)
 fast=execute_fast(code,{'a':x},use_fp64=fp64,on_fallback=fallback.append)
 public,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=True,use_fp64=fp64)
 if low:(OUT/(name+'-'+str(fp64)+'.lowered.py')).write_text(low)
 r={'name':name,'fp64':fp64,'dynamic':_module_dynamic_namespace(ast.parse(code)),'constants':_module_string_constants(ast.parse(code)),'lowered':low is not None,'declines':reasons,'fallbacks':fallback,'slow':np.asarray(slow['output']).tolist(),'fast':None if fast is None else np.asarray(fast['output']).tolist(),'public':np.asarray(public['output']).tolist()}
 (OUT/(name+'-'+str(fp64)+'.json')).write_text(json.dumps(r,indent=2)+'\n')
 return r
@pytest.mark.parametrize('fp64',[True,False])
@pytest.mark.parametrize('name',list(REFLECT))
def test_dynamic_dtype_reflection_must_not_silently_misround(name,fp64):
 code=H+'DT="bf16"\n'+REFLECT[name]+K+ALLOC+RUN
 r=observe(name,code,fp64)
 assert np.array_equal(np.asarray(r['public']),np.asarray(r['slow'],dtype=np.float64 if fp64 else np.float32)),r
 assert r['lowered'] is False and r['declines'][0].startswith('kernel_output_dtype_unestablished:k'),r

CAUGHT={
'sys_alias':'import sys as runtime\nruntime.modules[__name__].DT="fp32"\n',
'getattr_builtins_exec':'import builtins\ngetattr(builtins,"exec")("DT=\'fp32\'")\n',
'builtins_import_alias':'from builtins import exec as run\nrun("DT=\'fp32\'")\n',
'module_type_setattr':'import types\nm=types.ModuleType("x")\nsetattr(m,"DT","fp32")\n',
'class_globals':'class C:\n    globals()["DT"]="fp32"\n',
'function_attr_globals':'def marker():\n    pass\nmarker.__globals__["DT"]="fp32"\n',
}
@pytest.mark.parametrize('name',list(CAUGHT))
def test_declared_dynamic_patterns_remain_typed(name):
 code=H+'DT="bf16"\n'+CAUGHT[name]+K+ALLOC+RUN
 reasons=[];low=numba_lower(code,on_decline=reasons.append)
 assert _module_dynamic_namespace(ast.parse(code)) is not None and low is None and reasons[0].startswith('kernel_output_dtype_unestablished:k')
 (OUT/('caught-'+name+'.json')).write_text(json.dumps({'dynamic':_module_dynamic_namespace(ast.parse(code)),'declines':reasons},indent=2)+'\n')

INIT='    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype="fp32",fill=1.003)\n    Z=Tensor(M=A.shape[0],N=A.shape[1],dtype="bf16",fill=1.003)\n'
HANDLERS={
'starred_tuple':'(*errors,)',
'parenthesized_walrus':'(E := Exception)',
'helper_call':'pick()',
'call_in_tuple':'(ValueError,pick())',
'conditional_no_walrus':'(ValueError if True else TypeError)',
'walrus_lambda_default':'(lambda e=(Y := Z): Exception)()',
}
@pytest.mark.parametrize('fp64',[True,False])
@pytest.mark.parametrize('name',list(HANDLERS))
def test_extra_handler_expressions_decline_typed(name,fp64):
 body=INIT+'    errors=(ValueError,)\n    pick=lambda:Exception\n    try:\n        raise ValueError("synthetic")\n    except '+HANDLERS[name]+':\n        pass\n    return Y\n'
 r=observe('handler-'+name,H+K+body+RUN,fp64)
 assert not r['lowered'] and r['declines'] and r['declines'][0].startswith('kernel_output_dtype_unestablished:k') and 'exception/expression provenance not modelled' in r['declines'][0],r
 np.testing.assert_array_equal(r['public'],r['slow'])

LOOP='    for m in numba.prange(3):\n        s += m\n'
NEW='    for _m_prange in numba.prange(3):\n        m=np.intp(_m_prange)\n        s += m\n'
def program(post,pre=''):
 return 'def f():\n    s=0\n'+pre+LOOP+post
NORM={
'locals_string_key':program('    return s+locals().get("_m_prange",100)\n'),
'vars_string_key':program('    return s+vars().get("_m_prange",100)\n'),
'fstring_locals_lookup':program('    return f"{s+locals().get(\'_m_prange\',100)}"\n'),
'format_map_locals':program('    return "{_m_prange}".format_map(dict({"_m_prange":100},**locals()))\n'),
'fstring_identifier':program('    return f"{_m_prange}"\n'),
'fstring_walrus':program('    return s\n','    note=f"{(_m_prange := 100)}"\n'),
'class_annotation_string':('def f():\n    class C:\n        __annotations__={"_m_prange":int}\n        s=0\n        for m in numba.prange(3):\n            s+=m\n    return C.s+getattr(C,"_m_prange",100)\n'),
'comment_positive':program('    return s\n','    # _m_prange is only a comment\n'),
'string_positive':program('    return s\n','    note="_m_prange"\n'),
'keyword_overdecline':program('    return sink(_m_prange=s)\n','    def sink(**kw):\n        return kw["_m_prange"]\n'),
}
@pytest.mark.parametrize('name',list(NORM))
def test_normalizer_reflection_fstrings_and_safe_strings(name):
 before=NORM[name]
 after=before.replace(LOOP,NEW)
 if name=='class_annotation_string':
  after=before.replace('for m in numba.prange(3):\n            s+=m','for _m_prange in numba.prange(3):\n            m=np.intp(_m_prange)\n            s+=m')
 assert before!=after
 values=[]
 for src in [before,after]:
  ns={'np':np,'numba':numba}
  try:
   exec(src,ns);v=ns['f']();values.append(v.item() if isinstance(v,np.generic) else v)
  except Exception as e:values.append(type(e).__name__+': '+str(e))
 norms=[t._normalise_prange_rebind(src) for src in [before,after]]
 data={'before':before,'after':after,'values':values,'normalizations':norms,'after_identifiers':list(t._identifiers(ast.parse(after))).count('_m_prange'),'after_tokens':t._name_token_count(after,'_m_prange')}
 (OUT/('normalizer-'+name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 if name in {'comment_positive','string_positive'}:
  assert values[0]==values[1] and norms[0]==norms[1]
 else:
  assert norms[0]!=norms[1],data
  if name=='keyword_overdecline':assert values[0]==values[1]

