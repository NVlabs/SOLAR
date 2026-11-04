import ast,importlib.util,json,os,pathlib,textwrap
import numpy as np
import numba
import pytest
from solar.einsum_lib import clear_dims,stop_trace
from solar.einsum_lib.numba_accel import numba_lower,execute_fast,_module_string_constants,_syntactic_writes
from solar.einsum_lib.verify import execute_translated

C=pathlib.Path(__file__).resolve().parents[1]
OUT=C/'evidence/new-controls';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('head_rounding_tests',pathlib.Path(os.environ['REVIEW_SOURCE'])/'tests/test_lowering_final_rounding.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
H='from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n'
RUN="""\ndef run_model(inputs):
    clear_dims()
    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')
    y=k(a)
    return {'output':y.data}
"""
ALLOC='    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)\n    return Y\n'
def const(prefix, alloc=ALLOC):
    return H+prefix+K+alloc+RUN
INIT="""    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    Z=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
"""
def flow(body,prefix=''):
    return H+prefix+K+INIT+body+'    return Y\n'+RUN

CASES={
 'exec_write':(const('DT="bf16"\nexec("DT = \'fp32\'")\n'),None),
 'eval_write':(const('DT="bf16"\neval("globals().__setitem__(\'DT\', \'fp32\')")\n'),None),
 'globals_subscript':(const('DT="bf16"\nglobals()["DT"]="fp32"\n'),None),
 'globals_alias_update':(const('DT="bf16"\nns=globals()\nns.update(DT="fp32")\n'),None),
 'setattr_module':(const('DT="bf16"\nimport types\nmodule=types.ModuleType("review_private_module")\nmodule.__dict__.update(globals())\nsetattr(module,"DT","fp32")\nglobals().update(vars(module))\n'),None),
 'delete_reassign':(const('DT="bf16"\ndel DT\nDT="fp32"\n'),'unestablished'),
 'import_reassign':(const('DT="bf16"\nimport math as DT\nDT="fp32"\n'),'unestablished'),
 'annassign_only':(const('DT: str = "bf16"\n'),'unestablished'),
 'attribute_store_positive':(const('DT="bf16"\nclass C:\n    pass\nC.DT="fp32"\nread=DT\n'),'lower'),
 'module_getattr_hook':(const('import types\nmodule=types.ModuleType("review_private_hook")\nmodule.__getattr__=lambda name:"fp32"\n',ALLOC.replace('dtype=DT','dtype=module.DT')),'unestablished'),
 'with_walrus_in_try':(flow('    try:\n        with nullcontext():\n            _ = (Y := Z)\n    except Exception:\n        pass\n','from contextlib import nullcontext\n'),'expression'),
 'finally_walrus':(flow('    try:\n        pass\n    finally:\n        _ = (Y := Z)\n'),'unestablished'),
 'comprehension_walrus_in_try':(flow('    try:\n        values = [(Y := Z) for i in range(1)]\n    except Exception:\n        pass\n'),'expression'),
 'except_augassign_dtype':(H+K+'    DT="bf"\n    try:\n        raise ValueError("t")\n    except Exception:\n        DT += "16"\n'+ALLOC+RUN,'unestablished'),
 'except_type_walrus_tuple':(flow('    try:\n        raise ValueError("t")\n    except ((Y := Z), Exception)[1]:\n        pass\n'),'expression'),
 'except_type_walrus_conditional':(flow('    try:\n        raise ValueError("t")\n    except (Exception if (Y := Z) is not None else ValueError):\n        pass\n'),'expression'),
}
def observe(case,fp64):
    code,expect=CASES[case];clear_dims();stop_trace()
    (OUT/(case+'.py')).write_text(code)
    x=np.array([[1.003,1.009]],np.float64)
    slow,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=False)
    reasons=[];fallback=[]
    low=numba_lower(code,use_fp64=fp64,boundscheck=True,on_decline=reasons.append)
    fast=execute_fast(code,{'a':x},use_fp64=fp64,on_fallback=fallback.append)
    fallen,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=True,use_fp64=fp64)
    if low:(OUT/(case+f'-{fp64}.lowered.py')).write_text(low)
    record=dict(case=case,fp64=fp64,consts=_module_string_constants(ast.parse(code)),writes=_syntactic_writes(ast.parse(code)),lowered=low is not None,declines=reasons,fallback=fallback,slow=np.asarray(slow['output']).tolist(),fast=None if fast is None else np.asarray(fast['output']).tolist(),public_fast_or_fallback=np.asarray(fallen['output']).tolist())
    (OUT/(case+f'-{fp64}.json')).write_text(json.dumps(record,indent=2)+'\n')
    return record
@pytest.mark.parametrize('fp64',[True,False])
@pytest.mark.parametrize('case',list(CASES))
def test_semantics_and_class_declines(case,fp64):
    r=observe(case,fp64)
    expect=CASES[case][1]
    if expect=='lower':
        assert r['lowered'] and r['fast'] is not None and not r['declines']
    elif expect:
        assert not r['lowered'] and r['declines'] and r['declines'][0].startswith('kernel_output_dtype_unestablished:k'),r
        if expect=='expression':assert 'exception/expression provenance not modelled' in r['declines'][0],r
    if r['lowered'] and r['fast'] is not None:
        np.testing.assert_array_equal(np.asarray(r['fast']),np.asarray(r['slow'],dtype=np.float64 if fp64 else np.float32))
    else:
        np.testing.assert_array_equal(r['public_fast_or_fallback'],r['slow'])

def test_import_DT_and_annassign_syntactic_rule():
    assert _syntactic_writes(ast.parse('DT="bf16"\nimport DT\n'))['DT']==2
    assert 'DT' not in _module_string_constants(ast.parse('DT="bf16"\nimport DT\n'))
    assert _syntactic_writes(ast.parse('DT: str = "bf16"'))['DT']==1
    assert 'DT' not in _module_string_constants(ast.parse('DT: str = "bf16"'))

LOOP='    for m in numba.prange(3):\n        s += m\n'
NEW='    for _m_prange in numba.prange(3):\n        m=np.intp(_m_prange)\n        s += m\n'
def loop_program(pre='',post='    return s\n',module=''):
    return module+'def f():\n    s=0\n'+pre+LOOP+post
NORMAL={
 'nested_function_read':loop_program('    def inner():\n        return _m_prange\n'),
 'default_argument_read':loop_program('    _m_prange=100\n    def inner(x=_m_prange):\n        return x\n'),
 'global_declaration':loop_program('    global _m_prange\n','    return s+globals()["_m_prange"]\n','_m_prange=100\n'),
 'import_alias':loop_program('    import math as _m_prange\n','    return s, type(locals()["_m_prange"]).__name__\n'),
 'function_name':loop_program('    def _m_prange():\n        return 100\n','    return s, type(locals()["_m_prange"]).__name__\n'),
 'match_capture':loop_program('    match 100:\n        case _m_prange:\n            pass\n','    return s+locals()["_m_prange"]\n'),
 'keyword_argument_name':loop_program('    def sink(**kwargs):\n        return kwargs["_m_prange"]\n','    return sink(_m_prange=s)\n'),
}
# A keyword label is not a local name; normalization is safe for that positive.
@pytest.mark.parametrize('case',list(NORMAL))
def test_normalizer_binding_and_usage_classes(case):
    before=NORMAL[case];after=before.replace(LOOP,NEW)
    a={'np':np,'numba':numba};b=dict(a)
    values=None
    if case not in {'nested_function_read'}:
        exec(before,a);exec(after,b)
        values=[a['f'](),b['f']()]
    norm=[t._normalise_prange_rebind(s) for s in [before,after]]
    def clean(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        return v
    (OUT/('normalizer-'+case+'.json')).write_text(json.dumps(dict(before=before,after=after,values=clean(values),normalized=norm),indent=2)+'\n')
    if case=='keyword_argument_name':
        assert norm[0]==norm[1] and values[0]==values[1]
    else:
        assert norm[0]!=norm[1]
        assert norm[1]==ast.unparse(ast.parse(after))

def test_normalizer_nonlocal_declaration():
    before='def f():\n    _m_prange=100\n    def inner():\n        nonlocal _m_prange\n        s=0\n        for m in numba.prange(3):\n            s+=m\n        return s\n    s=inner()\n    return s+_m_prange\n'
    after=before.replace('for m in numba.prange(3):\n            s+=m','for _m_prange in numba.prange(3):\n            m=np.intp(_m_prange)\n            s+=m')
    a={'np':np,'numba':numba};b=dict(a);exec(before,a);exec(after,b)
    values=[int(a['f']()),int(b['f']())]
    norm=[t._normalise_prange_rebind(s) for s in [before,after]]
    (OUT/'normalizer-nonlocal.json').write_text(json.dumps(dict(before=before,after=after,values=values,normalized=norm),indent=2)+'\n')
    assert values==[103,5]
    assert norm[0]!=norm[1]

