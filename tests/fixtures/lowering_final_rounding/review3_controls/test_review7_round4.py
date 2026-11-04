import ast, importlib.util, json, os, pathlib
import numpy as np
import pytest
from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import numba_lower, execute_fast, _module_string_constants
from solar.einsum_lib.verify import execute_translated

C=pathlib.Path(__file__).resolve().parents[1]
OUT=C/'evidence/adversarial'; OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('rounding_tests',pathlib.Path(os.environ['REVIEW_SOURCE'])/'tests/test_lowering_final_rounding.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
HEADER='from solar.einsum_lib import Tensor, Dim, kernel, clear_dims\nDT="bf16"\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n'
RUN="""\ndef run_model(inputs):
    clear_dims()
    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')
    y=k(a)
    return {'output':y.data}
"""
ALLOC='Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)'
cases={
'nested_try_loop_continue':(HEADER+K+"""    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    for i in range(1):
        try:
            try:
                Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
                continue
            except Exception:
                pass
        except Exception:
            pass
    return Y
"""+RUN,'kernel_output_dtype_conflict'),
'finally_rebind':(HEADER+K+"""    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    try:
        pass
    finally:
        Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
    return Y
"""+RUN,'kernel_output_dtype_unestablished'),
'class_nested_function_global':(HEADER+"""class C:
    def rebind():
        global DT
        DT='fp32'
C.rebind()
"""+K+'    Y='+ALLOC+'\n    return Y\n'+RUN,'kernel_output_dtype_unestablished'),
'conditional_class_global':(HEADER+"""if True:
    class C:
        global DT
        if True:
            DT='fp32'
"""+K+'    Y='+ALLOC+'\n    return Y\n'+RUN,'kernel_output_dtype_unestablished'),
'default_expr_write':(HEADER+'def side(x=(DT := "fp32")):\n    pass\n'+K+'    Y='+ALLOC+'\n    return Y\n'+RUN,None),
'annotation_expr_write':(HEADER+'def side(x: (DT := "fp32")):\n    pass\n'+K+'    Y='+ALLOC+'\n    return Y\n'+RUN,None),
'class_base_expr_write':(HEADER+'class C((DT := "fp32") and object):\n    pass\n'+K+'    Y='+ALLOC+'\n    return Y\n'+RUN,None),
'raise_expr_alias':(HEADER+K+'''    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    Z=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
    try:
        raise ValueError((Y := Z).shape[0])
    except Exception:
        pass
    return Y
'''+RUN,None),
'raise_expr_write':(HEADER+K+"""    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    try:
        raise ValueError((Y := Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)).shape[0])
    except Exception:
        pass
    return Y
"""+RUN,None),
}

@pytest.mark.parametrize('use_fp64',[True,False])
@pytest.mark.parametrize('case',list(cases))
def test_semantic_control(case,use_fp64):
    code,expect=cases[case];clear_dims();stop_trace()
    (OUT/(case+'.py')).write_text(code)
    x=np.array([[1.003,1.009]],np.float64)
    slow,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=False)
    reasons=[];fallback=[]
    low=numba_lower(code,use_fp64=use_fp64,boundscheck=True,on_decline=reasons.append)
    if low:(OUT/(case+f'-{use_fp64}.lowered.py')).write_text(low)
    fast=execute_fast(code,{'a':x},use_fp64=use_fp64,on_fallback=fallback.append)
    record={'case':case,'fp64':use_fp64,'consts':_module_string_constants(ast.parse(code)),
        'lowered':low is not None,'declines':reasons,'fallback':fallback,
        'slow':np.asarray(slow['output']).tolist(),'fast':None if fast is None else np.asarray(fast['output']).tolist()}
    (OUT/(case+f'-{use_fp64}.json')).write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
    if expect:
        assert low is None and reasons and reasons[0].startswith(expect)
    if low is None:
        assert reasons and reasons[0].startswith('kernel_output_dtype_')
        assert fast is None
        fallen,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=True)
        np.testing.assert_array_equal(fallen['output'],slow['output'])
    else:
        assert fast is not None, fallback
        if use_fp64:
            np.testing.assert_array_equal(fast['output'],slow['output'])
        else:
            np.testing.assert_array_equal(fast['output'],np.asarray(slow['output'],dtype=np.float32))

BASE="for m in numba.prange(n):\n    s += m\n"
REWRITE="for _m_prange in numba.prange(n):\n    m = np.intp(_m_prange)\n    s += m\n"
@pytest.mark.parametrize('extra',[
    '    m = np.intp(m + 1)\n',
    '    s += _m_prange\n',
])
def test_normalizer_preserves_extra_body_changes(extra):
    mutant=REWRITE+extra
    assert t._normalise_prange_rebind(mutant)!=t._normalise_prange_rebind(BASE)
    if '_m_prange' in extra:
        assert t._normalise_prange_rebind(mutant)==ast.unparse(ast.parse(mutant))

def test_normalizer_fresh_used_after_loop():
    before="def f():\n    s=0\n    _m_prange=100\n    for m in numba.prange(3):\n        s+=m\n    return s+_m_prange\n"
    after=before.replace('for m in numba.prange(3):\n        s+=m','for _m_prange in numba.prange(3):\n        m=np.intp(_m_prange)\n        s+=m')
    import numba
    a={'np':np,'numba':numba};b=dict(a);exec(before,a);exec(after,b)
    norms=[t._normalise_prange_rebind(x) for x in [before,after]]
    (OUT/'normalizer-after-loop.json').write_text(json.dumps({'before':before,'after':after,'before_value':a['f'](),'after_value':int(b['f']()),'normalized':norms},indent=2))
    assert a['f']()==103 and b['f']()==5
    assert norms[0]!=norms[1]

