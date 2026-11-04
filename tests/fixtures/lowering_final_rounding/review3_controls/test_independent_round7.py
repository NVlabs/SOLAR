"""Independent round7 allowlist / normalizer controls; only private evidence is written."""
import ast,copy,importlib.util,json,os,pathlib
import numpy as np,numba,pytest
from solar.einsum_lib import clear_dims,stop_trace
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
R=pathlib.Path(__file__).parent
OUT=R/"runs/independent";OUT.mkdir(parents=True,exist_ok=True)
SOURCE=pathlib.Path(os.environ["REVIEW_SOURCE"])
spec=importlib.util.spec_from_file_location("head_rounding",SOURCE/"tests/test_lowering_final_rounding.py")
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
H='from solar.einsum_lib import Tensor, kernel, clear_dims\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)\n    return Y\n'
RUN="\ndef run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')\n    return {'output':k(a).data}\n"
IMPORT='from numpy import __builtins__ as space\n'
E='space["exec"]("DT = \'fp32\'", space["globals"]())'
CASES={
"import_alias_function_default":IMPORT+f'def marker(arg={E}):\n    pass\n',
"import_alias_argument_annotation":IMPORT+f'def marker(arg: {E}):\n    pass\n',
"import_alias_return_annotation":IMPORT+f'def marker() -> {E}:\n    pass\n',
"import_alias_module_annotation":IMPORT+f'flag: {E} = 0\n',
"import_alias_method_default":IMPORT+f'class C:\n    def marker(self, arg={E}):\n        pass\n',
"import_alias_class_annotation":IMPORT+f'class C:\n    flag: {E} = 0\n',
"types_module":'import types\nm=types.ModuleType("scratch")\nm.DT="fp32"\n',
"sys_alias":'import sys as runtime\nruntime.modules[__name__].DT="fp32"\n',
"decorator_mutation":'def inject(f):\n    f.__globals__["DT"]="fp32"\n    return f\n@inject\ndef marker():\n    pass\n',
"class_globals":'class C:\n    globals()["DT"]="fp32"\n',
"comprehension":'v=[0 for i in range(1)]\n',
"with":'from contextlib import nullcontext\nwith nullcontext():\n    pass\n',
"if":'if True:\n    pass\n',
"rebound_float":'float=lambda x:0\nEPS=float("1e-3")\n',
"fake_kernel":'def kernel(f):\n    return f\n',
"dunder_def":'def __getattr__(name):\n    return None\n',
"adjacent_dunder":'def marker():\n    pass\ngetattr(marker,"__glo" "bals__")["DT"]="fp32"\n',
"chr_join":'def marker():\n    pass\ngetattr(marker,"".join(chr(c) for c in [95,95,103,108,111,98,97,108,115,95,95]))["DT"]="fp32"\n',
}
POSITIVE={"float_literal":'EPS=float("1e-3")\n',"static_container":'N=2\nVALUES=[N,3]\n',"kernel_identity":'class C:\n    flag=1\n    def value(self):\n        return 1\n'}
def observe(name,extra,fp64):
 code=H+'DT="bf16"\n'+extra+K+RUN
 (OUT/(name+'.py')).write_text(code)
 clear_dims();stop_trace();x=np.array([[1.003,1.009]],np.float64)
 rec={"name":name,"fp64":fp64,"static":na._module_init_not_static(ast.parse(code)),"reflective":na._module_reflective_construct(ast.parse(code)),"constants":na._module_string_constants(ast.parse(code))}
 reasons=[];fallback=[]
 try:
  slow,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=False)
  low=na.numba_lower(code,boundscheck=True,use_fp64=fp64,on_decline=reasons.append)
  fast=na.execute_fast(code,{'a':x},use_fp64=fp64,on_fallback=fallback.append)
  public,_=execute_translated(code,{'a':x.shape},all_inputs={'a':x},fast=True,use_fp64=fp64)
  rec.update(lowered=low is not None,declines=reasons,fallbacks=fallback,slow=slow['output'].tolist(),slow_dtype=str(slow['output'].dtype),fast=None if fast is None else fast['output'].tolist(),public=public['output'].tolist(),public_dtype=str(public['output'].dtype))
  if low:(OUT/(name+'-'+str(fp64)+'.lowered.py')).write_text(low)
 except Exception as exc:rec.update(error=type(exc).__name__,message=str(exc),declines=reasons,fallbacks=fallback)
 (OUT/(name+'-'+str(fp64)+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
 return rec
@pytest.mark.parametrize("fp64",[True,False])
@pytest.mark.parametrize("name",list(CASES))
def test_dynamic_or_nonstatic_module_declines(name,fp64):
 r=observe(name,CASES[name],fp64)
 assert "error" not in r,r
 assert r["lowered"] is False and r["declines"] and r["declines"][0].startswith("kernel_output_dtype_unestablished:k"),r
 np.testing.assert_array_equal(r["public"],r["slow"])
 assert r["public_dtype"]=="float64"
@pytest.mark.parametrize("fp64",[True,False])
@pytest.mark.parametrize("name",list(POSITIVE))
def test_static_positive(name,fp64):
 r=observe("positive-"+name,POSITIVE[name],fp64)
 assert "error" not in r and r["static"] is None and r["reflective"] is None and r["lowered"],r
 np.testing.assert_array_equal(r["public"],r["slow"])
LOOP='    for m in numba.prange(3):\n        s+=m\n'
NEW='    for _m_prange in numba.prange(3):\n        m=np.intp(_m_prange)\n        s+=m\n'
def prog(post,pre=''):return 'def f():\n    s=0\n'+pre+LOOP+post
ALIA='    from builtins import locals as namespace\n'
NORM={
"vars":prog('    return s+vars().get("_m_prange",100)\n'),
"format_map":prog('    return "{_m_prange}".format_map(dict({"_m_prange":100},**locals()))\n'),
"nested_fstring":prog('    return f"{f\\\'{s+locals().get(chr(95)+chr(109)+chr(95)+chr(112)+chr(114)+chr(97)+chr(110)+chr(103)+chr(101),100)}\\\'}"\n'),
"bytes_literal_lookup":prog('    return s+namespace().get(b"_m_prange".decode(),100)\n',ALIA),
"translate_lookup":prog('    return s+namespace().get("_m_prangd".translate({100:101}),100)\n',ALIA),
"chr_join_lookup":prog('    return s+namespace().get("".join(map(chr,[95,109,95,112,114,97,110,103,101])),100)\n',ALIA),
"folded_string_lookup":prog('    return s+namespace().get("_m_"+"prange",100)\n',ALIA),
"annotations_bytes_alias":'def f():\n    from builtins import getattr as fetch\n    class C:\n        __annotations__={b"_m_prange".decode():int}\n        s=0\n        for m in numba.prange(3):\n            s+=m\n    return C.s+fetch(C,b"_m_prange".decode(),100)\n',
}
# Nested f-string represented directly; avoids host-string quoting changing the Python program.
NORM["nested_fstring"]=prog('    return f"{f\\\"{s}\\\"}"\n')
# Python 3.10 rejects a backslash inside f-string expressions; use alternating literal quotes instead.
NORM["nested_fstring"]=prog("""    return f"{f'{s+locals().get(chr(95)+chr(109)+chr(95)+chr(112)+chr(114)+chr(97)+chr(110)+chr(103)+chr(101),100)}'}"\n""")
@pytest.mark.parametrize("name",list(NORM))
def test_reflective_normalizer_preserves_difference(name):
 before=NORM[name];after=before.replace(LOOP,NEW)
 if name=="annotations_bytes_alias":after=before.replace('for m in numba.prange(3):\n            s+=m','for _m_prange in numba.prange(3):\n            m=np.intp(_m_prange)\n            s+=m')
 values=[]
 for src in (before,after):
  ns={"np":np,"numba":numba};exec(src,ns);v=ns["f"]();values.append(v.item() if isinstance(v,np.generic) else v)
 norms=[t._normalise_prange_rebind(src) for src in (before,after)]
 rec={"before":before,"after":after,"values":values,"normalizations":norms,"identifiers":list(t._identifiers(ast.parse(after))).count("_m_prange"),"tokens":t._name_token_count(after,"_m_prange")}
 (OUT/("normalizer-"+name+".json")).write_text(json.dumps(rec,indent=2)+'\n')
 assert values[0]!=values[1],rec
 assert norms[0]!=norms[1],rec

