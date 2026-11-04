# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round-7 contract probes, expectations frozen before execution.

Unmodelled calls must be typed, not guessed safe. Valid cases execute on the
interpreter; narrow cases only parse/convert and never execute invalid accesses.
"""
import ast, collections, json, os, pathlib, sys
import pytest
import test_eval_order as e
a=e.a
R=pathlib.Path(__file__).parent
ROLE=os.environ['REVIEW_ROLE']

RESET='def reset(*args):\n    nonlocal j\n    j=0\n    return 0'
VALID_UNMODELLED={
 'called_nonlocal': RESET+'\nreset()\nY[t,j]=0',
 'star_call_nonlocal': RESET+'\nY[t,j]=reset(*())',
 'decorator_call_nonlocal': 'def reset(f):\n    nonlocal j\n    j=0\n    return f\n@reset\ndef f():\n    pass\nY[t,j]=0',
 'generator_send_nonlocal': 'def reset():\n    nonlocal j\n    j=0\n    yield 0\ng=reset()\nv=g.send(None)\nY[t,j]=0',
 'lambda_calls_nonlocal': RESET+'\nf=lambda:reset()\nv=f()\nY[t,j]=0',
 'class_cell_method_nonlocal': 'class C:\n    def reset(self):\n        nonlocal j\n        j=0\n        return __class__\nv=C().reset()\nY[t,j]=0',
}
NARROW_UNMODELLED={
 'lambda_called_read': 'v=(lambda:Y[t,j])()',
 'nested_def_called_read': 'def read():\n    return Y[t,j]\nv=read()',
 'generator_send_read': 'g=(Y[t,j] for q in range(1))\nv=g.send(None)',
 'generator_next_read': 'v=next(Y[t,j] for q in range(1))',
 'eval_alias': 'f=eval\nv=f("0")\nY[t,j]=0',
 'exec_attribute': 'import builtins\nv=builtins.exec("pass")\nY[t,j]=0',
 'locals_alias': 'f=locals\nv=f()\nY[t,j]=0',
 'module_setattr': 'import types\nm=types.ModuleType("review_probe")\nv=setattr(m,"j",0)\nY[t,j]=0',
}
TYPED_CONTROLS={
 'exec_direct':'v=exec("pass")\nY[t,j]=0',
 'eval_direct':'v=eval("0")\nY[t,j]=0',
 'locals_direct':'v=locals()\nY[t,j]=0',
 'globals_store':'globals()["review_probe"]=0\nY[t,j]=0',
 'yield_read':'v=(yield Y[t,j])',
 'await_read':'v=await Y[t,j]',
 'generator_walrus':'v=((j:=0) for q in range(1))\nY[t,j]=0',
}
VALID_MODELLED={
 'try_finally':'try:\n    pass\nfinally:\n    j=0\nY[t,j]=0',
 'ifexp_unselected_read':'v=Y[t,j] if False else 0\nY[t,0]=0',
 'bool_and_unselected_read':'v=False and Y[t,j]\nY[t,0]=0',
 'bool_or_unselected_read':'v=True or Y[t,j]\nY[t,0]=0',
 'compare_unselected_read':'v=1<0<Y[t,j]\nY[t,0]=0',
}
NARROW_MODELLED={
 'nonlocal_never_called':RESET+'\nY[t,j]=0',
 'lambda_never_called':'f=lambda:Y[t,j]\nY[t,j]=0',
 'comp_first_iter_local_target':'v=[0 for j in [Y[t,j]]]',
 'comp_first_iter_later_target':'v=[0 for q in [Y[t,j]] for j in range(1)]',
}
def make(st):
 src=e.source(st)
 return a.prog(src[src.index('    Y='):])
def record(tmp_path,name,src,data):
 d=R/'runs'/('model-boundary-'+ROLE);d.mkdir(parents=True,exist_ok=True)
 (d/(name+'.json')).write_text(json.dumps({'source':src,**data},indent=2,default=str)+'\n')
def parsed(tmp_path,name,st):
 src=make(st);L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0')
 record(tmp_path,name,src,{'layer':L,'analysis':A,'parsed_only':True})
 return L,A
def executed(tmp_path,name,st):
 src=make(st);hits=collections.Counter()
 fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='k')
 lines={n.lineno for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and ast.unparse(t.value)=='Y' for t in n.targets)}
 def trace(f,event,arg):
  if event=='line' and f.f_code.co_filename=='<string>' and f.f_code.co_name=='k' and f.f_lineno in lines:hits[f.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 record(tmp_path,name,src,{'observation':r,'runtime_output_writes':dict(hits)})
 assert 'error' not in r and sum(hits.values())==32,(r,hits)
 return r['layer'],r['analysis']
def typed(L,A):
 assert 'loop_dim_extent_mismatch' not in L,L
 notes=L.get('witness_unmodelled',[])
 assert notes and all(n.startswith('witness_unmodelled:') for n in notes),L
 assert L.get('loop_iterations_complete') is False and set(notes)<=set(L.get('loop_iterations_omitted',[])),L
 layer=A['layers']['k#0']
 assert layer.get('status')=='incomplete',layer
 assert 'witness_unmodelled:' in json.dumps(layer),layer
@pytest.mark.parametrize('name',VALID_UNMODELLED)
def test_valid_unmodelled(tmp_path,name):typed(*executed(tmp_path,name,VALID_UNMODELLED[name]))
@pytest.mark.parametrize('name',NARROW_UNMODELLED)
def test_narrow_unmodelled(tmp_path,name):typed(*parsed(tmp_path,name,NARROW_UNMODELLED[name]))
@pytest.mark.parametrize('name',TYPED_CONTROLS)
def test_typed_controls(tmp_path,name):typed(*parsed(tmp_path,name,TYPED_CONTROLS[name]))
@pytest.mark.parametrize('name',VALID_MODELLED)
def test_valid_modelled(tmp_path,name):
 L,A=executed(tmp_path,name,VALID_MODELLED[name]);assert 'loop_dim_extent_mismatch' not in L,L
@pytest.mark.parametrize('name',NARROW_MODELLED)
def test_narrow_modelled(tmp_path,name):
 L,A=parsed(tmp_path,name,NARROW_MODELLED[name]);assert 'loop_dim_extent_mismatch' in L,L
