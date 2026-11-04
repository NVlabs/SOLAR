# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent r6 evaluation-order probes; frozen before first execution."""
import ast,collections,json,os,pathlib,sys,types
import pytest
import test_live_binding as p
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl import einsum_converter as cv
R=pathlib.Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE'];a=p.a
VALID={
 'dict_value_before_key':'v={0:(j:=0),Y[t,j]:1}',
 'dict_unpack_before_key':'v={**{0:(j:=0)},Y[t,j]:1}',
 'dict_value_before_value_control':'v={0:(j:=0),1:Y[t,j]}',
 'dict_key_before_value_control':'v={(j:=0):Y[t,j]}',
 'subscript_first_component_rebind_control':'Y[(j:=0)+t,j]=0',
 'rhs_tuple_before_read_control':'v=((j:=0),Y[t,j])',
}
NARROW={
 'index_read_before_later_walrus':'v=Y[j,(j:=0)]',
 'index_write_before_later_walrus':'Y[j,(j:=0)]=0',
 'index_annotated_write_before_walrus':'Y[j,(j:=0)]:float=0',
 'repeated_spelling_before_walrus':'Y[j,j+(j:=0)]=0',
 'aug_index_before_later_walrus':'Y[j,(j:=0)]+=0',
 'dict_value_before_later_key_walrus':'v={0:Y[t,j],(j:=0):1}',
 'dict_read_before_later_value_walrus_control':'v={0:Y[t,j],1:(j:=0)}',
 'dict_key_read_before_value_walrus_control':'v={Y[t,j]:(j:=0)}',
 'rhs_read_before_walrus_control':'Y[t,j]=Y[t,j]+(j:=0)*0',
}
def source(st):return a.H+a.SIG+p.old.N+p.old.Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n'+''.join('            '+s+'\n' for s in st.splitlines())+'    return Y\n'
def save(tmp_path,name,src,data):
 d=R/'runs'/('eval-order-'+ROLE);d.mkdir(parents=True,exist_ok=True);f=tmp_path/'source.py';f.write_text(src);info=extract_kernel_info(f)['k'];stmts=[s.strip() for s in info['loop_expr'].split(';') if s.strip()]
 sites=cv._access_witness_sites(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_access_witness_sites') else None
 intra=cv._access_witness_intra(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_access_witness_intra') else None
 (d/(name+'.json')).write_text(json.dumps({'role':ROLE,'source':src,'parser':info,'stmts':stmts,'sites':sites,'intra':intra,**data},indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+'\n')
@pytest.mark.parametrize('name',VALID)
def test_valid_order(tmp_path,name):
 st=VALID[name];src=source(st);body=src[src.index('    Y='):];src=a.prog(body);hits=collections.Counter();fn=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='k');loop=next(x for x in fn.body if isinstance(x,ast.For));lines={x.lineno for x in ast.walk(loop) if isinstance(x,(ast.Assign,ast.AnnAssign,ast.AugAssign))}
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno in lines:hits[frame.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 save(tmp_path,'valid-'+name,src,{'observation':r,'runtime_hits':dict(hits)});assert sum(hits.values())==32 and 'error' not in r,(hits,r)
 assert 'loop_dim_extent_mismatch' not in r['layer'] and r['layer'].get('loop_iterations')==32,r['layer']
@pytest.mark.parametrize('name',NARROW)
def test_access_uses_earlier_value(tmp_path,name):
 src=source(NARROW[name]);L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');save(tmp_path,'narrow-'+name,src,{'layer':L,'analysis':A,'parsed_only_invalid':True});assert 'loop_dim_extent_mismatch' in L and L.get('loop_iterations_complete') is False,L

SCOPES={
 'class_local':'class C:\n    j=0',
 'class_nonlocal':'class C:\n    nonlocal j\n    j=0',
 'class_nonlocal_walrus':'class C:\n    nonlocal j\n    v=(j:=0)',
 'nested_class_nonlocal':'class C:\n    class D:\n        nonlocal j\n        j=0',
 'class_default_outer':'class C:\n    def f(x=(j:=0)):\n        pass',
 'class_default_nonlocal':'class C:\n    nonlocal j\n    def f(x=(j:=0)):\n        pass',
}
@pytest.mark.parametrize('name',['class_nonlocal','class_nonlocal_walrus','nested_class_nonlocal','class_default_nonlocal'])
def test_executed_class_nonlocal(tmp_path,name):
 src=source(SCOPES[name]+'\nY[t,j]=X[t,j]');src=a.prog(src[src.index('    Y='):]);hits=collections.Counter();fn=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='k');line=next(x.lineno for x in ast.walk(fn) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Subscript) for t in x.targets))
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno==line:hits[line]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 save(tmp_path,'valid-'+name,src,{'observation':r,'runtime_hits':dict(hits)});assert sum(hits.values())==32 and 'error' not in r,(hits,r)
 assert 'loop_dim_extent_mismatch' not in r['layer'],r['layer']
@pytest.mark.parametrize('name',['class_local','class_default_outer'])
def test_class_local_preserves_outer(tmp_path,name):
 src=source(SCOPES[name]+'\nY[t,j]=X[t,j]');L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');save(tmp_path,'narrow-'+name,src,{'layer':L,'analysis':A,'parsed_only_invalid':True});assert 'loop_dim_extent_mismatch' in L,L

BINDERS={
 'with_name':'with context() as j:\n    pass',
 'with_tuple':'with context() as (j,q):\n    pass',
 'match_name':'match 0:\n    case j: pass',
 'match_mapping_rest':'match {}:\n    case {**j}: pass',
 'match_sequence_star':'match []:\n    case [*j]: pass',
 'global':'global j', 'nonlocal':'nonlocal j',
 'import_dotted':'import j.sub',
 'except_name':'try:\n    pass\nexcept Exception as j:\n    pass',
}
@pytest.mark.parametrize('name',BINDERS)
def test_all_string_and_target_bindings(tmp_path,name):
 if not hasattr(cv,'_bound_in_scope'):pytest.skip('new helper absent on baseline')
 n=ast.parse(BINDERS[name]).body[0];actual=cv._bound_in_scope(n);d=R/'runs'/('eval-order-'+ROLE);d.mkdir(parents=True,exist_ok=True);(d/('bind-'+name+'.json')).write_text(json.dumps({'code':BINDERS[name],'actual':sorted(actual)},indent=2)+'\n');assert 'j' in actual,actual
