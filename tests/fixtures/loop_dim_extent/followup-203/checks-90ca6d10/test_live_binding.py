# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round-5 probes, expectations frozen before first execution."""
import ast,collections,json,os,pathlib,sys,types
import pytest
import test_site_flow as old
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl import einsum_converter as cv
R=pathlib.Path(__file__).parent; ROLE=os.environ['REVIEW_ROLE']; a=old.a
# Expected j governor at the following access. Comprehension targets and deferred bodies are local.
PREFIXES={
 'plain':('j=0',False), 'tuple':('j,v=0,1',False), 'augmented':('j-=j',False),
 'delete':('del j',False), 'walrus':('v=(j:=0)',False),
 'conditional':('if X[t,0]>0:\n    j=0',False),
 'plain_for':('for j in range(3):\n    v=0',False),
 'plain_while':('while j>0:\n    j=0',False),
 'list_comp_local':('v=[j for j in range(2)]',True),
 'set_comp_local':('v={j for j in range(2)}',True),
 'dict_comp_local':('v={j:j for j in range(2)}',True),
 'generator_local':('v=(j for j in range(2))',True),
 'nested_def_local':('def f():\n    j=0',True),
 'lambda_local_walrus':('v=lambda: (j:=0)',True),
 'lambda_parameter':('v=lambda j:j',True),
 'lambda_default_walrus':('v=lambda x=(j:=0):x',False),
 'def_default_walrus':('def f(x=(j:=0)):\n    pass',False),
 'comp_walrus_outer':('v=[(j:=0) for q in range(1)]',False),
 'def_name':('def j():\n    pass',False),
 'class_name':('class j:\n    pass',False),
 'import_name':('import math as j',False),
 'from_import_name':('from math import pi as j',False),
 'except_name':('try:\n    v=0\nexcept Exception as j:\n    pass',False),
}
def code(prefix,tail='Y[t,j]=X[t,j]'):
 return a.H+a.SIG+old.N+old.Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n'+''.join('            '+s+'\n' for s in prefix.splitlines())+'            '+tail+'\n    return Y\n'
def record(tmp_path,name,src,extra=None):
 p=tmp_path/'probe.py';p.write_text(src);info=extract_kernel_info(p)['k'];stmts=[s.strip() for s in info['loop_expr'].split(';') if s.strip()];sites=cv._access_witness_sites(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_access_witness_sites') else None
 row={'role':ROLE,'source':src,'parser':info,'stmts':stmts,'sites':sites,**(extra or {})};d=R/'runs'/('live-binding-'+ROLE);d.mkdir(parents=True,exist_ok=True);(d/(name+'.json')).write_text(json.dumps(row,indent=2,default=str)+'\n');return sites,stmts
@pytest.mark.parametrize('name',PREFIXES)
def test_following_site(tmp_path,name):
 prefix,expected=PREFIXES[name];src=code(prefix);sites,stmts=record(tmp_path,'helper-'+name,src)
 if sites is None:pytest.skip('helper unavailable on baseline')
 ix=stmts.index('Y[t, j] = X[t, j]');assert ('j' in sites[ix]) is expected,(name,expected,stmts,sites)

@pytest.mark.parametrize('name',['list_comp_local','set_comp_local','dict_comp_local','generator_local','nested_def_local','lambda_local_walrus','lambda_parameter'])
def test_genuine_narrow_after_local_binding(tmp_path,name):
 src=code(PREFIXES[name][0]);L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');record(tmp_path,'narrow-'+name,src,{'layer':L,'analysis':A,'invalid_program_parsed_only':True});assert 'loop_dim_extent_mismatch' in L and L.get('loop_iterations_complete') is False,L

VALID={
 'rhs_subscript_walrus':('Y[t,j]=X[t,(j:=0)]',32),
 'rhs_value_walrus':('Y[t,j]=(j:=0)',32),
 'rhs_call_walrus':('Y[t,j]=float((j:=0))',32),
 'lhs_earlier_index_walrus':('Y[(j:=0)+t,j]=X[t,0]',32),
 'same_statement_no_mutation_control':('Y[t,j%3]=X[t,j]',32),
}
@pytest.mark.parametrize('name',VALID)
def test_valid_same_statement_order(tmp_path,name):
 st,expected=VALID[name];src=a.prog(old.N+old.Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n            '+st+'\n    return Y\n');hits=collections.Counter()
 fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='k');line=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets))
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno==line:hits[line]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 record(tmp_path,'valid-'+name,src,{'observation':r,'runtime_assignment_hits':dict(hits),'expected':expected});assert sum(hits.values())==expected,(hits,r)
 a.counted(r,expected)
