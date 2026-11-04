# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supplemental controls; the first probe file and its fixture failure remain unchanged."""
import ast,collections,json,os,pathlib,sys,types
import pytest
import test_live_binding as p
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl import einsum_converter as cv
R=pathlib.Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE'];a=p.a
@pytest.mark.parametrize('kind',['plain_for_then_dim_sibling','while_then_dim_sibling','inner_for_site','header_walrus'])
def test_plain_loop_sites(tmp_path,kind):
 stem=a.H+a.SIG+p.old.N+p.old.Q+'    for j in Dim("E"):\n'
 bodies={
 'plain_for_then_dim_sibling':'        for j in range(3):\n            v=0\n        for t in Dim("T"):\n            Y[t,j]=X[t,j]\n',
 'while_then_dim_sibling':'        while j>0:\n            j=0\n        for t in Dim("T"):\n            Y[t,j]=X[t,j]\n',
 'inner_for_site':'        for j in range(3):\n            for t in Dim("T"):\n                Y[t,j]=X[t,j]\n',
 'header_walrus':'        for t in Dim("T",size=(j:=0)+4):\n            Y[t,j]=X[t,j]\n',
 }
 src=stem+bodies[kind]+'    return Y\n';sites,stmts=p.record(tmp_path,'control-'+kind,src)
 if sites is None:pytest.skip('helper unavailable on baseline')
 ix=stmts.index('Y[t, j] = X[t, j]');assert 'j' not in sites[ix],(stmts,sites)

@pytest.mark.parametrize('kind',['import_F_OK','chain_target_first','tuple_target_first'])
def test_valid_binding_before_access(tmp_path,kind):
 body={
 'import_F_OK':'            from os import F_OK as j\n            Y[t,j]=X[t,j]\n',
 'chain_target_first':'            j=Y[t,j]=0\n',
 'tuple_target_first':'            j,Y[t,j]=0,1\n',
 }[kind]
 src=a.prog(p.old.N+p.old.Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n'+body+'    return Y\n');hits=collections.Counter();fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='k');lines={n.lineno for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for target in n.targets for t in ast.walk(target))}
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno in lines:hits[frame.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 p.record(tmp_path,'valid-extra-'+kind,src,{'observation':r,'runtime_assignment_hits':dict(hits)});assert sum(hits.values())==32 and 'error' not in r,(hits,r)
 assert 'loop_dim_extent_mismatch' not in r['layer'],r['layer']

@pytest.mark.parametrize('tail',['Y[t,j]=j=0','Y[t,j]+=(j:=0)'])
def test_access_before_rebinding_retains_refusal(tmp_path,tail):
 src=a.H+a.SIG+p.old.N+p.old.Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n            '+tail+'\n    return Y\n';L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');p.record(tmp_path,'access-first-'+str(len(tail)),src,{'layer':L,'analysis':A,'parsed_only_invalid':True});assert 'loop_dim_extent_mismatch' in L,L
