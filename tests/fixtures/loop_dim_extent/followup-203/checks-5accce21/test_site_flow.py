# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round4 governing-index probes; expectations frozen before execution."""
import ast,collections,json,os,pathlib,sys,types
import pytest
import test_sequence_sites as old
from solar.einsum_dsl import einsum_converter as cv
from solar.einsum_dsl.ast_parser import extract_kernel_info
R=pathlib.Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE'];a=old.a
N='    Y=Tensor(T=4,K=3)\n';Q='    Q=Tensor(T=4,E=8)\n'
VALID={
 'three_alternating_siblings':(N+Q+'    A=Q\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    A=Y\n    for t in Dim("T"):\n        for e in Dim("K"):\n            A[t,e]=X[t,e]\n    A=Q\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    return Y\n',76),
 'outer_governor_across_inner_bodies':(N+Q+'    for e in Dim("E"):\n        for t in Dim("T"):\n            Q[t,e]=X[t,e]\n        for t in Dim("T"):\n            Q[t,e]=X[t,e]\n    for k in Dim("K"):\n        for t in Dim("T"):\n            Y[t,k]=Q[t,k]\n    return Y\n',76),
 'same_dim_sequential_sizes':(N+Q+'    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n    for t in Dim("T"):\n        for e in Dim("E",size=3):\n            Y[t,e]=Q[t,e]\n    return Y\n',44),
 'nested_same_spelling_inner_governs':(N+Q+'    for j in Dim("E"):\n        for j in Dim("K"):\n            for t in Dim("T"):\n                Y[t,j]=X[t,j]\n    return Y\n',96),
 'derived_non_loop_index_control':(N+Q+'    for t in Dim("T"):\n        for e in Dim("E"):\n            j=0\n            Y[t,j]=X[t,j]\n    return Y\n',64),
 'reassigned_index':(N+Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n            j=0\n            Y[t,j]=X[t,j]\n    return Y\n',64),
 'reassigned_index_tuple':(N+Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n            j,v=0,1\n            Y[t,j]=X[t,j]\n    return Y\n',64),
 'shadow_survives_inner_loop':(N+Q+'    for j in Dim("E"):\n        for j in Dim("K"):\n            for t in Dim("T"):\n                Y[t,j]=X[t,j]\n        for t in Dim("T"):\n            Y[t,j]=X[t,j]\n    return Y\n',128),
 'assignment_before_inner_loop':(N+Q+'    for j in Dim("E"):\n        j=0\n        for t in Dim("T"):\n            Y[t,j]=X[t,j]\n    return Y\n',40),
}
def record(name,code,data):
 d=R/'runs'/('site-flow-'+ROLE)/name;d.mkdir(parents=True,exist_ok=True);p=d/'probe.py';p.write_text(code);info=extract_kernel_info(p)['k'];stmts=[x.strip() for x in info['loop_expr'].split(';') if x.strip()]
 sites=cv._access_witness_sites(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_access_witness_sites') else None
 data.update(role=ROLE,parser=info,sites=sites,statements=stmts);(d/'record.json').write_text(json.dumps(data,indent=2,default=str)+'\n')
 return sites
@pytest.mark.parametrize('name',VALID)
def test_valid_governing_index(tmp_path,name):
 body,count=VALID[name];code=a.prog(body);fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='k');assigns={id(n):n for loop in ast.walk(fn) if isinstance(loop,ast.For) for n in ast.walk(loop) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))};weights=collections.Counter(n.lineno for n in assigns.values());hits=collections.Counter()
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno in weights:hits[frame.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,code,inputs=a.INPUTS)
 finally:sys.settrace(None)
 actual=sum(weights[k]*v for k,v in hits.items());record(name,code,{'observation':r,'runtime_assignments':actual,'runtime_line_hits':hits,'assignments_per_line':weights});assert actual==count,(hits,weights,count)
 a.counted(r,count)

@pytest.mark.parametrize('scope',['sibling','outer'])
def test_real_narrow_site_refuses(tmp_path,scope):
 if scope=='sibling':body=N+Q+'    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n    for t in Dim("T"):\n        for j in Dim("E"):\n            Y[t,j]=X[t,j]\n    return Y\n'
 else:body=N+Q+'    for j in Dim("E"):\n        for k in Dim("K"):\n            Y[j,k]=Z[0,k]\n    return Y\n'
 code=a.H+a.SIG+body;L,A=a._isolated_layer(tmp_path,code,a._probe_metadata(),'k#0');record('narrow-'+scope,code,{'layer':L,'analysis':A,'parsed_only_invalid':True});assert 'loop_dim_extent_mismatch' in L and L.get('loop_iterations_complete') is False,L

@pytest.mark.parametrize('case',['nested_shadow','outer_shared','unmatched','plain_shadow'])
def test_sites_helper_placement(tmp_path,case):
 if not hasattr(cv,'_access_witness_sites'):pytest.skip('site helper absent on baseline')
 if case=='nested_shadow':body=VALID['nested_same_spelling_inner_governs'][0]
 elif case=='outer_shared':body=VALID['outer_governor_across_inner_bodies'][0]
 elif case=='unmatched':body=VALID['three_alternating_siblings'][0]
 else:body=N+'    for j in Dim("E"):\n        for j in range(3):\n            for t in Dim("T"):\n                Y[t,j]=X[t,j]\n    return Y\n'
 code=a.H+a.SIG+body;p=tmp_path/'probe.py';p.write_text(code);info=extract_kernel_info(p)['k'];stmts=[x.strip() for x in info['loop_expr'].split(';')]
 if case=='unmatched':stmts[1]=stmts[1].replace('A[t, e]','A[t, e+0]')
 sites=cv._access_witness_sites(types.SimpleNamespace(**info),stmts);record('helper-'+case,code,{'direct_statements':stmts,'direct_sites':sites})
 if case=='nested_shadow':assert sites[0]['j'][1]=='K'
 elif case=='outer_shared':assert sites[0]['e']==sites[1]['e'] and sites[0]['t']!=sites[1]['t']
 elif case=='unmatched':assert sites[1]=={} and sites[0] and sites[2]
 else:assert 'j' not in sites[0]
