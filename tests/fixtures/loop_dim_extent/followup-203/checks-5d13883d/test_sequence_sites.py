# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent the review round-3 probes. No expectations changed after execution."""
import ast,collections,hashlib,json,os,pathlib,sys,types
import numpy as np
import pytest
import test_epoch_edges as old
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl import einsum_converter as cv
a=old.a; R=pathlib.Path(__file__).parent; ROLE=os.environ['REVIEW_ROLE']
N='    Y=Tensor(T=4,K=3)\n';Q='    Q=Tensor(T=4,E=8)\n'
LOOP='    for t in Dim("T"):\n        for e in Dim("E"):\n            v=X[t,e]\n            A[t,e]=v\n'
VALID={
 'call_walrus_then_capture':(N+Q+'    A=Y\n    v=float((A:=Q)[0,0]); B=A; A=B\n'+LOOP+'    return Y\n',64),
 'call_walrus_source_then_capture':(N+Q+'    Out=Y\n    v=float((Y:=Q)[0,0]); A=Y\n'+LOOP+'    return Out\n',64),
 'two_walrus_call':(N+Q+'    A=Y\n    v=max(float((A:=Y)[0,0]),float((A:=Q)[0,0])); B=A; A=B\n'+LOOP+'    return Y\n',64),
 'chained_assignment':(N+Q+'    A=B=Q\n'+LOOP+'    return Y\n',64),
 'swap_names':(N+Q+'    A=Y; B=Q\n    A,B=B,A\n'+LOOP+'    return Y\n',64),
 'own_line_accessed_binding':(N+Q+'    for t in Dim("T"):\n        for e in Dim("E"):\n            A=Q; A[t,e]=X[t,e]\n    return Y\n',64),
 'own_line_unaccessed_binding':(N+Q+'    A=Y\n    for t in Dim("T"):\n        for k in Dim("K"):\n            B=A; A[t,k]=X[t,k]\n    return Y\n',24),
 'normalized_duplicate_same_epoch':(N+Q+'    A=Q\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]:float=X[t,e]\n            A[t,e]=X[t,e]\n    return Y\n',64),
 'normalized_scalar_before_valid':(N+Q+'    A=Q\n    for t in Dim("T"):\n        for e in Dim("E"):\n            v:float=X[t,e]\n            A[t,e]=v\n    return Y\n',64),
 'duplicate_sites_different_loop_extents':(N+Q+'    A=Q\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    A=Y\n    for t in Dim("T"):\n        for e in Dim("K"):\n            A[t,e]=X[t,e]\n    return Y\n',44),
 'normalized_duplicate_different_epochs':(N+Q+'    A=Q\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]:float=X[t,e]\n    A=Y\n    for t in Dim("T"):\n        for e in Dim("K"):\n            A[t,e]=X[t,e]\n    return Y\n',44),
}
def record(name,code,data,stmts=None,info=None):
 d=R/'runs'/('sequence-'+ROLE)/name;d.mkdir(parents=True,exist_ok=True);p=d/'probe.py';p.write_text(code)
 info=info or extract_kernel_info(p)['k'];stmts=stmts or [s.strip() for s in info['loop_expr'].split(';')]
 epochs=cv._access_witness_epochs(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_access_witness_epochs') else None
 data.update(role=ROLE,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parser=info,stmts=stmts,epochs=epochs)
 (d/'record.json').write_text(json.dumps(data,indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+'\n')
 return epochs
@pytest.mark.parametrize('name',VALID)
def test_valid(tmp_path,name):
 body,count=VALID[name];code=a.prog(body);fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='k')
 assigns={id(n):n for loop in ast.walk(fn) if isinstance(loop,ast.For) for n in ast.walk(loop) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))}
 weights=collections.Counter(n.lineno for n in assigns.values());hits=collections.Counter()
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno in weights:hits[frame.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,code,inputs=a.INPUTS)
 finally:sys.settrace(None)
 actual=sum(v*weights[k] for k,v in hits.items());record(name,code,{'observation':r,'runtime_line_hits':hits,'assignments_per_line':weights,'runtime_assignments':actual})
 assert actual==count,(hits,weights,count)
 if 'duplicate' in name and 'different' in name:assert np.array_equal(np.asarray(r['out']['out']),a.X[:,:3])
 a.counted(r,count)

@pytest.mark.parametrize('case',['chain','swap','walrus','own_accessed','own_unaccessed','duplicate_plain','duplicate_normalized','unmatched_first','unmatched_second'])
def test_helper_boundaries(tmp_path,case):
 if not hasattr(cv,'_access_witness_epochs'):pytest.skip('epoch helper absent on target')
 stem=N+Q;body='    for t in Dim("T"):\n        for k in Dim("K"):\n            A[t,k]=X[t,k]\n'
 expected=[set()]
 if case=='chain':stem+='    A=B=Y\n'
 elif case=='swap':stem+='    A=Q; B=Y\n    A,B=B,A\n'
 elif case=='walrus':stem+='    v=float((A:=Y)[0,0]); B=A\n'
 elif case in ['own_accessed','own_unaccessed']:
  stem+='    A=Y\n';line='A=Y' if case=='own_accessed' else 'B=A';body='    for t in Dim("T"):\n        for k in Dim("K"):\n            '+line+'; A[t,k]=X[t,k]\n';expected=[set(),set() if case=='own_accessed' else {'A'}]
 elif case.startswith('duplicate'):
  stem+='    A=Y\n';ann=':float' if case.endswith('normalized') else '';body='    for t in Dim("T"):\n        for k in Dim("K"):\n            A[t,k]'+ann+'=X[t,k]\n            A[t,k]=X[t,k]\n';expected=[{'A'},{'A'}]
 else:
  stem+='    A=Y\n';body='    for t in Dim("T"):\n        for k in Dim("K"):\n            A[t,k]=X[t,k]\n            A[t,k]=X[t,k]\n';expected=[set(),{'A'}] if case=='unmatched_first' else [{'A'},set()]
 code=a.H+a.SIG+stem+body+'    return Y\n';p=tmp_path/'probe.py';p.write_text(code);info=extract_kernel_info(p)['k'];stmts=[s.strip() for s in info['loop_expr'].split(';')]
 if case.startswith('unmatched'):stmts[0 if case.endswith('first') else 1]=stmts[0].replace('A[t, k]','A[t, k + 0]')
 epochs=record('boundary-'+case,code,{},stmts,info);assert epochs[0]==expected,epochs

@pytest.mark.parametrize('annotated',[False,True])
def test_narrow_duplicate_refuses(tmp_path,annotated):
 ann=':float' if annotated else '';code=a.H+a.SIG+N+'    A=Y\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]'+ann+'=X[t,e]\n            A[t,e]=X[t,e]\n    return Y\n'
 L,A=a._isolated_layer(tmp_path,code,a._probe_metadata(),'k#0');record('narrow-duplicate-'+str(annotated),code,{'layer':L,'analysis':A,'parsed_only_invalid':True});assert 'loop_dim_extent_mismatch' in L and L.get('loop_iterations_complete') is False,L
