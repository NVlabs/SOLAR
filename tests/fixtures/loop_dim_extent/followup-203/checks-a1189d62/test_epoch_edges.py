# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent the review round2 binding epoch probes. Assertions are not changed after execution."""
import ast,collections,hashlib,importlib.util,json,os,pathlib,sys,types
import pytest
from solar.einsum_dsl import einsum_converter as cv
from solar.einsum_dsl.ast_parser import extract_kernel_info
R=pathlib.Path(__file__).parent;role=os.environ['REVIEW_ROLE']
p=R/'source/head/solar/tests/test_loop_dim_extent_access_witness.py';spec=importlib.util.spec_from_file_location('author_epochs',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
N='    Y=Tensor(T=4,K=3)\n';Q='    Q=Tensor(T=4,E=8)\n'
L='    for t in Dim("T"):\n        for e in Dim("E"):\n            v=X[t,e]\n            A[t,e]=v\n'
VALID={
 'same_line_rebind_then_capture':(N+'    Out=Y\n    Y=Tensor(T=4,E=8); A=Y\n'+L+'    return Out\n',64),
 'same_line_alias_rebind_then_capture':(N+Q+'    Out=Y\n    Y=Q; A=Y\n'+L+'    return Out\n',64),
 'same_line_alias_chain':(N+Q+'    Out=Y\n    Y=Q; B=Y; A=B\n'+L+'    return Out\n',64),
 'multiline_rebind_then_capture_control':(N+'    Out=Y\n    Y=Tensor(T=4,E=8)\n    A=Y\n'+L+'    return Out\n',64),
 'delete_and_rebind':(N+Q+'    A=Y\n    del A\n    A=Q\n'+L+'    return Y\n',64),
 'tuple_unpack_alias':(N+Q+'    A,= (Q,)\n'+L+'    return Y\n',64),
 'two_allocations_one_statement':('    Y,Q = Tensor(T=4,K=3),Tensor(T=4,E=8)\n    A=Q\n'+L+'    return Y\n',64),
 'two_allocations_one_line':('    Y=Tensor(T=4,K=3); Q=Tensor(T=4,E=8)\n    A=Q\n'+L+'    return Y\n',64),
 'loop_rebinding_alias':(N+Q+'    A=Y\n    for t in Dim("T"):\n        A=Q\n        for e in Dim("E"):\n            v=X[t,e]\n            A[t,e]=v\n    return Y\n',68),
 'nested_loop_alias_capture':(N+'    for t in Dim("T"):\n        for k in Dim("K"):\n            A=Y\n            A[t,k]=X[t,k]\n    return Y\n',24),
 'author_in_loop_alias_16':(N+'    for t in Dim("T"):\n        A=Y\n        for k in Dim("K"):\n            A[t,k]=X[t,k]*Z[t,k]\n    return Y\n',16),
}
BAD={
 'same_line_alloc_and_alias':('    Y=Tensor(T=4,K=3); A=Y\n'+'    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    return Y\n'),
 'same_line_switch_to_returned_alias':(Q+N+'    A=Q\n    A=Y; B=A\n'+'    for t in Dim("T"):\n        for e in Dim("E"):\n            B[t,e]=X[t,e]\n    return Y\n'),
 'annotated_direct_access':(N+'    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]: float = X[t,e]\n    return Y\n'),
 'annotated_scalar_in_same_body':(N+'    for t in Dim("T"):\n        for e in Dim("E"):\n            v: float = X[t,e]\n            Y[t,e]=v\n    return Y\n'),
 'plain_direct_access_control':(N+'    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]=X[t,e]\n    return Y\n'),
}

def save(name,code,r):
 d=R/'runs'/('epochs-'+role)/name;d.mkdir(parents=True,exist_ok=True);p=d/'probe.py';p.write_text(code);info=extract_kernel_info(p)['k'];fn=getattr(cv,'_access_witness_epochs',None);stmts=[s.strip() for s in (info.get('loop_expr') or '').split(';') if s.strip()];epochs=fn(types.SimpleNamespace(**info),stmts) if fn else None
 r.update(role=role,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parser=info,epochs=epochs);(d/'record.json').write_text(json.dumps(r,indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+'\n')

@pytest.mark.parametrize('name',VALID)
def test_valid_epoch_programs(tmp_path,name):
 body,n=VALID[name];code=a.prog(body);tree=ast.parse(code);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='k');lines={x.lineno for loop in ast.walk(fn) if isinstance(loop,ast.For) for x in ast.walk(loop) if isinstance(x,(ast.Assign,ast.AnnAssign,ast.AugAssign))};hits=collections.Counter()
 def tracer(frame,event,arg):
  if event=='line' and frame.f_code.co_filename=='<string>' and frame.f_code.co_name=='k' and frame.f_lineno in lines:hits[frame.f_lineno]+=1
  return tracer
 sys.settrace(tracer)
 try:rec=a.observe(tmp_path,code,inputs=a.INPUTS)
 finally:sys.settrace(None)
 save(name,code,{'runtime_assignment_count':sum(hits.values()),'runtime_lines':dict(hits),'observation':rec});assert sum(hits.values())==n,(hits,n);a.counted(rec,n)

@pytest.mark.parametrize('name',BAD)
def test_proven_narrow_access(tmp_path,name):
 code=a.H+a.SIG+BAD[name];L,A=a._isolated_layer(tmp_path,code,a._probe_metadata(),'k#0');save(name,code,{'parsed_only_invalid_program':True,'layer':L,'analysis':A});assert 'loop_dim_extent_mismatch' in L and L.get('loop_iterations_complete') is False and 'loop_iterations' not in L,L

@pytest.mark.parametrize('kind',['missing_segment','conditional','loop_carried','tuple_alias','deleted','same_statement_unknown'])
def test_unproved_does_not_borrow(tmp_path,kind):
 if not hasattr(cv,'_access_witness_epochs'):pytest.skip('new helper absent on baseline')
 stems={
  'missing_segment':N+'    A=Y\n',
  'conditional':N+Q+'    A=Y\n    if X[0,0]>0:\n        A=Q\n',
  'loop_carried':N+'    A=Y\n',
  'tuple_alias':N+'    A,= (Y,)\n',
  'deleted':N+'    A=Y\n    del A\n',
  'same_statement_unknown':N+'    A=0\n    A,Y=Y,Tensor(T=4,K=3)\n',
 }
 body=stems[kind]+L+'    return Y\n'
 if kind=='loop_carried':body=N+'    for t in Dim("T"):\n        A=Y\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    return Y\n'
 code=a.H+a.SIG+body;p=tmp_path/'probe.py';p.write_text(code);info=extract_kernel_info(p)['k']
 if kind=='missing_segment':info['source_text']=None
 stmts=[s.strip() for s in info['loop_expr'].split(';')];borrow,shadow=cv._access_witness_epochs(types.SimpleNamespace(**info),stmts);save('boundary-'+kind,code,{'direct_helper_borrow':borrow,'direct_helper_shadowed':shadow});assert all(not s for s in borrow),(borrow,shadow)
