# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent the review witness lifetime and alias-access controls. Original expectations retained."""
import ast, collections, hashlib, importlib.util, json, os, pathlib, sys, types
import numpy as np
import pytest
from solar.einsum_dsl import einsum_converter as cv
from solar.einsum_dsl.ast_parser import extract_kernel_info
R=pathlib.Path(__file__).parent
p=R/'source/head/solar/tests/test_loop_dim_extent_access_witness.py'
spec=importlib.util.spec_from_file_location('author_witness',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
ROLE=os.environ['REVIEW_ROLE']
WIDE='    Q=Tensor(T=4,E=8)\n'
NARROW='    Y=Tensor(T=4,K=3)\n'
LOOP='    for t in Dim("T"):\n        for e in Dim("E"):\n            v=X[t,e]\n            Q[t,e]=v\n'
VALID={
 'later_allocation_same_name':(WIDE+LOOP+'    Q=Tensor(T=4,K=3)\n    return Q\n',64),
 'earlier_return_name_rebound':('    Out=Tensor(T=4,E=8)\n'+LOOP.replace('Q[t,e]','Out[t,e]')+NARROW+'    Out=Y\n    return Out\n',64),
 'original_alloc_name_rebound':('    Q=Tensor(T=4,K=3)\n    Out=Q\n'+WIDE+LOOP+'    return Out\n',64),
 'returned_name_different_epoch':(NARROW+'    Out=Y\n    Y=Tensor(T=4,E=8)\n'+LOOP.replace('Q[t,e]','Y[t,e]')+'    return Out\n',64),
 'simple_alias_valid':(NARROW+'    A=Y\n    for t in Dim("T"):\n        for k in Dim("K"):\n            A[t,k]=X[t,k]\n    return Y\n',12),
 'conditional_return_binding':(WIDE+NARROW+'    Out=Y\n    if X[0,0]>0:\n        Out=Q\n'+LOOP+'    return Out\n',64),
 'input_name_reused_as_returned_local':('    X=Tensor(T=4,K=3)\n    for t in Dim("T"):\n        for k in Dim("K"):\n            X[t,k]=Z[t,k]\n    return X\n',12),
}
BAD={
 'access_alias_return_original':NARROW+'    A=Y\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    return Y\n',
 'access_middle_alias_return_chain':NARROW+'    A=Y\n    Out=A\n    for t in Dim("T"):\n        for e in Dim("E"):\n            A[t,e]=X[t,e]\n    return Out\n',
 'direct_returned_access_control':NARROW+'    for t in Dim("T"):\n        for e in Dim("E"):\n            Y[t,e]=X[t,e]\n    return Y\n',
 'narrow_parameter_control':NARROW+WIDE+'    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=Z[t,e]\n    return Y\n',
}

def save(name,code,record):
 d=R/'runs'/('witness-'+ROLE)/name;d.mkdir(parents=True,exist_ok=True)
 source=d/'probe.py';source.write_text(code)
 info=extract_kernel_info(source)['k']
 witness=getattr(cv,'_access_witness_names',None)
 record.update(role=ROLE,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parser=info,witness_names=sorted(witness(types.SimpleNamespace(**info)) or []) if witness else 'legacy_first_access')
 (d/'record.json').write_text(json.dumps(record,indent=2,default=str)+'\n')

@pytest.mark.parametrize('name',VALID)
def test_valid_lifetimes(tmp_path,name):
 body,expected=VALID[name];code=a.prog(body)
 tree=ast.parse(code);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='k')
 lines={n.lineno for loop in ast.walk(fn) if isinstance(loop,ast.For) for n in ast.walk(loop) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign))}
 hits=collections.Counter()
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_name=='k' and frame.f_code.co_filename=='<string>' and frame.f_lineno in lines:hits[frame.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:r=a.observe(tmp_path,code,inputs=a.INPUTS)
 finally:sys.settrace(None)
 save(name,code,{'runtime_assignment_count':sum(hits.values()),'runtime_lines':dict(hits),'observation':r})
 assert sum(hits.values())==expected,(hits,expected)
 a.counted(r,expected)

@pytest.mark.parametrize('name',BAD)
def test_genuine_narrow_access_still_refuses(tmp_path,name):
 code=a.H+a.SIG+BAD[name]
 L,A=a._isolated_layer(tmp_path,code,a._probe_metadata(),'k#0')
 save(name,code,{'parsed_only_invalid_program':True,'layer':L,'analysis':A})
 assert L.get('loop_iterations_complete') is False and 'loop_iterations' not in L,L
 assert 'loop_dim_extent_mismatch' in L,L
 assert A['layers']['k#0']['status']=='incomplete'

UNCERTAIN={
 'conditional_rebind': NARROW+WIDE+'    if X[0,0]>0:\n        Y=Q\n    return Y\n',
 'tuple_return': NARROW+WIDE+'    return Y,Q\n',
 'slice_return':NARROW+'    return Y[:,:]\n',
 'subscript_alias':NARROW+'    aliases=[Y]\n    Out=aliases[0]\n    return Out\n',
 'multiple_return_different_allocs':NARROW+WIDE+'    if X[0,0]>0:\n        return Y\n    return Q\n',
 'multiple_return_same_alloc':NARROW+'    if X[0,0]>0:\n        return Y\n    return Y\n',
}
@pytest.mark.parametrize('name',UNCERTAIN)
def test_identity_boundary(tmp_path,name):
 witness=getattr(cv,'_access_witness_names',None)
 if witness is None:pytest.skip('new helper absent on baseline; integration tests above still compare baseline')
 code=a.H+a.SIG+UNCERTAIN[name];p=tmp_path/'probe.py';p.write_text(code);info=extract_kernel_info(p)['k'];k=types.SimpleNamespace(**info)
 names=witness(k);save(name,code,{'direct_helper_witness':sorted(names or [])})
 assert names==({'Y'} if name=='multiple_return_same_alloc' else set())

@pytest.mark.parametrize('fields,expect',[(dict(),None),(dict(returned_names=[]),set()),(dict(returns=[]),set())])
def test_legacy_fallback_boundary(fields,expect):
 witness=getattr(cv,'_access_witness_names',None)
 if witness is None:pytest.skip('new helper absent on baseline')
 assert witness(types.SimpleNamespace(**fields))==expect
