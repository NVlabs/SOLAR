# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round10 probes, frozen before execution: ancestor transfers, return routing and coverage."""
from pathlib import Path
import ast,collections,json,os,sys,types
import pytest
import test_minimal_control_flow as p
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl import einsum_converter as cv
a=p.a;R=Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE']
INNER='        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n            Y[t,j]=X[t,j]\n'
ANCESTOR={
 'outer_break':'break',
 'outer_continue':'continue',
 'outer_if_break':'if t>=0:\n    break',
 'outer_if_continue':'if t>=0:\n    continue',
 'outer_except_continue':'try:\n    raise ValueError("stop")\nexcept ValueError:\n    continue',
}
def record(tmp_path,name,src,extra):
 d=R/'runs'/('round10-transfer-'+ROLE);d.mkdir(parents=True,exist_ok=True);f=tmp_path/'probe.py';f.write_text(src);info=extract_kernel_info(f)['k'];stmts=[s.strip() for s in (info.get('loop_expr') or '').split(';') if s.strip()]
 model=cv._witness_model_v2(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_witness_model_v2') else None
 row={'source':src,'parser':info,'model':model,**extra};(d/(name+'.json')).write_text(json.dumps(row,indent=2,default=str)+'\n');return row
@pytest.mark.parametrize('name',ANCESTOR)
def test_ancestor_prevents_narrow_access(tmp_path,name):
 src=a.prog(p.PRE+'    for t in Dim("T"):\n'+''.join('        '+x+'\n' for x in ANCESTOR[name].splitlines())+INNER+'    return Y\n');hits=collections.Counter()
 fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='k');lines={n.lineno for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets)}
 def trace(f,event,arg):
  if event=='line' and f.f_code.co_filename=='<string>' and f.f_code.co_name=='k' and f.f_lineno in lines:hits[f.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:obs=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 row=record(tmp_path,name,src,{'observation':obs,'runtime_assignment_hits':dict(hits),'runtime_assignments':sum(hits.values())});assert 'error' not in obs and sum(hits.values())==0,row
 assert 'loop_dim_extent_mismatch' not in obs['layer'],row

RETURNS={
 'direct':'return Y',
 'conditional':'if j==0:\n    return Y',
 'if_true':'if True:\n    return Y',
 'if_false':'if False:\n    return Y',
 'try_finally':'try:\n    v=0\nfinally:\n    return Y',
 'except':'try:\n    raise ValueError("stop")\nexcept ValueError:\n    return Y',
 'match':'match j:\n    case 0:\n        return Y',
 'with':'with contextlib.nullcontext():\n    return Y',
 'nested_def':'def local():\n    return 0',
 'nested_lambda':'local=lambda:0',
}
@pytest.mark.parametrize('name',RETURNS)
def test_return_policy_keeps_existing_refusal(tmp_path,name):
 src=a.H+a.SIG+p.PRE+p.LOOP+'            Q[t,j]=X[t,j]\n'+''.join('            '+x+'\n' for x in RETURNS[name].splitlines())+'    return Y\n';L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');row=record(tmp_path,'return-'+name,src,{'layer':L,'analysis':A,'parsed_only':True})
 notes=L.get('loop_iterations_omitted',[])
 if name not in ['if_false','nested_def','nested_lambda']:
  assert L.get('loop_entry_unresolved') and L.get('loop_iterations_complete') is False,row
  assert not any('control_transfer_in_dim_loop' in x and ': return ' in x for x in notes),row
 else:assert not L.get('loop_entry_unresolved'),row
 assert 'loop_dim_extent_mismatch' not in L,row

COVERAGE={
 'partial_placement':'Q[t,j]=X[t,j]',
 'all_constants':'Q[0,0]=X[0,0]',
 'derived_indices':'Q[t,j+0]=X[t,j+0]',
 'unknown_tensor':'U=Y\nU[t,j]=X[t,j]',
}
@pytest.mark.parametrize('name',COVERAGE)
def test_coverage_controls(tmp_path,name):
 src=a.H+a.SIG+p.PRE+p.LOOP+''.join('            '+x+'\n' for x in COVERAGE[name].splitlines())+'    return Y\n';L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');record(tmp_path,'coverage-'+name,src,{'layer':L,'analysis':A,'parsed_only':True});assert L.get('witness_coverage') in ['full','partial','none'] if hasattr(cv,'_witness_model_v2') else True
