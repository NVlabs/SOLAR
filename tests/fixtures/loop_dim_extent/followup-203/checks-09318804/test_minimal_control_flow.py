# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent r9 minimal-predicate controls, frozen before execution."""
from pathlib import Path
import ast,collections,json,os,sys,types
import pytest
import test_model_boundary as old
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl import einsum_converter as cv
a=old.a;R=Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE']
PRE='    Y=Tensor(T=X.shape[0],K=Z.shape[1])\n    Q=Tensor(T=X.shape[0],E=X.shape[1])\n'
LOOP='    for t in Dim("T"):\n        for j in Dim("E"):\n'
COPY='    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Q[t,r]\n'
VALID={
 'plain_control':('Q[t,j]=X[t,j]',True),
 'break_wide_counter':('Q[t,j]=X[t,j]\nbreak',True),
 'break_after_narrow':('Q[t,j]=X[t,j]\nY[t,j]=X[t,j]\nbreak',False),
 'break_before_narrow':('Q[t,j]=X[t,j]\nif j>=3:\n    break\nY[t,j]=X[t,j]',False),
 'conditional_break_after_narrow':('Q[t,j]=X[t,j]\nY[t,j]=X[t,j]\nif j==2:\n    break',False),
 'continue_filter':('Q[t,j]=X[t,j]\nif j>=3:\n    continue\nY[t,j]=X[t,j]',False),
 'continue_unreachable':('Q[t,j]=X[t,j]\ncontinue\nY[t,j]=X[t,j]',False),
 'break_unreachable':('break\nQ[t,j]=X[t,j]\nY[t,j]=X[t,j]',False),
}
def save(name,src,data):
 d=R/'runs'/('minimal-control-'+ROLE);d.mkdir(parents=True,exist_ok=True);p=d/(name+'.py');p.write_text(src)
 info=extract_kernel_info(p)['k'];stmts=[s.strip() for s in info['loop_expr'].split(';') if s.strip()]
 model=cv._witness_model_v2(types.SimpleNamespace(**info),stmts) if hasattr(cv,'_witness_model_v2') else None
 row={'source':src,'parser':info,'statements':stmts,'model':model,**data};(d/(name+'.json')).write_text(json.dumps(row,indent=2,default=str)+'\n');return row
@pytest.mark.parametrize('name',VALID)
def test_valid_control_flow_never_false_refuses(tmp_path,name):
 st,copy=VALID[name];src=a.prog(PRE+LOOP+''.join('            '+x+'\n' for x in st.splitlines())+(COPY if copy else '')+'    return Y\n')
 fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='k')
 lines={n.lineno for st in fn.body if isinstance(st,ast.For) for n in ast.walk(st) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets)};hits=collections.Counter()
 def trace(f,event,arg):
  if event=='line' and f.f_code.co_filename=='<string>' and f.f_code.co_name=='k' and f.f_lineno in lines:hits[f.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:rec=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 row=save(name,src,{'observation':rec,'runtime_assignment_hits':dict(hits),'runtime_total':sum(hits.values())})
 assert 'error' not in rec,row
 L=rec['layer'];assert 'loop_dim_extent_mismatch' not in L,row
 if L.get('loop_iterations_complete') is True:assert L.get('loop_iterations')==sum(hits.values()),row

PARSED={
 'narrow_body_with_else':(PRE+'    for t in Dim("T"):\n        for j in Dim("E"):\n            Y[t,j]=X[t,j]\n        else:\n            for j in Dim("K"):\n                Q[t,j]=X[t,j]\n    return Y\n',True),
 'narrow_orelse_only':(PRE+'    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n        else:\n            for r in Dim("E"):\n                Y[t,r]=X[t,r]\n    return Y\n',False),
 'nested_def_narrow_no_candidate':(PRE+LOOP+'            def read():\n                return Y[t,j]\n            v=read()\n    return Y\n',False),
}
@pytest.mark.parametrize('name',PARSED)
def test_domain_controls(tmp_path,name):
 body,mismatch=PARSED[name];src=a.H+a.SIG+body;L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');row=save(name,src,{'layer':L,'analysis':A,'parsed_only':True})
 assert ('loop_dim_extent_mismatch' in L)==mismatch,row
 if name=='narrow_orelse_only':assert L.get('loop_iterations_complete') is False and any('else' in x for x in L.get('loop_iterations_omitted',[])),row
 if name=='nested_def_narrow_no_candidate' and hasattr(cv,'_witness_model_v2'):assert L.get('witness_coverage')=='full' and not L.get('witness_unmodelled'),row
