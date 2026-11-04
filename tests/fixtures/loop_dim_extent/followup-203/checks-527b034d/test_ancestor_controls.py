# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent non-vacuous ancestor controls; a separate frozen file after the first ancestor finding."""
import collections,sys
import pytest
import test_round10_transfer as q
a=q.a;p=q.p
@pytest.mark.parametrize('transfer',['break','continue'])
def test_real_work_and_skipped_narrow_nest(tmp_path,transfer):
 body=p.PRE+'    for t in Dim("T"):\n        for j in Dim("E"):\n            Q[t,j]=X[t,j]\n    for t in Dim("T"):\n        '+transfer+'\n        for j in Dim("E"):\n            Y[t,j]=X[t,j]\n'+p.COPY+'    return Y\n';src=a.prog(body);hits=collections.Counter();lines={i for i,line in enumerate(src.splitlines(),1) if ']=' in line and ('Q[' in line or 'Y[' in line)}
 def trace(f,event,arg):
  if event=='line' and f.f_code.co_filename=='<string>' and f.f_code.co_name=='k' and f.f_lineno in lines:hits[f.f_lineno]+=1
  return trace
 sys.settrace(trace)
 try:obs=a.observe(tmp_path,src,inputs=a.INPUTS)
 finally:sys.settrace(None)
 row=q.record(tmp_path,'nonvacuous-'+transfer,src,{'observation':obs,'runtime_assignment_hits':dict(hits),'runtime_assignments':sum(hits.values())});assert 'error' not in obs and sum(hits.values())==44,row;assert 'loop_dim_extent_mismatch' not in obs['layer'],row

def test_transfer_in_separate_loop_does_not_hide_real_narrow(tmp_path):
 body=p.PRE+'    for t in Dim("T"):\n        for r in Dim("K"):\n            Y[t,r]=Z[t,r]\n            break\n    for t in Dim("T"):\n        for j in Dim("E"):\n            Y[t,j]=X[t,j]\n    return Y\n';src=a.H+a.SIG+body;L,A=a._isolated_layer(tmp_path,src,a._probe_metadata(),'k#0');row=q.record(tmp_path,'separate-loop-narrow',src,{'layer':L,'analysis':A,'parsed_only':True});assert 'loop_dim_extent_mismatch' in L,row
