# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent legacy object-identity boundary, frozen before execution."""
import dataclasses,json,os,pathlib
import pytest
import test_site_flow as p
from solar.einsum_dsl.ast_parser import extract_kernel_info
from solar.einsum_dsl.model import DSLKernel,DSLTensor,DSLInput
from solar.einsum_dsl import einsum_converter as cv
R=pathlib.Path(__file__).parent;ROLE=os.environ['REVIEW_ROLE']
BODY=p.N+p.Q+'    for t in Dim("T"):\n        for e in Dim("E"):\n            Q[t,e]=X[t,e]\n    for t in Dim("T"):\n        for e in Dim("K"):\n            Y[t,e]=Z[t,e]\n    return Y\n'
@pytest.mark.parametrize('degrade',['none','no_return_info','no_return_and_source'])
def test_valid_legacy_scratch(tmp_path,degrade):
 code=p.a.prog(BODY);obs=p.a.observe(tmp_path,code,inputs=p.a.INPUTS);p.a.counted(obs,44);f=tmp_path/'probe.py';f.write_text(code);info=extract_kernel_info(f)['k'];fields={x.name for x in dataclasses.fields(DSLKernel)}
 ins=[DSLInput(DSLTensor('x',{'T':4,'E':8},'fp32','input'),'input'),DSLInput(DSLTensor('z',{'T':4,'K':3},'fp32','input'),'input')]
 k=DSLKernel('k#0','k',ins,DSLTensor('k#0',{'T':4,'K':3},'fp32',''),**{n:v for n,v in info.items() if n in fields});k.registration_order=[0,1];k.inputs_in_signature_order=True;k.loop_source_parsed=True
 if degrade!='none':k.returns=None;k.returned_names=None
 if degrade=='no_return_and_source':k.source_text=None
 row={'degradation':degrade,'fresh_runtime':obs,'expected_assignments':44}
 try:row['counter']=cv._kernel_loop_iterations(k)
 except Exception as exc:row.update(error=type(exc).__name__,code=getattr(exc,'code',None),message=str(exc))
 d=R/'runs'/('legacy-identity-'+ROLE);d.mkdir(parents=True,exist_ok=True);(d/(degrade+'.json')).write_text(json.dumps(row,indent=2,default=str)+'\n')
 assert row.get('code')!='loop_dim_extent_mismatch',row
 assert row.get('counter',[None])[0]==44,row
