# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json,os,pathlib
import numpy as np
import pytest
import test_binding_edges as original

NAMES=['nested_assignment_only','nested_parameter_only','lambda_parameter_only',
       'class_member_only','reexport_in_allowed_namespace','reexport_external_module']

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('fp64',[False,True])
def test_annotated_positive_binding(name,fp64):
    # A separate variant, preserving the original unannotated records and assertions.
    code,expect=original.CASES[name]
    code=code.replace('def k(A):','def k(A:Tensor("M,N")) -> Tensor("M,N"):')
    code=code.replace('return {"out":k(inputs["a"]).data}',
                      'a=Tensor.from_numpy(inputs["a"],"M,N")\n    return {"out":k(a).data}')
    newname='annotated_'+name;original.CASES[newname]=(code,expect)
    r=original.observe(newname,fp64)
    assert r['slow']['value']['out']['data']==[[2.0,2.0]],r
    assert r['public']['value']['out']['data']==[[2.0,2.0]],r
    assert r['lowered'] and not r['declines'] and not r['fallback'],r
