# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Annotated controls separate lexical regression from old unannotated rounding."""
import pytest
import test_round3_scope as original

@pytest.mark.parametrize('name',['comprehension_local','lambda_body_local_walrus'])
@pytest.mark.parametrize('fp64',[False,True])
def test_annotated_lexical_positive(name,fp64):
    code,expected=original.CASES[name]
    code=code.replace('def k(A):','def k(A:Tensor("M,N")) -> Tensor("M,N"):')
    code=code.replace('return {"out":k(inputs["a"]).data}',
                      'a=Tensor.from_numpy(inputs["a"],"M,N")\n    return {"out":k(a).data}')
    key='annotated_'+name
    original.CASES[key]=(code,expected)
    r=original.observe(key,fp64)
    assert r['slow']['value']['out']['data']==[[2.0,2.0]],r
    assert r['public']['value']['out']['data']==[[2.0,2.0]],r
    assert r['lowered'] and not r['declines'] and not r['fallback'],r
