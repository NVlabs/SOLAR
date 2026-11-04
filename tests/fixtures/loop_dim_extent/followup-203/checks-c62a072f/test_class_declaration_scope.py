# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nested-class declarations belong to their own class, not its parent/sibling."""
import ast,json,os,pathlib
import pytest
import test_model_boundary as b
R=pathlib.Path(__file__).parent
CASES={
 'child_decl_parent_local':'class C:\n    j=0\n    class D:\n        nonlocal j\nY[t,j]=0',
 'child_decl_parent_default':'class C:\n    def f(x=(j:=0)):\n        pass\n    class D:\n        nonlocal j\nY[t,j]=0',
 'sibling_decl_sibling_local':'class C:\n    class A:\n        j=0\n    class B:\n        nonlocal j\nY[t,j]=0',
}
@pytest.mark.parametrize('name',CASES)
def test_nested_class_decl_stays_local(tmp_path,name):
 st=CASES[name]
 # Independent interpreter control uses plain integers; demonstrates the outer
 # loop index survives every class execution unchanged, including index 7.
 prefix=st.rsplit('\nY[t,j]=0',1)[0]
 src='def control():\n    seen=[]\n    for j in range(8):\n'+''.join('        '+s+'\n' for s in prefix.splitlines())+'        seen.append(j)\n    return seen\n'
 ns={};exec(src,ns);seen=ns['control']()
 d=R/'runs'/('class-declaration-'+os.environ['REVIEW_ROLE']);d.mkdir(parents=True,exist_ok=True)
 (d/(name+'.json')).write_text(json.dumps({'python_source':src,'seen':seen},indent=2))
 assert seen==list(range(8)),seen
 L,A=b.parsed(tmp_path,'class-'+name,st)
 assert 'loop_dim_extent_mismatch' in L,L
