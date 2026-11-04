# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen independent scope-resolution tests: nested lambda/comprehension bindings are not bindings in their containing function."""
import pytest
from test_round11_compatible import record,H,MODEL,SIDE,END
SCOPES={
 "lambda_parameter":"    unused=lambda model:model\n",
 "lambda_walrus":"    unused=lambda: (model:=0)\n",
 "list_target":"    unused=[model for model in ()]\n",
 "generator_target":"    unused=(model for model in ())\n",
 "dict_target":"    unused={model:model for model in ()}\n",
 "set_target":"    unused={model for model in ()}\n",
}
@pytest.mark.parametrize('fp64',[False,True])
@pytest.mark.parametrize('name',SCOPES)
def test_nested_scope_does_not_hide_global_escape(name,fp64):
 code=H+MODEL+'def mutate():\n'+SCOPES[name]+'    alias=model\n    alias.__code__=replacement.__code__\nmutate()\n'+SIDE+END
 row=record('escape-'+name,code,fp64)
 assert row['slow']=={'value':{'output':5}},row
 assert row['public']==row['slow'],row
 assert not row['lowered'] and row['reasons'],row
