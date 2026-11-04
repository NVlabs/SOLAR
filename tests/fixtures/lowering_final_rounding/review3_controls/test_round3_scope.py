# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent round-3 controls. No edits to author or prior reviewer tests."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest
import solar.einsum_lib as el
import solar.einsum_lib.core as core
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
import test_binding_edges as old

R = Path(__file__).parent
H, K, Y, RUN = old.H, old.K, old.Y, old.RUN

def program(body='', prefix='', header=H, tail=Y, run=RUN):
    return header + prefix + K + body + tail + run

CASES = {
    'module_function_local': (program(prefix='def unused():\n    Tensor=int\n'), 'lower'),
    'module_type_checking': (program(prefix='from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    Tensor=int\n'), 'decline'),
    'module_try_reimport': (program(prefix='try:\n    from solar.einsum_lib import Tensor\nexcept ImportError:\n    from solar.einsum_lib.core import Tensor\n'), 'lower'),
    'module_attribute_write': (program(prefix='import solar.einsum_lib as m\nm.Tensor=int\n'), 'decline'),
    'module_star_import': (program(prefix='from review4_tensor_reexport import *\n'), 'decline'),
    'nested_local': (program('    def helper():\n        Tensor=int\n        return 2\n    unused=helper()\n'), 'lower'),
    'nested_nonlocal_declaration': (program('    from solar.einsum_lib import Tensor\n    def helper():\n        nonlocal Tensor\n        return 2\n    unused=helper()\n'), 'decline'),
    'comprehension_local': (program('    unused=[Tensor for Tensor in range(3)]\n'), 'lower'),
    'kernel_with_binding': (program('    with nullcontext(int) as Tensor:\n        pass\n', prefix='from contextlib import nullcontext\n'), 'decline'),
    'kernel_except_binding': (program('    try:\n        raise ValueError("probe")\n    except ValueError as Tensor:\n        pass\n'), 'decline'),
    'lambda_default_outer_walrus': (program('    unused=lambda x=(Tensor := int): x\n'), 'decline'),
    'lambda_body_local_walrus': (program('    unused=lambda: (Tensor := int)\n'), 'lower'),
    'module_argument_annotation_walrus': (program(prefix='def unrelated(x: (Tensor := int)):\n    pass\n'), 'decline'),
    'module_kwonly_annotation_walrus': (program(prefix='def unrelated(*, x: (Tensor := int)):\n    pass\n'), 'decline'),
    'module_vararg_annotation_walrus': (program(prefix='def unrelated(*x: (Tensor := int)):\n    pass\n'), 'decline'),
    'module_return_annotation_walrus': (program(prefix='def unrelated() -> (Tensor := int):\n    pass\n'), 'decline'),
    'module_default_walrus': (program(prefix='def unrelated(x=(Tensor := int)):\n    pass\n'), 'decline'),
    'module_called_global_writer': (program(prefix='def change():\n    global Tensor\n    Tensor=int\nchange()\n'), 'decline'),
    'module_uncalled_global_writer': (program(prefix='def change():\n    global Tensor\n    Tensor=int\n'), 'lower'),
    'module_class_global_writer': (program(prefix='class Change:\n    global Tensor\n    Tensor=int\n'), 'decline'),
    'unannotated_different_dtypes': (program(tail='    if A[0,0] > 0:\n        Y=Tensor(M=1,N=2,dtype="bf16",fill=2.003)\n    else:\n        Y=Tensor(M=1,N=2,dtype="fp32",fill=2.003)\n    return Y\n'), 'decline'),
    'unannotated_tuple': (program(tail='    Y=Tensor(M=1,N=2,dtype="bf16",fill=2.003)\n    Z=Tensor(M=1,N=2,dtype="fp32",fill=2.003)\n    return Y,Z\n', run='\ndef run_model(inputs):\n    clear_dims()\n    y,z=k(inputs["a"])\n    return {"out":y.data,"wide":z.data}\n'), 'decline'),
    'unannotated_parameter': (program(tail='    return A\n', run='\ndef run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs["a"],"M,N")\n    return {"out":k(a).data}\n'), 'decline'),
    'unannotated_fp32': (program(tail=Y.replace('"bf16"','"fp32"')), 'lower'),
}

def isolated_call(f):
    # The attribute-write probe may mutate the imported module in THIS process.
    # Restore its original object before the next path; no disk mutation.
    tensor = core.Tensor
    saved = el.Tensor
    try:
        core.clear_dims()
        core.stop_trace()
        return old.call(f)
    finally:
        el.Tensor = saved
        core.Tensor = tensor
        core.clear_dims()
        core.stop_trace()

def observe(name, fp64):
    code, expected = CASES[name]
    why, fallback = [], []
    low = na.numba_lower(code, use_fp64=fp64, on_decline=why.append)
    try:
        na._extract_kernel_output_dtypes(ast.parse(code))
        exception = None
    except Exception as exc:
        exception = {'type':type(exc).__name__, 'reason':getattr(exc,'reason',None),
                     'subcode':getattr(exc,'subcode',None),'message':str(exc)}
    inputs = {'a':np.full((1,2),2.003,dtype=np.float64)}
    slow = isolated_call(lambda: execute_translated(code,{},all_inputs=inputs,fast=False))
    raw = isolated_call(lambda: na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append))
    public = isolated_call(lambda: execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64))
    row = dict(name=name,fp64=fp64,expected=expected,code=code,lowered=isinstance(low,str),
               declines=why,exception=exception,fallback=fallback,slow=slow,raw=raw,public=public)
    dest=R/'runs'/('scope-'+os.environ['REVIEW_ROLE']);dest.mkdir(exist_ok=True)
    stem=name+'-'+str(fp64)
    (dest/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    (dest/(stem+'.py')).write_text(code)
    if isinstance(low,str):(dest/(stem+'.lowered.py')).write_text(low)
    return row

@pytest.mark.parametrize('name',CASES)
@pytest.mark.parametrize('fp64',[False,True])
def test_round3_scope_and_return(name,fp64):
    row=observe(name,fp64)
    if 'error' in row['slow']:
        assert row['public'].get('error')==row['slow']['error'],row
    else:
        assert 'value' in row['public'],row
        for key,v in row['slow']['value'].items():
            expected=np.asarray(v['data'])
            if not fp64 and not row['fallback']:expected=expected.astype(np.float32)
            np.testing.assert_array_equal(row['public']['value'][key]['data'],expected,err_msg=str(row))
    if row['expected']=='lower':
        assert row['lowered'] and not row['declines'] and not row['fallback'],row
    else:
        assert not row['lowered'] and row['declines'],row
        assert all(s.startswith('kernel_') for s in row['declines']),row
    if name in ('unannotated_parameter','unannotated_fp32'):
        assert row['public']['value']['out']['data'][0][0] != 2.0,row
