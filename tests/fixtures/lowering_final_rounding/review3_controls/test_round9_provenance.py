# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent r9 probes. Assertions fixed before execution; no source edits."""
import json, os
from pathlib import Path
import numpy as np
import pytest
from solar.einsum_lib import clear_dims, stop_trace
from solar.einsum_lib.numba_accel import numba_lower, execute_fast
from solar.einsum_lib.verify import execute_translated

R = Path(__file__).parent
H = '''from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
import numpy as np
class Box:
    data = 5
    def __int__(self):
        return 9
@kernel
def k(A: Tensor("M,N", dtype="fp32")) -> Tensor("M,N", dtype="fp32"):
    Y = Tensor(M=1, N=2, dtype="fp32", fill=2.0)
    return Y
'''
END = '''def run_model(inputs):
    clear_dims()
    return {"output": np.array(SIDE)}
'''
CASES = {
    'kernel_local_shadow': '''def run_model(inputs):
    clear_dims()
    k = lambda a: Box()
    return {"output": np.array(int(k(inputs["a"]).data))}
''',
    'kernel_parameter_return': '''def model(a, k):
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"], lambda a: Box())
    return {"output": np.array(int(out.data))}
''',
    'kernel_default_return': '''def model(a, k=lambda a: Box()):
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"])
    return {"output": np.array(int(out.data))}
''',
    'kernel_local_return': '''def model(a):
    k = lambda a: Box()
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"])
    return {"output": np.array(int(out.data))}
''',
    'kernel_class_shadow': '''class C:
    k = lambda a: Box()
    SIDE = int(k(np.ones((1,2))).data)
SIDE = C.SIDE
''' + END,
    'plain_class_shadow': '''def model(a):
    return k(a)
class C:
    model = lambda a: Box()
    SIDE = int(model(np.ones((1,2))).data)
SIDE = C.SIDE
''' + END,
    'plain_decorated': '''def deco(f):
    return lambda a: Box()
@deco
def model(a):
    return k(a)
SIDE = int(model(np.ones((1,2))).data)
''' + END,
    'plain_varargs': '''def model(*a):
    return k(a[0])
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"])
    return {"output": out.data}
''',
    'plain_safe_recursion': '''def model(a, n):
    if n:
        return model(a, n-1)
    return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"], 1)
    return {"output": out.data}
''',
    'plain_fallthrough': '''def model(a):
    if a is not None:
        return k(a)
def run_model(inputs):
    clear_dims()
    out = model(inputs["a"])
    return {"output": out.data}
''',
    'plain_parameter_unestablished': '''def model(x):
    return x
SIDE = int(model(Box()).data)
''' + END,
    'plain_callee_rebound': '''def model(a):
    return k(a)
model = lambda a: Box()
SIDE = int(model(np.ones((1,2))).data)
''' + END,
    'plain_code_replaced': '''def model(a):
    return k(a)
def replacement(a):
    return Box()
model.__code__ = replacement.__code__
SIDE = int(model(np.ones((1,2))).data)
''' + END,
}

def outcome(fn):
    clear_dims(); stop_trace()
    try:
        return {'value': {k: np.asarray(v).tolist() for k,v in fn().items()}}
    except Exception as e:
        return {'error': type(e).__name__, 'detail': str(e)}
    finally:
        clear_dims(); stop_trace()

@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('fp64', [False, True])
def test_provenance(case, fp64):
    code = H + CASES[case]
    compile(code, '<probe>', 'exec')
    inputs={'a': np.ones((1,2))}; reasons=[]; fallback=[]
    low = numba_lower(code, use_fp64=fp64, boundscheck=True, on_decline=reasons.append)
    slow = outcome(lambda: execute_translated(code, {}, all_inputs=inputs, fast=False)[0])
    raw = outcome(lambda: execute_fast(code, inputs, use_fp64=fp64, on_fallback=fallback.append))
    public = outcome(lambda: execute_translated(code, {}, all_inputs=inputs, fast=True, use_fp64=fp64)[0])
    row = dict(case=case, fp64=fp64, code=code, lowered=low is not None, reasons=reasons, fallback=fallback,
               slow=slow, raw=raw, public=public)
    out=R/'runs'/('round9-'+os.environ['REVIEW_ROLE']);out.mkdir(parents=True,exist_ok=True)
    stem=case+'-'+str(fp64);(out/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    (out/(stem+'.py')).write_text(code)
    if low is not None:(out/(stem+'.lowered.py')).write_text(low)
    assert 'value' in slow, row
    assert public == slow, row
    assert low is None and any(x.startswith('kernel_') for x in reasons), row
