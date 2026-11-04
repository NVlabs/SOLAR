# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools,json
import pytest,torch
from solar.graph import _torchview_param_shim as shim
from test_sdpa_scalar_kwargs_capture import _source,_capture

def test_mixed_tensor_and_mapping_keyword_must_refuse(tmp_path):
 def legacy(input,mat2):return input+mat2
 @functools.wraps(legacy)
 def adapter(input,mat2,**kw):return input+mat2+kw['payload']['extra']
 x,y,z=[torch.ones(2,2) for _ in range(3)]
 assert torch.equal(adapter(x,mat2=y,payload={'extra':z},unknown_scalar=.5),x+y+z)
 kwargs={'mat2':y,'payload':{'extra':z},'unknown_scalar':.5}
 leaves=shim._semantic_operand_leaves(adapter,(x,),kwargs)
 info=shim.bind_operand_leaves(adapter,(x,),kwargs)[1] if hasattr(shim,'bind_operand_leaves') else None
 (tmp_path/'binding.json').write_text(json.dumps({'leaf_shapes':None if leaves is None else [list(t.shape) for t in leaves if isinstance(t,torch.Tensor)],'info':info},indent=2))
 assert leaves is None,'nested tensor payload must not be silently classified and dropped as a scalar'

@pytest.mark.parametrize('force_cpu',[False,True],ids=['meta','forced-cpu'])
def test_capture_nested_tensor_retry_refuses(tmp_path,force_cpu):
 preamble='''import functools
def legacy(input,mat2):
    return input+mat2
@functools.wraps(legacy)
def adapter(input,mat2,**kwargs):
    extra=kwargs['payload']['extra']
    if torch.overrides.has_torch_function_variadic(input,mat2,extra):
        return torch.overrides.handle_torch_function(adapter,(input,mat2,extra),input,mat2=mat2,**kwargs)
    return input+mat2+extra
'''
 ok,out,graph=_capture(tmp_path,_source("adapter(q,mat2=k,payload={'extra':v},unknown_scalar=.5)",force_cpu,preamble))
 nodes=[] if graph is None else [v for v in graph['layers'].values() if v.get('type')=='legacy']
 (tmp_path/'observations.json').write_text(json.dumps({'ok':ok,'nodes':nodes,'stdout':out},indent=2))
 assert ok is False,'the graph must refuse the tensor hidden in the unplaceable payload keyword'
 assert 'UnsupportedFrontendOperand' in out
