# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent routing/ABI tests; API client is always mocked."""
import inspect, json, os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import test_translation_agent as a
from solar.einsum_lib import agent as ag, verify as v, numba_accel as na

R=Path(__file__).parent
def save(name, row):
    d=R/'runs'/('agent-boundary-'+os.environ['REVIEW_ROLE']);d.mkdir(parents=True,exist_ok=True)
    (d/(name+'.json')).write_text(json.dumps(row,indent=2,default=str)+'\n')

@pytest.mark.parametrize('reason', [
    'RuntimeError: lowered execution failed',
    'lowered code failed to import: NameError: missing',
    'numba_lower could not lower the generated code',
])
@pytest.mark.parametrize('old_policy', [False,True])
def test_untyped_failure_is_not_the_typed_decline_policy(reason,old_policy):
    ref,inputs=a._matmul_reference()
    agent,client=a._make_agent(max_retries=1,reprompt_on_numba_decline=old_policy)
    client.chat.completions.create.return_value=a._mock_openai_response(f'```python\n{a.GOOD_MATMUL_CODE}\n```')
    def failure(*args,**kwargs):
        kwargs['on_fallback'](reason)
        return None
    with patch.object(na,'execute_fast',side_effect=failure):
        result=agent.translate('x @ w',a.MATMUL_SHAPES,reference_outputs=ref,all_inputs=inputs)
    row={'reason':reason,'old_policy':old_policy,'verified':result.verified,'status':result.verification_status,'attempts':result.attempts,'error':result.verification_error}
    save('untyped-'+str(old_policy)+'-'+reason.split(':')[0].replace(' ','_'),row)
    assert not result.verified and result.verification_status=='execution_failed',row

def test_old_positional_constructor_contract():
    # Positional arguments valid before commit B, through mem_budget_bytes.
    args=['fake-key',None,None,'test-model',0.2,1,None,3,60,True,True,10000,True,'pytorch',True,False,17,123456789]
    sig=inspect.signature(ag.TranslationAgent)
    bound=sig.bind(*args);bound.apply_defaults()
    save('positional-contract',dict(bound.arguments))
    assert bound.arguments['source_type']=='pytorch'
    assert bound.arguments['use_fp64'] is True
    assert bound.arguments['verify_timeout']==17
    assert bound.arguments['mem_budget_bytes']==123456789

def test_default_typed_decline_keeps_existing_isolation():
    ref,inputs=a._matmul_reference()
    agent,client=a._make_agent(max_retries=1,verify_timeout=30,mem_budget_bytes=1024**3)
    client.chat.completions.create.return_value=a._mock_openai_response(f'```python\n{a.DECLINING_GOOD_CODE}\n```')
    original=v._run_isolated;calls=[]
    def record(fn,timeout,mem_budget_bytes,*args,**kwargs):
        calls.append({'function':fn.__name__,'timeout':timeout,'mem_budget_bytes':mem_budget_bytes})
        return original(fn,timeout,mem_budget_bytes,*args,**kwargs)
    with patch.object(v,'_run_isolated',side_effect=record):
        result=agent.translate('x @ w',a.MATMUL_SHAPES,reference_outputs=ref,all_inputs=inputs)
    save('isolation',{'calls':calls,'verified':result.verified,'status':result.verification_status,'attempts':result.attempts})
    assert result.verified and result.attempts==1
    assert [x['function'] for x in calls]==['_isolated_execute_fast','_isolated_run_model']
    assert all(x['timeout']==30 and x['mem_budget_bytes']==1024**3 for x in calls)
