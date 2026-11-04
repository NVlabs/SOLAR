# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-3 controls for the fallback dispatcher (an earlier review), carried
verbatim into the repository.

Test bodies are the reviewer's (``checks-9623e5ae/test_round3_adversarial.py``); only the
header differs: fixtures resolve inside the repository and evidence is written under
pytest's tmp_path instead of the reviewer's evidence directory.  Fake transports never
connect; no provider is constructed.

ONE policy adaptation, named here: ``test_usage_absent_finite_cap[True]`` asserted
``not r.verified`` for a transport with NO cost channel under an explicit pre-transport
per-request bound.  The reviewer recorded that stricter assertion as nonblocking ("an
established caller bound can legitimately support unmeasured usage"); the repaired
contract ADMITS at the bound and RECORDS the bound used (``cost.bound_used_usd``), and the
``[False]`` case (no explicit bound) stays fail-closed.  The verbatim reviewer file is
kept failing honestly in the delivery evidence.
"""
import json
import math
import pathlib

import numpy as np
import pytest

from solar.frontend import dispatcher as D
from solar.einsum_lib.transport_budget import charge_transport, record_transport_usage
from test_fallback_dispatcher_review7_round2_controls import (  # noqa: F401  (fixtures)
    _OUT, _out_dir, req, policy, provider, translate_good, verified, det, agent, Clock,
    adapter_case, sha)


def save(name, r=None, **kw):
    out = _OUT["path"].parent / "round3-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / (name + '.json')).write_text(json.dumps({'result': r.to_dict() if r else None, **kw},
                                                   indent=2, default=str) + '\n')


@pytest.mark.parametrize('changed',[False,True])
def test_executed_reference_bound_not_only_get_source(tmp_path,changed):
 rq,base,original=adapter_case(tmp_path)
 def consume(path):
  base(path)
  q=json.loads(pathlib.Path(path).read_text())
  if changed:
   p=pathlib.Path(q['adapter_path'])
   p.write_text(p.read_text().replace('exec(get_source(), ns)', 'exec(get_source().replace("return x * w + 1.0", "return x * w + 2.0"), ns)'))
   dsl=pathlib.Path(q['candidate_path']);dsl.write_text(dsl.read_text().replace('X[b, h] * W[h] + 1.0','X[b, h] * W[h] + 2.0'))
   response=pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json'
   ans=json.loads(response.read_text());ans['candidate_sha256']=sha(dsl);response.write_text(json.dumps(ans))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 p=pathlib.Path(r.attempts[0]['request_dir'])/'adapter.py'
 orig={};actual={};exec(original,orig);exec(p.read_text(),actual)
 inputs=orig['get_inputs']()
 delta=float(np.max(np.abs(orig['kernel_fn'](inputs)['output']-actual['kernel_fn'](inputs)['output'])))
 receipt=json.loads(pathlib.Path(r.validation['receipt_path']).read_text()) if r.validation and r.validation.get('receipt_path') else None
 save('actual-reference-'+str(changed),r,delta=delta,receipt=receipt,get_source_equal=orig['get_source']()==actual['get_source']())
 assert delta==(1.0 if changed else 0.0)
 assert not r.verified if changed else r.verified
@pytest.mark.parametrize('tokens',['unavailable',None,-1])
def test_malformed_tokens_preserve_debits(req,tokens):
 calls=[]
 def translate(*a,**kw):
  charge_transport('fake');calls.append(1);record_transport_usage('fake',{'total_tokens':tokens})
  return translate_good(*a,**kw)
 res=D.LedgerReservation(3,1,'fake')
 r=D.dispatch(req,policy(max_attempts=2,max_requests=1),provider(translate),reservation=res,deterministic_stage=det,validator=verified)
 save('malformed-tokens-'+str(tokens),r,calls=len(calls),reservation=res.to_dict())
 assert len(calls)<=1 and r.ledger['api_requests_used']==len(calls)
@pytest.mark.parametrize('explicit_bound',[False,True])
def test_usage_absent_finite_cap(req,explicit_bound):
 calls=[]
 def translate(*a,**kw):
  charge_transport('fake');calls.append(1);record_transport_usage('fake',None);return translate_good(*a,**kw)
 p=policy(max_cost_usd=.1)
 if not explicit_bound:p.cost_per_request_usd=None
 r=D.dispatch(req,p,provider(translate),reservation=D.LedgerReservation(1,.1,'fake'),deterministic_stage=det,validator=verified)
 save('usage-absent-'+str(explicit_bound),r,calls=len(calls))
 if not explicit_bound:
  assert not r.verified
  assert any(f.get('reason_code')=='cost_unbounded' for f in r.failure_receipts)
 else:
  # policy adaptation (reviewer: nonblocking): an explicit pre-transport bound
  # supports unmeasured usage, and the bound used is RECORDED
  assert r.verified and r.candidate['cost_basis']=='hard_cap_preauthorized'
  cost=r.candidate['provenance']['cost']
  assert cost['bound_used_usd']==0.1 and cost['measured'] is False and cost['bound_source']
@pytest.mark.parametrize('cost',[float('nan'),float('inf'),-.25])
def test_invalid_cost_cannot_verify_or_credit(req,cost):
 def translate(*a,**kw):
  charge_transport('fake');r=translate_good(*a,**kw);r.cost_usd=cost;return r
 res=D.LedgerReservation(1,.1,'fake')
 r=D.dispatch(req,policy(max_cost_usd=.1),provider(translate),reservation=res,deterministic_stage=det,validator=verified)
 save('invalid-cost-'+str(cost),r,reservation=res.to_dict())
 assert not r.verified and math.isfinite(res.usd_used) and res.usd_used>=0
def test_cost_overrun_does_not_invent_transport(req):
 calls=[]
 def translate(*a,**kw):
  charge_transport('fake');calls.append(1);r=translate_good(*a,**kw);r.cost_usd=.25;return r
 res=D.LedgerReservation(2,1,'fake')
 r=D.dispatch(req,policy(max_cost_usd=.1),provider(translate),reservation=res,deterministic_stage=det,validator=verified)
 save('phantom-request',r,calls=len(calls),reservation=res.to_dict())
 assert r.ledger['api_requests_used']==len(calls)==res.requests_used
@pytest.mark.parametrize('exc',[ValueError,RuntimeError])
def test_resume_validation_fault_has_fresh_durable_receipt(req,exc):
 first=D.dispatch(req,D.DispatchPolicy(),D.PureSkillProvider(),deterministic_stage=det)
 agent()(pathlib.Path(first.resume_token)/'TRANSLATION_REQUEST.json')
 def fail(*a):raise exc('offline validation storage decoder failure')
 result=None;error=None
 try:result=D.resume(req.out_dir,D.PureSkillProvider(),validator=fail)
 except Exception as e:error=repr(e)
 disk=json.loads((req.out_dir/'DISPATCH_RECEIPT.json').read_text())
 save('resume-fault-'+exc.__name__,result,error=error,disk=disk)
 assert result is not None and not result.verified and disk['status']!='awaiting_agent'
def test_cross_process_clock_rollback_cannot_extend_wall(req,monkeypatch):
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 first=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(wall_seconds=5)),D.PureSkillProvider(),deterministic_stage=det)
 agent()(pathlib.Path(first.resume_token)/'TRANSLATION_REQUEST.json')
 realos=D.os
 class Proxy:
  def __getattr__(self,name):return getattr(realos,name)
  def getpid(self):return realos.getpid()+100000
 monkeypatch.setattr(D,'os',Proxy())
 clock.epoch=103;clock.mono=1006
 r=D.resume(req.out_dir,D.PureSkillProvider(),validator=verified)
 save('cross-process-clock-rollback',r,epoch_elapsed=3,monotonic_elapsed=6,wall_cap=5)
 assert not r.verified
