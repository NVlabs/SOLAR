# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-4 controls for the fallback dispatcher (an earlier review), carried
verbatim into the repository (``checks-5c5d9c75/test_round4_adversarial.py``).  Only the
header differs: fixtures resolve inside the repository and evidence is written under
pytest's tmp_path.  Fake transports never connect; no GPU; no environment change.

These are the PROOF-INTEGRITY controls: nothing the candidate or adapter executes or
imports may touch any input, output, tolerance, comparator or dependency of the trusted
side (INPUT-21, TOLERANCE-22, ISOLATION-23), plus unknown boot identity (WALL-15) and
post-reconciliation cost coherence (ACCOUNTING-24).
"""
import json
import pathlib
import sys

import numpy as np
import pytest

from solar.frontend import dispatcher as D
from solar.einsum_lib.transport_budget import charge_transport, record_transport_usage
from test_fallback_dispatcher_review7_round2_controls import (  # noqa: F401  (fixtures)
    _OUT, _out_dir, req, policy, provider, translate_good, verified, det, agent, Clock,
    adapter_case, sha)


def save(name, r=None, **kw):
    out = _OUT["path"].parent / "round4-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / (name + '.json')).write_text(json.dumps({'result': r.to_dict() if r else None, **kw},
                                                   indent=2, default=str) + '\n')


def adapted(tmp_path,adapter_edit=lambda a:a,dsl_edit=lambda d:d,source_edit=None):
 rq,base,original=adapter_case(tmp_path)
 reference=rq.source_path.read_text()
 if source_edit:
  reference=source_edit(reference);rq.source_path.write_text(reference)
 def consume(path):
  base(path);q=json.loads(pathlib.Path(path).read_text())
  ap=pathlib.Path(q['adapter_path']);ds=pathlib.Path(q['candidate_path'])
  a=ap.read_text()
  if source_edit:a=a.replace('_REFERENCE_SRC = (','_REFERENCE_SRC = '+repr(reference)+'\n_UNUSED = (')
  ap.write_text(adapter_edit(a));ds.write_text(dsl_edit(ds.read_text()))
  rp=pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json'
  response=json.loads(rp.read_text());response['candidate_sha256']=sha(ds);rp.write_text(json.dumps(response))
 return rq,consume,original
def diverge(a):
 return a.replace('exec(get_source(), ns)','exec(get_source().replace("return x * w + 1.0", "return x * w + 2.0"), ns)')
def wrongdsl(d):return d.replace('X[b, h] * W[h] + 1.0','X[b, h] * W[h] + 2.0')
@pytest.mark.parametrize('mutates',[False,True])
def test_trusted_comparison_uses_pre_candidate_inputs(tmp_path,mutates):
 def dsl(d):
  d=wrongdsl(d)
  if mutates:d=d.replace('return {"output": model(x, w).data}','result = model(x, w).data\n    inputs["x"][:] += 1.0 / inputs["w"]\n    return {"output": result}')
  return d
 rq,consume,original=adapted(tmp_path,diverge,dsl)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 ns={};exec(original,ns);inputs=ns['get_inputs']()
 original_outputs=ns['kernel_fn'](inputs)['output']
 wrong_outputs=inputs['x']*inputs['w']+2.0
 delta=float(np.max(np.abs(wrong_outputs-original_outputs)))
 save('mutated-inputs-'+str(mutates),r,original_reference_error=delta)
 assert delta==1.0
 assert not r.verified
@pytest.mark.parametrize('patch_target',['reference_module','comparison'])
def test_adapter_parent_monkeypatch_cannot_fake_trusted_pass(tmp_path,monkeypatch,patch_target):
 import solar.einsum_lib.verify as V
 original_compare=V.compare_outputs
 def edit(a):
  a=diverge(a)
  if patch_target=='reference_module':
   return a+'\nimport sys, types\n_shadow=types.ModuleType("__trusted_reference__")\n_shadow.run=lambda x,w:x*w+2.0\nsys.modules["__trusted_reference__"]=_shadow\n'
  return a+'\nimport solar.einsum_lib.verify as _v\nfrom types import SimpleNamespace as _NS\n_v.compare_outputs=lambda *args,**kwargs:_NS(success=True,max_abs_error=0.0,vacuous_outputs=(),error=None)\n'
 rq,consume,_=adapted(tmp_path,edit,wrongdsl)
 try:r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 finally:
  V.compare_outputs=original_compare;sys.modules.pop('__trusted_reference__',None)
 save('parent-patch-'+patch_target,r)
 assert not r.verified
def test_get_source_different_is_refused_before_trusted_check(tmp_path):
 rq,consume,_=adapter_case(tmp_path,wrong_reference=True)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 save('different-get-source',r)
 assert not r.verified
@pytest.mark.parametrize('boot_known',[True,False])
def test_unknown_boot_cannot_fall_back_to_epoch_only(req,monkeypatch,boot_known):
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 monkeypatch.setattr(D,'_boot_id',lambda:'fixed-boot' if boot_known else None)
 first=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(wall_seconds=5)),D.PureSkillProvider(),deterministic_stage=det)
 agent()(pathlib.Path(first.resume_token)/'TRANSLATION_REQUEST.json')
 realos=D.os
 class Proxy:
  def __getattr__(self,k):return getattr(realos,k)
  def getpid(self):return realos.getpid()+100000
 monkeypatch.setattr(D,'os',Proxy());clock.epoch=103;clock.mono=1006
 r=D.resume(req.out_dir,D.PureSkillProvider(),validator=verified)
 save('boot-known-'+str(boot_known),r)
 assert not r.verified
@pytest.mark.parametrize('phase',['exit_code','log'])
def test_post_reconcile_fault_preserves_measured_cost(req,phase):
 class Result:
  cost_usd=.03
  verification_status='verified'
  @property
  def exit_code(self):
   if phase=='exit_code':raise ValueError('offline result exit-code decoding')
   return 7
  @property
  def log(self):raise ValueError('offline log decoding')
 def translate(*a,**kw):
  charge_transport('fake');translate_good(*a,**kw);return Result()
 reservation=D.LedgerReservation(1,.1,'fake')
 r=D.dispatch(req,policy(max_requests=1,max_cost_usd=.1),provider(translate),reservation=reservation,deterministic_stage=det,validator=verified)
 save('post-reconcile-'+phase,r,reservation=reservation.to_dict())
 assert r.ledger['api_requests_used']==reservation.requests_used==1
 assert r.ledger['cost_usd_used']==pytest.approx(reservation.usd_used)==pytest.approx(.03)
@pytest.mark.parametrize('narrow_output',[False,True])
def test_adapter_cannot_widen_original_reference_tolerance(tmp_path,narrow_output):
 def edit(a):
  a=a.replace('exec(get_source(), ns)','exec(get_source().replace("return x * w + 1.0", "return (x * w + 1.0) * 1.001"), ns)')
  if narrow_output:a=a.replace('.numpy()}', '.numpy().astype(np.float16)}')
  return a
 def dsl(d):return d.replace('X[b, h] * W[h] + 1.0','(X[b, h] * W[h] + 1.0) * 1.001')
 rq,consume,original=adapted(tmp_path,edit,dsl)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 ns={};exec(original,ns);inputs=ns['get_inputs']()
 ref=ns['kernel_fn'](inputs)['output'];wrong=(inputs['x']*inputs['w']+1.0)*1.001
 strict=bool(np.allclose(wrong,ref,rtol=rq.tolerance['rtol'],atol=rq.tolerance['atol']))
 save('untrusted-tolerance-'+str(narrow_output),r,approved_tolerance=rq.tolerance,strict_original_pass=strict,max_original_error=float(np.max(np.abs(wrong-ref))))
 assert not strict and not r.verified

def test_reference_import_cannot_readd_excluded_adapter_path(tmp_path):
 adir=tmp_path/'dispatch/agentic/attempt_01'
 def source_edit(s):
  return s.replace('    return x * w + 1.0','    import sys\n    sys.path.insert(0, '+repr(str(adir))+')\n    import review7_injected_helper\n    return x * w + review7_injected_helper.OFFSET')
 rq,base,_=adapted(tmp_path,lambda a:a,wrongdsl,source_edit)
 def consume(path):
  pathlib.Path(path).parent.joinpath('review7_injected_helper.py').write_text('OFFSET=2.0\n')
  base(path)
 oldpath=list(sys.path)
 try:r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 finally:
  sys.path[:]=oldpath;sys.modules.pop('review7_injected_helper',None)
 save('reference-import-side-effect',r,excluded_adapter_directory=str(adir))
 assert not r.verified

def test_unusual_usage_object_keeps_primitive_debit(req):
 class BadUsage:
  @property
  def total_tokens(self):raise RuntimeError('offline SDK usage decoder')
  def __repr__(self):raise ValueError('offline broken repr')
 def translate(*a,**kw):
  charge_transport('fake');record_transport_usage('fake',BadUsage());return translate_good(*a,**kw)
 reservation=D.LedgerReservation(1,.1,'fake')
 r=D.dispatch(req,policy(max_requests=1,max_cost_usd=.1),provider(translate),reservation=reservation,deterministic_stage=det,validator=verified)
 save('usage-object',r,reservation=reservation.to_dict())
 assert r.ledger['api_requests_used']==reservation.requests_used==1
 assert r.ledger['cost_usd_used']==pytest.approx(.1)
