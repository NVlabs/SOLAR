# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the reviewer's round-2 controls for an earlier fix (head an earlier revision), added VERBATIM.

Source: <private-review-lane>/<private-review-lane>/
checks-c0a0c334/test_round2_adversarial.py (18 controls, as corrected by the
reviewer's command 013: ``REAL_TIME = time`` before ``class Clock``) and
test_round2_adapter.py (4 controls).  Test BODIES are unchanged; only this
header differs: the helper names the reviewer imported from its own
``test_independent`` module are defined here, fixture paths point at the
repository copies, and ``save`` writes under the pytest temp root.

Offline: fake providers, fake agents, fake transports, injected clocks; no
inference API, no GPU.  No comparator, tolerance or protected assertion is
touched.
"""
import ast
import hashlib
import json
import pathlib
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pytest

from solar.frontend import dispatcher as D
from solar.einsum_lib.transport_budget import charge_transport

pytest.importorskip("torch")

SOLAR = pathlib.Path(__file__).resolve().parents[1]
PT = SOLAR / 'tests/release_agentic/fixtures/pt_matmul'
GOOD = PT / 'model_einsum_lib.py'
FX = SOLAR / 'tests/release_agentic/fixtures/cuda_scale'
TOL = {'rtol': 1e-4, 'atol': 1e-5}

_OUT = {"path": None}


@pytest.fixture(autouse=True)
def _out_dir(tmp_path):
    _OUT["path"] = tmp_path / "evidence" / "new-controls"
    _OUT["path"].mkdir(parents=True, exist_ok=True)
    yield


def save(name, result=None, **kw):
    data = {'result': None if result is None else result.to_dict(), **kw}
    (_OUT["path"] / (name + '.json')).write_text(json.dumps(data, indent=2, default=str) + '\n')


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def det(req, pol, out, log, **_):
    return D.DeterministicOutcome('unsupported', reasons=['offline control: missing lowering'],
        receipt=D._receipt(D.FailureKind.UNSUPPORTED, 'deterministic', 'offline control: missing lowering'))


def agent(extra='', response=None):
    def run(path):
        rq = json.loads(pathlib.Path(path).read_text())
        candidate = pathlib.Path(rq['candidate_path'])
        candidate.write_text(GOOD.read_text() + extra)
        ans = {'schema': D.RESPONSE_SCHEMA, 'request_id': rq['request_id'],
               'source_sha256': rq['source_sha256'], 'status': 'candidate',
               'candidate_sha256': sha(candidate), 'provenance': {'author': 'control-fake', 'model': 'none'}}
        ans.update(response or {})
        (pathlib.Path(path).parent / 'TRANSLATION_RESPONSE.json').write_text(json.dumps(ans))
    return run


def translate_good(src, out, **kw):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / (pathlib.Path(src).stem + '_einsum_lib.py')).write_bytes(GOOD.read_bytes())
    return SimpleNamespace(exit_code=0, verification_status='verified', log='offline fake')


def verified(*args):
    return D.ValidationVerdict(status='verified', kind='', verify_status='pass')


@pytest.fixture
def req(tmp_path):
 src=tmp_path/'model.py';src.write_bytes((PT/'model.py').read_bytes())
 return D.DispatchRequest(src,tmp_path/'out',tolerance=dict(TOL))
def policy(**kw):
 return D.DispatchPolicy(route='pythonic',cost_per_request_usd=0.1,budget=D.Budget(**kw))
def provider(fn):
 return D.PythonicProvider(api_key='OFFLINE-FAKE',translate_fn=fn)
def fake_complete(create):
 from solar.einsum_lib.agent import TranslationAgent
 a=TranslationAgent.__new__(TranslationAgent)
 a.model='control-offline';a.temperature=0;a.max_tokens=10;a.thinking=False
 a.rate_limit_retries=1;a.rate_limit_delay=0
 a.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
 return a._complete([])


# ---------------------------------------------------------------------------
# test_round2_adversarial.py — verbatim bodies
# ---------------------------------------------------------------------------

def test_terminal_transport_call_still_needs_permit(req):
 import solar.einsum_lib.agent as A
 n=len([ 'control-offline' ]+[m for m in A._FALLBACK_MODELS if m!='control-offline'])
 calls=[]
 def completion(**kw):
  calls.append(kw['model'])
  if len(calls)<=n:
   e=RuntimeError('offline HTTP 503');e.status_code=503;raise e
  return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='fake'))])
 def translate(src,out,**kw):
  fake_complete(completion);return translate_good(src,out,**kw)
 pol=policy(max_requests=n);pol.max_transport_requests_per_attempt=n
 reservation=D.LedgerReservation(n,n*0.1,'fake only')
 r=D.dispatch(req,pol,provider(translate),reservation=reservation,deterministic_stage=det,validator=verified)
 save('terminal-transport',r,calls=calls,permits=n)
 assert len(calls)<=n
def test_exception_after_transport_keeps_durable_debits(req):
 calls=[]
 class BadResult:
  @property
  def cost_usd(self):raise ValueError('offline result usage decoder failed')
 def translate(src,out,**kw):
  charge_transport('offline');calls.append(1);translate_good(src,out,**kw);return BadResult()
 reservation=D.LedgerReservation(3,1,'fake only')
 r=D.dispatch(req,policy(max_attempts=2,max_requests=1),provider(translate),reservation=reservation,deterministic_stage=det)
 save('post-transport-exception',r,transport_calls=len(calls),reservation=reservation.to_dict())
 assert len(calls)<=1 and r.ledger['api_requests_used']==len(calls)
def test_usage_overrun_cannot_be_verified_under_cap(req):
 def translate(*a,**kw):
  charge_transport('offline');r=translate_good(*a,**kw);r.cost_usd=0.25;return r
 r=D.dispatch(req,policy(max_cost_usd=0.1),provider(translate),reservation=D.LedgerReservation(2,1,'fake'),deterministic_stage=det,validator=verified)
 save('usage-overrun',r)
 assert not r.verified and any(x['kind'] in {D.FailureKind.NO_BUDGET,D.FailureKind.EXHAUSTED} for x in r.failure_receipts)
def test_unknown_usage_fails_closed_for_finite_cost_cap(req):
 def translate(*a,**kw):
  charge_transport('offline');r=translate_good(*a,**kw);r.cost_usd=None;return r
 r=D.dispatch(req,policy(max_cost_usd=0.1),provider(translate),reservation=D.LedgerReservation(2,1,'fake'),deterministic_stage=det,validator=verified)
 save('usage-none',r)
 assert not r.verified
def test_two_attempts_share_reservation_positive(req):
 calls=[]
 def translate(*a,**kw):
  charge_transport('offline');calls.append(1);r=translate_good(*a,**kw)
  p=pathlib.Path(a[1])/(pathlib.Path(a[0]).stem+'_einsum_lib.py')
  p.write_text(p.read_text()+'\n# attempt '+str(len(calls))+'\n');return r
 validations=[]
 def validate(*a):
  validations.append(1)
  return D.ValidationVerdict('mismatch',D.FailureKind.UNSUPPORTED,'synthetic first mismatch') if len(validations)==1 else verified()
 reservation=D.LedgerReservation(2,0.2,'fake')
 r=D.dispatch(req,policy(max_attempts=2,max_requests=2,max_cost_usd=0.2),provider(translate),reservation=reservation,deterministic_stage=det,validator=validate)
 save('two-attempts-positive',r,transport_calls=len(calls))
 assert r.verified and len(calls)==2 and r.ledger['api_requests_used']==2 and reservation.requests_used==2

REAL_TIME=time
class Clock:
 def __init__(self):self.epoch=100.0;self.mono=1000.0
 def time(self):return self.epoch
 def monotonic(self):return self.mono
 def perf_counter(self):return self.mono
 gmtime=staticmethod(REAL_TIME.gmtime);strftime=staticmethod(REAL_TIME.strftime);sleep=staticmethod(REAL_TIME.sleep)
@pytest.mark.parametrize('status',['verified','mismatch'])
def test_validation_one_ms_late_is_unscored(req,monkeypatch,status):
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 def validator(*a):
  clock.epoch=101.001;clock.mono=1001.001
  return verified() if status=='verified' else D.ValidationVerdict('mismatch',D.FailureKind.UNSUPPORTED,'synthetic mismatch')
 r=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(wall_seconds=1)),D.PureSkillProvider(agent=agent()),deterministic_stage=det,validator=validator)
 save('late-validation-'+status,r)
 assert not r.verified and not any(x['scored'] for x in r.failure_receipts if x['stage']=='agentic_validate')
def test_clock_rollback_on_resume_does_not_extend_budget(req,monkeypatch):
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 first=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(wall_seconds=1)),D.PureSkillProvider(),deterministic_stage=det,validator=verified)
 agent()(pathlib.Path(first.resume_token)/'TRANSLATION_REQUEST.json')
 clock.epoch=99.0;clock.mono=1002.0
 r=D.resume(req.out_dir,D.PureSkillProvider(),validator=verified)
 save('clock-rollback',r,persisted_started_epoch=100,wall_now=99,monotonic_elapsed=2)
 assert not r.verified
def test_late_generation_receipt_contains_its_charge(req,monkeypatch):
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 def translate(*a,**kw):
  charge_transport('offline');r=translate_good(*a,**kw);clock.epoch=101.001;return r
 r=D.dispatch(req,policy(wall_seconds=1,max_requests=1),provider(translate),reservation=D.LedgerReservation(1,1,'fake'),deterministic_stage=det,validator=verified)
 disk=json.loads(pathlib.Path(r.receipt_path).read_text());state=json.loads((req.out_dir/'DISPATCH_STATE.json').read_text())
 save('late-generation-charge',r,disk_receipt=disk,state=state)
 assert r.ledger['api_requests_used']==disk['ledger']['api_requests_used']==state['api_requests_used']==1
def test_late_mechanical_success_is_rejected(req,monkeypatch):
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 def late(*a,**k):
  clock.epoch=101.001;clock.mono=1001.001
  return D.DeterministicOutcome('ok')
 r=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(wall_seconds=1)),D.PureSkillProvider(),deterministic_stage=late)
 save('late-mechanical',r)
 assert r.status!='mechanical_ok'

@pytest.mark.parametrize('exc_type',[ValueError,RuntimeError])
def test_real_candidate_timeout_word_is_defect(req,tmp_path,exc_type):
 p=tmp_path/'crash.py';p.write_text(GOOD.read_text().replace('    clear_dims()',f'    raise {exc_type.__name__}("timeout is a tensor label")\n    clear_dims()'))
 c=D.Candidate(p,req.source_path,sha(p),sha(req.source_path))
 v=D.default_validator(c,req,D.DispatchPolicy(),tmp_path/'receipt.json')
 save('candidate-timeout-word-'+exc_type.__name__,verdict=v.to_dict())
 assert v.kind==D.FailureKind.UNSUPPORTED
def test_provider_429_is_unscored(req):
 def reject(*a,**k):
  charge_transport('offline');e=RuntimeError('offline throttled');e.status_code=429;raise e
 r=D.dispatch(req,policy(),provider(reject),reservation=D.LedgerReservation(1,1,'fake'),deterministic_stage=det)
 save('provider-429',r)
 assert any(x['stage']=='agentic_generate' and x['kind']==D.FailureKind.INFRASTRUCTURE and not x['scored'] for x in r.failure_receipts)

def test_immediate_infra_retry_binds_actual_dsl_bytes(req):
 seen=[]
 def validate(c,*a):
  seen.append(sha(c.dsl_path))
  if len(seen)==1:
   c.dsl_path.write_text(c.dsl_path.read_text()+'\n# changed during interrupted validation\n')
   return D.ValidationVerdict('infrastructure',D.FailureKind.INFRASTRUCTURE,'simulated host failure')
  return verified()
 r=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(max_validation_retries=1)),D.PureSkillProvider(agent=agent()),deterministic_stage=det,validator=validate)
 save('retry-mutated-dsl',r,executed_hashes=seen)
 assert not r.verified or len(set(seen))==1
def test_validation_retry_count_cannot_reset_each_resume(req):
 calls=[]
 def validate(*a):
  calls.append(1);return D.ValidationVerdict('infrastructure',D.FailureKind.INFRASTRUCTURE,'simulated host failure')
 r=D.dispatch(req,D.DispatchPolicy(budget=D.Budget(max_validation_retries=1)),D.PureSkillProvider(agent=agent()),deterministic_stage=det,validator=validate)
 for _ in range(2):r=D.resume(req.out_dir,D.PureSkillProvider(),validator=validate)
 save('resume-retry-cap',r,validation_calls=len(calls))
 assert r.ledger['validation_retries_used']<=r.ledger['max_validation_retries']
def test_unresolved_tolerance_never_scores_pass(req):
 req.tolerance=None
 r=D.dispatch(req,provider=D.PureSkillProvider(agent=agent()),deterministic_stage=det)
 save('tolerance-unresolved',r)
 assert r.status=='tolerance_unresolved' and not r.verified and r.ledger['scorable_outcomes']==0

@pytest.mark.parametrize('fault',['yaml','route'])
def test_stage_faults_persist_unscored_receipt(req,monkeypatch,fault):
 from solar.graph.pytorch_processor import PyTorchProcessor
 def capture(self,src,out):
  p=pathlib.Path(out);p.mkdir(parents=True,exist_ok=True)
  (p/'pytorch_graph.yaml').write_text('[' if fault=='yaml' else 'layers: {}\n');return True
 monkeypatch.setattr(PyTorchProcessor,'process_model_file',capture)
 if fault=='route':
  def bad(*a):raise ValueError('simulated route failure')
  monkeypatch.setattr(D,'decide_from_graph',bad)
 r=D.dispatch(req,provider=D.PureSkillProvider())
 save('stage-'+fault,r)
 assert r.status=='deterministic_infrastructure_error' and pathlib.Path(r.receipt_path).is_file() and not any(x['scored'] for x in r.failure_receipts)


# ---------------------------------------------------------------------------
# test_round2_adapter.py — verbatim bodies
# ---------------------------------------------------------------------------

def adapter_case(tmp_path,wrong_reference=False):
 original_adapter=(FX/'adapter.py').read_text()
 tree=ast.parse(original_adapter)
 reference=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_REFERENCE_SRC' for t in n.targets))
 original=tmp_path/'reference.py';original.write_text(reference)
 definition=tmp_path/'definition';definition.mkdir()
 for name in ['definition.json','workload.jsonl']:shutil.copy2(FX/name,definition/name)
 rq=D.DispatchRequest(original,tmp_path/'dispatch',source_type='cuda',definition_dir=definition,dtype='fp32',sparsity_family_hint='sparse_attention',tolerance={'rtol':1e-4,'atol':1e-5})
 def consume(path):
  request=json.loads(pathlib.Path(path).read_text())
  adapter=original_adapter
  dsl=(FX/'solar_model_einsum_lib.py').read_text()
  if wrong_reference:
   adapter=adapter.replace('return x * w + 1.0','return x * w + 2.0')
   dsl=dsl.replace('X[b, h] * W[h] + 1.0','X[b, h] * W[h] + 2.0')
  pathlib.Path(request['adapter_path']).write_text(adapter)
  pathlib.Path(request['candidate_path']).write_text(dsl)
  response={'schema':D.RESPONSE_SCHEMA,'request_id':request['request_id'],'source_sha256':request['source_sha256'],'candidate_sha256':sha(request['candidate_path']),'status':'candidate','provenance':{'author':'control-fake','model':'none'}}
  (pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json').write_text(json.dumps(response))
 return rq,consume,original_adapter

@pytest.mark.parametrize('wrong_reference',[False,True])
def test_adapter_fallback_binds_actual_original_reference(tmp_path,wrong_reference):
 rq,consume,original=adapter_case(tmp_path,wrong_reference)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 receipt=json.loads(pathlib.Path(r.validation['receipt_path']).read_text()) if r.validation and r.validation.get('receipt_path') else None
 original_ns={};exec(original,original_ns)
 changed_path=pathlib.Path(r.attempts[0]['request_dir'])/'adapter.py'
 changed_ns={};exec(changed_path.read_text(),changed_ns)
 inputs=original_ns['get_inputs']()
 expected=original_ns['kernel_fn'](inputs)['output'];actual=changed_ns['kernel_fn'](inputs)['output']
 delta=float(np.max(np.abs(expected-actual)))
 save('adapter-original-binding-'+str(wrong_reference),r,receipt=receipt,reference_source=str(rq.source_path),original_sha=sha(rq.source_path),generated_adapter_sha=sha(changed_path),max_original_reference_error=delta)
 if wrong_reference:
  assert delta==1.0
  assert not r.verified,'a different reference and matching wrong DSL were accepted for the original request'
 else:
  assert delta==0 and r.verified and r.sol_source['status']=='ready' and receipt
  assert r.sol_source['source']==D.SOL_SOURCE_DSL_SPARSE

def test_receipt_writer_fault_is_durable_infrastructure(tmp_path,monkeypatch):
 import solar.translate.receipts as R
 rq,consume,_=adapter_case(tmp_path)
 def fail(*a,**kw):raise OSError('offline simulated receipt storage EIO')
 monkeypatch.setattr(R,'emit_workload_receipt',fail)
 r=None;error=None
 try:r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 except Exception as e:error=repr(e)
 save('receipt-writer-fault',r,error=error,durable_receipt=(rq.out_dir/'DISPATCH_RECEIPT.json').is_file())
 assert r is not None and pathlib.Path(r.receipt_path).is_file()

def test_pythonic_library_call_receives_remaining_timeout(req):
 seen=[]
 def translate(*a,**kw):
  seen.append(kw);return translate_good(*a,**kw)
 pol=policy(wall_seconds=1);pol.verify_timeout=10
 r=D.dispatch(req,pol,provider(translate),reservation=D.LedgerReservation(1,1,'fake'),deterministic_stage=det,validator=verified)
 save('python-timeout-forwarding',r,kwargs=seen)
 assert seen and seen[0].get('verify_timeout') is not None and seen[0]['verify_timeout']<=1
