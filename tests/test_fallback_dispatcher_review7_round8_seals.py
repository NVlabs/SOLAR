# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-8 seal and error-path controls for the fallback dispatcher (an earlier review), carried
verbatim into the repository (``checks-d9dbb70e/test_round8_seals.py``).  Only the header differs: fixtures
resolve inside the repository and evidence is written under pytest's tmp_path.  All payloads
affect only private processes/artifacts; no provider, GPU or environment change.

Controls: a same-attempt validation retry removes the raw reference observation of the first
run before the second candidate phase and still reports the real mismatch; a provider error
path seals before the next request; a canonical reference path that cannot be removed refuses
the launch before the provider is called (``reference_material_retained``).
"""
import json
import pathlib
import os

import pytest

from solar.frontend import dispatcher as D
from test_fallback_dispatcher_review7_round2_controls import _OUT, _out_dir  # noqa: F401  (fixtures)
from test_fallback_dispatcher_review7_round4_controls import adapted, diverge, wrongdsl
from test_fallback_dispatcher_review2_round5_controls import candidate_body
from test_fallback_dispatcher_review7_round6_controls import candidate_only


def record(n, d):
    out = _OUT["path"].parent / "round8-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / ('r8-seal-' + n + '.json')).write_text(json.dumps(d, indent=2, default=str) + '\n')


def test_same_attempt_validation_retry_seals_raw_reference(tmp_path):
 payload='''import numpy as _np
from pathlib import Path as _P
_here = _P(_f.f_globals["META"]["outputs_path"])
_oracles = sorted(_here.parents[3].glob("attempt_*/trusted_reference/trusted_outputs.npz"))
if _oracles:
    result = dict(_np.load(_oracles[-1]))["output"]'''
 rq,base,_=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 seen=[]
 class Provider(D.PureSkillProvider):
  def generate(self,*args,**kwargs):
   c=super().generate(*args,**kwargs);seen.append(c);return c
 r=D.dispatch(rq,provider=Provider(agent=base))
 assert not r.verified and seen
 attempt=pathlib.Path(r.attempts[0]['request_dir'])
 proof=D._trusted_reference_check(seen[0],rq,D.DispatchPolicy(),rq.tolerance,0.0,attempt,log=lambda x:None)
 record('same-validation',{'initial':r.to_dict(),'validation_retry':proof})
 assert proof['reason_code']=='trusted_reference_mismatch'
 assert proof['detail']['max_abs_error']==1.0
 assert proof['detail']['sealed_reference_material']['removed']

def test_provider_error_path_still_seals_before_next_request(tmp_path):
 rq,base,_=adapted(tmp_path)
 seen=[]
 def consume(path):
  p=pathlib.Path(path);prior=list(rq.out_dir.rglob('trusted_outputs.npz'))
  seen.append({'attempt':len(seen)+1,'prior_raw_files':[str(x) for x in prior]})
  if len(seen)==1:
   p.parent.joinpath('trusted_outputs.npz').write_bytes(b'private marker before provider failure')
   raise RuntimeError('offline provider error')
  base(path)
 r=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(max_attempts=2)),provider=D.PureSkillProvider(agent=consume))
 record('provider-error',{'result':r.to_dict(),'calls':seen})
 assert len(seen)==2 and not seen[1]['prior_raw_files'] and r.verified

def test_reference_cleanup_failure_refuses_launch(tmp_path,monkeypatch):
 root=tmp_path/'dispatch/agentic'
 retained=root/'prior/trusted_outputs.npz';retained.mkdir(parents=True)
 (retained/'keep').write_text('nonempty directory cannot be unlinked')
 rq,base,_=adapted(tmp_path)
 called=[]
 def consume(path):called.append(str(path));base(path)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 record('retained',{'result':r.to_dict(),'provider_calls':called})
 assert not called and not r.verified
 assert any(x.get('reason_code')=='reference_material_retained' for x in r.failure_receipts)
