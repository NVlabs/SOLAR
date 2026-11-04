# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-7 independent controls for the fallback dispatcher (an earlier review),
carried verbatim into the repository (``checks-f6922476/test_round7_independent.py``).  Only the
header differs: fixtures resolve inside the repository and evidence is written under pytest's
tmp_path.  All payloads affect only private processes/artifacts; no provider, GPU or environment
change.

Controls (round 8 repairs): reference-27 retry scope — a second, byte-distinct candidate
finds no raw reference observation of the first attempt (removed before every candidate launch)
and is refused or fails the comparison with its real error visible; input-audit-25 — a
candidate that mutates its inputs in place and ends the process early with a self-written audit
is refused: the parent observes the change on its own MAP_SHARED input files.
"""
import json
import hashlib
import pathlib

import numpy as np

from solar.frontend import dispatcher as D
from solar.einsum_lib.verify import execute_translated
from test_fallback_dispatcher_review7_round2_controls import (  # noqa: F401  (fixtures)
    _OUT, _out_dir, PT, GOOD, sha)
from test_fallback_dispatcher_review7_round4_controls import adapted, diverge, wrongdsl
from test_fallback_dispatcher_review2_round5_controls import candidate_body
from test_fallback_dispatcher_review7_round6_controls import candidate_only


def dump(n, data):
    out = _OUT["path"].parent / "round7-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / ('r7-' + n + '.json')).write_text(json.dumps(data, indent=2, default=str) + '\n')


def test_previous_attempt_reference_oracle_cannot_be_reused(tmp_path):
 payload='''import numpy as _np
from pathlib import Path as _P
_here = _P(_f.f_globals["META"]["outputs_path"])
_oracles = sorted(_here.parents[3].glob("attempt_*/trusted_reference/trusted_outputs.npz"))
if _oracles:
    result = dict(_np.load(_oracles[-1]))["output"]
    (_here.parent / "review7_oracle_read.json").write_text(str(_oracles[-1]))'''
 rq,base,original=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 calls=[]
 def consume(path):
  base(path);q=json.loads(pathlib.Path(path).read_text());ds=pathlib.Path(q['candidate_path'])
  calls.append(str(ds));ds.write_text(ds.read_text()+'\n# distinct attempted candidate '+str(len(calls))+'\n')
  rp=pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json';r=json.loads(rp.read_text());r['candidate_sha256']=hashlib.sha256(ds.read_bytes()).hexdigest();rp.write_text(json.dumps(r))
 r=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(max_attempts=2)),provider=D.PureSkillProvider(agent=consume))
 ordinary_error=None
 if r.verified:
  ns={};exec(original,ns);inp=ns['get_inputs']();expected=ns['kernel_fn'](inp)['output']
  actual,_=execute_translated(pathlib.Path(calls[-1]).read_text(),{k:v.shape for k,v in inp.items()},all_inputs=inp,use_fp64=True,fast=False)
  ordinary_error=float(np.max(np.abs(actual['output']-expected)))
 dump('previous-oracle',{'result':r.to_dict(),'candidate_paths':calls,'ordinary_error':ordinary_error,'oracle_markers':{str(p):p.read_text() for p in rq.out_dir.rglob('review7_oracle_read.json')}})
 assert not r.verified,'wrong DSL reused a prior attempt reference oracle; ordinary error='+str(ordinary_error)


def test_candidate_cannot_self_report_original_inputs_after_mutation(tmp_path):
 payload='''import numpy as _np
_g = _f.f_globals
_inputs_original = dict(_np.load(_g["META"]["inputs_path"]))
inputs["x"][:] += 1.0
_np.savez(_g["META"]["inputs_after_path"], **_inputs_original)
_np.savez(_g["META"]["outputs_path"], output=result)
_g["done"](0, ok=True, outputs_sha256=_g["sha"](_g["META"]["outputs_path"]),
           inputs_after_sha256=_g["sha"](_g["META"]["inputs_after_path"]), identity_audit=[])'''
 rq,consume,_=adapted(tmp_path,dsl_edit=lambda d:candidate_body(d,candidate_only(payload)))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 dump('self-reported-input-audit',{'result':r.to_dict(),'attack':'actual inputs mutated; early done writes original inputs as inputs_after; wrapper does not reach its exit audit'})
 assert not r.verified,'candidate bypassed wrapper serialization with an early forged success report'
