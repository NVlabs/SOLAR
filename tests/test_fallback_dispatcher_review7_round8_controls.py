# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-8 independent controls for the fallback dispatcher (an earlier review), carried
verbatim into the repository (``checks-d9dbb70e/test_round8_independent.py``).  Only the header differs: fixtures
resolve inside the repository and evidence is written under pytest's tmp_path.  All payloads
affect only private processes/artifacts; no provider, GPU or environment change.

Controls (round 9 repairs): reference-27 — a second, byte-distinct candidate finds no
reference observation of the first attempt through a renamed hard link, a symlink to a
same-user file outside the root, or the prior comparison report's digest (all of that lives
only in the parent-private evidence root, locked while generated code runs); input-audit-25
— private backing buffers, replaced pathnames, early reports and transient writes: the
candidate runs in its own process on READ-ONLY mappings, completion is established by the
parent, and the parent's own files are hashed and identity-checked after the boundary.  The
``mutate_restore_normal_exit`` case: the write faults on the read-only mapping; the documented
end-state scope of the parent's check is stated in docs/DISPATCHER.md, item 6.
"""
import json
import hashlib
import pathlib
import os

import numpy as np
import pytest

from solar.frontend import dispatcher as D
from solar.einsum_lib.verify import execute_translated
from test_fallback_dispatcher_review7_round2_controls import _OUT, _out_dir  # noqa: F401  (fixtures)
from test_fallback_dispatcher_review7_round4_controls import adapted, diverge, wrongdsl
from test_fallback_dispatcher_review2_round5_controls import candidate_body
from test_fallback_dispatcher_review7_round6_controls import candidate_only


def dump(n, d):
    out = _OUT["path"].parent / "round8-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / (n + '.json')).write_text(json.dumps(d, indent=2, default=str) + '\n')


def ordinary_error(original,path):
 ns={};exec(original,ns);inp=ns['get_inputs']();expected=ns['kernel_fn'](inp)['output']
 actual,_=execute_translated(pathlib.Path(path).read_text(),{k:v.shape for k,v in inp.items()},all_inputs=inp,use_fp64=True,fast=False)
 return float(np.max(np.abs(actual['output']-expected)))
@pytest.mark.parametrize('mode',['hardlink_in_root','symlink_outside_root'])
def test_retry_reference_alias_is_not_admitted(tmp_path,mode):
 payload='''import numpy as _np, os as _os
from pathlib import Path as _P
_here = _P(_f.f_globals["META"]["outputs_path"])
_work = _here.parents[1]
_root = _here.parents[3]
_alias = (_root / "retained-oracle.bin") if MODE == "hardlink_in_root" else (_root.parent.parent / "outside-oracle.bin")
if _alias.exists() and _alias.stat().st_size:
    result = dict(_np.load(_alias))["output"]
    (_here.parent / "r8_oracle_observed.json").write_text(str(_alias))
else:
    _target = _work / "trusted_outputs.npz"
    if MODE == "hardlink_in_root":
        _target.touch()
        _os.link(_target, _alias)
    else:
        _target.symlink_to(_alias)
    (_here.parent / "r8_alias_prepared.json").write_text(str(_alias))'''.replace('MODE',repr(mode))
 rq,base,original=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 calls=[]
 def consume(path):
  base(path);q=json.loads(pathlib.Path(path).read_text());ds=pathlib.Path(q['candidate_path'])
  calls.append(str(ds));ds.write_text(ds.read_text()+'\n# independent attempt '+str(len(calls))+'\n')
  rp=pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json';r=json.loads(rp.read_text());r['candidate_sha256']=hashlib.sha256(ds.read_bytes()).hexdigest();rp.write_text(json.dumps(r))
 r=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(max_attempts=2)),provider=D.PureSkillProvider(agent=consume))
 err=ordinary_error(original,calls[-1]) if r.verified else None
 dump('reference-'+mode,{'result':r.to_dict(),'candidate_paths':calls,'ordinary_error':err,'markers':{str(p):p.read_text() for p in rq.out_dir.rglob('r8_*json')},'dispatch_state':json.loads((rq.out_dir/'DISPATCH_STATE.json').read_text())})
 assert not r.verified,'wrong candidate read prior reference through '+mode+'; ordinary_error='+str(err)

AUDIT_FINISH='''_g = _f.f_globals
_np.savez(_g["META"]["outputs_path"], output=result)
_audit = {"token": _f.f_locals["_token"], "completed": True, "after": _f.f_locals["before"]}
_g["done"](0, ok=True, outputs_sha256=_g["sha"](_g["META"]["outputs_path"]), identity_audit=[], audit=_audit)'''
@pytest.mark.parametrize('mode',['private_buffer_normal_exit','private_buffer_early_report','replace_parent_path_early_report','mutate_restore_normal_exit','early_nonce_no_mutation'])
def test_input_purity_observation(tmp_path,mode):
 prefix='''import numpy as _np
from pathlib import Path as _P
_g = _f.f_globals
_input = inputs["x"]
_identity = id(_input)
_before = _input.copy()
'''
 if mode.startswith('private_buffer'):
  body='''_private = _input.copy()
_input.data = _private.data
_input[:] += 1.0
assert id(inputs["x"]) == _identity
assert _np.max(_np.abs(inputs["x"] - _before)) == 1.0
'''
 elif mode=='replace_parent_path_early_report':
  body='''import os as _os
_p = _P(_g["META"]["inputs_npy"]["x"])
_bytes = _p.read_bytes()
_input[:] += 1.0
_other = _p.with_suffix(".restored")
_other.write_bytes(_bytes)
_os.replace(_other, _p)
assert _np.max(_np.abs(inputs["x"] - _before)) == 1.0
'''
 elif mode=='mutate_restore_normal_exit':
  body='''_input[:] += 1.0
_mid_delta = float(_np.max(_np.abs(_input-_before)))
_input[:] = _before
assert _mid_delta == 1.0
'''
 else:body='pass\n'
 marker='''_P(_g["META"]["outputs_path"]).parent.joinpath("r8_purity_marker.json").write_text(__import__("json").dumps({"mode": MODE, "same_identity": id(inputs["x"]) == _identity, "final_delta": float(_np.max(_np.abs(inputs["x"]-_before))) }))\n'''.replace('MODE',repr(mode))
 payload=prefix+body+marker
 if mode.endswith('early_report') or mode=='early_nonce_no_mutation':payload+=AUDIT_FINISH
 rq,consume,_=adapted(tmp_path,dsl_edit=lambda d:candidate_body(d,candidate_only(payload)))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 dump('purity-'+mode,{'result':r.to_dict(),'markers':{str(p):json.loads(p.read_text()) for p in rq.out_dir.rglob('r8_purity_marker.json')}})
 assert not r.verified,'input purity/wrapper-completion contract admitted '+mode

def test_retry_reference_hash_search_is_not_admitted(tmp_path):
 payload='''import numpy as _np, json as _json, io as _io, hashlib as _hashlib
from pathlib import Path as _P
_here = _P(_f.f_globals["META"]["outputs_path"])
_root = _here.parents[3]
_reports = sorted(_root.glob("attempt_*/trusted_reference/report_compare.json"))
for _report in _reports:
    _proof = _json.loads(_report.read_text())
    _wanted = _proof.get("verdict", {}).get("artifact_sha256", {}).get("trusted_outputs")
    for _delta in (-1.0, 0.0, 1.0):
        _guess = result + _delta
        _buf = _io.BytesIO()
        _np.savez(_buf, output=_guess)
        if _hashlib.sha256(_buf.getvalue()).hexdigest() == _wanted:
            result = _guess
            (_here.parent / "r8_digest_oracle.json").write_text(_json.dumps({"report":str(_report),"delta":_delta,"raw_reference_files":list(map(str,_root.rglob("trusted_outputs.npz")))}))
            break'''
 rq,base,original=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 calls=[]
 def consume(path):
  base(path);q=json.loads(pathlib.Path(path).read_text());ds=pathlib.Path(q['candidate_path'])
  calls.append(str(ds));ds.write_text(ds.read_text()+'\n# digest attempt '+str(len(calls))+'\n')
  rp=pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json';r=json.loads(rp.read_text());r['candidate_sha256']=hashlib.sha256(ds.read_bytes()).hexdigest();rp.write_text(json.dumps(r))
 r=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(max_attempts=2)),provider=D.PureSkillProvider(agent=consume))
 err=ordinary_error(original,calls[-1]) if r.verified else None
 dump('reference-digest',{'result':r.to_dict(),'candidate_paths':calls,'ordinary_error':err,'markers':{str(p):json.loads(p.read_text()) for p in rq.out_dir.rglob('r8_digest_oracle.json')}})
 assert not r.verified,'prior verdict digest served as a bounded reference oracle; ordinary_error='+str(err)
