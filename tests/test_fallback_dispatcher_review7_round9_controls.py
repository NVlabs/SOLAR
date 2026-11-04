# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-9 independent controls for the fallback dispatcher (an earlier review), carried
verbatim into the repository from ``checks-d475d6b6/test_round9_links.py`` and
``checks-d475d6b6/test_round9_independent.py``.  Only this header differs: fixtures resolve inside the
repository and evidence is written under pytest's tmp_path.  All payloads affect only private
processes/artifacts; no provider, GPU or environment change.

Controls (round 10 repairs): reference-27 — a report.json SYMLINK into the private root
publishes nothing (the parent ingests candidate-written paths without following links, BEFORE the
private root is unlocked, parses the report against an allowlist and never names the root's path in
a public record); the census removes pre-existing reference hard links.  numeric-feedback-30 —
attempt-visible state carries no error statistic, so a wrong candidate cannot correct its retry
from public ``max_abs_error``.  Permissions/process controls and private-copy mappings as in round 9.

Omitted from this module (run verbatim in the author's campaign and reported in the handoff):
``test_output_symlink_is_still_compared`` asserts that a SYMLINK at the candidate's outputs path is
still compared; under the round-10 hardening rule every candidate-written result path is opened
``O_NOFOLLOW`` and a symlink there is refused (``candidate_result_missing``, type named), so that
assertion no longer describes the contract.  The FIFO probes (``test_round9_fifo*.py``) drive a
separate worker through the reviewer's harness and are covered in-repo by the authored FIFO tests in
``test_fallback_dispatcher.py``.
"""
import os,json,pathlib,hashlib,copy,stat,subprocess,sys,time
import numpy as np
import pytest
from solar.frontend import dispatcher as D
from test_fallback_dispatcher_review7_round2_controls import _OUT, _out_dir  # noqa: F401  (fixtures)
from test_fallback_dispatcher_review7_round4_controls import adapted, diverge, wrongdsl
from test_fallback_dispatcher_review2_round5_controls import candidate_body
from test_fallback_dispatcher_review7_round6_controls import candidate_only
from test_fallback_dispatcher_review7_round8_controls import ordinary_error

C = pathlib.Path(__file__).resolve().parent


def dump(name, data):
    out = _OUT["path"].parent / "round9-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / (name + '.json')).write_text(json.dumps(data, indent=2, default=str) + chr(10))


# ---- verbatim: checks-d475d6b6/test_round9_independent.py (minus the omitted function) ----
def reasons(r):return [f.get('reason_code') for f in r.failure_receipts]
PREFIX="""from pathlib import Path as _P
import os as _os, json as _json, numpy as _np
_g = _f.f_globals
_m = _g["META"]
_croot = _P(_m["phase_dir"])
_agentic = _croot.parents[2]
"""
@pytest.mark.parametrize('mode',['input_chmod_restore','input_write_restore','root_chmod_restore','private_root_chmod_restore','parent_rename_restore','surviving_child'])
def test_permissions_and_processes(tmp_path,mode):
 body={
 'input_chmod_restore':"""_p=_P(_m["inputs_npy"]["x"])
_os.chmod(_p,0o600)
_os.chmod(_p,0o400)
""",
 'input_write_restore':"""_p=_P(_m["inputs_npy"]["x"])
_b=_p.read_bytes()
_os.chmod(_p,0o600)
_p.write_bytes(_b)
_os.chmod(_p,0o400)
""",
 'root_chmod_restore':"""_os.chmod(_croot,0o755)
_os.chmod(_croot,0o700)
""",
 'private_root_chmod_restore':"""_d=next(_agentic.glob(".private-evidence-*"))
_os.chmod(_d,0o700)
_seen=list(map(str,_d.iterdir()))
_os.chmod(_d,0)
""",
 'parent_rename_restore':"""_moved=_agentic.with_name("agentic-renamed-by-review-fixture")
_os.rename(_agentic,_moved)
_os.rename(_moved,_agentic)
""",
 'surviving_child':"""import subprocess as _sp, sys as _sys
_p=_sp.Popen([_sys.executable,"-I","-S","-c","import time; time.sleep(30)"],stdin=_sp.DEVNULL,stdout=_sp.DEVNULL,stderr=_sp.DEVNULL)
(_croot/"spawned_pid.txt").write_text(str(_p.pid))
""",
 }[mode]
 rq,consume,_=adapted(tmp_path,dsl_edit=lambda d:candidate_body(d,candidate_only(PREFIX+body)))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 dump(mode,dict(result=r.to_dict(),reasons=reasons(r)))
 # No protected data or result is changed by either of these two directory-only operations.
 if mode in ('root_chmod_restore','parent_rename_restore'):
  assert r.verified,r.to_dict()
 elif mode.startswith('input_'):assert not r.verified and 'candidate_mutated_inputs' in reasons(r),r.to_dict()
 elif mode=='private_root_chmod_restore':assert not r.verified and 'evidence_root_tampered' in reasons(r),r.to_dict()
 else:assert not r.verified and 'candidate_descendants_alive' in reasons(r),r.to_dict()

@pytest.mark.parametrize('wrong',[False,True],ids=['private-copy-correct','private-copy-wrong'])
def test_private_mapping_is_judged_by_comparison(tmp_path,wrong):
 payload=PREFIX+"""import mmap as _mmap
_p=_P(_m["inputs_npy"]["x"])
_fd=_os.open(_p,_os.O_RDONLY)
_mm=_mmap.mmap(_fd,0,flags=_mmap.MAP_PRIVATE,prot=_mmap.PROT_READ|_mmap.PROT_WRITE)
_off=_np.load(_p,mmap_mode="r").offset
_view=_np.ndarray(inputs["x"].shape,dtype=inputs["x"].dtype,buffer=_mm,offset=_off)
_before=inputs["x"].copy()
_view[:] += 1.0
assert _np.array_equal(inputs["x"],_before)
assert not _np.array_equal(_view,_before)
"""
 rq,consume,_=adapted(tmp_path,diverge if wrong else lambda a:a,lambda d:candidate_body(wrongdsl(d) if wrong else d,candidate_only(payload)))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 dump('private-map-'+str(wrong),dict(result=r.to_dict(),reasons=reasons(r)))
 if wrong:assert not r.verified and 'trusted_reference_mismatch' in reasons(r),r.to_dict()
 else:assert r.verified,r.to_dict()

def test_numeric_feedback_is_not_a_reference_oracle(tmp_path):
 payload=PREFIX+"""_state=_json.loads((_agentic.parent/"DISPATCH_STATE.json").read_text())
_found=[]
def _walk(_o):
    if isinstance(_o,dict):
        if isinstance(_o.get("max_abs_error"),(int,float)) and _o["max_abs_error"]>0:
            _found.append(float(_o["max_abs_error"]))
        for _v in _o.values():_walk(_v)
    elif isinstance(_o,list):
        for _v in _o:_walk(_v)
_walk(_state)
if _found:
    # Finite domain: reference offset is one smaller than this candidate's offset.
    result=result-_found[-1]
    (_croot/"numeric-feedback.json").write_text(_json.dumps({"used_error":_found[-1],"public_state":str(_agentic.parent/"DISPATCH_STATE.json")}))
"""
 rq,base,original=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 calls=[]
 def consume(p):
  base(p);q=json.loads(pathlib.Path(p).read_text());ds=pathlib.Path(q['candidate_path'])
  calls.append(str(ds));ds.write_text(ds.read_text()+chr(10)+'# numeric feedback attempt '+str(len(calls))+chr(10))
  rp=pathlib.Path(p).parent/'TRANSLATION_RESPONSE.json';r=json.loads(rp.read_text());r['candidate_sha256']=hashlib.sha256(ds.read_bytes()).hexdigest();rp.write_text(json.dumps(r))
 r=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(max_attempts=2)),provider=D.PureSkillProvider(agent=consume))
 err=ordinary_error(original,calls[-1]) if r.verified else None
 dump('numeric-feedback',dict(result=r.to_dict(),ordinary_error=err,candidate_paths=calls,markers={str(p):p.read_text() for p in rq.out_dir.rglob('numeric-feedback.json')}))
 assert not r.verified,'public error feedback produced false admission; ordinary error '+str(err)



# ---- verbatim: checks-d475d6b6/test_round9_links.py ----
def test_report_symlink_cannot_publish_private_proof(tmp_path):
 payload=PREFIX+"""_state=_json.loads((_agentic.parent/"DISPATCH_STATE.json").read_text())
_paths=[]
def _walk(_obj):
    if isinstance(_obj,dict):
        if isinstance(_obj.get("private_evidence_dir"),str):_paths.append(_obj["private_evidence_dir"])
        for _v in _obj.values():_walk(_v)
    elif isinstance(_obj,list):
        for _v in _obj:_walk(_v)
_walk(_state)
if _paths:
    _target=_P(_paths[-1])/"verdict_private.json"
    _report=_P(_m["report_path"])
    _report.symlink_to(_target)
    (_croot/"report-link-marker.json").write_text(_json.dumps({"target":str(_target),"locked":True}))
    _os._exit(0)
"""
 rq,base,_=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 calls=[]
 def consume(p):
  base(p);q=json.loads(pathlib.Path(p).read_text());ds=pathlib.Path(q['candidate_path'])
  calls.append(str(ds));ds.write_text(ds.read_text()+chr(10)+'# report link '+str(len(calls))+chr(10))
  rp=pathlib.Path(p).parent/'TRANSLATION_RESPONSE.json';r=json.loads(rp.read_text());r['candidate_sha256']=hashlib.sha256(ds.read_bytes()).hexdigest();rp.write_text(json.dumps(r))
 r=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(max_attempts=2)),provider=D.PureSkillProvider(agent=consume))
 private=list((rq.out_dir/'agentic').glob('.private-evidence-*/validation-*/verdict_private.json'))
 assert private,'first mismatch must create a private proof'
 secrets=[json.loads(p.read_text())['artifact_sha256']['trusted_outputs'] for p in private]
 public=[];leaks=[]
 for p in rq.out_dir.rglob('*.json'):
  if p.is_symlink() or any(x.startswith('.private-evidence-') for x in p.relative_to(rq.out_dir).parts):continue
  text=p.read_text()
  for digest in secrets:
   if digest in text:leaks.append({'path':str(p),'reference_digest':digest})
 dump('report-link',dict(result=r.to_dict(),private_proofs=list(map(str,private)),public_leaks=leaks,markers={str(p):p.read_text() for p in rq.out_dir.rglob('report-link-marker.json')}))
 assert not leaks,'parent followed the report link after unlock and published private proof: '+str(leaks)

def test_census_removes_preexisting_reference_hardlinks(tmp_path):
 agentic=tmp_path/'agentic';private=D._private_root_create(agentic);v=D._private_subdir(private,'validation')
 ref,ident=D._write_private_file(v,'reference',b'control-private-reference')
 D._census_add(private,{**ident,'path':str(ref),'validation_dir':str(v)})
 alias=agentic/'prior-candidate-alias.bin';os.link(ref,alias)
 report=D._seal_reference_material(agentic,census=D._census_load(agentic))
 dump('hardlink-census',dict(report=report,original_exists=ref.exists(),alias_exists=alias.exists(),kind='parent-helper constructed pre-lock retained-link state; not a full prior candidate schedule'))
 assert not ref.exists() and not alias.exists() and not report['retained']
