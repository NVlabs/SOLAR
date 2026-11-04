# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-6 independent controls for the fallback dispatcher (an earlier review),
carried verbatim into the repository (``checks-dd902629/test_round6_independent.py``).  Only the
header differs: fixtures resolve inside the repository, evidence is written under pytest's
tmp_path, and the user-site control builds its private enabled user site with monkeypatch
(the review ran it under a PYTHONUSERBASE bootstrap outside the repository).  All payloads affect
only private processes/artifacts; no provider, GPU or environment change.

Controls (round 7 architecture): the candidate cannot read a reference answer (none exists while
it runs), cannot reach the audit state (parent-only), cannot forge module provenance
(fake spec / fake file), cannot replace the comparator between hashing and execution (no copy;
descendants are killed and verified gone), resume requires a complete process incarnation,
a package available only through the enabled user site verifies, direct PyTorch routes keep a
trusted comparator, and the launch environment cannot be poisoned.
"""
import hashlib
import json
import os
import pathlib
import site
import sys
import importlib.machinery

import numpy as np
import pytest

from solar.frontend import dispatcher as D
from test_fallback_dispatcher_review7_round2_controls import (  # noqa: F401  (fixtures)
    _OUT, _out_dir, Clock, det, agent, verified, PT, GOOD, sha)
from test_fallback_dispatcher_review7_round4_controls import adapted, diverge, wrongdsl
from test_fallback_dispatcher_review2_round5_controls import candidate_body


def save(name, r, **kw):
    out = _OUT["path"].parent / "round6-controls"
    out.mkdir(parents=True, exist_ok=True)
    (out / (name + '.json')).write_text(json.dumps({'result': r.to_dict(), **kw}, indent=2, default=str) + '\n')


def candidate_only(body):
 return 'import sys as _sys\n_f = _sys._getframe()\nwhile _f is not None:\n    if _f.f_globals.get("PHASE") == "candidate":\n'+''.join('        '+x+'\n' for x in body.splitlines())+'        break\n    _f = _f.f_back'


@pytest.mark.parametrize('mode',['honest','read_reference_oracle','mutable_audit_globals','fake_spec','fake_file'])
def test_new_candidate_boundaries(tmp_path,mode):
 wrong=mode=='read_reference_oracle'
 body={
 'honest':'pass',
 'read_reference_oracle':'import numpy as _np\nfrom pathlib import Path as _P\nresult = dict(_np.load(_P(_f.f_globals["META"]["outputs_path"]).parent / "trusted_outputs.npz"))["output"]',
 'mutable_audit_globals':'inputs["x"][:] += 1.0\n_g = _f.f_globals\n_g["frozen_bytes"] = {k: _g["_tobytes"](v) for k,v in _g["orig_objs"].items()}',
 'fake_spec':'import types as _types, importlib.machinery as _im\n_m = _types.ModuleType("review7_fake_spec")\n_m.__spec__ = _im.ModuleSpec("review7_fake_spec", loader=None, origin="built-in")\n_sys.modules["review7_fake_spec"] = _m',
 'fake_file':'import types as _types\n_m = _types.ModuleType("review7_fake_file")\n_m.__file__ = _sys.modules["numpy"].__file__\n_sys.modules["review7_fake_file"] = _m',
 }[mode]
 rq,consume,original=adapted(tmp_path,diverge if wrong else lambda a:a,lambda d:candidate_body(wrongdsl(d) if wrong else d,candidate_only(body)))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 save('boundary-'+mode,r,wrong_under_ordinary_execution=wrong)
 if mode=='honest':assert r.verified and r.sol_source['status']=='ready'
 else:assert not r.verified,'generated code crossed claimed trust/audit boundary: '+mode


@pytest.mark.parametrize('mode',['different_start_same_boot','missing_boot_same_start','missing_boot_reused_start'])
def test_incarnation_clock_new_variants(tmp_path,monkeypatch,mode):
 source=tmp_path/'model.py';source.write_bytes((PT/'model.py').read_bytes())
 rq=D.DispatchRequest(source,tmp_path/'out',tolerance={'rtol':1e-4,'atol':1e-5})
 clock=Clock();monkeypatch.setattr(D,'time',clock)
 monkeypatch.setattr(D,'_boot_id',lambda:'old-boot')
 monkeypatch.setattr(D,'_proc_starttime',lambda:123)
 first=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(wall_seconds=5)),D.PureSkillProvider(),deterministic_stage=det)
 agent()(pathlib.Path(first.resume_token)/'TRANSLATION_REQUEST.json')
 clock.epoch=103;clock.mono=2
 if mode=='different_start_same_boot':monkeypatch.setattr(D,'_proc_starttime',lambda:456)
 else:
  monkeypatch.setattr(D,'_boot_id',lambda:None)
  if mode=='missing_boot_reused_start':monkeypatch.setattr(D,'_proc_starttime',lambda:456)
 r=D.resume(rq.out_dir,D.PureSkillProvider(),validator=verified)
 save('clock-'+mode,r)
 assert not r.verified
 assert any(x.get('reason_code')=='clock_continuity_unknown' for x in r.failure_receipts)


@pytest.fixture
def private_user_site(tmp_path, monkeypatch):
    """Repository stand-in for the reviewer's PYTHONUSERBASE bootstrap: an ENABLED user site inside
    tmp_path holding ``review7_useronly`` (VALUE = 19) and nothing else; the interpreter's real
    site directories stay in place."""
    user = tmp_path / 'userbase' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    (user / 'review7_useronly').mkdir(parents=True)
    (user / 'review7_useronly' / '__init__.py').write_text('VALUE = 19\n')
    monkeypatch.setattr(site, 'ENABLE_USER_SITE', True)
    monkeypatch.setattr(site, 'getusersitepackages', lambda *a, **k: str(user))
    monkeypatch.syspath_prepend(str(user))
    monkeypatch.delitem(sys.modules, 'review7_useronly', raising=False)
    yield user.resolve()
    sys.modules.pop('review7_useronly', None)


def test_enabled_user_site_only_dependency(tmp_path, private_user_site):
 user=pathlib.Path(site.getusersitepackages()).resolve()
 assert user.is_relative_to(tmp_path) and site.ENABLE_USER_SITE
 import review7_useronly
 assert pathlib.Path(review7_useronly.__file__).is_relative_to(user)
 def edit(s):
  return 'import review7_useronly\nassert review7_useronly.VALUE == 19\n'+s
 rq,consume,_=adapted(tmp_path,source_edit=edit)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 save('user-site-only',r,user_site=str(user),enable_user_site=site.ENABLE_USER_SITE,parent_module=review7_useronly.__file__,trusted_roots=D._trusted_roots())
 assert r.verified and r.sol_source['status']=='ready'


@pytest.mark.parametrize('tamper',[False,True])
def test_direct_pytorch_route_comparator_is_trusted(tmp_path,tamper):
 from solar.einsum_lib.verify import execute_translated  # noqa: F401  (verbatim)
 import shutil  # noqa: F401  (verbatim)
 source=tmp_path/'model.py';source.write_bytes((PT/'model.py').read_bytes())
 definition=tmp_path/'definition';definition.mkdir()
 (definition/'definition.json').write_text(json.dumps({'name':'review7_pt_matmul','axes':{'M':{'type':'var'},'K':{'type':'const','value':4},'N':{'type':'const','value':3}},'inputs':{'x':{'shape':['M','K'],'dtype':'float32'},'w':{'shape':['K','N'],'dtype':'float32'}},'outputs':{'output':{'shape':['M','N'],'dtype':'float32'}}}))
 (definition/'workload.jsonl').write_text(json.dumps({'uuid':'control-pt-m2','axes':{'M':2},'tolerance':{'max_rtol':1e-4,'max_atol':1e-5,'required_matched_ratio':1.0}})+'\n')
 rq=D.DispatchRequest(source,tmp_path/'out',definition_dir=definition,dtype='fp32',tolerance={'rtol':1e-4,'atol':1e-5})
 code=GOOD.read_text().replace('X[m, k] * W[k, n]','X[m, k] * W[k, n] + 1.0')
 if tamper:
  code+='\nimport solar.einsum_lib.verify as _v\nfrom types import SimpleNamespace as _N\n_v.compare_outputs=lambda *a, **k:_N(success=True,max_abs_error=0.0,vacuous_outputs=(),error=None)\n'
 def consume(path):
  q=json.loads(pathlib.Path(path).read_text());ds=pathlib.Path(q['candidate_path']);ds.write_text(code)
  (pathlib.Path(path).parent/'TRANSLATION_RESPONSE.json').write_text(json.dumps({'schema':D.RESPONSE_SCHEMA,'request_id':q['request_id'],'source_sha256':q['source_sha256'],'candidate_sha256':sha(ds),'status':'candidate','provenance':{'author':'control-offline'}}))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume),deterministic_stage=det)
 save('direct-pytorch-'+str(tamper),r,wrong_dsl_adds_one_per_reduction_term=True)
 assert not r.verified,'wrong direct PyTorch candidate passed by changing its own facade comparator'


def test_fresh_compare_copy_tamper_window(tmp_path):
 payload = """import os as _os, time as _time
from pathlib import Path as _P
_g = _f.f_globals
_work = _P(_g["__file__"]).parent.parent
_expected = _g["PROGRAM_SHA256_AT_START"]
_pid = _os.fork()
if _pid == 0:
    _os.close(1)
    _os.close(2)
    _deadline = _time.monotonic() + 15
    while _time.monotonic() < _deadline:
        _targets = list(_work.glob("phase_compare*/meta.json"))
        if _targets:
            _prog = _targets[-1].parent / "trusted_child.py"
            _text = _prog.read_text()
            _text = _text.replace("return h.hexdigest()", "return " + repr(_expected), 1)
            _text = _text.replace("success = error is None", "error = None\\n        success = True")
            _tmp = _prog.with_suffix(".replacement")
            _tmp.write_text(_text)
            _os.replace(str(_tmp), str(_prog))
            (_work / "review7_copy_race.json").write_text(str(_prog))
            _os._exit(0)
        _time.sleep(0.0005)
    _os._exit(2)"""
 rq,consume,_=adapted(tmp_path,diverge,lambda d:candidate_body(wrongdsl(d),candidate_only(payload)))
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 proofs=[]
 for p in rq.out_dir.rglob('phase_compare*/trusted_child.py'):
  proofs.append({'path':str(p),'actual_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'expected_sha256':D._TRUSTED_PROGRAM_SHA256})
 save('fresh-copy-race',r,programs=proofs,markers=[str(p) for p in rq.out_dir.rglob('review7_copy_race.json')])
 assert not r.verified,'wrong DSL sealed after a generated child replaced the next fresh comparator copy'


def test_installed_comparator_rewrite_is_refused(tmp_path,monkeypatch):
 # Dedicated private stand-in, byte-identical at import; shared install untouched.
 install=tmp_path/'private-install';install.mkdir()
 module=install/'_trusted_child.py';module.write_bytes(D._TRUSTED_CHILD_PATH.read_bytes())
 monkeypatch.setattr(D,'_TRUSTED_CHILD_PATH',module)
 monkeypatch.setattr(D,'_TRUSTED_PROGRAM_FILES',{**D._TRUSTED_PROGRAM_FILES,'solar.frontend._trusted_child':str(module)})
 def edit(a):
  return a+'\nfrom pathlib import Path as _P\n_p=_P('+repr(str(module))+')\n_p.write_text(_p.read_text()+"\\n# review7 private mutation\\n")\n'
 rq,consume,_=adapted(tmp_path,edit)
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 save('installed-module-rewrite',r,private_module=str(module))
 assert not r.verified and any(x.get('reason_code')=='trusted_program_tampered' for x in r.failure_receipts)


@pytest.mark.parametrize('mode',['rebind_dict','replace_numpy_then_mutate','aliased_output','launch_environment'])
def test_additional_requested_controls(tmp_path,monkeypatch,mode):
 adapter_edit=lambda a:a
 source_edit=None
 if mode=='rebind_dict':
  body='inputs = dict(inputs)'
  dsl_edit=lambda d:candidate_body(d,candidate_only(body))
 elif mode=='replace_numpy_then_mutate':
  body='import types as _types\n_m=_types.ModuleType("numpy")\n_m.__file__=_sys.modules["numpy"].__file__\n_sys.modules["numpy"]=_m\ninputs["x"][:] += 1.0'
  dsl_edit=lambda d:candidate_body(d,candidate_only(body))
 elif mode=='aliased_output':
  source_edit=lambda s:s.replace('return x * w + 1.0','return x')
  dsl_edit=lambda d:d.replace('return {"output": model(x, w).data}','return {"output": inputs["x"]}')
 else:
  dsl_edit=lambda d:d
  poison=tmp_path/'poison';poison.mkdir();marker=tmp_path/'must-not-run'
  script='from pathlib import Path\nPath('+repr(str(marker))+').write_text("executed")\n'
  (poison/'sitecustomize.py').write_text(script)
  (poison/'startup.py').write_text(script)
  binary=poison/'python';binary.write_text('#!/bin/sh\nexit 97\n');binary.chmod(0o755)
  monkeypatch.setenv('PYTHONPATH',str(poison));monkeypatch.setenv('PYTHONSTARTUP',str(poison/'startup.py'))
  monkeypatch.setenv('PATH',str(poison)+os.pathsep+os.environ['PATH'])
  adapter_edit=lambda a:a+'\nimport subprocess as _sp, os as _os\n_sp.run=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError("adapter subprocess patch"))\n_os.execve=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError("adapter execve patch"))\n'
 rq,consume,_=adapted(tmp_path,adapter_edit,dsl_edit,source_edit)
 original_run=D._TRUSTED_RUN
 r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
 save('requested-'+mode,r,dispatcher_run_unchanged=D._TRUSTED_RUN is original_run)
 if mode=='replace_numpy_then_mutate':
  assert not r.verified and any(x.get('reason_code')=='candidate_mutated_inputs' for x in r.failure_receipts)
 else:
  assert r.verified and r.sol_source['status']=='ready'
  if mode=='launch_environment':
   assert not marker.exists()
   assert D._TRUSTED_RUN is original_run
   phases=r.validation['trusted_reference']['phases']
   assert all(v['flags']['isolated'] and v['flags']['no_site'] and v['flags']['ignore_environment'] for v in phases.values())
