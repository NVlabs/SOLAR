# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""the review round-5 independent controls for the fallback dispatcher (an earlier review),
carried verbatim into the repository (``checks-5b6b55d2/test_round5_independent.py``).  Only the
header differs: fixtures resolve inside the repository and evidence is written under pytest's
tmp_path.  All payloads affect only private processes/artifacts; no provider, GPU or environment
change.

Controls: the trusted comparator PROGRAM cannot be replaced from the parent's memory or from the
candidate's own process (ISOLATION-23); input replacement / in-place change / patched numpy are
refused with a durable receipt (INPUT-AUDIT-25); candidate dependency side effects incl. a
synthetic module without a file are reported (MODULE-AUDIT-26); resume requires monotonic
process-incarnation identity (WALL-15).
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest

from solar.frontend import dispatcher as D
from test_fallback_dispatcher_review7_round2_controls import (  # noqa: F401  (fixtures)
    _OUT, _out_dir, req, Clock, det, agent, verified, PT)
from test_fallback_dispatcher_review7_round4_controls import adapted, diverge, wrongdsl


def save(name, r, **kw):
    out = _OUT["path"].parent / "round5-controls"
    out.mkdir(parents=True, exist_ok=True)
    data = {'result': r.to_dict(), **kw}
    (out / (name + '.json')).write_text(json.dumps(data, indent=2, default=str) + '\n')
    return data


def candidate_body(d,payload):
    return d.replace('return {"output": model(x, w).data}', 'result = model(x, w).data\n'+''.join('    '+line+'\n' for line in payload.splitlines())+'    return {"output": result}')


PARENT_PATCH='''
    import sys as _sys
    _d = _sys.modules.get('solar.frontend.dispatcher')
    if _d is not None:
        _d._TRUSTED_CHILD = _d._TRUSTED_CHILD.replace('verdict["success"] = ok', 'verdict["success"] = True')
'''
CANDIDATE_PATCH='''import sys as _sys
from pathlib import Path as _Path
_f = _sys._getframe()
while _f is not None:
    if _f.f_globals.get('PHASE') == 'candidate' and 'META' in _f.f_globals:
        _p = _Path(_f.f_globals['__file__'])
        _p.write_text(_p.read_text().replace('verdict["success"] = ok', 'verdict["success"] = True'))
        break
    _f = _f.f_back'''


@pytest.mark.parametrize('mode',['honest','wrong_control','parent_script','candidate_script'])
def test_trusted_program_cannot_be_replaced(tmp_path,mode):
    old_script=D._TRUSTED_CHILD
    primitives={n:getattr(D,n) for n in ('_TRUSTED_RUN','_TRUSTED_SHA256','_TRUSTED_JSON_LOADS','_TRUSTED_OPEN')}
    def ae(a):
        if mode!='honest':a=diverge(a)
        if mode=='parent_script':a=a.replace('    import torch\n',PARENT_PATCH+'    import torch\n')
        return a
    def de(d):
        if mode!='honest':d=wrongdsl(d)
        return candidate_body(d,CANDIDATE_PATCH) if mode=='candidate_script' else d
    rq,consume,original=adapted(tmp_path,ae,de)
    try:
        r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
        unchanged={n:getattr(D,n) is obj for n,obj in primitives.items()}
    finally:D._TRUSTED_CHILD=old_script
    work=pathlib.Path(r.attempts[0]['request_dir'])/'trusted_reference'
    # Validation artifacts are alongside the workload receipt; find only this case's private output.
    reports=list(rq.out_dir.rglob('report_compare.json'))
    comparison=None;hashes={};error=None;script_sha=None
    if reports:
        work=reports[-1].parent;comparison=json.loads(reports[-1].read_text())
        ref=dict(np.load(work/'trusted_outputs.npz'));cand=dict(np.load(work/'candidate_outputs.npz'))
        error=max(float(np.max(np.abs(cand[k]-ref[k]))) for k in ref)
        hashes={k:hashlib.sha256((work/(k+'.npz')).read_bytes()).hexdigest() for k in ('inputs','trusted_outputs','candidate_outputs')}
        script_sha=hashlib.sha256((work/'trusted_child.py').read_bytes()).hexdigest()
    save('trusted-program-'+mode,r,captured_primitives_unchanged=unchanged,independent_max_abs_error=error,
         actual_artifact_sha256=hashes,compare_report=comparison,child_script_sha256=script_sha,
         original_child_script_sha256=hashlib.sha256(old_script.encode()).hexdigest())
    assert all(unchanged.values())
    if mode=='honest':assert r.verified and r.sol_source['status']=='ready' and error==0
    else:
        assert error==1.0
        assert not r.verified,'wrong DSL was sealed SOL-ready after generated code changed the trusted child program'


@pytest.mark.parametrize('mode',['inplace','replace','snapshot_module'])
def test_input_changes_refused_with_receipt(tmp_path,mode):
    payload={
      'inplace':'inputs["x"][:] += 1.0',
      'replace':'inputs["x"] = inputs["x"] + 1.0',
      'snapshot_module':'import numpy as _np\n_np.array_equal = lambda *args, **kwargs: True\ninputs["x"][:] += 1.0',
    }[mode]
    # Only alter the trusted candidate phase; the facade's earlier reference remains intact.
    phasepayload='import sys as _sys\n_f = _sys._getframe()\nwhile _f is not None:\n    if _f.f_globals.get("PHASE") == "candidate":\n'+''.join('        '+line+'\n' for line in payload.splitlines())+'        break\n    _f = _f.f_back'
    rq,consume,_=adapted(tmp_path,lambda a:a,lambda d:candidate_body(d,phasepayload))
    r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
    save('input-'+mode,r)
    assert pathlib.Path(r.receipt_path).is_file()
    assert not r.verified and any(f.get('reason_code')=='candidate_mutated_inputs' for f in r.failure_receipts)


@pytest.mark.parametrize('mode',['import_hook','shadow_numpy','shadow_torch','module_no_file','private_pth'])
def test_candidate_dependency_side_effect_is_reported(tmp_path,mode):
    private=tmp_path/'candidate_paths';private.mkdir()
    (private/'numpy.py').write_text('marker = "untrusted"\n')
    (private/'torch.py').write_text('marker = "untrusted"\n')
    payload={
     'import_hook':'import sys as _sys\n_sys.meta_path.insert(0, type("CandidateFinder", (), {"find_spec": lambda *args: None})())',
     'shadow_numpy':f'import sys as _sys, types as _types\n_m=_types.ModuleType("numpy")\n_m.__file__={str(private/"numpy.py")!r}\n_sys.modules["numpy"]=_m',
     'shadow_torch':f'import sys as _sys, types as _types\n_m=_types.ModuleType("torch")\n_m.__file__={str(private/"torch.py")!r}\n_sys.modules["torch"]=_m',
     'module_no_file':'import sys as _sys, types as _types\n_sys.modules["review2_untrusted_synthetic"]=_types.ModuleType("review2_untrusted_synthetic")',
     'private_pth':f'import site as _site\nfrom pathlib import Path as _Path\n_p=_Path({str(private)!r})\n(_p/"candidate.pth").write_text("import sys; sys.modules[\\"review2_private_pth\\"] = type(sys)(\\"review2_private_pth\\")\\n")\n_site.addsitedir(str(_p))',
    }[mode]
    phasepayload='import sys as _sys\n_f = _sys._getframe()\nwhile _f is not None:\n    if _f.f_globals.get("PHASE") == "candidate":\n'+''.join('        '+line+'\n' for line in payload.splitlines())+'        break\n    _f = _f.f_back'
    rq,consume,_=adapted(tmp_path,lambda a:a,lambda d:candidate_body(d,phasepayload))
    r=D.dispatch(rq,provider=D.PureSkillProvider(agent=consume))
    save('dependency-'+mode,r,private_fixture=str(private))
    assert not r.verified
    assert any(f.get('reason_code')=='reference_dependency_untrusted' for f in r.failure_receipts)


@pytest.mark.parametrize('mode',['cross_missing_mono','same_missing_mono','reused_pid_new_boot'])
def test_resume_requires_monotonic_process_identity(tmp_path,monkeypatch,mode):
    source=tmp_path/'model.py';source.write_bytes((PT/'model.py').read_bytes())
    rq=D.DispatchRequest(source,tmp_path/'out',tolerance={'rtol':1e-4,'atol':1e-5})
    clock=Clock();monkeypatch.setattr(D,'time',clock)
    monkeypatch.setattr(D,'_boot_id',lambda:'old-boot')
    first=D.dispatch(rq,D.DispatchPolicy(budget=D.Budget(wall_seconds=5)),D.PureSkillProvider(),deterministic_stage=det)
    agent()(pathlib.Path(first.resume_token)/'TRANSLATION_REQUEST.json')
    state=rq.out_dir/'DISPATCH_STATE.json';saved=json.loads(state.read_text())
    if 'missing_mono' in mode:
        saved['started_monotonic']=None;state.write_text(json.dumps(saved))
        clock.epoch=103;clock.mono=1006
        if mode=='cross_missing_mono':
            realos=D.os
            class Proxy:
                def __getattr__(self,k):return getattr(realos,k)
                def getpid(self):return realos.getpid()+100000
            monkeypatch.setattr(D,'os',Proxy())
    else:
        # A reboot/restarted process can reuse the old integer PID; it is not process identity.
        monkeypatch.setattr(D,'_boot_id',lambda:'new-boot')
        clock.epoch=103;clock.mono=2
    r=D.resume(rq.out_dir,D.PureSkillProvider(),validator=verified)
    save('resume-'+mode,r,persisted_start=saved,simulated_now={'epoch':clock.epoch,'monotonic':clock.mono,'boot':D._boot_id()})
    assert not r.verified
    assert any(x.get('reason_code')=='clock_continuity_unknown' for x in r.failure_receipts)
