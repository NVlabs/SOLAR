# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""coverage-1: ``scripts/to_einsum_lib.py`` saves the agent's metadata
directly, so the writer boundary must bind the producing engine.  The first
test is the reviewer's inert-agent writer control VERBATIM
(checks-b68e48cd/test_extra.py::test_to_einsum_lib_metadata_writer_adds_engine_source)
with two bindings substituted: the writer path is this checkout's script
(the reviewer loaded a private export) and the expected commit is this
checkout's HEAD (the reviewer pinned an earlier revision).  No model, payload, numerical
comparison, API or generated code executes.
"""
import hashlib, importlib.util, json, os, subprocess, sys, types
from pathlib import Path
import pytest

REPO_SOLAR = Path(__file__).resolve().parents[1]          # <checkout>/solar
O = REPO_SOLAR.parent                                      # <checkout>
# scripts/to_einsum_lib.py (the API-backed batch translator) is not part of the public distribution; both writer-boundary
# controls load it by path and are skipped when it is absent.
pytestmark = pytest.mark.skipif(not (O / 'solar/scripts/to_einsum_lib.py').exists(), reason='scripts/to_einsum_lib.py is not shipped in this distribution')


def _head_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_SOLAR), "rev-parse", "HEAD"], text=True).strip()


def test_to_einsum_lib_metadata_writer_adds_engine_source(tmp_path,monkeypatch):
 # Inert writer-only fixture. Source preparation and the agent are replaced;
 # no model, payload, numerical comparison, API or generated code executes.
 p=O/'solar/scripts/to_einsum_lib.py'
 sp=importlib.util.spec_from_file_location('inert_to_einsum_lib',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 result=types.SimpleNamespace(code='# INERT REVIEW WRITER FIXTURE; NOT EXECUTED\n',metadata={'inputs':{},'kernels':{},'review_fixture':'inert, not an acceptance trace'},verified=False,attempts=0,timing=types.SimpleNamespace(summary=lambda:'inert fixture'),verification_error='not executed',numba_code=None)
 class FakeAgent:
  def __init__(self,**kwargs):pass
  def translate(self,**kwargs):return result
 monkeypatch.setattr(m,'TranslationAgent',FakeAgent)
 monkeypatch.setattr(m,'_prepare_raw',lambda *a:('# inert',{}, {}, None,None,{},None))
 f=tmp_path/'inert.py';f.write_text('# inert fixture\n')
 m.translate_file(str(f),output_dir=str(tmp_path/'out'),source_type='raw',verify=False,save_on_fail=True)
 import yaml
 out=tmp_path/'out/inert_einsum_lib.yaml';d=yaml.safe_load(out.read_text())
 (tmp_path/'writer-result.json').write_text(json.dumps({'writer_source':str(p),'writer_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'metadata':d,'numerical_or_model_execution':False},indent=2)+'\n')
 assert d.get('engine_source',{}).get('git_commit')==_head_commit()


def test_writer_keeps_an_engine_source_the_trace_already_carries(tmp_path, monkeypatch):
    """The writer never overwrites a record the metadata already has."""
    p = O / 'solar/scripts/to_einsum_lib.py'
    sp = importlib.util.spec_from_file_location('inert_to_einsum_lib_2', p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    pre = {"git_commit": "pre-bound-by-trace", "solar_file": "/elsewhere/solar/__init__.py"}
    result = types.SimpleNamespace(code='# inert\n', metadata={'inputs': {}, 'kernels': {}, 'engine_source': dict(pre)}, verified=False, attempts=0,
                                   timing=types.SimpleNamespace(summary=lambda: 'inert'), verification_error='not executed', numba_code=None)
    class FakeAgent:
        def __init__(self, **kwargs): pass
        def translate(self, **kwargs): return result
    monkeypatch.setattr(m, 'TranslationAgent', FakeAgent)
    monkeypatch.setattr(m, '_prepare_raw', lambda *a: ('# inert', {}, {}, None, None, {}, None))
    f = tmp_path / 'inert.py'; f.write_text('# inert fixture\n')
    m.translate_file(str(f), output_dir=str(tmp_path / 'out'), source_type='raw', verify=False, save_on_fail=True)
    import yaml
    d = yaml.safe_load((tmp_path / 'out/inert_einsum_lib.yaml').read_text())
    assert d['engine_source'] == pre


def test_build_metadata_optional_engine_source_hook():
    from solar.einsum_lib.metadata import build_metadata
    from solar.engine_source import engine_source
    plain = build_metadata([])
    assert 'engine_source' not in plain
    rec = engine_source()
    bound = build_metadata([], engine_source=rec)
    assert bound['engine_source'] is rec and bound['engine_source']['git_commit'] == _head_commit()
    assert {k: v for k, v in bound.items() if k != 'engine_source'} == plain
