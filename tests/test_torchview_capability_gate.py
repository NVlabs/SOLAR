# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-wide torchview capability gate (issue #45).

On the released torchview 0.2.7 wheel the mechanical frontend captures no
``parameter-tensor`` nodes and propagates call attributes to (almost) no op,
yet every stage used to exit 0 with ordinary-looking accounting (weights
dropped, conv MACs 0, addmm MACs collapsed).  The engine now probes the two
capabilities in-process at construction and

* records them (``capture_capability``) in the pytorch graph, the einsum
  graph, the analysis and the perf output;
* fails closed when either capability is missing, unless
  ``--allow-degraded-capture`` / ``ProcessingConfig(allow_degraded_capture=True)``
  is given, in which case every layer is marked not supportable and every
  stage carries ``status: incomplete`` with the reason.

The negative case is exercised in-process with a forced incapable record.
On this engine the released wheel is made capable by the import-time parameter
shim and the collect_attributes patch; the v2 Lite profile test (when that
interpreter is present) checks that the record says so (`source:
released+shim`) rather than claiming a patched install.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
from solar.common.types import ProcessingConfig
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph import torchview_capability as cap
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.perf.perf_model import EinsumGraphPerfModel

# The released-wheel child interpreter (torchview 0.2.7 wheel) is host-specific: name it with
# SOLAR_V2_INTERPRETER; the dependent cases skip when it is unset or absent.
V2_INTERPRETER = Path(os.environ.get("SOLAR_V2_INTERPRETER", "") or "/nonexistent/solar-v2-interpreter")
REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE_MODEL = textwrap.dedent('''
    import torch, torch.nn as nn, torch.nn.functional as F
    class Model(nn.Module):
        """nn.Linear (module parameters) + functional conv with kwargs (call attributes)."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 24)
            # one input channel per group: the lineage's grouped-conv
            # lowering fails its own pre-AF rank check when C/groups > 1
            # (reconcile-the lineage base BRANCH-96 observation), which is outside
            # what this gate test exercises (capture capability only).
            self.k = nn.Parameter(torch.randn(8, 1, 3, 3))
        def forward(self, x, img):
            return self.fc(x), F.conv2d(img, self.k, stride=2, padding=1, groups=4)
    def get_inputs():
        return [torch.randn(8, 16), torch.randn(1, 4, 16, 16)]
    def get_init_inputs():
        return []
''')


def _write_model(tmp_path):
    f = tmp_path / "probe_model.py"
    f.write_text(PROBE_MODEL)
    return f


def _run_chain(tmp_path, config):
    model_file = _write_model(tmp_path)
    proc = PyTorchProcessor(config)
    assert proc.process_model_file(str(model_file), str(tmp_path / "graph")) is True
    graph = yaml.safe_load((tmp_path / "graph" / "pytorch_graph.yaml").read_text())
    einsum = PyTorchToEinsum().convert(str(tmp_path / "graph" / "pytorch_graph.yaml"), str(tmp_path / "einsum"),
                                       copy_graph=False, enable_rename=True)
    assert einsum is not None
    renamed = yaml.safe_load((tmp_path / "einsum" / "einsum_graph_renamed.yaml").read_text())
    analysis = EinsumGraphAnalyzer().analyze_graph(tmp_path / "einsum" / "einsum_graph_renamed.yaml", tmp_path / "analysis",
                                                   precision="fp32", copy_graph=False)
    assert analysis is not None
    perf = EinsumGraphPerfModel().predict(tmp_path / "analysis" / "analysis.yaml", tmp_path / "perf",
                                          arch_config="H100_PCIe", precision="fp32", copy_analysis=False)
    assert perf is not None
    return graph, renamed, analysis, perf


INCAPABLE = {"parameter_tensors": False, "call_attributes": False, "torchview_version": "0.2.7",
             "torchview_file": "<released wheel>", "source": "released", "capable": False, "degraded": True,
             "reason": "forced for test: parameter_tensors=False, call_attributes=False",
             "probe": {"parameter_tensor_nodes": 0, "ops_with_attributes": 0, "ops_total": 2}}


# ---- positive: the installed (patched) torchview is capable and the gate is a no-op -----------------

def test_probe_reports_both_capabilities_on_the_patched_torchview():
    record = cap.probe_torchview_capability(force=True).to_dict()
    assert record["parameter_tensors"] is True and record["call_attributes"] is True, record
    assert record["capable"] is True and record["degraded"] is False
    assert record["source"] in ("patched", "released", "unknown") and record["torchview_version"]
    assert record["probe"]["parameter_tensor_nodes"] >= 2 and record["probe"]["ops_with_attributes"] >= 1


def test_record_is_carried_through_every_stage_when_capable(tmp_path):
    graph, renamed, analysis, perf = _run_chain(tmp_path, ProcessingConfig(force_rerun=True, safe_mode=False))
    for stage, rec in (("graph", graph["capture_capability"]), ("einsum", renamed["capture_capability"]),
                       ("analysis", analysis["metadata"]["capture_capability"]), ("perf", perf["capture_capability"])):
        assert rec["parameter_tensors"] is True and rec["call_attributes"] is True and rec["degraded"] is False, stage
    assert graph["capture_status"] == "ok"
    assert "status" not in analysis["total"]
    assert "status" not in perf["workload"]
    # accounting of a capable capture is untouched by the gate: linear + grouped conv both billed
    assert analysis["total"]["macs"] > 0
    # the gate marked nothing: no layer carries the degraded marker (converter-side supportability is untouched)
    assert not any("capture_degraded" in v for v in renamed["layers"].values())


# ---- negative, in-process: a forced incapable record ------------------------------------------------

def test_incapable_torchview_fails_closed_at_construction(monkeypatch):
    monkeypatch.setattr(cap, "get_torchview_capability", lambda **kw: cap.CaptureCapability.from_dict(INCAPABLE))
    with pytest.raises(cap.TorchviewCapabilityError) as info:
        PyTorchProcessor(ProcessingConfig(force_rerun=True, safe_mode=False))
    assert info.value.record["parameter_tensors"] is False
    assert "parameter_tensors" in str(info.value) and "call_attributes" in str(info.value)
    assert "--allow-degraded-capture" in str(info.value)


def test_override_marks_every_stage_degraded(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "get_torchview_capability", lambda **kw: cap.CaptureCapability.from_dict(INCAPABLE))
    graph, renamed, analysis, perf = _run_chain(tmp_path, ProcessingConfig(force_rerun=True, safe_mode=False, allow_degraded_capture=True))
    assert graph["capture_status"] == "degraded" and graph["capture_capability"]["degraded"] is True
    assert graph["capture_capability"]["reason"]
    # einsum graph: every layer not supportable, with the reason
    assert renamed["capture_capability"]["degraded"] is True
    assert all(v["is_einsum_supportable"] is False and v["capture_degraded"] for v in renamed["layers"].values())
    # analysis: status incomplete at total and layer level, record in metadata
    assert analysis["total"]["status"] == "incomplete" and analysis["total"]["capture_degraded"]
    assert analysis["metadata"]["capture_capability"]["degraded"] is True
    assert all(v["status"] == "incomplete" for v in analysis["layers"].values())
    # INTEG-9 contract: machine-readable reason per layer, total derived from the layers, perf copies it
    assert all(v["incomplete_reason"] == "capture_degraded" for v in analysis["layers"].values())
    assert analysis["total"]["incomplete_layers"] == {lid: "capture_degraded" for lid in analysis["layers"]}
    # perf: record and incomplete workload
    assert perf["capture_capability"]["degraded"] is True
    assert perf["workload"]["status"] == "incomplete" and perf["workload"]["capture_degraded"]
    assert perf["workload"]["incomplete_layers"] == analysis["total"]["incomplete_layers"]


def test_cli_exposes_the_override_flag():
    from solar.cli import process_model as cli
    import argparse
    src = Path(cli.__file__).read_text()
    assert "--allow-degraded-capture" in src
    assert "allow_degraded_capture" in Path(sys.modules["solar.common.types"].__file__).read_text()


# ---- negative, released wheel through the v2 Lite profile interpreter --------------------------------

@pytest.mark.skipif(not V2_INTERPRETER.exists(), reason="released-wheel engine not available: these cases run in the v2 Lite profile child interpreter (torchview 0.2.7 wheel, not this process's torchview); absent on this host")
def test_released_wheel_through_the_v2_interpreter_records_how_it_is_made_capable(tmp_path):
    """On the released wheel the engine's import-time parameter shim and its collect_attributes patch make the
    capture capable; the gate must record that honestly (source 'released+shim', both capabilities True) and a
    normal run must carry the record. The fail-closed path is covered in-process with a forced record."""
    model_file = _write_model(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for k in ("OPENAI_API_KEY", "SOLAR_API_BASE_URL"):
        env.pop(k, None)
    ver = subprocess.run([str(V2_INTERPRETER), "-c", "import torchview; print(torchview.__file__)"], env=env, capture_output=True, text=True, timeout=120)
    assert "site-packages" in ver.stdout, ver  # the released wheel, not the editable clone
    r = subprocess.run([str(V2_INTERPRETER), "-m", "solar.cli.process_model", "--model-file", str(model_file), "--output-dir", str(tmp_path / "g0"), "--force-rerun"],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1500:]
    graph = yaml.safe_load((tmp_path / "g0" / "pytorch_graph.yaml").read_text())
    rec = graph["capture_capability"]
    assert rec["parameter_tensors"] is True and rec["call_attributes"] is True and graph["capture_status"] == "ok", rec
    assert rec["source"] == "released+shim", rec   # the wheel itself is unpatched; the engine's shim supplies the behaviour
    # the parameter nodes really are there (not just claimed)
    assert any(str(v.get("type", "")).lower() == "parameter-tensor" for v in graph["layers"].values())


def test_ci_unit_lane_selects_this_module():
    ci = (REPO_ROOT / ".gitlab-ci.yml").read_text()
    unit = ci[ci.index("unit-tests:"):ci.index("regression-tests:")]
    assert "tests/test_torchview_capability_gate.py" in unit
    # the install contract: validate-install must reproduce the #128 propagation too
    assert "torchview-collect-attributes-inputs.patch" in ci


def test_docs_state_the_install_preflight_contract():
    doc = (REPO_ROOT / "docs" / "SOLAR_V1_FLOW.md").read_text()
    assert "capability" in doc.lower() and "--allow-degraded-capture" in doc and "fail" in doc.lower()


# ---- issue #45 review P2-1: ordinary resume must not reuse stale / unrecorded / degraded artifacts -------

def _process_once(tmp_path, force):
    model_file = _write_model(tmp_path)
    proc = PyTorchProcessor(ProcessingConfig(force_rerun=force, safe_mode=False))
    assert proc.process_model_file(str(model_file), str(tmp_path / "graph")) is True
    return tmp_path / "graph" / "pytorch_graph.yaml"


def _mark(path, mutate=None):
    doc = yaml.safe_load(path.read_text())
    doc["_resume_marker"] = "kept-only-if-reused"
    if mutate:
        mutate(doc)
    path.write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))


def _marker_survived(path):
    return "_resume_marker" in yaml.safe_load(path.read_text())


def test_resume_reuses_a_matching_capable_artifact(tmp_path):
    g = _process_once(tmp_path, True)
    _mark(g)
    _process_once(tmp_path, False)
    assert _marker_survived(g), "a matching, capable, provenance-complete artifact is reused"


def test_resume_recaptures_when_the_capability_record_is_missing(tmp_path):
    g = _process_once(tmp_path, True)
    _mark(g, lambda d: (d.pop("capture_capability", None), d.pop("capture_status", None)))
    _process_once(tmp_path, False)
    assert not _marker_survived(g), "a saved graph without a capability record (e.g. produced at the lineage base) is stale"
    assert yaml.safe_load(g.read_text())["capture_status"] == "ok"


def test_resume_recaptures_a_previously_degraded_artifact(tmp_path):
    g = _process_once(tmp_path, True)
    def degrade(d):
        d["capture_status"] = "degraded"; d["capture_capability"]["degraded"] = True; d["capture_capability"]["capable"] = False
    _mark(g, degrade)
    _process_once(tmp_path, False)
    assert not _marker_survived(g), "a degraded artifact is re-captured once the gate is capable"


def test_resume_recaptures_when_the_capture_implementation_changed(tmp_path):
    g = _process_once(tmp_path, True)
    _mark(g, lambda d: d["provenance"].__setitem__("capture_rev", "0" * 64))
    _process_once(tmp_path, False)
    assert not _marker_survived(g)


def test_resume_recaptures_when_the_effective_capability_differs(tmp_path):
    g = _process_once(tmp_path, True)
    _mark(g, lambda d: d["capture_capability"].__setitem__("source", "released"))   # record from another dependency identity
    _process_once(tmp_path, False)
    assert not _marker_survived(g)


def test_capture_identity_covers_the_whole_capture_implementation():
    files = [Path(f).name for f in PyTorchProcessor._capture_implementation_files()]
    assert {"pytorch_processor.py", "torchview_processor.py", "torchview_capability.py", "_torchview_param_shim.py"} <= set(files)
    assert "recorder_tensor.py" in files and "torchview.py" in files      # the installed torchview modules the wrappers patch
    assert len(PyTorchProcessor._capture_rev()) == 64
    g = yaml.safe_load(yaml.dump({"a": 1}))  # yaml available
    ident = PyTorchProcessor._capability_identity(cap.get_torchview_capability(draw_context=None))
    assert set(ident) == {"source", "torchview_version", "torchview_file", "parameter_tensors", "call_attributes", "degraded", "dynamo_warmup"}


# ---- P2-2: capability records are cached per draw context ---------------------------------------------

def test_capability_cache_is_keyed_by_draw_context():
    from solar.graph.pytorch_processor import _patch_torchview_collect_attributes
    cap.reset_capability_cache()
    a = cap.get_torchview_capability(draw_context=None)                            # explicit bare probe (no engine wrapper)
    b = cap.get_torchview_capability(draw_context=_patch_torchview_collect_attributes)   # the processor's context
    assert a.probe["draw_context"] == "none" and "_patch_torchview_collect_attributes" in b.probe["draw_context"]
    assert set(cap._CACHED) == {"none", b.probe["draw_context"]}
    assert cap.get_torchview_capability(draw_context=_patch_torchview_collect_attributes) is b   # memoized per context
    # every engine entry point and the public default use ONE canonical context: the processor's
    assert cap.get_torchview_capability() is b and cap.engine_context_key() == b.probe["draw_context"]
    assert a is not b
    # the record the processor carries is the contextual one, with the warm-up flag serialized
    proc = PyTorchProcessor(ProcessingConfig(safe_mode=False))
    assert proc.capture_capability.probe["draw_context"] == b.probe["draw_context"]
    assert proc.capture_capability.dynamo_warmup is True and "dynamo_warmup" in proc.capture_capability.to_dict()


def test_capability_cache_tolerates_the_legacy_none_reset():
    cap._CACHED = None                                   # how pre-P2-2 harnesses reset the single-record cache
    rec = cap.get_torchview_capability()
    assert isinstance(cap._CACHED, dict) and cap._CACHED[rec.probe["draw_context"]] is rec
    cap.reset_capability_cache()


# ---- P2-4: the probe exercises a functional producer -> consumer chain ----------------------------------

def test_probe_validates_attributes_on_hidden_tensor_consumers():
    rec = cap.probe_torchview_capability(force=True)          # canonical engine context
    chain = rec.probe["chain_consumers_with_attributes"]
    assert set(chain) == {"softmax", "relu"} and all(chain.values()), rec.probe
    assert rec.probe["ops_total"] >= 6                       # linear, add, softmax, conv2d, mul, relu
    assert rec.call_attributes is True
    # the chain is what distinguishes propagation from first-op attributes: a bare probe records the
    # consumers separately, so a context that loses them is reported per consumer, never summed away
    bare = cap.probe_torchview_capability(force=True, draw_context=None)
    assert set(bare.probe["chain_consumers_with_attributes"]) == {"softmax", "relu"}
    assert bare.probe["draw_context"] == "none" and rec.probe["draw_context"] == cap.engine_context_key()


V2_DRIVER = textwrap.dedent('''
    """v2 (released torchview wheel) controls: call orders of the capability cache and an input-only draw context."""
    import contextlib, json, sys
    import torch
    import torchview.recorder_tensor as _rt
    # the released wheel's own recorder dispatch, captured BEFORE solar installs its capture shim (which, since
    # R8, propagates call attributes through hidden tensors by itself): the negative control must draw with it
    RAW_TORCH_FUNCTION = _rt.RecorderTensor.__dict__["__torch_function__"]
    import solar.graph.torchview_capability as cap
    from solar.graph.pytorch_processor import PyTorchProcessor, _patch_torchview_collect_attributes
    from solar.common.types import ProcessingConfig

    @contextlib.contextmanager
    def input_only_context():
        """Reviewer fixture: the input wrapper only (collect_attributes reaches the model inputs) but NOT the
        hidden-output propagation wrapper, so attributes stop at the first hidden tensor.  The wheel's raw
        recorder dispatch is installed for the draw (the capture shim's own propagation, R8, would otherwise make
        this a positive control on any tree carrying it: INTEG-14)."""
        import torchview.torchview as tv_mod
        shim_tf = _rt.RecorderTensor.__dict__["__torch_function__"]
        _rt.RecorderTensor.__torch_function__ = RAW_TORCH_FUNCTION
        from torchview.torchview import (set_device, traverse_data, get_recorder_tensor, get_correct_input_sizes,
                                         get_input_tensor, RecorderTensor, reduce_data_info, collect_tensor_node)
        from torchview.recorder_tensor import NodeContainer
        orig = tv_mod.process_input
        def patched(input_data, input_size, kwargs, device, dtypes=None, collect_attributes=False):
            x = None
            kw = traverse_data(kwargs, lambda t: get_recorder_tensor(t, collect_attributes), type)
            if input_data is not None:
                x = traverse_data(set_device(input_data, device), lambda t: get_recorder_tensor(t, collect_attributes), type)
                if isinstance(x, RecorderTensor):
                    x = [x]
            if input_size is not None:
                dt = dtypes or [torch.float] * len(input_size)
                x = get_input_tensor(get_correct_input_sizes(input_size), dt, device, collect_attributes)
            return x, kw, reduce_data_info([x, kw], collect_tensor_node, NodeContainer())
        tv_mod.process_input = patched
        try:
            yield
        finally:
            tv_mod.process_input = orig
            _rt.RecorderTensor.__torch_function__ = shim_tf

    mode = sys.argv[1]; out = {"mode": mode}
    if mode == "bare-first":
        a = cap.get_torchview_capability(draw_context=None); proc = PyTorchProcessor(ProcessingConfig(safe_mode=False))
        out.update(bare=a.to_dict(), processor=proc.capture_capability.to_dict(), default=cap.get_torchview_capability().to_dict())
    elif mode == "processor-first":
        proc = PyTorchProcessor(ProcessingConfig(safe_mode=False)); a = cap.get_torchview_capability(draw_context=None)
        out.update(bare=a.to_dict(), processor=proc.capture_capability.to_dict(), default=cap.get_torchview_capability().to_dict())
    elif mode == "input-only-chain":
        r = cap.probe_torchview_capability(force=True, draw_context=input_only_context)
        full = cap.probe_torchview_capability(force=True, draw_context=_patch_torchview_collect_attributes)
        out.update(input_only=r.to_dict(), full=full.to_dict())
    out["cache_keys"] = sorted(cap._CACHED)
    print(json.dumps(out))
''')


def _v2(tmp_path, mode):
    (tmp_path / "v2_driver.py").write_text(V2_DRIVER)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for k in ("OPENAI_API_KEY", "SOLAR_API_BASE_URL"):
        env.pop(k, None)
    r = subprocess.run([str(V2_INTERPRETER), str(tmp_path / "v2_driver.py"), mode], env=env, capture_output=True, text=True, timeout=600, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not V2_INTERPRETER.exists(), reason="released-wheel engine not available: these cases run in the v2 Lite profile child interpreter (torchview 0.2.7 wheel, not this process's torchview); absent on this host")
@pytest.mark.parametrize("order", ["bare-first", "processor-first"])
def test_v2_released_wheel_capability_cache_survives_both_call_orders(tmp_path, order):
    """P2-2 on the released wheel: a bare probe (fewer attributes) must never be reused by, nor poison, the
    processor's contextual record; the processor constructs in both call orders and the public default record
    is the processor's."""
    out = _v2(tmp_path, order)
    assert out["processor"]["capable"] is True and out["processor"]["source"] == "released+shim", out["processor"]
    assert "none" in out["cache_keys"] and len(out["cache_keys"]) == 2, out["cache_keys"]
    assert out["bare"]["probe"]["draw_context"] == "none" and "_patch_torchview_collect_attributes" in out["processor"]["probe"]["draw_context"]
    assert out["default"]["probe"]["draw_context"] == out["processor"]["probe"]["draw_context"]
    assert out["bare"]["call_attributes"] is False, out["bare"]["probe"]      # the released wheel's bare draw loses the chain


@pytest.mark.skipif(not V2_INTERPRETER.exists(), reason="released-wheel engine not available: these cases run in the v2 Lite profile child interpreter (torchview 0.2.7 wheel, not this process's torchview); absent on this host")
def test_v2_released_wheel_input_only_context_is_not_capable(tmp_path):
    """P2-4 on the released wheel: with the input wrapper only (no hidden-output propagation) the chain consumers
    lose their attributes and the probe must report call_attributes False; the full context reports True."""
    out = _v2(tmp_path, "input-only-chain")
    assert out["full"]["call_attributes"] is True and all(out["full"]["probe"]["chain_consumers_with_attributes"].values()), out["full"]
    assert out["input_only"]["call_attributes"] is False, out["input_only"]
    assert not all(out["input_only"]["probe"]["chain_consumers_with_attributes"].values()), out["input_only"]["probe"]
