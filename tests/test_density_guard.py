# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention convention guard and real CLI/API metadata regression checks."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from solar.einsum_lib import build_metadata, get_trace, set_density_profiling
from solar.einsum_lib.verify import execute_translated
from test_sparse_cost import _analyze

SOLAR = Path(__file__).resolve().parents[1]
DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims

@kernel
def probs(X: Tensor("S")) -> Tensor("S,T"):
    Y = Tensor(S=X.shape[0], T=X.shape[0])
    for s in Dim("S"):
        denom = 0.0
        for t in Dim("T"):
            denom += exp(X[t]) * max(0, min(1, s-t+1))
        for t in Dim("T"):
            Y[s,t] = exp(X[t]) * max(0, min(1, s-t+1)) / denom
    return Y

@kernel
def pv(P: Tensor("S,T"), V: Tensor("T,D")) -> Tensor("S,D"):
    Y = Tensor(S=P.shape[0], D=V.shape[1])
    for s in Dim("S"):
        for d in Dim("D"):
            for t in Dim("T"):
                Y[s,d] += P[s,t] * V[t,d]
    return Y

def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "S")
    v = Tensor.from_numpy(inputs["v"], "T,D")
    return {"output": pv(probs(x), v).data}
'''


def _inputs(s=4):
    return {"x": np.ones(s), "v": np.ones((s, 3))}


@pytest.mark.parametrize("allow", [False, True])
def test_attention_guard_controls_pv_cost(tmp_path, allow):
    inputs = _inputs()
    set_density_profiling(True, measure_attention=allow)
    try:
        outputs, _ = execute_translated(DSL, {}, all_inputs=inputs,
                                        trace=True, fast=False)
        md = build_metadata(get_trace())
    finally:
        set_density_profiling(False)
    np.testing.assert_allclose(outputs["output"], np.ones((4, 3)))
    if allow:
        assert md["tensors"]["probs#0"]["sparsity"] == pytest.approx(10 / 16)
    else:
        assert "sparsity" not in md["tensors"]["probs#0"]
    analysis = _analyze(DSL, md, tmp_path)
    assert analysis["layers"]["pv#0"]["macs"] == pytest.approx(
        4 * 4 * 3 * (10 / 16 if allow else 1))


def test_explicit_role_guards_opaque_kernel_and_annotation_wins():
    code = DSL.replace("probs", "opaque").replace(
        'Tensor(S=X.shape[0], T=X.shape[0])',
        'Tensor(S=X.shape[0], T=X.shape[0], role="attention_probability")')
    for annotation in (False, True):
        source = code.replace('role="attention_probability"',
                              'role="attention_probability", sparsity=0.25') if annotation else code
        set_density_profiling(True)
        try:
            execute_translated(source, {}, all_inputs=_inputs(), trace=True, fast=False)
            meta = build_metadata(get_trace())["tensors"]["opaque#0"]
        finally:
            set_density_profiling(False)
        assert meta.get("sparsity") == (0.25 if annotation else None)


@pytest.mark.parametrize("allow", [False, True])
def test_kernel_guard_rechecks_from_numpy_output(allow):
    # The constructor sees all zeros before the kernel fills its output.
    # That provisional measurement must not bypass the output-name guard.
    code = DSL.replace('Y = Tensor(S=X.shape[0], T=X.shape[0])',
                       'Dim("T", size=X.shape[0])\n'
                       '    Y = Tensor.from_numpy(np.zeros((X.shape[0], X.shape[0])), "S,T")')
    set_density_profiling(True, measure_attention=allow)
    try:
        execute_translated(code, {}, all_inputs=_inputs(), trace=True, fast=False)
        meta = build_metadata(get_trace())["tensors"]["probs#0"]
    finally:
        set_density_profiling(False)
    assert meta.get("sparsity") == (10 / 16 if allow else None)


@pytest.mark.parametrize("full_size", [4, 8])
@pytest.mark.parametrize("mode", ["guarded", "allow", "override"])
def test_trace_cli_guard_same_shape_and_revalidation(tmp_path, full_size, mode):
    definition = {"axes": {"S": {"type": "const", "value": full_size}},
                  "inputs": {"x": {"shape": ["S"]}, "v": {"shape": ["S", 3]}}}
    (tmp_path / "definition.json").write_text(json.dumps(definition))
    (tmp_path / "model_einsum_lib.py").write_text(DSL)
    (tmp_path / "adapter.py").write_text(
        'import numpy as np\ndef get_inputs():\n'
        '    return {"x": np.ones(4), "v": np.ones((4,3))}\n')
    flags = ["--measure-attention-density"] if mode == "allow" else []
    if mode == "override":
        flags += ["--density-override", "probs#0=0.25"]
    p = subprocess.run(
        [sys.executable, "scripts/trace_saved_ir.py", str(tmp_path / "model_einsum_lib.py"),
         "--definition", str(tmp_path), "--out-dir", str(tmp_path / "out"),
         "--fallback-adapter", str(tmp_path / "adapter.py"), "--measure-density", *flags],
        cwd=SOLAR, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    md = yaml.safe_load((tmp_path / "out/model_einsum_lib.yaml").read_text())
    meta = md["tensors"]["probs#0"]
    assert list(meta["dims"].values()) == [full_size, full_size]
    if mode == "guarded":
        assert "sparsity" not in meta
        assert "GUARDED probs" in p.stderr
    else:
        expected = 0.25 if mode == "override" else (full_size + 1) / (2 * full_size)
        assert meta["sparsity"] == pytest.approx(expected)
        if mode == "allow" and full_size == 8:
            assert meta["sparsity_source"] == "revalidated_original"


@pytest.mark.parametrize("allow,full_size", [(False, 4), (True, 4), (True, 8)])
def test_api_path_exports_measurements_without_network(monkeypatch, allow, full_size):
    from solar.einsum_lib.agent import TranslationAgent
    monkeypatch.setattr(TranslationAgent, "_init_client", lambda self: None)
    monkeypatch.setattr(TranslationAgent, "_call_llm", lambda *args: DSL)
    agent = TranslationAgent(api_key="test-only", max_retries=1, fast=True,
                             use_fp64=True, verify_timeout=30)
    inputs = _inputs()
    set_density_profiling(True, source="measured_capped" if full_size == 8 else "measured",
                          measure_attention=allow)
    try:
        result = agent.translate(
            "test reference", {k: v.shape for k, v in _inputs(full_size).items()},
            reference_outputs={"output": np.ones((4, 3))}, all_inputs=inputs,
            full_inputs=_inputs(full_size))
    finally:
        set_density_profiling(False)
    assert result.verified, result.verification_error
    meta = result.metadata["tensors"]["probs#0"]
    assert meta.get("sparsity") == (10 / 16 if allow and full_size == 4 else None)


@pytest.mark.parametrize("command", [["scripts/trace_saved_ir.py", "fake.py",
                                     "--definition", ".", "--out-dir", "unused"],
                                    ["-m", "solar.cli.toDSL_model", "fake.py"]])
def test_attention_flag_requires_measure_density(command):
    p = subprocess.run([sys.executable, *command, "--measure-attention-density"],
                       cwd=SOLAR, capture_output=True, text=True, timeout=30)
    assert p.returncode == 2
    assert "requires --measure-density" in p.stderr
