# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mechanical-frontend regression: learned-parameter matmul operands/counts.

Defect (reproduced from the two-path docs smoke, an earlier revision): a bare
``nn.Parameter`` used directly in a function call (``torch.matmul(self.weight,
x)``) is dropped from the traced graph when the installed torchview lacks the
repo's parameter-tensors patch — the matmul node gets ONE input, the weight's
elements vanish from memory accounting, and the batch dim carried only by the
weight is lost from the MAC count (262,144 instead of 1,048,576 for
``(4,32,64) @ (64,128)``).

The repair must keep the explicit two-RecorderTensor path byte-identical:
``a(32,64) @ b(64,128)`` stays 262,144 MACs.

These tests drive the real 3-stage mechanical pipeline (process -> einsum ->
analyze) in subprocesses, exactly as documented for the supported mechanical
graph-lowering path.  CPU-only, bounded.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

SOLAR_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run_pipeline(tmp_path: Path, model_src: str) -> dict:
    model_file = tmp_path / "model.py"
    model_file.write_text(model_src)
    graph_dir = tmp_path / "graph"
    einsum_dir = tmp_path / "einsum"
    analysis_dir = tmp_path / "analysis"
    env_cmd = [PY, "-m"]
    stages = [
        env_cmd + ["solar.cli.process_model", "--model-file", str(model_file),
                   "--output-dir", str(graph_dir)],
        env_cmd + ["solar.cli.toeinsum_model", "--graph-path",
                   str(graph_dir / "pytorch_graph.yaml"),
                   "--output-dir", str(einsum_dir), "--enable-rename"],
        env_cmd + ["solar.cli.analyze_model", "--einsum-graph-path",
                   str(einsum_dir / "einsum_graph_renamed.yaml"),
                   "--output-dir", str(analysis_dir), "--no-copy-graph"],
    ]
    for cmd in stages:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=SOLAR_ROOT, timeout=300)
        assert proc.returncode == 0, (
            f"stage failed: {' '.join(cmd[2:4])}\n{proc.stdout[-1500:]}"
            f"\n{proc.stderr[-1500:]}")
    return {
        "graph": yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text()),
        "analysis": yaml.safe_load(
            (analysis_dir / "analysis.yaml").read_text()),
    }


def _total_macs(analysis: dict) -> int:
    return analysis["total"]["macs"]


def _matmul_layer(graph: dict) -> dict:
    layers = {k: v for k, v in graph["layers"].items()
              if v.get("type") == "matmul"}
    assert len(layers) == 1, f"expected one matmul layer, got {list(layers)}"
    return next(iter(layers.values()))


LEARNED_W_FIRST = textwrap.dedent("""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 32, 64))
        def forward(self, x):
            return torch.matmul(self.weight, x)

    def get_inputs():
        torch.manual_seed(0)
        return [torch.randn(64, 128)]

    def get_init_inputs():
        return []
""")

LEARNED_W_SECOND = textwrap.dedent("""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 128))
        def forward(self, x):
            return torch.matmul(x, self.weight)

    def get_inputs():
        torch.manual_seed(0)
        return [torch.randn(32, 64)]

    def get_init_inputs():
        return []
""")

EXPLICIT_TWO_INPUT = textwrap.dedent("""
    import torch

    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    def get_inputs():
        return [torch.randn(32, 64), torch.randn(64, 128)]

    def get_init_inputs():
        return []
""")


def test_learned_parameter_weight_first(tmp_path):
    """(4,32,64) Parameter @ (64,128) input: weight operand captured, MACs
    4*32*64*128 = 1,048,576 (not the weightless 262,144)."""
    out = _run_pipeline(tmp_path, LEARNED_W_FIRST)
    mm = _matmul_layer(out["graph"])
    shapes = [tuple(s) for s in mm["input_shapes"]]
    assert (4, 32, 64) in shapes, (
        f"learned weight operand missing from matmul inputs: {shapes}")
    assert (64, 128) in shapes, shapes
    assert "weight" in mm.get("input_types", []), mm.get("input_types")
    assert _total_macs(out["analysis"]) == 4 * 32 * 64 * 128


def test_learned_parameter_weight_second(tmp_path):
    """(32,64) input @ (64,128) Parameter: positional order preserved,
    MACs 32*64*128 = 262,144 and the weight's 8,192 elements captured."""
    out = _run_pipeline(tmp_path, LEARNED_W_SECOND)
    mm = _matmul_layer(out["graph"])
    shapes = [tuple(s) for s in mm["input_shapes"]]
    assert shapes == [(32, 64), (64, 128)], (
        f"positional operand order not preserved: {shapes}")
    assert _total_macs(out["analysis"]) == 32 * 64 * 128


def test_explicit_two_input_matmul_unchanged(tmp_path):
    """Pure-activation a @ b path must stay exactly as before the repair."""
    out = _run_pipeline(tmp_path, EXPLICIT_TWO_INPUT)
    mm = _matmul_layer(out["graph"])
    shapes = sorted(tuple(s) for s in mm["input_shapes"])
    assert shapes == [(32, 64), (64, 128)], shapes
    assert _total_macs(out["analysis"]) == 32 * 64 * 128


# ---- independent review P1s (REVIEW.json, 2026-10-21) ----

KW_REVERSED_LEARNED = textwrap.dedent("""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 32, 64))
        def forward(self, x):
            return torch.matmul(other=x, input=self.weight)

    def get_inputs():
        torch.manual_seed(0)
        return [torch.randn(64, 128)]

    def get_init_inputs():
        return []
""")

KW_REVERSED_EXPLICIT = textwrap.dedent("""
    import torch

    class Model(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(other=b, input=a)

    def get_inputs():
        return [torch.randn(32, 64), torch.randn(64, 128)]

    def get_init_inputs():
        return []
""")

SCALAR_REDUCTION_CONSUMER = textwrap.dedent("""
    import torch

    class Model(torch.nn.Module):
        def forward(self, x):
            s = x.sum()
            return s * s

    def get_inputs():
        return [torch.randn(8)]

    def get_init_inputs():
        return []
""")


def test_kwargs_spelling_is_order_invariant_learned(tmp_path):
    """torch.matmul(other=x, input=w) is the SAME operation as
    torch.matmul(w, x): keyword spelling must bind to semantic parameter
    order, never to dict insertion order (reviewer probe: 2,097,152 MACs)."""
    out = _run_pipeline(tmp_path, KW_REVERSED_LEARNED)
    mm = _matmul_layer(out["graph"])
    shapes = [tuple(s) for s in mm["input_shapes"]]
    assert shapes == [(4, 32, 64), (64, 128)], shapes
    assert _total_macs(out["analysis"]) == 4 * 32 * 64 * 128


def test_kwargs_spelling_is_order_invariant_explicit(tmp_path):
    """matmul(other=b, input=a) == a @ b: 262,144 MACs (reviewer probe
    reported 524,288 from the reversed kwargs insertion order)."""
    out = _run_pipeline(tmp_path, KW_REVERSED_EXPLICIT)
    mm = _matmul_layer(out["graph"])
    shapes = [tuple(s) for s in mm["input_shapes"]]
    assert shapes == [(32, 64), (64, 128)], shapes
    assert _total_macs(out["analysis"]) == 32 * 64 * 128


# ---- round-2 review: unresolvable ordering must FAIL CLOSED (no silent
# ----                  mechanical success with incomplete accounting) ----

UNKNOWN_SIGNATURE_INJECTED = KW_REVERSED_LEARNED.replace(
    "return torch.matmul(other=x, input=self.weight)",
    "from solar.graph import _torchview_param_shim as shim\n"
    "        shim._SIG_CACHE[torch.matmul] = None\n"
    "        return torch.matmul(other=x, input=self.weight)")

KWARGS_CATCHALL_INJECTED = KW_REVERSED_LEARNED.replace(
    "return torch.matmul(other=x, input=self.weight)",
    "import inspect\n"
    "        from solar.graph import _torchview_param_shim as shim\n"
    "        shim._SIG_CACHE[torch.matmul] = "
    "inspect.signature(lambda *a, **kw: None)\n"
    "        return torch.matmul(other=x, input=self.weight)")


def _run_stage1(tmp_path: Path, model_src: str):
    model_file = tmp_path / "model.py"
    model_file.write_text(model_src)
    graph_dir = tmp_path / "graph"
    proc = subprocess.run(
        [PY, "-m", "solar.cli.process_model", "--model-file",
         str(model_file), "--output-dir", str(graph_dir)],
        capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
    return proc, graph_dir


def test_unknown_signature_fails_closed(tmp_path):
    """Controlled signature-unavailable injection (reviewer probe): with a
    learned Parameter whose kwargs cannot be semantically bound, mechanical
    extraction must raise a typed unsupported diagnostic — never exit 0 with
    the weight silently dropped (262,144-MAC incomplete accounting)."""
    proc, graph_dir = _run_stage1(tmp_path, UNKNOWN_SIGNATURE_INJECTED)
    assert proc.returncode != 0, (
        "stage 1 succeeded on an unbindable-tensor-kwargs call:\n"
        + proc.stdout[-800:])
    assert "UnsupportedFrontendOperand" in (proc.stdout + proc.stderr)
    assert not (graph_dir / "pytorch_graph.yaml").exists(), (
        "trusted mechanical artifact produced despite fail-closed diagnostic")


def test_kwargs_catchall_fails_closed(tmp_path):
    """Tensors landing in a **kwargs catch-all have no semantic operand
    order: must fail closed with the typed diagnostic."""
    proc, graph_dir = _run_stage1(tmp_path, KWARGS_CATCHALL_INJECTED)
    assert proc.returncode != 0, proc.stdout[-800:]
    assert "UnsupportedFrontendOperand" in (proc.stdout + proc.stderr)
    assert not (graph_dir / "pytorch_graph.yaml").exists()


def test_scalar_reduction_consumer_edge_preserved(tmp_path):
    """s = x.sum(); s * s — the 0-d RecorderTensor operand must keep the
    sum -> mul connectivity and both scalar operand occurrences; unpatched
    torchview retains this edge and the shim must not remove it."""
    model_file = tmp_path / "model.py"
    model_file.write_text(SCALAR_REDUCTION_CONSUMER)
    graph_dir = tmp_path / "graph"
    proc = subprocess.run(
        [PY, "-m", "solar.cli.process_model", "--model-file",
         str(model_file), "--output-dir", str(graph_dir)],
        capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    graph = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
    mul_layers = {k: v for k, v in graph["layers"].items()
                  if "mul" in str(v.get("type", ""))}
    assert mul_layers, f"mul node missing: {list(graph['layers'])}"
    mul = next(iter(mul_layers.values()))
    assert mul["connections"]["inputs"], (
        "scalar producer edge dropped: mul has no inputs")
    shapes = [tuple(s) for s in mul["input_shapes"]]
    assert shapes == [(), ()], (
        f"expected two scalar operand occurrences, got {shapes}")
