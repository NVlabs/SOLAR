# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration reconciliation regressions (isolated integration checkout).

The recovery shim (per-occurrence, fail-closed operand capture) meets the
final-release staged converter (PR #10 signature resolution + PR #7
repair A' constant synthesis).  These tests pin the reconciliation
decisions:

* complete shim capture means A' must NOT fire (no appended synthetic
  operand, no position scramble);
* repeated parameter ``w @ w`` keeps BOTH occurrences (the on-disk PR10
  torchview patch dedups by id; the shim's per-occurrence slots must win)
  and must not trigger a spurious A' synthesis;
* estimate() classifies the shim's typed fail-closed diagnostic as
  UNSUPPORTED (agentic territory), not a pipeline error.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

SOLAR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOLAR_ROOT))
PY = sys.executable

REPEATED_W = textwrap.dedent("""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            return torch.matmul(self.weight, self.weight) + x

    def get_inputs():
        torch.manual_seed(0)
        return [torch.randn(4, 4)]

    def get_init_inputs():
        return []
""")


def _stage12(tmp_path, src):
    model = tmp_path / "model.py"
    model.write_text(src)
    graph_dir = tmp_path / "graph"
    proc = subprocess.run(
        [PY, "-m", "solar.cli.process_model", "--model-file", str(model),
         "--output-dir", str(graph_dir)],
        capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    from solar.einsum import PyTorchToEinsum
    out = tmp_path / "einsum"
    graph = PyTorchToEinsum(enable_agent=False).convert(
        graph_dir / "pytorch_graph.yaml", out)
    return graph_dir, graph


REPEATED_PARAM_ONE_CALL = textwrap.dedent("""
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            return torch.addcmul(x, self.weight, self.weight)

    def get_inputs():
        torch.manual_seed(0)
        return [torch.randn(4, 4)]

    def get_init_inputs():
        return []
""")


class TestRepeatedParameter:
    def test_same_parameter_twice_in_one_call_keeps_both_slots(
            self, tmp_path):
        """addcmul(x, w, w): the SAME parameter appears twice in one traced
        call — per-occurrence capture must keep three operand slots (the
        PR10 on-disk patch dedups by id; the shim's recorder must win), and
        no spurious operand may be synthesized on top."""
        model = tmp_path / "model.py"
        model.write_text(REPEATED_PARAM_ONE_CALL)
        graph_dir = tmp_path / "graph"
        proc = subprocess.run(
            [PY, "-m", "solar.cli.process_model", "--model-file",
             str(model), "--output-dir", str(graph_dir)],
            capture_output=True, text=True, cwd=SOLAR_ROOT, timeout=300)
        assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
        pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
        node = next(v for v in pg["layers"].values()
                    if v.get("type") == "addcmul")
        assert node["input_shapes"] == [[4, 4], [4, 4], [4, 4]], (
            f"per-occurrence parameter slots lost: {node['input_shapes']}")

    def test_all_parameter_call_stays_untraced_constant(self, tmp_path):
        """matmul(w, w) with NO traced operand never reaches
        __torch_function__: it remains an untraced constant subgraph (the
        documented derived-parameter-view limitation class). Pinned so the
        limitation is explicit, not silently assumed traceable."""
        graph_dir, _ = _stage12(tmp_path, REPEATED_W)
        pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
        types = {v.get("type") for v in pg["layers"].values()}
        assert "matmul" not in types  # untraced: constant feeding the add
        assert "add" in types


class TestAPrimeSuppression:
    def test_complete_capture_means_no_synthesis_and_correct_order(
            self, tmp_path):
        """The learned-parameter case the recovery repaired: with complete
        shim capture the staged converter must keep EXACTLY the two real
        operands in positional order (weight first) — A' would have
        appended the missing one at the END as an 'input'."""
        src = (SOLAR_ROOT / "examples" / "Matmul" / "Matmul.py").read_text()
        graph_dir, graph = _stage12(tmp_path, src)
        pg = yaml.safe_load((graph_dir / "pytorch_graph.yaml").read_text())
        mm = next(v for v in pg["layers"].values()
                  if v.get("type") == "matmul")
        assert mm["input_shapes"] == [[4, 32, 64], [64, 128]], (
            "operand order/arity wrong — A' interplay regressed the "
            "recovery's learned-parameter capture")


class TestEstimateFailClosed:
    def test_typed_unsupported_routes_agentic_not_pipeline_error(
            self, tmp_path):
        """The shim's fail-closed diagnostic is a genuine unsupported-source
        signal: estimate(mode='v0') must raise UnsupportedError (agentic
        fallback territory), never PipelineError."""
        from solar.estimate import PipelineError, UnsupportedError, estimate
        src = REPEATED_W.replace(
            "return torch.matmul(self.weight, self.weight) + x",
            "import inspect\n"
            "        from solar.graph import _torchview_param_shim as shim\n"
            "        shim._SIG_CACHE[torch.matmul] = "
            "inspect.signature(lambda *a, **kw: None)\n"
            "        return torch.matmul(other=x, input=self.weight)")
        model = tmp_path / "model.py"
        model.write_text(src)
        with pytest.raises(UnsupportedError):
            try:
                estimate(model, tmp_path / "out", mode="v0")
            except PipelineError as e:
                pytest.fail(f"typed fail-closed diagnostic surfaced as "
                            f"PipelineError: {e}")
