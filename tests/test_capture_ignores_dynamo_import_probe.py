# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The capture must not record torch._dynamo's import-time builtin probe (issue #45 KB delta finding).

``torch._dynamo.variables.torch_function.populate_builtin_to_tensor_fn_map()`` executes every builtin
operator on a one-element tensor subclass when ``torch._dynamo`` is first imported.  If that import is
triggered lazily inside a model's forward during a capture, the recorder sees the probe calls and
torchview emits stray FunctionNodes (``__ilshift__``, ``positive``, scalar ``matmul``, ...) into the
graph: Mamba2 at the lineage base shows 224 nodes when captured first in a process and 86 after any earlier
meta draw.  ``PyTorchProcessor`` now imports ``torch._dynamo`` before capturing.  Each case runs in a
fresh interpreter so ``torch._dynamo`` is really not yet imported.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL = textwrap.dedent('''
    import torch, torch.nn as nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
        def forward(self, x):
            import torch._dynamo   # lazy import inside the traced forward (what einops/einsum paths do)
            return self.fc(x)
    def get_inputs():
        return [torch.randn(2, 4)]
    def get_init_inputs():
        return []
''')

DRIVER = textwrap.dedent('''
    import json, sys, pathlib, yaml
    import solar.graph.pytorch_processor as pp
    if sys.argv[3] == "no-warmup":
        pp._import_torch_dynamo_before_capture = lambda: False     # the hazard, for the control
    assert "torch._dynamo" not in sys.modules, "precondition: dynamo not imported before the processor"
    proc = pp.PyTorchProcessor(pp.ProcessingConfig(force_rerun=True))
    assert proc.process_model_file(sys.argv[1], sys.argv[2]) is True
    layers = yaml.safe_load((pathlib.Path(sys.argv[2]) / "pytorch_graph.yaml").read_text())["layers"]
    fn = sorted(v["type"] for v in layers.values() if v.get("node_class") == "FunctionNode")
    print(json.dumps({"function_nodes": fn, "n_layers": len(layers), "dynamo_warmup": getattr(proc, "dynamo_warmup", None)}))
''')


def _run(tmp_path, mode):
    (tmp_path / "model.py").write_text(MODEL)
    (tmp_path / "driver.py").write_text(DRIVER)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(tmp_path / "driver.py"), str(tmp_path / "model.py"), str(tmp_path / f"graph_{mode}"), mode],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_lazy_dynamo_import_inside_forward_adds_no_nodes(tmp_path):
    out = _run(tmp_path, "warmup")
    assert out["dynamo_warmup"] is True
    assert out["function_nodes"] == ["linear"], out


def test_without_the_warmup_the_probe_is_recorded_on_an_affected_torch(tmp_path):
    """Negative control, conditional on the installed torch actually exhibiting the hazard
    (issue #45 review P2-3): torch 2.7.1 records ~138 stray nodes when torch._dynamo is first imported
    during a capture; torch 2.9.0 does not.  Documents the hazard where it exists and is skipped (never required)
    on an unaffected dependency; the positive test above stays mandatory."""
    import torch
    out = _run(tmp_path, "no-warmup")
    assert out["dynamo_warmup"] is False
    stray = [t for t in out["function_nodes"] if t != "linear"]
    if not stray:
        pytest.skip(f"torch {torch.__version__} does not execute the dynamo import-time builtin probe inside a "
                    "capture; hazard not present on this dependency (positive test still enforced)")
    assert {"positive", "__ilshift__"} <= set(stray), out


def test_ci_unit_lane_selects_this_module():
    ci = (REPO_ROOT / ".gitlab-ci.yml").read_text()
    unit = ci.split("unit-tests:")[1].split("regression-tests:")[0]
    assert "tests/test_capture_ignores_dynamo_import_probe.py" in unit
