# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank labelling is reproducible across Python processes (issue #43).

The original finding: ``EinsumRankRenamer`` allocated fresh labels while
iterating a ``set`` of tokens, so byte-identical pre-rename graphs were
renamed differently from one Python process to the next (hash-seed
dependent).  The renamer was removed at the lineage base in favour of
``solar.einsum.af_graph_builder`` (``R0``, ``R1``, ... in topological order
of first appearance).  These tests pin the guarantee on the successor: the
same captured graph converts to byte-identical ``einsum_graph.yaml``,
``einsum_graph_renamed.yaml`` and ``af_einsum_graph.yaml`` in fresh
subprocesses under different ``PYTHONHASHSEED`` values, and twice in one
process.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph.pytorch_processor import ProcessingConfig, PyTorchProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("einsum_graph.yaml", "einsum_graph_renamed.yaml", "af_einsum_graph.yaml")

TINY_MODEL = textwrap.dedent('''
    import torch, torch.nn as nn, torch.nn.functional as F
    class Model(nn.Module):
        """Two linears sharing a hidden axis, a residual add and a softmax: enough distinct axes to relabel."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 10)
            self.fc2 = nn.Linear(10, 6)
        def forward(self, x):
            h = F.relu(self.fc1(x))
            return F.softmax(self.fc2(h) + x, dim=-1)
    def get_inputs():
        return [torch.randn(3, 6)]
    def get_init_inputs():
        return []
''')


@pytest.fixture(scope="module")
def captured_graph(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rank_label_determinism")
    model_file = root / "tiny_model.py"
    model_file.write_text(TINY_MODEL)
    assert PyTorchProcessor(ProcessingConfig()).process_model_file(str(model_file), str(root / "graph")) is True
    return root / "graph" / "pytorch_graph.yaml"


def _digests(out_dir: Path) -> dict:
    return {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest() for name in OUTPUTS}


def _convert_in_subprocess(graph_path: Path, seed: int, out_dir: Path) -> dict:
    code = textwrap.dedent(f'''
        from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
        assert PyTorchToEinsum().convert({str(graph_path)!r}, {str(out_dir)!r}, copy_graph=False, enable_rename=True) is not None
    ''')
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return _digests(out_dir)


def test_converted_graphs_are_byte_identical_across_hash_seeds(captured_graph, tmp_path):
    outputs = {seed: _convert_in_subprocess(captured_graph, seed, tmp_path / f"seed{seed}") for seed in (0, 1, 2, 42)}
    first = next(iter(outputs.values()))
    assert all(d == first for d in outputs.values()), outputs
    for name in OUTPUTS:
        assert (tmp_path / "seed0" / name).stat().st_size > 0, name


def test_renamed_graph_is_a_copy_of_the_einsum_graph_at_this_lineage(captured_graph, tmp_path):
    digests = _convert_in_subprocess(captured_graph, 7, tmp_path / "copy")
    assert digests["einsum_graph_renamed.yaml"] == digests["einsum_graph.yaml"]


def test_two_in_process_runs_agree(captured_graph, tmp_path):
    for i in (1, 2):
        assert PyTorchToEinsum().convert(str(captured_graph), str(tmp_path / f"run{i}"), copy_graph=False, enable_rename=True) is not None
    assert _digests(tmp_path / "run1") == _digests(tmp_path / "run2")


def test_af_graph_uses_canonical_r_labels_in_first_appearance_order(captured_graph, tmp_path):
    import re
    import yaml
    _convert_in_subprocess(captured_graph, 3, tmp_path / "labels")
    af = yaml.safe_load((tmp_path / "labels" / "af_einsum_graph.yaml").read_text())
    sizes = af["workload"]["rank_sizes"]
    labels = list(sizes)
    assert labels == [f"R{i}" for i in range(len(labels))], labels   # dense R0..Rn-1 in order of appearance
    assert all(isinstance(v, int) and v > 0 for v in sizes.values()), sizes


def test_ci_unit_lane_selects_this_module():
    ci = (REPO_ROOT / ".gitlab-ci.yml").read_text()
    unit = ci.split("unit-tests:")[1].split("regression-tests:")[0]
    assert "tests/test_rank_label_determinism.py" in unit
    assert "tests/test_einsum_rank_renamer_determinism.py" not in unit
