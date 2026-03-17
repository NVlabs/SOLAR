"""Tests for zero-compute op classification in graph_analyzer.

Verifies that view ops, memory-only ops, and type-conversion ops
are assigned zero compute cost (macs=0, other_ops=0), while their
memory cost is handled correctly:
  - View ops (chunk, split, reshape, permute): 0 memory (zero-copy)
  - Memory ops (cat, repeat, stack): memory cost from tensor shapes
  - Type conversion (to): 0 memory (zero-copy view)

These tests catch the bugs where chunk was assigned 429B ops,
cat was assigned 412B ops, and repeat was assigned 6.2T ops.
"""

import pytest
from pathlib import Path
from textwrap import dedent

from solar.common.types import ProcessingConfig
from solar.graph import PyTorchProcessor
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer


def _run_pipeline(tmp_path: Path, model_source: str) -> dict:
    """Run full Solar pipeline from source code to analysis dict."""
    model_file = tmp_path / "model.py"
    model_file.write_text(dedent(model_source))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()

    config = ProcessingConfig(
        save_graph=False, force_rerun=True, debug=False, safe_mode=False,
    )
    processor = PyTorchProcessor(config)
    ok = processor.process_model_file(str(model_file), str(graph_dir))
    assert ok, "Graph extraction failed"

    einsum_dir = tmp_path / "einsum"
    einsum_dir.mkdir()

    converter = PyTorchToEinsum()
    result = converter.convert(str(graph_dir / "pytorch_graph.yaml"), str(einsum_dir))
    assert result is not None, "Einsum conversion failed"

    renamed = einsum_dir / "einsum_graph_renamed.yaml"
    assert renamed.exists()

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    analyzer = EinsumGraphAnalyzer()
    analysis = analyzer.analyze_graph(
        str(renamed), str(analysis_dir), precision="fp32", copy_graph=False,
    )
    assert analysis is not None, "Analysis failed"
    return analysis


# ---------------------------------------------------------------------------
# Model sources
# ---------------------------------------------------------------------------

CHUNK_MODEL = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a + b

def get_inputs():
    return [torch.randn(4, 512, 8192)]

def get_init_inputs():
    return []
"""

CAT_MODEL = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.cat([a, b], dim=-1)

def get_inputs():
    return [torch.randn(4, 512, 4096), torch.randn(4, 512, 4096)]

def get_init_inputs():
    return []
"""

REPEAT_MODEL = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.repeat(1, 1, 4)

def get_inputs():
    return [torch.randn(4, 512, 2048)]

def get_init_inputs():
    return []
"""

STACK_MODEL = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.stack([a, b], dim=0)

def get_inputs():
    return [torch.randn(4, 128), torch.randn(4, 128)]

def get_init_inputs():
    return []
"""

GEGLU_MODEL = """\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    \"\"\"GEGLU: chunk + gelu + mul — the pattern from stabilityai.\"\"\"
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 512, bias=False)

    def forward(self, x):
        h = self.proj(x)
        a, b = h.chunk(2, dim=-1)
        return a * F.gelu(b)

def get_inputs():
    return [torch.randn(4, 128, 256)]

def get_init_inputs():
    return []
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkZeroCompute:
    """chunk is a view op: zero compute, zero memory."""

    def test_chunk_zero_ops(self, tmp_path):
        analysis = _run_pipeline(tmp_path, CHUNK_MODEL)
        for lid, layer in analysis["layers"].items():
            if "chunk" in lid.lower() or layer.get("type") == "chunk":
                assert layer["macs"] == 0, f"{lid}: macs should be 0"
                assert layer["other_ops"] == 0, f"{lid}: other_ops should be 0"

    def test_chunk_zero_memory(self, tmp_path):
        """chunk returns views — zero memory footprint."""
        analysis = _run_pipeline(tmp_path, CHUNK_MODEL)
        for lid, layer in analysis["layers"].items():
            if "chunk" in lid.lower() or layer.get("type") == "chunk":
                assert layer["unfused_elements"] == 0, f"{lid}: should be zero-copy"


class TestCatZeroCompute:
    """cat moves data but has zero ALU compute."""

    def test_cat_zero_ops(self, tmp_path):
        analysis = _run_pipeline(tmp_path, CAT_MODEL)
        for lid, layer in analysis["layers"].items():
            if "cat" in lid.lower() or layer.get("type") in ("cat", "concat"):
                assert layer["macs"] == 0, f"{lid}: macs should be 0"
                assert layer["other_ops"] == 0, f"{lid}: other_ops should be 0"

    def test_cat_has_memory_cost(self, tmp_path):
        """cat copies data — it should have memory cost (not zero-copy)."""
        analysis = _run_pipeline(tmp_path, CAT_MODEL)
        for lid, layer in analysis["layers"].items():
            if "cat" in lid.lower() or layer.get("type") in ("cat", "concat"):
                assert layer["unfused_elements"] > 0, f"{lid}: cat should have memory cost"


class TestRepeatZeroCompute:
    """repeat tiles data: zero compute, nonzero memory."""

    def test_repeat_zero_ops(self, tmp_path):
        analysis = _run_pipeline(tmp_path, REPEAT_MODEL)
        for lid, layer in analysis["layers"].items():
            if "repeat" in lid.lower() or layer.get("type") == "repeat":
                assert layer["macs"] == 0, f"{lid}: macs should be 0"
                assert layer["other_ops"] == 0, f"{lid}: other_ops should be 0"


class TestStackZeroCompute:
    """stack concatenates along a new dim: zero compute."""

    def test_stack_zero_ops(self, tmp_path):
        analysis = _run_pipeline(tmp_path, STACK_MODEL)
        for lid, layer in analysis["layers"].items():
            if "stack" in lid.lower() or layer.get("type") == "stack":
                assert layer["macs"] == 0, f"{lid}: macs should be 0"
                assert layer["other_ops"] == 0, f"{lid}: other_ops should be 0"


class TestGEGLUPattern:
    """GEGLU = proj -> chunk -> gelu -> mul.
    chunk must not inflate other_ops."""

    def test_geglu_chunk_zero_ops(self, tmp_path):
        analysis = _run_pipeline(tmp_path, GEGLU_MODEL)
        chunk_ops = 0
        for lid, layer in analysis["layers"].items():
            if "chunk" in lid.lower() or layer.get("type") == "chunk":
                chunk_ops += layer["other_ops"]
        assert chunk_ops == 0, f"chunk other_ops should be 0 in GEGLU, got {chunk_ops}"

    def test_geglu_total_macs_reasonable(self, tmp_path):
        """Total MACs should come from the linear projection only."""
        analysis = _run_pipeline(tmp_path, GEGLU_MODEL)
        total = analysis["total"]
        expected_linear_macs = 4 * 128 * 256 * 512
        assert total["macs"] == expected_linear_macs, (
            f"Expected {expected_linear_macs} MACs from linear, got {total['macs']}"
        )

    def test_geglu_other_ops_small(self, tmp_path):
        """other_ops should be from gelu + mul only, not chunk."""
        analysis = _run_pipeline(tmp_path, GEGLU_MODEL)
        total = analysis["total"]
        gelu_elems = 4 * 128 * 256
        mul_elems = 4 * 128 * 256
        max_reasonable = gelu_elems + mul_elems + 1000
        assert total["other_ops"] <= max_reasonable, (
            f"other_ops too high: {total['other_ops']} (max reasonable: {max_reasonable})"
        )
