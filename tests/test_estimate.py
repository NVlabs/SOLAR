# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing/error-contract tests, plus actual validation and analysis integration."""
from pathlib import Path
from unittest.mock import Mock

import json
import pytest
import solar.estimate as api

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "skills/verified-translation/example"


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "model.py"
    p.write_text("# source fixture\n")
    return p


def test_explicit_v0_never_calls_llm(monkeypatch, source, tmp_path):
    llm = Mock(side_effect=AssertionError("LLM must not be called"))
    monkeypatch.setattr(api, "_translate_api", llm)
    with pytest.raises(api.UnsupportedError):
        api.estimate(source, tmp_path, mode="v0")
    llm.assert_not_called()
    assert not list(tmp_path.glob("run-*/result.json"))


def test_auto_supported_no_llm(monkeypatch, source, tmp_path):
    monkeypatch.setattr(api, "_mechanical", lambda *args: (source, {"kind": "mechanical"}))
    llm = Mock(side_effect=AssertionError("LLM must not be called"))
    monkeypatch.setattr(api, "_translate_api", llm)
    typed = {"aggregate": 3, "contraction": {"macs": 3, "by_compute_dtype": {"workload_precision": 3}}, "scalar_statement_multiplies": {"macs": 0, "layers": {}}, "components_sum_to_aggregate": True}
    monkeypatch.setattr(api, "_analyze", lambda *args: (1, 2, 3, typed))   # issue #58: _analyze returns (fused, unfused, macs, macs_typed)
    r = api.estimate(source, tmp_path)
    assert r.requested_mode == "auto" and r.used_mode == "v0"
    llm.assert_not_called()
    assert r.macs == 3 and r.macs_typed == typed                              # the typed components propagate to the result ...
    assert json.loads(next(tmp_path.glob("run-*/result.json")).read_text())["macs_typed"] == typed   # ... and to result.json


def test_auto_only_falls_back_on_unsupported(monkeypatch, source, tmp_path):
    llm = Mock(return_value=source)
    monkeypatch.setattr(api, "_translate_api", llm)
    monkeypatch.setattr(api, "_validated_ir", lambda *args: (source, {"numeric_comparison": True}))
    typed = {"aggregate": 3, "contraction": {"macs": 3, "by_compute_dtype": {"workload_precision": 3}}, "scalar_statement_multiplies": {"macs": 0, "layers": {}}, "components_sum_to_aggregate": True}
    monkeypatch.setattr(api, "_analyze", lambda *args: (1, 2, 3, typed))   # issue #58: four-value private result contract
    for error in (api.PipelineError("broken environment"), MemoryError("oom"), TimeoutError("timeout")):
        monkeypatch.setattr(api, "_mechanical", Mock(side_effect=error))
        with pytest.raises(type(error)):
            api.estimate(source, tmp_path)
    llm.assert_not_called()
    monkeypatch.setattr(api, "_mechanical", Mock(side_effect=api.UnsupportedError("sort")))
    r = api.estimate(source, tmp_path)
    assert r.used_mode == "v1" and r.fallback_reason == "sort"
    llm.assert_called_once()
    assert r.macs_typed == typed                                               # propagates on the fallback path too


@pytest.mark.parametrize("mode", ["v1", "v2"])
def test_invalid_ir_cannot_reach_analysis(monkeypatch, source, tmp_path, mode):
    monkeypatch.setattr(api, "_translate_api", Mock(return_value=source))
    monkeypatch.setattr(api, "_validated_ir", Mock(side_effect=api.ValidationError("mismatch")))
    analysis = Mock(side_effect=AssertionError("must not emit SOL"))
    monkeypatch.setattr(api, "_analyze", analysis)
    with pytest.raises(api.ValidationError):
        api.estimate(source, tmp_path, mode=mode, **({"ir_path": source} if mode == "v2" else {}))
    analysis.assert_not_called()
    assert not list(tmp_path.glob("run-*/result.json"))


def test_v2_requires_ir_without_calling_llm(monkeypatch, source, tmp_path):
    monkeypatch.setattr(api, "_translate_api", Mock(side_effect=AssertionError("no hidden API")))
    with pytest.raises(api.AgentTranslationRequired):
        api.estimate(source, tmp_path, mode="v2")
    assert list(tmp_path.glob("run-*/AGENT_TASK.md"))
    assert not list(tmp_path.glob("run-*/result.json"))


def test_v2_end_to_end_example(tmp_path):
    result = api.estimate(EXAMPLE / "adapter.py", tmp_path, mode="v2", source_type="cuda",
                          ir_path=EXAMPLE / "solar_model_einsum_lib.py", precision="fp32")
    assert result.validation["numeric_comparison"] is True
    assert result.fused_ms > 0 and result.unfused_ms >= result.fused_ms
    assert result.validation["max_abs_error"] < 1e-4


def test_wrong_ir_fails_actual_numerical_gate(tmp_path):
    ir = tmp_path / "bad.py"
    ir.write_text("def run_model(inputs):\n    return {'output': inputs['hidden_states'] * 0}\n")
    with pytest.raises(api.ValidationError):
        api.estimate(EXAMPLE / "adapter.py", tmp_path, mode="v2", source_type="cuda", ir_path=ir)
    assert not list(tmp_path.glob("run-*/result.json"))
