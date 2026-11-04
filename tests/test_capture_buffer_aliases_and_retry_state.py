# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture restores EVERY registered buffer name and a valid pre-attempt state (an earlier review, issue #29).

Independent review of the first repair found two holes:

* alias-1: ``named_buffers()`` deduplicates, so a second registered name
  for the same tensor (same module, or shared across children) was never
  snapshotted; after ``to_empty`` separated the slots the alias kept
  uninitialized memory.
* state-1: a failed meta attempt that resized a buffer left the CPU
  retry with a mismatched slot that the shape guard silently skipped.

Retry boundary: every capture attempt starts from the pre-capture buffer
state (values, shapes, alias identities, registered names); whatever a
FAILED attempt mutated is discarded; a SUCCESSFUL capture keeps its
mutations.  A buffer that disappears during a failed attempt cannot be
restored and the retry is refused.

Round 4 (meta-alias-1): the FIRST attempt is a capture attempt too.
``to_empty(device="meta")`` allocates one fresh meta tensor per registered
slot, so two names that referred to one tensor stop being aliases before the
meta draw; a model whose program depends on that identity (``a is b``) would
be captured on the wrong branch.  The capture loop therefore restores the
snapshot (shapes and alias identities on meta) before the meta draw as well
as before the module-less CPU retry; the module-backed CPU fallback
re-instantiates the model from its source module instead.
"""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from solar.common.types import ProcessingConfig
from solar.graph.pytorch_processor import PyTorchProcessor as P


def _poison(model):
    for b in model.buffers():
        b.fill_(-999)


def test_same_module_alias_is_restored_and_rebound():
    m = nn.Module()
    v = torch.tensor([5, 6, 7])
    m.register_buffer("primary", v)
    m.register_buffer("alias", v, persistent=False)
    saved = P._snapshot_buffers(m)
    assert set(saved) == {"primary", "alias"}
    m.to_empty(device="meta")
    m.to_empty(device="cpu")
    assert m.primary is not m.alias
    _poison(m)
    P._restore_buffers(m, saved)
    assert torch.equal(m.primary, torch.tensor([5, 6, 7]))
    assert torch.equal(m.alias, torch.tensor([5, 6, 7]))
    assert m.alias is m.primary                       # identity, not a copy
    assert "alias" in m._non_persistent_buffers_set  # persistence flags untouched
    assert "primary" not in m._non_persistent_buffers_set


def test_buffer_shared_across_children_is_restored_and_rebound():
    m = nn.Module()
    m.a = nn.Module()
    m.b = nn.Module()
    v = torch.tensor([2, 4])
    m.a.register_buffer("state", v)
    m.b.register_buffer("state", v)
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    m.to_empty(device="cpu")
    _poison(m)
    P._restore_buffers(m, saved)
    assert torch.equal(m.a.state, torch.tensor([2, 4]))
    assert torch.equal(m.b.state, torch.tensor([2, 4]))
    assert m.a.state is m.b.state


def test_resized_buffer_is_restored_to_saved_shape_and_values():
    m = nn.Module()
    m.register_buffer("cache", torch.tensor([2, 4]))
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    m.cache.resize_(3)                                 # a failed attempt grew the cache
    m.to_empty(device="cpu")
    _poison(m)
    P._restore_buffers(m, saved)
    assert list(m.cache.shape) == [2]
    assert torch.equal(m.cache, torch.tensor([2, 4]))
    assert m.cache.device.type == "cpu"


def test_failed_meta_attempt_that_resizes_never_reaches_cpu_capture_with_mutated_state():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.tensor([2, 4]))

        def to_empty(self, *, device, recurse=True):
            super().to_empty(device=device, recurse=recurse)
            if str(device) == "cpu":
                self.cache.fill_(-999)
            return self

    m = M()
    proc = P(ProcessingConfig(debug=False))
    seen = []

    def draw(model, input_data, device, **kwargs):
        if device == "meta":
            model.cache.resize_(3)
            raise RuntimeError("failed meta attempt resized the cache")
        seen.append({"shape": list(model.cache.shape), "values": model.cache.tolist()})
        return object()

    with patch("solar.graph.pytorch_processor.torchview.draw_graph", side_effect=draw):
        graph = proc._generate_torchview_graph(m, torch.tensor([1.0]))
    assert graph is not None
    assert seen == [{"shape": [2], "values": [2, 4]}]


def test_buffer_registered_during_failed_attempt_is_dropped_before_retry():
    m = nn.Module()
    m.register_buffer("state", torch.tensor([1, 2]))
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    m.register_buffer("lazy_cache", torch.empty(3, device="meta"))  # created by the failed attempt
    m.to_empty(device="cpu")
    P._restore_buffers(m, saved)
    assert dict(m.named_buffers()).keys() == {"state"}
    assert torch.equal(m.state, torch.tensor([1, 2]))


def test_buffer_that_disappeared_during_failed_attempt_refuses_retry():
    m = nn.Module()
    m.register_buffer("state", torch.tensor([1, 2]))
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    del m._buffers["state"]
    m.to_empty(device="cpu")
    with pytest.raises(RuntimeError, match="state"):
        P._restore_buffers(m, saved)


def test_non_persistent_index_buffers_round_trip():
    # transformers 5.6.2 registers BigBird's position_ids / token_type_ids with persistent=False.
    m = nn.Module()
    m.register_buffer("position_ids", torch.arange(6).unsqueeze(0), persistent=False)
    m.register_buffer("token_type_ids", torch.zeros(1, 6, dtype=torch.long), persistent=False)
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    m.to_empty(device="cpu")
    _poison(m)
    P._restore_buffers(m, saved)
    assert torch.equal(m.position_ids, torch.arange(6).unsqueeze(0))
    assert int(m.token_type_ids.max()) == 0
    assert {"position_ids", "token_type_ids"} <= m._non_persistent_buffers_set


# --- review (round 2): the registry is complete, not just the copyable values -------------


def test_pre_existing_meta_buffer_keeps_its_registration_and_flag():
    # Meta tensors carry no values (not snapshotted), but their REGISTRATION is original state.
    m = nn.Module()
    m.register_buffer("real", torch.tensor([2, 4]))
    m.register_buffer("metadata", torch.empty(3, device="meta"), persistent=False)
    saved = P._snapshot_buffers(m)
    assert set(saved) == {"real"}                       # copyable values only
    assert set(saved.registry) == {"real", "metadata"}  # complete registry
    assert saved.registry["metadata"]["persistent"] is False
    m.to_empty(device="meta")
    P._restore_buffers(m, saved)
    assert "metadata" in m._buffers and tuple(m.metadata.shape) == (3,)
    assert "metadata" in m._non_persistent_buffers_set
    assert "real" in m._buffers and tuple(m.real.shape) == (2,)


def test_mixed_cpu_meta_model_reaches_capture_with_original_registry():
    m = nn.Module()
    m.register_buffer("real", torch.tensor([2, 4]))
    m.register_buffer("metadata", torch.empty(3, device="meta"), persistent=False)
    marker = object()
    seen = []

    def draw(model, input_data, device, **kwargs):
        seen.append((device, tuple(model.metadata.shape), sorted(dict(model.named_buffers()))))
        return marker

    proc = P(ProcessingConfig(debug=False))  # the lineage probes torchview at construction: build before the double
    with patch("solar.graph.pytorch_processor.torchview.draw_graph", side_effect=draw):
        result = proc._generate_torchview_graph(m, torch.tensor([1.0]))
    assert result is marker
    assert seen == [("meta", (3,), ["metadata", "real"])]


def test_empty_original_registry_still_drops_buffers_created_by_a_failed_attempt():
    class M(nn.Module):
        def to_empty(self, *, device, recurse=True):
            super().to_empty(device=device, recurse=recurse)
            if str(device) == "cpu":
                for b in self.buffers():
                    b.fill_(-999)
            return self

    m = M()
    seen = []
    marker = object()

    def draw(model, input_data, device, **kwargs):
        if device == "meta":
            model.register_buffer("lazy_cache", torch.empty(3, device="meta"))
            raise RuntimeError("failed meta capture created a lazy buffer")
        seen.append({k: v.tolist() for k, v in model.named_buffers()})
        return marker

    proc = P(ProcessingConfig(debug=False))  # the lineage probes torchview at construction: build before the double
    with patch("solar.graph.pytorch_processor.torchview.draw_graph", side_effect=draw):
        result = proc._generate_torchview_graph(m, torch.tensor([1.0]))
    assert result is marker
    assert seen == [{}]            # the poisoned lazy_cache never reaches the CPU capture


def test_none_slot_registered_during_failed_attempt_is_dropped():
    m = nn.Module()
    m.register_buffer("state", torch.tensor([1, 2]))
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    m.register_buffer("lazy_none", None, persistent=False)   # named_buffers() does not list None slots
    m.to_empty(device="cpu")
    P._restore_buffers(m, saved)
    assert "lazy_none" not in m._buffers
    assert "lazy_none" not in m._non_persistent_buffers_set
    assert torch.equal(m.state, torch.tensor([1, 2]))


def test_original_none_slot_is_kept_and_reset_to_none():
    m = nn.Module()
    m.register_buffer("optional", None)
    m.register_buffer("state", torch.tensor([1, 2]))
    saved = P._snapshot_buffers(m)
    assert set(saved) == {"state"} and set(saved.registry) == {"optional", "state"}
    m.to_empty(device="meta")
    m.optional = torch.empty(2, device="meta")   # a failed attempt filled the optional slot
    m.to_empty(device="cpu")
    P._restore_buffers(m, saved)
    assert "optional" in m._buffers and m._buffers["optional"] is None
    assert torch.equal(m.state, torch.tensor([1, 2]))


def test_meta_original_buffer_that_disappeared_refuses_retry():
    m = nn.Module()
    m.register_buffer("metadata", torch.empty(3, device="meta"), persistent=False)
    saved = P._snapshot_buffers(m)
    assert not saved and set(saved.registry) == {"metadata"}   # no values, but a registry
    m.to_empty(device="meta")
    del m._buffers["metadata"]
    m._non_persistent_buffers_set.discard("metadata")
    with pytest.raises(RuntimeError, match="disappeared"):
        P._restore_buffers(m, saved)


def test_non_module_stand_in_is_an_unsupported_registry_not_an_empty_one():
    saved = P._snapshot_buffers(object())
    assert saved.supported is False
    P._restore_buffers(object(), saved)     # no-op, never raises


def test_valid_empty_registry_is_supported():
    saved = P._snapshot_buffers(nn.Module())
    assert saved.supported is True and not saved and not saved.registry


# --- review (round 3): an originally VALUED slot that a failed attempt set to None -------------


@pytest.mark.parametrize("shared", [False, True], ids=["plain", "with-alias"])
def test_valued_slot_set_to_none_by_failed_attempt_is_restored_on_the_cpu_attempt(shared):
    m = nn.Module()
    state = torch.tensor([2, 4], dtype=torch.int64)
    m.register_buffer("state", state, persistent=False)
    if shared:
        m.register_buffer("alias", state)
    marker = object()
    seen = []

    def draw(model, input_data, device, **kwargs):
        if device == "meta":
            model.state = None            # the failed attempt clobbers the slot
            raise RuntimeError("failed meta capture replaced an original valued buffer with None")
        seen.append((device, model.state.device.type, tuple(model.state.shape), model.state.tolist(),
                     model.state.dtype, (not shared) or (model.state is model.alias)))
        return marker

    proc = P(ProcessingConfig(debug=False))  # the lineage probes torchview at construction: build before the double
    with patch("solar.graph.pytorch_processor.torchview.draw_graph", side_effect=draw):
        graph = proc._generate_torchview_graph(m, torch.tensor([1.0]))
    assert graph is marker
    # the CPU attempt sees the ORIGINAL values on the CPU, with the alias identity restored
    assert seen == [("cpu", "cpu", (2,), [2, 4], torch.int64, True)]
    assert "state" in m._non_persistent_buffers_set


def test_restore_rebuilds_a_none_valued_slot_on_the_requested_device():
    m = nn.Module()
    m.register_buffer("state", torch.tensor([2, 4], dtype=torch.int64))
    saved = P._snapshot_buffers(m)
    m.to_empty(device="meta")
    m.state = None
    m.to_empty(device="cpu")                      # to_empty skips the None slot
    P._restore_buffers(m, saved, device="cpu")
    assert m.state.device.type == "cpu" and torch.equal(m.state, torch.tensor([2, 4]))
    # the meta attempt only needs the shape
    m.state = None
    P._restore_buffers(m, saved, device="meta")
    assert m.state.device.type == "meta" and tuple(m.state.shape) == (2,) and m.state.dtype == torch.int64


def test_restore_without_device_restores_values_where_the_saved_copy_lives():
    m = nn.Module()
    m.register_buffer("state", torch.tensor([2, 4]))
    saved = P._snapshot_buffers(m)
    m.state = None
    P._restore_buffers(m, saved)                  # no device given: a valued slot gets its values back
    assert m.state.device.type == "cpu" and torch.equal(m.state, torch.tensor([2, 4]))


# ---- round 4: aliases survive into a SUCCESSFUL meta capture (meta-alias-1) --------------------

def _alias_branch_model_class(across_children):
    """Two registered names for one int64 [2] buffer; the program depends on their identity."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            v = torch.tensor([2, 4], dtype=torch.int64)
            if across_children:
                self.a = nn.Module(); self.b = nn.Module()
                self.a.register_buffer("state", v)
                self.b.register_buffer("state", v, persistent=False)
            else:
                self.register_buffer("primary", v)
                self.register_buffer("alias", v, persistent=False)

        def alias_pair(self):
            return (self.a.state, self.b.state) if across_children else (self.primary, self.alias)

        def forward(self, x):
            a, b = self.alias_pair()
            return x.reshape(1, 2) if a is b else x.reshape(2, 1)
    return M


@pytest.mark.parametrize("across_children", [False, True], ids=["same_module", "children"])
@pytest.mark.parametrize("source_module", [False, True], ids=["same_object", "module_available"])
@pytest.mark.parametrize("fail_meta", [False, True], ids=["meta_success", "cpu_retry"])
def test_every_capture_attempt_sees_the_original_alias_identity(across_children, source_module, fail_meta):
    from types import SimpleNamespace
    M = _alias_branch_model_class(across_children)
    m = M()
    proc = P(ProcessingConfig(debug=False))
    seen = []
    marker = object()

    def draw(model, input_data, device, **kwargs):
        if fail_meta and device == "meta":
            raise RuntimeError("deliberate meta failure")
        a, b = model.alias_pair()
        seen.append({"device": device, "alias": a is b, "shape": tuple(model(input_data).shape)})
        return marker

    kw = {}
    if source_module:
        kw["module"] = SimpleNamespace(Model=M, get_init_inputs=lambda: [], get_inputs=lambda: torch.ones(2))
    with patch("solar.graph.pytorch_processor.torchview.draw_graph", side_effect=draw):
        assert proc._generate_torchview_graph(m, torch.ones(2), **kw) is marker
    assert seen == [{"device": "cpu" if fail_meta else "meta", "alias": True, "shape": (1, 2)}], seen


@pytest.mark.parametrize("across_children", [False, True], ids=["same_module", "children"])
def test_real_meta_capture_records_the_original_alias_branch(across_children):
    """Real torchview on meta, no draw double: the captured program is the alias branch (reshape to (1, 2))."""
    Base = _alias_branch_model_class(across_children)
    seen = []

    class M(Base):
        def forward(self, x):
            a, b = self.alias_pair()
            y = super().forward(x)
            seen.append({"device": str(x.device), "alias": a is b, "output_shape": tuple(y.shape)})
            return y

    proc = P(ProcessingConfig(debug=False))
    graph = proc._generate_torchview_graph(M(), torch.ones(2))
    assert graph is not None and seen
    assert all(v == {"device": "meta", "alias": True, "output_shape": (1, 2)} for v in seen), seen
    # the recorded graph carries the alias branch's output shape, not the other branch's
    shapes = {tuple(getattr(n, "tensor_shape", ())) for _, n in graph.edge_list if type(n).__name__ == "TensorNode"}
    assert (1, 2) in shapes and (2, 1) not in shapes, shapes


# ---- round 5 (meta-alias-2 / cache-1) ---------------------------------------------------------

@pytest.mark.parametrize("across_children", [False, True], ids=["same_module", "children"])
def test_initially_meta_alias_is_recorded_and_rebound(across_children):
    """Two names for one tensor that is ALREADY on meta: no values to copy, but the alias identity is state."""
    with torch.device("meta"):
        m = _alias_branch_model_class(across_children)()
    a, b = m.alias_pair()
    assert a is b and a.device.type == "meta"
    saved = P._snapshot_buffers(m)
    assert saved.alias_of and not list(saved)            # alias recorded, nothing copyable
    m.to_empty(device="meta")
    a, b = m.alias_pair()
    assert a is not b                                     # to_empty separated them
    P._restore_buffers(m, saved, device="meta")
    a, b = m.alias_pair()
    assert a is b and a.device.type == "meta" and tuple(a.shape) == (2,)


@pytest.mark.parametrize("across_children", [False, True], ids=["same_module", "children"])
def test_real_meta_capture_of_an_initially_meta_alias_records_the_alias_branch(across_children):
    Base = _alias_branch_model_class(across_children)
    seen = []

    class M(Base):
        def forward(self, x):
            a, b = self.alias_pair()
            y = super().forward(x)
            seen.append({"device": str(x.device), "alias": a is b, "output_shape": tuple(y.shape)})
            return y

    with torch.device("meta"):
        m = M()
    graph = P(ProcessingConfig(debug=False))._generate_torchview_graph(m, torch.ones(2))
    assert graph is not None and seen
    assert all(v == {"device": "meta", "alias": True, "output_shape": (1, 2)} for v in seen), seen
    shapes = {tuple(getattr(n, "tensor_shape", ())) for _, n in graph.edge_list if type(n).__name__ == "TensorNode"}
    assert (1, 2) in shapes and (2, 1) not in shapes, shapes


_ALIAS_SOURCE = """import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        v = torch.tensor([2, 4], dtype=torch.int64{device})
        {init}
    def forward(self, x):
        a, b = {pair}
        return x.reshape(1, 2) if a is b else x.reshape(2, 1)
def get_inputs(): return [torch.ones(2)]
def get_init_inputs(): return []
"""


def _alias_source(across_children, device=""):
    init = ("self.a = nn.Module(); self.b = nn.Module(); self.a.register_buffer('state', v); self.b.register_buffer('state', v, persistent=False)"
            if across_children else "self.register_buffer('primary', v); self.register_buffer('alias', v, persistent=False)")
    pair = "self.a.state, self.b.state" if across_children else "self.primary, self.alias"
    return _ALIAS_SOURCE.format(device=device, init=init, pair=pair)


def _reshape_shapes(graph_yaml):
    import yaml
    doc = yaml.safe_load(graph_yaml.read_text())
    return [s for n in doc["layers"].values() if n.get("type") == "reshape" for s in n.get("output_shapes", [])]


@pytest.mark.parametrize("across_children", [False, True], ids=["same_module", "children"])
def test_source_route_with_a_meta_constructed_alias_records_the_alias_branch(tmp_path, across_children):
    src = tmp_path / "model.py"
    src.write_text(_alias_source(across_children, device=", device='meta'"))
    proc = P(ProcessingConfig(debug=False, save_graph=False, force_rerun=True))
    assert proc.process_model_file(str(src), str(tmp_path / "graph"))
    assert _reshape_shapes(tmp_path / "graph" / "pytorch_graph.yaml") == [[1, 2]]


def test_capture_identity_covers_the_restore_implementation_and_rejects_prior_semantics(tmp_path):
    """cache-1: the capture revision is the earlier six-file identity (processor incl. snapshot/restore, node
    writer, capability gate, shim, installed torchview recorder/driver); an artifact whose provenance carries the
    pre-an earlier change shim-only revision, or a revision of another processor, is not reused on ordinary resume."""
    import hashlib, yaml
    from solar.graph import _torchview_param_shim as shim
    files = [Path(f).name for f in P._capture_implementation_files()]
    assert {"pytorch_processor.py", "torchview_processor.py", "torchview_capability.py", "_torchview_param_shim.py", "recorder_tensor.py", "torchview.py"} <= set(files)
    src = tmp_path / "model.py"
    src.write_text(_alias_source(True, device=", device='cpu'"))
    proc = P(ProcessingConfig(debug=False, save_graph=False, force_rerun=True))
    assert proc.process_model_file(str(src), str(tmp_path / "graph"))
    g = tmp_path / "graph" / "pytorch_graph.yaml"
    assert _reshape_shapes(g) == [[1, 2]]
    resume = P(ProcessingConfig(debug=False, save_graph=False, force_rerun=False))
    assert resume._is_already_processed(tmp_path / "graph", str(src)) is True            # same identity: reused
    doc = yaml.safe_load(g.read_text())
    shim_only = hashlib.sha256(Path(shim.__file__).read_bytes()).hexdigest()         # the pre-an earlier change scheme (target an earlier revision)
    doc["provenance"]["capture_rev"] = shim_only
    g.write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))
    assert resume._is_already_processed(tmp_path / "graph", str(src)) is False
    doc["provenance"]["capture_rev"] = "0" * 64                                         # another processor's identity
    g.write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))
    assert resume._is_already_processed(tmp_path / "graph", str(src)) is False
    assert resume.process_model_file(str(src), str(tmp_path / "graph"))                 # re-captured, not skipped
    assert _reshape_shapes(g) == [[1, 2]]
