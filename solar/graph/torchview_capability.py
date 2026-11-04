# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-wide torchview capability gate (issue #45).

The mechanical frontend needs two behaviours from the installed torchview that
the released 0.2.7 wheel does not provide on its own:

* **parameter tensors**: a bare ``nn.Parameter`` passed to a torch function
  (``F.linear(x, weight, bias)``, ``F.conv2d(img, weight, ...)``, ``x @ W``)
  appears as a ``parameter-tensor`` graph node, so weights are billed and
  their traffic counted (``patches/torchview-parameter-tensors.patch``);
* **call attributes**: ``draw_graph(collect_attributes=True)`` propagates the
  flag to the model-input and hidden recorder tensors so every FunctionNode
  carries its stringified arguments (upstream #128,
  ``patches/torchview-collect-attributes-inputs.patch``); stride/padding/
  dilation/groups, transpose dims, softmax dim, einsum equations and the
  norm-parameter attachment are read from them.

Without them the capture is *incapable*: weights vanish, conv MACs become 0,
addmm MACs collapse, and every stage still exits 0.  This module probes both
behaviours in-process on a tiny model (no files, no model download) and
produces a machine-readable record that the processor, converter, analyzer
and perf model carry in their outputs.  The processor fails closed when a
capability is missing unless ``allow_degraded_capture`` is set, in which
case every layer is marked not supportable and every stage reports
``status: incomplete``.

Two identities are recorded separately (issue #45 review P3-1):

* the **installed dependency**: ``torchview_version``, ``torchview_file`` and
  ``source`` (``patched`` when the installed recorder source itself emits
  parameter-tensor nodes; ``released`` for a plain distribution;
  ``released+shim`` when a plain distribution is made capable by the engine's
  import-time parameter recorder, ``solar.graph._torchview_param_shim``;
  ``unknown`` when the source cannot be read);
* the **effective behaviour**: what a capture sees.  The engine does supply
  runtime behaviour on top of the installed wheel — the lineage's recorder
  shim is installed when ``solar.graph.pytorch_processor`` is imported and
  attribute propagation is wrapped (``_patch_torchview_collect_attributes``)
  around every probe and capture — so the probe runs under the processor's
  draw context and reports what that context delivers.  The install contract
  (a versioned, patched torchview) and this behavioural preflight are distinct:
  passing the preflight does not certify the install, and the record names
  which of the two made the capture capable.
"""
from __future__ import annotations

import dataclasses
import datetime
import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchview


class TorchviewCapabilityError(RuntimeError):
    """The installed torchview lacks a capability the mechanical frontend needs.

    Raised at ``PyTorchProcessor`` construction (fail closed).  ``record`` is
    the machine-readable capability record.
    """

    def __init__(self, record: "CaptureCapability"):
        self.record = record.to_dict()
        super().__init__(
            "TorchviewCapabilityError: the installed torchview cannot support "
            f"mechanical capture ({record.reason}); torchview {record.torchview_version} "
            f"from {record.torchview_file} [{record.source}]. Install the patched torchview "
            "(install.sh / patches/torchview-parameter-tensors.patch + "
            "patches/torchview-collect-attributes-inputs.patch) or pass "
            "--allow-degraded-capture to proceed with every layer marked not supportable.")


# Default draw context of every public entry point: the engine's canonical capture wrapper
# (issue #45 review P2-2). ``None`` is the explicit bare probe.
ENGINE_CONTEXT = type("_EngineContext", (), {"__repr__": lambda self: "ENGINE_CONTEXT"})()

@dataclasses.dataclass
class CaptureCapability:
    parameter_tensors: bool
    call_attributes: bool
    torchview_version: str
    torchview_file: str
    source: str                      # 'patched' | 'released+shim' | 'released' | 'unknown'
    probe: Dict[str, Any] = dataclasses.field(default_factory=dict)
    checked_utc: str = ""
    error: Optional[str] = None
    # processor-level mitigation flag (torch._dynamo imported before capture); serialized with the record
    dynamo_warmup: Optional[bool] = None

    @property
    def capable(self) -> bool:
        return bool(self.parameter_tensors and self.call_attributes)

    @property
    def degraded(self) -> bool:
        return not self.capable

    @property
    def reason(self) -> str:
        if self.capable:
            return "capable"
        missing = [n for n, ok in (("parameter_tensors", self.parameter_tensors), ("call_attributes", self.call_attributes)) if not ok]
        base = "missing capability: " + ", ".join(missing)
        return base + (f"; probe error: {self.error}" if self.error else "")

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update({"capable": self.capable, "degraded": self.degraded, "reason": self.reason})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptureCapability":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


class _ProbeModel(nn.Module):
    """nn.Linear (module parameters), a functional conv with kwargs, and a
    functional producer->consumer chain with non-default arguments
    (``softmax(h + 1.0, dim=-1)``, ``relu(c * 2.0)``) so the probe exercises
    attribute propagation through hidden tensors, not only on model inputs
    (issue #45 review P2-4)."""

    CHAIN_CONSUMERS = ("softmax", "relu")      # downstream FunctionNodes that must carry attributes

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.k = nn.Parameter(torch.randn(2, 1, 3, 3))

    def forward(self, x: torch.Tensor, img: torch.Tensor):  # type: ignore[override]
        h = self.fc(x)
        y = F.softmax(h + 1.0, dim=-1)                       # hidden tensor -> functional consumer
        c = F.conv2d(img, self.k, stride=2, padding=1, groups=2)
        return y, F.relu(c * 2.0)


def _torchview_source() -> str:
    """'patched' when the installed recorder source emits parameter-tensor nodes, 'released+shim' when a plain
    distribution carries the engine's in-memory parameter recorder (``_solar_param_capture`` sentinel),
    'released' for a plain distribution, 'unknown' when the source cannot be read."""
    try:
        from torchview import recorder_tensor
        text = inspect.getsource(recorder_tensor)
        if "parameter-tensor" in text:
            return "patched"
        tf = recorder_tensor.RecorderTensor.__dict__.get("__torch_function__")
        fn = getattr(tf, "__func__", tf)
        return "released+shim" if getattr(fn, "_solar_param_capture", False) else "released"
    except Exception:
        return "unknown"


def probe_torchview_capability(force: bool = False, draw_context=ENGINE_CONTEXT) -> CaptureCapability:
    """Probe the installed torchview once per process and draw context (``force`` re-probes).

    ``draw_context`` defaults to :data:`ENGINE_CONTEXT`, the one canonical
    context every engine entry point uses (the processor's
    ``_patch_torchview_collect_attributes`` wrapper), so the public record and
    the processor's record are the same cache entry. ``draw_context=None`` is
    the explicit *bare* probe (no engine wrapper): on a released wheel it
    legitimately sees fewer attributes and is cached under its own key.

    The probe traces :class:`_ProbeModel` on CPU with the same ``draw_graph``
    options the processor uses (inside ``draw_context()`` when the engine
    wraps ``draw_graph``, e.g. its collect_attributes propagation patch, so
    the probe measures what a real capture would see), then inspects the
    flattened edge list:
    parameter tensors are present when ``TensorNode`` parents named
    ``parameter-tensor`` feed the function nodes; call attributes propagate
    when at least one FunctionNode carries a non-empty ``attributes`` string.
    """
    global _CACHED
    if not isinstance(_CACHED, dict):   # tolerate an external reset such as ``_CACHED = None`` (pre-P2-2 harnesses)
        _CACHED = {}
    draw_context = _resolve_context(draw_context)
    key = _context_key(draw_context)
    if key in _CACHED and not force:
        return _CACHED[key]
    version = str(getattr(torchview, "__version__", "unknown"))
    tv_file = str(getattr(torchview, "__file__", "unknown"))
    source = _torchview_source()
    draw = getattr(torchview, "draw_graph", None)
    if getattr(draw, "__module__", None) != "torchview.torchview":
        # A test double or wrapper replaced draw_graph in this process: keep the
        # previous record for this context if any, never fabricate one.
        if key in _CACHED:
            return _CACHED[key]
        rec = CaptureCapability(False, False, version, tv_file, source, {"ops_total": 0},
                                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                error="torchview.draw_graph is replaced in this process; capability not probed")
        return rec  # not cached: a later genuine probe may succeed
    param_nodes = 0
    ops_total = 0
    ops_with_attrs = 0
    chain_seen: Dict[str, bool] = {}
    error: Optional[str] = None
    try:
        model = _ProbeModel().eval()
        import contextlib
        ctx = draw_context() if draw_context is not None else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            graph = torchview.draw_graph(model, input_data=[torch.randn(2, 4), torch.randn(1, 2, 6, 6)], device="cpu",
                                         save_graph=False, expand_nested=True, depth=float("inf"), hide_module_functions=False,
                                         hide_inner_tensors=False, roll=False, strict=False, collect_attributes=True)
        seen_params = set()
        seen_ops = set()
        for tail, head in getattr(graph, "edge_list", []) or []:
            for node in (tail, head):
                cls = type(node).__name__
                if cls == "TensorNode" and str(getattr(node, "name", "")).lower() == "parameter-tensor":
                    seen_params.add(id(node))
                elif cls == "FunctionNode" and id(node) not in seen_ops:
                    seen_ops.add(id(node))
                    ops_total += 1
                    name = str(getattr(node, "name", "")).lower()
                    has = bool(getattr(node, "attributes", None))
                    if has:
                        ops_with_attrs += 1
                    if name in _ProbeModel.CHAIN_CONSUMERS:
                        chain_seen[name] = has
        param_nodes = len(seen_params)
    except Exception as exc:  # the probe itself failing is an incapable capture
        error = f"{type(exc).__name__}: {exc}"[:300]
    chain_ok = all(chain_seen.get(n) for n in _ProbeModel.CHAIN_CONSUMERS)
    rec = CaptureCapability(
        parameter_tensors=param_nodes >= 1,
        # every FunctionNode carries attributes AND the hidden-tensor consumers do (propagation through hidden tensors)
        call_attributes=ops_total > 0 and ops_with_attrs == ops_total and chain_ok,
        torchview_version=version, torchview_file=tv_file, source=source,
        probe={"parameter_tensor_nodes": param_nodes, "ops_with_attributes": ops_with_attrs, "ops_total": ops_total,
               "chain_consumers_with_attributes": {n: bool(chain_seen.get(n)) for n in _ProbeModel.CHAIN_CONSUMERS},
               "draw_context": key,
               "model": "nn.Linear(4,3) -> softmax(h + 1.0, dim=-1); F.conv2d(stride=2, padding=1, groups=2) -> relu(c * 2.0) on CPU"},
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), error=error)
    _CACHED[key] = rec
    return rec


# Records are cached PER DRAW CONTEXT (issue #45 review P2-2): a contextless
# public probe on a released wheel legitimately sees fewer attributes than the
# processor's wrapped capture and must never be reused for it (or vice versa).
_CACHED: Dict[str, CaptureCapability] = {}


def _resolve_context(draw_context):
    """``ENGINE_CONTEXT`` -> the processor's canonical draw wrapper (resolved lazily: the processor imports this module)."""
    if draw_context is ENGINE_CONTEXT:
        from solar.graph.pytorch_processor import _patch_torchview_collect_attributes
        return _patch_torchview_collect_attributes
    return draw_context


def _context_key(draw_context) -> str:
    if draw_context is None:
        return "none"
    return f"{getattr(draw_context, '__module__', '')}.{getattr(draw_context, '__qualname__', repr(draw_context))}"


def engine_context_key() -> str:
    """The cache key of the canonical engine draw context."""
    return _context_key(_resolve_context(ENGINE_CONTEXT))


def get_torchview_capability(draw_context=ENGINE_CONTEXT) -> CaptureCapability:
    """The capability record for ``draw_context`` (probed on first use per context; engine context by default)."""
    return probe_torchview_capability(draw_context=draw_context)


def reset_capability_cache() -> None:
    """Forget every cached record (tests / explicit re-probe)."""
    _CACHED.clear()


def degraded_marker(record: Dict[str, Any]) -> Optional[str]:
    """The reason string when ``record`` describes a degraded capture, else None."""
    if not record or not record.get("degraded"):
        return None
    return f"degraded capture: {record.get('reason', 'capability missing')}"


__all__ = ["CaptureCapability", "TorchviewCapabilityError", "ENGINE_CONTEXT", "probe_torchview_capability",
           "get_torchview_capability", "engine_context_key", "reset_capability_cache", "degraded_marker"]
