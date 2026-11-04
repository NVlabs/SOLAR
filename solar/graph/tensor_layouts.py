# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout (contiguity / stride) records bound to the tensors torchview actually traced.

an earlier decision round 4 (an earlier decision review P2 input-3): a layout record must belong
to the tensor it describes.  Matching records to tensor nodes by kind and
shape let a contiguous, unused model input donate its record to a
same-shape noncontiguous nested input.  Records are therefore bound to the
*identity* torchview stores on each ``TensorNode`` (``tensor_id = id(...)``):

* model inputs: torchview turns every traced input (nested lists / tuples /
  dicts included) into a ``RecorderTensor`` through
  ``torchview.torchview.get_recorder_tensor``; :func:`bind_input_layouts`
  wraps that function for the duration of a capture and records the layout
  of the source tensor under ``id(recorder_tensor)``, which is the node's
  ``tensor_id``;
* parameters: the capture shim creates one ``parameter-tensor`` node per
  ``nn.Parameter`` object with ``tensor_id = id(param)``;
  :func:`parameter_layouts` maps the model's parameter objects the same way.

A node without a bound record gets no ``layout`` entry and the analyzer
treats its layout as unknown (charged as a copy).  Nothing is inferred.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import torch

_INPUT_BINDINGS: Dict[int, Dict[str, Any]] = {}
# id(traced tensor / parameter) -> storage identity and strides AT BINDING TIME
# (storage-5): the executed call is compared against this record.
_BOUND_STORAGE: Dict[int, Dict[str, Any]] = {}
# id(stand-in tensor) -> layout of the real tensor it stands for (written by the
# processor's input mover; a meta stand-in cannot always carry the layout itself)
_SOURCE_LAYOUTS: Dict[int, Dict[str, Any]] = {}


def note_source_layout(stand_in: torch.Tensor, source: torch.Tensor) -> None:
    try:
        _SOURCE_LAYOUTS[id(stand_in)] = layout_record(source, "recorded at capture (bound to the traced input tensor)")
    except Exception:
        pass


def storage_record(t: torch.Tensor) -> Dict[str, Any]:
    """Storage identity + geometry of ``t`` now.  ``storage_ptr`` is 0 on meta
    (no data), so storage identity is *unavailable* there and only the
    geometry can be compared.

    Read under ``DisableTorchFunctionSubclass``: ``t`` is usually torchview's
    ``RecorderTensor``, and ``untyped_storage()`` on a tensor subclass is a
    torch-function call.  Dispatched, torchview's recorder walked the returned
    storage element by element (``reduce_data_info`` over ``torch.storage``'s
    sequence protocol), which cost O(numel) Python per bound receiver and took
    the Reformer capture from 40 s to over 2400 s (regression found at
    an earlier revision, introduced with an earlier review round).  Metadata reads are not graph
    events and must never reach the recorder.
    """
    with torch._C.DisableTorchFunctionSubclass():
        try:
            storage = t.untyped_storage()
            ptr = int(storage.data_ptr())
            nbytes = int(storage.nbytes())
        except Exception:
            ptr, nbytes = 0, -1
        return {"storage_ptr": ptr, "storage_nbytes": nbytes, "storage_offset": int(t.storage_offset()),
                "shape": [int(d) for d in t.shape], "stride": [int(s) for s in t.stride()],
                "contiguous": bool(t.is_contiguous()), "device": t.device.type,
                # the recorded tensor itself (TARGET-ALIAS-FALLBACK-1): the maps are
                # keyed by id(), which CPython recycles once a tensor is freed — a
                # strong reference keeps the id alive while the record lives, and
                # readers verify `record["tensor"] is receiver` before trusting it
                "tensor": t}


def record_is_for(rec: Optional[Dict[str, Any]], t: torch.Tensor) -> bool:
    """A binding-time record describes ``t`` only if it was recorded FROM ``t`` (same object);
    a record found under ``id(t)`` that belongs to another (possibly freed) tensor is no record."""
    return bool(rec) and rec.get("tensor") is t


def reset_attempt_records() -> None:
    """Forget every binding-time record.  Called at the start of EACH capture attempt (the meta
    attempt and the module-backed CPU fallback): a failed attempt's traced tensors are freed and
    the fallback's fresh parameters may be allocated at the same addresses, so a record of the
    dead attempt must never be visible to the next one (TARGET-ALIAS-FALLBACK-1)."""
    _INPUT_BINDINGS.clear()
    _SOURCE_LAYOUTS.clear()
    _BOUND_STORAGE.clear()


def receiver_layout_validity(receiver: torch.Tensor) -> Dict[str, Any]:
    """Does the layout/storage evidence recorded for ``receiver`` still describe
    the tensor the call executed on?  Identity (same object) is not enough:
    ``x.data = other``, ``x.set_(other)``, ``as_strided_`` or ``resize_``
    keep ``id(x)`` and change the storage and/or the strides.  Returns the
    executed geometry, the comparison against the binding-time record and the
    verdict ``layout_valid`` (None when no binding-time record exists: a
    hidden tensor; its executed contiguity is still reported)."""
    now = storage_record(receiver)
    rec = _BOUND_STORAGE.get(id(receiver))
    out: Dict[str, Any] = {"at_call": {k: now[k] for k in ("shape", "stride", "contiguous")}}
    if rec is None:
        out["layout_valid"] = None
        out["storage_identity"] = "no binding-time record (hidden tensor)"
        return out
    if not record_is_for(rec, receiver):
        # a record under this id written for ANOTHER tensor (an id recycled from
        # a freed traced tensor of an earlier attempt): not evidence about the receiver
        out["layout_valid"] = None
        out["storage_identity"] = "no binding-time record (stale record of another tensor under a reused id; ignored)"
        return out
    geometry_same = rec["shape"] == now["shape"] and rec["stride"] == now["stride"] and rec["storage_offset"] == now["storage_offset"]
    if now["device"] == "meta" or rec["device"] == "meta" or rec["storage_ptr"] == 0 or now["storage_ptr"] == 0:
        storage = "unavailable (meta: no data pointer); geometry compared"
        storage_same = None
    else:
        storage_same = rec["storage_ptr"] == now["storage_ptr"] and rec["storage_nbytes"] == now["storage_nbytes"]
        storage = "unchanged" if storage_same else "changed"
    out["storage_identity"] = storage
    out["geometry_matches_record"] = geometry_same
    out["layout_valid"] = bool(geometry_same and storage_same is not False)
    return out


def layout_record(t: torch.Tensor, evidence: str) -> Dict[str, Any]:
    with torch._C.DisableTorchFunctionSubclass():      # metadata read, never a recorder event
        return {
            "contiguous": bool(t.is_contiguous()),
            "stride": [int(s) for s in t.stride()],
            "evidence": evidence,
        }


@contextlib.contextmanager
def bind_input_layouts():
    """Record the layout of every tensor torchview wraps as a traced input.

    Wraps ``torchview.torchview.get_recorder_tensor`` (looked up by name at
    call time by both torchview's ``process_input`` and the collect-attributes
    patch), keyed by the identity of the ``RecorderTensor`` it returns.  The
    registry is cleared on entry; it is read while the graph is written.
    """
    import torchview.torchview as tv_mod
    reset_attempt_records()
    orig = tv_mod.get_recorder_tensor

    def _recording(input_tensor, *args, **kwargs):
        rec = orig(input_tensor, *args, **kwargs)
        try:
            # prefer the real tensor's layout noted by the input mover; else the traced tensor's own
            _INPUT_BINDINGS[id(rec)] = dict(_SOURCE_LAYOUTS.get(id(input_tensor)) or layout_record(
                input_tensor, "recorded at capture (bound to the traced input tensor)"))
            _BOUND_STORAGE[id(rec)] = storage_record(rec)        # the traced tensor the model executes with
        except Exception:
            pass
        return rec

    tv_mod.get_recorder_tensor = _recording
    try:
        yield _INPUT_BINDINGS
    finally:
        tv_mod.get_recorder_tensor = orig
        _SOURCE_LAYOUTS.clear()


def parameter_layouts(model: Any) -> Dict[int, Dict[str, Any]]:
    """``id(param) -> layout`` for the model object that was traced.

    Only the traced object binds: the lineage's CPU fallback re-instantiates
    the model, whose parameters are different objects, so their nodes stay
    unbound (unknown layout, charged as a copy).
    """
    out: Dict[int, Dict[str, Any]] = {}
    if model is None:
        return out
    try:
        for _, p in model.named_parameters():
            out[id(p)] = layout_record(p, "recorded at capture (bound to the parameter object)")
            _BOUND_STORAGE[id(p)] = storage_record(p)
    except Exception:
        pass
    return out


def _tensor_leaves(obj: Any) -> list:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in _tensor_leaves(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _tensor_leaves(o)]
    return []


def bind_call_operands(fn_node: Any, args: Any, kwargs: Optional[dict]) -> Dict[str, Any]:
    """Identity record of a recorded call's tensor arguments against the graph operands
    torchview attached to ``fn_node`` (an earlier decision round 5, input-4).

    ``receiver_bound`` is True only when the first positional tensor argument
    (the method receiver / first operand) IS one of the FunctionNode's parent
    tensor nodes: a ``RecorderTensor`` whose own node is a parent, or an
    ``nn.Parameter`` whose ``id`` is a parent's ``tensor_id``.  A registered
    buffer or plain tensor attribute never becomes a graph operand, so it is
    reported unbound whatever its shape; a receiver passed under a keyword is
    not a receiver at all.  ``destination`` records an ``out=`` tensor (shape
    and whether it is a graph operand) without making it an operand.
    """
    from torchview.recorder_tensor import RecorderTensor
    parents = list(getattr(fn_node, "parents", None) or [])
    parent_ids = {id(p) for p in parents}
    parent_tensor_ids = {getattr(p, "tensor_id", None) for p in parents}

    def bound(t: torch.Tensor) -> bool:
        if isinstance(t, torch.nn.Parameter):
            return id(t) in parent_tensor_ids
        if isinstance(t, RecorderTensor):
            return any(id(n) in parent_ids for n in (getattr(t, "tensor_nodes", None) or []))
        return False

    kwargs = dict(kwargs or {})
    out = kwargs.pop("out", None)
    positional = _tensor_leaves(list(args))
    keyword = _tensor_leaves(kwargs)
    receiver = positional[0] if positional else None
    rec: Dict[str, Any] = {
        "tensor_args": len(positional) + len(keyword),
        "positional_tensor_args": len(positional),
        "graph_operands": sum(1 for t in positional + keyword if bound(t)),
        "receiver_bound": (bound(receiver) if receiver is not None else None),
    }
    if rec["receiver_bound"]:
        # storage-5: identity alone never authorizes reuse of the layout
        # record; the executed receiver must still match it.
        rec["receiver_layout"] = receiver_layout_validity(receiver)
    outs = _tensor_leaves(out)
    if outs:
        rec["destination"] = {"keyword": "out", "shape": [int(d) for d in outs[0].shape],
                              "graph_operand": bool(bound(outs[0]))}
    return rec


def layout_for_node(node: Any, node_type: str, param_layouts: Optional[Dict[int, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The bound layout record for a torchview TensorNode, or None."""
    tid = getattr(node, "tensor_id", None)
    if tid is None:
        return None
    nt = str(node_type or "").lower()
    if nt in ("input-tensor", "auxiliary-tensor"):
        rec = _INPUT_BINDINGS.get(tid)
        # trusted only while the storage record under the same id is still the
        # recorded tensor's (its id not recycled by a later tensor)
        bound = _BOUND_STORAGE.get(tid)
        if rec and bound is not None and bound.get("tensor") is not None and id(bound["tensor"]) != tid:
            return None
    elif nt == "parameter-tensor":
        rec = (param_layouts or {}).get(tid)
    else:
        return None
    return dict(rec) if rec else None
