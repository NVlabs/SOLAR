# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime parameter-operand capture for vanilla torchview.

The mechanical frontend's learned-parameter support (``parameter-tensor``
nodes, ``ordered_input_nodes``, weight billing) was designed against
``patches/torchview-parameter-tensors.patch``.  When the installed torchview
is unpatched (e.g. vanilla 0.2.7), a bare ``nn.Parameter`` passed directly to
a torch function is silently dropped from the trace: the function node loses
that operand, its elements vanish from memory accounting, and any dims
carried only by the parameter vanish from the MAC count (regression:
``tests/test_learned_param_matmul.py``).

This module applies the same behavior in-process, so a clean checkout works
against an unpatched site-packages torchview without mutating the shared
environment.  If the installed torchview already carries the patch
(detected by ``ComputationGraph._render_parameter_nodes``), it is a no-op.

Deviations from the on-disk patch (independent review 2026-10-21, both P1):

* Keyword-spelled tensor arguments are bound to the function's SEMANTIC
  parameter order (``inspect.signature`` / ``torch.overrides``), never to
  dict insertion order — ``matmul(other=x, input=w)`` is the same operation
  as ``matmul(w, x)``.  When kwargs hold tensors and no signature can bind
  them (absent signature, bind failure, or tensors in a ``**kwargs``
  catch-all), the call FAILS CLOSED by raising
  ``UnsupportedFrontendOperandError``: mechanical extraction stops with a
  typed diagnostic instead of emitting trusted artifacts with incomplete
  accounting (round-2 review: the earlier vanilla fallback let all three
  stages exit 0 with the Parameter silently dropped).  The same diagnostic
  fires when a bare ``nn.Parameter`` is present but could not be captured
  as an operand.  Vanilla collection remains ONLY for calls where binding
  succeeded, no operand node could be resolved, and no Parameter is
  involved — connectivity-preserving, with no operand order claimed.
* Zero-dimensional activation operands are retained (vanilla torchview keeps
  ``sum -> mul`` for ``s = x.sum(); s * s``); repeated occurrences of one
  operand keep their slots in the ordered operand list and input shapes,
  while graph parents stay deduplicated.

Known limitation (same as the patch): only tensors that are *directly*
``nn.Parameter`` instances are captured; a derived view such as
``self.weight.t()`` is a plain untraced Tensor and is still dropped.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_SIG_CACHE: dict = {}
_TESTING_OVERRIDES = None


class UnsupportedFrontendOperandError(RuntimeError):
    """Typed fail-closed diagnostic: a call's tensor operands cannot be
    semantically ordered (or a bare nn.Parameter operand cannot be captured),
    so mechanical extraction must not produce trusted accounting artifacts.
    Route such models to the agentic path or mark them unsupported."""

    def __init__(self, func_name: str, reason: str):
        super().__init__(
            f"UnsupportedFrontendOperand (accounting-incomplete): "
            f"'{func_name}' — {reason}. Mechanical lowering is fail-closed "
            f"for this call; use the agentic/verified path or an explicit "
            f"positional spelling.")
        self.func_name = func_name
        self.reason = reason


def _contains_bare_parameter(obj) -> bool:
    if isinstance(obj, nn.Parameter):
        return True
    if isinstance(obj, (tuple, list)):
        return any(_contains_bare_parameter(v) for v in obj)
    if isinstance(obj, dict):
        return any(_contains_bare_parameter(v) for v in obj.values())
    return False


def _signature_for(func):
    """Best-effort semantic signature for a torch callable (cached)."""
    global _TESTING_OVERRIDES
    try:
        return _SIG_CACHE[func]
    except (KeyError, TypeError):
        pass
    sig = None
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        try:
            if _TESTING_OVERRIDES is None:
                from torch.overrides import get_testing_overrides
                _TESTING_OVERRIDES = get_testing_overrides()
            stub = _TESTING_OVERRIDES.get(func)
            if stub is not None:
                sig = inspect.signature(stub)
        except Exception:
            sig = None
    try:
        _SIG_CACHE[func] = sig
    except TypeError:
        pass
    return sig


_SIG_SOURCE_CACHE: dict = {}


def _signature_with_source(func):
    """(signature, source) for a callable: ``('signature')`` when
    ``inspect.signature(func)`` succeeds (the callable's own signature, honouring
    ``__signature__`` and ``__wrapped__``), ``('testing_override')`` when only
    torch's testing-override stub describes it (C builtins), else (None, None)."""
    global _TESTING_OVERRIDES
    try:
        return _SIG_SOURCE_CACHE[func]
    except (KeyError, TypeError):
        pass
    result = (None, None)
    try:
        # the callable's OWN parameters (``__signature__`` is honoured; a
        # ``functools.wraps`` ``__wrapped__`` chain is NOT followed: the wrapped
        # function's signature is metadata about another callable)
        result = (inspect.signature(func, follow_wrapped=False), "signature")
    except (ValueError, TypeError):
        try:
            if _TESTING_OVERRIDES is None:
                from torch.overrides import get_testing_overrides
                _TESTING_OVERRIDES = get_testing_overrides()
            stub = _TESTING_OVERRIDES.get(func)
            if stub is not None:
                result = (inspect.signature(stub), "testing_override")
        except Exception:
            result = (None, None)
    try:
        _SIG_SOURCE_CACHE[func] = result
    except TypeError:
        pass
    return result


def _operand_bindings(func, args, kwargs):
    """Bind the executed call to the callable's signature and return
    ``(positional_names, keyword_names, source, parameter_order)``: the bound
    parameter NAME of every positional argument (by index) and of every
    keyword argument, the signature's source ('signature' /
    'testing_override') and its parameter order; ``None`` when no signature
    binds the call.  Evidence for consumers that must know which semantic
    slot each recorded argument occupied (issue #34, callee-7): it is
    tied to the callable's actual signature, not to its ``__name__`` /
    ``__qualname__`` (which ``functools.wraps`` copies onto other callables).
    A ``*args`` element is named ``<param>[k]``, a ``**kwargs`` entry
    ``**<param>``."""
    sig, source = _signature_with_source(func)
    if sig is None:
        return None
    # does the signature the metadata points to (``__wrapped__`` followed, the
    # one torch / older binders would report) order the parameters differently?
    wrapped_differs = False
    if source == "signature":
        try:
            followed = inspect.signature(func)
            wrapped_differs = list(followed.parameters) != list(sig.parameters)
        except (ValueError, TypeError):
            wrapped_differs = False
    # torch's universal aliasing destination ``out=`` names where the result is
    # written; it is not an operand and many signatures resolvable only through
    # the testing stubs omit it (``torch.mm`` -> ``(input, mat2)``): it is left
    # out of the binding (same rule as the operand-order binder's destination
    # split, INTEG-10 / an earlier change).
    out_v = kwargs.get("out") if kwargs else None
    out_leaves = _flatten_args([out_v]) if out_v is not None else []
    operand_kwargs = ({k: v for k, v in kwargs.items() if k != "out"}
                      if out_leaves and all(isinstance(v, torch.Tensor) for v in out_leaves) else kwargs)
    pos_sentinels = [("__POS__", i) for i in range(len(args))]
    kw_sentinels = {k: ("__KW__", k) for k in operand_kwargs}
    try:
        bound = sig.bind(*pos_sentinels, **kw_sentinels)
    except TypeError:
        return None
    pos_names = [None] * len(args)
    kw_names = {}
    for name, value in bound.arguments.items():
        kind = sig.parameters[name].kind
        if kind == inspect.Parameter.VAR_POSITIONAL:
            for k, v in enumerate(value):
                if isinstance(v, tuple) and v and v[0] == "__POS__":
                    pos_names[v[1]] = f"{name}[{k}]"
        elif kind == inspect.Parameter.VAR_KEYWORD:
            for k in value:
                kw_names[str(k)] = f"**{name}"
        elif isinstance(value, tuple) and value and value[0] == "__POS__":
            pos_names[value[1]] = name
        elif isinstance(value, tuple) and value and value[0] == "__KW__":
            kw_names[str(value[1])] = name
    # Resolved scalar defaults of the parameters the call OMITTED, from the same
    # signature the operands were bound to (default-8): an update-mode flag
    # such as ``training=True`` left to its default is as real as a supplied one.
    defaults = {}
    for name, param in sig.parameters.items():
        if name in bound.arguments or param.default is inspect.Parameter.empty:
            continue
        if isinstance(param.default, (bool, int, float, str)) or param.default is None:
            defaults[name] = param.default
    return pos_names, kw_names, source, list(sig.parameters), wrapped_differs, defaults
def _split_destination(kwargs):
    """Separate torch's universal aliasing destination keyword ``out=`` from
    the operand keywords (INTEG-10).  ``out`` names where the result is
    written; it is not an operand, carries no semantic order, and many
    ``torch.*`` signatures resolvable only through the testing stubs omit it
    (``torch.mm`` -> ``(input, mat2)``).  Returns ``(operand_kwargs, out)``;
    ``out`` is None when the keyword is absent or not a tensor (tuple of
    tensors)."""
    if not kwargs or "out" not in kwargs:
        return kwargs, None
    out = kwargs["out"]
    leaves = _flatten_args([out])
    if leaves and all(isinstance(v, torch.Tensor) for v in leaves):
        return {k: v for k, v in kwargs.items() if k != "out"}, out
    return kwargs, None


def _has_tensor(value) -> bool:
    """True when *value* is a tensor or holds one at any depth: through the
    lists/tuples ``_flatten_args`` orders AND through mapping values (keys are
    never operands).  A keyword is tensor-bearing on this test alone; the
    scalar-only retry of :func:`bind_operand_leaves` must never see a tensor
    hidden in a dict as a scalar (nested-1)."""
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_has_tensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_tensor(v) for v in value)
    return False


def _mapping_holds_tensor(value) -> bool:
    """True when a tensor sits inside a mapping anywhere in *value*.  Operand
    leaves are ordered through lists/tuples only (``_flatten_args`` and the
    recorded ``operand_kinds`` treat a mapping as one leaf), so such a tensor
    has no semantic operand placement: the call is refused, never bound with
    an invented order and never dropped."""
    if isinstance(value, Mapping):
        return any(_has_tensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_mapping_holds_tensor(v) for v in value)
    return False


def bind_operand_leaves(func, args, kwargs):
    """``(leaves, info)``: leaf argument values in the function's semantic
    parameter order, or ``leaves is None`` when the call must be refused.

    ``info`` is a dict with ``unbound_scalar_kwargs`` (keyword arguments that
    carry NO tensor and that the available signature does not know; they were
    left out of the bind because they carry no operand order: for
    ``scaled_dot_product_attention`` torch's testing-override stub predates
    ``scale`` / ``is_causal`` / ``enable_gqa``, ACC-140-SDPA-KWARGS) and, on
    refusal, ``refusal`` naming the cause: the unplaceable TENSOR keyword(s),
    tensors held inside a mapping keyword, tensors in a ``**kwargs``
    catch-all, or an unavailable signature.  The
    caller FAILS CLOSED on ``leaves is None`` (it must never invent an operand
    order or fall back to vanilla collection for such a call).
    """
    info = {"unbound_scalar_kwargs": [], "refusal": None}
    # torch's aliasing destination ``out=`` is not an operand (INTEG-10, an earlier change): split it
    # off before any binding so a stub signature without ``out`` still binds
    kwargs, _destination = _split_destination(kwargs or {})
    if not kwargs or not any(_has_tensor(v) for v in kwargs.values()):
        return _flatten_args(list(args)), info
    # A tensor inside a mapping (``payload={'extra': v}``, at any depth) is
    # tensor-bearing but has no semantic operand placement in the capture
    # contract: refuse, naming the keyword(s), before any signature bind or
    # scalar-only retry can classify or drop it (nested-1).
    mapped = sorted(k for k, v in kwargs.items() if _mapping_holds_tensor(v))
    if mapped:
        info["refusal"] = ("tensor keyword argument(s) " + ", ".join(mapped)
                           + " hold tensors inside a mapping, which has no semantic operand order "
                             "(operand leaves are ordered through lists/tuples only)")
        return None, info
    sig = _signature_for(func)
    if sig is None:
        info["refusal"] = ("tensor keyword argument(s) "
                           + ", ".join(sorted(k for k, v in kwargs.items() if _has_tensor(v)))
                           + " cannot be placed: no signature is available for the callable")
        return None, info
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        # A keyword the signature does not know is a problem only when it
        # carries a TENSOR (an operand whose semantic slot cannot be placed).
        # Non-tensor keywords carry no operand order: bind the tensor-bearing
        # arguments alone and record the dropped scalar keywords.
        tensor_kwargs = {k: v for k, v in kwargs.items() if _has_tensor(v)}
        try:
            bound = sig.bind(*args, **tensor_kwargs)
        except TypeError as exc:
            unplaceable = sorted(k for k in tensor_kwargs if k not in sig.parameters
                                 and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))
            info["refusal"] = (("tensor keyword argument(s) " + ", ".join(unplaceable) + " cannot be placed in the "
                                "callable's signature " + str(sig)) if unplaceable
                               else f"the call does not bind to the callable's signature {sig}: {exc}")
            return None, info
        info["unbound_scalar_kwargs"] = sorted(k for k in kwargs if k not in tensor_kwargs)
    leaves = []
    for name, value in bound.arguments.items():
        kind = sig.parameters[name].kind
        if kind == inspect.Parameter.VAR_KEYWORD:
            # a **kwargs catch-all has no semantic order between its tensors
            tensor_keys = sorted(k for k, v in value.items() if _has_tensor(v))
            if tensor_keys:
                info["refusal"] = ("tensor keyword argument(s) " + ", ".join(tensor_keys)
                                   + f" fall into the **{name} catch-all, which has no semantic operand order")
                return None, info
            continue
        leaves.extend(_flatten_args(value))
    return leaves, info


def _semantic_operand_leaves(func, args, kwargs):
    """Leaf argument values in the function's semantic parameter order, or
    None when the call must be refused (see :func:`bind_operand_leaves`)."""
    return bind_operand_leaves(func, args, kwargs)[0]


def ensure_parameter_capture() -> bool:
    """Idempotently enable parameter-operand capture on ANY torchview
    installation. Returns True if this call changed anything, False if all
    guards were already active.

    The recorder-side guards (semantic operand binding, scalar retention,
    fail-closed diagnostics) are installed REGARDLESS of any on-disk
    rendering patch: ``ComputationGraph._render_parameter_nodes`` proves only
    rendering support, not safe operand binding (round-3 review: the shipped
    parameter-tensors patch applied to the current torchview produced
    ZERO-input function nodes and 0 MACs while that marker was present).
    Each component carries its own sentinel so repeated calls are no-ops and
    the renderer is never double-wrapped; an on-disk ``_render_parameter_
    nodes`` is kept as the renderer (it consumes the same ``is_param`` nodes
    this recorder emits).
    """
    from torchview import computation_graph as cg
    from torchview import recorder_tensor as rt

    changed = False
    tf = rt.RecorderTensor.__dict__.get("__torch_function__")
    fn = getattr(tf, "__func__", tf)
    if not getattr(fn, "_solar_param_capture", False):
        _patch_recorder_tensor(rt)
        changed = True
    if not hasattr(cg.ComputationGraph, "_render_parameter_nodes"):
        _patch_computation_graph(cg)
        changed = True
    return changed


def _flatten_args(args):
    out = []
    if isinstance(args, (tuple, list)):
        for item in args:
            out.extend(_flatten_args(item))
    else:
        out.append(args)
    return out


def _patch_recorder_tensor(rt) -> None:
    ScriptMethod = rt.ScriptMethod
    RecorderTensor = rt.RecorderTensor
    reduce_data_info = rt.reduce_data_info
    collect_tensor_node = rt.collect_tensor_node
    collect_tensor = rt.collect_tensor
    collect_shape = rt.collect_shape
    attach_node = rt.attach_node
    traverse_data_inplace = rt.traverse_data_inplace
    NodeContainer = rt.NodeContainer
    TensorNode = rt.TensorNode
    FunctionNode = rt.FunctionNode
    OrderedSet = rt.OrderedSet
    stringify_attributes = rt.stringify_attributes

    def _operand_kinds(obj):
        """Mirror of the call's (args, kwargs) with every tensor leaf replaced by
        how THIS capture bound it: ``'parameter'`` (an nn.Parameter -> a
        parameter-tensor node), ``'activation'`` (a RecorderTensor with a
        producer node) or ``'detached'`` (any other tensor, e.g. a buffer: no
        node, no edge); every other value is None.  Recorded on the
        FunctionNode next to the stringified attributes so that a consumer can
        bind each recorded argument slot (by position / keyword against the
        callee's signature) to the executed call's actual operand identity
        instead of to a shape (issue #34 round 6, identity-5)."""
        if isinstance(obj, (list, tuple)):
            return [_operand_kinds(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): _operand_kinds(v) for k, v in obj.items()}
        if isinstance(obj, torch.Tensor) and hasattr(obj, "shape"):
            if isinstance(obj, nn.Parameter):
                return "parameter"
            if isinstance(obj, RecorderTensor) and getattr(obj, "tensor_nodes", None):
                return "activation"
            return "detached"
        # Scalar leaves are recorded STRUCTURALLY — the Python type and, for
        # Booleans / None / numbers, the value — so a consumer binds e.g. an
        # update-mode flag from the executed call's actual argument and never
        # from a substring of the stringified attributes (flagtext-10).
        # Anything else (a string, a dtype, a device, an arbitrary object) is
        # ``opaque``: its type is recorded, its value establishes nothing.
        if isinstance(obj, bool):
            return {"scalar": "bool", "value": bool(obj)}
        if obj is None:
            return {"scalar": "NoneType", "value": None}
        if isinstance(obj, (int, float)):
            return {"scalar": type(obj).__name__, "value": obj}
        return {"scalar": "opaque", "type": type(obj).__name__}

    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Activation TensorNodes (from RecorderTensor args/kwargs only)
        recorder_nodes = reduce_data_info(
            [args, kwargs], collect_tensor_node, NodeContainer())

        # necessary for torch version < 1.10
        if func in [F.linear, F.embedding]:
            out = nn.parameter.Parameter.__torch_function__(
                func, types, args, kwargs).as_subclass(RecorderTensor)
        else:
            out = super(RecorderTensor, cls).__torch_function__(
                func, types, args, kwargs)

        # if no RecorderTensor is found in input or output,
        # dont create any node, give the result only
        if not recorder_nodes:
            return out
        if not reduce_data_info(out, collect_tensor, OrderedSet()):
            return out

        cur_depth = next(iter(recorder_nodes)).depth
        input_context = next(iter(recorder_nodes)).context
        collect_attributes = next(iter(recorder_nodes)).collect_attributes

        # Resolve tensor operands in SEMANTIC parameter order (keyword
        # spellings bind through the signature; dict insertion order is never
        # an operand order).  None => FAIL CLOSED with a typed diagnostic —
        # trusted mechanical artifacts must not be produced from a call whose
        # operand accounting cannot be established.
        operand_leaves, bind_info = bind_operand_leaves(func, args, kwargs)
        if operand_leaves is None:
            func_name_early = (
                func.name if isinstance(func, ScriptMethod)
                else getattr(func, "__name__", repr(func)))
            raise UnsupportedFrontendOperandError(
                func_name_early,
                (bind_info.get("refusal") or "tensor keyword arguments cannot be bound to a semantic parameter order")
                + " (keyword tensor operands need a semantic order; scalar keywords alone never refuse)")

        # Build ordered_input_nodes over the resolved operands.  For each
        # tensor operand (0-d scalars included — vanilla torchview keeps
        # them and dropping one severs its producer edge):
        #   - RecorderTensor -> its existing TensorNode
        #   - bare nn.Parameter -> a parameter-tensor TensorNode (one node
        #     per distinct parameter, re-appended per occurrence so operand
        #     slots survive while graph parents stay deduplicated)
        ordered_input_nodes = []
        all_input_shapes = []
        param_nodes_by_id = {}

        for arg in operand_leaves:
            if not isinstance(arg, torch.Tensor) or not hasattr(arg, "shape"):
                continue
            # nn.Parameter is a torch.Tensor subclass; on meta device it may
            # also be a RecorderTensor, so check Parameter first.
            is_model_param = isinstance(arg, nn.Parameter)
            if is_model_param:
                param_node = param_nodes_by_id.get(id(arg))
                if param_node is None:
                    param_node = TensorNode(
                        tensor=arg,
                        depth=cur_depth,
                        name="parameter-tensor",
                        context=input_context,
                        is_aux=False,
                        main_node=None,
                        collect_attributes=collect_attributes,
                    )
                    param_node.is_param = True
                    param_nodes_by_id[id(arg)] = param_node
                ordered_input_nodes.append(param_node)
                all_input_shapes.append(tuple(arg.shape))
            elif isinstance(arg, RecorderTensor):
                if getattr(arg, "tensor_nodes", None):
                    ordered_input_nodes.append(arg.tensor_nodes[-1])
                all_input_shapes.append(tuple(arg.shape))

        if not ordered_input_nodes:
            # Binding succeeded but no operand node was resolved (e.g. the
            # traced tensors sit inside containers the flattener treats as
            # leaves).  A dropped Parameter here would silently shrink the
            # accounting -> typed fail-closed diagnostic; otherwise keep
            # vanilla parents/shapes (connectivity intact, no operand order
            # claimed, no parameter capture).
            if _contains_bare_parameter((args, kwargs)):
                func_name_early = (
                    func.name if isinstance(func, ScriptMethod)
                    else getattr(func, "__name__", repr(func)))
                raise UnsupportedFrontendOperandError(
                    func_name_early,
                    "a bare nn.Parameter operand could not be captured as an "
                    "ordered operand")
            args_nodes = recorder_nodes
            ordered_input_nodes = None
            all_input_shapes = reduce_data_info(
                [args, kwargs], collect_shape, [])
        else:
            args_nodes = NodeContainer(ordered_input_nodes)

        func_name = (
            func.name if isinstance(func, ScriptMethod) else func.__name__)
        attributes = (stringify_attributes((args, kwargs))
                      if collect_attributes else None)

        cur_node = FunctionNode(
            func, cur_depth, args_nodes, name=func_name, attributes=attributes)
        # Identity of the callee torchview actually executed (module-qualified
        # name of the function object; torchview keeps only id()).  A consumer
        # that binds recorded arguments to a signature must pick that signature
        # from this, never from an argument count (issue #34, binding-6).
        cur_node.callee = (
            f"ScriptMethod.{func.name}" if isinstance(func, ScriptMethod)
            else f"{getattr(func, '__module__', None)}.{getattr(func, '__qualname__', getattr(func, '__name__', '?'))}")
        # The callable's TYPE (a C builtin such as torch.batch_norm cannot be
        # produced by functools.wraps) and the executed call bound to the
        # callable's actual signature: the parameter name of every positional
        # and keyword argument (callee-7).  None when nothing binds.
        cur_node.callable_type = type(func).__name__
        _bindings = None if isinstance(func, ScriptMethod) else _operand_bindings(func, args, kwargs)
        if _bindings is not None:
            cur_node.operand_bindings = [_bindings[0], _bindings[1]]
            cur_node.binding_source = _bindings[2]
            cur_node.binding_param_order = _bindings[3]
            cur_node.binding_wrapped_differs = bool(_bindings[4])
            cur_node.binding_defaults = dict(_bindings[5])
        if ordered_input_nodes is not None:
            cur_node.ordered_input_nodes = ordered_input_nodes
            # per-slot binding evidence (same classification as the loop above)
            cur_node.operand_kinds = [_operand_kinds(list(args)), _operand_kinds(kwargs)]
        if bind_info.get("unbound_scalar_kwargs"):
            # scalar keywords the signature did not know (recorded so accounting
            # / perf can read e.g. SDPA's ``scale`` / ``is_causal`` later)
            cur_node.unbound_scalar_kwargs = list(bind_info["unbound_scalar_kwargs"])
        if func_name == "interpolate" and not isinstance(func, ScriptMethod):
            # Structured, typed record of F.interpolate's scalar arguments from
            # the ACTUAL objects (issue #50 / parser-5): the identity-copy
            # rule consumes only this, never the stringified attributes.
            # Guarded at the shim boundary (capture-evidence-11): a
            # failure inside the builder must never abort the capture; the
            # node carries a structured error record instead (no evidence
            # fields) and the consumer refuses it (unrecognised / incomplete).
            try:
                from solar.einsum.op_families import interpolate_evidence_record
                _ev = interpolate_evidence_record(func, args, kwargs)
            except Exception as _ev_exc:
                _ev = None
                cur_node.interpolate_evidence_error = {
                    "schema": "solar-interpolate-evidence-error-v1",
                    "type": type(_ev_exc).__name__, "message": str(_ev_exc)[:2000]}
            if _ev is not None:
                cur_node.interpolate_evidence = _ev

        for i in args_nodes:
            i.add_child(cur_node)

        input_context.append(cur_node)
        attach_kwargs = {
            "parents": cur_node, "depth": cur_depth, "context": input_context,
            "is_aux": False, "parent_hierarchy": {cur_depth: cur_node},
            "name": "output-tensor" if cur_depth == 0 else "hidden-tensor",
            # Carry the flag onto the result TensorNodes here, exactly like
            # torchview's collect-attributes patch does in-source.  Without
            # it the result node defaults to False and an IN-PLACE op
            # (unsqueeze_, add_) leaves its operand's latest node unflagged,
            # so every later call on that tensor records no attributes
            # (BigBird: 19,982 of 32,614 FunctionNodes; INTEG-6 / R8).
            "collect_attributes": collect_attributes,
        }
        traverse_data_inplace(out, attach_node(attach_kwargs))

        cur_node.set_input_shape(all_input_shapes)
        cur_node.set_output_shape(reduce_data_info(out, collect_shape, []))

        # Record the ACTUAL ordered output dtypes from the executed
        # results: the recorder holds `out` here, and a cast's target
        # dtype (to(dtype)/dtype= keyword/convenience casts) exists in no
        # other trace evidence — stage 1 previously filled the model
        # default and billed a BF16 result at FP32.  The same traversal
        # as set_output_shape keeps slots ordered; the result TensorNodes
        # are stamped so tensor-level extraction agrees.  This records
        # evidence only — nothing is inferred from an op name.
        def _record_dtype(recorded, collected):
            dtype = str(recorded.dtype)
            collected.append(dtype)
            nodes = getattr(recorded, "tensor_nodes", None)
            if nodes:
                nodes[-1].tensor_dtype = dtype

        cur_node.recorded_output_dtypes = reduce_data_info(
            out, _record_dtype, [])

        return out

    # sentinel: lets ensure_parameter_capture() detect that the recorder-side
    # guards are active without trusting any rendering-side marker
    __torch_function__._solar_param_capture = True
    RecorderTensor.__torch_function__ = classmethod(__torch_function__)


def _patch_computation_graph(cg) -> None:
    ComputationGraph = cg.ComputationGraph
    TensorNode = cg.TensorNode
    FunctionNode = cg.FunctionNode
    ModuleNode = cg.ModuleNode

    def _render_parameter_nodes(self) -> None:
        """Parameter-tensor TensorNodes are parents of visited Function/
        ModuleNodes but live outside the hierarchy tree, so collect_graph
        never reaches them; add them and their param -> func edges here."""
        visited, compute_nodes = set(), []
        for tail, head in self.edge_list:
            for n in (tail, head):
                if isinstance(n, (FunctionNode, ModuleNode)) \
                        and id(n) not in visited:
                    visited.add(id(n))
                    compute_nodes.append(n)
        for compute_node in compute_nodes:
            for parent in compute_node.parents:
                if (isinstance(parent, TensorNode)
                        and getattr(parent, "is_param", False)
                        and id(parent) not in self.node_set):
                    self.add_node(parent, self.visual_graph)
                    self.edge_list.append((parent, compute_node))

    orig_render_nodes = ComputationGraph.render_nodes

    def render_nodes(self) -> None:
        orig_render_nodes(self)
        self._render_parameter_nodes()

    orig_is_node_visible = ComputationGraph.is_node_visible

    def is_node_visible(self, compute_node) -> bool:
        if isinstance(compute_node, TensorNode) \
                and getattr(compute_node, "is_param", False):
            return True
        return orig_is_node_visible(self, compute_node)

    orig_get_node_color = ComputationGraph.get_node_color

    @staticmethod
    def get_node_color(node) -> str:
        if isinstance(node, TensorNode) and getattr(node, "is_param", False):
            return "lightsalmon"
        return orig_get_node_color(node)

    ComputationGraph._render_parameter_nodes = _render_parameter_nodes
    ComputationGraph.render_nodes = render_nodes
    ComputationGraph.is_node_visible = is_node_visible
    ComputationGraph.get_node_color = get_node_color
