# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a metadata sidecar dict from runtime trace entries and write as YAML."""

from collections import Counter
import sys
import yaml
import logging

logger = logging.getLogger(__name__)


def _declared_outputs(entries, id_to_name, tensors, graph):
    """Resolve only recorded return identities, never names/shapes as guesses."""
    records = [e for e in entries if e["type"] == "model_outputs"]
    if not records:
        return None  # Historical trace: no current return-provenance claim.
    if any(e.get("unsupported_return") for e in entries if e["type"] == "call"):
        # a kernel returned a tuple / a non-Tensor (TARGET-DSL-TUPLE-13): it has no output node, so no return
        # identity can be resolved; the converter refuses that kernel with the typed ``unsupported_tuple_return``
        # (an earlier change) -- the provenance builder states "unrecorded" instead of raising a generic producer error (an earlier change merge)
        return None
    if len(records) != 1 or records[0].get("schema") != "dsl-return-v1":
        raise ValueError("output provenance: missing/ambiguous return record")
    tensor_rows = [e for e in entries if e["type"] == "tensor"]
    by_id = {e["id"]: e for e in tensor_rows}
    if len(by_id) != len(tensor_rows):
        raise ValueError("output provenance: reused tensor identity")
    producers = Counter(e.get("output_id") for e in entries if e["type"] == "call")
    available = {tid for tid, e in by_id.items() if e.get("input_name") or tid not in producers}
    if {e["input_name"] for e in tensor_rows if e.get("input_name")} & set(graph):
        raise ValueError("output provenance: input/producer name collision")
    for e in entries:
        if e["type"] != "call":
            continue
        tid = e.get("output_id")
        if (tid not in by_id or producers[tid] != 1 or by_id[tid].get("input_name")
                or any(i not in by_id or i not in available for i in e["input_ids"])):
            raise ValueError("output provenance: missing/ambiguous producer or unsupported in-place version")
        available.add(tid)
    result, seen, paths = [], {}, set()
    for leaf in records[0].get("outputs", []):
        path = leaf.get("path")
        if not isinstance(path, list) or any(type(p) not in (str, int) for p in path):
            raise ValueError("output provenance: invalid return path")
        path_key = tuple((type(p).__name__, p) for p in path)
        if path_key in paths:
            raise ValueError("output provenance: duplicate return path")
        paths.add(path_key)
        if "tensor_id" in leaf:
            tid = leaf["tensor_id"]
            entry = by_id.get(tid)
            if entry is None or tid not in id_to_name:
                raise ValueError("output provenance: unknown returned tensor")
            name = id_to_name[tid]
            is_input = bool(entry.get("input_name"))
            if (is_input and not entry.get("input_alias")) or (not is_input and producers[tid] != 1):
                raise ValueError("output provenance: missing producer or converted input return")
            shape, dtype = list(entry["shape"]), entry["dtype"]
            op = None if is_input else name
        elif "input" in leaf:
            inp = leaf["input"]
            name, shape, dtype = inp["name"], inp["shape"], inp["dtype"]
            if name in graph:
                raise ValueError("output provenance: input name collides with producer")
            tensors.setdefault(name, {"dims": {"D"+str(i): s for i, s in enumerate(shape)},
                                      "dtype": dtype, "role": "input", "origin": "registered-input"})
            is_input, op = True, None
        else:
            raise ValueError("output provenance: missing returned identity")
        if name in seen:
            previous = seen[name]
            if previous["shape"] != shape or previous["op"] != op or previous["is_input"] != is_input:
                raise ValueError("output provenance: conflicting returned identity")
            previous["paths"].append(path)
        else:
            row = {"tensor": name, "op": op, "is_input": is_input, "shape": shape,
                   "dtype": dtype, "paths": [path]}
            seen[name] = row
            result.append(row)
    if not result:
        raise ValueError("output provenance: no returned identities")
    return result


def _resolve_input_ref(inp_id, id_to_name, id_to_meta):
    """Build an input reference dict for a graph node."""
    if inp_id is not None and inp_id in id_to_name:
        src = id_to_name[inp_id]
        role = id_to_meta.get(inp_id, {}).get("role")
        from_field = (
            "input"
            if role in ("input", "weight", "mask")
            else src
        )
        return {"tensor": src, "from": from_field}
    return {"tensor": "unknown", "from": "unknown"}


def _process_tensor_entry(entry, id_to_name, id_to_meta, tensors):
    """Process a tensor trace entry into the metadata structures."""
    tid = entry["id"]
    meta = {
        "dims": dict(zip(entry["dim_names"], entry["shape"])),
        "dtype": entry["dtype"],
    }
    if entry.get("role"):
        meta["role"] = entry["role"]
    if entry.get("sparsity") is not None:
        meta["sparsity"] = entry["sparsity"]
        if entry.get("sparsity_source"):
            # provenance tag ("measured" / "measured_capped") — the analyzer
            # ignores unknown keys; consumers use it to judge scale-freeness
            meta["sparsity_source"] = entry["sparsity_source"]
    id_to_meta[tid] = meta
    if entry.get("input_name"):
        name = entry["input_name"]
        id_to_name[tid] = name
        meta["role"] = meta.get("role", "input")
        # Actual provenance, carried through to the analyzed graph: this
        # tensor came from the registered PUBLIC inputs dict, whatever it
        # is named — __input_0 is a valid identifier a user may register,
        # so a name spelling is never provenance (review C19-P1).
        meta["origin"] = "registered-input"
        tensors[name] = meta


def _process_call_entry(entry, id_to_name, id_to_meta, tensors, graph,
                        call_counter):
    """Process a call trace entry into the metadata structures."""
    func_name = entry["name"]
    count = call_counter[func_name]
    call_counter[func_name] += 1
    node_name = f"{func_name}#{count}"
    out_id = entry["output_id"]
    if out_id is not None:
        id_to_name[out_id] = node_name
        if out_id in id_to_meta:
            tensors[node_name] = id_to_meta[out_id]
    inputs_list = [
        _resolve_input_ref(inp_id, id_to_name, id_to_meta)
        for inp_id in entry["input_ids"]
    ]
    node = {
        "type": entry["call_type"],
        "function": func_name,
        "inputs": inputs_list,
        "output": node_name,
    }
    if entry.get("call_site_line") is not None:
        node["call_site_line"] = entry["call_site_line"]
    if entry.get("input_order") is not None:
        # "signature": inputs are in parameter order; the converter requires
        # this marker before evaluating arithmetic size specs.
        node["input_order"] = entry["input_order"]
    if entry.get("registration_order") is not None:
        # run-time dim-registry registration order of the inputs (indices into
        # ``inputs``): positional arguments, then keywords in call order
        node["registration_order"] = list(entry["registration_order"])
    if entry.get("unsupported_return") is not None:
        # TARGET-DSL-TUPLE-13: no output Tensor was returned; the assembler refuses this node with a typed error
        node["unsupported_return"] = dict(entry["unsupported_return"])
    if entry.get("scalar_args"):
        # TARGET-KB-MAXPOOL: the call's int arguments by parameter name (Dim sizes may name them)
        node["scalar_args"] = dict(entry["scalar_args"])
    graph[node_name] = node


def build_metadata(trace_entries, *, engine_source=None):
    """Walk trace entries and produce the ``tensors`` + ``graph`` metadata dict.

    ``engine_source`` (optional, coverage-1): a provenance record such
    as :func:`solar.engine_source.engine_source` returns.  When given it is
    stored under the top-level key ``engine_source`` so the metadata binds
    the engine that produced it; when omitted nothing is added (the writer
    boundary -- ``scripts/to_einsum_lib.py`` -- injects it before saving).

    Naming rules:
    * Input tensors: named by their key in the inputs dict (matched via
      input_name set during Tensor.from_numpy with registered inputs).
    * Kernel/builtin outputs: node name is ``function#index`` where
      *function* is the ``@kernel`` or builtin name and *index* is the
      zero-based invocation count.  The ``#`` separator cannot appear in
      Python identifiers, so no collision with real function names.
    """
    id_to_name = {}
    id_to_meta = {}
    tensors = {}
    graph = {}
    call_counter = Counter()

    # First pass: register all tensor entries and name inputs
    for entry in trace_entries:
        if entry["type"] == "tensor":
            _process_tensor_entry(entry, id_to_name, id_to_meta, tensors)

    # Auto-name unregistered tensors (e.g. scalar constants created inline)
    unnamed_counter = Counter()
    for entry in trace_entries:
        if entry["type"] == "tensor" and entry["id"] not in id_to_name:
            tid = entry["id"]
            meta = id_to_meta.get(tid, {})
            role = meta.get("role", "")
            prefix = role if role else "const"
            idx = unnamed_counter[prefix]
            name = f"__{prefix}_{idx}"
            # A registered public input may legally own a __<role>_<n>
            # name; never clobber it — advance to a free auto-name.
            while name in tensors:
                idx += 1
                name = f"__{prefix}_{idx}"
            unnamed_counter[prefix] = idx + 1
            id_to_name[tid] = name
            meta.setdefault("role", "input")
            # Actual provenance: created inline by the trace, not a
            # registered public input (review C19-P1).
            meta["origin"] = "internal-constant"
            tensors[name] = meta

    # Second pass: process call entries
    for entry in trace_entries:
        if entry["type"] == "call":
            _process_call_entry(entry, id_to_name, id_to_meta, tensors,
                                graph, call_counter)

    result = {"tensors": tensors, "graph": graph}
    outputs = _declared_outputs(trace_entries, id_to_name, tensors, graph)
    if outputs is not None:
        result.update(model_outputs=outputs, model_output_provenance="dsl-return-v1")
    unrecorded = [e for e in trace_entries if e.get("type") == "model_outputs_unrecorded"]
    _unsupported = [e for e in trace_entries if e.get("type") == "call" and e.get("unsupported_return")]
    if _unsupported and not outputs and not unrecorded:
        _u = _unsupported[0]
        unrecorded = [{"reason": f"kernel {_u.get('function')!r} returned {(_u['unsupported_return'] or {}).get('kind', 'a non-Tensor')!s}: "
                                 "no output node, no return identity (the converter refuses the kernel with a typed error)"}]
    if unrecorded and not outputs:
        # the returned arrays could not be bound to traced identities: the
        # external-output footprint is unknown and the analyzer fails closed
        result["returned_outputs_unrecorded"] = str(unrecorded[0].get("reason"))
    _metadata = result
    if engine_source is not None:
        _metadata["engine_source"] = engine_source
    return _metadata


def merge_measured_density(metadata, measured):
    """Copy same-shape measurements without replacing analytic densities.

    The API path does not run the original-scale revalidation ladder. Refuse
    changed shapes here; trace_saved_ir can revalidate them under a budget.
    """
    for name, sample in measured.get("tensors", {}).items():
        target = metadata.get("tensors", {}).get(name)
        if (target is None or target.get("sparsity") is not None
                or sample.get("sparsity") is None
                or not sample.get("sparsity_source", "").startswith("measured")):
            continue
        if sample.get("dims") != target.get("dims"):
            logger.info(f"[density] REFUSED {name}: verification/original shapes differ; "
                  "use scripts/trace_saved_ir.py --measure-density for "
                  "original-scale revalidation")
            continue
        target["sparsity"] = sample["sparsity"]
        target["sparsity_source"] = sample["sparsity_source"]


def write_metadata_yaml(metadata, path):
    """Serialize *metadata* to a YAML file at *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)
