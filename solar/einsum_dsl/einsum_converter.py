# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert DSL dim names to solar einsum equation format.

The solar einsum format uses tokens matching [A-Z][0-9]* (single capital letter
optionally followed by digits), as required by solar.common.utils.parse_dim_tokens.

Mapping rules for DSL dim names:
  - Single letter, all caps (B, S, H, D, K, N): use as-is
  - Letter + pure digits (H2, B0, S1): keep base letter + digits (e.g. H2 → H2)
  - Multi-char alpha (Sk, Nh, Qs): take first char uppercase; if conflict, append counter
"""

from __future__ import annotations

import ast
import re
from typing import Optional
from .model import DSLGraph, DSLKernel, DSLTensor
from .builtin_table import DSL_TENSOR_BUILTINS


# ---------------------------------------------------------------------------
# Dim-name → einsum token mapping
# ---------------------------------------------------------------------------

def _dim_name_to_token(dim_name: str, taken: set) -> str:
    """Map one DSL dim name to a unique valid einsum token.

    Valid token: single capital letter optionally followed by digits ([A-Z][0-9]*).

    Examples:
        B      → B
        S      → S
        H2     → H2   (digit suffix preserved)
        Sk     → S    (non-digit suffix dropped; or S0 if S is taken)
        Nh     → N    (non-digit suffix dropped)
    """
    if not dim_name:
        raise ValueError("Empty dim name")

    base = dim_name[0].upper()
    rest = dim_name[1:]

    # Keep pure-digit suffix (e.g. H2 → H2)
    candidate = base + rest if rest.isdigit() else base

    if candidate not in taken:
        return candidate

    # Conflict: find the next available base+N
    i = 0
    while True:
        candidate = base + str(i)
        if candidate not in taken:
            return candidate
        i += 1


def build_dim_map(kernels_tensors: list) -> dict:
    """Build a global dim_name → token mapping across a list of tensors.

    Processes dims in first-appearance order so the assignment is deterministic.
    """
    dim_to_token: dict = {}
    taken: set = set()
    for tensor in kernels_tensors:
        for dim_name in tensor.dims.keys():
            if dim_name not in dim_to_token:
                token = _dim_name_to_token(dim_name, taken)
                dim_to_token[dim_name] = token
                taken.add(token)
    return dim_to_token


# ---------------------------------------------------------------------------
# Einsum equation builders
# ---------------------------------------------------------------------------

def _tensor_tokens(tensor: DSLTensor, dim_map: dict) -> str:
    """Return the concatenated token string for a tensor's dims."""
    return "".join(dim_map[d] for d in tensor.dims.keys())



# ---------------------------------------------------------------------------
# Loop-expr based einsum equation (compound dim support)
# ---------------------------------------------------------------------------

def _parse_tensor_accesses(expr: str) -> list:
    """Parse tensor accesses from an expression string.

    Returns [(tensor_name, [index_expr, ...]), ...].
    E.g. 'Y[b, o, p, q] += Xp[b, c, p * stride + r, q * stride + s] * W[o, c, r, s]'
    → [('Y', ['b','o','p','q']), ('Xp', ['b','c','p * stride + r','q * stride + s']),
       ('W', ['o','c','r','s'])]
    """
    results = []
    for m in re.finditer(r'(\w+)\[([^\]]+)\]', expr):
        name = m.group(1)
        indices = [idx.strip() for idx in m.group(2).split(',')]
        results.append((name, indices))
    return results


def _identify_loop_vars(all_accesses: list) -> set:
    """Identify loop variables: identifiers that appear as a standalone index in any access."""
    loop_vars = set()
    for _, indices in all_accesses:
        for idx in indices:
            if re.fullmatch(r'[a-zA-Z_]\w*', idx):
                loop_vars.add(idx)
    return loop_vars


def _vars_in_index(index_expr: str, loop_vars: set) -> list:
    """Extract loop variables from an index expression, preserving order.

    'p * stride + r' with loop_vars={p,r} → ['p', 'r']
    'b' → ['b']
    """
    seen = set()
    result = []
    for m in re.finditer(r'[a-zA-Z_]\w*', index_expr):
        ident = m.group()
        if ident in loop_vars and ident not in seen:
            result.append(ident)
            seen.add(ident)
    return result


def _build_var_to_token(ordered_vars: list) -> dict:
    """Map loop variables to einsum tokens ([A-Z][0-9]*)."""
    var_to_token = {}
    taken = set()
    for var in ordered_vars:
        base = var[0].upper()
        if base not in taken:
            var_to_token[var] = base
            taken.add(base)
        else:
            i = 0
            while f"{base}{i}" in taken:
                i += 1
            candidate = f"{base}{i}"
            var_to_token[var] = candidate
            taken.add(candidate)
    return var_to_token


def _index_to_tokens(index_expr: str, loop_vars: set, var_to_token: dict) -> str:
    """Convert one index expression to einsum token(s).

    Simple: 'b' → 'B'
    Compound: 'p * stride + r' → '(P+R)'
    Literal int: '0' → empty (handled by dim-name fallback at operand level)
    """
    vars_found = _vars_in_index(index_expr, loop_vars)
    if len(vars_found) == 0:
        return ''
    if len(vars_found) == 1:
        return var_to_token[vars_found[0]]
    return '(' + '+'.join(var_to_token[v] for v in vars_found) + ')'


def _access_to_operand(indices: list, loop_vars: set, var_to_token: dict,
                       tensor: DSLTensor = None) -> Optional[str]:
    """Convert a tensor's index list to an einsum operand string.

    For indices that are loop variables or compound expressions, tokens come from
    var_to_token.  For literal-int indices (e.g. '0' in W[c, 0, r, s]), the
    tensor's dim name at that position is used to generate a token — this handles
    singleton dims like groups=1 in depthwise convolutions.

    Returns None if the resulting token count doesn't match the tensor's ndim.
    """
    taken = set(var_to_token.values())
    tensor_dim_names = list(tensor.dims.keys()) if tensor else []

    parts = []
    for i, idx in enumerate(indices):
        tok = _index_to_tokens(idx, loop_vars, var_to_token)
        if tok:
            parts.append(tok)
        elif i < len(tensor_dim_names):
            # Literal index (e.g. '0') — use tensor's dim name to generate token
            dim_name = tensor_dim_names[i]
            token = _dim_name_to_token(dim_name, taken)
            taken.add(token)
            parts.append(token)

    # Validate: token count must match tensor ndim
    if tensor and len(parts) != len(tensor.dims):
        return None

    return ''.join(parts)


def _balanced_tensor_accesses(expr: str) -> list:
    """Bracket-aware tensor-access parser.

    Unlike ``_parse_tensor_accesses`` (regex, breaks on nested brackets), this
    handles gather-style nested indexing: ``EW[IDS[b, s], h]`` parses as
    ``('EW', ['IDS[b, s]', 'h'])`` plus the inner ``('IDS', ['b', 's'])``.
    """
    results = []
    i, n = 0, len(expr)
    while i < n:
        m = re.match(r"[A-Za-z_]\w*", expr[i:])
        if not m:
            i += 1
            continue
        name = m.group()
        j = i + len(name)
        if j < n and expr[j] == '[':
            depth, k = 0, j
            while k < n:
                if expr[k] == '[':
                    depth += 1
                elif expr[k] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth == 0:
                inner = expr[j + 1:k]
                idxs, buf, d = [], [], 0
                for ch in inner:
                    if ch == '[':
                        d += 1
                    elif ch == ']':
                        d -= 1
                    if ch == ',' and d == 0:
                        idxs.append(''.join(buf).strip())
                        buf = []
                    else:
                        buf.append(ch)
                if buf:
                    idxs.append(''.join(buf).strip())
                results.append((name, idxs))
                results.extend(_balanced_tensor_accesses(inner))
                i = k + 1
                continue
        i = j
    return results


def _gather_read_overrides(kernel: DSLKernel) -> dict:
    """{input_position: elements_actually_read} for gather-indexed inputs.

    A tensor indexed by a gathered value — ``EW[IDS[b, s], h]`` inline, or
    ``e = SE[n, k]`` then ``W[e, i, h]`` — reads only the looked-up rows, not
    the whole table.  Billing the full table as DRAM traffic overstates
    memory-bound SOL by table_rows/lookups (measured 40x on an embedding).

    Reads = product over the access's index positions of:
      * gather position  -> element count of the index-source tensor (an upper
        bound on distinct rows) x extents of any loop vars in the same expr
      * loop-var position -> the loop var's extent
      * literal           -> 1
    capped at the full tensor size.  Emitted only when every access to the
    tensor is analyzable, at least one is gather-indexed, and the bound is
    smaller than the full tensor.
    """
    if not kernel.loop_expr or not kernel.param_names:
        return {}
    stmts = [s.strip() for s in kernel.loop_expr.split(';') if s.strip()]
    if not stmts:
        return {}
    accesses = _balanced_tensor_accesses('; '.join(stmts))
    if not accesses:
        return {}

    name_to_shape = {
        name: inp.tensor.shape
        for name, inp in zip(kernel.param_names, kernel.inputs)
    }
    first = _balanced_tensor_accesses(stmts[0])
    if first and first[0][0] not in name_to_shape:
        name_to_shape[first[0][0]] = kernel.output.shape

    def _elems(name):
        shape = name_to_shape.get(name)
        if not shape:
            return None
        p = 1
        for d in shape:
            p *= int(d)
        return p

    # Assigned gather values: v = T[...] / v = int(T[...])
    gather_src = {}
    for s in stmts:
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(?:int\s*\(\s*)?([A-Za-z_]\w*)\[", s)
        if m and m.group(2) in name_to_shape:
            gather_src[m.group(1)] = m.group(2)

    # Loop vars (standalone indices that aren't gather values) and extents.
    loop_vars = set()
    for _, idxs in accesses:
        for idx in idxs:
            if re.fullmatch(r"[A-Za-z_]\w*", idx) and idx not in gather_src:
                loop_vars.add(idx)
    extents: dict = {}
    for name, idxs in accesses:
        shape = name_to_shape.get(name)
        if not shape or len(idxs) != len(shape):
            continue
        for pos, idx in enumerate(idxs):
            if idx in loop_vars:
                extents[idx] = max(extents.get(idx, 0), int(shape[pos]))

    overrides = {}
    for pos, pname in enumerate(kernel.param_names):
        full = _elems(pname)
        if not full:
            continue
        pname_accs = [(n, idxs) for n, idxs in accesses if n == pname]
        if not pname_accs:
            continue
        any_gather, all_ok, max_prod = False, True, 0
        for _, idxs in pname_accs:
            shape = name_to_shape.get(pname)
            if len(idxs) != len(shape):
                all_ok = False
                break
            prod, has_gather, ok = 1, False, True
            for idx in idxs:
                nested = re.match(r"(?:int\s*\(\s*)?([A-Za-z_]\w*)\[", idx)
                idents = re.findall(r"[A-Za-z_]\w*", idx)
                gvars = [v for v in idents if v in gather_src]
                lvars = [v for v in idents if v in loop_vars]
                factor = 1
                if nested and nested.group(1) in name_to_shape:
                    src = _elems(nested.group(1))
                    if src is None:
                        ok = False
                        break
                    factor *= src
                    has_gather = True
                    # loop vars in the expr but OUTSIDE the nested access:
                    # strip the balanced NAME[...] span, scan the remainder.
                    start = nested.start(1)
                    depth, k2 = 0, idx.index('[', start)
                    while k2 < len(idx):
                        if idx[k2] == '[':
                            depth += 1
                        elif idx[k2] == ']':
                            depth -= 1
                            if depth == 0:
                                break
                        k2 += 1
                    outside = idx[:start] + idx[k2 + 1:] if k2 < len(idx) else idx[:start]
                    lvars = [v for v in re.findall(r"[A-Za-z_]\w*", outside)
                             if v in loop_vars]
                elif gvars:
                    has_gather = True
                    for g in gvars:
                        src = _elems(gather_src[g])
                        if src is None:
                            ok = False
                            break
                        factor *= src
                    if not ok:
                        break
                for lv in lvars:
                    if lv not in extents:
                        ok = False
                        break
                    factor *= extents[lv]
                if not ok:
                    break
                prod *= factor
            if not ok:
                all_ok = False
                break
            any_gather = any_gather or has_gather
            max_prod = max(max_prod, prod)
        if all_ok and any_gather and 0 < max_prod < full:
            overrides[pos] = int(max_prod)
    return overrides


def _spec_text(spec) -> str:
    """The source form of a size spec, for refusal texts."""
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and spec[0] == "scalar":
            return str(spec[1])
        if len(spec) == 3 and spec[0] == "shape":
            return f"{spec[1]}.shape[{spec[2]}]"
        if len(spec) == 4 and spec[0] == "binop":
            return f"({_spec_text(spec[2])} {spec[1]} {_spec_text(spec[3])})"
        if len(spec) == 4 and spec[0] == "opaque":
            return str(spec[1])
    return repr(spec)


_DIM_UNBINDABLE = "Dim call arguments not statically bindable"   # ast_parser._DIM_UNBINDABLE (dim-call-binding-23)


def _is_unbindable_spec(spec) -> bool:
    """An opaque spec that stands for a ``Dim(...)`` call whose ARGUMENTS the parser could not bind to
    ``Dim(name, parallel=False, size=None)`` (``**{"size": 3}``, ``*args`` ...): the runtime may register a size the
    parser cannot see, so the call is never treated as size-less."""
    return _is_opaque_spec(spec) and str(spec[1]).startswith(_DIM_UNBINDABLE)


def _is_opaque_spec(spec) -> bool:
    """An explicit ``size=`` the parser could not reduce to a supported form (opaque-size-20)."""
    return isinstance(spec, (tuple, list)) and len(spec) == 4 and spec[0] == "opaque"


def _has_shape_arithmetic(spec) -> bool:
    return isinstance(spec, (tuple, list)) and len(spec) == 4 and spec[0] == "binop"


def _shape_leaves(spec) -> list:
    """The ("shape", param, axis) leaves of a size spec."""
    if isinstance(spec, (tuple, list)):
        if len(spec) == 3 and spec[0] == "shape":
            return [tuple(spec)]
        if len(spec) == 4 and spec[0] == "binop":
            return _shape_leaves(spec[2]) + _shape_leaves(spec[3])
    return []


def _scalar_leaves(spec) -> list:
    """The ("scalar", name) parameters a size spec names (for the refusal text when the trace did not record them)."""
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and spec[0] == "scalar":
            return [spec[1]]
        if len(spec) == 4 and spec[0] == "binop":
            return _scalar_leaves(spec[2]) + _scalar_leaves(spec[3])
    return []


def _eval_size_spec(spec, param_to_shape, binop_shapes=None, scalars=None):
    """Evaluate an ast_parser size spec at the call site.

    int -> itself; ("shape", param, axis) -> that input tensor's extent;
    ("binop", op, left, right) -> recursively evaluated integer arithmetic
    (ported from an earlier revision: sizes such as ``W.shape[0] // 2`` resolve to the
    loop's real extent instead of falling back to the bare-index axis, which
    billed the L1/029 half-range projection at the full weight axis).
    Returns None when any leaf cannot be resolved.

    ``binop_shapes`` is the parameter->shape map that arithmetic specs may
    use; pass None when the correspondence between recorded inputs and
    parameter names is not established, so every BinOp over a shape leaf
    stays unresolved (base behaviour) instead of computing from the wrong
    tensor.  Plain shape specs keep using ``param_to_shape`` as before.
    """
    if spec is None or isinstance(spec, int):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and spec[0] == "scalar":
        # TARGET-KB-MAXPOOL (an earlier change round 4): a non-Tensor kernel argument, from the call's recorded ``scalar_args``
        # (traces saved before they were recorded resolve nothing here -> the loop is refused, naming the argument)
        val = (scalars or {}).get(spec[1])
        return int(val) if isinstance(val, int) and not isinstance(val, bool) else None
    if isinstance(spec, (tuple, list)) and len(spec) == 3 and spec[0] == "shape":
        shape = param_to_shape.get(spec[1])
        axis = spec[2]
        return shape[axis] if shape and axis < len(shape) else None
    if isinstance(spec, (tuple, list)) and len(spec) == 4 and spec[0] == "binop":
        if binop_shapes is None:
            return None
        left = _eval_size_spec(spec[2], binop_shapes, binop_shapes, scalars)
        right = _eval_size_spec(spec[3], binop_shapes, binop_shapes, scalars)
        if left is None or right is None:
            return None
        op = spec[1]
        try:
            if op == "//":
                return left // right
            if op == "*":
                return left * right
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "%":
                return left % right
        except ZeroDivisionError:
            return None
    return None


class LoopDimExtentMismatch(ValueError):
    """A ``Dim()`` loop's dimension cannot be trusted at this call: its runtime
    bindings disagree (two arguments / allocations register the same name with
    different sizes), the loop variable indexes a tensor axis NARROWER than the
    bound extent (an out-of-bounds or guarded traversal the source alone cannot
    size), or the name is bound only by a local allocation whose size is not
    statically resolvable.  The converter records it (``loop_dim_extent_mismatch``
    / ``loop_dim_extent_unresolved``) and the kernel's count is incomplete."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


LOOP_DIM_ORDER_UNKNOWN = "loop_dim_order_unknown"
LOOP_ENTRY_UNRESOLVED = "loop_entry_unresolved"
LOOP_DIM_SCALAR_UNRESOLVED = "loop_dim_scalar_unresolved"     # round 5: an explicit size over a scalar argument the trace did not record


class _Unresolved(str):
    """A reason text that also names the refusal code the loop reading the name must raise with."""
    code = "loop_dim_extent_unresolved"

    def __new__(cls, reason, code="loop_dim_extent_unresolved"):
        obj = super().__new__(cls, reason)
        obj.code = code
        return obj


def _input_registrations(kernel: DSLKernel) -> tuple:
    """Replay the registrations of the kernel's ARGUMENTS (core.kernel: positional, then keyword, in CALL order; the
    last registration of a name wins) -> ``(extents, sources, order_unknown)``.

    legacy-kworder-12: a trace without ``registration_order`` (saved before TARGET-DSL-DIM-KWORDER-7) fixes no
    call order -- its inputs are listed in signature order, and a keyword call ``k(Z=z, X=x)`` registered them the
    other way round.  When two inputs carry the SAME dim name with DIFFERENT extents, which binding won is unknowable
    from such a trace: ``order_unknown[name]`` lists the sites and the loop reading that name is refused
    (``loop_dim_order_unknown``).  Inputs that agree, or a trace that recorded the order, bind the name."""
    extents: dict = {}
    sources: dict = {}
    inputs = list(kernel.inputs)
    pnames = list(kernel.param_names or [f"arg{i}" for i in range(len(inputs))])
    order = getattr(kernel, "registration_order", None)
    ordered = isinstance(order, list) and sorted(order) == list(range(len(inputs)))
    if ordered:
        seq = [(pnames[i] if i < len(pnames) else f"arg{i}", inputs[i]) for i in order]
    else:
        seq = list(zip(pnames, inputs))
    disagree = set()
    for pname, inp in seq:
        for dname, dsize in inp.tensor.dims.items():
            if isinstance(dsize, int) and dsize > 0:
                if not ordered and dname in extents and extents[dname] != int(dsize):
                    disagree.add(dname)
                extents[dname] = int(dsize); sources.setdefault(dname, []).append((f"input {pname}.{dname}", int(dsize)))
    order_unknown = {d: list(sources[d]) for d in sorted(disagree)}
    return extents, sources, order_unknown


def _order_unknown_reason(kernel: DSLKernel, dname: str, sites: list) -> "_Unresolved":
    return _Unresolved(
        f"bound by inputs that carry the name {dname!r} with different extents ("
        + ", ".join(f"{site}={ext}" for site, ext in sites)
        + ") in a trace that did not record the registration order (no `registration_order`): which binding the "
        f"kernel {kernel.id!r} read depends on the call's keyword order, which this trace cannot tell; retrace",
        LOOP_DIM_ORDER_UNKNOWN)


def _runtime_dim_bindings(kernel: DSLKernel, param_to_shape: dict, binop_shapes, before_line=None):
    """The dim registry a ``Dim("NAME")`` loop reads at THIS call, mirrored
    statically (annotation-binding-1 / TARGET-DSL-DIM-LOCAL-2, round 3).

    ``@kernel`` opens a fresh registry scope and registers each argument's
    ACTUAL dim names in CALL order (``dims[name] = size``: the LAST registration
    of a name wins), then the body runs and every ``Tensor(NAME=size, ...)``
    allocation registers its names as it executes (again last wins), and
    ``Dim`` reads the registry.  Annotations never touch it.  The mirror
    applies, in order:

    * every recorded input's dims by NAME, in the trace's recorded
      ``registration_order`` (positional arguments, then keywords in call
      order -- TARGET-DSL-DIM-KWORDER-7); a legacy trace without that record
      is replayed in its recorded (signature) order, which equals the call
      order for positional calls;
    * every local allocation whose line precedes ``before_line`` (the entry
      of the loop being sized; all of them when the line is unknown), in
      source order, sized from this call's shapes.  An allocation whose
      execution is not statically certain (``conditional``: under if / while /
      try / with / match, for-else, a non-Dim loop body) is an UNCERTAIN
      binding (conditional-binding-4): it leaves the name's extent
      alone when it binds the already established value, otherwise the name
      is unresolved until a later certain allocation re-establishes it.  The
      RETURNED allocation -- identified by NAME binding: the unique certain
      allocation assigned to the name a ``return`` statement returns
      (return-identity-5), never by its dim-name set -- may take the
      trace's recorded output sizes for a size the converter cannot evaluate;
      any other allocation with a non-static size makes the name unresolved.

    Returns ``(extents, conflicts, unresolved)``; ``conflicts`` is non-empty
    only when ``before_line`` is unknown (legacy metadata) and the bindings
    disagree, the order being unrecoverable then; ``unresolved`` maps a name
    to the reason it cannot be trusted."""
    extents, sources, order_unknown = _input_registrations(kernel)
    unresolved: dict = {}
    conflicts: dict = {}
    for dname, sites in order_unknown.items():                  # legacy-kworder-12
        unresolved[dname] = _order_unknown_reason(kernel, dname, sites)
        extents.pop(dname, None)
    allocs = list(getattr(kernel, "local_tensor_dims", None) or [])
    out_dims = {k: int(v) for k, v in kernel.output.dims.items() if isinstance(v, int) and v > 0}
    returned = [n for n in (getattr(kernel, "returned_names", None) or []) if n]
    returned_idx = None
    if returned and len(set(returned)) == 1 and None not in (getattr(kernel, "returned_names", None) or [None]):
        candidates = [i for i, a in enumerate(allocs) if a.get("target") == returned[0] and not a.get("conditional")]
        if len(candidates) == 1 and not any(a.get("target") == returned[0] and a.get("conditional") for a in allocs):
            returned_idx = candidates[0]
    for i, alloc in enumerate(allocs):
        line = alloc.get("line")
        if before_line is not None and line is not None and line >= before_line:
            continue                                                # executes after the loop being sized is entered
        certain = not alloc.get("conditional")
        for dname, spec in (alloc.get("dims") or {}).items():
            val = _eval_size_spec(spec, param_to_shape, binop_shapes, getattr(kernel, "scalar_args", None))
            if val is None and i == returned_idx and dname in out_dims:
                val = out_dims[dname]                                # the recorded size of the RETURNED allocation
            site = f"Tensor({dname}=...) line {line}"
            if not (isinstance(val, int) and val > 0):
                if certain or dname not in extents:
                    unresolved[dname] = (f"registered by a local Tensor allocation at line {line} whose size is not "
                                         f"statically resolvable at this call" + ("" if certain else " (and whose execution is not certain)"))
                    extents.pop(dname, None)
                else:
                    unresolved[dname] = (f"may be re-registered by a conditional allocation at line {line} whose size is not "
                                         f"statically resolvable; the binding at loop entry is not established")
                    extents.pop(dname, None)
                continue
            val = int(val)
            if certain:
                prev = {v for _, v in sources.get(dname, [])}
                if before_line is None and prev and val not in prev:        # legacy metadata: order unrecoverable
                    conflicts.setdefault(dname, list(sources.get(dname, []))).append((site, val))
                extents[dname] = val; unresolved.pop(dname, None); sources.setdefault(dname, []).append((site, val))
            else:
                if dname in extents and extents[dname] == val and dname not in unresolved:
                    continue                                        # rebinds the established extent: no uncertainty
                unresolved[dname] = (f"may be re-registered by a conditional allocation at line {line} (Tensor({dname}={val})) "
                                     f"while the established binding is {extents.get(dname, 'absent')}; the binding at loop "
                                     f"entry is not statically certain")
                extents.pop(dname, None)
    for dname in list(conflicts):
        if len({v for _, v in conflicts[dname]}) == 1:
            conflicts.pop(dname)
    return extents, conflicts, unresolved


def _returned_allocation_index(kernel: DSLKernel):
    """Index (into ``local_tensor_dims``) of the allocation the kernel RETURNS, or None.

    The recorded output dims may size that allocation alone (return-identity-5).  Round 4
    (return-rebind-9): the returned object is the LAST certain binding of the returned name before the return,
    followed through plain aliases (``Alias = Y``) to a certain allocation; a last binding that is conditional, a
    non-allocation (``Y = Existing``, a call, an unpacking), a parameter, or a return of a non-name expression
    identifies no allocation -- nothing is borrowed and an unresolvable size is refused.  Several returns must all
    resolve to the same allocation."""
    allocs = list(getattr(kernel, "local_tensor_dims", None) or [])
    bindings = getattr(kernel, "name_bindings", None)
    returns = getattr(kernel, "returns", None)
    if bindings is None or returns is None:
        # graphs parsed before the bindings were recorded (round 3 rule): the unique certain allocation of the name
        returned = [n for n in (getattr(kernel, "returned_names", None) or []) if n]
        if returned and len(set(returned)) == 1 and None not in (getattr(kernel, "returned_names", None) or [None]):
            candidates = [i for i, a in enumerate(allocs) if a.get("target") == returned[0] and not a.get("conditional")]
            if len(candidates) == 1 and not any(a.get("target") == returned[0] and a.get("conditional") for a in allocs):
                return candidates[0]
        return None
    if not returns or any(r.get("name") is None for r in returns):
        return None

    def resolve(name, before_seq, depth=0):
        if depth > 8:
            return None
        prior = [b for b in bindings if b.get("name") == name and isinstance(b.get("seq"), int) and b["seq"] < before_seq]
        if not prior:
            return None                                          # a parameter, or never bound: not an allocation
        last = prior[-1]
        if last.get("conditional"):
            return None                                          # which binding the return sees is not certain
        if last.get("kind") == "alloc" and isinstance(last.get("alloc"), int) and 0 <= last["alloc"] < len(allocs):
            return last["alloc"]
        if last.get("kind") == "alias" and last.get("source"):
            return resolve(last["source"], last["seq"], depth + 1)
        return None

    found = {resolve(r["name"], r.get("seq", 10 ** 9)) for r in returns}
    if len(found) == 1 and None not in found:
        return found.pop()
    return None


def _access_witness_names(kernel: DSLKernel):
    """The local name(s) whose bare accesses in the loop body witness the kernel's OUTPUT shape (P2-203-SCRATCH-EXTENT-1).

    The positional accesses are a witness against the registry binding of a loop's dimension: a bare access of the loop
    variable on an axis narrower than the bound extent is refused (``loop_dim_extent_mismatch``, TARGET-DSL-DIM-EXTENT-13).
    The witness knows the recorded inputs by parameter name and one local shape, the kernel's output; that shape belongs
    to the tensor the loop WRITES as its output -- the RETURNED allocation, identified exactly as the registry mirror
    identifies it (``_returned_allocation_index``: the last certain binding of the returned name before the return,
    through plain aliases; return-identity-5 / RETURN-REBIND-9) -- and to the returned name itself.  Up to an earlier revision
    the output shape was given to the FIRST tensor the body accessed: in the Lite 203 candidate's ``topk_select`` that is
    the E-wide scratch ``QV`` [T, E] while the returned ``Y`` is [T, K], so the synthetic K-element axis on the scratch
    refused the E loop (the review, issue #58 note 78936488) -- a false diagnostic, not a candidate defect.

    Returns the empty set when the returned allocation is not certain (a conditional or in-loop rebinding, a return of an
    expression, several returns of different allocations, a returned parameter -- whose shape the witness already has by
    name): the output shape then witnesses nothing, and the count rests on the registry binding alone, never on a guessed
    name.  Returns None only for a record without return information (parsed before ``returned_names`` / ``returns``
    were recorded): the caller keeps the first-access convention for it."""
    returned_names = getattr(kernel, "returned_names", None)
    returns = getattr(kernel, "returns", None)
    if returned_names is None and returns is None:
        return None
    names: set = set()
    allocs = list(getattr(kernel, "local_tensor_dims", None) or [])
    idx = _returned_allocation_index(kernel)
    if idx is not None and 0 <= idx < len(allocs) and allocs[idx].get("target"):
        names.add(allocs[idx]["target"])
        if returned_names and None not in returned_names and len(set(returned_names)) == 1:
            names.add(returned_names[0])                              # a plain alias of the allocation (``Out = Y; return Out``)
    return names


def _access_witness_epochs(kernel: DSLKernel, stmts: list):
    """Per innermost statement of ``loop_expr``: which accessed names provably denote the RETURNED allocation at THAT
    access (they borrow the output shape), and which names are bound locally before it (they are no longer the
    parameter of that spelling) -- witness-lifetime-1 / access-alias-2 (the review on an earlier revision), ordered by
    binding sequence and matched through the parser's normalized statements (same-line-epoch-4 /
    source-normalization-5, the review on an earlier revision).

    A return-time identity is not a global fact about a name: ``Q = Tensor(T, E); <loop writes Q>; Q = Tensor(T, K);
    return Q`` returns the second allocation while the loop wrote the first.  Every ``Tensor(...)`` allocation or
    rebinding of a name starts a new binding epoch (``name_bindings``, execution order ``seq``); an access borrows the
    output shape only when the binding live at the access -- the last one executed before it, followed through plain
    ``A = B`` aliases in the same scope, each resolved at ITS OWN sequence position (so ``Y = Tensor(T, E); A = Y`` on
    one line captures that allocation, never an earlier same-name one) -- is the returned allocation
    (``_returned_allocation_index``).  Every plain alias of that allocation live at the access is mapped the same way
    (``Y = Tensor(T, K); A = Y; <loop writes A[t, e]>; return Y`` keeps its narrow-axis refusal).  Where the association
    is not proved -- the live binding or a prior rebinding is conditional, the name is (re)bound inside a loop
    enclosing the access (loop-carried) or on the access's own line (order unrecorded), an alias through anything but
    a plain name, the record lacks the source segment / bindings / loop spans, or a statement of the segment does not
    reproduce its ``loop_expr`` statement -- THAT access borrows nothing (the registry binding alone sizes the loop);
    never the first-access convention, and never at the expense of the other statements' witnesses.

    Access lines come from the kernel's own source segment (``source_text``), normalized exactly as the parser
    normalizes the body before it extracts ``loop_expr`` (``statement_class.normalize_statements``: a valued
    annotated assignment is the plain assignment), matched statement by statement, and placed on absolute lines
    through the ``loop_nest`` roots (the segment's top-level ``Dim`` loops in order).  Returns ``(borrow, shadowed)``:
    two lists of name sets, one entry per statement."""
    n = len(stmts)
    borrow = [set() for _ in range(n)]; shadowed = [set() for _ in range(n)]
    bindings = [b for b in (getattr(kernel, "name_bindings", None) or []) if isinstance(b, dict) and b.get("name")]
    if not bindings:
        return borrow, shadowed                                  # nothing bound locally: parameters only, nothing to borrow
    bound_names = {b["name"] for b in bindings}
    accessed = [{name for name, _ in _parse_tensor_accesses(text)} for text in stmts]
    conservative = ([set() for _ in range(n)], [set(bound_names & accessed[i]) for i in range(n)])   # no borrow; an accessed name bound locally anywhere shadows its parameter spelling
    seg = getattr(kernel, "source_text", None); nest = getattr(kernel, "loop_nest", None)
    if not seg or not nest or n == 0:
        return conservative
    try:
        import textwrap
        from .statement_class import normalize_statements
        tree = ast.parse(textwrap.dedent(seg))
    except SyntaxError:
        return conservative
    fn = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return conservative
    normalize_statements(fn.body)                                # the parser's own view of the body (positions kept)

    def _is_dim_for(st):
        return (isinstance(st, ast.For) and isinstance(st.iter, ast.Call)
                and ((isinstance(st.iter.func, ast.Name) and st.iter.func.id == "Dim")
                     or (isinstance(st.iter.func, ast.Attribute) and st.iter.func.attr == "Dim")))

    def innermost(for_node):
        if any(isinstance(st, ast.For) for st in for_node.body):
            out = []
            for st in for_node.body:
                if isinstance(st, ast.For):
                    out.extend(innermost(st))
            return out
        return [st for st in for_node.body
                if isinstance(st, (ast.Assign, ast.AugAssign)) or (isinstance(st, ast.AnnAssign) and st.value is not None)]

    inner_nodes = []
    for st in fn.body:
        if isinstance(st, ast.For):
            inner_nodes.extend(innermost(st))
    roots_rel = [st.lineno for st in fn.body if _is_dim_for(st)]
    roots_abs = [node.get("line") for node in nest if isinstance(node, dict)]
    if not roots_rel or len(roots_rel) != len(roots_abs) or not all(isinstance(a, int) for a in roots_abs):
        return conservative
    offsets = {a - r for a, r in zip(roots_abs, roots_rel)}
    if len(offsets) != 1:
        return conservative
    offset = offsets.pop()
    spans = [(node.lineno + offset, getattr(node, "end_lineno", node.lineno) + offset)
             for node in ast.walk(fn) if isinstance(node, (ast.For, ast.While, ast.AsyncFor))]
    ordered = sorted(bindings, key=lambda b: (b.get("seq") if isinstance(b.get("seq"), int) else 10 ** 9))
    allocs = list(getattr(kernel, "local_tensor_dims", None) or [])
    returned_idx = _returned_allocation_index(kernel)
    # a statement is MATCHED when the segment's normalized statement at the same index reproduces its loop_expr text;
    # an unmatched statement is unproved on its own (conservative for ITS accesses only)
    matched = [i < len(inner_nodes) and ast.unparse(inner_nodes[i]) == stmts[i] for i in range(n)]

    def of(name):
        return [b for b in ordered if b["name"] == name]

    def live(name, line, seq=None, depth=0):
        """Identity of *name* at an access on *line* (``seq`` None) or at the alias binding with sequence *seq*:
        ("alloc", i), ("param", name) for a never-bound name, or None when unproved."""
        if depth > 8 or not isinstance(line, int):
            return None
        mine = of(name)
        if seq is None:
            if any(b.get("line") == line for b in mine):
                return None                                      # bound on the access's own line: the order is not recorded
            prior = [b for b in mine if isinstance(b.get("line"), int) and b["line"] < line]
        else:
            prior = [b for b in mine if isinstance(b.get("seq"), int) and b["seq"] < seq]
        if not prior:
            return ("param", name)
        enclosing = [(s, e) for s, e in spans if s <= line <= e]
        if any(isinstance(b.get("line"), int) and any(s <= b["line"] <= e for s, e in enclosing) for b in mine):
            return None                                          # (re)bound inside a loop enclosing this line: loop-carried
        last = prior[-1]
        if last.get("conditional") or not isinstance(last.get("seq"), int):
            return None
        if last.get("kind") == "alloc" and isinstance(last.get("alloc"), int) and 0 <= last["alloc"] < len(allocs):
            return ("alloc", last["alloc"])
        if last.get("kind") == "alias" and last.get("source") and isinstance(last.get("line"), int):
            return live(last["source"], last["line"], seq=last["seq"], depth=depth + 1)
        return None

    for i in range(n):
        if not matched[i]:
            shadowed[i] = set(bound_names & accessed[i]); continue
        line = inner_nodes[i].lineno + offset
        for name in accessed[i]:
            if name not in bound_names:
                continue                                         # a parameter (or an unbound name): its recorded shape applies
            if not any(isinstance(b.get("line"), int) and b["line"] <= line for b in of(name)):
                continue                                         # bound only after this access: still the parameter here
            shadowed[i].add(name)
            if returned_idx is not None and live(name, line) == ("alloc", returned_idx):
                borrow[i].add(name)
    return borrow, shadowed


WITNESS_UNMODELLED = "witness_unmodelled"
_DYNAMIC_NAMESPACE_SPELLINGS = frozenset({"exec", "eval", "locals", "globals", "vars", "setattr", "delattr", "__import__"})
_WITNESS_STATEMENT_NODES = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Name, ast.Constant, ast.Subscript, ast.BinOp, ast.UnaryOp,
                            ast.Compare, ast.Attribute, ast.Tuple, ast.Call, ast.keyword, ast.Load, ast.Store, ast.Del,
                            ast.operator, ast.unaryop, ast.cmpop, ast.expr_context)


def _dim_for_name(st):
    """(is a ``for ... in Dim(...)`` loop, the Dim name when it is a string literal else None)."""
    if not (isinstance(st, ast.For) and isinstance(st.iter, ast.Call)):
        return False, None
    f = st.iter.func
    if not ((isinstance(f, ast.Name) and f.id == "Dim") or (isinstance(f, ast.Attribute) and f.attr == "Dim")):
        return False, None
    arg = st.iter.args[0] if st.iter.args else next((kw.value for kw in st.iter.keywords if kw.arg == "name"), None)
    return True, (arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None)


def _placed_segment(kernel: DSLKernel, stmts: list):
    """The kernel's source segment parsed and normalized as the parser normalizes it, with its innermost statements in
    ``loop_expr`` order: ``(fn, inner_nodes, matched, offset, chains)`` where ``matched[i]`` says the i-th innermost statement
    reproduces ``stmts[i]``, ``offset`` maps the segment's lines to absolute lines and ``chains[i]`` is the list of ``for``
    nodes enclosing the i-th statement, outermost first; None when the segment cannot be parsed or placed (no ``source_text``
    / ``loop_nest``, unparsable text, a different number or order of top-level Dim loops)."""
    n = len(stmts)
    seg = getattr(kernel, "source_text", None); nest = getattr(kernel, "loop_nest", None)
    if not seg or not nest or n == 0:
        return None
    try:
        import textwrap
        from .statement_class import normalize_statements
        tree = ast.parse(textwrap.dedent(seg))
    except SyntaxError:
        return None
    fn = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return None
    normalize_statements(fn.body)
    inner, chains = [], []

    def leaves(for_node, chain):
        chain = chain + [for_node]
        if any(isinstance(st, ast.For) for st in for_node.body):
            for st in for_node.body:
                if isinstance(st, ast.For):
                    leaves(st, chain)
            return
        for st in for_node.body:
            if isinstance(st, (ast.Assign, ast.AugAssign)) or (isinstance(st, ast.AnnAssign) and st.value is not None):
                inner.append(st); chains.append(chain)

    for st in fn.body:
        if isinstance(st, ast.For):
            leaves(st, [])
    roots_rel = [st.lineno for st in fn.body if _dim_for_name(st)[0]]
    roots_abs = [node.get("line") for node in nest if isinstance(node, dict)]
    if not roots_rel or len(roots_rel) != len(roots_abs) or not all(isinstance(a, int) for a in roots_abs):
        return None
    offsets = {a - r for a, r in zip(roots_abs, roots_rel)}
    if len(offsets) != 1:
        return None
    matched = [i < len(inner) and ast.unparse(inner[i]) == stmts[i] for i in range(n)]
    return fn, inner, matched, offsets.pop(), chains


def _witness_model_v2(kernel: DSLKernel, stmts: list):
    """WITNESS_MODEL v2 (round 8, minimal): which bare index components of the innermost statements are witness-eligible,
    and the Dim loop that governs each -- by LEXICAL nesting only.  No statement-order, evaluation-order or scope emulation.

    A witness CANDIDATE is a bare ``Name`` component of a plain ``Subscript`` on a ``Name`` whose spelling is the target of
    some ``for <spelling> in Dim(...)`` loop in the kernel function; a ``Constant`` or expression component, and a spelling
    never bound as a Dim target (a derived index, a scalar parameter), witness no Dim loop and are outside the domain
    (no witness, no note).  A candidate IS a witness only when every condition of the model holds:
    (b) the spelling is bound anywhere in the kernel function ONLY as a Dim-loop target; (b') no Dim loop over it is nested
    inside another Dim loop over it; (c) the statement's AST holds only names, constants, subscripts, arithmetic, single
    comparisons, attribute loads, tuples of those and calls to names (or attribute chains rooted at names) bound nowhere in
    the kernel and not dynamic-namespace spellings; (d) every loop enclosing the statement is a Dim loop; (e) the kernel has
    no class statement, no nested def / lambda mentioning the spelling, no dynamic-namespace spelling (``exec``, ``eval``,
    ``locals``, ``globals``, ``vars``, ``setattr``, ``delattr``, ``__import__``) and no call to a kernel-bound name; (f) a Dim
    loop over the spelling lexically encloses the statement (the innermost one is the governor), and (v5, (f)) NO lexically
    enclosing Dim loop of the statement -- the governor or any ancestor -- has a break / continue / raise / return anywhere in its
    body (through nested compound statements, not into nested def / lambda / class bodies).  A candidate failing any condition gets NO witness and a typed ``witness_unmodelled:<first failing
    condition>`` note (the typed boundary).  v4 coverage: accesses outside the extracted innermost statements are recorded on
    ``kernel.witness_uninspected_notes`` (``access_outside_loop_expr``), so ``full`` means every access was inspected and eligible.

    Returns None when the segment cannot be placed; else one entry per statement: None for a statement the segment does
    not reproduce, otherwise ``(accesses, notes)`` with ``accesses`` = [(tensor name, [(spelling or None, (entry line, dim
    name) or None), ...]), ...] (a component's governor, or None when it witnesses nothing) and ``notes`` = [(tag, line)]."""
    placed = _placed_segment(kernel, stmts)
    if placed is None:
        return None
    fn, inner, matched, offset, chains = placed
    nodes = list(ast.walk(fn))
    dim_fors: dict = {}                                                  # spelling -> [For nodes over Dim with that Name target]
    dim_target_ids = set()
    for node in nodes:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and _dim_for_name(node)[0]:
            dim_fors.setdefault(node.target.id, []).append(node); dim_target_ids.add(id(node.target))
    from .ast_parser import _control_transfers                           # (f): the same recursive body walk as the counter's control_transfer_in_dim_loop
    transfer_loops = {id(f) for fors in dim_fors.values() for f in fors if _control_transfers(f.body)}
    bound_otherwise: set = set()                                         # every spelling bound by any form other than a Dim-loop target
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in dim_target_ids:
            bound_otherwise.add(node.id)
        elif isinstance(node, ast.NamedExpr):
            bound_otherwise.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound_otherwise.update(a.asname if a.asname else a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            bound_otherwise.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound_otherwise.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound_otherwise.update(node.names)
        elif isinstance(node, ast.arg):
            bound_otherwise.add(node.arg)                                # the kernel's own parameters and those of nested defs / lambdas
        else:
            for attr in ("name", "rest"):                                # match captures
                if type(node).__name__.startswith("Match") and isinstance(getattr(node, attr, None), str):
                    bound_otherwise.add(getattr(node, attr))
    bound_in_kernel = bound_otherwise | set(dim_fors)
    # (e) kernel-wide reasons, in order
    kernel_reason = None
    if any(isinstance(node, ast.ClassDef) for node in nodes):
        kernel_reason = "class_statement"
    elif any((isinstance(node, ast.Name) and node.id in _DYNAMIC_NAMESPACE_SPELLINGS) or (isinstance(node, ast.Attribute) and node.attr in _DYNAMIC_NAMESPACE_SPELLINGS) for node in nodes):
        kernel_reason = "dynamic_namespace_spelling"
    else:
        for node in nodes:
            if isinstance(node, ast.Call):
                root = node.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not isinstance(root, ast.Name) or root.id in bound_in_kernel:
                    kernel_reason = "opaque_call"; break               # a call to a kernel-bound name (a closure, an alias, an import inside the kernel) or to a non-name
    nested_mentions: set = set()                                         # Dim spellings a nested def / lambda mentions
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not fn:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in dim_fors:
                    nested_mentions.add(sub.id)
                elif isinstance(sub, ast.arg) and sub.arg in dim_fors:
                    nested_mentions.add(sub.arg)
    shadowed: set = set()                                                # (b'): a Dim loop over the spelling nested in the BODY of another Dim loop over it
    for spelling, fors in dim_fors.items():                              # (v3: For.orelse is a sibling segment -- a for-else loop runs after the loop finished)
        for outer in fors:
            inner_ids = {id(x) for st in outer.body for x in ast.walk(st)}
            if any(id(f) in inner_ids for f in fors):
                shadowed.add(spelling); break

    def statement_reason(st):
        ann = getattr(st, "annotation", None)                            # the parser's Assign view keeps an AnnAssign's annotation (never evaluated in function scope)
        if ann is not None and any(not isinstance(x, (ast.Name, ast.Attribute, ast.Constant, ast.Subscript, ast.Tuple, ast.Load)) for x in ast.walk(ann)):
            return "statement_annotation"
        for node in ast.walk(st):
            if isinstance(node, ast.Call):
                root = node.func
                while isinstance(root, ast.Attribute):
                    if root.attr in _DYNAMIC_NAMESPACE_SPELLINGS:
                        return "statement_call"
                    root = root.value
                if not isinstance(root, ast.Name) or root.id in bound_in_kernel or root.id in _DYNAMIC_NAMESPACE_SPELLINGS:
                    return "statement_call"
            elif isinstance(node, ast.Compare) and len(node.comparators) != 1:
                return "statement_compare_chain"
            elif isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
                return "statement_attribute_store"
            elif not isinstance(node, _WITNESS_STATEMENT_NODES):
                return "statement_" + type(node).__name__.lower()
        return None

    out = []
    for i in range(len(stmts)):
        if not matched[i]:
            out.append(None); continue
        st, chain = inner[i], chains[i]
        line = st.lineno + offset
        plain_loop = any(not _dim_for_name(f)[0] for f in chain)
        st_reason = statement_reason(st)
        # (f) v5 (ancestor-transfer-26): a break / continue / raise / return anywhere in the body of ANY lexically enclosing
        # Dim loop -- the governor or an ancestor -- can keep this statement from executing at all indices (or at all), so the
        # full extent of no enclosing loop is proven reached at the access: every candidate component is unmodelled.  A
        # transfer in a non-enclosing (sibling) nest does not affect the statement.
        enclosing_transfer = any(id(f) in transfer_loops for f in chain)
        accesses, notes = [], []
        for node in ast.walk(st):
            if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
                continue
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            comps = []
            for e in elts:
                if not (isinstance(e, ast.Name) and e.id in dim_fors):
                    comps.append((e.id if isinstance(e, ast.Name) else None, None)); continue        # outside the witness domain
                sp = e.id
                reason = (kernel_reason or ("nested_scope_mentions_spelling" if sp in nested_mentions else None)
                          or ("spelling_bound_otherwise" if sp in bound_otherwise else None)
                          or ("spelling_shadowed_by_nested_dim_loop" if sp in shadowed else None)
                          or ("statement_under_plain_loop" if plain_loop else None) or st_reason
                          or ("control_transfer_in_enclosing_loop" if enclosing_transfer else None))
                gov = None
                if reason is None:
                    enclosing = [f for f in chain if isinstance(f.target, ast.Name) and f.target.id == sp and _dim_for_name(f)[0]]
                    if enclosing:
                        f = enclosing[-1]; gov = (f.lineno + offset, _dim_for_name(f)[1])
                        if gov[1] is None:
                            reason, gov = "statement_dim_name_not_literal", None
                    else:
                        reason = "spelling_not_enclosed"
                if reason is not None:
                    notes.append((f"{WITNESS_UNMODELLED}:{reason}", line))
                comps.append((sp, gov))
            accesses.append((node.value.id, comps))
        out.append((accesses, notes))
    # WITNESS_MODEL v4 §2 coverage: every Subscript-on-Name access of the kernel FUNCTION that is not in an extracted innermost
    # statement (outer-level in-loop statements, if / while / try / with / orelse bodies, nested def / lambda / class bodies) was
    # not inspected by the witness -- one typed note per access; ``witness_coverage`` can be ``full`` only without them.
    inspected = {id(n) for st in inner for n in ast.walk(st)}
    kernel.witness_uninspected_notes = sorted({(f"{WITNESS_UNMODELLED}:access_outside_loop_expr", node.lineno + offset) for node in nodes
                                               if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and id(node) not in inspected})
    return out


_REPLAY_DISTINCT_STATES = 8


def _replay_loop_nest(kernel: DSLKernel, param_to_shape: dict, binop_shapes, resolve, explicit=None, only_path=None, only_line=None) -> tuple:
    """Count the kernel's loop nest by replaying the dim registry through it iteration by iteration
    (round 4, loop-carried-10) -> ``(innermost weighted trips, innermost trips, outer-level statements)``.

    The registry is loop-carried state: an allocation anywhere in an outer body re-registers its names before the
    NEXT outer iteration, so an inner loop's extent may differ between the first outer iteration and the rest.  Each
    loop is replayed from its entry state; as soon as an iteration starts from the same registry state as the one
    before it, every remaining iteration repeats that one (first + (n-1) x steady).  A loop whose entry states keep
    changing is refused; so is a loop whose dimension is bound uncertainly at the entry of ANY replayed iteration
    (``resolve`` raises).  A ``return`` that is not strictly after a loop (conditional or not, inside the body or
    before it) makes the loop's execution count unprovable from the source: refused as ``loop_entry_unresolved``
    (TARGET-DSL-EARLY-RETURN-11).  Counting conventions are those of the flat paths: a leaf's ``weight`` per trip,
    a non-leaf's ``level_assigns`` per trip of its own level.

    Round 5: statement identity is the AST POSITION ``(line, col)`` against each loop's span -- never a line range --
    so a one-line body ``for t in Dim("T"): Tmp = Tensor(U=3)`` is the loop's own statement (same-line-16);
    explicit ``Dim("U", size=...)`` registrations -- in a for-header or standalone -- are replayed as registry state
    exactly like allocations (dim-registration-17); and the RETURNED allocation's recorded size is borrowed
    only when its site executes once -- a returned allocation inside a loop has one dynamic instance per iteration
    and the output witnesses only the last, so earlier instances stay unresolved (return-instance-15).

    Reconciliation with an earlier change (statement unit classes, issue #59): ``only_path`` -- a Dim prefix ``[(dim, size), ...]``
    as recorded in ``loop_statement_classes`` -- switches the counting convention to "one trip per execution of the
    loop whose root chain is exactly that prefix, nothing else": ``_kernel_statement_class_totals`` /
    ``_kernel_gather_statement_trips`` size every class record with the SAME replay (registry state, refusals, first +
    steady) as the trip counter, so the class sums cannot drift from ``loop_iterations``."""
    import json as _json
    only_norm = _json.loads(_json.dumps([[d, s] for d, s in only_path])) if only_path is not None else None
    def _chain_norm(chain):
        return _json.loads(_json.dumps([[d, s] for d, s in chain]))
    nest = list(getattr(kernel, "loop_nest", None) or [])
    allocs = list(getattr(kernel, "local_tensor_dims", None) or [])
    dimregs = list(getattr(kernel, "dim_registrations", None) or [])
    returns = [r for r in (getattr(kernel, "returns", None) or []) if isinstance(r.get("line"), int)]
    ret_idx = _returned_allocation_index(kernel)
    out_dims = {k: int(v) for k, v in kernel.output.dims.items() if isinstance(v, int) and v > 0}
    scalar_args = getattr(kernel, "scalar_args", None)
    state, _sources, order_unknown = _input_registrations(kernel)
    for dname, sites in order_unknown.items():                  # legacy-kworder-12
        state[dname] = _order_unknown_reason(kernel, dname, sites)

    all_nodes: list = []
    def flatten(nodes):
        for n in nodes:
            all_nodes.append(n)
            flatten(n.get("children") or [])
    flatten(nest)

    def _pos(item):
        return item.get("line"), item.get("col")

    def _inside(p, node):
        """Is the statement at AST position ``p`` inside loop ``node``'s span?  With columns (graphs parsed since
        round 5) the span is ``(line, col) < p <= (end_line, end_col)``; a graph parsed before columns existed
        falls back to line containment INCLUDING the header line (an allocation never sits in a for-header)."""
        line, col = p
        if line is None:
            return False
        if col is None or node.get("col") is None or node.get("end_col") is None:
            return node["line"] <= line <= node.get("end", node["line"])
        return (node["line"], node["col"]) < (line, col) <= (node.get("end", node["line"]), node["end_col"])

    ret_in_loop = ret_idx is not None and any(_inside(_pos(allocs[ret_idx]), n) for n in all_nodes)

    def bind(state, dname, val, certain, line, what, spelled):
        """One registration of ``dname`` with ``val`` (an int, or None when not statically resolvable); ``what`` names
        the registering statement ("allocation" / "explicit Dim registration"), ``spelled`` its source form."""
        if not (isinstance(val, int) and val > 0):
            state[dname] = _Unresolved(f"registered by a local {spelled} at line {line} whose size is not statically "
                                       f"resolvable at this call" + ("" if val is None else f" (it evaluates to {val})")
                                       + ("" if certain else " (and whose execution is not certain)"))
            return
        val = int(val)
        if certain:
            state[dname] = val
        elif state.get(dname) != val:
            state[dname] = _Unresolved(f"may be re-registered by a conditional {what} at line {line} ({spelled}({dname}={val})) "
                                       f"while the established binding is {state.get(dname, 'absent')}; the binding at loop "
                                       f"entry is not statically certain")

    def register(state, i, alloc):
        certain = not alloc.get("conditional")
        line = alloc.get("line")
        for dname, spec in (alloc.get("dims") or {}).items():
            val = _eval_size_spec(spec, param_to_shape, binop_shapes, scalar_args)
            if val is None and i == ret_idx and dname in out_dims:
                if ret_in_loop:
                    # return-instance-15: the site runs once per iteration of its enclosing loop; the recorded
                    # output size is the LAST instance's and cannot stand for the earlier ones
                    state[dname] = _Unresolved(f"registered by the returned Tensor allocation at line {line}, which executes "
                                               f"once per iteration of its enclosing loop (a single instance when that loop "
                                               f"runs once); the recorded output size {out_dims[dname]} witnesses only the "
                                               f"returned instance and the size expression is not statically resolvable per "
                                               f"iteration")
                    continue
                val = out_dims[dname]                            # the recorded size of the RETURNED allocation (executed once)
            bind(state, dname, val, certain, line, "allocation", "Tensor")

    def register_dim(state, reg):
        dname, line = reg.get("dim"), reg.get("line")
        if dname is None:
            raise LoopDimExtentMismatch(
                "loop_dim_extent_unresolved",
                f"an explicit Dim(..., size=...) at line {line} of kernel {kernel.id!r} registers a dimension whose name is "
                f"not a string literal; its effect on the registry cannot be replayed")
        if reg.get("order_unknown"):
            # nested-dim-eval-25: the registration sits inside an expression whose evaluation order (or whether it
            # executes) the parser does not model -> the registry binding after it is not statically known; typed
            # refusal for every later loop over the name, naming the nested call -- never a silent skip
            state[dname] = _Unresolved(f"re-registered by the explicit Dim registration {reg.get('text') or 'Dim(...)'} at line "
                                       f"{line}, nested in {reg['order_unknown']} whose evaluation order the parser does not model")
            return
        spec = reg.get("size")
        missing = [n for n in _scalar_leaves(spec) if not isinstance((scalar_args or {}).get(n), int)
                   or isinstance((scalar_args or {}).get(n), bool)]
        if missing:
            raise LoopDimExtentMismatch(
                LOOP_DIM_SCALAR_UNRESOLVED,
                f"the explicit Dim({dname!r}, size=...) at line {line} of kernel {kernel.id!r} is sized by the scalar "
                f"argument(s) {', '.join(repr(n) for n in missing)}, which this trace did not record; its registration cannot be replayed")
        val = _eval_size_spec(spec, param_to_shape, binop_shapes, scalar_args)
        if val is None:
            # opaque-size-20 / HEADER-FALLBACK-19 (round 6): the registration's explicit size is unsupported or
            # unevaluable at this call -> the name is unresolved for every later loop over it (never the registry value)
            if _is_unbindable_spec(spec):
                why = str(spec[1])                            # dim-call-binding-23
            elif _is_opaque_spec(spec):
                why = f"explicit size expression not supported: {spec[1]}"
            elif _has_shape_arithmetic(spec) and binop_shapes is None:
                why = (f"the explicit size {_spec_text(spec)} is shape arithmetic this legacy trace (no signature input-order "
                       f"marker) cannot evaluate")
            else:
                why = f"the explicit size {_spec_text(spec)} is not statically resolvable at this call"
            state[dname] = _Unresolved(f"re-registered by the explicit Dim registration at line {line} ({why})")
            return
        bind(state, dname, val, not reg.get("conditional"), line, "explicit Dim registration", "Dim")

    def items_for(parent, children):
        """The statements replayed in ``parent``'s body (``None`` = the kernel's top level), in RUNTIME order: the
        child loops, and every allocation / explicit registration positioned inside the parent and not inside a child.
        nested-dim-eval-25 (round 8): a registration nested in a child loop's HEADER (``header_of``) is evaluated
        once when that ``for`` statement starts -- a statement of THIS body, replayed right before the loop, never per
        iteration; within one statement the parser's evaluation-order index ``eseq`` (arguments before the enclosing
        call, left to right) orders the registrations, so ``Dim("U", parallel=Dim("U", size=4), size=3)`` binds 4 then
        3 as the runtime does.  Graphs parsed without those keys keep the source-position order."""
        child_pos = {(c["line"], c.get("col")) for c in children}
        out = [("loop", c) for c in children]
        for kind, seq in (("alloc", allocs), ("dimreg", dimregs)):
            for i, a in enumerate(seq):
                hdr = a.get("header_of")
                if hdr is not None:
                    if (hdr[0], hdr[1]) in child_pos:
                        out.append((kind, i))
                    continue
                p = _pos(a)
                if p[0] is None:
                    if parent is None:
                        out.append((kind, i))                    # unplaceable: replayed first, at the top level
                    continue
                if parent is not None and not _inside(p, parent):
                    continue
                if parent is None and any(_inside(p, r) for r in nest):
                    continue
                if any(_inside(p, c) for c in children):
                    continue
                out.append((kind, i))
        def key(it):
            if it[0] == "loop":
                return (it[1]["line"], it[1].get("col") or 0, 1, 0)
            a = (allocs if it[0] == "alloc" else dimregs)[it[1]]
            hdr = a.get("header_of")
            if hdr is not None:
                return (hdr[0], hdr[1] or 0, 0, a.get("eseq") or 0)
            line = a.get("stmt_line", a.get("line")); col = a.get("stmt_col", a.get("col"))
            return (line if line is not None else -1, col or 0, 1, a.get("eseq") or 0)
        return sorted(out, key=key)

    def run_items(items, state, chain=()):
        inner = trips = outer = 0
        for kind, ref in items:
            if kind == "alloc":
                register(state, ref, allocs[ref])
            elif kind == "dimreg":
                register_dim(state, dimregs[ref])
            else:
                a, b, c = run_loop(ref, state, chain)
                inner += a; trips += b; outer += c
        return inner, trips, outer

    def run_loop(node, state, chain=()):
        dim, line, end = node["dim"], node["line"], node.get("end", node["line"])
        chain = tuple(chain) + ((dim, node.get("size")),)
        probe_hit = (only_norm is not None and _chain_norm(chain) == only_norm
                     and (only_line is None or node.get("line") == only_line))   # the producing loop, not a same-prefix sibling
        for r in returns:
            if r["line"] <= end:
                raise LoopDimExtentMismatch(
                    LOOP_ENTRY_UNRESOLVED,
                    f"the loop over Dim({dim!r}) at line {line} of kernel {kernel.id!r} "
                    + ("encloses a " if line <= r["line"] else "is preceded by a ")
                    + ("conditional " if r.get("conditional") else "")
                    + f"return at line {r['line']} (the kernel may end "
                    + ("mid-loop" if line <= r["line"] else "before the loop runs")
                    + "): whether, and how often, the loop runs is not provable from the source; refusing to count it")
        extent = int(resolve(dim, node.get("size"), state, line=node.get("line")))      # the witness of THIS loop's site (witness-site-6)
        if node.get("size") is not None:
            # dim-registration-17: ``for u in Dim("U", size=...)`` REGISTERS U before iterating, for every later
            # loop over the name.  ``resolve`` above has already REFUSED an explicit size this call cannot evaluate
            # (header-fallback-19, round 6: no registry / positional / assumed fallback for an explicit size --
            # an earlier revision's "one assumption, applied once" certified 3 where the runtime executed 9), so ``extent`` here
            # IS the evaluated explicit size and the registration carries exactly that value.
            state[dim] = extent
        body = items_for(node, node.get("children") or [])
        leaf = not node.get("children")
        inner = trips = outer = 0
        entries: list = []
        last = (0, 0, 0)
        done = 0
        while done < extent:
            snap = dict(state)
            if entries and snap == entries[-1]:
                k = extent - done                                # steady state: the rest repeats the last iteration
                inner += last[0] * k; trips += last[1] * k; outer += last[2] * k
                break
            if len(entries) >= _REPLAY_DISTINCT_STATES:
                raise LoopDimExtentMismatch(
                    "loop_dim_extent_unresolved",
                    f"the loop over Dim({dim!r}) at line {line} of kernel {kernel.id!r} starts {len(entries)} "
                    f"iterations in a row from different dim-registry states (allocations in its body keep "
                    f"re-binding names); no steady state to count from")
            entries.append(snap)
            if only_norm is not None:
                ci, ct, co = (1, 1, 0) if probe_hit else (0, 0, 0)      # class probe: trips of exactly this prefix
            else:
                ci, ct, co = (int(node.get("weight", 1)), 1, 0) if leaf else (0, 0, int(node.get("level_assigns", 0)))
            a, b, c = run_items(body, state, chain)
            last = (ci + a, ct + b, co + c)
            inner += last[0]; trips += last[1]; outer += last[2]
            done += 1
        return inner, trips, outer

    return run_items(items_for(None, nest), state)


def _kernel_loop_iterations(kernel: DSLKernel, dim_sizes: Optional[dict] = None) -> Optional[int]:
    """The kernel's actual iteration count — ground truth for op counting.

    The einsum equation can misrepresent the loop nest (derived indices like
    ``k // 128`` or ``n * D + d`` become independent dims; fused extra inputs
    such as residuals or block scales widen the index union), which inflates
    equation-derived op counts multiplicatively.

    Preferred source: ``kernel.loop_paths`` — the explicit ``for v in Dim(...)``
    structure parsed from the kernel source.  Iterations = sum over
    root-to-innermost paths of the product of loop-dim extents (explicit
    ``size=`` kwarg, else the graph-wide dim registry).

    Fallback: infer each loop variable's extent from a standalone index
    position in the loop body accesses.

    When explicit loop paths exist, every loop must be sized. An incomplete
    path is an error: standalone-index inference cannot recover loops whose
    indices appear only inside arithmetic expressions. For older metadata
    without loop paths, returns None when fallback inference is unavailable.
    """
    # -- Preferred: explicit Dim() loop structure from the kernel source --
    if kernel.loop_paths:
        dim_sizes = dim_sizes or {}
        # Resolve extents against THIS call site's own tensors first: a shared
        # kernel function (e.g. linear_bias called for q-proj with F=8192 and
        # for kv-proj with F=1024) must not be sized by the graph-wide max
        # registry, which counts every call at the largest caller's size.
        # Inputs first, then the OUTPUT overrides: the kernel's output is
        # authoritative for the output-space loop dims it defines.  A shared
        # kernel whose output reuses dim names (a conv producing B,O,OT,OH,OW
        # that feeds the next conv as input) would otherwise take the max of
        # this call's small output and the producer's larger same-named dims.
        # Reduction dims (only in inputs) are untouched by the override.
        local_sizes: dict = {}
        for inp in kernel.inputs:
            for dname, dsize in inp.tensor.dims.items():
                if isinstance(dsize, int) and dsize > 0:
                    local_sizes[dname] = max(local_sizes.get(dname, 0), dsize)
        for dname, dsize in kernel.output.dims.items():
            if isinstance(dsize, int) and dsize > 0:
                local_sizes[dname] = dsize
        param_to_shape = {
            name: inp.tensor.shape
            for name, inp in zip(kernel.param_names or [], kernel.inputs)
        }
        # Arithmetic size specs (new) may only use this map when the recorded
        # inputs provably correspond to the leading Tensor parameters: the
        # parser certified a clean tensor prefix, the saved trace carries the
        # "signature" input-order marker (legacy artifacts traced in call
        # order do not, so they keep the base behaviour), and the call
        # recorded exactly that many tensors.
        tpc = getattr(kernel, "tensor_param_count", None)
        binop_shapes = (param_to_shape
                        if tpc is not None
                        and getattr(kernel, "inputs_in_signature_order", False)
                        and len(kernel.inputs) == tpc
                        and len(kernel.param_names or []) >= tpc else None)
        # Positional extents: a loop var standing alone at axis i of a call
        # tensor pins its extent to THAT call's shape.  This is the most
        # call-accurate source — a shared kernel's signature dim name ("H")
        # often doesn't exist in the call's tensor dim names at all (the
        # caller's tensors are named R/Q), in which case name-based lookup
        # falls through to the graph-wide registry and sizes the loop by an
        # unrelated same-named dim (measured 4.67x on shared projections).
        # Loop vars conventionally lowercase their Dim name, so match
        # dim_name.lower() == var.
        var_extents: dict = {}
        # WITNESS_MODEL v2 (round 8, minimal): the narrow-access witness is kept PER GOVERNING LOOP -- keyed by the Dim loop
        # (entry line, dim name) that lexically encloses the access and whose variable spelling is bound ONLY as a Dim-loop
        # target in the whole kernel -- and compared only with that loop's own binding.  Every candidate component outside
        # the minimal predicate gets no witness and a typed ``witness_unmodelled:<first failing condition>`` note
        # (WITNESS_MODEL v3 §2, B′: kernel.witness_notes -> the informational ``witness_unmodelled`` / ``witness_coverage``
        # node fields ONLY -- a declined witness changes no count, no completeness, no reason, no basis).  A record whose
        # source segment cannot be placed witnesses NOTHING.  The pooled maximum (``var_extents``) is the positional
        # fallback for a name no binding covers and is unchanged.
        site_min: dict = {}
        site_max: dict = {}
        if kernel.loop_expr:
            _stmts = [s.strip() for s in kernel.loop_expr.split(';') if s.strip()]
            _accs = _parse_tensor_accesses('; '.join(_stmts))
            # The output shape witnesses the tensor the loop WRITES as its output -- the returned allocation -- never
            # the first tensor the body happens to access (P2-203-SCRATCH-EXTENT-1: an E-wide scratch accessed first
            # took the K-wide output shape and the E loop was refused as indexing a K-element axis), and only for an
            # access whose binding epoch IS the returned allocation's (witness-lifetime-1 / ACCESS-ALIAS-2): a
            # later allocation with the same spelling, or a plain alias of the returned allocation, is judged by what
            # the name denotes at that access, never by its spelling at the return.
            _witness = _access_witness_names(kernel)
            _notes: list = []
            kernel.witness_pass = _witness is not None                     # v3 §2: ``witness_coverage`` none (legacy record) vs full / partial
            if _witness is None:
                # WITNESS_MODEL v2 §3 -- a record without return information: the first tensor the body accesses takes the
                # output shape for the pooled-maximum FALLBACK only (the original convention); no site governor, no witness.
                _n2s = dict(param_to_shape)
                _first = _parse_tensor_accesses(_stmts[0]) if _stmts else []
                if _first:
                    _n2s.setdefault(_first[0][0], kernel.output.shape)
                for _si, _s in enumerate(_stmts):
                    for _name, _indices in _parse_tensor_accesses(_s):
                        _shape = _n2s.get(_name)
                        if not _shape or len(_indices) != len(_shape):
                            continue
                        for _pos, _idx in enumerate(_indices):
                            _iv = _idx.strip()
                            if re.fullmatch(r"[a-zA-Z_]\w*", _iv):
                                var_extents[_iv] = max(var_extents.get(_iv, 0), int(_shape[_pos]))
            else:
                _borrow, _shadowed = _access_witness_epochs(kernel, _stmts)
                _model = _witness_model_v2(kernel, _stmts)                     # WITNESS_MODEL v4 §1 / §2: eligible components and their lexical governors
                if _model is None and getattr(kernel, "loop_source_parsed", False):
                    _notes.append(f"{WITNESS_UNMODELLED}:segment_not_placed")
                _notes.extend(f"{_tag} (line {_line})" for _tag, _line in (getattr(kernel, "witness_uninspected_notes", None) or []))   # v4 coverage: accesses outside loop_expr
                for _si, _s in enumerate(_stmts):
                    def _shape_of(_name):
                        if _name in _borrow[_si]:
                            return kernel.output.shape
                        if _name in _shadowed[_si]:
                            return None                                            # a local whose identity at this access is not the returned allocation (or not proved)
                        return param_to_shape.get(_name)
                    if _model is None or _model[_si] is None:
                        if _model is not None:
                            _notes.append(f"{WITNESS_UNMODELLED}:statement_not_placed (statement {_si + 1})")
                        for _name, _indices in _parse_tensor_accesses(_s):             # no witness; the pooled maximum fallback only
                            _shape = _shape_of(_name)
                            if not _shape or len(_indices) != len(_shape):
                                continue
                            for _pos, _idx in enumerate(_indices):
                                _iv = _idx.strip()
                                if re.fullmatch(r"[a-zA-Z_]\w*", _iv):
                                    var_extents[_iv] = max(var_extents.get(_iv, 0), int(_shape[_pos]))
                        continue
                    _accesses, _tags = _model[_si]
                    _notes.extend(f"{_tag} (line {_line})" for _tag, _line in _tags)
                    for _name, _comps in _accesses:
                        _shape = _shape_of(_name)
                        if not _shape or len(_comps) != len(_shape):
                            continue
                        for _pos, (_sp, _gov) in enumerate(_comps):
                            if not _sp:
                                continue
                            _ext = int(_shape[_pos])
                            var_extents[_sp] = max(var_extents.get(_sp, 0), _ext)
                            if _gov is not None:
                                site_min[_gov] = min(site_min.get(_gov, _ext), _ext)              # (entry line, dim name) of the lexically innermost Dim loop over this spelling
                                site_max[_gov] = max(site_max.get(_gov, 0), _ext)
            kernel.witness_notes = sorted(set(_notes)) if _notes else None
        def _witness_bounds(dim_name, line):
            """(narrowest, widest) bare-access extents witnessed for the loop over *dim_name* entered at *line* -- that loop's
            own site only.  An unplaceable segment, a legacy record or a record without loop lines witnesses nothing: the
            count then rests on the registry binding alone."""
            if line is None:
                return None, None
            return site_min.get((line, dim_name)), site_max.get((line, dim_name))
        # Runtime bindings come FIRST (TARGET-DSL-DIM-EXTENT-13, annotation-binding-1,
        # TARGET-DSL-DIM-LOCAL-2): a loop bound to Dim("X") iterates the dimension X
        # the kernel's registry holds at this call -- the arguments' actual dim
        # names and the body's Tensor allocations, never a parameter annotation
        # and never the widest positional access that shares the index letter.
        # The positional extents above remain a fallback for a name no binding
        # covers, and a witness: a bare access of the loop variable on an axis
        # narrower than the bound extent is an inconsistency, flagged, never a max.
        path_lines = list(getattr(kernel, "loop_path_lines", None) or [])
        outer_lines = list(getattr(kernel, "loop_outer_lines", None) or [])

        scalar_args = getattr(kernel, "scalar_args", None)

        def _explicit_extent(dim_name, spec):
            """The loop's explicit ``size=``, which ALWAYS takes precedence over a registry binding of the same name
            (scalar-extent-18): None only when the loop has NO explicit size (the registry binding then
            applies); else a positive int, or a typed refusal -- an explicit size in a form the parser does not
            support (``loop_dim_extent_unresolved``, opaque-size-20: a helper call, ``*args`` / ``**kwargs``
            subscripts ...), a scalar leaf the trace did not record (``loop_dim_scalar_unresolved``, whether or not
            the registry happens to bind the name: that binding is unrelated to the explicit size), a supported
            expression this call cannot EVALUATE (``loop_dim_extent_unresolved``, header-fallback-19: shape
            arithmetic on a legacy trace without the signature input-order marker, a parameter the call did not
            record as a tensor input -- never the registry binding, the positional fallback or any assumed value),
            or a non-positive value (a zero-trip loop is not certified; the count must never fall through to an
            unrelated extent)."""
            if spec is None:
                return None
            if _is_unbindable_spec(spec):
                # dim-call-binding-23: the header's arguments cannot be bound to Dim(name, parallel=False, size=None)
                # (``**{"size": 3}``, ``*args``): the runtime may register a size the parser cannot see -- refused, never "absent"
                raise LoopDimExtentMismatch(
                    "loop_dim_extent_unresolved",
                    f"the loop over Dim({dim_name!r}) of kernel {kernel.id!r}: {spec[1]} (line {spec[2]}); the call may pass "
                    f"an explicit size the parser cannot see, so the loop is not sized from the registry binding of the name. "
                    f"Bindable forms: a positional size (third argument), size=<expr>, or no size at all")
            if _is_opaque_spec(spec):
                raise LoopDimExtentMismatch(
                    "loop_dim_extent_unresolved",
                    f"the loop over Dim({dim_name!r}) of kernel {kernel.id!r} has an explicit size expression not "
                    f"supported: {spec[1]} (line {spec[2]}); an explicit size is never read from the registry binding "
                    f"of the same name. Supported forms: an int literal, a scalar parameter recorded by the trace, "
                    f"<tensor parameter>.shape[i] (and integer arithmetic over it when the trace carries the signature "
                    f"input-order marker), or a local name bound exactly once to one of those")
            missing = [n for n in _scalar_leaves(spec) if not isinstance((scalar_args or {}).get(n), int)
                       or isinstance((scalar_args or {}).get(n), bool)]
            if missing:
                # TARGET-KB-MAXPOOL (round 4) / SCALAR-EXTENT-18 (round 5): the size names a scalar argument this trace did not record
                raise LoopDimExtentMismatch(
                    LOOP_DIM_SCALAR_UNRESOLVED,
                    f"Cannot count kernel {kernel.id!r}: the loop dimension {dim_name!r} is sized by the scalar "
                    f"argument(s) {', '.join(repr(n) for n in missing)} (Dim({dim_name!r}, size=...)), which this trace "
                    f"did not record (traced before int arguments and defaults were recorded as `scalar_args`, or passed "
                    f"as a non-int). Expose its extent as a call-local tensor dimension or retrace with the current "
                    f"einsum_lib so the call's value is in the metadata.")
            val = _eval_size_spec(spec, param_to_shape, binop_shapes, scalar_args)
            if val is None:
                # header-fallback-19 (round 6, manager ruling): an explicit size this call cannot evaluate is a
                # typed refusal -- not the registry binding, not the positional fallback, not "the count the loop was
                # counted with" (an earlier revision certified 3 where the runtime executed 9)
                if _has_shape_arithmetic(spec) and binop_shapes is None:
                    why = ("integer arithmetic over a tensor shape can only be evaluated when the saved trace carries the "
                           "signature input-order marker; this trace does not (a legacy trace -- retrace with the current "
                           "einsum_lib so the call's input order is recorded)")
                else:
                    missing_params = sorted({s[1] for s in _shape_leaves(spec) if s[1] not in param_to_shape})
                    why = (f"the parameter(s) {', '.join(repr(p) for p in missing_params)} are not recorded tensor inputs of this call"
                           if missing_params else "a leaf of the expression does not resolve at this call")
                raise LoopDimExtentMismatch(
                    "loop_dim_extent_unresolved",
                    f"the explicit size {_spec_text(spec)} of the loop over Dim({dim_name!r}) of kernel {kernel.id!r} cannot "
                    f"be evaluated at this call: {why}; the loop is not sized from the registry binding of the name")
            if val <= 0:
                raise LoopDimExtentMismatch(
                    "loop_dim_extent_unresolved",
                    f"the explicit size of the loop over Dim({dim_name!r}) of kernel {kernel.id!r} evaluates to {val} at this "
                    f"call; a loop with no trips (or a negative size) is not certified from the source, and the count must "
                    f"not fall through to the registry binding of the same name")
            return int(val)

        def _pick_extent(dim_name, explicit, bound, refusal, line=None):
            # explicit size, else the registry binding (with the narrow-access witness of THIS loop's site), else the positional fallback
            explicit = _explicit_extent(dim_name, explicit)
            if explicit is not None:
                return explicit
            if refusal is not None:
                raise LoopDimExtentMismatch(getattr(refusal, "code", "loop_dim_extent_unresolved"),
                                            f"Dim({dim_name!r}) of kernel {kernel.id!r} is {refusal}")
            if bound is not None:
                narrowest, widest = _witness_bounds(dim_name, line)
                if narrowest is not None and narrowest < bound:
                    raise LoopDimExtentMismatch(
                        "loop_dim_extent_mismatch",
                        f"Dim({dim_name!r}) of kernel {kernel.id!r} is bound to extent {bound} but its loop "
                        f"variable indexes an axis of only {narrowest} elements (accessed extents "
                        f"{narrowest}..{widest})")
                return bound
            extent = (var_extents.get(dim_name.lower())
                      or local_sizes.get(dim_name)
                      or dim_sizes.get(dim_name))
            if not extent or extent <= 0:
                raise ValueError(
                    f"Cannot count kernel {kernel.id!r}: unresolved or "
                    f"non-positive loop dimension {dim_name!r}. "
                    "Expose its extent as a call-local tensor dimension "
                    "or an explicitly resolvable Dim(size=...) and retrace. "
                    "Refusing partial standalone-index fallback because "
                    "it can silently omit outer loops.")
            return int(extent)

        if getattr(kernel, "loop_nest", None):
            # Round 4 (loop-carried-10 / RETURN-REBIND-9 / EARLY-RETURN-11 / LEGACY-KWORDER-12): the registry
            # replayed through the nest iteration by iteration; the flat per-level path below serves graphs parsed
            # before the tree existed.
            def _resolve(dim_name, explicit, state, line=None):
                bound = state.get(dim_name)
                if isinstance(bound, _Unresolved):
                    return _pick_extent(dim_name, explicit, None, bound, line=line)
                return _pick_extent(dim_name, explicit, bound if isinstance(bound, int) else None, None, line=line)
            _probe_path = getattr(kernel, "class_probe_path", None)       # an earlier change class totals: size ONE Dim prefix by the same replay
            total, trips, outer_total = _replay_loop_nest(kernel, param_to_shape, binop_shapes, _resolve, _explicit_extent,
                                                          only_path=_probe_path, only_line=getattr(kernel, "class_probe_line", None))
            if _probe_path is not None:
                return (int(total), int(trips), "paths", int(outer_total)) if total > 0 else None
            if total > 0:
                return (int(total), int(trips), "paths", int(outer_total))
        _bind_cache: dict = {}
        def _bindings_at(line):
            if line not in _bind_cache:
                _bind_cache[line] = _runtime_dim_bindings(kernel, param_to_shape, binop_shapes, before_line=line)
            return _bind_cache[line]
        def _prefix_product(path, lines=None) -> int:
            # ``lines``: the entry line of EACH loop along the path (round 3); a legacy int applies to every level
            if isinstance(lines, int):
                lines = [lines] * len(path)
            prod = 1
            for level, (dim_name, explicit) in enumerate(path):
                line = lines[level] if isinstance(lines, (list, tuple)) and level < len(lines) else None
                bound_extents, bound_conflicts, bound_unresolved = _bindings_at(line)
                explicit = _explicit_extent(dim_name, explicit)      # round 5: precedence, missing scalars, zero
                if explicit is not None:
                    extent = explicit
                elif dim_name in bound_conflicts:
                    raise LoopDimExtentMismatch(
                        "loop_dim_extent_mismatch",
                        f"Dim({dim_name!r}) of kernel {kernel.id!r} is bound to different sizes at this call: "
                        + ", ".join(f"{site}={ext}" for site, ext in bound_conflicts[dim_name]))
                elif dim_name in bound_extents:
                    extent = bound_extents[dim_name]
                    narrowest, widest = _witness_bounds(dim_name, line)
                    if narrowest is not None and narrowest < extent:
                        raise LoopDimExtentMismatch(
                            "loop_dim_extent_mismatch",
                            f"Dim({dim_name!r}) of kernel {kernel.id!r} is bound to extent {extent} but its loop "
                            f"variable indexes an axis of only {narrowest} elements (accessed extents "
                            f"{narrowest}..{widest})")
                elif dim_name in bound_unresolved:
                    raise LoopDimExtentMismatch(
                        getattr(bound_unresolved[dim_name], "code", "loop_dim_extent_unresolved"),
                        f"Dim({dim_name!r}) of kernel {kernel.id!r} is {bound_unresolved[dim_name]}")
                else:
                    extent = (var_extents.get(dim_name.lower())
                              or local_sizes.get(dim_name)
                              or dim_sizes.get(dim_name))
                if not extent or extent <= 0:
                    raise ValueError(
                        f"Cannot count kernel {kernel.id!r}: unresolved or "
                        f"non-positive loop dimension {dim_name!r}. "
                        "Expose its extent as a call-local tensor dimension "
                        "or an explicitly resolvable Dim(size=...) and retrace. "
                        "Refusing partial standalone-index fallback because "
                        "it can silently omit outer loops.")
                prod *= int(extent)
            return prod

        total = 0
        trips = 0
        for idx, entry in enumerate(kernel.loop_paths):
            # (path, weight) — older graphs may carry bare paths
            path, weight = entry if (len(entry) == 2 and isinstance(entry[1], int)
                                     and not isinstance(entry[0], str)) else (entry, 1)
            prod = _prefix_product(path, path_lines[idx] if idx < len(path_lines) else None)
            total += prod * weight
            trips += prod
        # Assignments at non-innermost loop levels execute once per trip of
        # their own level (statement-coverage-1): sized with the same
        # extents, kept apart from the innermost (MAC-bearing) count.
        outer_total = 0
        for idx, entry in enumerate(getattr(kernel, "loop_outer_statements", None) or []):
            path, count = entry
            outer_total += _prefix_product(path, outer_lines[idx] if idx < len(outer_lines) else None) * int(count)
        if total > 0:
            return (int(total), int(trips), "paths", int(outer_total))

    # -- Fallback: standalone-index inference from the loop body --
    if not kernel.loop_expr:
        return None
    stmts = [s.strip() for s in kernel.loop_expr.split(';') if s.strip()]
    if not stmts:
        return None
    all_accesses = _parse_tensor_accesses('; '.join(stmts))
    if not all_accesses:
        return None
    loop_vars = _identify_loop_vars(all_accesses)
    # Exclude locally-assigned variables: gather values (``e = SE[n, k]``)
    # AND arithmetic index helpers (``lo = n * hd + d`` in double-write RoPE).
    # They are data/derived indices, not loop trip counts; treating them as
    # loops multiplies the indexed tensor's full extent into the work
    # (measured 1e4x+ inflations).  A tensor-element write ``Y[b, s] = ...``
    # does not match (the '[' follows the name before '=').
    assigned_vars = {
        m.group(1)
        for s in stmts
        for m in [re.match(r"\s*([A-Za-z_]\w*)\s*=[^=]", s)]
        if m
    }
    loop_vars -= assigned_vars
    if not loop_vars:
        return None

    name_to_shape = {}
    for pname, inp in zip(kernel.param_names or [], kernel.inputs):
        name_to_shape[pname] = inp.tensor.shape
    first = _parse_tensor_accesses(stmts[0])
    if first:
        # LHS of the first statement is the output tensor's local name.
        name_to_shape.setdefault(first[0][0], kernel.output.shape)

    extents: dict = {}
    for name, indices in all_accesses:
        shape = name_to_shape.get(name)
        if not shape or len(indices) != len(shape):
            continue
        for pos, idx in enumerate(indices):
            iv = idx.strip()
            if iv in loop_vars:
                extents[iv] = max(extents.get(iv, 0), int(shape[pos]))
    if set(extents) != set(loop_vars) or any(v <= 0 for v in extents.values()):
        return None
    # Sum per-statement products, NOT the product of the union: sibling loop
    # nests (3-pass layernorm/softmax: mean stmt, var stmt, apply stmt) execute
    # sequentially — multiplying all their loop vars together inflated LN-class
    # kernels by |D|^2 (measured up to 7500x).  A statement's iterations are
    # the product of the loop vars appearing in ITS OWN accesses.
    total = 0
    for s in stmts:
        accs = _parse_tensor_accesses(s)
        svars = set()
        for _, indices in accs:
            for idx in indices:
                for v in _vars_in_index(idx, loop_vars):
                    svars.add(v)
        if not svars:
            continue
        prod = 1
        for v in svars:
            prod *= extents[v]
        total += prod
    return (int(total), int(total), "fallback", 0) if total > 0 else None


def build_kernel_einsum_from_loop_expr(kernel: DSLKernel) -> Optional[str]:
    """Build einsum equation from loop_expr with compound dim support.

    Parses tensor accesses in the loop expression to detect compound indexing
    like 'p * stride + r' and generate '(P+R)' notation.

    Returns equation string like 'BC(P+R)(Q+S),OCRS,O->BOPQ' or None on failure.
    """
    if not kernel.loop_expr or not kernel.param_names:
        return None

    stmts = [s.strip() for s in kernel.loop_expr.split(';')]

    # Find the primary accumulation statement (prefer +=, else last =)
    primary_stmt = None
    for stmt in stmts:
        if '+=' in stmt:
            primary_stmt = stmt
            break
    if primary_stmt is None:
        primary_stmt = stmts[-1] if stmts else None
    if not primary_stmt:
        return None

    # Collect all tensor accesses across all statements for variable discovery
    all_text = '; '.join(stmts)
    all_accesses = _parse_tensor_accesses(all_text)
    if not all_accesses:
        return None

    loop_vars = _identify_loop_vars(all_accesses)
    if not loop_vars:
        return None

    # Parse the primary statement
    primary_accesses = _parse_tensor_accesses(primary_stmt)
    if not primary_accesses:
        return None

    # LHS is the output tensor
    output_name = primary_accesses[0][0]
    output_indices = primary_accesses[0][1]

    # Collect RHS tensor accesses (name → indices), deduped
    rhs_accesses = {}
    for name, indices in primary_accesses[1:]:
        if name != output_name and name not in rhs_accesses:
            rhs_accesses[name] = indices
    # Also pick up inputs from non-primary statements (e.g. bias init)
    for stmt in stmts:
        if stmt == primary_stmt:
            continue
        for name, indices in _parse_tensor_accesses(stmt):
            if name != output_name and name not in rhs_accesses:
                rhs_accesses[name] = indices

    # Build ordered variable list: output vars first, then input vars by appearance
    seen = set()
    ordered_vars = []
    for idx in output_indices:
        for v in _vars_in_index(idx, loop_vars):
            if v not in seen:
                ordered_vars.append(v)
                seen.add(v)
    for _, indices in primary_accesses[1:]:
        for idx in indices:
            for v in _vars_in_index(idx, loop_vars):
                if v not in seen:
                    ordered_vars.append(v)
                    seen.add(v)
    for stmt in stmts:
        for _, indices in _parse_tensor_accesses(stmt):
            for idx in indices:
                for v in _vars_in_index(idx, loop_vars):
                    if v not in seen:
                        ordered_vars.append(v)
                        seen.add(v)

    var_to_token = _build_var_to_token(ordered_vars)

    output_tokens = _access_to_operand(output_indices, loop_vars, var_to_token,
                                       tensor=kernel.output)
    if output_tokens is None:
        return None  # token count mismatch with output ndim

    # Build input operands in kernel parameter order, using tensor dim names to
    # resolve literal-int indices (e.g. W[c, 0, r, s] on tensor with dims {C,g,R,S}).
    input_parts = []
    param_to_tensor = {name: inp.tensor for name, inp in zip(kernel.param_names, kernel.inputs)}
    for param_name in kernel.param_names:
        if param_name in rhs_accesses:
            tensor = param_to_tensor.get(param_name)
            tokens = _access_to_operand(rhs_accesses[param_name], loop_vars, var_to_token,
                                        tensor=tensor)
            if tokens is None:
                return None  # token count mismatch with input ndim
            input_parts.append(tokens)
        else:
            return None  # param not found in loop_expr, can't build

    return ','.join(input_parts) + '->' + output_tokens


def build_kernel_einsum(kernel: DSLKernel) -> str:
    """Derive the einsum equation for a kernel from its tensor dims.

    The equation format follows solar convention: inputs separated by commas,
    arrow, then output. Tokens are [A-Z][0-9]* derived from dim names.

    Example for linear_q_proj (dims B,S,H and H2,H → B,S,H2):
        BSH,H2H->BSH2

    Example for attention_scores (Q: B,Nh,S,D  K: B,Nh,Sk,D  → B,Nh,S,Sk):
        With B→B, Nh→N, S→S, D→D, Sk→S0:
        BNSD,BNS0D->BNSS0
    """
    # Build dim map over all tensors in this kernel (input order matters for determinism)
    all_tensors = [inp.tensor for inp in kernel.inputs] + [kernel.output]
    dim_map = build_dim_map(all_tensors)

    input_parts = [_tensor_tokens(inp.tensor, dim_map) for inp in kernel.inputs]
    output_part = _tensor_tokens(kernel.output, dim_map)

    return ",".join(input_parts) + "->" + output_part


def build_start_einsum(graph: DSLGraph) -> str:
    """Derive the start node einsum (->XYZ) from the model input tensors."""
    model_inputs = graph.model_inputs
    if not model_inputs:
        return "->A"

    # Build dim map over all model input tensors
    all_tensors = model_inputs
    dim_map = build_dim_map(all_tensors)

    output_part = "".join(
        _tensor_tokens(t, dim_map) for t in model_inputs
    )
    return "->" + output_part


# ---------------------------------------------------------------------------
# Elementwise / reduction op inference from loop expression
# ---------------------------------------------------------------------------

# The exporter's own body-inference vocabulary (the codomain of _infer_ops for
# a kernel WITH a body).  The graph analyzer imports these to decide whether a
# pre-marker (legacy) record's ops prove that a body existed at export time
# (legacy-schema-14): a reduction token "add"/"sub" is only produced by
# an augmented assignment, "none" only together with an inferred elementwise
# op; a bodiless kernel gets (None, None).  Any other string is not evidence.
BODY_REDUCTION_TOKENS = frozenset({"add", "sub", "none"})
BODY_ACCUMULATION_TOKENS = frozenset({"add", "sub"})
BODY_ELEMENTWISE_TOKENS = frozenset({"mul", "add", "exp", "max", "copy"})


# Value-expression grammar of a kernel body statement (legacy-parse-15):
# arithmetic over indexed tensor accesses, loop variables, local scalars and
# numeric constants, with plain positional calls (``max``, ``math.exp``) and
# conditional expressions.  No strings, containers, lambdas, comprehensions,
# keyword/starred arguments, matrix products or attribute access on anything
# but a dotted name.
_BODY_VALUE_NODES = (ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Name, ast.Constant, ast.Call,
                     ast.IfExp, ast.Compare, ast.BoolOp, ast.Attribute, ast.Tuple, ast.Slice)
_BODY_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
                ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)


def _dotted_name(node) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _body_value_ok(node) -> bool:
    """True when *node* is an expression the body grammar admits (recursive)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _dotted_name(node)
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and _body_value_ok(node.slice)
    if isinstance(node, ast.Tuple):  # multi-index subscript
        return all(_body_value_ok(e) for e in node.elts)
    if isinstance(node, ast.Slice):
        return all(e is None or _body_value_ok(e) for e in (node.lower, node.upper, node.step))
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, _BODY_BINOPS) and _body_value_ok(node.left) and _body_value_ok(node.right)
    if isinstance(node, ast.UnaryOp):
        return _body_value_ok(node.operand)
    if isinstance(node, ast.BoolOp):
        return all(_body_value_ok(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return _body_value_ok(node.left) and all(_body_value_ok(c) for c in node.comparators)
    if isinstance(node, ast.IfExp):
        return _body_value_ok(node.test) and _body_value_ok(node.body) and _body_value_ok(node.orelse)
    if isinstance(node, ast.Call):
        return (_dotted_name(node.func) and not node.keywords
                and all(not isinstance(a, ast.Starred) and _body_value_ok(a) for a in node.args))
    return False


def loop_expr_is_body(expr) -> bool:
    """True when *expr* is a string that PARSES as a kernel loop body.

    Syntax-aware validation without execution (legacy-parse-15):
    ``ast.parse`` must succeed (a ``SyntaxError``/``ValueError`` is not a
    body); the module must contain at least one statement (a comment-only
    string has none); EVERY statement must be an ``ast.Assign`` or
    ``ast.AugAssign`` whose target(s) are indexed accesses (``ast.Subscript``
    on a plain tensor name) and whose value is admitted by the body grammar
    (``_body_value_ok``: no strings, containers, lambdas, keyword arguments,
    call statements, comparisons-as-statements ...).  Only then is the
    exporter's own access extractor consulted: the statements must carry at
    least one standalone loop-variable index (``_parse_tensor_accesses`` /
    ``_identify_loop_vars``), so ``Y[0] = X[0]`` and ``Y = X`` are not bodies
    while ``Y[0] += X[h] * W[h]`` is.  Total over saved-artifact value types:
    non-strings and empty strings are ``False``, never an exception.
    """
    if not isinstance(expr, str) or not expr.strip():
        return False
    try:
        tree = ast.parse(expr, mode="exec")
    except (SyntaxError, ValueError):
        return False
    if not tree.body:
        return False
    for st in tree.body:
        if isinstance(st, ast.Assign):
            targets, value = st.targets, st.value
        elif isinstance(st, ast.AugAssign):
            if not isinstance(st.op, _BODY_BINOPS):
                return False
            targets, value = [st.target], st.value
        else:
            return False
        if not targets or not all(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                  and _body_value_ok(t.slice) for t in targets):
            return False
        if not _body_value_ok(value):
            return False
    # Loop-variable evidence from the VALIDATED AST only (legacy-comment-18):
    # a standalone identifier index of any tensor access in the statements,
    # under the exporter's identifier rule (ASCII ``[a-zA-Z_][a-zA-Z0-9_]*``,
    # the same class ``_identify_loop_vars`` accepts).  Comments and other
    # text outside the statements cannot contribute an index.
    return bool(_ast_loop_vars(tree))


_LOOP_VAR_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _ast_loop_vars(tree) -> set:
    """Standalone identifier indices of every ``Name[...]`` access in *tree*."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for e in elts:
                if isinstance(e, ast.Name) and _LOOP_VAR_RE.fullmatch(e.id):
                    found.add(e.id)
    return found


def ops_prove_body(elementwise_op, reduction_op) -> bool:
    """True when an exported (elementwise_op, reduction_op) pair can only have
    come from ``_infer_ops`` of a real loop body: an accumulation token, or
    ``"none"`` together with an inferred elementwise token.

    Total over saved-artifact value types (legacy-token-16): the type
    is checked BEFORE any membership test, so containers, bytes, numbers,
    ``None`` and unknown strings are ``False`` and nothing can raise.
    ``(None, None)`` (bodiless), ``""``, ``"null"``, ``"ADD"`` -> False."""
    if not isinstance(reduction_op, str):
        return False
    if reduction_op in BODY_ACCUMULATION_TOKENS:
        return True
    return (reduction_op == "none" and isinstance(elementwise_op, str)
            and elementwise_op in BODY_ELEMENTWISE_TOKENS)


def _kernel_gather_statement_trips(kernel: DSLKernel, dim_sizes: Optional[dict] = None):
    """Executed trips of the statements that address a tensor through a declared
    GATHER index (``loop_statement_class_gather``, parallel to the classes) plus
    the executed evaluations of statement-``if`` TESTS that read through a gather
    (``loop_gather_controls``, one per trip of the ``if``), summed with the class
    totals' extent rules; None when the kernel records none."""
    import copy as _copy
    classes = getattr(kernel, "loop_statement_classes", None) or []
    gather = getattr(kernel, "loop_statement_class_gather", None) or []
    controls = getattr(kernel, "loop_gather_controls", None) or []
    if not kernel.loop_paths or (not controls and not (classes and gather)):
        return None
    if gather and len(gather) != len(classes):
        return None
    probe = _copy.copy(kernel)
    probe.loop_outer_statements = None
    cls_lines = list(getattr(kernel, "loop_statement_class_lines", None) or []) or [None] * len(classes)
    ctl_lines = list(getattr(kernel, "loop_gather_control_lines", None) or []) or [None] * len(controls)
    if len(cls_lines) != len(classes) or len(ctl_lines) != len(controls):
        return None
    sites = [(path, ln) for (path, _kind, _extra), g, ln in zip(classes, gather, cls_lines) if g] + [(list(p), ln) for p, ln in zip(controls, ctl_lines)]
    if not sites:
        return None
    total = 0
    for path, ln in sites:
        probe.loop_paths = [(list(path), 1)]
        probe.class_probe_path = list(path)                  # the loop-nest replay sizes exactly this prefix (an earlier change reconciliation)
        probe.class_probe_line = ln                          # ... at exactly the loop that produced the record (sibling nests share prefixes)
        li = _kernel_loop_iterations(probe, dim_sizes)
        if not li or li[2] != "paths":
            return None
        total += int(li[1])
    return total


def _kernel_singleton_statement_trips(kernel: DSLKernel, dim_sizes: Optional[dict] = None):
    """Executed singleton-scale read evaluations (ISSUE61-SINGLETON-RECEIPT-4, the gather receipt's
    algebra): the trips of the statements flagged in ``loop_statement_class_singleton`` (parallel to
    the classes) plus the trips of the statement-``if`` TESTS that read a singleton
    (``loop_singleton_controls``, one evaluation per trip of the ``if``), summed with the class
    totals' extent rules, plus the reads outside every loop, once per call
    (``loop_singleton_outer_reads``); None when the kernel records none."""
    import copy as _copy
    classes = getattr(kernel, "loop_statement_classes", None) or []
    single = getattr(kernel, "loop_statement_class_singleton", None) or []
    controls = getattr(kernel, "loop_singleton_controls", None) or []
    outer = int(getattr(kernel, "loop_singleton_outer_reads", None) or 0)
    if single and len(single) != len(classes):
        return None
    cls_lines = list(getattr(kernel, "loop_statement_class_lines", None) or []) or [None] * len(classes)
    ctl_lines = list(getattr(kernel, "loop_singleton_control_lines", None) or []) or [None] * len(controls)
    if len(cls_lines) != len(classes) or len(ctl_lines) != len(controls):
        return None
    sites = [(path, ln) for (path, _kind, _extra), s, ln in zip(classes, single, cls_lines) if s] + [(list(p), ln) for p, ln in zip(controls, ctl_lines)]
    if not sites and not outer:
        return None
    if sites and not kernel.loop_paths:
        return None
    probe = _copy.copy(kernel)
    probe.loop_outer_statements = None
    total = outer
    for path, ln in sites:
        probe.loop_paths = [(list(path), 1)]
        probe.class_probe_path = list(path)                  # the loop-nest replay sizes exactly this prefix (an earlier change reconciliation)
        probe.class_probe_line = ln                          # ... at exactly the loop that produced the record (sibling nests share prefixes)
        li = _kernel_loop_iterations(probe, dim_sizes)
        if not li or li[2] != "paths":
            return None
        total += int(li[1])
    return total


def _singleton_reads_proven(kernel: DSLKernel):
    """The EXTENT half of the SINGLETON-SCALE-READ production (issue #61): every
    candidate read (``loop_singleton_reads``: parameter -> declared rank) is proven
    from the call site's DECLARED SHAPE BINDING -- the recorded input tensor in the
    parameter's position, which requires the trace's signature input order and the
    parser's clean tensor prefix -- to have exactly that rank with every dim extent
    1.  Returns ``(proven, reason)``; an unproven read names its binding."""
    reads = getattr(kernel, "loop_singleton_reads", None) or {}
    if not reads:
        return True, ""
    names = list(kernel.param_names or [])
    tpc = getattr(kernel, "tensor_param_count", None)
    problems = []
    if not getattr(kernel, "inputs_in_signature_order", False) or tpc is None or len(kernel.inputs) != tpc:
        return False, ("singleton read of %s not proven: the call's inputs are not bound to the signature "
                       "(signature input order and a clean tensor prefix are required)" % ", ".join(sorted(reads)))
    for name in sorted(reads):
        rank = int(reads[name])
        if name not in names or names.index(name) >= tpc:
            problems.append(f"singleton read of {name} not proven: not a bound input tensor")
            continue
        dims = list(kernel.inputs[names.index(name)].tensor.dims.items())
        if len(dims) != rank or any(not (isinstance(v, int) and not isinstance(v, bool) and v == 1) for _, v in dims):
            problems.append(f"singleton read {name}[{', '.join(['0'] * rank)}] not proven: bound extent {dict(dims)} is not a rank-{rank} singleton")
    return (not problems), "; ".join(problems)


def _kernel_statement_class_totals(kernel: DSLKernel, dim_sizes: Optional[dict] = None):
    """``(mac_statements, other_statements, extra_multiplies)`` executed by the kernel: per
    recorded assignment (``loop_statement_classes``) the product of its Dim()
    prefix extents, summed per class; a MAC statement with extra value
    factors adds one standalone multiply per trip per extra factor.

    The extent of each prefix is resolved by ``_kernel_loop_iterations``
    itself, called on a copy of the kernel whose only loop path is that
    prefix (weight 1, no outer statements) -- so the class totals use exactly
    the trip counter's extent rules and cannot drift from them.  None when
    the kernel has no parsed statement classes (builtin, legacy artifact) or
    a prefix cannot be sized."""
    import copy as _copy
    classes = getattr(kernel, "loop_statement_classes", None)
    if not classes or not kernel.loop_paths:
        return None
    probe = _copy.copy(kernel)
    probe.loop_outer_statements = None
    cls_lines = list(getattr(kernel, "loop_statement_class_lines", None) or []) or [None] * len(classes)
    if len(cls_lines) != len(classes):
        return None
    mac = other = extras = 0
    for entry, ln in zip(classes, cls_lines):
        path, kind, extra = entry
        probe.loop_paths = [(list(path), 1)]
        probe.class_probe_path = list(path)                  # the loop-nest replay sizes exactly this prefix (an earlier change reconciliation)
        probe.class_probe_line = ln                          # ... at exactly the loop that produced the record (sibling nests share prefixes)
        li = _kernel_loop_iterations(probe, dim_sizes)
        if not li or li[2] != "paths":
            return None
        trips = int(li[1])
        if kind == "mac":
            mac += trips
            other += trips * int(extra or 0)
            extras += trips * int(extra or 0)
        else:
            other += trips
    return (int(mac), int(other), int(extras))


def _kernel_mac_operand_records(kernel: DSLKernel, dim_sizes: Optional[dict] = None):
    """The MAC statements of the kernel grouped by the operands they contract, each group's trips summed with the
    class totals' extent rules (one-path probes through ``_kernel_loop_iterations``).  Returns two parallel lists
    or None (legacy artifact):

    * the RECEIPT ``loop_mac_operands``: ``[{"macs", "payloads": [{name, input, dtype}] (one per payload
      OCCURRENCE), "scales": [...] (one per scale occurrence), "conversions": [target per occurrence],
      "operands": {name: declared dtype}, "unproven"?, "arms_differ"?}, ...]`` -- ``input`` is the operand's
      POSITION among the kernel inputs (the graph node's ``tensor_dtypes.inputs`` order, None for a local
      allocation) and ``dtype`` its DECLARED logical dtype;
    * the OCCURRENCE CHANNEL ``loop_mac_reads``: ``[{"macs", "occurrences": [{name, input, index, role}] (the
      parser's recorded tensor-element reads of the MAC statement, in source order), "conversions", "names"}, ...]``
      -- carries no dtypes; the analyzer reconciles every receipt entry against it by name AND position AND role
      and never takes the receipt's own map as the sole binding evidence (DTYPE-EVIDENCE-BINDING-5)."""
    import copy as _copy
    classes = getattr(kernel, "loop_statement_classes", None)
    operands = getattr(kernel, "loop_statement_class_operands", None)
    if not classes or operands is None or len(operands) != len(classes) or not kernel.loop_paths:
        return None
    names = list(kernel.param_names or [])
    dtypes = {name: inp.tensor.dtype for name, inp in zip(names, kernel.inputs)}
    positions = {name: i for i, name in enumerate(names[:len(kernel.inputs)])}
    cls_lines = list(getattr(kernel, "loop_statement_class_lines", None) or []) or [None] * len(classes)
    if len(cls_lines) != len(classes):
        return None
    probe = _copy.copy(kernel)
    probe.loop_outer_statements = None
    receipts: list = []
    reads: list = []

    def entry(name):
        return {"name": str(name), "input": positions.get(str(name)), "dtype": dtypes.get(str(name))}

    for (path, kind, _extra), spec, ln in zip(classes, operands, cls_lines):
        if kind != "mac":
            continue
        probe.loop_paths = [(list(path), 1)]
        probe.class_probe_path = list(path)                  # the loop-nest replay sizes exactly this prefix (an earlier change reconciliation)
        probe.class_probe_line = ln                          # ... at exactly the loop that produced the record (sibling nests share prefixes)
        li = _kernel_loop_iterations(probe, dim_sizes)
        if not li or li[2] != "paths":
            return None
        spec = spec or {"occurrences": [], "payloads": [], "scales": [], "conversions": [], "names": [], "unproven": True, "arms_differ": False}
        occ = list(spec.get("occurrences", []))
        listed = [o["name"] for o in occ] + list(spec.get("names", []))
        rec = {"macs": int(li[1]),
               "payloads": [entry(o["name"]) for o in occ if o.get("role") == "payload"],
               "scales": [entry(o["name"]) for o in occ if o.get("role") == "scale"],
               "conversions": [str(c) for c in spec.get("conversions", [])],
               "operands": {str(n): dtypes.get(str(n)) for n in listed},
               "unproven": bool(spec.get("unproven")), "arms_differ": bool(spec.get("arms_differ"))}
        read = {"macs": int(li[1]),
                "occurrences": [{"name": str(o["name"]), "input": positions.get(str(o["name"])), "index": str(o["index"]), "role": str(o["role"])} for o in occ],
                "conversions": [str(c) for c in spec.get("conversions", [])],
                "names": []}
        for n in listed:
            if str(n) not in read["names"]:
                read["names"].append(str(n))
        for g, r in zip(receipts, reads):
            if (all(g[k] == rec[k] for k in ("payloads", "scales", "conversions", "operands", "unproven", "arms_differ"))
                    and all(r[k] == read[k] for k in ("occurrences", "conversions", "names"))):
                g["macs"] += rec["macs"]
                r["macs"] += read["macs"]
                break
        else:
            receipts.append(rec)
            reads.append(read)
    for g in receipts:
        if not g["unproven"] and not g["arms_differ"]:
            g.pop("unproven"); g.pop("arms_differ")
    return receipts, reads


def _infer_ops(loop_expr: str | None) -> tuple:
    """Heuristically infer (elementwise_op, reduction_op) from a loop expression string.

    Returns (elementwise_op, reduction_op) strings drawn from
    BODY_ELEMENTWISE_TOKENS / BODY_REDUCTION_TOKENS, or (None, None) when
    there is no body.
    """
    if not loop_expr:
        return (None, None)

    # Reduction: augmented assignment
    if "+=" in loop_expr:
        reduction_op = "add"
    elif "-=" in loop_expr:
        reduction_op = "sub"
    else:
        reduction_op = "none"

    # Elementwise: look at the RHS expression.  ``fp32(...)`` is the DSL's
    # rounding-boundary cast (zero-cost, no MACs): it is invisible here, so a
    # reduction that only snaps its terms is never billed as a multiply-
    # accumulate (snap-mac-1).
    loop_expr = re.sub(r"\bfp32\(", "(", loop_expr)
    # Simple scalar multiply: Y[...] += A[...] * B[...]
    if re.search(r"\*", loop_expr):
        elementwise_op = "mul"
    elif re.search(r"\+", loop_expr.split("=", 1)[-1]):
        elementwise_op = "add"
    elif re.search(r"exp\(", loop_expr):
        elementwise_op = "exp"
    elif re.search(r"max\(", loop_expr):
        elementwise_op = "max"
    elif re.search(r"[=/]\s*[A-Za-z_]\w*\[", loop_expr) and "*" not in loop_expr:
        # Simple copy: Y[...] = X[...]
        elementwise_op = "copy"
    else:
        elementwise_op = None

    return (elementwise_op, reduction_op)


# ---------------------------------------------------------------------------
# Top-level: annotate a DSLGraph dict (as from dsl_graph_to_dict) with equations
# ---------------------------------------------------------------------------

def annotate_einsum_equations(graph: DSLGraph, graph_dict: dict) -> dict:
    """Return a copy of graph_dict with einsum_equation, elementwise_op, reduction_op filled in.

    graph      - DSLGraph domain object (for dim access)
    graph_dict - the dict produced by dsl_graph_to_dict (will not be mutated)
    """
    import copy
    result = copy.deepcopy(graph_dict)

    kernel_map = {k.id: k for k in graph.kernels}

    # Graph-wide dim-name -> size registry for loop-path extent resolution.
    dim_sizes: dict = {}
    for t in graph.tensors.values():
        for dname, dsize in t.dims.items():
            if isinstance(dsize, int) and dsize > 0:
                dim_sizes[dname] = max(dim_sizes.get(dname, 0), dsize)

    for node_id, node_data in result["layers"].items():
        if node_id == "start":
            node_data["einsum_equation"] = build_start_einsum(graph)
            node_data["is_real_einsum"] = False
            node_data["elementwise_op"] = "copy"
            node_data["reduction_op"] = "none"
        else:
            kernel = kernel_map[node_id]
            elem_op, red_op = _infer_ops(kernel.loop_expr)
            n_inputs = len(kernel.inputs)
            # A true einsum (multiply-accumulate) requires mul+add AND at
            # least 2 input tensors.  Single-input kernels with mul+add
            # (e.g. cumsum via Y[b,n] += X[b,k] * (k<=n)) are self-
            # reductions, not real contractions.
            is_real = (elem_op == "mul" and red_op == "add" and n_inputs >= 2)
            # Try loop_expr-based equation first (supports compound dims like (P+R))
            equation = None
            eq_from_loop_expr = None
            if is_real:
                eq_from_loop_expr = build_kernel_einsum_from_loop_expr(kernel)
                equation = eq_from_loop_expr
            # Fall back to dim-name-based equation
            if equation is None:
                equation = build_kernel_einsum(kernel)
            node_data["einsum_equation"] = equation
            node_data["elementwise_op"] = elem_op
            node_data["reduction_op"] = red_op
            node_data["is_real_einsum"] = is_real
            # Explicit DSL provenance (kernel-origin-10): every kernel
            # node says it came through the DSL route and which einsum_lib
            # builtin it resolved to (None for a parsed kernel body).  A
            # bodiless node whose function is NOT a known builtin has nothing
            # to cost: it is exported as unrecognised and the analyzer reports
            # it incomplete instead of pricing it from its shapes.
            node_data["dsl_route"] = True
            # the traced kernel IDENTITY of this node, established at conversion from the trace (the node key is
            # `<function>#<n>` for the same reason): the analyzer binds the re-derived source to it, never to the
            # name a receipt asserts (DTYPE-KERNEL-IDENTITY-10)
            node_data["kernel_function"] = kernel.function_name
            _bodiless = not getattr(kernel, "loop_source_parsed", False) and not kernel.loop_expr
            _builtin = kernel.function_name if (_bodiless and kernel.function_name in DSL_TENSOR_BUILTINS) else None
            node_data["builtin"] = _builtin
            node_data["kernel_body"] = not _bodiless        # a parsed @kernel body (or a legacy loop_expr) exists
            if _bodiless and _builtin is None:
                node_data["unrecognised_op"] = kernel.function_name
            # Ground-truth work: product of actual loop-variable extents.
            # The analyzer prefers this over equation-derived counts, which
            # inflate when equations misrepresent the loop nest (derived
            # indices, fused extra inputs).
            _extent_mismatch = None
            _extent_code = "loop_dim_extent_mismatch"
            try:
                li = _kernel_loop_iterations(kernel, dim_sizes)
            except LoopDimExtentMismatch as exc:
                # TARGET-DSL-DIM-EXTENT-13 / an earlier fix: a loop dimension whose runtime
                # binding is inconsistent or unresolvable at this call cannot be
                # counted; the kernel is marked incomplete (the analyzer keeps its
                # equation basis) rather than priced from a guessed extent.
                li = None
                _extent_mismatch = str(exc)
                _extent_code = exc.code
            loop_iters, li_trips, li_src, li_outer = li if li else (None, None, None, 0)
            # Per-statement unit split (issue #59): MAC statements vs other
            # statements, each summed over its own loop extents.  Exported only
            # from the explicit Dim() paths; the analyzer bills MACs and
            # other_ops from these two fields, never from the whole-kernel
            # ``is_real_einsum`` flag.
            if li and li_src == "paths":
                _cls = _kernel_statement_class_totals(kernel, dim_sizes)
                if _cls is not None:
                    node_data["loop_mac_statements"], node_data["loop_other_statements"], node_data["loop_extra_multiplies"] = _cls
                    _gt = _kernel_gather_statement_trips(kernel, dim_sizes)
                    if _gt:
                        node_data["gather_index_statements"] = int(_gt)   # the row relies on the gather rule (GATHER-INDEX production)
                    # the MAC statements' operands with their declared dtypes: the analyzer
                    # binds the contraction's compute dtype from them (DSL-COMPUTE-DTYPE-BINDING-1)
                    _ops = _kernel_mac_operand_records(kernel, dim_sizes)
                    if _ops is not None:
                        node_data["loop_mac_operands"], node_data["loop_mac_reads"] = _ops
                    if getattr(kernel, "source_text", None) is not None:
                        # the kernel's source segment bound to the node with its hash, the module hash and the declared
                        # signature: the analyzer re-derives the operand occurrences from it (DTYPE-OCCURRENCE-CHANNEL-8)
                        node_data["kernel_source"] = {"name": kernel.function_name, "sha256": kernel.source_sha256,
                                                      "module_sha256": kernel.module_source_sha256, "text": kernel.source_text,
                                                      "signature": [dict(s) for s in (kernel.signature or [])]}
            if getattr(kernel, "loop_source_parsed", False):
                # The classifier's verdict is stated whenever the kernel's source was classified, whether or not the
                # trip count certified (an earlier change round 7 reconciliation): a loop-extent refusal (an unbindable or opaque
                # explicit size, dim-call-binding-23 / OPAQUE-SIZE-20) must not hide a class-evidence refusal
                # (an earlier fix value_as_dim_size).  Without class totals the analyzer bills nothing from these flags.
                _gv = getattr(kernel, "loop_statement_grammar_violations", None)
                if _gv:
                    node_data["class_grammar_violations"] = "; ".join(str(v) for v in _gv)
                _unc = list(getattr(kernel, "loop_statement_classes_uncertain", None) or [])
                if getattr(kernel, "loop_singleton_reads", None):
                    # SINGLETON-SCALE-READ production (issue #61): the classifier proved the SHAPE of the read;
                    # the EXTENT is proven here from this call's declared shape binding, or the layer is unproven
                    # (stated like every other class verdict, whether or not the trip count certified); the executed
                    # read count is sized by the certified Dim() path replay only (an earlier change x an earlier change reconciliation)
                    _ok, _why = _singleton_reads_proven(kernel)
                    if _ok:
                        if li and li_src == "paths":
                            _st = _kernel_singleton_statement_trips(kernel, dim_sizes)
                            if _st:
                                node_data["singleton_scale_reads"] = int(_st)   # executed trips reading a singleton scale
                    else:
                        if "class_evidence_unproven" not in _unc:
                            _unc.append("class_evidence_unproven")
                        _prev = node_data.get("class_grammar_violations")
                        node_data["class_grammar_violations"] = "; ".join([x for x in (_prev, _why) if x])
                if _unc:
                    # branches of different class composition: the classes are a
                    # per-class maximum, the analyzer marks the layer incomplete
                    node_data["class_evidence_uncertain"] = "+".join(str(u) for u in _unc)
            if loop_iters and eq_from_loop_expr is not None and li_src == "fallback":
                # A faithful loop-expr equation is a better witness than the
                # standalone-index fallback: never let the fallback OVERRIDE it
                # upward (measured: fallback multiplying sibling nests inflated
                # a correct softmax equation by 6e4x).
                union = 1
                seen = {}
                for t in [inp.tensor for inp in kernel.inputs] + [kernel.output]:
                    for dname, dsize in t.dims.items():
                        if isinstance(dsize, int) and dsize > 0:
                            seen[dname] = max(seen.get(dname, 0), dsize)
                for v in seen.values():
                    union *= v
                loop_iters = min(loop_iters, union) if union > 1 else loop_iters
            # Completeness proof for the analyzer's loop-count basis
            # (loop-undercount-1 / -COMPLETENESS-4): the count is complete
            # only when it came from the explicit Dim() paths AND the parser's
            # counted traversal recorded no omission (a range()/while loop, a
            # for-else / with / try body, a comprehension — anything the
            # counter did not represent or did not visit).  The standalone-
            # index fallback cannot prove anything about omitted loops either.
            # The omissions are carried whether or not a numeric count exists
            # (loop-marker-5): a kernel whose only loop is a ``while``
            # has no count, but its omission is a known fact the analyzer must
            # see.  Only PARSED kernels carry that fact — a builtin
            # (``gather`` / ``slice_at`` ...) or a legacy artifact without
            # source has nothing observed and is not marked here.
            _omitted = list(getattr(kernel, "loop_paths_omitted", None) or [])
            # WITNESS_MODEL v3 §2 (B′, an earlier change round 9): a construct the witness does not model leaves its access without a
            # witness and is REPORTED in two informational node fields -- ``witness_unmodelled`` (the typed notes) and
            # ``witness_coverage`` (partial / full / none).  A declined witness is a declined consistency cross-check: it
            # never joins ``loop_iterations_omitted``, never clears ``loop_iterations_complete`` and changes no count or
            # basis (the round-7/8 plumbing that did so is removed).
            _witness_notes = list(getattr(kernel, "witness_notes", None) or [])
            if _witness_notes:
                node_data["witness_unmodelled"] = _witness_notes
            if kernel.loop_expr:
                node_data["witness_coverage"] = "partial" if _witness_notes else ("full" if getattr(kernel, "witness_pass", False) else "none")
            if _extent_mismatch:
                _omitted.append(_extent_mismatch)
                node_data[_extent_code] = _extent_mismatch
            if loop_iters:
                # Every executed in-loop assignment, all levels (statement-
                # COVERAGE-1): innermost weighted trips + statements at outer loop
                # levels.  A reduction kernel keeps ``loop_iterations`` = its
                # innermost (MAC-bearing) trips; a kernel that is NOT a reduction
                # is SM work statement by statement, so its count is the full
                # statement total (the assignment-execution convention the
                # other-ops side bills).  Both figures are exported.
                if li_src == "paths":
                    node_data["loop_statements"] = int(loop_iters) + int(li_outer or 0)
                    if li_outer:
                        node_data["loop_outer_statements"] = int(li_outer)
                    if not is_real and li_outer:
                        loop_iters = node_data["loop_statements"]
                node_data["loop_iterations"] = loop_iters
                _complete = bool(li_src == "paths" and getattr(kernel, "loop_paths_complete", False))
                node_data["loop_iterations_complete"] = _complete
                if not _complete:
                    if li_src == "fallback":
                        _omitted.append("standalone-index fallback: no complete Dim() loop structure")
                    node_data["loop_iterations_omitted"] = _omitted or ["loop structure not proven complete"]
                # (recompute-7: the fingerprint below belongs to the counted
                # branch — round 3 had nested it under the no-count ``elif``.)
                # Recompute-per-element fingerprint: a "real einsum" whose loop
                # nest iterates MORE than the union of its own tensors' dim
                # extents recomputes a reduction per output element (inline
                # mean/var in single-kernel norms).  That work is not a tensor-
                # core contraction — reclassify so its cost lands on SM ops
                # (the reference hides such work under the fused roofline).
                if node_data.get("is_real_einsum"):
                    union = 1
                    seen_dims: dict = {}
                    for t in [inp.tensor for inp in kernel.inputs] + [kernel.output]:
                        for dname, dsize in t.dims.items():
                            if isinstance(dsize, int) and dsize > 0:
                                seen_dims[dname] = max(seen_dims.get(dname, 0), dsize)
                    for v in seen_dims.values():
                        union *= v
                    # Use the UNWEIGHTED trip count: a multi-accumulation body
                    # (2 statements/iteration) is still a TC contraction; only
                    # a nest that TRIPS more than its tensors' index space
                    # recomputes reductions per element (inline-stat norms).
                    if (li_trips or loop_iters) > 1.5 * union:
                        node_data["is_real_einsum"] = False
                        # The adjudicated recompute convention (loop-evidence
                        # rounds 2-4) bills a per-element recomputed reduction as
                        # SM work by its innermost trip count; the per-statement
                        # unit split does not apply to it (issue #59 keeps the
                        # pin; the marker makes the exception visible).
                        node_data["recompute_fingerprint"] = True
                        node_data.pop("loop_mac_statements", None)
                        node_data.pop("loop_other_statements", None)
                        node_data.pop("loop_extra_multiplies", None)
                        # the billing convention changes HOW the work is billed, never WHETHER the
                        # kernel's classes are proven: class_evidence_uncertain, class_grammar_violations
                        # and gather_index_statements stay on the node (recompute-refusal-30)
                        node_data.pop("loop_mac_operands", None)
                        node_data.pop("loop_mac_reads", None)
            elif _omitted and getattr(kernel, "loop_source_parsed", False):
                node_data["loop_iterations_complete"] = False
                node_data["loop_iterations_omitted"] = _omitted
            # A declared-dims fallback equation is NOT a faithful description
            # of the computation for a real contraction (this is where the
            # dequant-matmul / flat-index inflations came from) — mark it so
            # downstream consumers never trust it for op counting.
            node_data["equation_faithful"] = bool(
                (not is_real) or eq_from_loop_expr is not None
            )
            # Gather-indexed inputs read only the looked-up rows, not the
            # whole table (embedding lookups, slot-dense expert weights).
            # The analyzer uses these per-position element counts in place of
            # the full tensor size for memory traffic.
            read_overrides = _gather_read_overrides(kernel)
            if read_overrides:
                node_data["input_read_elements"] = {
                    str(k): int(v) for k, v in read_overrides.items()
                }

    return result
