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

"""AST-parse *_einsum_lib.py to extract innermost loop body expressions per @kernel function."""

import ast
import re
from pathlib import Path


def _is_kernel_decorated(func_def: ast.FunctionDef) -> bool:
    """Return True if the function has a @kernel decorator."""
    for dec in func_def.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "kernel":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "kernel":
            return True
    return False


def _innermost_stmts_in_for(for_node: ast.For) -> list:
    """Recursively collect innermost assignment statements in a for-loop nest.

    A for loop is innermost if its body contains no nested for statements.
    Returns a list of unparsed assignment strings.
    """
    has_nested_for = any(isinstance(s, ast.For) for s in for_node.body)
    if has_nested_for:
        result = []
        for stmt in for_node.body:
            if isinstance(stmt, ast.For):
                result.extend(_innermost_stmts_in_for(stmt))
        return result
    # Innermost for loop: collect assignment/augmented-assignment statements
    result = []
    for stmt in for_node.body:
        if isinstance(stmt, (ast.Assign, ast.AugAssign)) or (isinstance(stmt, ast.AnnAssign) and stmt.value is not None):
            result.append(ast.unparse(stmt))
    return result


def _extract_kernel_loop_expr(func_def: ast.FunctionDef) -> str:
    """Extract innermost loop expression(s) from a @kernel function.

    Scans top-level for-loop nests in the function body and collects their
    innermost assignment statements. Multiple statements are joined with '; '.
    """
    stmts = []
    for stmt in func_def.body:
        if isinstance(stmt, ast.For):
            stmts.extend(_innermost_stmts_in_for(stmt))
    return "; ".join(stmts) if stmts else ""


def extract_loop_exprs(py_path: Path) -> dict:
    """Parse *_einsum_lib.py and return {function_name: loop_expr_string}.

    Only processes @kernel-decorated functions. Returns an empty string for
    functions where no innermost loop statements are found.
    """
    source = Path(py_path).read_text()
    tree = ast.parse(source)

    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_kernel_decorated(node):
            expr = _extract_kernel_loop_expr(node)
            result[node.name] = expr if expr else None

    return result


_BINOP_NAMES = {ast.FloorDiv: "//", ast.Mult: "*", ast.Add: "+",
                ast.Sub: "-", ast.Mod: "%"}
# Key under which _local_size_consts records the positional tensor parameters
# whose ``.shape[i]`` may be used as a size leaf.  It contains spaces, so it can
# never collide with a Python identifier looked up through ``consts[name]``.
_SHAPE_PARAMS_KEY = " shape params "
_SCALAR_PARAMS_KEY = " scalar params "   # kernel parameters that are NOT Tensors and are never rebound: a size may name one


def _int_size_spec(v: ast.expr, consts: dict):
    """Resolve an int-valued size expression into a deferred spec.

    Returns an int, ("shape", param, axis), ("scalar", param) for a non-Tensor
    kernel parameter (resolved from the call's recorded ``scalar_args``), a
    recursive ("binop", op, left, right) tree over those leaves, or None.  BinOp
    support (ported from an earlier revision) lets sizes such as
    ``half = W.shape[0] // 2`` resolve to the loop's real extent; before,
    the unresolved size pushed the kernel onto the bare-index fallback and
    billed the L1/029 half-range projection at the full weight axis.
    """
    if isinstance(v, ast.Constant) and isinstance(v.value, int):
        return v.value
    if isinstance(v, ast.Name) and consts.get(v.id) is not None:
        return consts[v.id]
    if isinstance(v, ast.Name) and v.id in (consts.get(_SCALAR_PARAMS_KEY) or ()):
        # a SCALAR kernel argument (``Dim("R", size=kernel_size)``): the trace records int arguments per call
        # (``scalar_args``) and the converter resolves the leaf from them (TARGET-KB-MAXPOOL, an earlier change round 4)
        return ("scalar", v.id)
    if (isinstance(v, ast.Subscript)
            and isinstance(v.value, ast.Attribute) and v.value.attr == "shape"
            and isinstance(v.value.value, ast.Name)
            and isinstance(v.slice, ast.Constant)
            and isinstance(v.slice.value, int)):
        # Inside a kernel, a shape leaf is only meaningful when the name is
        # the unchanged positional tensor parameter the converter will map
        # the call-site input to (``X = Z`` or a keyword-only tensor breaks
        # that correspondence).  Outside a kernel no restriction applies.
        allowed = consts.get(_SHAPE_PARAMS_KEY)
        if allowed is not None and v.value.value.id not in allowed:
            return None
        return ("shape", v.value.value.id, v.slice.value)
    if isinstance(v, ast.BinOp) and type(v.op) in _BINOP_NAMES:
        left = _int_size_spec(v.left, consts)
        right = _int_size_spec(v.right, consts)
        if left is not None and right is not None:
            return ("binop", _BINOP_NAMES[type(v.op)], left, right)
    return None


def _explicit_size_spec(v: ast.expr, consts: dict):
    """The spec of an EXPLICIT ``size=`` argument (opaque-size-20, round 6): a supported form from
    :func:`_int_size_spec`, else ``("opaque", <source text>, line, col)`` -- an explicit size the parser cannot reduce
    to a supported form (a helper call ``choose()``, ``sizes[0]`` from ``*args``, ``sizes["n"]`` from ``**kwargs``, an
    attribute, a name bound conditionally or more than once ...).  The distinction "no ``size=``" (``None``: the
    registry binding applies) versus "explicit but unsupported" is PRESERVED so the converter refuses the latter
    instead of reading an unrelated registry binding."""
    spec = _int_size_spec(v, consts)
    if spec is not None:
        return spec
    try:
        text = ast.unparse(v)
    except Exception:                                      # pragma: no cover - unparse failure on an exotic node
        text = type(v).__name__
    return ("opaque", text, getattr(v, "lineno", None), getattr(v, "col_offset", None))


def _target_names(target: ast.expr) -> list:
    """Variable names actually (re)bound by an assignment target.

    ``n``, ``a, b = ...``, ``[a, *rest] = ...`` bind names; ``Y[i + n] = ...``
    and ``obj.attr = ...`` store into an object and bind nothing, so the
    names they READ (``Y``, ``i``, ``n``) are not collected.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            out.extend(_target_names(elt))
        return out
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_scope_bindings(func) -> set:
    """Names a nested def/lambda binds in ITS OWN scope: parameters plus
    direct (non-nested) assignment-like targets, minus names it declares
    ``nonlocal``/``global`` (those refer elsewhere)."""
    if isinstance(func, ast.Lambda):
        return set(_all_parameter_names(func))
    names = set(_all_parameter_names(func))
    # Same binding inventory as the kernel scope itself (assignments, walrus,
    # for/with/except/import/def/class/del, MATCH CAPTURES, and walrus inside
    # the defaults/decorators/annotations of nested defs and lambdas, which
    # Python evaluates in THIS scope), never entering nested bodies.
    for sub in _scope_nodes(func.body, module_scope=False):
        if isinstance(sub, (ast.Global, ast.Nonlocal)):
            continue        # declarations are not bindings of this scope
        writes, shadow = _node_binding_names(sub)
        names.update(writes)
        names.update(shadow)
    # Names this function declares nonlocal/global DIRECTLY (not in deeper
    # defs) refer elsewhere and are not its own bindings.
    declared = set()
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _NESTED_SCOPES):
            continue
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        stack.extend(ast.iter_child_nodes(node))
    return names - declared


def _node_binding_names(sub) -> tuple:
    """(value_writes, shadow_only) names bound by ONE AST node.  Shadow-only
    is a bare annotation (``n: int``): it makes the name local without
    writing a value.  Global/Nonlocal are handled by the callers."""
    writes, shadow = [], []
    if isinstance(sub, ast.Assign):
        for t in sub.targets:
            writes.extend(_target_names(t))
    elif isinstance(sub, ast.AnnAssign):
        (writes if sub.value is not None else shadow).extend(_target_names(sub.target))
    elif isinstance(sub, (ast.AugAssign, ast.NamedExpr)):
        writes.extend(_target_names(sub.target))
    elif isinstance(sub, (ast.For, ast.AsyncFor)):
        writes.extend(_target_names(sub.target))
    elif isinstance(sub, (ast.With, ast.AsyncWith)):
        for item in sub.items:
            if item.optional_vars is not None:
                writes.extend(_target_names(item.optional_vars))
    elif isinstance(sub, ast.ExceptHandler) and sub.name:
        writes.append(sub.name)
    elif isinstance(sub, (ast.Import, ast.ImportFrom)):
        for alias in sub.names:
            writes.append((alias.asname or alias.name).split(".")[0])
    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        writes.append(sub.name)
    elif isinstance(sub, ast.Delete):
        for t in sub.targets:
            writes.extend(_target_names(t))
    elif isinstance(sub, ast.MatchAs) and sub.name:
        writes.append(sub.name)
    elif isinstance(sub, ast.MatchStar) and sub.name:
        writes.append(sub.name)
    elif isinstance(sub, ast.MatchMapping) and sub.rest:
        writes.append(sub.rest)
    return writes, shadow


def _outer_writes_from_nested(scope_node, module_scope: bool) -> list:
    """Names that code nested inside ``scope_node`` (a def/lambda/class) can
    write in the ENCLOSING scope, resolved lexically.

    * ``global NAME`` anywhere inside targets the module: it is a write to
      the enclosing scope only when that scope IS the module.
    * ``nonlocal NAME`` declared directly in ``scope_node`` targets the
      nearest enclosing function binding, i.e. the enclosing scope when it is
      a function (never the module).
    * ``nonlocal NAME`` declared in a deeper nested def resolves to the
      nearest intermediate def that binds NAME in its own scope; only when no
      intermediate def binds it does it reach the enclosing scope.
    Returns ``ast.Name``-free plain strings wrapped as ast.Nonlocal/ast.Global
    nodes so callers can treat them like declarations.
    """
    out = []
    if isinstance(scope_node, ast.Lambda):
        return out                       # a lambda body cannot declare anything

    def visit(node, intermediate: set):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = intermediate | _own_scope_bindings(node)
            for child in node.body:
                visit(child, inner)
        elif isinstance(node, ast.ClassDef):
            # a class body is not an enclosing function scope for the free
            # variables of functions defined inside it
            for child in node.body:
                visit(child, intermediate)
        elif isinstance(node, ast.Lambda):
            return
        elif isinstance(node, ast.Global):
            if module_scope:
                out.append(ast.Global(names=list(node.names)))
        elif isinstance(node, ast.Nonlocal):
            if not module_scope:
                reaching = [n for n in node.names if n not in intermediate]
                if reaching:
                    out.append(ast.Nonlocal(names=reaching))
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, intermediate)

    # Declarations directly in scope_node's body resolve past scope_node
    # itself; deeper ones must first get past scope_node's own bindings.
    start = set() if isinstance(scope_node, ast.ClassDef) else _own_scope_bindings(scope_node)
    for child in scope_node.body:
        visit(child, start)
    return out


def _scope_nodes(stmts, module_scope: bool = False):
    """Yield every AST node of ``stmts`` that belongs to the SAME lexical
    scope.  Nested function/class/lambda BODIES are not entered (their locals
    are not the kernel's variables), but everything evaluated in the
    enclosing scope at definition time is: decorators, default values,
    annotations, class bases and keywords.  The def/class node itself is
    yielded so its name counts as a binding, and every ``nonlocal``/``global``
    declaration inside a nested body is yielded too, because such code can
    write the enclosing name."""
    stack = list(stmts)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.decorator_list)
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                        + [a for a in (node.args.vararg, node.args.kwarg) if a]):
                if arg.annotation is not None:
                    stack.append(arg.annotation)
            if node.returns is not None:
                stack.append(node.returns)
            stack.extend(_outer_writes_from_nested(node, module_scope))
        elif isinstance(node, ast.Lambda):
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
        elif isinstance(node, ast.ClassDef):
            stack.extend(node.decorator_list)
            stack.extend(node.bases)
            stack.extend(kw.value for kw in node.keywords)
            stack.extend(_outer_writes_from_nested(node, module_scope))
        else:
            stack.extend(ast.iter_child_nodes(node))


def _binding_occurrences(stmts, module_scope: bool = False) -> tuple:
    """Variable bindings in the lexical scope of ``stmts``.

    Returns ``(value_writes, shadow_names)``: ``value_writes`` lists one entry
    per site that writes a value to a name (assignments, augmented
    assignments, annotated assignments WITH a value, walrus, for targets,
    ``with ... as``, ``except ... as``, imports, nested def/class names,
    ``del``, match captures), so repeats mean rebinding; ``shadow_names``
    additionally includes bare annotations (``n: int``), which make a name
    local without writing a value."""
    writes, shadow = [], []
    for sub in _scope_nodes(stmts, module_scope):
        if isinstance(sub, (ast.Global, ast.Nonlocal)):
            # a declaration (direct, or resolved here from a nested body by
            # _outer_writes_from_nested): nested code may write this name
            writes.extend(sub.names)
            continue
        w, s = _node_binding_names(sub)
        writes.extend(w)
        shadow.extend(s)
    return writes, shadow + writes


def _is_tensor_annotation(ann) -> bool:
    if isinstance(ann, ast.Call):
        ann = ann.func
    if isinstance(ann, ast.Name):
        return ann.id == "Tensor"
    if isinstance(ann, ast.Attribute):
        return ann.attr == "Tensor"
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return ann.value.strip().startswith("Tensor")
    return False


_TENSOR_NON_DIM_KEYWORDS = frozenset({"role", "dtype", "sparsity", "fill"})


def _is_tensor_call(node) -> bool:
    return (isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == "Tensor")
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == "Tensor")))


def _returned_names(func: ast.FunctionDef) -> list:
    """The names the kernel returns, one entry per ``return`` statement in its
    body (nested defs excluded): the identifier for ``return NAME``, ``None`` for
    any other return expression.  The converter identifies the RETURNED
    allocation by this name binding, never by its dim-name set
    (return-identity-5)."""
    out = []
    def visit(stmts):
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, ast.Return):
                out.append(stmt.value.id if isinstance(stmt.value, ast.Name) else None)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, field, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    visit(sub)
            for h in getattr(stmt, "handlers", []) or []:
                visit(h.body)
            for c in getattr(stmt, "cases", []) or []:
                visit(c.body)
    visit(func.body)
    return out


def _static_truth(test: ast.expr):
    """The truth value of an ``if`` test that is a literal constant (``if True:`` / ``if 0:``), else None: the one
    condition that is statically decidable (TARGET-DSL-EARLY-RETURN-11)."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    return None


def _pattern_names(pattern) -> list:
    names = []
    for sub in ast.walk(pattern):
        for attr in ("name", "rest"):
            v = getattr(sub, attr, None)
            if isinstance(v, str):
                names.append(v)
    return names


def _target_names(t) -> list:
    """Plain names a binding target writes (``Y``, ``a, b``, ``*rest``); a subscript or attribute target mutates an
    object and rebinds no name."""
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in _target_names(e)]
    if isinstance(t, ast.Starred):
        return _target_names(t.value)
    return []


def _local_bindings(func: ast.FunctionDef, consts: dict) -> tuple:
    """One walk of the kernel body (nested defs excluded) -> ``(allocations, bindings, returns)``.

    ``allocations``: every ``Tensor(NAME=size, ...)`` call in source order:
    ``{"line", "dims": {name: size spec}, "conditional", "target"}``.  At run time each such allocation REGISTERS its
    dimension names in the kernel's dim registry (core.Tensor.__init__), exactly as the kernel's arguments register
    their actual dim names on entry -- these registrations, not the parameter annotations, are what ``Dim("NAME")``
    reads (annotation-binding-1 / TARGET-DSL-DIM-LOCAL-2).  A size that is not statically resolvable is recorded
    as ``None`` (the converter then refuses to trust a loop over that name).  ``conditional`` marks an allocation whose
    execution is not statically certain -- under if / while / try / with / match, in a for-else block, or in the body
    of a non-``Dim()`` loop (conditional-binding-4); an ``if`` whose test is a literal constant is decided
    statically (its dead branch is not walked at all).  ``target`` is the plain name the allocation is assigned to.

    ``bindings``: EVERY binding of a name in the body, in execution (source) order -- assignment, augmented and
    annotated assignment, tuple unpacking, walrus, for / with / except targets, imports, del, nested def / class names,
    global / nonlocal declarations, match captures: ``{"seq", "name", "line", "kind", "source", "alloc",
    "conditional"}`` with ``kind`` = ``"alloc"`` (``NAME = Tensor(...)``, ``alloc`` = index into ``allocations``),
    ``"alias"`` (``NAME = OTHER``, ``source`` = OTHER) or ``"other"``.  The converter identifies the RETURNED
    allocation as the LAST certain binding of the returned name, through plain aliases; a returned name whose last
    binding is anything else borrows nothing (return-rebind-9).

    ``returns``: every ``return`` statement: ``{"seq", "line", "name", "conditional"}`` (``name`` for ``return NAME``,
    else None).  A return that is not strictly after a loop nest -- conditional or not -- leaves the trip count of
    that nest unproven (TARGET-DSL-EARLY-RETURN-11)."""
    allocs, bindings, returns, dimregs = [], [], [], []
    seq = [0]
    def bind(name, line, kind, conditional, source=None, alloc=None):
        seq[0] += 1
        bindings.append({"seq": seq[0], "name": name, "line": line, "kind": kind, "source": source, "alloc": alloc,
                         "conditional": bool(conditional)})
    def walk_stmts(stmts, conditional):
        for stmt in stmts:
            line = getattr(stmt, "lineno", None)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(stmt.name, line, "other", conditional)
                continue
            exprs = _expr_nodes(stmt)
            alloc_of = {}
            for node in exprs:
                if not _is_tensor_call(node):
                    continue
                dims = {kw.arg: _int_size_spec(kw.value, consts) for kw in node.keywords
                        if kw.arg and kw.arg not in _TENSOR_NON_DIM_KEYWORDS}
                if dims:
                    target = None
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and stmt.value is node:
                        target = stmt.targets[0].id
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is node:
                        target = stmt.target.id
                    allocs.append({"line": getattr(node, "lineno", None), "col": getattr(node, "col_offset", None),
                                   "dims": dims, "conditional": conditional, "target": target})
                    alloc_of[id(node)] = len(allocs) - 1
            # dim-registration-17 (round 5): an explicit ``Dim("U", size=...)`` anywhere but a for-header's OWN
            # call (a standalone ``D = Dim("U", size=3)``, a bare expression, an argument) REGISTERS the name in the
            # runtime registry exactly like a Tensor allocation does; the converter replays it as loop-carried state.
            # The for-header's own call is the loop node's ``size`` (registered by the replay when the loop starts).
            # nested-dim-eval-25 (round 8): ONLY the header's own call is excluded -- a ``Dim(...)`` nested in
            # another Dim call's ARGUMENTS (``Dim("T", parallel=bool(Dim("T", size=3)))``), in a header or standalone,
            # is a registration the runtime performs BEFORE the enclosing call; every allocation / registration of the
            # statement carries its Python EVALUATION-ORDER index ``eseq`` (arguments before the enclosing call, left to
            # right -- see :func:`_eval_order`) so the converter replays them in runtime order, never in source order;
            # a header-nested registration carries ``header_of`` (the loop it precedes); a registration inside an
            # expression whose evaluation order the parser does not model carries ``order_unknown`` (typed refusal in
            # the converter -- never a silent skip).
            is_for = isinstance(stmt, (ast.For, ast.AsyncFor))
            own_header = {id(stmt.iter)} if is_for and _is_dim_call(stmt.iter) else set()
            loop_pos = [stmt.lineno, getattr(stmt, "col_offset", None)] if is_for and _dim_call_info(stmt.iter, consts) is not None else None
            header_ids = {id(n) for n in ast.walk(stmt.iter)} - own_header if is_for else set()
            order = _eval_order(stmt)
            regs_here = []
            for node in exprs:
                if id(node) in own_header or not isinstance(node, ast.Call):
                    continue
                if not _is_dim_call(node):
                    continue
                # dim-call-binding-23: the SAME binding as a for-header -- ``D = Dim("U", False, 3)`` registers U=3,
                # ``Dim("U", **{"size": 3})`` is an opaque explicit size (refused), a call that provably passes no size
                # registers nothing
                name, size = _dim_call_binding(node, consts)
                if size is None:
                    continue
                eseq, econd, why = order.get(id(node), (None, False, "a statement form the parser does not model"))
                reg = {"line": getattr(node, "lineno", line), "col": getattr(node, "col_offset", None), "dim": name,
                       "size": size, "conditional": bool(conditional or econd),
                       "eseq": eseq, "stmt_line": line, "stmt_col": getattr(stmt, "col_offset", None)}
                if id(node) in header_ids and loop_pos is not None:
                    reg["header_of"] = loop_pos
                if why is not None:
                    try:
                        reg["text"] = ast.unparse(node)
                    except Exception:                      # pragma: no cover - exotic node
                        reg["text"] = "Dim(...)"
                    reg["order_unknown"] = why
                regs_here.append(reg)
            regs_here.sort(key=lambda r: r["eseq"] if r["eseq"] is not None else 10 ** 9)   # recorded in EVALUATION order
            dimregs.extend(regs_here)
            # the statement's allocations: evaluation-order keys only when the order within the statement matters
            # (two or more registering nodes, or a header-nested one), so single-allocation graphs are unchanged
            here = [allocs[i] for i in alloc_of.values()]
            if here and (len(here) + len(regs_here) >= 2 or any(id(n) in header_ids for n in exprs if id(n) in alloc_of)):
                for nid, i in alloc_of.items():
                    eseq, econd, _why = order.get(nid, (None, False, None))
                    allocs[i].update({"eseq": eseq, "stmt_line": line, "stmt_col": getattr(stmt, "col_offset", None)})
                    if econd:
                        allocs[i]["conditional"] = True
                    if nid in header_ids and loop_pos is not None:
                        allocs[i]["header_of"] = loop_pos
            for node in exprs:
                if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                    bind(node.target.id, getattr(node, "lineno", line), "other", conditional)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and getattr(stmt, "value", None) is not None:
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    name = targets[0].id
                    if id(stmt.value) in alloc_of:
                        bind(name, line, "alloc", conditional, alloc=alloc_of[id(stmt.value)])
                    elif isinstance(stmt.value, ast.Name):
                        bind(name, line, "alias", conditional, source=stmt.value.id)
                    else:
                        bind(name, line, "other", conditional)
                else:
                    for t in targets:
                        for name in _target_names(t):
                            bind(name, line, "other", conditional)
            elif isinstance(stmt, ast.AugAssign):
                for name in _target_names(stmt.target):
                    bind(name, line, "other", conditional)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for a in stmt.names:
                    bind((a.asname or a.name).split(".")[0], line, "other", conditional)
            elif isinstance(stmt, (ast.Global, ast.Nonlocal)):
                for name in stmt.names:
                    bind(name, line, "other", conditional)
            elif isinstance(stmt, ast.Delete):
                for t in stmt.targets:
                    for name in _target_names(t):
                        bind(name, line, "other", conditional)
            elif isinstance(stmt, ast.Return):
                seq[0] += 1
                returns.append({"seq": seq[0], "line": line, "conditional": bool(conditional),
                                "name": stmt.value.id if isinstance(stmt.value, ast.Name) else None})
            # statement containers
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                # a Dim() loop body runs (positive extents); any other loop may run zero times
                body_conditional = conditional or isinstance(stmt, ast.AsyncFor) or _dim_call_info(stmt.iter, consts) is None
                for name in _target_names(stmt.target):
                    bind(name, line, "other", body_conditional)
                walk_stmts(stmt.body, body_conditional); walk_stmts(stmt.orelse, True)
            elif isinstance(stmt, ast.If):
                truth = _static_truth(stmt.test)
                if truth is None:
                    walk_stmts(stmt.body, True); walk_stmts(stmt.orelse, True)
                elif truth:
                    walk_stmts(stmt.body, conditional)          # decided statically: the body always runs
                else:
                    walk_stmts(stmt.orelse, conditional)        # the body is dead code
            elif isinstance(stmt, ast.While):
                walk_stmts(stmt.body, True); walk_stmts(stmt.orelse, True)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            bind(name, line, "other", True)
                walk_stmts(stmt.body, True)
            elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
                walk_stmts(stmt.body, True); walk_stmts(stmt.orelse, True); walk_stmts(stmt.finalbody, True)
                for h in stmt.handlers:
                    if h.name:
                        bind(h.name, getattr(h, "lineno", line), "other", True)
                    walk_stmts(h.body, True)
            elif isinstance(stmt, ast.Match):
                for c in stmt.cases:
                    for name in _pattern_names(c.pattern):
                        bind(name, getattr(c.pattern, "lineno", line), "other", True)
                    walk_stmts(c.body, True)
    walk_stmts(func.body, False)
    return allocs, bindings, returns, dimregs


def _is_dim_call(node) -> bool:
    """``Dim(...)`` spelled as a name or an attribute (``lib.Dim``)."""
    return isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "Dim")
                                           or (isinstance(node.func, ast.Attribute) and node.func.attr == "Dim"))


def _local_tensor_dims(func: ast.FunctionDef, consts: dict) -> list:
    """The allocations of ``_local_bindings`` alone (kept for callers of the round-2 interface)."""
    return _local_bindings(func, consts)[0]


def _expr_nodes(stmt):
    """Expression nodes directly owned by *stmt* (not those of its nested statement bodies)."""
    found = []
    def rec(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            found.append(child); rec(child)
    rec(stmt)
    return found


# Expression node types whose children (``ast.iter_child_nodes`` order) are evaluated unconditionally, left to right,
# before the node itself -- the post-order over them IS Python's evaluation order (nested-dim-eval-25).
_ORDERED_EXPR_TYPES = (ast.Call, ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.Set,
                       ast.Dict, ast.BinOp, ast.UnaryOp, ast.Starred, ast.keyword, ast.Slice, ast.JoinedStr,
                       ast.FormattedValue, ast.NamedExpr)


def _eval_order(stmt) -> dict:
    """``{id(node): (eseq, conditional, order_unknown)}`` for every expression node directly owned by *stmt*, in PYTHON
    EVALUATION order (nested-dim-eval-25): a call's function, then its positional arguments left to right, then
    its keyword values, then the call itself; a subscript's value before its slice; an assignment's value before its
    targets; an augmented assignment's target operands before its value; a ``for`` header before anything of its body.
    ``conditional`` is True for the operands of a conditional expression / the non-first operands of ``and`` / ``or``
    / the non-first comparators of a chained comparison (short-circuit: they may not execute).  ``order_unknown``
    names the construct for every node inside an expression form the parser does not model (a comprehension, a
    generator, ``await`` / ``yield``, a loop target, an annotation, a ``match`` pattern ...): a registration there is
    REFUSED by the converter, never skipped.  Nested defs / lambdas / classes are their own scope (not evaluated here)."""
    out: dict = {}
    seq = [0]
    def put(node, cond, why):
        seq[0] += 1
        out[id(node)] = (seq[0], bool(cond), why)
    def rec(node, cond, why):
        if node is None or isinstance(node, (ast.stmt, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return
        if isinstance(node, ast.IfExp):
            rec(node.test, cond, why); rec(node.body, True, why); rec(node.orelse, True, why)
        elif isinstance(node, ast.BoolOp):
            for i, v in enumerate(node.values):
                rec(v, cond or i > 0, why)
        elif isinstance(node, ast.Compare):
            rec(node.left, cond, why)
            for i, c in enumerate(node.comparators):
                rec(c, cond or i > 0, why)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):             # key, value, key, value ... (``**d``: key is None)
                rec(k, cond, why); rec(v, cond, why)
        elif isinstance(node, _ORDERED_EXPR_TYPES):
            for child in ast.iter_child_nodes(node):
                rec(child, cond, why)
        else:
            inner = why or f"a {type(node).__name__} expression"
            for child in ast.iter_child_nodes(node):
                rec(child, cond, inner)
            put(node, cond, inner)
            return
        put(node, cond, why)
    def unknown(node, what):
        for n in ast.walk(node) if node is not None else ():
            if not isinstance(n, ast.stmt):
                put(n, False, what)
    if isinstance(stmt, ast.Assign):
        rec(stmt.value, False, None)
        for t in stmt.targets:
            rec(t, False, None)
    elif isinstance(stmt, ast.AnnAssign):
        rec(stmt.value, False, None); rec(stmt.target, False, None); unknown(stmt.annotation, "an annotation")
    elif isinstance(stmt, ast.AugAssign):
        rec(stmt.target, False, None); rec(stmt.value, False, None)
    elif isinstance(stmt, (ast.Expr, ast.Return)):
        rec(stmt.value, False, None)
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        rec(stmt.iter, False, None); unknown(stmt.target, "a loop target")
    elif isinstance(stmt, (ast.If, ast.While)):
        rec(stmt.test, False, None)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            rec(item.context_expr, False, None); unknown(item.optional_vars, "a with target")
    elif isinstance(stmt, ast.Assert):
        rec(stmt.test, False, None); rec(stmt.msg, True, None)
    elif isinstance(stmt, ast.Raise):
        rec(stmt.exc, False, None); rec(stmt.cause, False, None)
    elif isinstance(stmt, ast.Delete):
        for t in stmt.targets:
            rec(t, False, None)
    else:
        for child in ast.iter_child_nodes(stmt):
            unknown(child, f"a {type(stmt).__name__} statement")
    return out


def _tensor_param_count(func: ast.FunctionDef):
    """Number of leading positional parameters annotated ``Tensor(...)``, or
    None when the recorded call inputs cannot be mapped onto ``param_names``
    positionally: a Tensor parameter after a non-Tensor one, an unannotated
    positional parameter, ``*args``, ``**kwargs``, or a keyword-only
    parameter that is (or may be) a tensor.  The tracer records tensor
    arguments in signature order, so with a clean tensor prefix the i-th
    recorded input IS the i-th positional parameter."""
    a = func.args
    positional = a.posonlyargs + a.args
    if a.vararg is not None or a.kwarg is not None:
        return None
    for kw in a.kwonlyargs:
        if kw.annotation is None or _is_tensor_annotation(kw.annotation):
            return None
    count = 0
    seen_non_tensor = False
    for p in positional:
        if p.annotation is None:
            return None
        if _is_tensor_annotation(p.annotation):
            if seen_non_tensor:
                return None
            count += 1
        else:
            seen_non_tensor = True
    return count


def _all_parameter_names(func: ast.FunctionDef) -> list:
    a = func.args
    params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        params.append(a.vararg.arg)
    if a.kwarg:
        params.append(a.kwarg.arg)
    return params


def _local_size_consts(func: ast.FunctionDef, consts: dict, param_names: list) -> dict:
    """Size bindings that provably reach the kernel's Dim loops.

    Conservative by construction (reviewer finding on the an earlier revision port):
    * kernel parameters shadow module constants of the same name and are
      unresolved (``DIVISOR`` as a parameter is not the module ``DIVISOR``);
    * bindings are followed in straight-line order; a reassignment that does
      not resolve (``n = max(n, ...)``), an augmented assignment (``n *= 2``)
      or a rebinding inside a loop body invalidates the name instead of
      keeping an earlier spec;
    * assignments inside ``if``/``while``/``try``/``with`` (and loop ``else``
      blocks) may not execute, so every name they bind is invalidated and
      none of their specs is accepted;
    * a name first bound inside a ``for`` body from loop-invariant leaves is
      accepted, as the pre-existing parser did for ``n = X.shape[1]`` inside
      an outer Dim loop.
    * a name bound at MORE THAN ONE site in the function (including a
      later resolvable reassignment after the loop, a ``with ... as``
      target or an ``except ... as`` name) is never resolved: one map is
      applied to every loop in the body, so only a unique, unconditional
      binding is valid at every use;
    * item/attribute stores (``Y[i + n] = ...``) bind nothing and do not
      invalidate the names they read.
    An invalidated name falls back to the previous behaviour (registry/
    standalone-index resolution), never to a stale or wrong extent.
    """
    params = _all_parameter_names(func)
    writes, shadow = _binding_occurrences(func.body)
    counts = {}
    for name in writes + params:
        counts[name] = counts.get(name, 0) + 1

    local_consts = dict(consts)
    for name in set(shadow) | set(params):   # anything local hides the module value
        local_consts.pop(name, None)
    # ``X.shape[i]`` leaves are only valid for positional tensor parameters
    # that the function never rebinds; the converter maps those positionally
    # to the call-site inputs (see extract_kernel_info's param_names).
    positional = [p.arg for p in func.args.posonlyargs + func.args.args]
    local_consts[_SHAPE_PARAMS_KEY] = frozenset(
        p for p in positional if counts.get(p, 0) == 1)
    tensor_params = {p.arg for p in func.args.posonlyargs + func.args.args + func.args.kwonlyargs
                     if p.annotation is not None and _is_tensor_annotation(p.annotation)}
    local_consts[_SCALAR_PARAMS_KEY] = frozenset(
        p for p in positional + [k.arg for k in func.args.kwonlyargs]
        if p not in tensor_params and counts.get(p, 0) == 1)

    def visit(stmts):
        for stmt in stmts:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if (len(targets) == 1 and isinstance(targets[0], ast.Name)
                        and stmt.value is not None):
                    name = targets[0].id
                    if counts.get(name) == 1:      # unique, unconditional binding
                        spec = _int_size_spec(stmt.value, local_consts)
                        if spec is not None:
                            local_consts[name] = spec
            elif isinstance(stmt, ast.For):
                # A for body runs whenever the loop has trips; a name bound
                # exactly once in the function and first bound here is
                # loop-invariant by construction (``n = X.shape[1]``).
                visit(stmt.body)
            # if / while / try / with / match / nested defs: bindings inside
            # may not execute; nothing from them is accepted.
    visit(func.body)
    return local_consts


_DIM_PARAMS = ("name", "parallel", "size")       # the runtime signature: Dim(name, parallel=False, size=None) (core.Dim)
_DIM_UNBINDABLE = "Dim call arguments not statically bindable"


def _dim_call_binding(node: ast.Call, consts: dict) -> tuple:
    """``(dim_name, size_spec)`` of a ``Dim(...)`` call with its arguments bound against the RUNTIME signature
    ``Dim(name, parallel=False, size=None)`` -- identically for a for-header and a standalone registration
    (dim-call-binding-23).  Positional index 2 IS the size (``Dim("T", False, 3)``), a keyword ``size=`` IS the
    size, and ``parallel`` passed positionally (index 1) is never taken for it.  ``dim_name`` is the string literal
    bound to ``name`` (positionally or ``name=``), else None.  ``size_spec`` is None only when the call PROVABLY passes
    no size (no third argument, no ``size=``, or an explicit ``size=None``: the runtime registers nothing); otherwise
    the spec of :func:`_explicit_size_spec`.  Any argument form the parser cannot bind to a parameter -- ``*args``,
    ``**kwargs`` (``Dim("T", **{"size": 3})``), more than three positionals, an unknown or repeated keyword -- is an
    OPAQUE explicit size ``("opaque", "Dim call arguments not statically bindable: <call>", line, col)``: refused by
    the converter, never "absent" (the runtime may well register a size the parser could not see)."""
    try:
        text = ast.unparse(node)
    except Exception:                                      # pragma: no cover - exotic node
        text = "Dim(...)"
    opaque = ("opaque", f"{_DIM_UNBINDABLE}: {text}", getattr(node, "lineno", None), getattr(node, "col_offset", None))
    # what binds BEFORE anything unbindable (the name is read from it even when the call as a whole is refused)
    bound = {}
    for i, a in enumerate(node.args[:len(_DIM_PARAMS)]):
        if isinstance(a, ast.Starred):
            break
        bound[_DIM_PARAMS[i]] = a
    ok = len(node.args) <= len(_DIM_PARAMS) and not any(isinstance(a, ast.Starred) for a in node.args)
    for kw in node.keywords:
        if kw.arg not in _DIM_PARAMS or kw.arg in bound:          # **kwargs (arg None), an unknown or a repeated keyword
            ok = False
            continue
        bound[kw.arg] = kw.value
    name_node = bound.get("name")
    name = name_node.value if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str) else None
    if not ok:
        return name, opaque
    size_node = bound.get("size")
    if size_node is None or (isinstance(size_node, ast.Constant) and size_node.value is None):
        return name, None
    return name, _explicit_size_spec(size_node, consts)


def _dim_call_info(node: ast.expr, consts: dict):
    """If node is a Dim("NAME", ...) call, return (dim_name, size_spec).

    size_spec is an int (literal or module constant), a tuple
    ("shape", param_name, axis) for ``size=<param>.shape[i]`` (resolved by the
    converter against the call site's input tensors), ("scalar", param) for a
    recorded scalar parameter, a recursive ("binop", ...) tree over those
    leaves, ("opaque", text, line, col) for an explicit ``size=`` in any other
    form (opaque-size-20: refused by the converter, never read from the
    registry), or None when the call has no ``size=`` (resolve the dim name
    via the registry).
    """
    if not (isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "Dim")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Dim"))):
        return None
    dim_name, size = _dim_call_binding(node, consts)      # dim-call-binding-23: bound against the runtime signature
    if dim_name is None:
        return None                                         # the dimension name is not a string literal: not a countable Dim() loop
    return (dim_name, size)


def _count_assigns(stmts: list) -> int:
    """Assignment statements executed per trip of an innermost loop body.  An
    ``if`` contributes the LARGER of its branches — never their sum — which is
    the as-written (dense) convention for a conditional accumulation such as a
    masked attention update; measured density, not the loop count, discounts
    the untaken trips."""
    n = 0
    for s in stmts:
        if isinstance(s, (ast.Assign, ast.AugAssign)) or (isinstance(s, ast.AnnAssign) and s.value is not None):
            n += 1                                                       # an annotated assignment executes like a plain one
        elif isinstance(s, ast.If):
            n += max(_count_assigns(s.body), _count_assigns(s.orelse))
    return n


def _statement_context(node: ast.FunctionDef, module: ast.Module = None):
    """The classifier's view of one kernel: Tensor-annotated parameters and
    local ``Tensor(...)`` allocations are Tensor names, the other parameters
    runtime scalars, every local binding carries the provenance of its value
    at each statement (statement_class.build_context, a fixpoint over the
    loop nest), the MODULE namespace is scanned for shadowing and literals,
    and the body is checked against the supported grammar: any violation
    makes the whole kernel unproven (``ctx.violations``)."""
    from .statement_class import build_context, grammar_violations
    tensor_names, scalar_names, int_params, input_ranks = [], [], [], {}
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        ann = arg.annotation
        is_tensor = isinstance(ann, ast.Call) and ((isinstance(ann.func, ast.Name) and ann.func.id == "Tensor")
                                                   or (isinstance(ann.func, ast.Attribute) and ann.func.attr == "Tensor"))
        (tensor_names if is_tensor else scalar_names).append(arg.arg)
        if is_tensor and ann.args and isinstance(ann.args[0], ast.Constant) and isinstance(ann.args[0].value, str):
            # the declared shape binding: Tensor("A,B") declares a rank-2 input (SINGLETON-SCALE-READ production)
            input_ranks[arg.arg] = len([d for d in ann.args[0].value.split(",") if d.strip()])
        if isinstance(ann, ast.Name) and ann.id == "int":
            int_params.append(arg.arg)                                   # a size input (SIZE slot only)
    if node.args.vararg is not None or node.args.kwarg is not None:
        scalar_names.extend(a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None)
    ctx = build_context(node.body, tensor_names, scalar_names, module=module, int_params=int_params, input_ranks=input_ranks)
    ctx.violations.extend(grammar_violations(node.body, ctx))
    if node.args.vararg is not None or node.args.kwarg is not None:
        ctx.violations.append("kernel takes *args / **kwargs")
    return ctx


def _classify_assignments(stmts: list, loop_vars: frozenset, flags: set = None, ctx=None) -> tuple:
    """``(records, controls, singleton_controls)`` for the assignments executed per trip of a body
    (``singleton_controls``: the statement-``if`` TEST evaluations per trip that read a SINGLETON-SCALE
    candidate, with the same branch algebra as ``controls`` -- ISSUE61-SINGLETON-RECEIPT-4):
    ``records`` is ``(kind, extra_other, gather, singleton)`` per assignment (the
    singleton flag: the statement reads a SINGLETON-SCALE candidate, issue #61,
    carried by the same per-record algebra as the gather flag), in the same
    shape as :func:`_count_assigns` counts them, ``controls`` the number of
    gather-addressed control-test EVALUATIONS per trip of the list.  An
    ``if`` whose branches have the SAME class composition contributes that
    composition; branches of different composition contribute the per-class
    MAXIMUM over the branches and set the ``branch_class_uncertain`` flag --
    never one branch's classes silently (TARGET-KB-BRANCH-CLASS-4).  An
    accumulated inline conditional is classified per arm by the classifier
    itself (the same rule, ifexp-class-6); unproven factor provenance
    sets ``class_evidence_unproven`` (shape-divisor-5).

    ONE branch-aware receipt algebra for controls AND assignments
    (gather-receipt-29, gather-control-join-31): a statement's
    gather flag is its own; for an ``if`` each branch's figure is its control
    evaluations plus its gather-bearing records, the branch with the MAXIMUM
    figure is kept -- its records carry the flags, its controls are counted,
    never summed across the exclusive branches -- and the enclosing test's
    own evaluation (one per trip when the TEST reads through a gather) is
    added.  Controls are not class records, so the statement counts stay
    exact.  Every record carries the tensors its MAC multiplies (the
    compute-dtype operands, DSL-COMPUTE-DTYPE-BINDING-1); a branch rule unions
    both branches' MAC operands because the per-class maximum cannot name
    which MAC runs."""
    from .statement_class import classify_records, mac_operand_specs, MAC, OTHER
    out = []
    controls = 0
    scontrols = 0
    for s in stmts:
        if isinstance(s, (ast.Assign, ast.AugAssign)) or (isinstance(s, ast.AnnAssign) and s.value is not None):
            records, stmt_flags = classify_records(s, loop_vars, ctx)
            gather = ctx is not None and id(s) in ctx.gather_statements          # GATHER-INDEX production: declared gather
            single = ctx is not None and id(s) in ctx.singleton_statements       # SINGLETON-SCALE-READ production (issue #61)
            specs = mac_operand_specs(s, loop_vars, ctx)
            out.extend((kind, extra, gather, single, dict(specs) if (kind == MAC and specs) else None) for kind, extra in records)
            if flags is not None:
                flags.update(stmt_flags)
        elif isinstance(s, ast.If):
            test_gather = 0
            test_single = 0
            if ctx is not None:
                from .statement_class import provenance, STATIC, UNPROVEN_FLAG
                ctx.select(s)
                if provenance(s.test, set(loop_vars), ctx) != STATIC and flags is not None:
                    flags.add(UNPROVEN_FLAG)                                  # a statement-if TEST must be a proven mask
                if ctx._gather_hit:
                    test_gather = 1                                           # the TEST reads through a gather: one evaluation per trip
                if ctx._singleton_hit:
                    test_single = 1                                           # the TEST reads a singleton scale: one evaluation per trip (issue #61)
                ctx.select(None)
            body, c_body, sc_body = _classify_assignments(s.body, loop_vars, flags, ctx)
            orelse, c_else, sc_else = _classify_assignments(s.orelse, loop_vars, flags, ctx)
            g_body, g_else = sum(1 for _, _, g, _, _ in body if g), sum(1 for _, _, g, _, _ in orelse if g)
            if c_body + g_body >= c_else + g_else:
                source, c_src, g_src = body, c_body, g_body                   # the branch carrying the gather maximum
            else:
                source, c_src, g_src = orelse, c_else, g_else
            # the singleton flag: the same per-record algebra, the branch carrying the singleton maximum (issue #61)
            s_body, s_else = sum(1 for _, _, _, sg, _ in body if sg), sum(1 for _, _, _, sg, _ in orelse if sg)
            if sc_body + s_body >= sc_else + s_else:
                source_s, sc_src, s_src = body, sc_body, s_body
            else:
                source_s, sc_src, s_src = orelse, sc_else, s_else
            mac_specs = [sp for k, _, _, _, sp in body + orelse if k == MAC]

            def _k(sp):
                return (tuple((o["name"], o["index"], o["role"]) for o in sp["occurrences"]), tuple(sp["conversions"]), sp["unproven"], sp["arms_differ"]) if sp else None
            joint = None
            if mac_specs:
                joint = dict(mac_specs[0]) if all(_k(sp) == _k(mac_specs[0]) for sp in mac_specs) else dict(mac_specs[0], arms_differ=True)
                for sp in mac_specs[1:]:
                    for a in ("payloads", "scales", "names"):
                        joint[a] = joint[a] + [n for n in sp[a] if n not in joint[a]]
                    joint["unproven"] = joint["unproven"] or sp["unproven"]
            if sorted((k, e) for k, e, _, _, _ in body) == sorted((k, e) for k, e, _, _, _ in orelse):
                merged = [(k, e) for k, e, _, _, _ in body]
            else:
                n_mac = max(sum(1 for k, _, _, _, _ in body if k == MAC), sum(1 for k, _, _, _, _ in orelse if k == MAC))
                n_other = max(sum(1 for k, _, _, _, _ in body if k == OTHER), sum(1 for k, _, _, _, _ in orelse if k == OTHER))
                extra = max(sum(e for k, e, _, _, _ in body if k == MAC), sum(e for k, e, _, _, _ in orelse if k == MAC))
                merged = [(MAC, extra if i == 0 else 0) for i in range(n_mac)] + [(OTHER, 0)] * n_other
                if flags is not None:
                    flags.add("branch_class_uncertain")
            gathers = [g for _, _, g, _, _ in source]
            if len(gathers) != len(merged):
                gathers = [i < g_src for i in range(len(merged))]
            singles = [sg for _, _, _, sg, _ in source_s]
            if len(singles) != len(merged):
                singles = [i < s_src for i in range(len(merged))]
            out.extend((k, e, g, sg, dict(joint) if (k == MAC and joint) else None) for (k, e), g, sg in zip(merged, gathers, singles))
            controls += test_gather + c_src
            scontrols += test_single + sc_src
    return out, controls, scontrols


GATHER_CONTROL = "gather_control"      # a records entry for a control TEST evaluated through a gather (not a class record)
SINGLETON_CONTROL = "singleton_control"   # a records entry for a control TEST that reads a singleton scale (issue #61; not a class record)
_CONTROL_KINDS = (GATHER_CONTROL, SINGLETON_CONTROL)


def _outer_singleton_reads(stmts: list, ctx) -> int:
    """Singleton-scale read EVALUATIONS outside every loop -- assignments reading ``GM[0]`` and statement-``if``
    TESTS reading it at the top level of the body or inside a top-level ``if`` (never inside a ``for``) -- once per
    kernel call, through the SAME branch-aware receipt algebra as the in-loop records
    (:func:`_classify_assignments`: per-record singleton flags, control evaluations, the per-branch MAXIMUM of
    controls + flagged records across exclusive arms, the enclosing test's own evaluation added once;
    ISSUE61-SINGLETON-RECEIPT-4, ISSUE61-OUTER-SINGLETON-BRANCH-6).  The class flags of these statements are
    not collected here (the post-fixpoint slot validation covers every statement)."""
    records, _gather_controls, singleton_controls = _classify_assignments(stmts, frozenset(), None, ctx)
    return int(singleton_controls) + sum(1 for r in records if r[3])


def _expression_loop_omissions(nodes, omissions: list) -> None:
    """Comprehension / generator loops inside the EXPRESSIONS of a statement the
    counted walk visits (a ``for`` target/iter, an ``if`` test, a plain
    statement).  They execute iterations the Dim() paths do not represent."""
    for node in nodes:
        if node is None:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                omissions.append(f"comprehension at line {getattr(sub, 'lineno', '?')}: not visited by the loop counter")


def _has_statement_children(stmt: ast.stmt) -> bool:
    """True for a statement that CONTAINS statements (``while``, ``with``, ``try``,
    ``match``, ``async for``, a nested ``def``/``class`` ...)."""
    kinds = (ast.stmt, ast.excepthandler) + ((ast.match_case,) if hasattr(ast, "match_case") else ())
    return any(isinstance(child, kinds) for child in ast.iter_child_nodes(stmt))


def _statement_kind(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.While):
        return f"while {ast.unparse(stmt.test)}"
    if isinstance(stmt, ast.AsyncFor):
        return f"async for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"{type(stmt).__name__} {stmt.name}"
    return {"With": "with", "AsyncWith": "async with", "Try": "try", "TryStar": "try*",
            "Match": "match"}.get(type(stmt).__name__, type(stmt).__name__.lower())


def _loop_paths(stmts: list, consts: dict, prefix: tuple, omissions: list = None, outer: list = None,
                records: list = None, loop_vars: frozenset = frozenset(), flags: set = None, ctx=None,
                lines: list = None, prefix_lines: tuple = (), nest: list = None) -> list:
    """Weighted multiplicative loop paths through a statement list.

    Each entry is ``(path, weight)`` where path is a tuple of
    (dim_name, explicit_size|None) for one root-to-innermost chain of
    ``for v in Dim(...)`` loops and weight is the number of assignment
    statements executed per innermost iteration (a body with two ``+=``
    accumulations does 2x the work per trip — e.g. fused gate+up
    projections).  Total kernel iterations = sum over paths of
    weight * product of the path's dim extents.  Sequential sibling loop
    nests contribute separate paths (their work adds).

    The counted traversal descends into exactly two containers: the BODY of a
    ``for`` statement (a ``Dim()`` loop extends the path; any other ``for``
    keeps the prefix, so its own trip count is not represented) and both
    branches of an ``if`` — whose loop nests, being mutually exclusive, are
    never added together: a loop nest under a branch is an omission (round 4).
    Completeness is a property of THIS walk
    (loop-completeness-4): while it runs it records in ``omissions``
    every loop it does not represent and every statement container it does
    not descend into — a non-``Dim()`` ``for``, a ``for ... else`` body, a
    ``while`` / ``async for`` / ``with`` / ``try`` / ``match`` / nested
    ``def`` (whatever they contain), and comprehension loops in the
    expressions it does visit.  A kernel whose omissions are empty is one the
    counted paths describe in full; nothing is accepted from a container the
    counter never visited.
    """
    paths = []
    if omissions is None:
        omissions = []
    if outer is None:
        outer = []
    if lines is None:
        lines = []
    if nest is None:
        nest = []
    if records is None:
        records = []
    if flags is None:
        flags = set()
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            info = _dim_call_info(stmt.iter, consts)
            # the loop variable(s) this ``for`` binds: never a value factor of a MAC
            _targets = frozenset(n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name))
            inner_vars = loop_vars | _targets
            if info is None:
                omissions.append(f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)} (line {stmt.lineno}): "
                                 + (f"{_DIM_UNBINDABLE} (its dimension name is not a string literal, or its arguments cannot be "
                                    f"bound to Dim(name, parallel=False, size=None)); the loop is not counted"
                                    if _is_dim_call(stmt.iter) else "not a Dim() loop"))
            _expression_loop_omissions((stmt.target, stmt.iter), omissions)
            if stmt.orelse:
                omissions.append(f"for ... else body (line {stmt.orelse[0].lineno}): not visited by the loop counter")
            if info is not None:
                # TARGET-BREAK-COUNT-23 (an earlier change round 10, counter soundness): a Dim() loop whose body -- through nested compound
                # statements, not into nested def / lambda / class bodies -- contains a break / continue / return / raise may
                # not run all its trips; the counted figure is kept but the structure is no longer proven complete (an honest
                # incomplete count instead of a wrong complete one).  Conservative: reachability and the transfer's
                # condition are not emulated.
                for _kind, _ln in _control_transfers(stmt.body, kinds=("break", "continue", "raise")):   # a return inside the loop is the
                    # round-4 return rule's (loop_entry_unresolved: "encloses a conditional return"): not recorded twice
                    omissions.append(f"control_transfer_in_dim_loop (line {stmt.lineno}): {_kind} at line {_ln} inside "
                                     f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)} -- the loop may not run all its trips")
            new_prefix = prefix + ((info,) if info else ())
            # the entry line of EACH Dim() loop along the nest: the registry state a loop reads is the one at ITS
            # entry (an allocation inside an enclosing loop body re-registers on every iteration before the inner
            # loop starts -- TARGET-DSL-DIM-INNER-6)
            new_lines = prefix_lines + ((stmt.lineno,) if info else ())
            # The same structure as a TREE (round 4, loop-carried-10): the converter replays the registry
            # through the nest iteration by iteration, so a registration later in an outer body is carried into
            # the next outer iteration.  A non-Dim() ``for`` (already an omission) attaches its loops to the parent.
            node = {"dim": info[0], "size": info[1], "line": stmt.lineno, "end": getattr(stmt, "end_lineno", stmt.lineno),
                    "col": getattr(stmt, "col_offset", None), "end_col": getattr(stmt, "end_col_offset", None),
                    "children": []} if info else None
            inner = _loop_paths(stmt.body, consts, new_prefix, omissions, outer, records, inner_vars, flags, ctx,
                                lines=lines, prefix_lines=new_lines, nest=node["children"] if node else nest)
            if node is not None:
                nest.append(node)
            if inner:
                paths.extend(inner)
                # Statements at THIS level (index computation, an accumulator
                # init, a per-slot scale / write) execute once per trip of this
                # loop; they are not innermost-path work but they ARE executed
                # in-loop assignments (statement-coverage-1).  Recorded
                # separately so the MAC-bearing innermost count stays what it is.
                level_stmts = [b for b in stmt.body if not isinstance(b, ast.For)]
                level = _count_assigns(level_stmts)
                # one class record per executed statement at this level (issue #59 unit contract); the control
                # evaluations of this level are collected whether or not it has class records (GATHER-CONTROL-JOIN-31)
                classes, controls, scontrols = _classify_assignments(level_stmts, inner_vars, flags, ctx) if new_prefix else ([], 0, 0)
                if level and new_prefix:
                    outer.append((new_prefix, level, list(new_lines)))
                    for kind, extra, gather, single, operands in classes:
                        records.append((new_prefix, kind, extra, gather, single, operands, stmt.lineno))
                for _c in range(controls):
                    records.append((new_prefix, GATHER_CONTROL, 0, True, False, None, stmt.lineno))       # a gather-addressed control test, once per trip
                for _c in range(scontrols):
                    records.append((new_prefix, SINGLETON_CONTROL, 0, False, True, None, stmt.lineno))    # a singleton-reading control test, once per trip
                if node is not None:
                    node["level_assigns"] = int(level)
            else:
                weight = max(1, _count_assigns(stmt.body))
                paths.append((new_prefix, weight))
                lines.append(list(new_lines))
                if node is not None:
                    node["weight"] = int(weight)
                if new_prefix:
                    classes, controls, scontrols = _classify_assignments(stmt.body, inner_vars, flags, ctx)
                    for kind, extra, gather, single, operands in (classes or [("other", 0, False, False, None)]):
                        records.append((new_prefix, kind, extra, gather, single, operands, stmt.lineno))
                    for _c in range(controls):
                        records.append((new_prefix, GATHER_CONTROL, 0, True, False, None, stmt.lineno))
                    for _c in range(scontrols):
                        records.append((new_prefix, SINGLETON_CONTROL, 0, False, True, None, stmt.lineno))
        elif isinstance(stmt, ast.If):
            # Mutually exclusive branches are NOT additive (loop-
            # COMPLETENESS-4, round 4).  Both branches are walked so that any
            # loop inside them is still reported, but a loop NEST under either
            # branch makes the kernel's trip count unprovable from the source
            # alone (which branch runs, and how often, is decided at run time;
            # no test is constant-folded): the branch is recorded as an omission
            # and the kernel fails closed to the equation basis.  An ``if``
            # whose branches contain no loops only gates assignments; its weight
            # is the larger branch (``_count_assigns``), never the sum.
            _expression_loop_omissions((stmt.test,), omissions)
            # Nothing under a branch is counted: its paths are discarded below, so its entry lines, outer-level
            # statements and nest nodes are discarded with them (round 4: the shared ``lines`` list had kept the
            # branch loops' entries, misaligning ``loop_path_lines`` against ``loop_paths`` for the loops after it).
            # The statement-class records and flags of the branch ARE shared, as on the target (an earlier change): the class
            # evidence of a kernel is one record over its whole body.
            body_paths = _loop_paths(stmt.body, consts, prefix, omissions, [], records, loop_vars, flags, ctx, lines=[], prefix_lines=prefix_lines, nest=[])
            else_paths = _loop_paths(stmt.orelse, consts, prefix, omissions, [], records, loop_vars, flags, ctx, lines=[], prefix_lines=prefix_lines, nest=[])
            if body_paths or else_paths:
                omissions.append(f"if {ast.unparse(stmt.test)} (line {stmt.lineno}): loop nest under a branch "
                                 f"(body {len(body_paths)} / else {len(else_paths)} paths): branches are exclusive, "
                                 f"not additive; trip count not provable")
        elif _has_statement_children(stmt):
            omissions.append(f"{_statement_kind(stmt)} (line {stmt.lineno}): not visited by the loop counter")
        else:
            _expression_loop_omissions((stmt,), omissions)
    return paths


_CONTROL_TRANSFER = {ast.Break: "break", ast.Continue: "continue", ast.Return: "return", ast.Raise: "raise"}
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _control_transfers(body, kinds=("break", "continue", "return", "raise")) -> list:
    """``(kind, line)`` of every break / continue / return / raise statement (restricted to *kinds*) reachable from *body* through
    compound statements (if / else / try / with / match / nested loops), NOT descending into nested def / lambda / class bodies."""
    found = []
    stack = list(body)
    while stack:
        node = stack.pop(0)
        if isinstance(node, _NESTED_SCOPES):
            continue
        kind = _CONTROL_TRANSFER.get(type(node))
        if kind:
            if kind in kinds:
                found.append((kind, getattr(node, "lineno", None)))
            continue
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt) or isinstance(child, (ast.excepthandler, ast.match_case)):
                stack.append(child)
    return found


def _module_int_consts(tree: ast.Module) -> dict:
    """Module-level integer constants: a top-level ``NAME = <int literal>``
    that is the ONLY write to that name anywhere in the module scope (a
    later ``NAME *= 2``, ``NAME = 1 + 3``, a conditional rebinding or a
    ``global NAME`` write inside a function disqualifies it, since its value
    at the use would not be the literal)."""
    writes, _shadow = _binding_occurrences(tree.body, module_scope=True)
    counts = {}
    for name in writes:
        counts[name] = counts.get(name, 0) + 1
    consts = {}
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, int)
                and counts.get(stmt.targets[0].id) == 1):
            consts[stmt.targets[0].id] = stmt.value.value
    return consts


def _multiplicative_params(func_def: ast.FunctionDef, param_names: list) -> list:
    """Parameter tensors that appear inside a product anywhere in the body.

    A tensor subscript ``P[...]`` under an ``ast.Mult`` operand means a zero in
    ``P`` kills that product term — the analyzer's zeros-are-free discount
    applies only to such inputs (additive/guard uses don't qualify).  Uses the
    FULL body (loop_expr keeps only innermost statements, which loses
    outer-level gating like ``Y[n, e] = acc * M[n, e]``).
    """
    params = set(param_names or [])
    found = []

    def subtree_params(node):
        return {
            sub.value.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id in params
        }

    # Gather-derived variables: `e = SE[n, k]` / `e = int(SE[n, k])`.  A sparse
    # param indexed by such a variable (RW[n, e]) is NOT a discount candidate:
    # the loop visits only the gathered (non-zero) entries already, so applying
    # its density again would double-discount slot-dense MoE kernels.
    gather_vars = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            v = node.value
            if isinstance(v, ast.Call) and v.args:
                v = v.args[0]
            if isinstance(v, ast.Subscript):
                gather_vars.add(node.targets[0].id)

    def gather_indexed(sub: ast.Subscript) -> bool:
        return any(
            isinstance(n, ast.Name) and n.id in gather_vars
            for n in ast.walk(sub.slice)
        )

    hits = set()
    gathered = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in params and gather_indexed(node):
            gathered.add(node.value.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            hits |= subtree_params(node.left)
            hits |= subtree_params(node.right)
    for name in param_names or []:
        if name in hits and name not in gathered:
            found.append(name)
    return found


def _kernel_signature(node: ast.FunctionDef) -> list:
    """The kernel's positional parameters in call order: ``{name, position, tensor, dtype}`` -- ``dtype`` is the
    ``dtype=`` keyword of a ``Tensor(...)`` annotation (None when absent / not a Tensor parameter)."""
    out = []
    for i, arg in enumerate(node.args.posonlyargs + node.args.args):
        ann = arg.annotation
        is_tensor = isinstance(ann, ast.Call) and ((isinstance(ann.func, ast.Name) and ann.func.id == "Tensor")
                                                   or (isinstance(ann.func, ast.Attribute) and ann.func.attr == "Tensor"))
        dtype = None
        if is_tensor:
            for kw in ann.keywords:
                if kw.arg == "dtype" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    dtype = kw.value.value
        out.append({"name": arg.arg, "position": i, "tensor": bool(is_tensor), "dtype": dtype})
    return out


def _kernel_info_for(node: ast.FunctionDef, tree: ast.Module, source: str, consts: dict) -> dict:
    """The parsed record of ONE @kernel (see :func:`extract_kernel_info`); a pure function of the module text
    (an earlier change DTYPE-OCCURRENCE-CHANNEL-8 re-derives it from the bound source).  an earlier change adds, from the same text: the
    loop nest as a tree with entry lines, the per-record producing-loop lines, the local allocations / name
    bindings / returns and the explicit Dim registrations the converter's registry replay reads."""
    import hashlib
    from .statement_class import normalize_statements
    normalize_statements(node.body)                          # AnnAssign is Assign before anything reads the body
    expr = _extract_kernel_loop_expr(node)
    # Positional parameters in call order (positional-only first), so
    # the converter's zip against the call-site inputs is aligned.
    param_names = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    # Function-local size bindings: `n = X.shape[1]` / `n = 128` make
    # `Dim("D2", size=n)` statically resolvable per call site.
    local_consts = _local_size_consts(node, consts, param_names)
    # One traversal yields both the counted Dim() paths and the list of
    # everything that traversal did not represent: the proof of
    # completeness is the counter's own record (loop-completeness-4).
    omissions: list = []
    outer: list = []
    lines: list = []
    nest: list = []
    records: list = []
    class_flags: set = set()
    class_ctx = _statement_context(node, tree)
    if class_ctx.violations:
        class_flags.add("class_evidence_unproven")           # outside the supported grammar: the whole kernel is unproven
    raw_paths = _loop_paths(node.body, local_consts, (), omissions, outer, records, frozenset(), class_flags, class_ctx,
                            lines=lines, prefix_lines=(), nest=nest)
    allocations, bindings, returns, dim_registrations = _local_bindings(node, local_consts)
    paths = [(list(p), w) for p, w in raw_paths if p]
    path_lines = [ln for (p, w), ln in zip(raw_paths, lines) if p]
    outer_statements = [(list(p), n) for p, n, ln in outer if p]
    outer_lines = [ln for p, n, ln in outer if p]
    # per executed assignment: (Dim prefix, "mac" | "other", standalone multiplies per trip) -- issue #59
    # records: (prefix, kind, extra, gather, single, operands, producing-loop line)
    statement_classes = [(list(p), k, int(e)) for p, k, e, _g, _s, _o, _ln in records if p and k not in _CONTROL_KINDS]
    statement_class_gather = [bool(g) for p, k, _e, g, _s, _o, _ln in records if p and k not in _CONTROL_KINDS]
    statement_class_singleton = [bool(s) for p, k, _e, _g, s, _o, _ln in records if p and k not in _CONTROL_KINDS]
    # control tests (statement-if) evaluated through a gather: (Dim prefix) per evaluation site, once per trip
    gather_controls = [list(p) for p, k, _e, _g, _s, _o, _ln in records if p and k == GATHER_CONTROL]
    # ... and through a singleton read (issue #61), plus the singleton reads outside every loop (once per call)
    singleton_controls = [list(p) for p, k, _e, _g, _s, _o, _ln in records if p and k == SINGLETON_CONTROL]
    singleton_outer = _outer_singleton_reads([b for b in node.body if not isinstance(b, ast.For)], class_ctx)
    # per class record: the tensor names a MAC statement multiplies ([] for other work)
    statement_class_operands = [(dict(o) if o else None) for p, k, _e, _g, _s, o, _ln in records if p and k not in _CONTROL_KINDS]
    # an earlier change reconciliation: the entry LINE of the loop that produced each record, parallel to the lists above --
    # sibling nests share a Dim prefix (lstm_gates: B/G/I, B/G/H and B/G), so the converter's per-record sizing
    # through the loop-nest replay needs the node, not only the prefix
    statement_class_lines = [ln for p, k, _e, _g, _s, _o, ln in records if p and k not in _CONTROL_KINDS]
    gather_control_lines = [ln for p, k, _e, _g, _s, _o, ln in records if p and k == GATHER_CONTROL]
    singleton_control_lines = [ln for p, k, _e, _g, _s, _o, ln in records if p and k == SINGLETON_CONTROL]
    # the kernel's own source segment, bound to the node with its hash (DTYPE-OCCURRENCE-CHANNEL-8): the analyzer
    # re-derives the MAC operand occurrences from it and treats the serialized channel as a validated cache
    seg = ast.get_source_segment(source, node) if source is not None else None
    return {
        "loop_expr": expr if expr else None,
        "param_names": param_names,
        "loop_paths": paths or None,
        # the Dim() paths describe the WHOLE loop nest only when nothing was omitted
        "loop_paths_complete": bool(paths) and not omissions,
        "loop_paths_omitted": omissions or None,
        # assignments at NON-innermost loop levels, (Dim prefix, statements per trip) — statement-coverage-1
        "loop_outer_statements": outer_statements or None,
        # the entry line of each Dim() loop along each path / outer-statement entry: the registry a
        # loop reads is the one at ITS entry (arguments, then the allocations executed before it)
        "loop_path_lines": path_lines or None,
        "loop_outer_lines": outer_lines or None,
        "returned_names": _returned_names(node) or None,
        "loop_statement_classes": statement_classes or None,
        "loop_statement_classes_uncertain": sorted(class_flags) or None,
        "loop_statement_grammar_violations": list(class_ctx.violations) or None,
        "loop_statement_class_gather": statement_class_gather if any(statement_class_gather) else None,
        "loop_gather_controls": gather_controls or None,
        "loop_statement_class_lines": statement_class_lines or None,
        "loop_gather_control_lines": gather_control_lines or None,
        # SINGLETON-SCALE-READ production (issue #61): per record, the statement reads a singleton candidate;
        # the candidates (parameter -> declared rank) whose extent the converter proves from the call's binding
        "loop_statement_class_singleton": statement_class_singleton if any(statement_class_singleton) else None,
        "loop_singleton_reads": dict(class_ctx.singleton_reads) or None,
        "loop_singleton_controls": singleton_controls or None,
        "loop_singleton_outer_reads": singleton_outer or None,
        "loop_singleton_control_lines": singleton_control_lines or None,
        "loop_statement_class_operands": statement_class_operands or None,
        "mult_params": _multiplicative_params(node, param_names) or None,
        "tensor_param_count": _tensor_param_count(node),
        "source_text": seg,
        "source_sha256": hashlib.sha256(seg.encode("utf-8")).hexdigest() if seg is not None else None,
        "module_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest() if source is not None else None,
        "signature": _kernel_signature(node),
        # the kernel's local Tensor allocations: their dim names are RUNTIME registry
        # bindings a Dim() loop reads (TARGET-DSL-DIM-EXTENT-13 / -LOCAL-2 / annotation-binding-1)
        "local_tensor_dims": allocations or None,
        # round 4: the loop nest as a tree (entry/end lines, per-level statement weights) for the iteration-
        # by-iteration registry replay (loop-carried-10); every name binding and every return of the
        # body (return-rebind-9 / TARGET-DSL-EARLY-RETURN-11)
        "loop_nest": nest or None,
        "name_bindings": bindings or None,
        "returns": returns or None,
        # round 5: explicit Dim("U", size=...) registrations outside for-headers (dim-registration-17)
        "dim_registrations": dim_registrations or None,
    }


def extract_kernel_info(py_path: Path) -> dict:
    """Parse *_einsum_lib.py and return per-kernel info including loop_expr and param_names.

    Returns {function_name: {"loop_expr": str|None, "param_names": [str, ...],
    "loop_paths": [([(dim_name, size|None), ...], weight), ...], ...}}.  loop_paths
    is the kernel's explicit ``Dim(...)`` loop structure with a per-path
    statement weight — the ground truth for iteration counting (see
    einsum_converter._kernel_loop_iterations).  Every record is a pure
    function of the module text (:func:`_kernel_info_for`); the kernel's own
    source segment, its sha256, the module's sha256 and the declared signature
    ride along so the analyzer can re-derive the MAC operand occurrences from
    the bound source (:func:`kernel_source_binding`).
    """
    source = Path(py_path).read_text()
    tree = ast.parse(source)
    consts = _module_int_consts(tree)
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_kernel_decorated(node):
            result[node.name] = _kernel_info_for(node, tree, source, consts)
    return result


def module_source_record(py_path: Path) -> dict:
    """``{"sha256", "text"}`` of the DSL module: the graph-level anchor the per-node ``kernel_source`` hashes refer to."""
    import hashlib
    source = Path(py_path).read_text()
    return {"sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(), "text": source}


def kernel_source_binding(module_text: str, kernel_name: str) -> dict:
    """RE-DERIVE, from the module text alone, what the analyzer needs to validate a node's compute-dtype evidence
    (DTYPE-OCCURRENCE-CHANNEL-8): ``{"sha256": <sha256 of the kernel's source segment>, "signature": [...],
    "specs": [<MAC operand spec per MAC class record, in record order>]}``.  A pure function of the text through
    the same parser (:func:`_kernel_info_for` -> ``statement_class.mac_operand_specs``), so a serialized
    channel that disagrees with it cannot have come from this source.  Raises ``KeyError`` when the module has no
    such @kernel."""
    tree = ast.parse(module_text)
    consts = _module_int_consts(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name and _is_kernel_decorated(node):
            info = _kernel_info_for(node, tree, module_text, consts)
            classes = info.get("loop_statement_classes") or []
            operands = info.get("loop_statement_class_operands") or []
            specs = [dict(o) if o else {"occurrences": [], "payloads": [], "scales": [], "conversions": [], "names": [], "unproven": True, "arms_differ": False}
                     for (_p, k, _e), o in zip(classes, operands) if k == "mac"]
            return {"sha256": info["source_sha256"], "signature": info["signature"], "specs": specs}
    raise KeyError(f"no @kernel named {kernel_name!r} in the bound module text")
