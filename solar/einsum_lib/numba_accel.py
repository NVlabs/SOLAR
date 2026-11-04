# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numba JIT acceleration for executing generated einsum_lib code.

Provides a mechanical AST transform that lowers @kernel functions to
@numba.njit(parallel=True) equivalents operating on raw numpy arrays.
The LLM-generated code is unchanged; this module produces a fast
execution path for verification.

Prescan import allowlist (reexport-policy-7).  Before a ``Tensor(...)``
allocation is lowered, the prescan establishes that the name ``Tensor`` -- in
the module and in the kernel scope -- is bound ONLY by imports of the DSL class
verified by object identity (``obj is solar.einsum_lib.core.Tensor``).  To
verify identity the prescan imports the named module, and it imports only
modules under the prefixes in ``_DSL_IMPORT_MODULES`` (``solar.einsum_lib`` and
its submodules).  An import from any other module -- including a private module
that re-exports the very same object -- is declined with the sub-reason
``import_disallowed_prefix:<module>`` (policy: the prescan executes no foreign
module), distinct from ``import_identity_mismatch:<module>.<symbol>`` (the
symbol exists and is not the DSL class), ``import_failed`` /
``import_symbol_missing`` and ``import_unresolvable_relative``.  Every one of
these is reported under the typed decline ``kernel_allocation_callee_unresolved:
<kernel>`` with the sub-reason as the first token of its detail; the fallback
runs the interpreter, whose own outcome (value or error) reaches the caller.

Module effects (an earlier fix round 5).  Executing the module -- its statements, the
class bodies they define, the calls / decorators / class bases they evaluate,
whatever those callees reach, and the harness's call of ``run_model`` -- must
not rebind the module's ``Tensor`` either.  :class:`_ModuleEffects` computes a
SOUND over-approximation of that by a fixpoint over every scope of the module
(defs, lambdas, class bodies; call edges, argument references, nested scopes;
cycles converge); anything it cannot resolve to a non-writer is an unresolved
effect and declines (``module_binding_uncertain``) rather than passing.  Under
``from __future__ import annotations`` annotations are never evaluated: the
prescan does not visit them and the lowered module keeps the future import.
The original source is compile-validated (``compile(..., dont_inherit=True)``)
before any rewrite; a source Python rejects is declined
``kernel_source_invalid:<kernel>`` and never lowered.  Module-level statements are
lowered by :class:`_ModuleStatementLowerer`, which touches only established DSL
values; a ``.data`` read it cannot establish is declined
``kernel_module_statement_unresolved:<kernel>`` (an earlier fix round 7).
"""

import ast
import copy
import math
import textwrap
from typing import Callable, List, Optional

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


_INT_DTYPE_MAP = {"int8": "int8", "int32": "int32", "int64": "int64"}


def _resolve_np_dtype_attr(keywords, default_float: str) -> str:
    """Return the numpy dtype attribute for a Tensor call's keyword list.

    If the call has ``dtype="int8|int32|int64"``, returns the matching numpy
    integer type string; otherwise returns *default_float* (e.g. ``"float64"``).
    """
    for kw in keywords:
        if kw.arg == "dtype" and isinstance(kw.value, ast.Constant):
            mapped = _INT_DTYPE_MAP.get(kw.value.value)
            if mapped is not None:
                return mapped
    return default_float


_NOT_A_LITERAL = object()


def _is_kernel_decorated(func_def: ast.FunctionDef) -> bool:
    """Check if a function has a @kernel decorator."""
    for dec in func_def.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "kernel":
            return True
    return False


class _KernelLowerer(ast.NodeTransformer):
    """Transform a @kernel function body to operate on raw numpy arrays.

    Transforms applied:
    - Tensor(B=X.shape[0], ..., dtype=...) -> np.zeros((...), dtype=np.float64|int64)
    - Tensor(..., fill=val) -> np.full((...), val, dtype=np.float64|int64)
    - for x in Dim("X"): -> for x in numba.prange(...) or range(...)
      (a prange iterates a fresh name and rebinds ``x = np.intp(...)`` so
      the body never sees Numba's unsigned gufunc index; see
      ``_prange_index_rebind``)
    - Strip type annotations from parameters
    - max(a, b) handled natively by numba
    - exp/log/sqrt/tanh/cos/sin/floor/log2 -> math.* equivalents
    - round(x) handled natively by numba (round-half-even, matches np.round)
    """

    def __init__(self, use_fp64: bool = True):
        self._dim_to_shape = {}
        self._np_float_attr = "float64" if use_fp64 else "float32"
        # Dim loops whose size could not be established from parameter
        # annotations, output constructors or an explicit size= kwarg.
        # Lowering such a loop to range(0) would silently skip the body
        # and return all-zero outputs, while the DSL contract
        # (core.Dim/_registry.get_size) raises for unregistered dims --
        # the caller must decline lowering instead.
        self.unresolved_dims: List[str] = []
        # Every name the current kernel mentions: fresh prange index names
        # must not collide with any of them.
        self._kernel_names: set = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._dim_to_shape = {}
        self._kernel_names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self._kernel_names.update(a.arg for a in node.args.args)
        self._extract_dims_from_annotations(node)
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None
        node.body = [self.visit(stmt) for stmt in node.body]
        node.body = [s for s in node.body if s is not None]
        return node

    def _extract_dims_from_annotations(self, node: ast.FunctionDef):
        """Pre-populate dim-to-shape from parameter Tensor annotations.

        E.g. ``A: Tensor("B,M,K")`` registers B=A.shape[0], M=A.shape[1], K=A.shape[2].
        This captures reduction dimensions that don't appear in output constructors.
        """
        for arg in node.args.args:
            ann = arg.annotation
            if ann is None:
                continue
            dims_str = self._extract_tensor_dims_string(ann)
            if dims_str is None:
                continue
            dim_names = [d.strip() for d in dims_str.split(",")]
            param_name = arg.arg
            for axis, dname in enumerate(dim_names):
                if dname not in self._dim_to_shape:
                    self._dim_to_shape[dname] = ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id=param_name, ctx=ast.Load()),
                            attr="shape",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=axis),
                        ctx=ast.Load(),
                    )

    @staticmethod
    def _extract_tensor_dims_string(ann: ast.AST) -> Optional[str]:
        """Extract the dims string from ``Tensor("B,M,K", ...)``."""
        if (isinstance(ann, ast.Call) and
                isinstance(ann.func, ast.Name) and
                ann.func.id == "Tensor" and
                ann.args and
                isinstance(ann.args[0], ast.Constant) and
                isinstance(ann.args[0].value, str)):
            return ann.args[0].value
        return None

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.AST]:
        if self._is_tensor_constructor(node.value):
            return self._lower_tensor_constructor(node)
        self.generic_visit(node)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        """Transform ``for x in Dim("X", parallel=True)`` to ``numba.prange``."""
        dim_name = self._extract_dim_name(node.iter)
        if dim_name is not None:
            explicit_size = self._extract_size_kwarg(node.iter)
            invalid_bound = False
            if (isinstance(explicit_size, ast.Constant)
                    and explicit_size.value is None):
                # core.Dim(size=None) means "use the registered size".
                explicit_size = None
            elif explicit_size is not None:
                try:
                    bound = ast.literal_eval(explicit_size)
                except (ValueError, TypeError, SyntaxError, MemoryError,
                        RecursionError):
                    bound = _NOT_A_LITERAL  # shape expression / name: keep
                if bound is not _NOT_A_LITERAL and (
                        isinstance(bound, bool) or not isinstance(bound, int)
                        or bound < 0):
                    # A statically-known bound (including signed literals
                    # such as -0.0 / +1.5) that is not a non-negative int can
                    # never be a valid extent: never emit range('x') etc.
                    invalid_bound = True
            if invalid_bound:
                shape_expr = None
            elif explicit_size is not None:
                shape_expr = explicit_size
            else:
                shape_expr = self._dim_to_shape.get(dim_name)
            if shape_expr is None:
                # Unresolvable loop bound: record it so numba_lower
                # declines -- never a silent zero-iteration range(0).
                self.unresolved_dims.append(dim_name)
                shape_expr = ast.Constant(value=0)

            is_parallel = self._has_parallel_flag(node.iter)
            rebind = None
            if is_parallel:
                iter_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="numba", ctx=ast.Load()),
                        attr="prange",
                        ctx=ast.Load(),
                    ),
                    args=[shape_expr],
                    keywords=[],
                )
                rebind = self._prange_index_rebind(node)
            else:
                iter_call = ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[shape_expr],
                    keywords=[],
                )
            node.iter = iter_call
        else:
            rebind = None

        node.body = [self.visit(stmt) for stmt in node.body]
        node.body = [s for s in node.body if s is not None]
        if rebind is not None:
            node.body.insert(0, rebind)
        # The for/else branch executes too: an unresolved Dim loop there
        # must be recorded (and lowered) exactly like one in the body.
        node.orelse = [self.visit(stmt) for stmt in node.orelse]
        node.orelse = [s for s in node.orelse if s is not None]
        return node

    def _prange_index_rebind(self, node: ast.For) -> Optional[ast.Assign]:
        """Iterate a fresh name and rebind the DSL index to ``np.intp``.

        Numba's parfor lowering moves the body of an outermost ``prange``
        into a gufunc that iterates a ``uintp`` schedule, so inside it the
        loop index is ``uint64`` although the outer typing sees ``int64``.
        Binary operators with a signed operand recover ``int64``, but
        ``min``/``max`` and conditional expressions unify their operands,
        and ``unify(uint64, int64)`` is ``float64``: a DSL index such as
        ``t = min(m, nt - 1); X[t // ns, t % ns, h]`` then fails only in
        native parfor lowering with ``getitem(..., Tuple(float64, ...))``
        (finding SHAPE0-205-NUMBA-FLOAT-INDEX-1).  ``Dim`` indices are
        Python ints in the DSL, so the lowered body gets the signed
        ``np.intp`` value under the DSL name::

            for _m_prange in numba.prange(mp):
                m = np.intp(_m_prange)
                ...

        Returns the rebind statement, or None when the loop target is not a
        plain name (the DSL contract only allows a name).
        """
        if not isinstance(node.target, ast.Name):
            return None
        dsl_name = node.target.id
        fresh = f"_{dsl_name}_prange"
        counter = 0
        while fresh in self._kernel_names:
            counter += 1
            fresh = f"_{dsl_name}_prange{counter}"
        self._kernel_names.add(fresh)
        node.target = ast.Name(id=fresh, ctx=ast.Store())
        return ast.Assign(
            targets=[ast.Name(id=dsl_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="intp",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=fresh, ctx=ast.Load())],
                keywords=[],
            ),
        )

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ("exp", "log", "sqrt", "tanh", "cos", "sin",
                                "floor", "log2", "erf"):
                node.func = ast.Attribute(
                    value=ast.Name(id="math", ctx=ast.Load()),
                    attr=node.func.id,
                    ctx=ast.Load(),
                )
            elif node.func.id == "fp32" and len(node.args) == 1 and not node.keywords:
                # the DSL's fp32 rounding-boundary cast: a float32 round trip,
                # never inline multiply/power arithmetic (snap-mac-1)
                np_name = lambda attr: ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=attr, ctx=ast.Load())
                return ast.Call(func=np_name("float64"),
                                args=[ast.Call(func=np_name("float32"), args=[node.args[0]], keywords=[])],
                                keywords=[])
        return node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Rewrite a 0-d read ``S[()]`` as ``S.item()``.

        ``S[()]`` is how the DSL reads a scalar ``Tensor("")``, and numpy accepts
        it, but by the time the value crosses the njit boundary it can arrive as
        a bare ``np.float64`` rather than a 0-d array -- and numba has no
        ``getitem(float64, Tuple())``.  It rejects the whole kernel, which used
        to mean a silent fall back to the interpreter.  ``.item()`` reads the
        scalar out of every form the value can take.
        """
        self.generic_visit(node)
        if (isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Tuple)
                and not node.slice.elts):
            return ast.Call(
                func=ast.Attribute(value=node.value, attr="item", ctx=ast.Load()),
                args=[], keywords=[],
            )
        return node

    def _is_tensor_constructor(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Tensor":
            return True
        return False

    def _lower_tensor_constructor(self, assign_node: ast.Assign) -> ast.Assign:
        """Convert Tensor(B=X.shape[0], ..., fill=..., dtype=...) to np.zeros/np.full."""
        call = assign_node.value
        shape_elts = []
        fill_value = None

        for kw in call.keywords:
            if kw.arg in ("dtype", "role", "sparsity"):
                continue
            if kw.arg == "fill":
                fill_value = kw.value
                continue
            shape_elts.append(kw.value)
            self._register_dim_shape(kw.arg, kw.value)

        shape_tuple = ast.Tuple(elts=shape_elts, ctx=ast.Load())

        np_attr = _resolve_np_dtype_attr(call.keywords, self._np_float_attr)
        dtype_node = ast.Attribute(
            value=ast.Name(id="np", ctx=ast.Load()),
            attr=np_attr,
            ctx=ast.Load(),
        )

        if fill_value is not None:
            new_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="full",
                    ctx=ast.Load(),
                ),
                args=[shape_tuple, fill_value],
                keywords=[ast.keyword(arg="dtype", value=dtype_node)],
            )
        else:
            new_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="zeros",
                    ctx=ast.Load(),
                ),
                args=[shape_tuple],
                keywords=[ast.keyword(arg="dtype", value=dtype_node)],
            )

        assign_node.value = new_call
        return assign_node

    def _register_dim_shape(self, dim_name: str, size_expr: ast.AST):
        """Record the shape expression for a dim name so Dim loops can reference it."""
        self._dim_to_shape[dim_name] = size_expr

    def _extract_dim_name(self, node: ast.AST) -> Optional[str]:
        """Extract dim name from ``Dim("X")`` or ``Dim("X", parallel=True)``."""
        if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "Dim" and
                node.args and
                isinstance(node.args[0], ast.Constant) and
                isinstance(node.args[0].value, str)):
            return node.args[0].value
        return None

    @staticmethod
    def _has_parallel_flag(node: ast.AST) -> bool:
        """Return True if ``Dim("X", parallel=True)`` has the parallel keyword."""
        if not isinstance(node, ast.Call):
            return False
        for kw in node.keywords:
            if kw.arg == "parallel":
                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return True
        return False

    @staticmethod
    def _extract_size_kwarg(node: ast.AST) -> Optional[ast.AST]:
        """Extract the ``size=`` expression from ``Dim("G", size=expr)``."""
        if not isinstance(node, ast.Call):
            return None
        for kw in node.keywords:
            if kw.arg == "size":
                return kw.value
        return None


def _extract_kernel_return_dims(tree: ast.Module) -> dict:
    """Pre-scan ``@kernel`` functions for return-type dim annotations.

    Returns ``{func_name: ["DIM1", "DIM2", ...]}`` for every kernel whose
    return annotation is ``Tensor("DIM1,DIM2,...", ...)``.
    """
    dims: dict = {}
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and _is_kernel_decorated(node)):
            continue
        ret = node.returns
        if ret is None or not isinstance(ret, ast.Call) or not ret.args:
            continue
        arg0 = ret.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            parsed = [d.strip() for d in arg0.value.split(",") if d.strip()]
            if parsed:
                dims[node.name] = parsed
    return dims


_NARROW_FLOAT_DTYPES_SET = frozenset({"bf16", "fp16"})


class KernelOutputDtypeUnresolved(Exception):
    """A ``@kernel``'s output dtype cannot be established statically.

    Raised by :func:`_extract_kernel_output_dtypes` and turned into a typed
    lowering decline (``numba_lower`` returns None and reports
    ``kernel_output_dtype_<unestablished|conflict>:<kernel>`` through
    ``on_decline``).  Final rounding must never be skipped silently
    (SHAPE0-BF16-FINAL-ROUNDING-1): a result whose declared dtype is unknown
    is not lowered; the interpreter path, which rounds by the allocated
    Tensor's dtype, runs instead.
    """

    def __init__(self, kernel: str, kind: str, detail: str, code: Optional[str] = None):
        reason = code or f"kernel_output_dtype_{kind}:{kernel}"
        super().__init__(f"{reason}: {detail}")
        self.kernel, self.kind, self.detail = kernel, kind, detail
        self.reason = reason
        self.detail = detail


_UNKNOWN_DTYPE = "<unknown>"      # an EXPLICIT dtype that cannot be resolved statically (never defaulted)
_DSL_DEFAULT_DTYPE = "fp32"        # the DSL's default when ``dtype`` is ABSENT (core.Tensor(..., dtype="fp32"))


def _bound_names(node: ast.AST) -> set:
    """Every name a statement subtree BINDS (assignment targets incl. tuple
    unpacking, augmented / annotated assignment, ``for`` / ``with`` targets,
    walrus, ``del``, ``import``, nested ``def`` / ``class``, ``except ... as``,
    ``match`` captures).  Used to decide which scope a dtype name belongs to
    (const-5)."""
    out: set = set()

    def targets(tgt):
        if isinstance(tgt, ast.Name):
            out.add(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for e in tgt.elts:
                targets(e)
        elif isinstance(tgt, ast.Starred):
            targets(tgt.value)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for tg in sub.targets:
                targets(tg)
        elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
            targets(sub.target)
        elif isinstance(sub, (ast.For, ast.AsyncFor)):
            targets(sub.target)
        elif isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                if item.optional_vars is not None:
                    targets(item.optional_vars)
        elif isinstance(sub, ast.NamedExpr):
            targets(sub.target)
        elif isinstance(sub, ast.Delete):
            for tg in sub.targets:
                targets(tg)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            out.add(sub.name)
        elif isinstance(sub, ast.MatchAs) and sub.name:
            out.add(sub.name)
        elif isinstance(sub, ast.MatchStar) and sub.name:
            out.add(sub.name)
        elif isinstance(sub, ast.MatchMapping) and sub.rest:
            out.add(sub.rest)
    return out


def _syntactic_writes(node: ast.AST) -> dict:
    """``{name: count}`` of EVERY syntactic binding of a name anywhere under
    ``node`` — one generic :func:`ast.walk`, no per-construct allow-list
    (const-11): a ``Name`` in Store / Del context (assignment, augmented
    / annotated assignment, ``for`` / ``with`` / comprehension targets, walrus,
    ``del``) wherever it occurs — statement bodies, defaults, annotations,
    decorators, bases, keywords, comprehensions, lambdas —, ``global`` /
    ``nonlocal`` declarations, ``def`` / ``class`` / import names, function
    and lambda parameters, ``except ... as``, ``match`` captures."""
    writes: dict = {}

    def bump(name: str) -> None:
        writes[name] = writes.get(name, 0) + 1

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bump(sub.id)
        elif isinstance(sub, (ast.Global, ast.Nonlocal)):
            for n in sub.names:
                bump(n)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bump(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                bump((alias.asname or alias.name).split(".")[0])
        elif isinstance(sub, ast.arg):
            bump(sub.arg)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bump(sub.name)
        elif isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
            bump(sub.name)
        elif isinstance(sub, ast.MatchMapping) and sub.rest:
            bump(sub.rest)
    return writes


def _module_string_constants(tree: ast.Module) -> dict:
    """Round 9 (literal-only delivery): NO dtype name is folded -- a ``dtype=``
    that is not a string literal is a typed decline (:func:`_explicit_tensor_dtype`).
    Kept as a thin compatibility shim for earlier control files: it reports no
    static constant."""
    return {}


def _module_dynamic_namespace(tree: ast.Module) -> Optional[str]:
    """Round 9 shim: no static judgement of the module is made (see
    :func:`_module_string_constants`); always None."""
    return None


_module_init_not_static = _module_dynamic_namespace
_module_reflective_construct = _module_dynamic_namespace


def _explicit_tensor_dtype(call: ast.AST, consts: Optional[dict] = None) -> Optional[str]:
    """The ``dtype=`` of a ``Tensor(...)`` call / annotation: ``None`` when the
    keyword is ABSENT (the DSL default applies); the string when it is a string
    LITERAL; and :data:`_UNKNOWN_DTYPE` for any other expression -- a name, an
    attribute, a call, anything (an earlier fix round 9, literal-only delivery: an
    explicit non-literal dtype is never folded, never evaluated at call time
    and never defaulted; the kernel is declined with a typed reason and the
    interpreter runs it).  ``consts`` is accepted for signature compatibility
    with earlier control files and ignored."""
    if not isinstance(call, ast.Call):
        return None
    for kw in call.keywords:
        if kw.arg == "dtype":
            v = kw.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
            return _UNKNOWN_DTYPE
    return None


def _is_tensor_ctor(call: ast.AST) -> bool:
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "Tensor")


def _is_plain_handler_type(expr) -> bool:
    """A handler type the walk models: absent (bare ``except``), a Name, a
    dotted Attribute chain of Names, or a Tuple of those (flow-15)."""
    def plain(e) -> bool:
        if isinstance(e, ast.Name):
            return True
        if isinstance(e, ast.Attribute):
            return plain(e.value)
        return False
    if expr is None:
        return True
    if isinstance(expr, ast.Tuple):
        return all(plain(e) for e in expr.elts)
    return plain(expr)


class _FlowNotModelled(Exception):
    """A control-flow construct the return-site walk does not model soundly
    (``try/finally``, ``with``, ``async`` constructs, a loop that does not
    reach its fixed point): the caller declines with a typed reason instead of
    guessing a dtype (flow-4)."""


_LOOP_FIXPOINT_PASSES = 1000     # the lattice is finite (name -> set of dtype strings), so this is never reached


def _return_value_dtypes(node: ast.FunctionDef, consts: dict) -> list:
    """Return-site provenance of a ``@kernel`` (provenance-1 / FLOW-4).

    An abstract interpretation of the body over a per-name environment that
    maps each local to the SET of dtypes its reaching definition may carry.
    Bindings: a ``Tensor(...)`` constructor binds its dtype (literal /
    resolved name / absent -> DSL default / explicit-unknown), ``x = y``
    copies ``y``'s set, a parameter starts from its annotation, and ANY other
    binding of a name kills it to unknown.  Control flow is modelled with
    distinct exits: a block yields its fall-through environment plus the
    environments carried by every ``break`` and ``continue`` it executes;
    ``if`` and ``match`` join the fall-throughs of every branch WITH the
    incoming path when no branch is taken (``if`` without ``else``, no
    irrefutable ``case``); a loop is iterated to a genuine fixed point over
    its back edges (fall-through and ``continue``), its exit is the join of
    the zero-trip path, the fixed point and every ``break`` environment, and
    the ``else`` block runs on the non-``break`` exits; ``return`` records the
    returned value's set and ends the path; ``raise`` ends the path.  A
    ``try`` handler starts from the join of the incoming environment, the
    body's fall-through AND every environment a statement of the body (at any
    nesting depth) may leave behind when it raises — an exception can exit
    the body after any prefix, so a rebinding before the raise reaches the
    handler (flow-6).
    Constructs not modelled soundly — ``try`` with a ``finally`` (which can
    override a return), ``with``, ``async`` loops / ``with``, a loop that does
    not converge, any assignment EXPRESSION anywhere inside a ``try`` or a
    ``match`` statement (one walk over the whole statement: body, handler
    types / names / bodies, else, finally, subject, patterns, guards —
    flow-10 / -15), and a handler type that is not a plain name /
    attribute / tuple of those — raise :class:`_FlowNotModelled` and the
    kernel is declined.
    Statements after an exit in a block are dead and never bind.  Returns one
    frozenset per reachable ``return``; ``_UNKNOWN_DTYPE`` marks a value that
    cannot be traced.
    """
    UNK = frozenset([_UNKNOWN_DTYPE])

    def names_bound(target: ast.AST) -> list:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for e in target.elts for n in names_bound(e)]
        if isinstance(target, ast.Starred):
            return names_bound(target.value)
        return []                                   # Y[i] = ... / attribute store: no rebinding

    def value_set(expr: ast.AST, env: dict) -> frozenset:
        if isinstance(expr, ast.Name):
            return env.get(expr.id, UNK)
        if _is_tensor_ctor(expr):
            dt = _explicit_tensor_dtype(expr, consts)
            return frozenset([_DSL_DEFAULT_DTYPE if dt is None else dt])
        if isinstance(expr, ast.IfExp):
            return value_set(expr.body, env) | value_set(expr.orelse, env)
        return UNK

    def walrus_kill(expr: ast.AST, env: dict) -> None:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.NamedExpr):
                for n in names_bound(sub.target):
                    env[n] = UNK

    def join(*envs) -> Optional[dict]:
        envs = [e for e in envs if e is not None]
        if not envs:
            return None
        keys = set().union(*(e.keys() for e in envs))
        out: dict = {}
        for k in keys:
            acc: frozenset = frozenset()
            for e in envs:
                acc = acc | e.get(k, UNK)           # unbound on a path -> unknown on that path
            out[k] = acc
        return out

    returns: list = []
    # Active ``try`` bodies, innermost last: every environment that a statement
    # executed inside the body may leave behind when it raises (flow-6).
    exc_prefixes: list = []
    # Depth of guarded regions (``try`` body / ``except`` / try-``else`` /
    # ``match`` case): an assignment EXPRESSION there can rebind a returned name
    # part-way through a statement that then raises or dispatches, which the
    # statement-prefix join cannot see (flow-10).  Any ``NamedExpr``
    # inside a guarded region is declined (fail closed); plain assignments keep
    # the statement-prefix join.
    guard_depth: list = [0]

    class Exits:
        __slots__ = ("fall", "breaks", "continues")

        def __init__(self, fall, breaks=None, continues=None):
            self.fall, self.breaks, self.continues = fall, list(breaks or []), list(continues or [])

    def run_block(stmts: list, env: dict) -> Exits:
        env = dict(env)
        breaks: list = []
        continues: list = []
        for s in stmts:
            if guard_depth[0] and any(isinstance(sub, ast.NamedExpr) for sub in ast.walk(s)):
                raise _FlowNotModelled("exception/expression provenance not modelled "
                                       "(assignment expression inside a try / except / match body)")
            for trace in exc_prefixes:
                trace.append(dict(env))         # an exception raised by (or inside) ``s`` leaves THIS environment
            for expr in ast.iter_child_nodes(s):
                if isinstance(expr, ast.expr):
                    walrus_kill(expr, env)
            if isinstance(s, ast.Return):
                returns.append(value_set(s.value, env) if s.value is not None else UNK)
                return Exits(None, breaks, continues)
            if isinstance(s, ast.Raise):
                return Exits(None, breaks, continues)
            if isinstance(s, ast.Break):
                breaks.append(dict(env)); return Exits(None, breaks, continues)
            if isinstance(s, ast.Continue):
                continues.append(dict(env)); return Exits(None, breaks, continues)
            if isinstance(s, ast.Assign):
                vs = value_set(s.value, env)
                for tgt in s.targets:
                    if isinstance(tgt, ast.Name):
                        env[tgt.id] = vs
                    else:
                        for n in names_bound(tgt):
                            env[n] = UNK
            elif isinstance(s, ast.AnnAssign):
                if isinstance(s.target, ast.Name):
                    env[s.target.id] = value_set(s.value, env) if s.value is not None else UNK
            elif isinstance(s, ast.AugAssign):
                for n in names_bound(s.target):
                    env[n] = UNK
            elif isinstance(s, ast.Delete):
                for tgt in s.targets:
                    for n in names_bound(tgt):
                        env[n] = UNK
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for alias in s.names:
                    env[(alias.asname or alias.name).split(".")[0]] = UNK
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                env[s.name] = UNK
            elif isinstance(s, ast.If):
                a = run_block(s.body, env)
                b = run_block(s.orelse, env) if s.orelse else Exits(dict(env))   # no else: the incoming path falls through
                breaks += a.breaks + b.breaks; continues += a.continues + b.continues
                env = join(a.fall, b.fall)
                if env is None:
                    return Exits(None, breaks, continues)
            elif isinstance(s, (ast.For, ast.While)):
                entry = dict(env)
                if isinstance(s, ast.For):
                    for n in names_bound(s.target):
                        entry[n] = UNK
                state = entry
                loop_breaks: list = []
                converged = False
                for _ in range(_LOOP_FIXPOINT_PASSES):
                    r = run_block(s.body, state)
                    loop_breaks = r.breaks
                    nxt = join(state, r.fall, *r.continues)       # back edges re-enter the loop
                    if nxt == state:
                        converged = True
                        break
                    state = nxt
                if not converged:
                    raise _FlowNotModelled("loop did not reach a fixed point")
                normal_exit = join(entry, state)                   # zero-trip path and normal termination
                if s.orelse:
                    e = run_block(s.orelse, normal_exit)
                    breaks += e.breaks; continues += e.continues
                    normal_exit = e.fall
                env = join(normal_exit, *loop_breaks)
                if env is None:
                    return Exits(None, breaks, continues)
            elif isinstance(s, (ast.AsyncFor, ast.With, ast.AsyncWith)):
                raise _FlowNotModelled(type(s).__name__ + " is not modelled")
            elif isinstance(s, ast.Try):
                if s.finalbody:
                    raise _FlowNotModelled("try/finally is not modelled (a finally block can override the return)")
                # ONE generic walk over the ENTIRE try statement — body, every
                # handler's type / name / body, else, finally — before anything
                # is modelled (flow-15): any assignment expression inside
                # declines; a handler type that is not a Name / Attribute / Tuple
                # of those (a conditional, a subscript, a call, a BoolOp ...)
                # executes arbitrary code while the handler is selected and is
                # not modelled either.
                if any(isinstance(sub, ast.NamedExpr) for sub in ast.walk(s)):
                    raise _FlowNotModelled("exception/expression provenance not modelled "
                                           "(assignment expression inside a try statement)")
                for h in s.handlers:
                    if not _is_plain_handler_type(h.type):
                        raise _FlowNotModelled("exception/expression provenance not modelled "
                                               f"(handler type {ast.unparse(h.type)!r} is not a plain name / attribute / tuple)")
                body_prefixes: list = []
                exc_prefixes.append(body_prefixes)
                guard_depth[0] += 1
                try:
                    body = run_block(s.body, env)
                finally:
                    exc_prefixes.pop()
                    guard_depth[0] -= 1
                breaks += body.breaks; continues += body.continues
                # a handler may start after ANY prefix of the body: the incoming env, the body's fall-through, and
                # every environment a statement of the body may leave behind when it raises (flow-6)
                paths: list = []
                guard_depth[0] += 1
                try:
                    for h in s.handlers:
                        start = join(env, body.fall, *body_prefixes) or dict(env)
                        if h.name:
                            start = dict(start); start[h.name] = UNK
                        hr = run_block(h.body, start)
                        breaks += hr.breaks; continues += hr.continues
                        paths.append(hr.fall)
                    after_body = body.fall
                    if s.orelse and after_body is not None:
                        er = run_block(s.orelse, after_body)
                        breaks += er.breaks; continues += er.continues
                        after_body = er.fall
                finally:
                    guard_depth[0] -= 1
                env = join(after_body, *paths)
                if env is None:
                    return Exits(None, breaks, continues)
            elif hasattr(ast, "Match") and isinstance(s, ast.Match):
                if any(isinstance(sub, ast.NamedExpr) for sub in ast.walk(s)):      # ENTIRE match: subject, patterns, guards, bodies
                    raise _FlowNotModelled("exception/expression provenance not modelled "
                                           "(assignment expression inside a match statement)")
                falls: list = []
                irrefutable = False
                guard_depth[0] += 1
                try:
                    for c in s.cases:
                        start = dict(env)
                        for n in _bound_names(c.pattern):
                            start[n] = UNK
                        cr = run_block(c.body, start)
                        breaks += cr.breaks; continues += cr.continues
                        falls.append(cr.fall)
                        if c.guard is None and isinstance(c.pattern, ast.MatchAs) and c.pattern.pattern is None:
                            irrefutable = True                          # ``case _:`` / ``case x:`` always matches
                finally:
                    guard_depth[0] -= 1
                env = join(*falls, *([] if irrefutable else [dict(env)]))   # the unmatched path keeps the incoming env
                if env is None:
                    return Exits(None, breaks, continues)
            # Expr / Pass / Assert / Global / Nonlocal: no binding of interest here
        return Exits(env, breaks, continues)

    env0: dict = {}
    for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
        if a.annotation is not None and _is_tensor_ctor(a.annotation):
            dt = _explicit_tensor_dtype(a.annotation, consts)
            env0[a.arg] = frozenset([_DSL_DEFAULT_DTYPE if dt is None else dt])
        else:
            env0[a.arg] = UNK
    if node.args.vararg:
        env0[node.args.vararg.arg] = UNK
    if node.args.kwarg:
        env0[node.args.kwarg.arg] = UNK
    top = run_block(node.body, env0)
    if top.breaks or top.continues:
        raise _FlowNotModelled("break/continue outside a loop")
    if top.fall is not None:
        returns.append(UNK)                   # falling off the end returns None: untraceable
    return returns


def _kernel_output_dtype(node: ast.FunctionDef, consts: Optional[dict] = None) -> Optional[str]:
    """The DECLARED output dtype of a ``@kernel``: the return annotation's
    ``dtype`` when present (a string literal; anything else is declined),
    checked against — or, when absent, taken from — the
    return-site provenance of every reachable ``return``
    (:func:`_return_value_dtypes`).  This is the dtype the interpreter's
    ``kernel`` wrapper rounds by (``result.dtype``).  Raises
    :class:`KernelOutputDtypeUnresolved`:

    * ``unestablished`` — an explicit dtype that cannot be resolved, a returned
      value that is not a traced allocation / parameter / alias on every path,
      or control flow the walk does not model soundly (declined, never guessed);
    * ``conflict`` — paths (or returns) disagree on the dtype, or the
      annotation and the traced allocation are both explicit and differ.
    """
    declared = _explicit_tensor_dtype(node.returns, consts) if node.returns is not None else None
    if declared == _UNKNOWN_DTYPE:
        raise KernelOutputDtypeUnresolved(node.name, "unestablished",
                                          "the return annotation's dtype is explicit but not a string literal")
    try:
        returned = _return_value_dtypes(node, consts)
    except _FlowNotModelled as exc:
        raise KernelOutputDtypeUnresolved(node.name, "unestablished", f"control flow not modelled soundly: {exc}") from exc
    union: set = set()
    for r in returned:
        union |= set(r)
    if not returned:
        raise KernelOutputDtypeUnresolved(node.name, "unestablished", "the kernel has no return statement")
    if _UNKNOWN_DTYPE in union:
        raise KernelOutputDtypeUnresolved(
            node.name, "unestablished",
            "a returned value cannot be traced to a Tensor(...) allocation or parameter on every path "
            "(rebinding to an expression / call / subscript, an explicit non-literal dtype, or a value from outside the kernel)")
    if len(union) > 1:
        raise KernelOutputDtypeUnresolved(node.name, "conflict", f"return paths disagree on the output dtype {sorted(union)}")
    from_alloc = next(iter(union))
    if declared is not None and declared != from_alloc:
        raise KernelOutputDtypeUnresolved(
            node.name, "conflict",
            f"the return annotation declares dtype={declared!r} but the returned allocation is dtype={from_alloc!r}")
    return declared if declared is not None else from_alloc


_DSL_KEYWORD_CALLEES = frozenset({"Tensor", "Dim"})      # the only DSL calls a kernel body spells with keywords
_TENSOR_OPTION_KEYWORDS = frozenset({"dtype", "role", "sparsity", "fill"})


_DSL_IMPORT_MODULES = ("solar.einsum_lib",)      # the prescan IMPORT ALLOWLIST: only modules under these prefixes are imported to resolve a binding
_LOWERABLE_BUILTINS = frozenset({"int", "float", "bool", "range", "len", "enumerate", "zip", "abs", "min", "max", "round"})
_NAMESPACE_CALLS = frozenset({"exec", "eval", "globals", "vars", "locals", "setattr", "delattr", "__import__"})   # a call that may rewrite a namespace
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _import_identity_reason(stmt: ast.ImportFrom, alias: ast.alias, expected: str = "Tensor"):
    """``None`` when ``from <module> import <alias>`` binds the DSL ``Tensor`` class
    itself (or, round 10, the DSL object ``core.<expected>`` -- a lowered builtin
    such as ``apply_mask``), resolved by IDENTITY; else ``(subcode, detail)`` saying
    truthfully WHY identity could not be established (reexport-policy-7):

    * ``import_unresolvable_relative`` -- a relative import names no module the
      prescan can resolve;
    * ``import_disallowed_prefix:<module>`` -- the module is outside the prescan
      import allowlist ``_DSL_IMPORT_MODULES``; the prescan imports no other
      module, so the object's identity is NEVER examined (an identity-preserving
      re-export from a private module is declined here by policy, not because
      the object differs -- see the module docstring);
    * ``import_failed:<module>`` / ``import_symbol_missing:<module>.<name>`` --
      the allowlisted module does not import, or has no such attribute;
    * ``import_identity_mismatch:<module>.<name>`` -- the symbol exists and is
      not ``core.Tensor`` (``Dim``, a subclass, anything else).
    The spelling of the alias proves nothing (``from solar.einsum_lib import Dim
    as Tensor`` binds the name ``Tensor`` to ``Dim``)."""
    spelled = ast.unparse(stmt)
    if stmt.level:
        return "import_unresolvable_relative", f"the relative import at line {stmt.lineno} ({spelled!r}) cannot be resolved at prescan"
    mod = stmt.module or ""
    if not any(mod == m or mod.startswith(m + ".") for m in _DSL_IMPORT_MODULES):
        return (f"import_disallowed_prefix:{mod}",
                f"the import at line {stmt.lineno} ({spelled!r}) names module {mod!r}, outside the prescan import allowlist "
                f"{list(_DSL_IMPORT_MODULES)}; the prescan imports no other module, so the object's identity was not checked")
    try:
        import importlib
        module = importlib.import_module(mod)
    except Exception as exc:
        return f"import_failed:{mod}", f"the import at line {stmt.lineno} ({spelled!r}) fails at prescan: {type(exc).__name__}: {exc}"
    missing = object()
    obj = getattr(module, alias.name, missing)
    if obj is missing:
        return f"import_symbol_missing:{mod}.{alias.name}", f"the import at line {stmt.lineno} ({spelled!r}): module {mod!r} has no attribute {alias.name!r}"
    from . import core as _core
    if obj is not getattr(_core, expected):
        return (f"import_identity_mismatch:{mod}.{alias.name}",
                f"the import at line {stmt.lineno} ({spelled!r}) binds '{expected}' to {mod}.{alias.name} = {obj!r}, which is not the DSL "
                + ("Tensor class" if expected == "Tensor" else f"builtin {expected}"))
    return None


def _import_binds_dsl_tensor(stmt: ast.ImportFrom, alias: ast.alias) -> bool:
    """True when the import binds the DSL ``Tensor`` class itself (see :func:`_import_identity_reason`)."""
    return _import_identity_reason(stmt, alias) is None


def _scope_statements(body):
    """The statements of ONE lexical scope, through EVERY compound statement --
    ``if`` / ``for`` / ``while`` / ``with`` / ``try`` / ``match`` bodies, ``else``,
    ``finally``, handlers, cases -- and never into a ``def`` / ``class`` body
    (another scope; the def / class itself IS yielded, for its name, decorators,
    defaults and bases).  ``ExceptHandler`` and ``match_case`` nodes are yielded
    too, for the names they capture."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list):
                yield from _scope_statements(inner)
        for handler in getattr(stmt, "handlers", None) or []:
            yield handler
            yield from _scope_statements(handler.body)
        for case in getattr(stmt, "cases", None) or []:
            yield case
            yield from _scope_statements(case.body)


def _all_arguments(args: ast.arguments):
    """Every parameter node of a def / lambda, in signature order."""
    return (args.posonlyargs + args.args + ([args.vararg] if args.vararg else [])
            + args.kwonlyargs + ([args.kwarg] if args.kwarg else []))


def _own_expressions(node, postponed: bool = False):
    """The expression children a statement evaluates IN ITS OWN SCOPE: every
    field except nested statement lists (bodies are other statements of the
    scope, yielded by :func:`_scope_statements`) and, for a def / class, except
    the body and the parameter NAMES (another scope) -- decorators, defaults,
    EVERY parameter annotation (positional-only, positional, ``*args``,
    keyword-only, ``**kwargs``; annotation-binding-9), the return
    annotation and bases are evaluated in the enclosing scope at definition
    time (Python 3.10).  With ``postponed`` (the module has ``from __future__
    import annotations``; future-annotations-16) no annotation -- of a
    parameter, a return or an annotated assignment -- is evaluated by Python,
    and none is yielded: a postponed annotation may legally spell
    ``globals().__setitem__(...)`` and never runs."""
    skip = {"body", "orelse", "finalbody", "handlers", "cases"}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield from node.decorator_list
        yield from (d for d in node.args.defaults + node.args.kw_defaults if d is not None)
        if not postponed:
            yield from (a.annotation for a in _all_arguments(node.args) if a.annotation is not None)
            if node.returns is not None:
                yield node.returns
        return
    if isinstance(node, ast.ClassDef):
        yield from node.decorator_list
        yield from node.bases
        yield from (k.value for k in node.keywords)
        return
    for field, value in ast.iter_fields(node):
        if field in skip or (postponed and field == "annotation" and isinstance(node, ast.AnnAssign)):
            continue
        if isinstance(value, ast.AST):
            yield value
        elif isinstance(value, list):
            yield from (v for v in value if isinstance(v, ast.AST))


def _names_stored(target) -> set:
    """Names a target expression binds (Name in Store / Del context, through tuple / list / starred unpacking)."""
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}


def _scope_expression_nodes(expr):
    """The nodes of an expression that are evaluated IN THE ENCLOSING SCOPE
    (lambda-scope-12 / comprehension-scope-11), where ``ast.walk``
    would also descend into other scopes:

    * a ``lambda``: its defaults are evaluated in the enclosing scope and are
      visited; its BODY is its own scope (a walrus there binds the lambda, PEP
      572) and is not;
    * a comprehension / generator expression: its iterables, conditions and
      element / key / value expressions are visited -- a walrus in any of them
      binds the ENCLOSING scope (PEP 572) and a namespace call there executes
      when the comprehension does; its TARGETS are comprehension-local names
      and are not visited at all.
    The lambda and comprehension nodes themselves are yielded."""
    stack = [expr]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, ast.Lambda):
            stack.extend(d for d in n.args.defaults + n.args.kw_defaults if d is not None)
        elif isinstance(n, ast.comprehension):
            stack.append(n.iter)
            stack.extend(n.ifs)
        else:
            stack.extend(ast.iter_child_nodes(n))


def _scope_tensor_binding(statements, scope_word: str, postponed: bool = False):
    """How ONE lexical scope (the statements of a module, or of a kernel body)
    binds the name ``Tensor`` (module-binding-5 / LEXICAL-SCOPE-6).
    Returns ``(verified_imports, reason)``: ``reason`` is ``None`` when every
    binding of the name is an identity-verified import of the DSL class, else
    ``(subcode, detail)`` for the first statement that

    * REBINDS it -- ``def`` / ``class`` named Tensor, an assignment target of any
      kind (plain, augmented, annotated, unpacking, walrus -- also a walrus
      inside a comprehension, a lambda default or a nested def's parameter
      annotation, all evaluated in this scope), a ``for`` / ``with``
      / ``except ... as`` / ``match`` capture, a plain
      ``import ... as Tensor``, an import of another symbol under the name
      (``import_*`` sub-reasons); ``subcode`` ``<scope>_binding_rebound``;
    * DELETES it -- ``del Tensor``; ``<scope>_binding_deleted``;
    * might rewrite the namespace in a way the scan cannot follow -- ``exec`` /
      ``eval`` / ``globals()`` / ``vars()`` / ``locals()`` / ``setattr`` /
      ``delattr`` / ``__import__`` calls -- or any other REFERENCE to one of
      those builtins or to an attribute so named (``f = setattr; f(...)``,
      ``builtins.exec(...)``), a reflective attribute (``__dict__``,
      ``__globals__``, ``__builtins__``, ``f_globals``, ``f_locals``), a store to
      an attribute named ``Tensor``, a star import, a ``global`` / ``nonlocal``
      declaration of the name; ``<scope>_binding_uncertain``.
    Statements are visited through every compound statement of the scope and
    never into nested def / class / lambda bodies or comprehension targets
    (their parameters and locals are theirs -- lexical-scope-6,
    LAMBDA-SCOPE-12, COMPREHENSION-SCOPE-11; see :func:`_scope_expression_nodes`);
    with ``postponed`` annotations are not visited (never evaluated).  A nested
    scope is judged by its name, by what its definition evaluates in this scope
    and by what it declares ``nonlocal`` / ``global`` (see
    :func:`_kernel_tensor_binding`); what EXECUTING it may do is
    :class:`_ModuleEffects`' question."""
    verified = 0
    for stmt in statements:
        line = getattr(stmt, "lineno", "?")
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if stmt.name == "Tensor":
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} defines the name 'Tensor' at line {line}")
        elif isinstance(stmt, ast.Assign):
            if any("Tensor" in _names_stored(t) for t in stmt.targets):
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} assigns the name 'Tensor' at line {line}")
        elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
            if "Tensor" in _names_stored(stmt.target):
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} assigns the name 'Tensor' at line {line}")
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            if "Tensor" in _names_stored(stmt.target):
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} binds 'Tensor' as a for-loop target at line {line}")
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            if any(item.optional_vars is not None and "Tensor" in _names_stored(item.optional_vars) for item in stmt.items):
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} binds 'Tensor' as a with-target at line {line}")
        elif isinstance(stmt, ast.ExceptHandler):
            if stmt.name == "Tensor":
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} binds 'Tensor' as an exception name at line {line}")
        elif isinstance(stmt, ast.match_case):
            captured = {n.name for n in ast.walk(stmt.pattern) if isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name}
            captured |= {n.rest for n in ast.walk(stmt.pattern) if isinstance(n, ast.MatchMapping) and n.rest}
            if "Tensor" in captured:
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} binds 'Tensor' as a match capture at line {getattr(stmt.pattern, 'lineno', '?')}")
        elif isinstance(stmt, ast.Delete):
            if any("Tensor" in _names_stored(t) for t in stmt.targets):
                return verified, (f"{scope_word}_binding_deleted", f"the {scope_word} deletes the name 'Tensor' at line {line}")
        elif isinstance(stmt, (ast.Global, ast.Nonlocal)):
            if "Tensor" in stmt.names:
                return verified, (f"{scope_word}_binding_uncertain", f"the {scope_word} declares 'Tensor' {type(stmt).__name__.lower()} at line {line}")
        elif isinstance(stmt, ast.Import):
            if any((a.asname or a.name).split(".")[0] == "Tensor" for a in stmt.names):
                return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} binds 'Tensor' by a plain import at line {line}")
        elif isinstance(stmt, ast.ImportFrom):
            for a in stmt.names:
                if a.name == "*":
                    return verified, (f"{scope_word}_binding_uncertain", f"the {scope_word} uses a star import at line {line}; the binding of 'Tensor' cannot be established")
                if (a.asname or a.name) == "Tensor":
                    why = _import_identity_reason(stmt, a)
                    if why is not None:
                        return verified, why
                    verified += 1
        # expressions this statement evaluates in THIS scope (lambda bodies and comprehension targets are not): walrus, namespace calls
        for expr in _own_expressions(stmt, postponed):
            for n in _scope_expression_nodes(expr):
                if isinstance(n, ast.NamedExpr) and "Tensor" in _names_stored(n.target):
                    return verified, (f"{scope_word}_binding_rebound", f"the {scope_word} binds 'Tensor' by a walrus at line {line}")
                reach = _namespace_reach(n)
                if reach is not None:
                    return verified, (f"{scope_word}_binding_uncertain",
                                      f"the {scope_word} {reach} at line {line}, which may rewrite the namespace; neither the binding of 'Tensor' nor the @kernel decorator in the live module can be followed statically")
    return verified, None


_REFLECTIVE_ATTRIBUTES = frozenset({"__dict__", "__globals__", "__builtins__", "f_globals", "f_locals", "f_back"})   # reach a namespace dict without naming a builtin


def _namespace_reach(n):
    """How the expression node ``n`` may reach a namespace the prescan cannot
    follow, as a verb phrase (``"calls globals(...)"``, ``"references
    'setattr'"``, ``"writes 'mod.Tensor'"``), else ``None``: a call of /
    reference to a namespace builtin (``_NAMESPACE_CALLS``) by name or as an
    attribute of anything (``builtins.exec``), a reflective attribute
    (``__dict__`` / ``__globals__`` / ``__builtins__`` / ``f_globals`` /
    ``f_locals`` / ``f_back``) or a store / deletion of an attribute named
    ``Tensor``."""
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _NAMESPACE_CALLS:
        return f"calls {n.func.id}(...)"
    if isinstance(n, ast.Name) and n.id in _NAMESPACE_CALLS and isinstance(n.ctx, ast.Load):
        return f"references {n.id!r}"
    if isinstance(n, ast.Attribute):
        if n.attr in _NAMESPACE_CALLS or n.attr in _REFLECTIVE_ATTRIBUTES:
            return f"references {ast.unparse(n)!r}"
        if n.attr == "Tensor" and isinstance(n.ctx, (ast.Store, ast.Del)):
            return f"writes {ast.unparse(n)!r}"
    return None


_HARNESS_NAMES = frozenset({"np", "math", "abs", "cos", "erf", "exp", "floor", "fp32", "log", "log2", "max", "min", "round", "sin", "softmax", "sqrt", "tanh"})   # verify.execute_translated injects these into every module namespace (_MATH_BUILTINS + math + np): never module-defined code
_BUILTIN_NAMES = frozenset(dir(__import__("builtins")))
_CAPABILITY_IMPORTS = frozenset({"builtins", "ctypes", "gc", "inspect"})     # importing one of these gives module code a handle on namespaces it never names


def _postponed_annotations(tree: ast.Module) -> bool:
    """True when the module's LEADING future statements -- after an optional
    docstring; Python allows nothing else before a future import, and the
    source is compile-validated before the prescan runs -- include
    ``annotations`` (PEP 563): every annotation is then kept as a string and
    NEVER evaluated (future-annotations-16)."""
    body = tree.body
    i = 1 if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)) else 0
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__" and not body[i].level:
        if any(a.name == "annotations" for a in body[i].names):
            return True
        i += 1
    return False


class _Scope:
    """One code scope of :class:`_ModuleEffects`: the module, a ``def`` /
    ``async def``, a ``lambda`` or a ``class`` body."""

    def __init__(self, node, kind: str, parent):
        self.node, self.kind, self.parent = node, kind, parent
        self.children = []          # scopes defined directly inside this one, in source order
        self.bindings = {}          # name -> set of binding kinds: "import" / "param" / ("def", node) / ("alias", name) / ("unknown", line)
        self.globals, self.nonlocals = set(), set()
        self.units = []             # [(statement or None, [expressions evaluated in this scope])]
        self.line = getattr(node, "lineno", 0)

    def desc(self, article: bool = True) -> str:
        the = "the " if article else ""
        if self.kind == "module":
            return "the module"
        if self.kind == "lambda":
            return f"{the}lambda at line {self.line}"
        return f"{the}{'class' if self.kind == 'class' else 'function'} {self.node.name!r} defined at line {self.line}"

    def bind(self, name: str, kind) -> None:
        self.bindings.setdefault(name, set()).add(kind)

    @property
    def stmts(self):
        return [u for u, _ in self.units if u is not None]

    def expression_nodes(self):
        for _stmt, exprs in self.units:
            for e in exprs:
                yield from _scope_expression_nodes(e)


class _ModuleEffects:
    """A SOUND over-approximation of what EXECUTING the module may do to its
    name ``Tensor`` (an earlier fix INDIRECT-MODULE-WRITE-10, TRANSITIVE-WRITER-13,
    CLASS-EXECUTION-14, INVOKED-LAMBDA-15), as a fixpoint over every scope:

    * every ``def`` / ``lambda`` / ``class`` body is a scope with an INTRINSIC
      effect -- a ``global Tensor`` declaration together with a binding /
      deletion of the name in that scope, or any reach into a namespace
      (:func:`_namespace_reach`: ``exec`` / ``eval`` / ``globals`` / ``vars`` /
      ``locals`` / ``setattr`` / ``delattr`` / ``__import__`` called OR merely
      referenced, by name or as an attribute, a reflective attribute, a store
      to an attribute named ``Tensor``);
    * a scope's CALL effect (what running it may do) is its intrinsic effect,
      the effect of every callee it may invoke -- a call, a decorator it
      applies, a class base it derives from, a module-defined function / class
      / lambda it passes as an argument to any call (the callee may invoke it)
      -- and, conservatively, the effect of every scope defined inside it (a
      nested def or lambda may be returned and invoked later; a nested class
      body executes).  A class's call effect (instantiation, ``Holder.m()``, use
      as a base) is the union of its methods' and nested classes' effects;
    * a class body's EXEC effect (it runs at definition) is its intrinsic
      effect, its calls and its nested class bodies -- not its methods;
    * the MODULE executes its statements in order -- class bodies, calls,
      decorators, bases, argument references -- and the harness then calls
      ``run_model`` -- whatever the name is FINALLY bound to, so every binding of
      the name is a possible entry (a ``def``, ``run_model = lambda``, an alias
      followed through its chain) and a binding the prescan cannot follow (an
      attribute such as ``Runner.run``, a call result, an import, a ``global``
      store elsewhere, a deletion) is the typed refusal
      ``harness_binding_unresolved``, capability or not (harness-binding-17);
      a callee that is never reached (an uncalled writer, a
      lambda that is only bound) contributes nothing;
    * a callee is resolved by lexical lookup through the enclosing scopes
      (class scopes are invisible to the functions nested in them; ``global``
      / ``nonlocal`` honoured): a def / class / ``name = lambda`` -> that scope;
      an import, a Python builtin, a harness-provided name (``np``, ``math``,
      the interpreter's math builtins) or an attribute of any of those -> safe,
      UNLESS the program itself stores something under that attribute name
      anywhere (``np.x = change``, ``np.random.x = ...``, ``other.x = ...``;
      import-attribute-writer-18): the attribute then stands for what was
      stored -- the stored module-defined code's effects, or unresolved when the
      store cannot be followed -- while untouched imported attributes
      (``np.random.seed``) keep the ordinary-import rule;
      a parameter, a comprehension variable, a name bound by a statement the prescan cannot follow, an
      attribute of a function object, a callee reached through a subscript /
      call result / ``getattr`` / container -> UNRESOLVED; a name bound nowhere
      -> UNKNOWN.  An unresolved effect declines (``module_binding_uncertain``)
      whenever the module holds any CAPABILITY it could reach -- a scope with
      an intrinsic effect, a string constant naming a namespace builtin, a
      reference to ``__builtins__``, an import of ``builtins`` / ``ctypes`` /
      ``gc`` / ``inspect``; an unknown callee always declines; a module without
      any such capability cannot rebind the name through any call and lowers.
    Mutual recursion is a cycle of the fixpoint and converges.  Postponed
    annotations are never evaluated and are not visited.  Outside the model
    (not claimed): a namespace builtin reached by a COMPUTED attribute name
    (``getattr(mod, "set" + "attr")``) in a module that holds none of the
    capabilities above."""

    def __init__(self, tree: ast.Module, postponed: bool):
        self.postponed = postponed
        self.scopes, self.by_node = [], {}
        self.module = self._build(tree, "module", None)
        self.intrinsic = {id(sc): (None if sc.kind == "module" else self._intrinsic(sc)) for sc in self.scopes}
        self.capability = self._capability(tree)
        self.attr_stores, self._attr_stack = self._attribute_stores(), set()   # import-attribute-writer-18, consulted by _callee
        self.edges = {id(sc): self._edges(sc) for sc in self.scopes}
        self.value = {}
        self._fixpoint()

    # ---- scopes and bindings ------------------------------------------------------------------
    def _build(self, node, kind: str, parent):
        sc = _Scope(node, kind, parent)
        self.scopes.append(sc)
        self.by_node[id(node)] = sc
        if parent is not None:
            parent.children.append(sc)
        if kind == "lambda":
            sc.units = [(None, [node.body])]
        else:
            sc.units = [(st, list(_own_expressions(st, self.postponed))) for st in _scope_statements(node.body)]
        if kind in ("function", "lambda"):
            for a in _all_arguments(node.args):
                sc.bind(a.arg, "param")
        for st in sc.stmts:
            self._bind_statement(sc, st)
        for n in sc.expression_nodes():
            if isinstance(n, ast.NamedExpr):
                for name in _names_stored(n.target):
                    sc.bind(name, ("unknown", getattr(n, "lineno", "?")))
            elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):   # a comprehension variable is a BOUND name whose
                for g in n.generators:                                                       # value the prescan cannot follow (its visibility is
                    for name in _names_stored(g.target):                                     # narrower than the scope's: recording it here is
                        sc.bind(name, ("unknown", getattr(n, "lineno", "?")))                 # conservative, never permissive)
        root = sc
        while root.parent is not None:
            root = root.parent
        for name in (sc.globals & set(sc.bindings)) - {"Tensor"}:   # a ``global x`` bound here rebinds the MODULE's x by a statement the module scan
            root.bind(name, ("unknown", sc.line))                     # does not see; 'Tensor' itself is what the effects track (its import binding stands unless a writer RUNS)
        for st in sc.stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._build(st, "function", sc)
            elif isinstance(st, ast.ClassDef):
                self._build(st, "class", sc)
        for n in sc.expression_nodes():
            if isinstance(n, ast.Lambda):
                self._build(n, "lambda", sc)
        return sc

    @staticmethod
    def _bind_statement(sc, st) -> None:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sc.bind(st.name, ("def", st))
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                if a.name != "*":
                    sc.bind((a.asname or a.name).split(".")[0], "import")
        elif isinstance(st, ast.Global):
            sc.globals.update(st.names)
        elif isinstance(st, ast.Nonlocal):
            sc.nonlocals.update(st.names)
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Name):
            sc.bind(st.targets[0].id, ("alias", st.value.id))
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Lambda):
            sc.bind(st.targets[0].id, ("def", st.value))
        else:
            targets = []
            if isinstance(st, ast.Assign):
                targets = st.targets
            elif isinstance(st, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor)):
                targets = [st.target]
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                targets = [i.optional_vars for i in st.items if i.optional_vars is not None]
            elif isinstance(st, ast.Delete):
                targets = st.targets
            for name in {n for t in targets for n in _names_stored(t)}:
                sc.bind(name, ("unknown", st.lineno))
            if isinstance(st, ast.ExceptHandler) and st.name:
                sc.bind(st.name, ("unknown", st.lineno))
            if isinstance(st, ast.match_case):
                for n in ast.walk(st.pattern):
                    nm = n.name if isinstance(n, (ast.MatchAs, ast.MatchStar)) else n.rest if isinstance(n, ast.MatchMapping) else None
                    if nm:
                        sc.bind(nm, ("unknown", getattr(st.pattern, "lineno", "?")))

    # ---- intrinsic effects and capabilities -----------------------------------------------------
    def _intrinsic(self, sc):
        """``(subcode, phrase)`` for what the scope's OWN code does to the module's 'Tensor' without following any call, else ``None``."""
        if "Tensor" in sc.globals:
            own = [st for st in sc.stmts if not isinstance(st, (ast.Global, ast.Nonlocal))]
            _v, why = _scope_tensor_binding(own, "module", self.postponed)
            if why is not None:
                verb = "deletes it" if why[0] == "module_binding_deleted" else "may rewrite it" if why[0] == "module_binding_uncertain" else "binds it"
                return why[0], f"declares 'Tensor' global and {verb}: {why[1]}"
        for n in sc.expression_nodes():
            reach = _namespace_reach(n)
            if reach is not None:
                return "module_binding_uncertain", f"{reach} at line {getattr(n, 'lineno', sc.line)}, which may rewrite the module namespace"
        return None

    def _capability(self, tree):
        """``(desc, phrase)`` of the first thing in the module that could let an UNRESOLVED call reach the module namespace, else ``None``."""
        for sc in self.scopes:
            v = self.intrinsic[id(sc)]
            if v is not None:
                return sc.desc(), v[1]
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in _NAMESPACE_CALLS:
                return "the module", f"spells the string {n.value!r} at line {n.lineno}, by which a namespace builtin can be looked up dynamically"
            if isinstance(n, ast.Name) and n.id == "__builtins__":
                return "the module", f"references '__builtins__' at line {n.lineno}"
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [(n.module or "").split(".")[0]]
                hit = next((m for m in mods if m in _CAPABILITY_IMPORTS), None)
                if hit:
                    return "the module", f"imports {hit!r} at line {n.lineno}, which gives a handle on namespaces it never names"
        return None

    # ---- call edges and resolution --------------------------------------------------------------
    def _edges(self, sc):
        """``[(what, targets)]``: everything the scope's own code may invoke, in source order."""
        out = []
        for st, exprs in sc.units:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in st.decorator_list:
                    out.append((f"applies the decorator {ast.unparse(d)!r} at line {st.lineno}", self._callee(sc, d)))
                if isinstance(st, ast.ClassDef):
                    for b in st.bases:          # __init_subclass__ / the metaclass of a base runs at definition
                        out.append((f"derives the class {st.name!r} from {ast.unparse(b)!r} at line {st.lineno}", self._callee(sc, b)))
                    if st.keywords:
                        out.append((f"defines the class {st.name!r} with class keywords at line {st.lineno}",
                                    [("unresolved", "a metaclass or class keyword runs code the prescan cannot follow")]))
            for e in exprs:
                for n in _scope_expression_nodes(e):
                    if not isinstance(n, ast.Call):
                        continue
                    out.append((f"calls {ast.unparse(n.func)!r} at line {n.lineno}", self._callee(sc, n.func)))
                    for a in n.args + [k.value for k in n.keywords]:
                        nodes = list(_scope_expression_nodes(a))
                        called = {id(c.func) for c in nodes if isinstance(c, ast.Call)}      # the callee of a nested call is its own edge, not a reference handed over
                        for ref in nodes:
                            if id(ref) in called:
                                continue
                            if isinstance(ref, ast.Lambda) or (isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load)):
                                targets = [t for t in self._callee(sc, ref) if t[0] == "scope"]      # module-defined code handed to a call may be invoked by it
                                if targets:
                                    out.append((f"passes {ast.unparse(ref)!r} to {ast.unparse(n.func)!r} at line {n.lineno}", targets))
        return out

    def _callee(self, sc, func):
        """What a callee expression may stand for: ``("scope", sc)``, ``("safe",)``, ``("unresolved", why)`` or ``("unknown", name)``."""
        if isinstance(func, ast.Lambda):
            return [("scope", self.by_node[id(func)])]
        if isinstance(func, ast.Name):
            return self._lookup(sc, func.id, ())
        if isinstance(func, ast.Attribute):
            # 18 + INTERMEDIATE-ATTRIBUTE-22: the COMPLETE path root.a.b.leaf -- every intermediate attribute the program itself
            # stored anywhere (``np._holder = Holder; np._holder.m()``) stands for what was stored there (a class: its methods; a
            # function: unresolved), and the leaf for what was stored under its name; untouched segments keep the import rule
            path, root = [], func
            while isinstance(root, ast.Attribute):
                path.append(root.attr); root = root.value
            path.reverse()                                            # [a, b, leaf]
            stored = []
            for seg in path[:-1]:
                for t in (self._stored_targets(seg) if seg in self.attr_stores else ()):
                    if t[0] == "scope" and t[1].kind != "class":
                        stored.append(("unresolved", f"{ast.unparse(func)!r} is an attribute of {t[1].desc()}, stored under {seg!r}"))
                    elif t[0] != "safe":
                        stored.append(t)                              # a stored class: its methods; unresolved / unknown stores as they are
            if path[-1] in self.attr_stores:
                stored.extend(self._stored_targets(path[-1]))
            if isinstance(root, ast.Name):
                out = []
                for t in self._lookup(sc, root.id, ()):
                    if t[0] == "scope" and t[1].kind != "class":
                        out.append(("unresolved", f"{ast.unparse(func)!r} is an attribute of {t[1].desc()}"))
                    else:
                        out.append(t)        # an attribute of an import / builtin / harness name is safe; of a module-defined class: its methods
                return out + stored
            return [("unresolved", f"{ast.unparse(func)!r} is reached through an expression the prescan cannot follow")] + stored
        return [("unresolved", f"{ast.unparse(func)!r} is the result of an expression the prescan cannot follow")]

    def _lookup(self, sc, name: str, seen):
        if (id(sc), name) in seen:
            return [("unresolved", f"{name!r} is bound through a cycle of aliases")]
        seen = seen + ((id(sc), name),)
        scope = sc
        if name in sc.globals:
            while scope.parent is not None:
                scope = scope.parent
        elif name in sc.nonlocals:
            scope = sc.parent
        while scope is not None:
            if scope.kind == "class" and scope is not sc:
                scope = scope.parent          # a class scope is invisible to the scopes nested in it
                continue
            kinds = scope.bindings.get(name)
            if kinds:
                out = []
                for k in kinds:
                    if k == "import":
                        out.append(("safe",))
                    elif k == "param":
                        return [("unresolved", f"{name!r} is a parameter of {scope.desc()}")]
                    elif k[0] == "unknown":
                        return [("unresolved", f"{name!r} is bound at line {k[1]} by a statement the prescan cannot follow")]
                    elif k[0] == "def":
                        out.append(("scope", self.by_node[id(k[1])]))
                    else:
                        out.extend(self._lookup(scope, k[1], seen))
                return out
            scope = scope.parent
        if name in _BUILTIN_NAMES or name in _HARNESS_NAMES:
            return [("safe",)]
        return [("unknown", name)]

    def _attribute_stores(self):
        """``{attribute: [(scope, value or None, line)]}`` for every store to an attribute anywhere in the module -- a plain
        ``root.x = value`` keeps its value, tuple / augmented / annotated-without-value / ``for`` / ``with`` targets and
        ``del root.x`` keep ``None`` (a store the prescan cannot follow).  Keyed by the attribute NAME alone, whatever the
        root (``np``, ``np.random``, an alias ``other = np``): an over-approximation (import-attribute-writer-18)."""
        stores = {}
        for sc in self.scopes:
            for st in sc.stmts:
                if isinstance(st, ast.Assign):
                    targets, value = st.targets, (st.value if len(st.targets) == 1 and isinstance(st.targets[0], ast.Attribute) else None)
                elif isinstance(st, ast.AnnAssign):
                    targets, value = [st.target], (st.value if isinstance(st.target, ast.Attribute) else None)
                elif isinstance(st, (ast.AugAssign, ast.For, ast.AsyncFor)):
                    targets, value = [st.target], None
                elif isinstance(st, (ast.With, ast.AsyncWith)):
                    targets, value = [i.optional_vars for i in st.items if i.optional_vars is not None], None
                elif isinstance(st, ast.Delete):
                    targets, value = st.targets, None
                else:
                    continue
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
                            stores.setdefault(n.attr, []).append((sc, value if n is t else None, getattr(st, "lineno", "?")))
        return stores

    def _stored_targets(self, attr: str):
        """What the program may have stored under the attribute name ``attr`` (every store in the module, any root): the
        stored module-defined code's scope(s) / safe import, or unresolved when a store's value cannot be followed."""
        if attr in self._attr_stack:
            return [("unresolved", f"the attribute {attr!r} is stored through a cycle of attribute stores")]
        self._attr_stack.add(attr)
        try:
            out = []
            for sc, value, line in self.attr_stores.get(attr, ()):
                if isinstance(value, (ast.Lambda, ast.Name, ast.Attribute)):
                    out.extend(self._callee(sc, value))
                else:
                    out.append(("unresolved", f"the attribute {attr!r} is stored at line {line} by a statement the prescan cannot follow"))
            return out
        finally:
            self._attr_stack.discard(attr)

    # ---- the fixpoint ---------------------------------------------------------------------------
    @staticmethod
    def _slots(sc):
        return ("exec", "call") if sc.kind == "class" else ("call",)

    def _fixpoint(self) -> None:
        slots = [(sc, slot) for sc in self.scopes if sc.kind != "module" for slot in self._slots(sc)]
        for _ in range(len(slots) + 1):          # values only move from None to a reason: monotone, so this bound suffices
            changed = False
            for sc, slot in slots:
                if self.value.get((id(sc), slot)) is None:
                    v = self._compute(sc, slot)
                    if v is not None:
                        self.value[(id(sc), slot)] = v
                        changed = True
            if not changed:
                return

    def _effect(self, sc, slot):
        return self.value.get((id(sc), slot))

    def _why(self, targets):
        """Why invoking any of ``targets`` may rebind the module's 'Tensor', as ``(subcode, text)``, else ``None``."""
        for t in targets:
            if t[0] == "scope":
                v = self._effect(t[1], "call")
                if v is not None:
                    joiner = ", " if v[1].startswith("whose ") else ", which "
                    return "module_binding_uncertain", f"resolves to {t[1].desc()}{joiner}{v[1]}"
            elif t[0] == "unresolved":
                if self.capability is not None:
                    return "module_binding_uncertain", f"cannot be resolved statically ({t[1]}), while {self.capability[0]} {self.capability[1]}"
            elif t[0] == "unknown":
                return "module_binding_uncertain", f"names {t[1]!r}, which is bound nowhere in the module and is neither a builtin nor a harness-provided name"
        return None

    def _compute(self, sc, slot: str):
        if slot == "call" and sc.kind == "class":
            for ch in sc.children:
                for s in self._slots(ch):
                    v = self._effect(ch, s)
                    if v is not None:
                        return "module_binding_uncertain", f"whose {ch.desc(False)} {v[1]}"
            return None
        v = self.intrinsic[id(sc)]
        if v is not None:
            return v
        for what, targets in self.edges[id(sc)]:
            why = self._why(targets)
            if why is not None:
                return why[0], f"{what}, which {why[1]}"
        for ch in sc.children:
            if sc.kind == "class":
                if ch.kind == "class":
                    v = self._effect(ch, "exec")
                    if v is not None:
                        return v[0], f"contains {ch.desc(False)}, whose body executes at definition and {v[1]}"
            else:
                for s in self._slots(ch):
                    v = self._effect(ch, s)
                    if v is not None:
                        return "module_binding_uncertain", f"defines {ch.desc(False)}, which {v[1]}"
        return None

    def module_reason(self):
        """``(subcode, detail)`` when executing the module -- or the harness's call of ``run_model`` -- may rebind its 'Tensor', else ``None``."""
        m = self.module
        for st in m.stmts:
            if isinstance(st, ast.ClassDef):
                v = self._effect(self.by_node[id(st)], "exec")
                if v is not None:
                    return v[0], f"the class body at line {st.lineno} executes at definition and {v[1]}"
        for what, targets in self.edges[id(m)]:
            why = self._why(targets)
            if why is not None:
                return why[0], f"the module {what}, which {why[1]}; executing it may rebind the module's 'Tensor'"
        for what, targets, direct in self._harness_entries():          # harness-binding-17: every possible FINAL binding of the name
            why = self._entry_why(targets, direct)
            if why is not None:
                return why[0], f"the harness calls 'run_model' ({what}), which {why[1]}; executing it may rebind the module's 'Tensor' before the kernel runs"
        return None

    def _harness_entries(self):
        """``[(what, targets, direct)]``: every way the module may bind the name ``run_model`` by the time the harness calls it
        (harness-binding-17, SAME-LINE-ENTRY-21).  The runtime uses the FINAL binding, so every binding STATEMENT is a
        possible entry, identified by its own AST node -- never by its line, which two bindings may share: a ``def`` / ``class``
        named run_model and ``run_model = lambda`` (``direct``: the scope itself), ``run_model = name`` / ``run_model: T = name``
        (an alias, followed through its chain), and every binding the prescan cannot follow -- a subscript / call / attribute
        value, a tuple target, an augmented assignment, a walrus, a ``for`` / ``with`` / ``except`` / ``match`` capture, an import,
        a ``global run_model`` store in another scope, a deletion -- as unresolved (``harness_binding_unresolved``)."""
        out = []
        def unresolved(node, how):
            line = getattr(node, 'lineno', '?')
            out.append((line, getattr(node, 'col_offset', 0), f"bound at line {line}", [("unresolved", f"'run_model' is bound at line {line} by {how}")], False))
        for st in self.module.stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if st.name == "run_model":
                    out.append((st.lineno, st.col_offset, f"defined at line {st.lineno}", [("scope", self.by_node[id(st)])], True))
                continue
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                if any((a.asname or a.name).split(".")[0] == "run_model" for a in st.names):
                    unresolved(st, "an import, not by module-defined code")
                continue
            targets, value = [], None
            if isinstance(st, ast.Assign):
                targets, value = st.targets, (st.value if len(st.targets) == 1 else None)
            elif isinstance(st, ast.AnnAssign):
                targets, value = [st.target], st.value
            elif isinstance(st, (ast.AugAssign, ast.For, ast.AsyncFor)):
                targets = [st.target]
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                targets = [i.optional_vars for i in st.items if i.optional_vars is not None]
            elif isinstance(st, ast.Delete):
                targets = st.targets
            elif isinstance(st, ast.ExceptHandler):
                if st.name == "run_model":
                    unresolved(st, "an 'except ... as' capture")
            elif isinstance(st, ast.match_case):
                if any((n.name if isinstance(n, (ast.MatchAs, ast.MatchStar)) else n.rest if isinstance(n, ast.MatchMapping) else None) == "run_model" for n in ast.walk(st.pattern)):
                    unresolved(st.pattern, "a 'match' capture")
            if not any("run_model" in _names_stored(t) for t in targets):
                continue
            simple = len(targets) == 1 and isinstance(targets[0], ast.Name)
            if simple and isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Lambda):
                out.append((st.lineno, st.col_offset, f"bound at line {st.lineno} to a lambda", [("scope", self.by_node[id(value)])], True))
            elif simple and isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Name):
                out.append((st.lineno, st.col_offset, f"bound at line {st.lineno} to {value.id!r}", self._lookup(self.module, value.id, ()), False))
            else:
                unresolved(st, "a statement the prescan cannot follow")
        for n in self.module.expression_nodes():                      # a walrus at module scope binds the module name
            if isinstance(n, ast.NamedExpr) and "run_model" in _names_stored(n.target):
                unresolved(n, "an assignment expression the prescan cannot follow")
        for sc in self.scopes:                                         # ``global run_model`` + a binding in another scope
            if sc is not self.module and "run_model" in sc.globals and "run_model" in sc.bindings:
                unresolved(sc.node, f"a 'global' store in {sc.desc()}")
        return [(what, targets, direct) for _, _, what, targets, direct in sorted(out, key=lambda e: (e[0] if isinstance(e[0], int) else 0, e[1], e[2]))]

    def _entry_why(self, targets, direct: bool):
        """The harness entry's effect: a scope's call effect (``direct``: the phrase itself, else through :meth:`_why`); an
        entry the prescan cannot establish as module-defined code -- unresolved, unknown, or an imported / builtin name -- is
        ALWAYS the typed refusal ``harness_binding_unresolved``, capability or not: the harness executes whatever it is."""
        for t in targets:
            if t[0] == "scope":
                if direct:
                    v = self._effect(t[1], "call")
                    if v is not None:
                        return v
                else:
                    why = self._why([t])
                    if why is not None:
                        return why
            elif t[0] == "safe":
                return "harness_binding_unresolved", "resolves to an imported, builtin or harness-provided name, not to module-defined code"
            elif t[0] == "unresolved":
                return "harness_binding_unresolved", f"cannot be resolved statically ({t[1]})"
            else:
                return "harness_binding_unresolved", f"names {t[1]!r}, which is bound nowhere in the module"
        return None


def _module_tensor_binding(tree: ast.Module, postponed: Optional[bool] = None):
    """How the module binds the name ``Tensor``: ``None`` when it is bound ONLY by
    identity-verified imports of the DSL class (``from solar.einsum_lib import
    Tensor``), else ``(subcode, detail)`` -- see :func:`_scope_tensor_binding`
    (every module-scope statement through all control flow; def / class bodies
    are other scopes, so ``class Holder: Tensor = int`` does not bind the module
    name), :class:`_ModuleEffects` (what EXECUTING the module -- class bodies,
    calls and everything they reach, ``run_model`` -- may do to the name; a
    sound over-approximation, an earlier fix rounds 4-5) and ``module_binding_missing``
    when no import of the class is seen (a binding that cannot be established)."""
    if postponed is None:
        postponed = _postponed_annotations(tree)
    verified, reason = _scope_tensor_binding(_scope_statements(tree.body), "module", postponed)
    if reason is not None:
        return reason
    reason = _ModuleEffects(tree, postponed).module_reason()
    if reason is not None:
        return reason
    if not verified:
        return "module_binding_missing", "the module never binds 'Tensor' by an import of the DSL class"
    return None


def _kernel_tensor_binding(node: ast.FunctionDef, postponed: bool = False):
    """``None`` when the KERNEL scope binds the name ``Tensor`` by nothing but
    identity-verified ``from solar.einsum_lib import Tensor`` imports; else
    ``(subcode, detail)``.  Lexically scoped (lexical-scope-6): a
    parameter of the kernel, any binding of the kernel's own scope (see
    :func:`_scope_tensor_binding`, including what a nested def's decorators,
    defaults and parameter / return annotations evaluate here) and a nested def
    / class NAMED Tensor are rejected; the parameters and locals of a nested def
    / lambda / class and comprehension targets are theirs and do not rebind the
    kernel's name -- unless the nested scope declares ``Tensor`` ``nonlocal`` or
    ``global`` (then it may)."""
    params = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    params += [node.args.vararg.arg] if node.args.vararg else []
    params += [node.args.kwarg.arg] if node.args.kwarg else []
    if "Tensor" in params:
        return "kernel_binding_rebound", "a parameter of the kernel is named 'Tensor'"
    for sub in ast.walk(node):
        if sub is node or not isinstance(sub, _NESTED_SCOPES):
            continue
        for inner in ast.walk(sub):
            if isinstance(inner, (ast.Global, ast.Nonlocal)) and "Tensor" in inner.names:
                return ("kernel_binding_uncertain",
                        f"a nested scope at line {sub.lineno} declares 'Tensor' {type(inner).__name__.lower()} at line {inner.lineno} "
                        f"and may rebind the enclosing name")
    _verified, reason = _scope_tensor_binding(_scope_statements(node.body), "kernel", postponed)
    return reason


def _lowerable_callee_names(tree: ast.Module) -> set:
    """Names a kernel body may call and the lowering can compile: the lowering's
    own namespace (DSL builtins, math, numpy, numba), the Python builtins numba
    supports, and the module's other ``@kernel`` functions (lowered alongside).
    A plain module helper is NOT among them: the lowering keeps it as a Python
    function, which compiled code cannot call."""
    import builtins as _builtins
    names = set(execution_namespace(True).keys()) | set(_LOWERABLE_BUILTINS)
    # exception classes (``raise ValueError("...")`` inside a kernel) are lowerable calls
    names |= {n for n in dir(_builtins) if isinstance(getattr(_builtins, n, None), type) and issubclass(getattr(_builtins, n), BaseException)}
    names |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and _is_kernel_decorated(n)}
    return names


def _reject_unresolved_tensor_binding(node: ast.FunctionDef, tree: ast.Module, memo: Optional[dict] = None) -> None:
    """callee-binding-1 / IMPORT-IDENTITY-2: before anything else, establish
    that the name ``Tensor`` -- in the kernel scope and in the module -- is bound
    only by identity-verified imports of the DSL class (see
    :func:`_kernel_tensor_binding` / :func:`_module_tensor_binding`); else the
    typed ``kernel_allocation_callee_unresolved`` decline, whose detail starts
    with the sub-reason (an earlier fix round 3: ``module_binding_rebound`` /
    ``module_binding_deleted`` / ``module_binding_uncertain`` /
    ``module_binding_missing``, ``harness_binding_unresolved`` (round 6), ``kernel_binding_*``, ``import_disallowed_prefix:<module>``
    / ``import_identity_mismatch:<module>.<symbol>`` / ``import_failed`` /
    ``import_symbol_missing`` / ``import_unresolvable_relative``) and which also
    carries it as ``exc.subcode``.  The decline CODE stays the one the contract
    and the reviewer's verbatim fixtures pin; the sub-reason makes it truthful.  ``memo``
    (a dict the caller shares across the kernels of one pre-scan) holds the
    module-wide answer: :class:`_ModuleEffects` depends on the tree alone, so it is
    built ONCE per module -- not once per kernel -- and only when the kernel's own
    scope is clean (round 5 census follow-up: 4-5x slower lowering of 30-40-kernel
    modules at an earlier revision, answers unchanged)."""
    postponed = _postponed_annotations(tree)
    # the kernel's own scope first (the more local diagnosis), then the module and what executing it reaches
    reason = _kernel_tensor_binding(node, postponed)
    if reason is None:
        if memo is None:
            memo = {}
        if "module" not in memo:          # the module-wide analysis: once per pre-scan, shared by every kernel of the module
            memo["module"] = _module_tensor_binding(tree, postponed)
        reason = memo["module"]
    if reason is not None:
        subcode, detail = reason
        exc = KernelOutputDtypeUnresolved(
            node.name, "callee_unresolved",
            f"{subcode}: {detail}; a Tensor(...) in the kernel cannot be established as the DSL constructor the lowering understands",
            code=f"kernel_allocation_callee_unresolved:{node.name}")
        exc.subcode = subcode      # module_binding_{rebound,deleted,uncertain,missing} / kernel_binding_* / import_* (see the module docstring)
        raise exc


def _reject_unresolved_allocation_calls(node: ast.FunctionDef, tree: ast.Module) -> None:
    """Issue #57 items 1 + 2 and an earlier fix round 2 (CALLEE-BINDING-1, IMPORT-IDENTITY-2,
    HELPER-BOUNDARY-3): the literal-only path accepts an allocation only in the
    ONE spelling the lowering understands -- the DSL constructor ``Tensor(...)``
    with explicit keywords -- and it establishes what ``Tensor`` IS by BINDING
    and identity, never by spelling:

    * the module must bind ``Tensor`` only by identity-verified imports of the
      DSL class (the imported symbol *is* ``core.Tensor``); any other module
      binding, a star import or no binding at all is
      ``kernel_allocation_callee_unresolved``;
    * the kernel scope must not bind ``Tensor`` at all -- not as a parameter
      (of the kernel or a nested def), a nested def / class, a target, a
      ``global`` / ``nonlocal`` -- except by an identity-verified import;
    * every call in the kernel body must name a callee the lowering can
      compile: ``Tensor`` (then the constructor-contract checks below), the
      lowering's namespace names, numba-supported builtins, the module's other
      ``@kernel`` functions, or an attribute of ``np`` / ``math`` / ``numba``.
      Anything else -- an alias, an attribute of another module, a callable
      from a parameter, a plain module helper whether called positionally or
      with keywords, a computed callee -- is ``kernel_allocation_callee_unresolved``
      (the lowering would otherwise fail in numba typing or run the wrong
      object);
    * ``kernel_allocation_signature_unresolved`` -- a ``Tensor(...)`` call
      outside the constructor contract: ``**kwargs`` / ``*args`` expansion,
      any positional argument other than a single dims string, a dims string
      combined with keyword dims.
    """
    lowerable = _lowerable_callee_names(tree)
    # names the kernel binds to nested functions: closures the lowering compiles with the kernel (a nested def named
    # Tensor was already refused by _reject_unresolved_tensor_binding)
    lowerable |= {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n is not node}
    for sub in (n for stmt in node.body for n in ast.walk(stmt)):
        if not isinstance(sub, ast.Call):
            continue
        starred = any(isinstance(a, ast.Starred) for a in sub.args) or any(k.arg is None for k in sub.keywords)
        f = sub.func
        line = getattr(sub, "lineno", "?")
        is_tensor = isinstance(f, ast.Name) and f.id == "Tensor"
        if not is_tensor:
            callee_ok = ((isinstance(f, ast.Name) and f.id in lowerable)
                         or (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("np", "math", "numba")))
            if callee_ok and not starred:
                continue
            raise KernelOutputDtypeUnresolved(
                node.name, "callee_unresolved",
                f"the call {ast.unparse(f)!r} at line {line} " + ("expands *args / **kwargs" if callee_ok else
                "names a callee the lowering cannot compile or identify (not the DSL constructor, not a lowering "
                "builtin, not another @kernel): an alias, a module helper, an attribute of another module, a callable "
                "from a parameter or a computed callee"),
                code=f"kernel_allocation_callee_unresolved:{node.name}")
        if starred:
            raise KernelOutputDtypeUnresolved(
                node.name, "signature_unresolved",
                f"the Tensor(...) call at line {line} expands *args / **kwargs; its keywords (dtype included) are not "
                f"visible statically",
                code=f"kernel_allocation_signature_unresolved:{node.name}")
        spec_form = len(sub.args) == 1 and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str)
        if sub.args and not spec_form:
            raise KernelOutputDtypeUnresolved(
                node.name, "signature_unresolved",
                f"the Tensor(...) call at line {line} passes {len(sub.args)} positional argument(s) "
                f"({', '.join(ast.unparse(a) for a in sub.args)}); the constructor takes keyword dims (or a single "
                f"dims string), and the lowering must not drop an argument the interpreter rejects",
                code=f"kernel_allocation_signature_unresolved:{node.name}")
        dim_keywords = [k.arg for k in sub.keywords if k.arg not in _TENSOR_OPTION_KEYWORDS]
        if spec_form and dim_keywords:
            raise KernelOutputDtypeUnresolved(
                node.name, "signature_unresolved",
                f"the Tensor(...) call at line {line} combines a dims string with keyword dims {dim_keywords}; "
                f"the interpreter raises TypeError on it",
                code=f"kernel_allocation_signature_unresolved:{node.name}")


def _reject_nonliteral_allocation_dtypes(node: ast.FunctionDef) -> None:
    """literal-boundary-30: EVERY ``Tensor(...)`` allocation inside a
    ``@kernel`` -- returned or not, at any nesting -- must spell its ``dtype``
    as a string literal that is a supported dtype, or the kernel is declined
    with the dtype-specific typed reason before anything is lowered or
    stripped.  The lowering drops the dtype keyword of an allocation it turns
    into ``np.full`` (the expression is never evaluated), so a non-literal
    dtype on a temporary is not merely unknown: its evaluation -- and any side
    effect of it -- would vanish from the lowered program while the
    interpreter performs it.  Return-site provenance only sees allocations
    that reach a ``return``; this check sees them all."""
    from .core import VALID_DTYPES as _valid
    for sub in (n for stmt in node.body for n in ast.walk(stmt)):          # the BODY's allocations; annotations are judged by _kernel_output_dtype
        if not _is_tensor_ctor(sub):                                       # an attribute / alias callee was already declined by _reject_unresolved_allocation_calls
            continue
        for kw in sub.keywords:
            if kw.arg != "dtype":
                continue
            v = kw.value
            if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                raise KernelOutputDtypeUnresolved(
                    node.name, "unestablished",
                    f"a Tensor allocation at line {getattr(sub, 'lineno', '?')} spells its dtype as "
                    f"{ast.unparse(v)!r}, not a string literal (literal-only path: the expression would be "
                    f"dropped by the lowering, never evaluated)")
            if v.value not in _valid:
                raise KernelOutputDtypeUnresolved(
                    node.name, "unestablished",
                    f"a Tensor allocation at line {getattr(sub, 'lineno', '?')} names dtype {v.value!r}, "
                    f"which is not a supported dtype (the interpreter refuses it too)")


def _invalid_literal_dtype(tree: ast.Module):
    """Round 12 (baseline-invalid-dtype-37, baseline-owned): the first ``Tensor`` call anywhere in the
    module -- ``Tensor.from_numpy(...)``, ``Tensor.from_random(...)``, ``Tensor(...)`` -- whose LITERAL ``dtype=`` the interpreter
    rejects (``core._validate_dtype``: not a member of ``VALID_DTYPES``, case-sensitive; a non-string constant included), as
    ``(callee text, line, value)``, else ``None``.  Where such a call executes, the interpreter raises ``ValueError("Unknown dtype
    ...")`` and the fast path used to map the unknown spelling to its default float and produce a value.  The scan is a CONSERVATIVE
    refusal of the visible literal SPELLING (P3 dtype-refusal-scope-38): it also refuses a malformed literal in a body that
    never runs or under a shadowed ``Tensor`` name, where the interpreter would not raise -- the program is then left to the
    interpreter, which decides.  A non-literal dtype is not judged here (the deferred dynamic-dtype contract); kernel-body
    allocations were already judged by ``_reject_nonliteral_allocation_dtypes``."""
    from .core import VALID_DTYPES as _valid
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        is_tensor = (isinstance(f, ast.Name) and f.id == "Tensor") or \
                    (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "Tensor" and f.attr in ("from_numpy", "from_random"))
        if not is_tensor:
            continue
        for kw in n.keywords:
            if kw.arg == "dtype" and isinstance(kw.value, ast.Constant) and kw.value.value not in _valid:
                return ast.unparse(f), getattr(n, "lineno", "?"), kw.value.value
    return None


def _extract_kernel_output_dtypes(tree: ast.Module) -> dict:
    """Pre-scan ``@kernel`` functions for narrow output dtypes.

    Returns ``{func_name: dtype_str}`` for kernels whose DECLARED output dtype
    (return annotation, else the returned allocation — see
    :func:`_kernel_output_dtype`) is a narrow float dtype (bf16 or fp16).
    Raises :class:`KernelOutputDtypeUnresolved` when a kernel's output dtype
    cannot be established or is contradictory; the caller declines lowering
    with that typed reason instead of skipping the rounding.
    """
    result: dict = {}
    module_memo: dict = {}      # the module-wide 'Tensor' answer (see _reject_unresolved_tensor_binding), computed once for all kernels
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and _is_kernel_decorated(node)):
            continue
        _reject_nonliteral_allocation_dtypes(node)      # every allocation, returned or not (LITERAL-BOUNDARY-30): the dtype-specific reason comes first
        _reject_unresolved_tensor_binding(node, tree, module_memo)    # then 'Tensor' IS the DSL class, by binding and identity (an earlier fix rounds 2-3; a module-scope
                                                         # namespace rewrite declines here, after the dtype contract -- rounds 4-8 attack pins keep their code)
        dt = _kernel_output_dtype(node)                 # return-site provenance / flow reasons keep their precedence (rounds 4-6 pins)
        _reject_unresolved_allocation_calls(node, tree)  # then every other callee / signature outside the understood spelling (#57 items 1+2, an earlier fix)
        if dt in _NARROW_FLOAT_DTYPES_SET:
            result[node.name] = dt
    return result


def _prescan_func_param_dims(tree: ast.Module, kernel_dims: dict) -> dict:
    """Map dimension names from callers to callee parameters, transitively.

    Starting from ``run_model``'s ``Tensor.from_numpy`` calls, traces which
    dims flow through every non-kernel function call in the file.  Handles
    multi-level call chains like ``run_model → model → attention_group``.

    Returns ``{func_name: {param_name: [dim_names]}}``.
    """
    func_defs: dict = {}
    run_model_node = None
    kernel_names: set = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if _is_kernel_decorated(node):
                kernel_names.add(node.name)
            else:
                func_defs[node.name] = node
            if node.name == "run_model":
                run_model_node = node
    if run_model_node is None:
        return {}

    def _dims_from_from_numpy(call: ast.Call) -> Optional[list]:
        if len(call.args) >= 2:
            arg = call.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return [d.strip() for d in arg.value.split(",") if d.strip()]
        return None

    def _is_from_numpy(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == "from_numpy")

    def _collect_local_dims(func_node: ast.FunctionDef,
                            seed: dict) -> dict:
        """Collect variable dims within a function from from_numpy + kernels."""
        local = dict(seed)
        for stmt in func_node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not (len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                continue
            tgt = stmt.targets[0].id
            val = stmt.value
            if _is_from_numpy(val):
                dims = _dims_from_from_numpy(val)
                if dims:
                    local[tgt] = dims
            elif (isinstance(val, ast.Call)
                  and isinstance(val.func, ast.Name)
                  and val.func.id in kernel_dims):
                local[tgt] = list(kernel_dims[val.func.id])
        return local

    def _propagate_calls(func_node: ast.FunctionDef,
                         local_dims: dict,
                         result: dict) -> bool:
        """Find calls to non-kernel functions and map arg dims to params."""
        changed = False
        for stmt in ast.walk(func_node):
            if not isinstance(stmt, ast.Call):
                continue
            if not isinstance(stmt.func, ast.Name):
                continue
            callee = stmt.func.id
            if callee not in func_defs or callee in kernel_names:
                continue
            callee_def = func_defs[callee]
            params = [a.arg for a in callee_def.args.args]
            existing = result.get(callee, {})
            for i, arg in enumerate(stmt.args):
                if (i < len(params) and isinstance(arg, ast.Name)
                        and arg.id in local_dims
                        and params[i] not in existing):
                    existing[params[i]] = list(local_dims[arg.id])
                    changed = True
            if existing:
                result[callee] = existing
        return changed

    # Seed from run_model.
    rm_dims = _collect_local_dims(run_model_node, {})
    result: dict = {}
    _propagate_calls(run_model_node, rm_dims, result)

    # Iteratively propagate through the call graph until stable.
    changed = True
    while changed:
        changed = False
        for func_name in list(result):
            if func_name not in func_defs:
                continue
            func_node = func_defs[func_name]
            local = _collect_local_dims(func_node, result[func_name])
            if _propagate_calls(func_node, local, result):
                changed = True

    return result


def _is_direct_kernel_call(e, kernel_names) -> bool:
    """``k(...)`` where ``k`` is SPELLED like a module-level ``@kernel``.  The spelling alone proves nothing (round 10,
    kernel-call-binding-27): whether the call RESOLVES to the kernel -- no nearer binding of the name between the
    call and the module, and the kernel's only module binding is its own ``def`` -- is ``_kernel_callee_status``."""
    return isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in kernel_names


def _nearer_binding(scopes, name):
    """The nearest scope between the reading scope (last of ``scopes``) and the module (first; excluded) that binds ``name``
    in any way -- a parameter, a lambda / comprehension parameter, a local or class-body assignment, an import, a nested
    ``def`` / ``class``, a ``del``, a ``global`` / ``nonlocal`` declaration -- as ``(desc, kind, line)``; ``None`` when the
    name resolves to the module scope."""
    for sc in reversed(scopes[1:]):
        sites = sc.sites.get(name)
        if sites:
            return sc.desc, sites[0][0], sites[0][1]
    return None


def _collect_mutations(tree):
    """Round 10 (function-code-mutation-28): every VISIBLE mutation of a module-level callable, anywhere in the module
    subtree (a nested function or class body included), as ``name -> [(statement text, line)]``:

    * an attribute or subscript STORE / DELETE whose root is a name -- ``f.__code__ = g.__code__``, ``f.__defaults__ = ...``,
      ``f.__wrapped__ = ...``, ``f.__dict__['x'] = ...``, ``f.__globals__['k'] = ...``, ``del f.__kwdefaults__``;
    * ``setattr(f, ...)`` / ``delattr(f, ...)`` with the callable as a name;
    * a write through a namespace dictionary that NAMES the callable -- ``globals()['f'] = ...`` / ``del vars()['f']`` with a
      literal key.

    Calls on the namespace dictionaries (``globals().update(...)``, ``globals().__setitem__(...)``, a non-literal key) and
    ``setattr`` on a non-name target are deliberately NOT recorded here: whether such a call can execute is what the round-5
    module-effects analysis decides soundly (an invoked lambda or an eager annotation declines as ``module_binding_uncertain``;
    an uninvoked body or a postponed annotation lowers -- the reviewer's lambda_namespace_uninvoked ruling), and a syntactic
    wildcard would re-refuse those settled programs.  A subscript store on an ordinary container (``box['x'] = ...``,
    ``Y[m, n] = ...``) is recorded and harmless: the record is consulted only for a name used as an established callee.
    Nothing is inferred from what the stored value is: the analysed ``def`` of a mutated function is simply not proved to be
    its implementation.  A store inside a body that never runs (``def never(): model.__code__ = ...``) is still a record:
    conservative, typed, reported."""
    found = {}

    def add(name, stmt):
        found.setdefault(name, []).append((ast.unparse(stmt)[:80], getattr(stmt, "lineno", "?")))

    def root(e):
        hops = 0
        while isinstance(e, (ast.Attribute, ast.Subscript)):
            e, hops = e.value, hops + 1
        return e, hops

    def namespace_call(e):
        return isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in ("globals", "vars", "locals")

    def visit(n, stmt):
        if isinstance(n, ast.stmt):
            stmt = n                                    # the record names the whole statement, not the target expression
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            base, hops = root(n)
            if isinstance(base, ast.Name):
                add(base.id, stmt)
            elif namespace_call(base) and isinstance(n, ast.Subscript) and n.value is base \
                    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                add(n.slice.value, stmt)                    # globals()['f'] = ... names the callable it rebinds
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("setattr", "delattr") \
                and n.args and isinstance(n.args[0], ast.Name):
            add(n.args[0].id, stmt)
        for child in ast.iter_child_nodes(n):
            visit(child, stmt)

    visit(tree, tree)
    return found


def _lexical_scope(node):
    """Round 12 (mutation-scope-36): the ACTUAL lexical bindings of ONE scope node, by Python's rules -- ``(locals, globals,
    nonlocals)`` as sets of names.  This is NOT ``_BindingScope.sites``: that inventory deliberately over-approximates for conservative
    REFUSAL (a nested lambda's parameters, a comprehension's targets and stores inside a nested lambda body are charged to the containing
    scope so a call of the spelling is refused), and may never be used to PROVE that a load does not reach the module definition.

    * a function / lambda: its parameters; every ``Name`` stored or deleted in its own body (assignment, augmented / annotated
      assignment, ``for`` / ``with`` / ``except`` / ``match`` targets, ``del``, a walrus -- including a walrus inside a comprehension,
      which PEP 572 binds in the containing function), the names of nested ``def`` / ``class`` statements, imports; a nested def's
      decorators / defaults / annotations and a nested lambda's defaults are evaluated here, their bodies are not; a comprehension's
      own targets are the comprehension's, not this scope's; ``global`` / ``nonlocal`` declarations in the body are returned separately;
    * a comprehension: its generators' targets (every generator) are its locals; nothing else binds in it;
    * a class body: NO locals (a class-body load before the class-local store reaches the enclosing scopes -- conservative), only its
      ``global`` / ``nonlocal`` declarations."""
    locs, glob, nonl = set(), set(), set()
    comps = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def params(a):
        return a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]

    def own(n, in_comp):
        # the parts of ``n`` evaluated in THIS scope; nested scope bodies are skipped
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            locs.add(n.name)
            for sub in n.decorator_list + n.args.defaults + [d for d in n.args.kw_defaults if d] + ([n.returns] if n.returns else []) \
                    + [a.annotation for a in params(n.args) if a.annotation is not None]:
                own(sub, in_comp)
            return
        if isinstance(n, ast.ClassDef):
            locs.add(n.name)
            for sub in n.decorator_list + n.bases + [k.value for k in n.keywords]:
                own(sub, in_comp)
            return
        if isinstance(n, ast.Lambda):
            for sub in n.args.defaults + [d for d in n.args.kw_defaults if d]:
                own(sub, in_comp)
            return
        if isinstance(n, comps):
            # the comprehension's generator targets are ITS locals; a walrus anywhere inside binds HERE (PEP 572)
            for g in n.generators:
                own(g.iter, True)
                for c in g.ifs:
                    own(c, True)
            for sub in ([n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]):
                own(sub, True)
            return
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, (ast.Store, ast.Del)) and not in_comp:
                locs.add(n.id)
            return
        if isinstance(n, ast.NamedExpr):
            locs.add(n.target.id); own(n.value, in_comp); return
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                locs.add((a.asname or a.name).split(".")[0])
            return
        if isinstance(n, ast.ExceptHandler) and n.name:
            locs.add(n.name)
        if isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            locs.add(n.name)
        if isinstance(n, ast.MatchMapping) and n.rest:
            locs.add(n.rest)
        if isinstance(n, ast.Global):
            glob.update(n.names); return
        if isinstance(n, ast.Nonlocal):
            nonl.update(n.names); return
        for child in ast.iter_child_nodes(n):
            own(child, in_comp)

    if isinstance(node, funcs):
        locs.update(a.arg for a in params(node.args))
        for st in (node.body if isinstance(node.body, list) else [node.body]):
            own(st, False)
    elif isinstance(node, comps):
        for g in node.generators:
            for n in ast.walk(g.target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    locs.add(n.id)
    elif isinstance(node, ast.ClassDef):
        for st in node.body:
            for n in ast.walk(st):
                if isinstance(n, ast.Global):
                    glob.update(n.names)
                elif isinstance(n, ast.Nonlocal):
                    nonl.update(n.names)
        # class-body stores are NOT locals here: conservative (see the docstring)
    else:
        raise TypeError(type(node).__name__)
    return frozenset(locs), frozenset(glob), frozenset(nonl)


def _load_resolves_to_module(chain, name):
    """Round 12 (mutation-scope-36): does a load of ``name`` whose enclosing scopes are ``chain`` (``_lexical_scope`` triples,
    outermost first, the module excluded) reach the MODULE binding?  Python's rule, innermost out: declared ``global`` -> the module;
    declared ``nonlocal`` -> keep looking outward; a real local of the scope -> not the module; otherwise outward; nothing found ->
    the module.  Only a proved non-reference returns False; a chain this function cannot read returns True (the escape is KEPT)."""
    try:
        for locs, glob, nonl in reversed(chain):
            if name in glob:
                return True
            if name in nonl:
                continue
            if name in locs:
                return False
        return True
    except Exception:                                    # unproved: conservative
        return True


def _collect_escapes(tree, program=None):
    """Round 11 (mutation-aliases-33): ``name -> [(statement text, line), ...]`` for every module-level ``def`` name that
    ESCAPES -- is loaded anywhere in the module (every scope, executed or not) other than as the function position of a
    direct call ``name(...)``, or is deleted.  An alias binding (``alias = model``), an alias chain (``other = alias`` starts
    with the same load), a call argument (``change(model)``, ``type(model)``), a container element, an attribute / subscript
    store of it, a return value, a default, a decorator: each hands the callable object to code whose mutations of it
    (``alias.__code__ = ...``, ``f.__code__ = ...`` in the callee, ``type(model).__setattr__(model, ...)``) cannot be followed
    by spelling.  Flow-insensitive and conservative, as finding 28: recorded, not reasoned about.

    A load counts only when it RESOLVES to the module definition (round 11 commit 5, finding 27's rule applied to the loads):
    a load whose spelling is a real local or free variable where it is read -- a parameter, a loop target ``for k in Dim("K")``
    inside the kernel ``k``, a local assignment, an import, a nested ``def`` -- never names the module ``def``.  Round 12
    (mutation-scope-36): that proof is ACTUAL lexical binding resolution (``_lexical_scope`` / ``_load_resolves_to_module``,
    Python's scoping rules), NEVER the conservative-refusal inventory ``_BindingScope.sites`` -- a nested lambda's parameter, a
    comprehension's target or a store inside a nested lambda body is charged to the containing scope there so that a CALL of the
    spelling is refused, but it binds nothing in that scope, and using it as a binding hid ``unused = lambda model: model;
    alias = model; alias.__code__ = ...`` (fast 9 vs interpreter 5).  A ``global`` declaration (in a function or a class body)
    makes the load the module name; a class body never shadows; a decorator, default, annotation and the first iterable of a
    comprehension are read in the ENCLOSING scope, as Python evaluates them; whatever resolution cannot prove is an escape.
    ``program`` is accepted for compatibility and unused."""
    found = {}
    defs = {st.name for st in tree.body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not defs:
        return found
    call_funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}

    def params(a):
        return a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]

    def scope(n, chain):
        try:
            return chain + [_lexical_scope(n)]
        except Exception:                                # unreadable scope: a chain that resolves every load to the module
            return chain + [(frozenset(), frozenset(defs), frozenset())]

    def visit(n, stmt, chain):
        if isinstance(n, ast.stmt):
            stmt = n
        if isinstance(n, ast.Name):
            if n.id in defs and id(n) not in call_funcs and isinstance(n.ctx, (ast.Load, ast.Del)) and _load_resolves_to_module(chain, n.id):
                found.setdefault(n.id, []).append((ast.unparse(stmt)[:80], getattr(stmt, "lineno", "?")))
            return
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in n.decorator_list + n.args.defaults + [d for d in n.args.kw_defaults if d] + ([n.returns] if n.returns else []) \
                    + [a.annotation for a in params(n.args) if a.annotation is not None]:
                visit(sub, stmt, chain)                     # evaluated in the enclosing scope
            inner = scope(n, chain)
            for st in n.body:
                visit(st, st, inner)
        elif isinstance(n, ast.ClassDef):
            for sub in n.decorator_list + n.bases + [k.value for k in n.keywords]:
                visit(sub, stmt, chain)
            inner = scope(n, chain)                         # only the class body's global / nonlocal declarations count
            for st in n.body:
                visit(st, st, inner)
        elif isinstance(n, ast.Lambda):
            for sub in n.args.defaults + [d for d in n.args.kw_defaults if d]:
                visit(sub, stmt, chain)
            visit(n.body, stmt, scope(n, chain))
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            visit(n.generators[0].iter, stmt, chain)        # the first iterable is evaluated in the enclosing scope
            inner = scope(n, chain)
            for i, g in enumerate(n.generators):
                if i:
                    visit(g.iter, stmt, inner)
                visit(g.target, stmt, inner)
                for c in g.ifs:
                    visit(c, stmt, inner)
            for sub in ([n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]):
                visit(sub, stmt, inner)
        else:
            for child in ast.iter_child_nodes(n):
                visit(child, stmt, chain)

    visit(tree, tree, [])
    return found


def _mutation_status(prog, name):
    """``(class, detail)`` when ``name`` -- used as an established callee -- is visibly mutated (``_collect_mutations``, round
    10) or ESCAPES as a value (``_collect_escapes``, round 11: a mutation through the other reference cannot be excluded),
    else ``None``."""
    hit = prog.mutations.get(name)
    if hit:
        text, line = hit[0]
        return ("function_implementation_unproved",
                f"'{name}' is visibly mutated by '{text}' at line {line}, so the analysed definition of '{name}' is not proved to be its implementation")
    esc = prog.escapes.get(name)
    if esc:
        text, line = esc[0]
        return ("function_implementation_unproved",
                f"'{name}' escapes as a value in '{text}' at line {line} (it is used other than as the callee of a direct call: bound to another "
                f"name, passed to a call, stored, returned ...), so a mutation of it through that other reference cannot be excluded and the "
                f"analysed definition of '{name}' is not proved to be its implementation")
    return None


def _kernel_resolves(scopes, name):
    """Round 11 (readless-callee-34): the call ``name(...)`` read in the innermost of ``scopes`` RESOLVES to the
    module-level ``@kernel`` -- no nearer binding of the spelling between the call and the module, and the module binds the
    name only by the kernel's own ``def`` (finding 27's rule, applied to the call-time rounding wrap and the kernel's return
    dims, which until round 11 were keyed on the spelling whenever no ``.data`` read requested provenance)."""
    return _nearer_binding(scopes, name) is None and scopes[0].program.module_scope.kernel_is_sole(name)


def _kernel_callee_status(scopes, name):
    """Round 10 (kernel-call-binding-27): ``None`` when a call ``name(...)`` read in the innermost of ``scopes``
    RESOLVES to the module-level ``@kernel`` definition, else ``(class, detail)``: ``kernel_name_shadowed`` when the
    spelling has a nearer binding (``k = lambda a: Box()`` in ``run_model``, a parameter or default parameter ``k`` of a
    plain function, a class-body ``k``, an import, a comprehension target, a ``global`` declaration -- the call then names
    an ordinary callable whose result is NOT a DSL value), ``kernel_name_rebound`` when the module binds the kernel name
    by more than its own ``def``.  Applied to direct reads and inside rule-(iii) return analysis alike."""
    near = _nearer_binding(scopes, name)
    if near is not None:
        desc, kind, line = near
        return ("kernel_name_shadowed", f"the @kernel name '{name}' is bound in {desc} ({kind} at line {line}), so the call "
                                        f"does not reach the module-level @kernel definition")
    prog = scopes[-1].program
    if not prog.module_scope.kernel_is_sole(name):
        return "kernel_name_rebound", f"the @kernel name '{name}' has more than its own definition as a binding site in the module"
    mutated = _mutation_status(prog, name)          # round 10 (FUNCTION-CODE-MUTATION-28): k.__code__ = ... is not the kernel;
    if mutated is not None:                         # round 11 (MUTATION-ALIASES-33): nor is a kernel that escapes as a value
        return mutated
    return None


class _Program:
    """Module-wide facts behind the established-value rule (an earlier fix rounds 8 and 9): the ``@kernel`` names, the module's own
    binding scope and the MODULE-DEFINED PLAIN FUNCTIONS -- top-level, undecorated ``def`` statements -- whose results rule
    (iii) may establish per call site (``_plain_call_status``).  ``_memo`` holds one analysis per (function, established
    parameters, required tuple arity)."""

    def __init__(self, tree, kernel_names, kernel_output_dtypes=None):
        self.kernel_names = set(kernel_names)
        self.kernel_output_dtypes = dict(kernel_output_dtypes or {})   # round 11 (BUILTIN-NARROW-32): narrow output dtypes, by kernel name
        self._memo = {}
        self._pristine = {}          # (node type, position) -> the scope node in the pristine snapshot (filled by the module scope)
        self.module_scope = _BindingScope(tree, self, "the module body")        # a pristine snapshot of the whole module
        self.functions = {st.name: st for st in self.module_scope.node.body
                          if isinstance(st, ast.FunctionDef) and not st.decorator_list}
        self.mutations = _collect_mutations(self.module_scope.node)   # round 10 (FUNCTION-CODE-MUTATION-28): name -> [(text, line)]
        self.escapes = _collect_escapes(self.module_scope.node, self)  # round 11 (MUTATION-ALIASES-33): def name -> [(text, line)] of its escapes
        self.rewritten = set()       # round 10 (BUILTIN-RESULT-29): positions of the DSL builtin calls the lowering rewrote / dispatched

    @staticmethod
    def _key(node):
        return (type(node).__name__, getattr(node, "lineno", None), getattr(node, "col_offset", None),
                getattr(node, "end_lineno", None), getattr(node, "end_col_offset", None))


class _BindingScope:
    """The binding sites of ONE scope -- the module, a class body, a function, a lambda, a comprehension -- for the
    established-or-refuse rule of an earlier fix round 8 (MODULE-PROVENANCE-24 / ORDINARY-VALUE-PROOF-25 / FUNCTION-DATA-26),
    limited to the established-value forms listed below.

    A value is ESTABLISHED as a DSL value when it is (i) the immediate result of a direct ``@kernel`` call; (ii) a name with
    exactly ONE binding site in its scope, that site being an unconditional, top-level assignment statement binding the name
    to the WHOLE of an established value (``name = v``, the chained ``x = name = v``, the annotated ``name: T = v``); (iii)
    the result of a direct call of a MODULE-DEFINED PLAIN FUNCTION whose EVERY return is established -- analysed in that
    function's own scope, a parameter counting as established only when this call site passes an established argument for
    it; see ``_plain_call_status`` for the exact conditions; (iv) a target of a flat tuple unpacking ``a, b = <expr>`` whose
    right-hand side is a tuple DISPLAY of established values of matching arity, or a direct call of a module-defined plain
    function whose every return is such a display -- the target itself still satisfying (ii)'s single-binding-site condition.
    Round 10 (function-code-mutation-28): a callee visibly MUTATED anywhere in the module -- ``f.__code__ = ...``,
    ``f.__defaults__``, ``f.__wrapped__``, ``setattr(f, ...)``, a ``globals()`` / ``vars()`` write -- is
    ``function_implementation_unproved``: its ``def`` is not proved to be what runs (``_collect_mutations``).  Round 11
    (mutation-aliases-33): so is a callee that ESCAPES as a value -- bound to another name (``alias = f``, and any
    chain from it), passed to a call (``change(f)``, ``type(f)``), stored, returned -- since a mutation through the other
    reference cannot be followed by spelling (``_collect_escapes``; flow-insensitive, every scope, executed or not; a load
    counts when it RESOLVES to the module ``def`` by ACTUAL lexical binding resolution -- round 12, mutation-scope-36:
    never by this inventory, which over-approximates on purpose: a nested lambda's parameters, a comprehension's targets and
    stores inside a nested lambda body are charged to the containing scope here so that a CALL of the spelling is refused).
    A second binding site of any kind -- an assignment under an ``if`` / loop / ``try`` / ``with`` / ``match`` body
    (executed or not), an augmented / walrus target, a ``for`` / ``with`` / ``except`` / ``match`` capture, an import, a
    ``def`` / ``class``, a parameter (of the scope or of a lambda / comprehension inside it), a ``del``, a ``global`` /
    ``nonlocal`` declaration -- leaves the name unestablished, and so does any other right-hand side (a builtin / class /
    attribute call, a subscript, an attribute, a container, an operator expression, a starred or nested unpacking).
    Nothing is inferred from the kind of value a name holds: provenance is never proved by name sets.  Round 10
    (kernel-call-binding-27): a callee is judged by RESOLUTION, not spelling -- a ``@kernel`` name or a plain-function
    name with a nearer binding in the reading scope or an enclosing scope (parameter, default, local / class-body
    assignment, comprehension target, import, ``global`` / ``nonlocal``) is ``kernel_name_shadowed`` / unestablished."""

    _SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def __init__(self, node, program, desc: str, established_params=None):
        # a PRISTINE snapshot: the lowerers rewrite the live tree statement by statement (kernel calls wrapped in their
        # rounding, '.data' reads stripped, function-local imports removed by _StripInnerImports BEFORE any function is
        # visited) while rules (ii)-(iv) evaluate right-hand sides and callee bodies lazily -- so a scope is taken from the
        # module snapshot (by node type and source position; round 10, KERNEL-CALL-BINDING-27: the stripped import of a
        # kernel spelling is a binding the scope must see), and only the module itself is copied from the live tree
        if isinstance(node, ast.Module):
            node = copy.deepcopy(node)
            program._pristine.update({program._key(n): n for n in ast.walk(node) if isinstance(n, self._SCOPES)})
        else:
            node = program._pristine.get(program._key(node)) or copy.deepcopy(node)
        self.node, self.desc, self.program, self.sites = node, desc, program, {}
        kernel_names = program.kernel_names
        self.established_params = None if established_params is None else frozenset(established_params)
        body = node.body if isinstance(getattr(node, "body", None), list) else []
        self.top = {}            # name -> (class, original statement text, line) for a name STORED by a top-level statement
        self.assign = {}         # name -> right-hand side of its whole-value assignment statement (plain / chained / annotated)
        self.unpack = {}         # name -> (index, arity, right-hand side) of its flat tuple-unpacking statement
        for st in body:
            names = [n.id for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
            if not names:
                continue
            text = ast.unparse(st)[:80]
            walrus = {id(n.target) for n in ast.walk(st) if isinstance(n, ast.NamedExpr)}
            scopes = [s for s in ast.walk(st) if isinstance(s, self._SCOPES)]
            inner = {id(n) for s in scopes for n in ast.walk(s)}
            if isinstance(st, ast.Assign) and all(isinstance(t, ast.Name) for t in st.targets):
                tag = "kernel" if _is_direct_kernel_call(st.value, kernel_names) else "non_kernel_value"
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None and st.simple:
                tag = "kernel" if _is_direct_kernel_call(st.value, kernel_names) else "non_kernel_value"
            elif isinstance(st, ast.Assign):
                tag = "unpacking"
            elif isinstance(st, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
                tag = "control_flow:" + type(st).__name__.lower()
            else:
                tag = type(st).__name__.lower()
            for n in ast.walk(st):
                if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)):
                    continue
                if id(n) in inner:                              # a store inside a nested scope of this statement: fallback record only
                    self.top.setdefault(n.id, ("expression_scope", text, st.lineno))
                elif n.id not in self.top or self.top[n.id][0] == "expression_scope":
                    self.top[n.id] = ("namedexpr", text, st.lineno) if id(n) in walrus else (tag, text, st.lineno)
            if tag in ("kernel", "non_kernel_value"):
                for tg in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                    self.assign[tg.id] = st.value
            elif tag == "unpacking" and len(st.targets) == 1 and isinstance(st.targets[0], ast.Tuple) \
                    and all(isinstance(e, ast.Name) for e in st.targets[0].elts):
                for i, e in enumerate(st.targets[0].elts):      # a FLAT tuple of names; starred / nested / attribute / subscript targets stay out
                    self.unpack[e.id] = (i, len(st.targets[0].elts), st.value)

        self._module = isinstance(node, ast.Module)
        self._function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [x for x in (node.args.vararg, node.args.kwarg) if x]:
                self._add(a.arg, "a parameter", a)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in node.generators:                   # the comprehension's OWN targets are its parameters
                for n in ast.walk(g.target):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        self._add(n.id, "a comprehension target", n)
            for sub in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]) + [g.iter for g in node.generators] + [c for g in node.generators for c in g.ifs]:
                self._leaf(sub, False); self._collect(sub, False)
        else:
            self._collect(node, nested=False)

    def _add(self, name, kind, n):
        self.sites.setdefault(name, []).append((kind, getattr(n, "lineno", "?")))

    def _collect(self, node, nested: bool):
        """Walk the children of ``node``; ``nested`` is True inside a nested def / class BODY, where only ``global`` /
        ``nonlocal`` declarations can bind a name of this scope."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not nested:
                    self._add(child.name, "defined", child)
                for sub in child.decorator_list + child.args.defaults + [d for d in child.args.kw_defaults if d] + ([child.returns] if child.returns else []):
                    self._collect(sub, nested)               # evaluated in THIS scope (a walrus there binds here)
                    self._leaf(sub, nested)
                for a in child.args.posonlyargs + child.args.args + child.args.kwonlyargs + [x for x in (child.args.vararg, child.args.kwarg) if x]:
                    if a.annotation is not None:
                        self._collect(a.annotation, nested); self._leaf(a.annotation, nested)
                for st in child.body:
                    self._leaf(st, True); self._collect(st, True)
            elif isinstance(child, ast.ClassDef):
                if not nested:
                    self._add(child.name, "defined", child)
                for sub in child.decorator_list + child.bases + [k.value for k in child.keywords]:
                    self._collect(sub, nested); self._leaf(sub, nested)
                for st in child.body:
                    self._leaf(st, True); self._collect(st, True)
            elif isinstance(child, ast.Lambda):
                if not nested:
                    for a in child.args.posonlyargs + child.args.args + child.args.kwonlyargs + [x for x in (child.args.vararg, child.args.kwarg) if x]:
                        self._add(a.arg, "a lambda parameter", a)             # per the ruling: a lambda parameter of the name disqualifies it
                for sub in child.args.defaults + [d for d in child.args.kw_defaults if d] + [child.body]:
                    self._leaf(sub, nested); self._collect(sub, nested)
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                if not nested:
                    for g in child.generators:
                        for n in ast.walk(g.target):
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                                self._add(n.id, "a comprehension target", n)    # per the ruling: a comprehension target of the name disqualifies it
                parts = ([child.key, child.value] if isinstance(child, ast.DictComp) else [child.elt]) + [g.iter for g in child.generators] + [c for g in child.generators for c in g.ifs]
                for sub in parts:
                    self._leaf(sub, nested); self._collect(sub, nested)         # a walrus in a comprehension binds in the enclosing scope
            else:
                self._leaf(child, nested)
                self._collect(child, nested)

    def _leaf(self, n, nested: bool):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            if (self._module and isinstance(n, ast.Global)) or (self._function and isinstance(n, ast.Nonlocal)) or not nested:
                for nm in n.names:
                    self._add(nm, "declared global" if isinstance(n, ast.Global) else "declared nonlocal", n)
            return
        if nested:
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            self._add(n.id, "deleted" if isinstance(n.ctx, ast.Del) else "stored", n)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                self._add((a.asname or a.name).split(".")[0], "imported", n)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            self._add(n.name, "except-captured", n)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            self._add(n.name, "match-captured", n)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            self._add(n.rest, "match-captured", n)


    def kernel_is_sole(self, name: str) -> bool:
        """The kernel name has exactly one binding site in the module subtree: its own ``def``."""
        return [k for k, _ in self.sites.get(name, ())] == ["defined"]

    def name_status(self, name: str, scopes, stack=(), seen=frozenset()):
        """``None`` when ``name`` is ESTABLISHED in this scope (the innermost of ``scopes``), else ``(class, detail)``."""
        sites = self.sites.get(name, [])
        if not sites:
            return "name_unbound_in_scope", f"'{name}' has no binding in {self.desc}, so it is a free variable resolved outside the scope"
        if len(sites) > 1:
            where = ", ".join(f"{k} at line {l}" for k, l in sites[:6]) + (", ..." if len(sites) > 6 else "")
            return "name_multiply_bound", f"'{name}' has {len(sites)} binding sites in {self.desc} or the scopes nested in it ({where})"
        kind, line = sites[0]
        if kind == "a parameter" and self.established_params is not None and name in self.established_params:
            return None                                       # rule (iii): an established argument reaches this parameter from the call site
        if kind == "stored":
            tag, text, at = self.top.get(name, ("expression_scope", "", line))
            if tag in ("kernel", "non_kernel_value"):
                if (id(self), name) in seen:
                    return "name_bound_to_non_kernel_value", f"'{name}' is bound once, by '{text}' at line {at}, whose value refers back to '{name}' itself"
                st = _established(scopes, self.assign[name], stack, seen | {(id(self), name)})
                if st is None:
                    return None
                return "name_bound_to_non_kernel_value", f"'{name}' is bound once, by '{text}' at line {at}, whose value is not established: {st[1]}"
            if tag == "unpacking":
                info = self.unpack.get(name)
                if info is None:
                    return "name_bound_by_unpacking", f"'{name}' is bound once, as one target of the unpacking assignment '{text}' at line {at}, whose targets are not a flat tuple of names"
                i, arity, value = info
                if (id(self), name) in seen:
                    return "name_bound_by_unpacking", f"'{name}' is bound once, as one target of the unpacking assignment '{text}' at line {at}, whose value refers back to '{name}' itself"
                why = _unpack_status(scopes, value, arity, stack, seen | {(id(self), name)})
                if why is None:
                    return None
                return "name_bound_by_unpacking", f"'{name}' is bound once, as target {i + 1} of {arity} of the unpacking assignment '{text}' at line {at}, whose value is not an established tuple: {why}"
            if tag.startswith("control_flow:"):
                return "name_bound_under_control_flow", f"'{name}' is bound once in {self.desc}, inside the '{tag.split(':')[1]}' statement at line {at} (stored at line {line})"
            if tag == "expression_scope":
                return "name_bound_inside_an_expression_scope", f"'{name}' is stored at line {line} inside a nested definition or expression scope of {self.desc}, not by a top-level statement of {self.desc}"
            return "name_bound_by_" + tag, f"'{name}' is bound once in {self.desc}, by the {tag} statement '{text}' at line {at}"
        more = " (no established argument reaches it from this call site)" if kind == "a parameter" and self.established_params is not None else ""
        return "name_bound_by_" + kind.replace("a parameter", "parameter").replace("declared ", "").replace("-", "_").replace(" ", "_"), f"'{name}' is {kind} at line {line} in {self.desc}{more}"


_DSL_BUILTIN_RESULTS = ("apply_mask", "top_k_mask", "gather", "concat", "stack", "slice_at")   # by NAME; plus Tensor.from_numpy


def _dsl_builtin_call(e):
    """``(binding name, operation)`` for a call spelled like a lowered DSL builtin -- ``apply_mask(...)`` -> ("apply_mask",
    "apply_mask"), ``Tensor.from_numpy(...)`` -> ("Tensor", "from_numpy") -- else ``None``."""
    if not isinstance(e, ast.Call):
        return None
    if isinstance(e.func, ast.Name) and e.func.id in _DSL_BUILTIN_RESULTS:
        return e.func.id, e.func.id
    if isinstance(e.func, ast.Attribute) and e.func.attr == "from_numpy" and isinstance(e.func.value, ast.Name) and e.func.value.id == "Tensor":
        return "Tensor", "from_numpy"
    return None


def _builtin_signature_reason(op, call):
    """The reason the call does not match the signature the lowering supports for ``op``, else ``None`` (the lowering of
    ``gather`` / ``concat`` / ``slice_at`` additionally needs the axis resolved -- see ``_Program.rewritten``)."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords):
        return "the call passes *args / **kwargs"
    kws = {kw.arg for kw in call.keywords}
    n = len(call.args)
    if op == "apply_mask":            # apply_mask(tensor, mask, masked_value=v) -> np.where(mask > 0.5, tensor, v)
        if not (2 <= n <= 3 and kws <= {"masked_value"} and not (n == 3 and "masked_value" in kws)):
            return "apply_mask takes (tensor, mask[, masked_value])"
    elif op == "top_k_mask":          # left as a plain call; the ndarray dispatcher _np_top_k_mask(arr, k, axis=-1) takes an INTEGER axis
        axis = call.args[2] if n == 3 else next((kw.value for kw in call.keywords if kw.arg == "axis"), None)
        if not (2 <= n <= 3 and kws <= {"axis"} and not (n == 3 and "axis" in kws)):
            return "top_k_mask takes (tensor, k[, axis])"
        if axis is not None and not (isinstance(axis, ast.Constant) and isinstance(axis.value, int) and not isinstance(axis.value, bool)):
            return "top_k_mask is dispatched on raw arrays with an INTEGER axis only (a dim name is not resolved on the fast path)"
    elif op == "gather":              # gather(table, index, axis) -> np.take(table, index.astype(int), axis=pos)
        if not ((n == 3 and not kws) or (n == 2 and kws == {"axis"})):
            return "gather takes (table, index, axis)"
    elif op == "concat":              # concat(tensors, along_dim) -> np.concatenate(tensors, axis=pos)
        if not ((n == 2 and not kws) or (n == 1 and kws == {"along_dim"})):
            return "concat takes (tensors, along_dim)"
    elif op == "stack":               # stack(tensors, new_dim, size=None) -> np.stack(tensors, axis=0)
        if not (1 <= n <= 3 and kws <= {"new_dim", "size"} and n + len(kws) >= 2):
            return "stack takes (tensors, new_dim[, size])"
    elif op == "slice_at":            # slice_at(tensor, axis, index) -> np.take(tensor, index, axis=pos)
        if not ((n == 3 and not kws) or (n == 2 and kws == {"index"})):
            return "slice_at takes (tensor, axis, index)"
    elif op == "from_numpy":          # Tensor.from_numpy(array, dims, *, role, dtype, sparsity) -> np.asarray(array, dtype=...)
        if not (1 <= n <= 2 and kws <= {"role", "dtype", "sparsity"}):
            return "Tensor.from_numpy takes (array, dims, *, role, dtype, sparsity)"
    return None


def _literal_number(e):
    """The value of a numeric literal ``e`` (``1.0``, ``-1e9``, ``+2``), else ``None``."""
    sign = 1.0
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
        sign, e = (-1.0 if isinstance(e.op, ast.USub) else 1.0), e.operand
    if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) and not isinstance(e.value, bool):
        return sign * float(e.value)
    return None


def _dual_representable(v):
    """``v`` is a finite number exactly representable in bf16 AND fp16, so rounding it to either is a no-op."""
    if v is None or not np.isfinite(v) or abs(v) > 65504.0:
        return False
    from .core import _truncate_to_dtype
    return float(np.float16(v)) == v and float(_truncate_to_dtype(np.array([v], dtype=np.float64), "bf16")[0]) == v


def _static_dtype(scopes, expr, seen=frozenset()):
    """Round 11 (builtin-narrow-32): the DECLARED dtype class of the DSL value ``expr`` evaluates to, when the syntax of
    the program proves it -- ``'bf16'`` / ``'fp16'`` (a literal narrow dtype), ``'wide'`` (fp32 / fp64 / an integer dtype: the
    interpreter stores these without rounding), else ``None`` (not proved).  Proved by: a ``Tensor.from_numpy`` with a literal
    (or absent, = fp32) dtype; a direct ``@kernel`` call that RESOLVES to the kernel (its established output dtype, narrow or
    wide); a name bound once, unconditionally, in its scope to such a value; a builtin applied to such an operand
    (``apply_mask`` / ``top_k_mask`` / ``slice_at`` / ``gather`` keep the operand's dtype; ``concat`` / ``stack`` take the FIRST
    operand's, so they are proved only when every operand agrees).  Nothing is inferred from spelling or from names."""
    prog = scopes[-1].program
    b = _dsl_builtin_call(expr)
    if b is not None:
        _, op = b
        if op == "from_numpy":
            kw = next((k.value for k in expr.keywords if k.arg == "dtype"), None)
            if kw is None:
                return "wide"
            if isinstance(kw, ast.Constant) and isinstance(kw.value, str):
                return kw.value if kw.value in _NARROW_FLOAT_DTYPES_SET else "wide"
            return None
        if op in ("apply_mask", "top_k_mask", "slice_at", "gather"):
            return _static_dtype(scopes, expr.args[0], seen) if expr.args else None
        if op in ("concat", "stack"):
            seq = expr.args[0] if expr.args else next((k.value for k in expr.keywords if k.arg == "tensors"), None)
            if isinstance(seq, (ast.List, ast.Tuple)) and seq.elts and not any(isinstance(e, ast.Starred) for e in seq.elts):
                dts = [_static_dtype(scopes, e, seen) for e in seq.elts]
                return dts[0] if dts[0] is not None and all(d == dts[0] for d in dts) else None
            return None
        return None
    if _is_direct_kernel_call(expr, prog.kernel_names) and _kernel_callee_status(scopes, expr.func.id) is None:
        return prog.kernel_output_dtypes.get(expr.func.id, "wide")
    if isinstance(expr, ast.Name):
        for i in range(len(scopes) - 1, -1, -1):
            sc = scopes[i]
            sites = sc.sites.get(expr.id)
            if not sites:
                continue
            if len(sites) == 1 and sites[0][0] == "stored" and sc.top.get(expr.id, ("",))[0] in ("kernel", "non_kernel_value") \
                    and (id(sc), expr.id) not in seen:
                return _static_dtype(scopes[:i + 1], sc.assign[expr.id], seen | {(id(sc), expr.id)})
            return None
    return None


def _builtin_rounding_reason(scopes, call, op):
    """Round 11 (builtin-narrow-32): the reason the rewrite of ``call`` does NOT reproduce the interpreter's rounding of
    the result to its declared dtype, else ``None`` -- the fifth proof of ``_builtin_call_status``.  ``Tensor.from_numpy`` is
    rounded inline for a literal narrow dtype (a non-literal dtype is the deferred dynamic-dtype contract); ``apply_mask``
    is wrapped in the truncation when its operand's dtype is proved narrow and left unrounded when proved wide, so with an
    UNPROVED operand dtype only a masked value representable in bf16 and fp16 alike is a faithful rewrite; ``concat`` /
    ``stack`` need every operand's dtype proved and equal; ``gather`` / ``slice_at`` / ``top_k_mask`` have nothing to round."""
    if op == "from_numpy":
        kw = next((k.value for k in call.keywords if k.arg == "dtype"), None)
        if kw is not None and not (isinstance(kw, ast.Constant) and isinstance(kw.value, str)):
            return "whose dtype is not a literal string, so the interpreter's rounding of its result is not reproduced (a dynamic dtype is deferred)"
        return None
    if op == "apply_mask":
        if call.args and _static_dtype(scopes, call.args[0]) is not None:
            return None
        mv = call.args[2] if len(call.args) > 2 else next((k.value for k in call.keywords if k.arg == "masked_value"), None)
        if mv is not None and _dual_representable(_literal_number(mv)):
            return None
        shown = ast.unparse(mv) if mv is not None else "-1e9 (the default)"
        return (f"whose first argument's declared dtype is not established, so the interpreter's rounding of the masked value {shown} "
                f"to that dtype is not reproduced (a masked value exactly representable in bf16 and fp16 would need no rounding)")
    if op in ("concat", "stack"):
        if _static_dtype(scopes, call) is None:
            return ("whose operands' declared dtypes are not all established as one dtype, so the interpreter's rounding of every "
                    "element to the first operand's dtype is not reproduced")
        return None
    return None


def _builtin_call_status(scopes, call):
    """Round 10 (builtin-result-29, ruling (i')): ``None`` when ``call`` is the immediate result of a direct
    DSL builtin operation the lowering handles -- established by construction -- else ``(class, detail)``.  Four proofs, none
    of them a spelling exemption: (a) the callee BINDING resolves to the DSL import (no nearer binding of the spelling between
    the call and the module -- finding 27's discipline; the module binds the name only by imports, each verified by object
    IDENTITY against ``solar.einsum_lib.core``); (b) the callee is not visibly mutated (finding 28); (c) the call matches the
    signature the lowering supports (``_builtin_signature_reason``); (d) the lowering has in fact rewritten / dispatched THIS
    call before the read (``_Program.rewritten``, by source position: the axis builtins ``gather`` / ``concat`` / ``slice_at``
    record only when their axis resolved; a module-level call is never rewritten; a call inside a function the lowering visits
    after the reader is not yet recorded); (e) round 11 (builtin-narrow-32): the rewrite reproduces the interpreter's
    rounding of the result to its declared dtype (``_builtin_rounding_reason``)."""
    prog = scopes[-1].program
    name, op = _dsl_builtin_call(call)
    src = ast.unparse(call)[:60]
    shown = "Tensor.from_numpy" if op == "from_numpy" else op
    near = _nearer_binding(scopes, name)
    if near is not None:
        return "call_result", (f"{src!r} is spelled like the DSL builtin {shown}, but '{name}' is bound in {near[0]} ({near[1]} at line {near[2]}), "
                               f"so the call does not reach the DSL import")
    msites = prog.module_scope.sites.get(name, [])
    if not msites or any(k != "imported" for k, _ in msites):
        where = ", ".join(f"{k} at line {l}" for k, l in msites[:4]) or "nowhere"
        return "call_result", f"{src!r} is spelled like the DSL builtin {shown}, but the module binds '{name}' {where}, not only by imports of solar.einsum_lib"
    for st in ast.walk(prog.module_scope.node):
        if isinstance(st, ast.ImportFrom):
            for alias in st.names:
                if (alias.asname or alias.name) == name:
                    why = _import_identity_reason(st, alias, expected=name)
                    if why is not None:
                        return "call_result", f"{src!r} is spelled like the DSL builtin {shown}, but its binding is not the DSL object: {why[1]}"
        elif isinstance(st, ast.Import) and any((a.asname or a.name).split(".")[0] == name for a in st.names):
            return "call_result", f"{src!r} is spelled like the DSL builtin {shown}, but '{name}' is bound by a plain import at line {st.lineno}"
    mutated = _mutation_status(prog, name)
    if mutated is not None:
        return mutated
    sig = _builtin_signature_reason(op, call)
    if sig is not None:
        return "call_result", f"{src!r} is a call of the DSL builtin {shown} whose form the lowering does not rewrite ({sig})"
    if prog._key(call) not in prog.rewritten:
        if op in ("gather", "concat", "slice_at"):
            return "call_result", (f"{src!r} is a call of the DSL builtin {shown} whose axis the lowering could not resolve (no dims are known for its "
                                   f"first argument at that statement), so the call is left to the runtime dispatcher and its result is not established")
        return "call_result", (f"{src!r} is a call of the DSL builtin {shown} that the lowering had not rewritten before this read (a module-level "
                               f"statement is never rewritten; a function defined after its reader is lowered later)")
    rounding = _builtin_rounding_reason(scopes, call, op)          # round 11 (BUILTIN-NARROW-32): the rewrite rounds as the interpreter does
    if rounding is not None:
        return "call_result", f"{src!r} is a call of the DSL builtin {shown} {rounding}"
    return None


def _established(scopes, expr, stack=(), seen=frozenset()):
    """``None`` when ``expr`` is an ESTABLISHED DSL value in the innermost scope of ``scopes`` (module first), else
    ``(class, detail)``: (i) a direct ``@kernel`` call whose callee RESOLVES to the kernel (no nearer binding of the name,
    the kernel name bound in the module only by its own ``def``; round 10); (ii) an established
    name (``_BindingScope.name_status``); (iii) a direct call, by name, of a module-defined plain function whose every
    return is established (``_plain_call_status``).  Any other expression -- a call of a builtin / class / attribute /
    lambda, a subscript, an attribute, a container, an operator expression -- is not followed."""
    sc = scopes[-1]
    prog = sc.program
    if _is_direct_kernel_call(expr, prog.kernel_names):
        return _kernel_callee_status(scopes, expr.func.id)       # round 10: the RESOLVED callee, never the spelling alone
    if isinstance(expr, ast.Name):
        return sc.name_status(expr.id, scopes, stack, seen)
    if _dsl_builtin_call(expr) is not None:
        return _builtin_call_status(scopes, expr)              # round 10 (i'): a lowered DSL builtin's result, binding and call proved
    src = ast.unparse(expr)
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name):
            why = _plain_call_status(scopes, expr, stack, seen)
            if why is None:
                return None
            if why[0] != "call_result":
                return why                                   # function_implementation_unproved keeps its own class
            return "call_result", f"{src!r} is the result of a call that is not a direct @kernel call nor an established call of a module-defined plain function ({why[1]})"
        return "call_result", f"{src!r} is the result of a call that is not a direct @kernel call, whose value the lowering does not follow"
    kind = {ast.Subscript: "subscript", ast.Attribute: "attribute"}.get(type(expr), "expression")
    what = {"subscript": "a subscript", "attribute": "an attribute of another value", "expression": f"a {type(expr).__name__} expression"}[kind]
    return kind, f"{src!r} is {what}, whose value the lowering does not follow"


def _plain_call_status(scopes, call, stack=(), seen=frozenset(), tuple_arity=None):
    """Rule (iii) / (iv) verdict for the call ``f(...)`` by NAME: ``None`` when established, else ``(class, text)`` --
    ``call_result`` for every unmet syntactic condition (``_plain_call_reason`` lists them), ``function_implementation_unproved``
    (round 10, function-code-mutation-28) when the callee is visibly mutated anywhere in the module."""
    why = _plain_call_reason(scopes, call, stack, seen, tuple_arity)
    if why is None:
        return None
    return why if isinstance(why, tuple) else ("call_result", why)


def _plain_call_reason(scopes, call, stack=(), seen=frozenset(), tuple_arity=None):
    """Rule (iii) -- and (iv) when ``tuple_arity`` is given: ``None`` when ``call`` (``f(...)`` by NAME) is a direct call of a
    MODULE-DEFINED PLAIN FUNCTION whose EVERY return is an established value (a tuple display of ``tuple_arity``
    established values), else the reason as text.  Conditions, all syntactic: the callee name has no binding in any scope
    between the call and the module and exactly one binding site in the module, its ``def``; that ``def`` is a top-level,
    undecorated, non-async ``FunctionDef`` without ``*args`` / ``**kwargs``; the call passes no ``*args`` / ``**kwargs`` and
    binds declared parameters only, each at most once and every required one; the function is not on the analysis stack
    (no recursion or mutual recursion); it is no generator / coroutine, has at least one ``return``, no bare ``return`` and
    ends in a top-level ``return`` (no fall-through); every ``return`` value is established in the function's own scope,
    where a parameter is established only when this call site passes an established argument for it."""
    prog = scopes[-1].program
    name = call.func.id
    near = _nearer_binding(scopes, name)          # round 10: one resolution for kernel and plain-function callees
    if near is not None:
        return f"'{name}' is bound in {near[0]} ({near[1]} at line {near[2]}), so the call does not reach a module-level definition"
    msites = prog.module_scope.sites.get(name, [])
    if [k for k, _ in msites] != ["defined"]:
        if not msites:
            return f"'{name}' has no module-level definition"
        return f"'{name}' has {len(msites)} binding sites in the module (" + ", ".join(f"{k} at line {l}" for k, l in msites[:4]) + ")"
    mutated = _mutation_status(prog, name)          # round 10 (FUNCTION-CODE-MUTATION-28): f.__code__ = ... is not the analysed def;
                                                    # round 11 (MUTATION-ALIASES-33): nor is a def that escapes as a value (alias / argument / type(f))
    if mutated is not None:
        return mutated
    f = prog.functions.get(name)
    if f is None:
        return f"'{name}' is not a module-defined plain function (a class, a decorated or an async definition)"
    if name in stack:
        return f"'{name}' is called recursively (" + " -> ".join(stack + (name,)) + ")"
    if f.args.vararg is not None or f.args.kwarg is not None:
        return f"'{name}' declares *args / **kwargs"
    if any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords):
        return f"the call {ast.unparse(call)[:60]!r} passes *args / **kwargs"
    posonly = [a.arg for a in f.args.posonlyargs]
    positional = posonly + [a.arg for a in f.args.args]
    kwonly = [a.arg for a in f.args.kwonlyargs]
    if len(call.args) > len(positional):
        return f"the call passes {len(call.args)} positional arguments, '{name}' declares {len(positional)}"
    bound = dict(zip(positional, call.args))
    for kw in call.keywords:
        if kw.arg not in positional + kwonly or kw.arg in posonly:
            return f"the keyword '{kw.arg}' is not a (keyword-bindable) parameter of '{name}'"
        if kw.arg in bound:
            return f"the parameter '{kw.arg}' of '{name}' is passed twice"
        bound[kw.arg] = kw.value
    required = positional[:len(positional) - len(f.args.defaults)] + [a for a, d in zip(kwonly, f.args.kw_defaults) if d is None]
    missing = [p for p in required if p not in bound]
    if missing:
        return f"the call does not bind the parameter(s) {missing} of '{name}'"
    est = frozenset(p for p, a in bound.items() if _established(scopes, a, stack, seen) is None)
    key = (name, est, tuple_arity)
    if key not in prog._memo:
        prog._memo[key] = _returns_status(prog, f, est, stack + (name,), tuple_arity)
    return prog._memo[key]


def _returns_status(prog, f, est, stack, tuple_arity):
    """Every ``return`` of the plain function ``f``, analysed in ``f``'s own scope with the parameters ``est`` established."""
    scopes = [prog.module_scope, _BindingScope(f, prog, f"the function '{f.name}'", established_params=est)]
    rets, other = [], []

    def walk(n):
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue                                           # another scope's returns / yields
            if isinstance(ch, (ast.Yield, ast.YieldFrom, ast.Await)):
                other.append(ch)
            elif isinstance(ch, ast.Return):
                rets.append(ch)
            walk(ch)
    walk(f)
    if other:
        return f"'{f.name}' is a generator / coroutine function ({type(other[0]).__name__} at line {other[0].lineno})"
    if not rets:
        return f"'{f.name}' has no return statement"
    bare = [r for r in rets if r.value is None]
    if bare:
        return f"'{f.name}' has a bare return at line {bare[0].lineno}"
    if not isinstance(f.body[-1], ast.Return):
        return f"'{f.name}' can fall through its last statement (line {f.body[-1].lineno}) without a return"
    for r in rets:
        v = r.value
        if tuple_arity is None:
            st = _established(scopes, v, stack)
            if st is not None:
                return f"the return at line {r.lineno} of '{f.name}' ({ast.unparse(v)[:60]!r}) is not established: {st[1]}"
            continue
        if not (isinstance(v, ast.Tuple) and len(v.elts) == tuple_arity and not any(isinstance(e, ast.Starred) for e in v.elts)):
            return f"the return at line {r.lineno} of '{f.name}' is not a tuple display of {tuple_arity} values"
        for i, e in enumerate(v.elts):
            st = _established(scopes, e, stack)
            if st is not None:
                return f"element {i + 1} of the tuple returned at line {r.lineno} of '{f.name}' ({ast.unparse(e)[:60]!r}) is not established: {st[1]}"
    return None


def _unpack_status(scopes, value, arity, stack=(), seen=frozenset()):
    """Rule (iv): ``None`` when ``value`` is a tuple DISPLAY of ``arity`` established values, or a direct call of a module-defined
    plain function whose every return is such a display; else the reason as text."""
    if isinstance(value, ast.Tuple):
        if len(value.elts) != arity:
            return f"the tuple display has {len(value.elts)} values for {arity} targets"
        for i, e in enumerate(value.elts):
            if isinstance(e, ast.Starred):
                return f"element {i + 1} of the tuple display is starred"
            st = _established(scopes, e, stack, seen)
            if st is not None:
                return f"element {i + 1} of the tuple display ({ast.unparse(e)[:60]!r}) is not established: {st[1]}"
        return None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        st = _plain_call_status(scopes, value, stack, seen, tuple_arity=arity)
        return None if st is None else st[1]
    return f"the right-hand side {ast.unparse(value)[:60]!r} is neither a tuple display nor a direct call of a module-defined plain function"


class _ScopeStack:
    """Mixin for the two non-kernel lowerers: the stack of ``_BindingScope`` objects (module first) and the single
    established-or-refuse decision for a ``.data`` read."""

    def _scoped(self, node, desc: str):
        self._scopes.append(_BindingScope(node, self._program, desc))
        try:
            self.generic_visit(node)
        finally:
            self._scopes.pop()
        return node

    def visit_Lambda(self, node):
        return self._scoped(node, f"a lambda at line {getattr(node, 'lineno', '?')}")

    def visit_ListComp(self, node):
        return self._scoped(node, f"a comprehension at line {getattr(node, 'lineno', '?')}")

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp

    def _data_status(self, node, pristine=None):
        """``None`` when the ``.data`` read ``node`` has an ESTABLISHED DSL base, else ``(kind, detail)``.  ``pristine`` is the
        base as written (the live base has been lowered first since round 10, so a builtin call read directly is recorded)."""
        if not isinstance(node.ctx, ast.Load):
            return "stored_or_deleted_attribute", "the attribute is a store / delete target, which the lowering never rewrites"
        return _established(self._scopes, pristine if pristine is not None else node.value)   # rounds 8-10: (i)-(iv) and (i')

    def _record_data(self, node, pristine=None):
        """Record an unestablished ``.data`` read (line, source, scope, kind, detail); the caller refuses the program."""
        kind, detail = self._data_status(node, pristine)
        self.unresolved.append((getattr(node, "lineno", "?"), ast.unparse(pristine if pristine is not None else node.value), self._scopes[-1].desc, kind, detail))

    def _visit_data_read(self, node):
        """``.data``: lower the base FIRST (round 10 -- a DSL builtin call the lowering rewrites is recorded by then), judge the
        read on the base as written, and replace it by the lowered base only when established."""
        pristine = copy.deepcopy(node.value)
        self.generic_visit(node)
        established = self._data_status(node, pristine) is None
        if not established:
            self._record_data(node, pristine)
        return node.value if established else node


class _ModuleStatementLowerer(_ScopeStack, ast.NodeTransformer):
    """The lowering of MODULE-LEVEL statements (entry-call-rounding-19 / MODULE-DATA-TRANSFORM-23 / round 8):
    it touches only values ESTABLISHED as DSL values by the syntax of the statement itself and leaves everything else
    byte-identical.

    * a call of a ``@kernel`` is a DSL value: a narrow-declared result is rounded at its call site (the transform
      ``run_model`` gets), and a ``.data`` read of it is the raw array;
    * a module name bound ONCE, unconditionally, at the top level of the module to an established value -- a direct
      ``@kernel`` call, an established name, a call of a module-defined plain function whose every return is established
      (round 9), a target of a flat tuple unpacking of such values -- is a DSL value in the module scope (``_BindingScope``);
      a class body is its own scope;
    * every other ``.data`` read -- a name bound otherwise or more than once, a lambda / comprehension parameter, a call
      result, a subscript, an attribute, a free variable of another scope -- is the typed refusal
      ``kernel_module_statement_unresolved:<kernel>`` rather than a guess (nothing is classified "ordinary");
    * a bare ``clear_dims()`` statement is dropped (the DSL registry does not exist on the fast path); a class body's
      methods are lowered like any plain function, its other statements like module statements."""

    def __init__(self, program, kernel_output_dtypes, function_lowerer):
        self._program, self._kernel_names, self._dtypes = program, set(program.kernel_names), dict(kernel_output_dtypes)
        self._functions, self._scopes, self.unresolved = function_lowerer, [program.module_scope], []

    def visit_ClassDef(self, node):
        self._scopes.append(_BindingScope(node, self._program, f"the body of the class '{node.name}'"))
        try:
            body = []
            for st in node.body:
                low = self._functions.visit(st) if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) else self.visit(st)
                if low is None:
                    continue
                body.extend(low if isinstance(low, list) else [low])
        finally:
            self._scopes.pop()
        node.body = body or [ast.Pass()]
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "clear_dims":
            return None
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        if node.attr == "data":
            return self._visit_data_read(node)                          # the raw array IS the lowered value
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if _is_direct_kernel_call(node, self._kernel_names) and node.func.id in self._dtypes \
                and _kernel_resolves(self._scopes, node.func.id):          # round 11 (READLESS-CALLEE-34): resolution, not spelling
            return _RunModelLowerer._rounding_call(node, self._dtypes[node.func.id], node.func.id)
        return node


class _RunModelLowerer(_ScopeStack, ast.NodeTransformer):
    """Transform non-kernel functions to operate on raw numpy arrays.

    Beyond the original duties (stripping ``clear_dims``, ``Tensor.from_numpy``
    and ``.data``), this now also tracks dimension names through the function
    so that axis-taking builtins (``gather``, ``concat``, ``stack``,
    ``slice_at``) can be lowered to their numpy equivalents with integer axes.

    Dim tracking works both within a function (``Tensor.from_numpy`` and kernel
    return annotations) and across functions (``run_model`` parameter dims are
    propagated to ``model`` parameter names via ``_prescan_func_param_dims``).
    If an axis still cannot be resolved, the original call is left unchanged
    and will raise at runtime, declining the Numba path as before.

    Round 8 (function-data-26): a ``.data`` read is replaced by its base ONLY when the base is ESTABLISHED as a
    DSL value by the syntax of the statement -- a direct ``@kernel`` call, or a name bound once, unconditionally, in the
    SAME scope to an established value (``_BindingScope``); round 9 (the narrowing) extends "established" through
    module-defined plain functions whose every return is established and through flat tuple-display unpacking
    (``_plain_call_status`` / ``_unpack_status``).  Every other ``.data`` read, in a plain function or a class method alike,
    is recorded in ``unresolved`` and the caller refuses the program with
    ``kernel_function_statement_unresolved:<kernel>`` instead of rewriting it on a guess.
    """

    def __init__(self, use_fp64: bool = True, kernel_dims: Optional[dict] = None,
                 func_param_dims: Optional[dict] = None,
                 kernel_output_dtypes: Optional[dict] = None, kernel_names: Optional[set] = None,
                 program: Optional["_Program"] = None):
        self._np_float_attr = "float64" if use_fp64 else "float32"
        self._kernel_dims: dict = kernel_dims or {}
        self._func_param_dims: dict = func_param_dims or {}
        self._kernel_output_dtypes: dict = kernel_output_dtypes or {}
        self._var_dims: dict = {}
        # kernel-call nodes whose rounding is emitted as the statement after
        # their plain assignment (``x = k(...)``; ``x = _truncate_bf16(x)``) —
        # visit_Call must not wrap those a second time
        self._rounded_by_assignment: set = set()
        self._kernel_names: set = set(kernel_names) if kernel_names is not None else set(self._kernel_output_dtypes) | set(self._kernel_dims)
        self._program = program if program is not None else _Program(ast.Module(body=[], type_ignores=[]), self._kernel_names)
        self._scopes: list = [self._program.module_scope]
        self.unresolved: list = []      # (line, source, scope, kind, detail) of every '.data' read whose base is not established

    # ------------------------------------------------------------------
    # AST visitors
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Seed ``_var_dims`` with parameter dims propagated from callers; the function body is its own binding scope."""
        old = self._var_dims
        self._var_dims = dict(self._func_param_dims.get(node.name, {}))
        self._scopes.append(_BindingScope(node, self._program, f"the function '{node.name}'"))
        try:
            self.generic_visit(node)
        finally:
            self._scopes.pop()
        self._var_dims = old
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return self._scoped(node, f"the body of the class '{node.name}'")

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        if self._is_clear_dims(node.value):
            return None
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign):
        target = self._target_name(node)

        # Tensor.from_numpy — lower and record dims.
        if self._is_tensor_from_numpy(node.value):
            from_dtype = self._from_numpy_narrow_dtype(node.value)
            if target:
                dims = self._from_numpy_dims(node.value)
                if dims:
                    self._var_dims[target] = dims
            self._note_rewrite(node.value, self._lower_from_numpy(node.value))
            node.value = self._lower_from_numpy(node.value)
            self.generic_visit(node)
            if from_dtype and target:
                trunc_fn = ("_truncate_bf16" if from_dtype == "bf16"
                            else "_truncate_fp16")
                trunc_call = ast.Assign(
                    targets=[ast.Name(id=target, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id=trunc_fn, ctx=ast.Load()),
                        args=[ast.Name(id=target, ctx=ast.Load())],
                        keywords=[],
                    ),
                    lineno=0,
                )
                return [node, trunc_call]
            return node

        # Axis-taking builtins — lower before generic_visit so their
        # arguments (plain Name nodes) are not transformed first.
        for checker, lowerer in (
            (self._is_gather_call, self._try_lower_gather),
            (self._is_concat_call, self._try_lower_concat),
            (self._is_stack_call, self._try_lower_stack),
            (self._is_slice_at, self._try_lower_slice_at),
        ):
            if checker(node.value):
                result = lowerer(node.value)
                if result is not None:
                    self._note_rewrite(node.value, result[0])
                    node.value, dims = result
                    if target and dims:
                        self._var_dims[target] = dims
                break

        # Dim-tracking for calls that this visitor does NOT lower:
        # kernel calls (dims from the annotation) and dim-preserving
        # builtins (apply_mask, top_k_mask).
        truncation_dtype = None
        if target and isinstance(node.value, ast.Call):
            fname = self._call_name(node.value)
            # round 11 (readless-callee-34): the kernel's dims and call-time rounding apply only when the call RESOLVES to
            # the module-level @kernel (no nearer binding of the spelling; the module binds it by its def alone) -- a local /
            # parameter / default `k` is an ordinary callable whose result is neither dimensioned nor rounded here
            resolves = bool(fname) and fname in self._kernel_names and _kernel_resolves(self._scopes, fname)
            if resolves and fname in self._kernel_dims:
                self._var_dims[target] = list(self._kernel_dims[fname])
            if resolves and fname in self._kernel_output_dtypes:
                # BASELINE-UNANNOTATED-NARROW-8: the rounding is gated on the ESTABLISHED output dtype of the kernel
                # (return-site provenance), never on the presence of a return-dimension annotation
                truncation_dtype = self._kernel_output_dtypes[fname]
                self._rounded_by_assignment.add(id(node.value))
            elif self._is_apply_mask(node.value) or self._is_top_k_mask(node.value):
                src = self._arg_var(node.value, 0)
                if src and src in self._var_dims:
                    self._var_dims[target] = list(self._var_dims[src])

        self.generic_visit(node)

        if truncation_dtype and target:
            trunc_call = ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=self._rounding_call(ast.Name(id=target, ctx=ast.Load()), truncation_dtype, fname),
                lineno=0,
            )
            return [node, trunc_call]
        return node

    @staticmethod
    def _rounding_call(value: ast.AST, dt: str, kernel_name: Optional[str]) -> ast.Call:
        """The call-site rounding of a kernel result: ``_truncate_bf16`` /
        ``_truncate_fp16`` for the declared (literal) narrow dtype.  There is no
        call-time dtype path (round 9): a non-literal dtype never reaches here,
        the kernel was declined at lowering."""
        return ast.Call(func=ast.Name(id="_truncate_bf16" if dt == "bf16" else "_truncate_fp16", ctx=ast.Load()),
                        args=[value], keywords=[])

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Final rounding of a narrow-declared kernel result follows the
        # DECLARED dtype whatever the syntactic form of the call site
        # (SHAPE0-BF16-FINAL-ROUNDING-1): a kernel call that is not the plain
        # value of an assignment — returned directly, an element of a returned
        # tuple / dict, nested as another call's argument, part of an
        # expression — is wrapped in the truncation helper here, exactly as
        # the interpreter's ``kernel`` wrapper rounds every narrow result.
        rounded_here = id(node) in self._rounded_by_assignment
        # round 11 (builtin-narrow-32): apply_mask's result takes its OPERAND's declared dtype in the interpreter, read here
        # from the PRISTINE first argument (before the children are lowered) and applied to the rewrite below
        apply_mask_dtype = _static_dtype(self._scopes, node.args[0]) if self._is_apply_mask(node) and node.args else None
        self.generic_visit(node)
        if not rounded_here:
            fname = self._call_name(node)
            if fname and fname in self._kernel_output_dtypes and _kernel_resolves(self._scopes, fname):
                # established dtype, annotation or not (UNANNOTATED-NARROW-8); round 11 (READLESS-CALLEE-34): wrapped only when
                # the callee RESOLVES to the kernel -- a shadowing lambda / parameter / default named like it is not rounded
                return self._rounding_call(node, self._kernel_output_dtypes[fname], fname)
        if self._is_apply_mask(node):
            # round 11 (BUILTIN-NARROW-32): the interpreter stores the masked value rounded to the operand's dtype
            # (``apply_mask(x_bf16, m)`` holds bf16(-1e9) = -998244352.0); a proved narrow operand wraps the rewrite in that rounding
            return self._note_rewrite(node, self._narrow_rounded(node, self._lower_apply_mask(node), apply_mask_dtype))
        if self._is_tensor_from_numpy(node):
            # round 11 (BUILTIN-NARROW-32): an expression-position from_numpy with a literal narrow dtype is rounded INLINE,
            # exactly as the assignment form appends ``x = _truncate_bf16(x)`` (visit_Assign) -- the interpreter's from_numpy truncates
            return self._note_rewrite(node, self._narrow_rounded(node, self._lower_from_numpy(node), self._from_numpy_narrow_dtype(node)))
        if self._is_top_k_mask(node):
            # left as a plain call: the execution namespace dispatches it to _np_top_k_mask on raw arrays (integer axis)
            if _builtin_signature_reason("top_k_mask", node) is None:
                self._program.rewritten.add(self._program._key(node))
            return node
        for checker, lowerer in (
            (self._is_gather_call, self._try_lower_gather),
            (self._is_concat_call, self._try_lower_concat),
            (self._is_stack_call, self._try_lower_stack),
            (self._is_slice_at, self._try_lower_slice_at),
        ):
            if checker(node):
                result = lowerer(node)
                return self._note_rewrite(node, result[0]) if result is not None else node
        return node

    def _narrow_rounded(self, original, lowered, dt):
        """Round 11 (BUILTIN-NARROW-32): ``lowered`` wrapped in the truncation of the literal narrow dtype ``dt`` (``'bf16'`` /
        ``'fp16'``), unchanged otherwise or when nothing was lowered."""
        if lowered is original or dt not in _NARROW_FLOAT_DTYPES_SET:
            return lowered
        return self._rounding_call(lowered, dt, None)

    def _note_rewrite(self, original, lowered):
        """Round 10 (BUILTIN-RESULT-29): record that the lowering rewrote the DSL builtin call ``original`` (by source position
        in the pristine program) when the rewrite is a real one; returns ``lowered``."""
        if lowered is not original:
            self._program.rewritten.add(self._program._key(original))
        return lowered

    def visit_Return(self, node: ast.Return) -> ast.Return:
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Replace ``tensor.data`` with ``tensor`` (raw array) -- for an ESTABLISHED DSL base only (round 8); any other
        ``.data`` read is recorded as unresolved and left exactly as written (the program is then refused)."""
        if node.attr == "data":
            return self._visit_data_read(node)
        self.generic_visit(node)
        return node

    # ------------------------------------------------------------------
    # Node classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_clear_dims(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "clear_dims")

    @staticmethod
    def _is_tensor_from_numpy(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and
                func.attr == "from_numpy" and
                isinstance(func.value, ast.Name) and
                func.value.id == "Tensor")

    @staticmethod
    def _is_apply_mask(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "apply_mask")

    @staticmethod
    def _is_top_k_mask(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "top_k_mask")

    @staticmethod
    def _is_slice_at(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "slice_at")

    @staticmethod
    def _is_stack_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "stack")

    @staticmethod
    def _is_concat_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "concat")

    @staticmethod
    def _is_gather_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "gather")

    # ------------------------------------------------------------------
    # Dim tracking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _target_name(node: ast.Assign) -> Optional[str]:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    @staticmethod
    def _call_name(node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None

    @staticmethod
    def _arg_var(call: ast.Call, pos: int) -> Optional[str]:
        if pos < len(call.args) and isinstance(call.args[pos], ast.Name):
            return call.args[pos].id
        return None

    @staticmethod
    def _string_arg(call: ast.Call, pos: int,
                    keyword: Optional[str] = None) -> Optional[str]:
        if pos < len(call.args):
            arg = call.args[pos]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        if keyword:
            for kw in call.keywords:
                if kw.arg == keyword and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        return kw.value.value
        return None

    @staticmethod
    def _from_numpy_dims(call: ast.Call) -> Optional[list]:
        if len(call.args) >= 2:
            dim_arg = call.args[1]
            if isinstance(dim_arg, ast.Constant) and isinstance(dim_arg.value, str):
                return [d.strip() for d in dim_arg.value.split(",") if d.strip()]
        return None

    @staticmethod
    def _from_numpy_narrow_dtype(call: ast.Call) -> Optional[str]:
        """Return the dtype string if it's a narrow float type, else None."""
        for kw in call.keywords:
            if kw.arg == "dtype" and isinstance(kw.value, ast.Constant):
                dt = kw.value.value
                if isinstance(dt, str) and dt in _NARROW_FLOAT_DTYPES_SET:
                    return dt
        return None

    def _resolve_axis(self, var: Optional[str], axis: str) -> Optional[int]:
        if var is None:
            return None
        dims = self._var_dims.get(var)
        if dims is None:
            return None
        try:
            return dims.index(axis)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Lowering: Tensor.from_numpy and apply_mask (unchanged)
    # ------------------------------------------------------------------

    def _lower_from_numpy(self, call: ast.Call) -> ast.Call:
        """Tensor.from_numpy(arr, dims, ...) -> np.asarray(arr, dtype=np.{float64|int64|...})"""
        arr_arg = call.args[0] if call.args else None
        if arr_arg is None:
            return call
        np_attr = _resolve_np_dtype_attr(call.keywords, self._np_float_attr)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="asarray",
                ctx=ast.Load(),
            ),
            args=[arr_arg],
            keywords=[ast.keyword(
                arg="dtype",
                value=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr=np_attr,
                    ctx=ast.Load(),
                ),
            )],
        )

    def _lower_apply_mask(self, call: ast.Call) -> ast.Call:
        """apply_mask(tensor, mask, masked_value=v) -> np.where(mask > 0.5, tensor, v)"""
        tensor_arg = call.args[0] if len(call.args) > 0 else None
        mask_arg = call.args[1] if len(call.args) > 1 else None
        masked_value = ast.Constant(value=-1e9)
        for kw in call.keywords:
            if kw.arg == "masked_value":
                masked_value = kw.value
        if len(call.args) > 2:
            masked_value = call.args[2]

        condition = ast.Compare(
            left=mask_arg,
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=0.5)],
        )
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="where",
                ctx=ast.Load(),
            ),
            args=[condition, tensor_arg, masked_value],
            keywords=[],
        )

    # ------------------------------------------------------------------
    # Lowering: axis-taking builtins → numpy equivalents
    # ------------------------------------------------------------------

    def _try_lower_gather(self, call: ast.Call) -> Optional[tuple]:
        """gather(table, index, axis) → np.take(table, index.astype(int), axis=pos)

        Returns ``(lowered_node, output_dims)`` or ``None`` when the axis
        cannot be resolved.
        """
        table_var = self._arg_var(call, 0)
        axis_str = self._string_arg(call, 2, "axis")
        if axis_str is None:
            return None
        axis_pos = self._resolve_axis(table_var, axis_str)
        if axis_pos is None:
            return None

        lowered = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="take", ctx=ast.Load()),
            args=[
                call.args[0],
                ast.Call(
                    func=ast.Attribute(
                        value=call.args[1],
                        attr="astype", ctx=ast.Load()),
                    args=[ast.Name(id="int", ctx=ast.Load())],
                    keywords=[]),
            ],
            keywords=[ast.keyword(arg="axis",
                                  value=ast.Constant(value=axis_pos))],
        )

        table_dims = self._var_dims.get(table_var) if table_var else None
        out_dims = None
        if table_dims:
            index_var = self._arg_var(call, 1)
            index_dims = self._var_dims.get(index_var) if index_var else None
            if index_dims is not None:
                out_dims = (table_dims[:axis_pos] + index_dims
                            + table_dims[axis_pos + 1:])
            else:
                out_dims = table_dims[:axis_pos] + table_dims[axis_pos + 1:]
        return lowered, out_dims

    def _try_lower_concat(self, call: ast.Call) -> Optional[tuple]:
        """concat(tensors, along_dim) → np.concatenate(tensors, axis=pos)"""
        axis_str = self._string_arg(call, 1, "along_dim")
        if axis_str is None:
            return None

        first_var = None
        if (call.args and isinstance(call.args[0], ast.List)
                and call.args[0].elts):
            first_elt = call.args[0].elts[0]
            if isinstance(first_elt, ast.Name):
                first_var = first_elt.id

        axis_pos = self._resolve_axis(first_var, axis_str)
        if axis_pos is None:
            return None

        lowered = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="concatenate", ctx=ast.Load()),
            args=[call.args[0]],
            keywords=[ast.keyword(arg="axis",
                                  value=ast.Constant(value=axis_pos))],
        )
        dims = list(self._var_dims[first_var]) if first_var and first_var in self._var_dims else None
        return lowered, dims

    def _try_lower_stack(self, call: ast.Call) -> Optional[tuple]:
        """stack(tensors, new_dim, ...) → np.stack(tensors, axis=0)

        ``stack`` always prepends a new leading dimension, so the integer axis
        is unconditionally 0.  Resolution cannot fail.
        """
        new_dim = self._string_arg(call, 1, "new_dim")

        inner_dims = None
        if (call.args and isinstance(call.args[0], ast.List)
                and call.args[0].elts):
            first_elt = call.args[0].elts[0]
            if isinstance(first_elt, ast.Name):
                inner_dims = self._var_dims.get(first_elt.id)

        lowered = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="stack", ctx=ast.Load()),
            args=[call.args[0]],
            keywords=[ast.keyword(arg="axis",
                                  value=ast.Constant(value=0))],
        )
        out_dims = ([new_dim] + inner_dims) if new_dim and inner_dims else None
        return lowered, out_dims

    def _try_lower_slice_at(self, call: ast.Call) -> Optional[tuple]:
        """slice_at(tensor, axis, index) → np.take(tensor, index, axis=pos)"""
        tensor_var = self._arg_var(call, 0)
        axis_str = self._string_arg(call, 1, "axis")
        if axis_str is None:
            return None
        axis_pos = self._resolve_axis(tensor_var, axis_str)
        if axis_pos is None:
            return None

        index_arg = call.args[2] if len(call.args) > 2 else None
        if index_arg is None:
            for kw in call.keywords:
                if kw.arg == "index":
                    index_arg = kw.value
                    break
        if index_arg is None:
            return None

        lowered = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="take", ctx=ast.Load()),
            args=[call.args[0], index_arg],
            keywords=[ast.keyword(arg="axis",
                                  value=ast.Constant(value=axis_pos))],
        )
        tensor_dims = self._var_dims.get(tensor_var) if tensor_var else None
        out_dims = ([d for d in tensor_dims if d != axis_str]
                    if tensor_dims else None)
        return lowered, out_dims


def _module_rebinds_kernel(tree: ast.Module):
    """``(kernel_name, lineno)`` when a MODULE-LEVEL statement rebinds the name
    ``kernel`` -- a ``def kernel`` / ``class kernel``, an assignment or
    annotated / augmented assignment to ``kernel``, or an import that binds
    ``kernel`` from anywhere but einsum_lib.  A program whose ``@kernel`` is not
    einsum_lib's decorator does not round like the interpreter, so it is never
    lowered (typed decline, same reason the run-time identity guard gives for a
    reflective replacement); ``None`` otherwise."""
    first_kernel = next((n.name for n in tree.body if isinstance(n, ast.FunctionDef) and _is_kernel_decorated(n)), "module")
    for stmt in tree.body:
        hit = False
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            hit = stmt.name == "kernel"
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            hit = any(isinstance(n, ast.Name) and n.id == "kernel" for t in targets for n in ast.walk(t))
        elif isinstance(stmt, ast.Import):
            hit = any((a.asname or a.name) == "kernel" for a in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            hit = any((a.asname or a.name) == "kernel" for a in stmt.names) and not (stmt.module or "").startswith("solar.einsum_lib")
        if hit:
            return first_kernel, stmt.lineno
    return None


def numba_lower(code: str, boundscheck: bool = False, use_fp64: bool = True,
                on_decline: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Transform generated einsum_lib code into Numba-accelerated equivalent.

    Args:
        code: Generated einsum_lib Python source.
        on_decline: Called with a typed reason when lowering is declined for a
            reason that must not be silent — today
            ``kernel_output_dtype_unestablished:<kernel>`` /
            ``kernel_output_dtype_conflict:<kernel>`` (final rounding of a
            narrow result cannot be established; SHAPE0-BF16-FINAL-ROUNDING-1),
            ``kernel_allocation_callee_unresolved:<kernel>`` (the binding of
            ``Tensor`` or the effects of executing the module cannot be
            established) and ``kernel_source_invalid:<kernel>`` (the original
            source parses but Python's compiler rejects it -- a misplaced
            ``from __future__`` import, a forbidden walrus -- so the
            interpreter would raise; future-annotations-16),
            ``kernel_module_statement_unresolved:<kernel>`` and
            ``kernel_function_statement_unresolved:<kernel>`` (a ``.data`` read
            whose base is not ESTABLISHED as a DSL value by the syntax of its
            own statement -- a direct @kernel call or a name bound once,
            unconditionally, in the same scope to one; an earlier fix round 8).  Other
            declines (unparsable source, unresolved dims, unsupported
            placement) keep returning None without a callback, as before.
        boundscheck: If True, emit ``@numba.njit(boundscheck=True, ...)``
            so out-of-bounds indexing raises ``IndexError`` instead of
            segfaulting.  Use during verification when the LLM-generated
            code may contain incorrect indices.  The saved Numba file
            (already verified) should use the default ``False``.
        use_fp64: When True (default), emit ``np.float64`` dtypes.
            When False, emit ``np.float32``.

    Returns the lowered code as a string, or None if lowering fails or
    numba is not available.
    """
    if not HAS_NUMBA:
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    first_kernel = next((n.name for n in tree.body if isinstance(n, ast.FunctionDef) and _is_kernel_decorated(n)), "module")
    try:
        # the ORIGINAL source must be a program Python accepts before anything is rewritten: ast.parse admits
        # forms the compiler rejects (a future import after other statements, a walrus in a comprehension
        # iterable or in a postponed annotation); the interpreter raises on them, so the fast path must too
        compile(code, "<dsl-module>", "exec", dont_inherit=True)
    except Exception as exc:          # SyntaxError, ValueError (null bytes), RecursionError, ...
        if on_decline is not None:
            where = f" (line {exc.lineno})" if getattr(exc, "lineno", None) else ""
            on_decline(f"kernel_source_invalid:{first_kernel}: {type(exc).__name__}: {getattr(exc, 'msg', None) or exc}{where}; "
                       f"Python's compiler rejects the original source, which is never lowered")
        return None
    postponed = _postponed_annotations(tree)

    rebind = _module_rebinds_kernel(tree)
    if rebind is not None:
        # the @kernel in this module is not einsum_lib's decorator (statically visible); the run-time
        # identity guard catches the reflective spellings -- same typed reason either way
        if on_decline is not None:
            on_decline(f"kernel_output_dtype_unestablished:{rebind[0]}: the @kernel decorator in the live module "
                       f"is not einsum_lib's kernel (the module rebinds the name 'kernel' at line {rebind[1]})")
        return None
    kernel_dims = _extract_kernel_return_dims(tree)
    try:
        kernel_output_dtypes = _extract_kernel_output_dtypes(tree)
    except KernelOutputDtypeUnresolved as exc:
        # never lower a kernel whose final rounding cannot be established
        if on_decline is not None:
            on_decline(exc.reason + ": " + exc.detail)
        return None
    bad = _invalid_literal_dtype(tree)      # round 12 (BASELINE-INVALID-DTYPE-37): after the kernel-body allocation contract, before any rewrite
    if bad is not None:
        if on_decline is not None:
            from .core import VALID_DTYPES as _valid
            on_decline(f"kernel_invalid_dtype_literal:{first_kernel}: the call '{bad[0]}' at line {bad[1]} passes dtype={bad[2]!r}, which is not "
                       f"one of the interpreter's dtypes {sorted(_valid)} (case-sensitive); a conservative refusal of the visible literal spelling: "
                       f"wherever that call executes the interpreter raises ValueError(\"Unknown dtype\") and the fast path does not reproduce it, "
                       f"and whether it executes (or names the DSL Tensor at all) is not decided here, so the program is left to the interpreter")
        return None
    func_param_dims = _prescan_func_param_dims(tree, kernel_dims)
    kernel_lowerer = _KernelLowerer(use_fp64=use_fp64)
    kernel_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and _is_kernel_decorated(n)}
    program = _Program(tree, kernel_names, kernel_output_dtypes)   # rounds 8 + 9: binding sites and module-defined plain functions, computed once
    run_model_lowerer = _RunModelLowerer(use_fp64=use_fp64,
                                         kernel_dims=kernel_dims,
                                         func_param_dims=func_param_dims,
                                         kernel_output_dtypes=kernel_output_dtypes,
                                         kernel_names=kernel_names, program=program)

    njit_keywords = [
        ast.keyword(arg="parallel", value=ast.Constant(value=True)),
    ]
    if boundscheck:
        njit_keywords.append(
            ast.keyword(arg="boundscheck", value=ast.Constant(value=True)),
        )

    # Strip import statements INSIDE function bodies as well: an
    # `import numpy as np` inside run_model() makes `np` function-local, so
    # converter-injected `np.asarray(...)` lines earlier in the body raise
    # UnboundLocalError at call time -> numba declines -> interpreter timeout.
    # (Module-level np/math/numba are re-injected below, so inner imports are
    # redundant anyway.)
    class _StripInnerImports(ast.NodeTransformer):
        def _clean(self, node):
            self.generic_visit(node)
            node.body = [s for s in node.body
                         if not isinstance(s, (ast.Import, ast.ImportFrom))]
            if not node.body:
                node.body = [ast.Pass()]
            return node

        def visit_FunctionDef(self, node):
            return self._clean(node)

        def visit_AsyncFunctionDef(self, node):
            return self._clean(node)

    tree = _StripInnerImports().visit(tree)

    # A @kernel defined inside another function would be routed to the
    # run_model lowerer and bypass the kernel lowerer (and its unresolved-
    # Dim check).  Unsupported placement: decline so the interpreter path
    # enforces the DSL contract.
    # Only kernels that are direct children of the module are dispatched to
    # the kernel lowerer; a kernel anywhere else (nested in a function, an
    # async wrapper, a class or a module-level if) is an unsupported
    # placement.
    dispatched = {id(n) for n in tree.body}
    for inner in ast.walk(tree):
        if (isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(inner) not in dispatched
                and _is_kernel_decorated(inner)):
            return None

    new_body = []
    module_lowerer = _ModuleStatementLowerer(
        program=program,
        kernel_output_dtypes=run_model_lowerer._kernel_output_dtypes,
        function_lowerer=run_model_lowerer)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            continue
        if isinstance(node, ast.Import):
            continue

        if isinstance(node, ast.FunctionDef):
            if _is_kernel_decorated(node):
                lowered = kernel_lowerer.visit(node)
                if kernel_lowerer.unresolved_dims:
                    # A Dim loop bound could not be established.  The DSL
                    # contract raises for unregistered dims; lowering to
                    # range(0) would instead return silent all-zero
                    # outputs.  Decline, so the interpreter path enforces
                    # the contract.
                    return None
                kernel_lowerer._dim_to_shape = {}
                njit_dec = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="numba", ctx=ast.Load()),
                        attr="njit",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=list(njit_keywords),
                )
                lowered.decorator_list = [njit_dec]
                # @kernel is recognised by OBJECT IDENTITY at run time, in the
                # live module, at the point where the decorator is applied: the
                # expression ``kernel`` is evaluated there and must be
                # einsum_lib's decorator, else the kernel is declined (round 8).
                guard = ast.Expr(value=ast.Call(func=ast.Name(id="_require_einsum_kernel", ctx=ast.Load()),
                                                args=[ast.Name(id="kernel", ctx=ast.Load()), ast.Constant(value=node.name)],
                                                keywords=[]))
                new_body.append(guard)
                new_body.append(lowered)
            else:
                lowered = run_model_lowerer.visit(node)
                new_body.append(lowered)
        else:
            # entry-call-rounding-19 / MODULE-DATA-TRANSFORM-23 / round 8: module-level code -- ``run_model = lambda inputs: ...``,
            # a class body, any statement executed at import -- goes through the NARROW transform that touches only
            # established DSL values (kernel calls rounded at the call site, their ``.data`` reads, a bare clear_dims(),
            # class methods as plain functions) and leaves every other expression byte-identical; a ``.data`` read whose base
            # is not established is a typed refusal after the loop
            lowered = module_lowerer.visit(node)
            if lowered is None:
                continue
            new_body.extend(lowered if isinstance(lowered, list) else [lowered])

    unresolved = ([("kernel_module_statement_unresolved", *u) for u in module_lowerer.unresolved]
                  + [("kernel_function_statement_unresolved", *u) for u in run_model_lowerer.unresolved])
    if unresolved:
        # round 8 (MODULE-PROVENANCE-24 / ORDINARY-VALUE-PROOF-25 / FUNCTION-DATA-26) narrowed in round 9 (the reviewed decision): a
        # '.data' read is rewritten only when its base is ESTABLISHED -- rules (i)-(iv) of _BindingScope; the first unestablished
        # read (by line) names the program's refusal
        code, line, src, where, kind, detail = min(unresolved, key=lambda u: (u[1] if isinstance(u[1], int) else 0, u[0]))
        if on_decline is not None:
            more = f" ({len(unresolved)} unestablished '.data' reads in this program)" if len(unresolved) > 1 else ""
            on_decline(f"{code}:{first_kernel}: {kind}: a '.data' attribute at line {line} is read from {src!r} in {where}; {detail}; the lowering "
                       f"rewrites a '.data' read only when its base is established -- a direct @kernel call, a name bound once and unconditionally "
                       f"in the same scope to an established value, a direct call of a module-defined plain function whose every return is "
                       f"established, a target of a flat tuple unpacking of such values, or the result of a DSL builtin (apply_mask, top_k_mask, "
                       f"gather, concat, stack, slice_at, Tensor.from_numpy) whose binding is proved and whose call the lowering rewrote with the "
                       f"interpreter's declared-dtype rounding -- so the "
                       f"program is left to the interpreter rather than rewritten on a guess{more}")
        return None
    # round 11 (readless-callee-34, with finding 33's rule applied to the kernels themselves): a kernel name the module
    # binds by more than its own def, or whose implementation is visibly mutated / escapes as a value, leaves every call of it
    # unproved -- which callable runs, and whether the call-time rounding belongs -- so the WHOLE program is a typed refusal.
    # Emitted after the '.data'-read refusals: a program with an unestablished read keeps its round 8-10 class.
    for kn in sorted(program.kernel_names):
        if not program.module_scope.kernel_is_sole(kn):
            sites = program.module_scope.sites.get(kn, [])
            if on_decline is not None:
                on_decline(f"kernel_name_rebound:{kn}: the @kernel name '{kn}' has {len(sites)} binding sites in the module ("
                           + ", ".join(f"{k} at line {l}" for k, l in sites[:4]) + "), so a call of '" + kn + "' is not proved to reach "
                           f"the kernel and its call-time rounding cannot be placed; the program is left to the interpreter")
            return None
        unproved = _mutation_status(program, kn)
        if unproved is not None:
            if on_decline is not None:
                on_decline(f"kernel_implementation_unproved:{kn}: {unproved[1]}; the lowered kernel is not proved to be what the "
                           f"interpreter runs, so the program is left to the interpreter")
            return None
    imports = ast.parse(
        # a module that postpones its annotations keeps doing so when lowered: the stripped future import is
        # re-emitted FIRST, so no annotation the interpreter never evaluates is executed by the fast path
        ("from __future__ import annotations\n" if postponed else "")
        + "import numpy as np\nimport math\nimport numba\n"
        "from math import exp, log, sqrt, cos, sin\n"
    ).body
    tree.body = imports + new_body
    ast.fix_missing_locations(tree)

    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _truncate_narrow_carrier(arr: np.ndarray, dtype_str: str, use_fp64: bool) -> np.ndarray:
    """Round to a narrow float dtype and return the SELECTED floating carrier.

    The lowered program allocates ``np.float64`` arrays under the default
    fp64 verification mode and ``np.float32`` arrays when
    ``numba_lower(..., use_fp64=False)``; the narrow-rounding helpers it calls
    on BF16/FP16 inputs, intermediates and returned outputs must hand back an
    array of that same carrier.  The shared :func:`core._truncate_to_dtype`
    is the fp64 interpreter's helper and always returns float64, which made a
    float32-mode program return a float64 ``output`` that the driver's
    output_map refuses ("candidate physical output dtype differs": logical
    bfloat16 expects a float32 carrier; SHAPE0-228-CANDIDATE-DTYPE-1).  The
    rounding itself is unchanged: bf16 through torch's round-to-nearest-even,
    fp16 through numpy's float16.
    """
    from .core import _truncate_to_dtype
    if use_fp64:
        return _truncate_to_dtype(arr, dtype_str)
    if dtype_str == "bf16":
        import torch
        src = np.asarray(arr)
        # torch needs a contiguous float32 buffer; np.ascontiguousarray would
        # promote a rank-0 array to shape (1,), so round on a flat contiguous
        # copy and restore the original shape (contiguous-scalar-p2-1).  Non-contiguous
        # (transposed / strided) inputs are copied the same way.
        flat = np.ascontiguousarray(src.reshape(-1), dtype=np.float32)
        return (torch.from_numpy(flat)
                .to(torch.bfloat16)
                .to(torch.float32)
                .numpy()
                .reshape(src.shape))
    if dtype_str == "fp16":
        return np.asarray(arr).astype(np.float16).astype(np.float32)
    return arr


def execution_namespace(use_fp64: bool = True) -> dict:
    """Globals that lowered code needs in order to run.

    ``numba_lower`` drops the generated import line, so every name the lowered
    module refers to has to be supplied here.  Exposed because a tool that
    executes the lowered module itself -- ``scripts/diff_ir_stages.py`` -- has
    to use the same globals as :func:`execute_fast`, or it is not observing the
    same execution.  ``use_fp64`` must match the mode the program was lowered
    with: it selects the floating carrier the narrow-rounding helpers return
    (float64 in the fp64 verification mode, float32 otherwise).
    """
    from .core import (Tensor, Dim, clear_dims, kernel, apply_mask,
                       slice_at, stack, concat, gather, top_k_mask, fp32)

    def _truncate_bf16(arr):
        return _truncate_narrow_carrier(arr, "bf16", use_fp64)

    def _truncate_fp16(arr):
        return _truncate_narrow_carrier(arr, "fp16", use_fp64)

    def _require_einsum_kernel(decorator, kernel_name):
        """The ``@kernel`` seen in the live module must BE einsum_lib's decorator
        (object identity), else the lowered kernel would not behave like the
        interpreted one: typed decline (kept from round 8)."""
        if decorator is not kernel:
            raise KernelOutputDtypeUnresolved(kernel_name, "unestablished",
                                              "the @kernel decorator in the live module is not einsum_lib's kernel")

    namespace = {
        "np": np, "math": math,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "cos": np.cos, "sin": np.sin, "tanh": np.tanh,
        "floor": np.floor, "log2": np.log2, "round": round,
        "erf": math.erf,
        "fp32": fp32,
        "abs": abs, "max": max, "min": min,
        "_truncate_bf16": _truncate_bf16,
        "_truncate_fp16": _truncate_fp16,
        "_require_einsum_kernel": _require_einsum_kernel,
    }
    if HAS_NUMBA:
        namespace["numba"] = numba
    namespace.update({
        "Tensor": Tensor,
        "Dim": Dim,
        "clear_dims": clear_dims,
        "kernel": kernel,
        "apply_mask": _nd(apply_mask, _np_apply_mask),
        "top_k_mask": _nd(top_k_mask, _np_top_k_mask),
        "slice_at": _nd(slice_at, _np_slice_at),
        "stack": _nd(stack, _np_stack),
        "concat": _nd(concat, _np_concat),
        "gather": _nd(gather, _np_gather),
    })
    return namespace


# ---------------------------------------------------------------------------
# ndarray-tolerant builtin wrappers for lowered code.
#
# In numba-lowered run_model() the tensors are raw np.ndarrays, but builtins
# that the AST pass leaves as plain calls (top_k_mask always; gather/stack/...
# when their lowering bails) were mapped to the core Tensor-based functions,
# which access .dim_names/.data and crash with
# "'numpy.ndarray' object has no attribute 'dim_names'".  Each builtin now
# dispatches on the first argument's type: Tensor -> core fn, ndarray -> the
# numpy-native equivalent below.
# ---------------------------------------------------------------------------

def _nd(core_fn, np_fn):
    def _dispatch(first, *args, **kwargs):
        if isinstance(first, np.ndarray):
            return np_fn(first, *args, **kwargs)
        return core_fn(first, *args, **kwargs)
    return _dispatch


def _np_apply_mask(tensor, mask, masked_value=-1e9):
    m = mask if isinstance(mask, np.ndarray) else mask.data
    return np.where(m > 0.5, tensor, masked_value).astype(tensor.dtype, copy=False)


def _np_top_k_mask(arr, k, axis=-1):
    ax = axis if axis >= 0 else arr.ndim + axis
    n = arr.shape[ax]
    kk = max(0, min(int(k), n))
    out = np.zeros_like(arr)
    if kk == n:
        out[...] = 1.0
    elif kk > 0:
        part = np.argpartition(arr, n - kk, axis=ax)
        idx = np.take(part, np.arange(n - kk, n), axis=ax)
        np.put_along_axis(out, idx, 1.0, axis=ax)
    return out


def _np_slice_at(arr, axis, index):
    return np.take(arr, int(index), axis=int(axis) if isinstance(axis, (int, np.integer)) else axis)


def _np_stack(arrays, new_dim, size=None):
    arrs = [a if isinstance(a, np.ndarray) else a.data for a in arrays]
    return np.stack(arrs, axis=0)


def _np_concat(arrays, along_dim=0):
    arrs = [a if isinstance(a, np.ndarray) else a.data for a in arrays]
    ax = along_dim if isinstance(along_dim, (int, np.integer)) else 0
    return np.concatenate(arrs, axis=ax)


def _np_gather(table, index, axis=0):
    idx = index if isinstance(index, np.ndarray) else index.data
    ax = axis if isinstance(axis, (int, np.integer)) else 0
    return np.take(table, idx.astype(np.int64), axis=ax)


def is_typed_decline(reason: Optional[str]) -> bool:
    """True when a fast-path fallback ``reason`` is a TYPED decline of the program -- one of numba_lower's refusal codes
    (``kernel_output_dtype_*``, ``kernel_allocation_callee_unresolved``, ``kernel_source_invalid``,
    ``kernel_module_statement_unresolved``, ``kernel_function_statement_unresolved``, ...) or the run-time
    ``KernelOutputDtypeUnresolved`` guard, all spelled ``kernel_<code>:<kernel>: ...``.  Everything else the fast path reports --
    generated code crashing in the lowered module (``IndexError: ...``), the lowered module failing to import, numba_lower
    returning None without a reason, numba absent -- is UNTYPED (``untyped_fast_failure_class``).  Round 10,
    agent-routing-scope-31: the fast agent's interpreter routing is typed-decline-only."""
    if not reason or not reason.startswith("kernel_"):
        return False
    code = reason.split(":", 1)[0]                      # kernel_<code>, then ':' or the end of the reason
    return len(code) > len("kernel_") and all(c.islower() or c.isdigit() or c == "_" for c in code)


def untyped_fast_failure_class(reason: Optional[str]) -> str:
    """The class of an UNTYPED fast-path failure, for the record: ``lowering_unavailable`` (numba absent, or numba_lower
    returned None without a typed reason), ``lowered_import_failure`` (the lowered module did not import or defines no
    run_model), ``fast_runtime_failure`` (the generated code raised while running on the fast path)."""
    r = reason or ""
    if r in ("numba is not installed", "numba_lower could not lower the generated code"):
        return "lowering_unavailable"
    if r.startswith("lowered code failed to import:") or r == "lowered code defines no run_model()":
        return "lowered_import_failure"
    return "fast_runtime_failure"


def execute_fast(code: str, inputs: dict, use_fp64: bool = True,
                 on_fallback: Optional[Callable[[str], None]] = None
                 ) -> Optional[dict]:
    """Lower and execute generated code with Numba acceleration.

    Uses ``boundscheck=True`` so incorrect LLM-generated indices raise
    ``IndexError`` instead of causing a segfault.

    Returns the outputs dict, or None if lowering/execution fails.

    *on_fallback* is called with the reason whenever None is returned.  Callers
    that respond by running the interpreter instead should pass one: that path
    is correct but around 100x slower, which is easy to mistake for a slow model
    if the reason is never reported.
    """
    def _decline(reason: str) -> None:
        if on_fallback is not None:
            on_fallback(reason)
        return None

    if not HAS_NUMBA:
        return _decline("numba is not installed")

    typed: list = []
    lowered_code = numba_lower(code, boundscheck=True, use_fp64=use_fp64,
                               on_decline=typed.append)
    if lowered_code is None:
        return _decline(typed[0] if typed else "numba_lower could not lower the generated code")

    namespace = execution_namespace(use_fp64=use_fp64)

    try:
        exec(lowered_code, namespace)  # noqa: S102
    except KernelOutputDtypeUnresolved as exc:
        return _decline(exc.reason + ": " + exc.detail)
    except Exception as exc:
        return _decline(f"lowered code failed to import: "
                        f"{type(exc).__name__}: {exc}")

    run_model = namespace.get("run_model")
    if run_model is None:
        return _decline("lowered code defines no run_model()")

    try:
        return run_model(inputs)
    except KernelOutputDtypeUnresolved as exc:
        return _decline(exc.reason + ": " + exc.detail)
    except Exception as exc:
        # Numba typing errors run to hundreds of lines; the head carries the
        # unsupported operation, which is the part worth reading.
        detail = " ".join(str(exc).split())
        if len(detail) > 400:
            detail = detail[:400] + " ..."
        return _decline(f"{type(exc).__name__}: {detail}")
