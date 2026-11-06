# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static integrality check of the index expressions of a lowered kernel.

``numba_lower`` turns ``@kernel`` functions into ``@numba.njit(parallel=True)``
kernels whose ``Dim(..., parallel=True)`` loops are ``numba.prange`` loops.
Numba's parfor lowering moves the body of an outermost ``prange`` into a
gufunc that iterates a ``uintp`` schedule
(``for m in range(sched[d], sched[d + n] + np.uint8(1))``), so inside that
gufunc the loop index is ``uint64`` although the outer typing of the same
function sees ``int64``.  Plain binary operators with a signed operand give
``int64`` again, but ``min``/``max`` and conditional expressions *unify*
their operands and ``unify(uint64, int64)`` is ``float64``.  An index built
that way (``t = min(m, nt - 1); X[t // ns, t % ns, h]``) fails only in
"native parfor lowering" with ``getitem(array(...), Tuple(float64, ...))``
(finding SHAPE0-205-NUMBA-FLOAT-INDEX-1, Lite rows 205 and 206).

This module types every index expression of every subscript inside the
``njit`` kernels of a lowered program with those rules and reports the ones
that are provably ``float64``.  It is a *proof of float-ness* checker, a
diagnostic: an expression whose type cannot be established (array elements
of parameters, unknown calls, values from unmodelled constructs) is not
reported, so **an empty finding list is not an integrality certificate**;
only Numba's own typing is.  Assignments are followed in program order;
``if``/``else`` branches and loop bodies are typed on copies of the
environment and joined afterwards (float on any path wins, otherwise
unknown if any path is unknown), ``and``/``or`` are value-typed over their
operands with constant short-circuits folded only for literals and aliases
bound by one statement on every path (a branch join of equal values is not a
constant; a constant a loop body reassigns is unknown-value in that body),
rounding of an untyped value is unknown (never assumed float64); a name a
loop body may rebind in the enclosing scope -- assignments, loop and
with/except targets, ``del``, nonlocal rebinds from nested functions (a nested ``global``
rebinds the module name only, never a kernel local) --
loses its constant before the body is typed, while nested-function locals and
comprehension targets do not (``except ... as`` / ``del`` are collected
conservatively; Numba rejects such kernels before index typing anyway), literal tuples/lists type their loop variable from their elements,
NumPy constructors default to float64 and array aliases keep the element
type, literal tuple/list unpacking binds each target to its element, a loop
over a known array takes its element kind, constant ``and``/``or``
short-circuits are folded as Numba folds them, and unsigned widths are kept
apart (only uint64 meets the float rule).  Loop-carried values that change
type across iterations are joined once (before/after one pass of the body),
not to a fixed point.  Known limits (recorded, not certified): values
returned by nested or unknown calls are unknown; array elements of
parameters are unknown; ``math.floor`` returns an int in Numba and is typed
so, ``np.floor`` returns a float and is typed so.

Command line::

    python -m solar.einsum_lib.lowered_index_check <lowered.py | dir>...

prints one line per finding and exits 1 when any program has a float-typed
index expression.
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

__all__ = [
    "IndexFinding",
    "check_lowered_source",
    "check_lowered_file",
    "iter_lowered_programs",
    "main",
]

# Kinds: "int" (int64), "uint" (uint64: the parfor index, np.uint64 casts and
# uint64 array elements), "uint8" / "uint16" / "uint32" (narrower unsigned,
# kept apart because Numba unifies them with int64 to int64 but with uint64
# to uint64), "float" (float64), "bool", or None (unknown).  A Kind carries
# the reason it was derived.
Kind = Tuple[Optional[str], str]

_UNKNOWN: Kind = (None, "unknown")
_PRANGE_REASON = "numba.prange index (uint64 inside the parfor gufunc)"

_INT_CASTS = {"int", "np.intp", "np.int64", "np.int32", "np.int16", "np.int8",
              "numba.intp", "numba.int64", "numba.int32", "np.intc", "round",
              "math.floor", "math.ceil", "math.trunc", "len"}
# Only 64-bit unsigned values hit Numba's unify(uint64, int64) -> float64
# rule; uint8/16/32 unify with int64 to int64 and with uint64 to uint64, so
# they keep their own kinds (never collapsed to "int", an earlier fix P2-2).
# A bool unifies with any integer kind to that kind (unify(uint64, boolean)
# is uint64), so it never makes the pair float.
_UINT_CASTS = {"np.uint64", "np.uintp", "numba.uintp", "numba.uint64"}
_NARROW_UINT_CASTS = {"np.uint32": "uint32", "np.uint16": "uint16", "np.uint8": "uint8",
                      "numba.uint32": "uint32", "numba.uint16": "uint16", "numba.uint8": "uint8"}
_NARROW_UINT_KINDS = frozenset({"uint8", "uint16", "uint32"})
_INTEGRAL_KINDS = frozenset({"int", "uint", "bool"}) | _NARROW_UINT_KINDS
_FLOAT_CALLS = {"float", "np.float32", "np.float64", "np.float16",
                "np.floor", "np.ceil", "np.trunc", "np.rint",
                "math.exp", "math.log", "math.sqrt", "math.tanh", "math.cos",
                "math.sin", "math.log2", "math.erf", "math.pow", "math.fabs",
                "exp", "log", "sqrt", "tanh", "cos", "sin", "log2", "erf",
                "np.exp", "np.log", "np.sqrt", "np.tanh", "np.cos", "np.sin"}
_ARRAY_CTORS = {"np.zeros", "np.full", "np.empty", "np.ones", "np.arange"}
_SHIFT_OPS = (ast.LShift, ast.RShift)


class IndexFinding(NamedTuple):
    """A subscript index expression that Numba types as float64."""

    file: str
    kernel: str
    lineno: int
    col: int
    index_text: str
    access: str   # "load" | "store" | "del"
    kind: str     # always "float64" today
    reason: str

    def format(self) -> str:
        return (f"{self.file}:{self.lineno}:{self.col}: {self.kernel}: "
                f"{self.access} index `{self.index_text}` is {self.kind}: "
                f"{self.reason}")


def _dotted(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _is_njit_kernel(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = _dotted(target)
        if name in ("numba.njit", "njit", "numba.jit", "jit"):
            return True
    return False


def _is_prange_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call)
            and _dotted(node.func) in ("numba.prange", "prange"))


def _is_range_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _dotted(node.func) == "range"


def _unify(kinds: List[Kind], what: str) -> Kind:
    """Numba's ``unify_types`` over min/max/ifexp operands."""
    tags = [k[0] for k in kinds]
    for k in kinds:
        if k[0] == "float":
            return ("float", f"{what} has a float64 operand: {k[1]}")
    if None in tags:
        return _UNKNOWN
    if "uint" in tags and "int" in tags:
        u = next(k for k in kinds if k[0] == "uint")
        s = next(k for k in kinds if k[0] == "int")
        return ("float", f"{what} unifies uint64 with int64 to float64 "
                         f"({u[1]}; {s[1]})")
    if "uint" in tags:
        # uint64 with bool or narrower unsigned operands stays uint64 (Numba:
        # (uint64, uint32) -> uint64, (uint64, boolean) -> uint64; a bool
        # never triggers the signed rule -- checked on numba 0.61 and 0.68)
        return ("uint", f"{what} of uint64 operands")
    if "int" in tags:
        # bool or narrower unsigned with a signed int64 unifies to int64
        return ("int", f"{what} of int64 operands")
    narrow = [t for t in tags if t in _NARROW_UINT_KINDS]
    if narrow:
        # narrow unsigned kinds (with or without bool): the widest of them
        widest = max(narrow, key=lambda t: int(t[4:]))
        return (widest, f"{what} of {widest} operands")
    return ("bool", f"{what} of bool operands")


class _KernelTyper:
    """Types expressions of one kernel body with Numba's gufunc rules."""

    def __init__(self, module_env: Dict[str, Kind]):
        self.module_env = module_env
        self.env: Dict[str, Kind] = {}
        self.arrays: Dict[str, Kind] = {}   # name -> element kind
        self.defs: Dict[str, str] = {}      # name -> defining statement text
        self.consts: Dict[str, tuple] = {}  # name -> (statically known numeric value, defining stmt position)

    # -- expressions -------------------------------------------------------

    def kind_of(self, node: ast.AST) -> Kind:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return ("bool", "bool literal")
            if isinstance(v, int):
                return ("int", "int literal")
            if isinstance(v, float):
                return ("float", f"float literal {v!r}")
            return _UNKNOWN
        if isinstance(node, ast.Name):
            if node.id in self.env:
                k = self.env[node.id]
                via = self.defs.get(node.id)
                if via and k[0] == "float":
                    return (k[0], f"`{via}`: {k[1]}")
                return k
            if node.id in self.module_env:
                return self.module_env[node.id]
            return _UNKNOWN
        if isinstance(node, ast.Attribute):
            if node.attr in ("size", "ndim"):
                return ("int", f"array .{node.attr}")
            return _UNKNOWN
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"):
                return ("int", "array shape extent")
            base = _dotted(node.value)
            if base in self.arrays and not isinstance(node.slice, ast.Slice):
                return self.arrays[base]
            return _UNKNOWN
        if isinstance(node, ast.UnaryOp):
            inner = self.kind_of(node.operand)
            if isinstance(node.op, ast.Not):
                return ("bool", "not")
            return inner   # -uint64 stays uint64 in Numba
        if isinstance(node, ast.Compare):
            return ("bool", "comparison")
        if isinstance(node, ast.BoolOp):
            # ``a or b`` / ``a and b`` evaluate to one of the OPERANDS, not to
            # a bool.  A statically decidable short circuit (constant
            # operands: ``0.5 and 1`` -> 1, ``0.0 or 1`` -> 1, ``1 or 0.5``
            # -> 1) is folded exactly as Numba folds it; otherwise the
            # operands that can still be the value are unified (``flag and
            # 0.5``: the value is ``flag`` or ``0.5`` -> float64).
            return self._boolop(node)
        if isinstance(node, ast.IfExp):
            return _unify([self.kind_of(node.body), self.kind_of(node.orelse)],
                          "conditional expression")
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        return _UNKNOWN

    def const_value(self, node: ast.AST):
        """The statically known numeric value of *node*, or None.

        Literals, negated / unary-plus literals and names bound (on every
        path) to such a literal are decidable; anything else is None.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = self.const_value(node.operand)
            if inner is None or isinstance(inner, bool):
                return None
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.Name) and node.id in self.consts:
            return self.consts[node.id][0]
        return None

    def _set_const(self, name: str, value, stmt: ast.AST) -> None:
        self.consts[name] = (value, (getattr(stmt, "lineno", None), getattr(stmt, "col_offset", None)))

    def invalidate_consts_assigned_in(self, body: Iterable[ast.AST]) -> None:
        """Forget every constant alias that *body* may reassign (an earlier fix P2-R2).

        A loop body is typed once, so a constant read in the body must not be
        the entry value when a later statement of the body (reached through
        the backedge) rebinds the name: such names are typed-but-unknown-value
        before the body is examined.  Names only read keep their constant.
        """
        for n in _assigned_names(body):
            self.consts.pop(n, None)
        # a nested ``global x; x = ...`` rebinds the MODULE name x: a module
        # constant the kernel reads is no longer known inside the loop (its
        # kind and value become unknown); a kernel-local x is untouched
        # (an earlier fix P2-R5-GLOBAL)
        for n in _module_rebinds(body):
            if n in self.module_env and n not in self.env:   # the kernel reads the MODULE name
                self.env[n] = _UNKNOWN
                self.defs[n] = f"module constant `{n}` rebound through a nested `global` declaration"
                self.consts.pop(n, None)
            # a kernel-local of the same name is untouched: `global` cannot reach it

    def _boolop(self, node: ast.BoolOp) -> Kind:
        is_or = isinstance(node.op, ast.Or)
        what = "`or`" if is_or else "`and`"
        reachable: List[Kind] = []
        for i, v in enumerate(node.values):
            last = i == len(node.values) - 1
            cv = self.const_value(v)
            if cv is not None and not last:
                truthy = bool(cv)
                if (is_or and truthy) or (not is_or and not truthy):
                    # the expression is this constant: short circuit decided
                    reachable.append(self.kind_of(v))
                    break
                # this constant can never be the value: skip it
                continue
            reachable.append(self.kind_of(v))
        if not reachable:
            return _UNKNOWN
        if len(reachable) == 1:
            k = reachable[0]
            return (k[0], f"{what} short-circuits to {k[1]}")
        return _unify(reachable, what)

    def _binop(self, node: ast.BinOp) -> Kind:
        left, right = self.kind_of(node.left), self.kind_of(node.right)
        if isinstance(node.op, ast.Div):
            return ("float", f"true division `{ast.unparse(node)}`")
        for side, k in (("left", left), ("right", right)):
            if k[0] == "float":
                return ("float", f"`{ast.unparse(node)}` has a float64 "
                                 f"{side} operand: {k[1]}")
        if isinstance(node.op, ast.MatMult):
            return _UNKNOWN
        if left[0] is None or right[0] is None:
            return _UNKNOWN
        if isinstance(node.op, _SHIFT_OPS):
            return left    # shifts keep the left operand's type
        if isinstance(node.op, ast.Pow):
            # int ** negative int is float in Numba; we cannot see the sign
            if left[0] == "uint" and right[0] == "uint":
                return ("uint", "power of uint64 operands")
            return ("int", "power of integer operands")
        unsigned = frozenset({"uint"}) | _NARROW_UINT_KINDS
        if left[0] in unsigned and right[0] in unsigned and "uint" in (left[0], right[0]):
            return ("uint", f"`{ast.unparse(node)}` of unsigned operands (uint64)")
        if left[0] in _NARROW_UINT_KINDS and right[0] in _NARROW_UINT_KINDS:
            # Numba promotes arithmetic on narrow unsigned operands to uint64
            # (uint8 + uint8 -> uint64), so the result meets the float rule
            # when unified with int64 (an earlier fix P2-2)
            return ("uint", f"`{ast.unparse(node)}` of narrow unsigned operands (promoted to uint64)")
        # Numba's integer binop cases pick the signed 64-bit result for
        # mixed signedness (unlike unification).
        return ("int", f"`{ast.unparse(node)}` of integer operands")

    def _call(self, node: ast.Call) -> Kind:
        name = _dotted(node.func)
        if name in ("min", "max"):
            return _unify([self.kind_of(a) for a in node.args], f"{name}()")
        if name in _INT_CASTS:
            if name == "round" and len(node.args) > 1:
                return ("float", "round() with ndigits")
            return ("int", f"{name}() cast")
        if name in _UINT_CASTS:
            return ("uint", f"{name}() cast")
        if name in _NARROW_UINT_CASTS:
            return (_NARROW_UINT_CASTS[name], f"{name}() cast ({_NARROW_UINT_CASTS[name]}: unifies with int64 to int64, with uint64 to uint64)")
        if name in _FLOAT_CALLS:
            return ("float", f"{name}() returns float64")
        if name in ("np.round", "np.around") and node.args:
            # Numba's scalar overload (numba/np/old_arraymath.py, impl_np_round,
            # integer branch: ``if decimals == 0: return a / else: return
            # int(round_ndigits(a, decimals))``) types BOTH returns and unifies
            # them, with or without an explicit `decimals`, so the result kind
            # is unify(kind(a), int64): signed -> int64, narrow unsigned
            # (uint8/16/32) -> int64, uint64 -> float64 (an earlier fix P2-R3-U; probed
            # on numba 0.61.2, evidence numba-round-return-types.json); a float
            # argument is float64; an UNTYPED argument is unknown -- this checker
            # reports PROVABLE float-ness only, it never assumes (ruling P3-R4).
            k = self.kind_of(node.args[0])
            if k[0] == "int" or k[0] in _NARROW_UINT_KINDS:
                return ("int", f"{name}() of a {k[0]} value is int64 (Numba unifies the integer branch with int64; {k[1]})")
            if k[0] == "uint":
                return ("float", f"{name}() of a uint64 value is float64 (Numba unifies uint64 with the int64 branch; {k[1]})")
            if k[0] == "float":
                return ("float", f"{name}() of a float64 value ({k[1]})")
            return _UNKNOWN
        if name == "abs" and node.args:
            return self.kind_of(node.args[0])
        if name and name.endswith(".item"):
            return _UNKNOWN
        return _UNKNOWN

    # -- statements --------------------------------------------------------

    def bind(self, target: ast.AST, kind: Kind, stmt: ast.AST,
             value: Optional[ast.AST] = None) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = kind
            self.defs[target.id] = ast.unparse(stmt)
            self.arrays.pop(target.id, None)
            cv = self.const_value(value) if value is not None else None
            if cv is not None:
                self._set_const(target.id, cv, stmt)
            else:
                self.consts.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # ``j, k = (0.5, 0)`` / ``j, k = k, j``: every RHS element is typed
            # BEFORE any target is rebound (the RHS reads the old bindings); a
            # homogeneous numeric LIST literal has one common element type
            # (``[0.5, 1]`` is a float64 list, so both targets are float);
            # anything else leaves the targets unknown (an earlier fix P2-1 / an earlier fix P2-1)
            typed = self._unpack_kinds(target, value)
            if typed is None:
                for elt in target.elts:
                    self.bind(elt, _UNKNOWN, stmt)
            else:
                self._bind_unpacked(target, typed, stmt)

    def _unpack_kinds(self, target: ast.AST, value: ast.AST):
        """Type every element of an unpacking RHS before anything is bound.

        Returns a tree mirroring *target*: ``(kind, const)`` for a name,
        a list for a nested tuple/list target, or None when the RHS is not
        a literal of matching length (the targets become unknown).
        """
        if not (isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(target.elts)
                and not any(isinstance(e, ast.Starred) for e in target.elts)):
            return None
        out = []
        for elt, val in zip(target.elts, value.elts):
            if isinstance(elt, (ast.Tuple, ast.List)):
                out.append(self._unpack_kinds(elt, val))
            else:
                out.append((self.kind_of(val), self.const_value(val)))
        if isinstance(value, ast.List):
            leaves = [x for x in out if isinstance(x, tuple)]
            if len(leaves) != len(out):
                return None  # a list literal holding nested sequences: not modelled
            common = _unify([k for k, _ in leaves], "list literal element") if leaves else _UNKNOWN
            out = [(common, None if common[0] == "float" else cv) for _, cv in leaves]
        return out

    def _bind_unpacked(self, target: ast.AST, typed, stmt: ast.AST) -> None:
        for elt, t in zip(target.elts, typed):
            if isinstance(elt, (ast.Tuple, ast.List)):
                if t is None:
                    for sub in elt.elts:
                        self.bind(sub, _UNKNOWN, stmt)
                else:
                    self._bind_unpacked(elt, t, stmt)
            elif isinstance(elt, ast.Name):
                k, cv = t
                self.env[elt.id] = k
                self.defs[elt.id] = ast.unparse(stmt)
                self.arrays.pop(elt.id, None)
                if cv is not None:
                    self._set_const(elt.id, cv, stmt)
                else:
                    self.consts.pop(elt.id, None)
            else:
                self.bind(elt, t[0] if isinstance(t, tuple) else _UNKNOWN, stmt)

    def _dtype_kind(self, dt: str, what: str) -> Kind:
        dt = dt.replace("np.", "").replace("numpy.", "").replace("numba.", "")
        if dt in ("uint64", "uintp", "uint"):
            return ("uint", f"element of {dt} {what}")
        if dt in _NARROW_UINT_KINDS:
            return (dt, f"element of {dt} {what}")
        if dt.startswith("int") or dt in ("intp", "intc", "long"):
            return ("int", f"element of {dt} {what}")
        if dt.startswith("float") or dt == "double":
            return ("float", f"element of {dt} {what}")
        if dt.startswith("bool"):
            return ("bool", f"element of {dt} {what}")
        return _UNKNOWN

    def _array_ctor_elem(self, value: ast.Call, ctor: str) -> Kind:
        """Element kind of ``np.zeros/ones/empty/full/arange(...)``."""
        for kw in value.keywords:
            if kw.arg == "dtype":
                return self._dtype_kind(_dotted(kw.value) or "", "array")
        # positional dtype: zeros(shape, dtype) / full(shape, fill, dtype) / arange(..., dtype)
        pos_dtype = {"np.zeros": 1, "np.ones": 1, "np.empty": 1, "np.full": 2, "np.arange": 3}.get(ctor)
        if pos_dtype is not None and len(value.args) > pos_dtype:
            dt = _dotted(value.args[pos_dtype])
            if dt:
                return self._dtype_kind(dt, "array")
        if ctor == "np.full":
            # NumPy infers the dtype from the fill value
            return self.kind_of(value.args[1]) if len(value.args) > 1 else _UNKNOWN
        if ctor == "np.arange":
            kinds = [self.kind_of(a) for a in value.args]
            if any(k[0] == "float" for k in kinds):
                return ("float", "element of np.arange with a float argument")
            if kinds and all(k[0] in ("int", "uint", "bool") for k in kinds):
                return ("int", "element of np.arange with integer arguments")
            return _UNKNOWN
        # zeros / ones / empty without dtype: NumPy's default is float64
        return ("float", f"element of {ctor} array (NumPy default dtype float64)")

    def assign(self, node: ast.AST, value: ast.AST, targets: List[ast.AST]) -> None:
        ctor = _dotted(value.func) if isinstance(value, ast.Call) else None
        if ctor in _ARRAY_CTORS:
            elem = self._array_ctor_elem(value, ctor)
            for t in targets:
                if isinstance(t, ast.Name):
                    self.arrays[t.id] = elem
                    self.env.pop(t.id, None)
            return
        # ``B = A`` for an array A: the alias keeps A's element kind
        if isinstance(value, ast.Name) and value.id in self.arrays:
            for t in targets:
                if isinstance(t, ast.Name):
                    self.arrays[t.id] = self.arrays[value.id]
                    self.env.pop(t.id, None)
            return
        kind = self.kind_of(value)
        for t in targets:
            self.bind(t, kind, node, value)

    # -- control-flow joins ----------------------------------------------

    def snapshot(self):
        return dict(self.env), dict(self.arrays), dict(self.defs), dict(self.consts)

    def restore(self, snap) -> None:
        self.env, self.arrays, self.defs = dict(snap[0]), dict(snap[1]), dict(snap[2])
        self.consts = dict(snap[3]) if len(snap) > 3 else {}

    def join(self, snaps, what: str) -> None:
        """Merge the environments of alternative paths into ``self``.

        A name's kind is float when any path makes it float (Numba unifies the
        incoming types of the merge, and float64 absorbs every integer),
        unknown when any path leaves it unknown, otherwise the unification of
        the paths' integer kinds; a name defined on only some paths takes the
        join of the paths that define it.  Arrays keep their element kind only
        when every path agrees.
        """
        envs = [s[0] for s in snaps]
        names = set().union(*(e.keys() for e in envs))
        env: Dict[str, Kind] = {}
        defs: Dict[str, str] = {}
        for n in names:
            kinds = [e[n] for e in envs if n in e]
            floats = [k for k in kinds if k[0] == "float"]
            if floats:
                env[n] = ("float", f"{what}: {floats[0][1]}")
            elif any(k[0] is None for k in kinds):
                env[n] = _UNKNOWN
            elif len({k[0] for k in kinds}) == 1:
                env[n] = kinds[0]
            else:
                env[n] = _unify(kinds, what)   # integral kinds of mixed width/signedness
            for s in snaps:
                if n in s[2]:
                    defs[n] = s[2][n]
        arrays: Dict[str, Kind] = {}
        for n in set().union(*(s[1].keys() for s in snaps)):
            elems = [s[1].get(n) for s in snaps]
            if all(e is not None for e in elems) and len({e[0] for e in elems}) == 1:
                arrays[n] = elems[0]
            else:
                arrays[n] = _UNKNOWN
        # A constant alias survives a join only when every path carries the
        # SAME definition (same value from the same statement, i.e. bound
        # before the paths diverged and untouched on all of them).  Equal
        # values assigned by DIFFERENT statements on different paths are an
        # SSA phi that Numba does not fold (`if flag: a = 0.5 else: a = 0.5;
        # a and 1` is typed float64), so they are typed-but-unknown-value here
        # (an earlier fix P2-R1); the type unification above is unaffected.
        consts: Dict[str, tuple] = {}
        common_names = set(snaps[0][3]).intersection(*(set(sn[3]) for sn in snaps[1:])) if snaps else set()
        for n in sorted(common_names):
            entries = {(repr(s[3][n][0]), s[3][n][1]) for s in snaps}
            if len(entries) == 1:
                consts[n] = snaps[0][3][n]
        self.env, self.arrays, self.defs, self.consts = env, arrays, defs, consts


def _bind_targets(t: ast.AST, out: set) -> None:
    if isinstance(t, ast.Name):
        out.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _bind_targets(e, out)
    elif isinstance(t, ast.Starred):
        _bind_targets(t.value, out)


def _scan_scope(stmts: Iterable[ast.AST]):
    """One lexical scope: ``(bound, nonlocals, globals_, nested_defs)``.

    ``bound`` are the names this scope's own statements bind (Assign /
    AugAssign / AnnAssign targets, ``for`` targets incl. nested loops and
    ``else:`` clauses, walrus targets, ``with ... as``, ``except ... as``,
    ``del``, ``import``, and the names of nested ``def`` / ``class``
    statements); ``nonlocals`` / ``globals_`` are the names its own
    ``nonlocal`` / ``global`` statements declare -- kept APART, because a
    ``nonlocal`` targets the nearest enclosing function scope that binds the
    name while a ``global`` targets the MODULE namespace and can never rebind
    any function's local (an earlier fix P2-R5-GLOBAL); ``nested_defs`` are the
    function definitions found directly in this scope.  Bodies of nested
    functions / lambdas / classes and comprehension iteration targets belong
    to OTHER scopes and are not entered (a walrus inside a comprehension
    binds in this scope and is).
    """
    bound: set = set()
    nonlocals: set = set()
    globals_: set = set()
    nested: list = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            nested.append(node)
            for d in node.decorator_list:
                visit(d)
            return
        if isinstance(node, ast.ClassDef):
            bound.add(node.name)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Nonlocal):
            nonlocals.update(node.names)
            return
        if isinstance(node, ast.Global):
            globals_.update(node.names)
            return
        if isinstance(node, ast.comprehension):
            visit(node.iter)                      # the target is the comprehension's own local
            for e in node.ifs:
                visit(e)
            return
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _bind_targets(t, bound)
            visit(node.value)
            return
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            _bind_targets(node.target, bound)
            if node.value is not None:
                visit(node.value)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            _bind_targets(node.target, bound)
            visit(node.iter)
            for stmt in node.body + node.orelse:
                visit(stmt)
            return
        if isinstance(node, ast.NamedExpr):
            _bind_targets(node.target, bound)
            visit(node.value)
            return
        if isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                _bind_targets(node.optional_vars, bound)
            visit(node.context_expr)
            return
        if isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)              # ``except E as a`` binds (and unbinds) a
            if node.type is not None:
                visit(node.type)
            for stmt in node.body:
                visit(stmt)
            return
        if isinstance(node, ast.Delete):
            for t in node.targets:
                _bind_targets(t, bound)           # ``del a`` ends a's binding: not a constant afterwards
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in stmts:
        visit(stmt)
    return bound, nonlocals, globals_, nested


def _fn_params(fn) -> set:
    params = set()
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        params.add(a.arg)
    for a in (fn.args.vararg, fn.args.kwarg):
        if a is not None:
            params.add(a.arg)
    return params


def _rebinds_enclosing(fn) -> set:
    """Names *fn* rebinds in its ENCLOSING FUNCTION scope through ``nonlocal``.

    A name *fn* itself declares ``nonlocal`` and assigns is such a rebind.  A
    deeper function's ``nonlocal`` passes through *fn* only when the name does
    not resolve to one of *fn*'s own locals (its parameters or the names its
    own scope binds without declaring them): a function that binds the name
    locally is where that ``nonlocal`` stops (an earlier fix P2-R4-SCOPE).  ``global``
    declarations never appear here: they target the module namespace, not any
    function's local (an earlier fix P2-R5-GLOBAL; see :func:`_rebinds_module`).
    """
    bound, nonlocals, globals_, nested = _scan_scope(fn.body)
    local = (bound | _fn_params(fn)) - nonlocals - globals_
    out = {n for n in nonlocals if n in bound}     # declared nonlocal here and assigned here
    for g in nested:
        for n in _rebinds_enclosing(g):
            if n not in local:                     # resolves past fn
                out.add(n)
    return out


def _rebinds_module(fn) -> set:
    """Names *fn* (or any function nested in it) rebinds in the MODULE namespace
    through a ``global`` declaration plus an assignment.  Intermediate
    functions never stop a ``global``: it always targets the module."""
    bound, nonlocals, globals_, nested = _scan_scope(fn.body)
    out = {n for n in globals_ if n in bound}
    for g in nested:
        out |= _rebinds_module(g)
    return out


def _assigned_names(body: Iterable[ast.AST]) -> set:
    """Every name the ENCLOSING lexical scope binds anywhere inside *body*.

    The scope's own bindings (see :func:`_scan_scope`: assignment-like
    constructs, nested loops and their ``else:`` clauses, walrus, ``with /
    except ... as``, ``del``) plus the names that nested functions rebind in
    this scope through ``nonlocal`` (see :func:`_rebinds_enclosing`).  Bodies
    of nested functions / lambdas / classes and comprehension iteration
    targets bind their own locals and are never attributed to this scope
    (an earlier fix P2-R3-SCOPE / P2-R4-SCOPE); a nested ``global`` rebinds the module
    namespace only and never a same-named local of this scope
    (P2-R5-GLOBAL; see :func:`_module_rebinds`).
    """
    bound, nonlocals, globals_, nested = _scan_scope(body)
    names = set(bound)
    for g in nested:
        names |= _rebinds_enclosing(g)
    return names


def _module_rebinds(body: Iterable[ast.AST]) -> set:
    """Module-level names that *body* rebinds through ``global`` declarations
    (its own ``global x; x = ...`` statements or those of functions nested in it)."""
    bound, nonlocals, globals_, nested = _scan_scope(body)
    out = {n for n in globals_ if n in bound}
    for g in nested:
        out |= _rebinds_module(g)
    return out


def _module_env(tree: ast.Module) -> Dict[str, Kind]:
    typer = _KernelTyper({})
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets:
            typer.assign(node, node.value, node.targets)
            # module constants are plain scalars: strip the def text so a
            # kernel finding does not quote the module line twice
    return {k: (v[0], f"module constant `{k}` ({v[1]})") for k, v in typer.env.items()}


def _index_elements(node: ast.Subscript) -> Iterator[ast.AST]:
    sl = node.slice
    elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
    for e in elts:
        if isinstance(e, ast.Slice):
            for part in (e.lower, e.upper, e.step):
                if part is not None:
                    yield part
        elif isinstance(e, ast.Constant) and e.value in (None, Ellipsis):
            continue
        else:
            yield e


def _check_kernel(fn: ast.FunctionDef, module_env: Dict[str, Kind],
                  filename: str) -> List[IndexFinding]:
    typer = _KernelTyper(module_env)
    findings: List[IndexFinding] = []

    def check_subscripts(expr: ast.AST) -> None:
        for sub in ast.walk(expr):
            if not isinstance(sub, ast.Subscript):
                continue
            access = {ast.Load: "load", ast.Store: "store", ast.Del: "del"}[type(sub.ctx)]
            for elt in _index_elements(sub):
                kind = typer.kind_of(elt)
                if kind[0] == "float":
                    findings.append(IndexFinding(
                        filename, fn.name, elt.lineno, elt.col_offset,
                        ast.unparse(elt), access, "float64", kind[1]))

    def visit(stmts: Iterable[ast.AST], prange_depth: int) -> None:
        for node in stmts:
            if isinstance(node, ast.For):
                if _is_prange_call(node.iter):
                    if prange_depth == 0:
                        typer.bind(node.target, ("uint", _PRANGE_REASON), node)
                    else:
                        typer.bind(node.target, ("int", "nested prange (serial range in the gufunc)"), node)
                    check_subscripts(node.iter)
                    typer.invalidate_consts_assigned_in(node.body)
                    before = typer.snapshot()
                    visit(node.body, prange_depth + 1)
                    visit(node.orelse, prange_depth)
                    typer.join([before, typer.snapshot()], "loop join")
                    continue
                if _is_range_call(node.iter):
                    typer.bind(node.target, ("int", "range index"), node)
                elif isinstance(node.iter, (ast.Tuple, ast.List)) and node.iter.elts:
                    # a literal iterable types its loop variable from its elements
                    typer.bind(node.target,
                               _unify([typer.kind_of(e) for e in node.iter.elts],
                                      "literal iterable element"), node)
                elif isinstance(node.iter, ast.Name) and node.iter.id in typer.arrays:
                    # iterating a known array yields its element kind
                    ek = typer.arrays[node.iter.id]
                    typer.bind(node.target, (ek[0], f"element of array `{node.iter.id}`: {ek[1]}"), node)
                elif isinstance(node.iter, ast.Call) and _dotted(node.iter.func) in _ARRAY_CTORS:
                    ek = typer._array_ctor_elem(node.iter, _dotted(node.iter.func))
                    typer.bind(node.target, (ek[0], f"element of {_dotted(node.iter.func)}(...): {ek[1]}"), node)
                else:
                    typer.bind(node.target, _UNKNOWN, node)
                check_subscripts(node.iter)
                # zero-or-more trips: the environment after the loop is the
                # join of "body never ran" and "body ran once"; a constant the
                # body reassigns is unknown-value already on the first pass
                # (the backedge makes the entry value non-invariant)
                typer.invalidate_consts_assigned_in(node.body)
                before = typer.snapshot()
                visit(node.body, prange_depth)
                visit(node.orelse, prange_depth)
                after = typer.snapshot()
                typer.join([before, after], "loop join")
            elif isinstance(node, ast.Assign):
                check_subscripts(node.value)
                for t in node.targets:
                    check_subscripts(t)
                typer.assign(node, node.value, node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                check_subscripts(node.value)
                check_subscripts(node.target)
                typer.assign(node, node.value, [node.target])
            elif isinstance(node, ast.AugAssign):
                check_subscripts(node.value)
                check_subscripts(node.target)
                if isinstance(node.target, ast.Name):
                    combined = ast.BinOp(left=node.target, op=node.op, right=node.value)
                    ast.copy_location(combined, node)
                    typer.bind(node.target, typer.kind_of(combined), node)
            elif isinstance(node, ast.If):
                check_subscripts(node.test)
                before = typer.snapshot()
                visit(node.body, prange_depth)
                taken = typer.snapshot()
                typer.restore(before)
                visit(node.orelse, prange_depth)
                not_taken = typer.snapshot()
                typer.join([taken, not_taken], "branch join")
            elif isinstance(node, ast.While):
                check_subscripts(node.test)
                typer.invalidate_consts_assigned_in(node.body)
                before = typer.snapshot()
                visit(node.body, prange_depth)
                visit(node.orelse, prange_depth)
                after = typer.snapshot()
                typer.join([before, after], "loop join")
            elif isinstance(node, ast.With):
                visit(node.body, prange_depth)
            elif isinstance(node, (ast.Expr, ast.Return)) and node.value is not None:
                check_subscripts(node.value)
            elif isinstance(node, ast.Delete):
                for t in node.targets:
                    check_subscripts(t)

    visit(fn.body, 0)
    return findings


def check_lowered_source(source: str, filename: str = "<lowered>") -> List[IndexFinding]:
    """Return every float-typed subscript index of the njit kernels in *source*."""
    tree = ast.parse(source, filename)
    module_env = _module_env(tree)
    findings: List[IndexFinding] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _is_njit_kernel(node):
            findings.extend(_check_kernel(node, module_env, filename))
    return findings


def check_lowered_file(path: str) -> List[IndexFinding]:
    with open(path, encoding="utf-8") as fh:
        return check_lowered_source(fh.read(), path)


def iter_lowered_programs(root: str, basename: str = "executed-lowered.py"
                          ) -> Iterator[Tuple[str, List[IndexFinding]]]:
    """Yield ``(path, findings)`` for every *basename* under *root* (sorted)."""
    if os.path.isfile(root):
        yield root, check_lowered_file(root)
        return
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        if basename in files:
            paths.append(os.path.join(dirpath, basename))
    for path in sorted(paths):
        yield path, check_lowered_file(path)


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__.strip().splitlines()[0])
        print("usage: python -m solar.einsum_lib.lowered_index_check <file|dir>...")
        return 2
    total = 0
    programs = 0
    for arg in args:
        for path, findings in iter_lowered_programs(arg):
            programs += 1
            for f in findings:
                print(f.format())
            total += len(findings)
            print(f"{path}: {len(findings)} float-typed index expression(s)")
    print(f"{programs} program(s), {total} float-typed index expression(s)")
    return 1 if total else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
