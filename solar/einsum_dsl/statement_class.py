# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-statement work classification for einsum_lib kernel bodies.

The unit contract (release decision ``mac_unit_convention``, issue #59):

* **MAC** = one multiply-accumulate: a statement that sums a product into its
  target (``T += a * b``, ``T -= a * b``, ``T = T + a * b``) where the
  product has at least two *value* factors.  Each loop trip of such a
  statement executes one MAC; every value factor beyond the second is one
  standalone multiply (``other``) per trip.
* **other** = everything else executed per trip: plain additions and
  subtractions, scales (``T = acc * s``), bias adds, masks, copies,
  initialisations, elementwise products that are not accumulated.

THE STRUCTURAL RULE (manager ruling, an earlier fix round 4).  What a factor IS is
decided by its value PROVENANCE, never by its syntax.  Provenance has three
classes and is DEFAULT-DENY, like the merged an earlier change pricing contract:

* ``STATIC`` -- a closed whitelist: literals; ``Dim(...)``; ``.shape[...]``,
  ``.size()``, ``.dim()``, ``.numel()`` or ``len()`` of a tensor NAME; loop
  indices and names bound to them; arithmetic / comparisons / explicit
  booleans of STATIC operands; a pure-static builtin (``int``, ``float``,
  ``abs``, ``min``, ``max``, ``round``, ``floor``, ``len``, ``range``,
  ``bool``) called through its UNSHADOWED name on STATIC arguments only.  A
  STATIC factor is a SCALE, never a value, on either side of a product.
  A LITERAL CONSTANT -- a module-level name bound exactly once,
  unconditionally, to an int or float literal (the LITERAL-CONSTANT
  PRODUCTION below, issue #61) -- is STATIC.
* ``VALUE`` -- a tensor element (a tensor NAME subscripted by a loop index),
  a runtime scalar (a scalar parameter, an accumulator, a SINGLETON-SCALE
  READ of a declared input -- the production below, issue #61), and any cast,
  arithmetic, DSL math function (``exp``, ``sqrt``, ``tanh``, ...) or
  operand selection (``a or b``, ``a if c else b``) of those.  A VALUE
  carries its origin: a *tensor* value is a value factor on either side of a
  product; a runtime *scalar* is a value factor as a multiplier (``acc +=
  X[m, k] * s`` is a fused multiply-add) but only a scale as a divisor
  (``/ s``, release decision divisor-class-3).
* ``UNPROVEN`` -- everything else: a subscript on a name that is not a
  tensor, a tensor subscripted without any loop index, an unknown or
  shadowed callable, a lambda, a method call on a tensor-derived receiver,
  a module-qualified call, an unknown name, an unsupported binding form, a
  name bound to differing classes anywhere in the kernel, a name used before
  it is bound.  UNPROVEN DOMINATES EVERY JOIN (binary operators, call
  arguments, comparisons, conditional arms, boolean operators, indices):
  uncertainty propagates independently of the value class.  An unproven
  factor is counted as a value and the statement carries the
  ``class_evidence_unproven`` flag: the kernel's classes are reported but
  never as complete evidence (incomplete analysis and pricing).

Bindings are FLOW-ORDERED: every binding form (plain, annotated, chained and
tuple targets, augmented assignment, walrus, ``for`` / ``with`` targets,
``def`` / ``class`` / ``import``) re-derives the provenance of all its target
names, and a name's provenance at a statement is its value AT THAT
STATEMENT.  Unpacking (tuple / list / starred targets) is not a proven
binding form: its names are unproven.  A loop variable rebound inside its body is no longer a loop
index.  A name bound to differing classes anywhere in the kernel is
UNPROVEN at every use (``denom = W[m, k]; denom = 2.0`` is refused, not
guessed); an accumulator (``acc += ...``) is a runtime scalar.

Masks are not values: a comparison, a boolean of comparisons, ``not`` or an
explicit ``bool()`` only gates a term -- but ``or`` / ``and`` RETURN AN
OPERAND and are classified by the join of their operands
(bool-value-11).  An accumulated right-hand side that is an inline
conditional is classified per arm (ifexp-class-6): equal arms give that
class exactly, differing arms give the per-class MAXIMUM over the arms with
the ``branch_class_uncertain`` flag -- the same rule the parser applies to a
statement-level ``if``.

THE SUPPORTED GRAMMAR (manager ruling, an earlier fix round 5).  The per-statement
classifier accepts only this grammar; anything outside it makes the WHOLE
KERNEL ``class_evidence_unproven`` (incomplete analysis and pricing; the
legacy whole-kernel split is never a fallback for a DSL kernel that declares
classes).  The 4 KB candidates with a lowered DSL are written in it (0 of 26
kernels refuse); the Lite programs are NOT all in it (the round-6 inventory:
91 of 360 kernels refused -- one-element global-scale reads such as ``GM[0]``
and module constants outside the block-index production; the issue #61
inventory on the round-7 head with the LITERAL-CONSTANT and SINGLETON-SCALE-READ
productions below: __LITE_REFUSED__ of 360 -- expression- and call-bound
constants, literal-initialised running-max rebinding, ``ST[0]`` state
reads); anything exotic refuses, never guesses.

1. Kernel bodies: ``for <name> in Dim(...)`` loops, plain ``Assign`` /
   ``AugAssign`` (one target: a Name or a tensor Subscript store), ``if`` /
   ``IfExp`` whose TEST is proven STATIC or a comparison / boolean of proven
   operands (otherwise UNPROVEN), ``return``.
2. Expressions: Names, Constants, tensor Subscripts indexed by loop Names /
   STATIC arithmetic, ``.shape[<literal>]`` / ``.size()`` / ``len()`` of a
   tensor Name, arithmetic ``BinOp`` / ``UnaryOp``, comparisons and booleans,
   calls ONLY to the whitelist -- the pure-static builtins (``int float abs
   min max round floor len range bool``), the DSL execution namespace's math
   functions (``exp log sqrt cos sin tanh log2 erf fp32``) and the DSL's
   ``Dim`` / ``Tensor`` -- with the callee proven unshadowed in BOTH the
   kernel and the MODULE namespace: one scope-aware collector over the WHOLE
   module tree (every suite: if / for / while / try / with / match; Assign,
   AnnAssign, AugAssign, walrus, for / with targets, def / class, import /
   star import, ``global`` anywhere, ``del``, and the DEFINITION-TIME
   expressions of a def / class -- decorators, argument defaults, every
   annotation, the return annotation, class bases and keywords -- which
   execute in the enclosing scope, so ``def f(arg=(abs := g))`` shadows
   ``abs`` and ``def f(arg=(_BLK := 2))`` rebinds ``_BLK``; the def / class
   BODY and a lambda BODY are nested scopes -- ``def f(arg=(lambda: (abs :=
   1)))`` shadows nothing, ``lambda x=(abs := g): x`` does through its
   default, and a comprehension's walrus binds in the containing scope,
   PEP 572) gives every module name its
   binding count and kinds, and any binding of a whitelisted name refuses the
   kernel; only the exact DSL import (``from solar.einsum_lib import ...``,
   or its relative form) is the vocabulary, never a module-name suffix.  A
   bare module-level name is UNPROVEN unless it is a LITERAL CONSTANT (the
   LITERAL-CONSTANT PRODUCTION below); the BLOCK-INDEX PRODUCTION below is
   its stricter, original form.  Every expression SLOT has a
   required role: a ``Tensor(...)`` argument and a ``Dim(..., size=...)``
   size must be STATIC (literal, ``Dim``, ``.shape[<literal>]``, STATIC
   arithmetic, the block-index production, and -- in the SIZE slot only --
   the DSL's own size inputs: a kernel parameter annotated ``int`` whose
   annotation RESOLVES to the builtin (a module that binds the name ``int``,
   or star-imports, proves no annotation) and that is NEVER REBOUND anywhere
   in the body by any binding form -- plain / annotated / augmented
   assignment, walrus, ``for`` / ``with`` target, ``del``, def / import --
   before or after the use (the extent counter resolves it from the call
   site), and a local bound to STATIC size arithmetic such as ``C =
   A.shape[1] // block_len``, and an INT literal constant; a float parameter,
   a tensor element, a FLOAT literal constant or any other module name is not
   a size), an ``if`` / ``IfExp`` TEST must
   be STATIC at every nesting depth, a ``return`` expression is classified
   like any value, and a tensor Subscript index is STATIC or a declared
   gather (below); a VALUE in any STATIC slot refuses the kernel.
   EVERY STATEMENT OF THE BODY is validated by slot after the fixpoint --
   inside or outside loops, used or unused (an earlier fix round 7): a loop bound
   and an ``if`` / ``while`` test STATIC, an assigned value and a ``return``
   expression not UNPROVEN, a subscript store through the index slot, an
   expression statement a literal; one violation anywhere refuses the
   kernel.  An outer ``unused = _BAD`` or ``if _FLAG:`` refuses exactly as
   an in-loop one.  A subscript / attribute STORE binds nothing: it mutates
   an object, its receiver and index expressions are reads (``Y[rows - 2 +
   m] += ...`` leaves the ``int`` parameter ``rows`` a size source); only a
   Name target and the Names of unpacking bind.  The subscript RECEIVER is a tensor OBJECT only (a
   declared input, a ``Tensor(...)`` allocation or a name bound to one): a
   tensor-origin VALUE (``a = X[m, k]``) is a number and ``a[...]`` refuses.
   A valued ``AnnAssign`` IS the ``Assign`` of its value (a local annotation
   is never evaluated): it is normalised in place at the front, so
   accumulator detection, binding, flow, counting and classification have
   one path (``acc: float = acc + X[m, k] * W[m, k]`` is the MAC the plain
   form is).
3. Scalar Name bindings are resolved by a REAL FIXPOINT over the loop nest:
   the flow-ordered binding analysis is iterated until the per-statement
   snapshots stop changing, bounded by N = number of bindings + 2; a run that
   has not converged, or a Name that joins STATIC with VALUE across
   iterations, is UNPROVEN.  Parameters are bindings too (a rebound scalar
   parameter is UNPROVEN).  The convergence key is the frozen VALUE image of
   the WHOLE snapshot table (:func:`freeze_state`): bindings, index names,
   the SIZE view and every component a later change appends to a snapshot
   take part, nothing is hand-picked and nothing is compared by identity (a
   float carried into a Tensor size through a 32-name chain is found,
   size-fixpoint-24); a component that is not a dict / set / sequence /
   scalar must be encoded structurally before it enters a snapshot (the
   ``repr`` fallback is identity for AST objects, not value).  Statements
   are validated on the CONVERGED
   snapshots, so a loop-carried name the fixpoint proves is never refused
   by an early pass.
   BLOCK-INDEX PRODUCTION (manager decision, an earlier fix round 5; the NVFP4
   block-scale decode form of the lowered Lite candidates 195, 204, 205,
   206, 207 and 209): ONLY in the index slot of a tensor Subscript and the
   size slot of ``Tensor(...)`` / ``Dim(...)``, an expression of the exact
   AST shape ``BinOp(FloorDiv, left=<loop index or STATIC arithmetic>,
   right=Name(<module-level name bound exactly once, unconditionally, to an
   int literal, never rebound, never deleted>))``, i.e. ``k // _BLK`` with
   ``_BLK = 16``, is STATIC; the same expression in any other slot (a factor,
   a plain assignment ``q = k // _BLK``) and any Name bound from it is
   UNPROVEN -- the role is passed down by the classifier, never exported.  It
   is the shape of the block amax (``Y[.., p // _BLK] = max(Y[.., p // _BLK],
   abs(X[.., p]))``), the quantisation (``sx = X[..] * EN[.., p // _BLK]``),
   the scaled contraction (``acc += XQ[.., k] * XS[.., k // _BLK] * (WV[n, k]
   * WS[n, k // _BLK])``, row 205's ``ws = wv * WS[0, n, k // _BLK]``) and the
   ``NB=X.shape[d] // _BLK`` allocation.  The dequantisation arithmetic is a
   VALUE, the block-scale tensor is read through this index (a scale, not a
   payload, for the compute-dtype binding) and the contraction operands keep
   their declared payload dtype.  Since the LITERAL-CONSTANT PRODUCTION
   (issue #61, below) admits the same once-bound int literal names in every
   STATIC role, ``k // _BLK`` is STATIC wherever both operands are; this
   production keeps its stricter form (an int literal divisor, the index and
   size slots) as the documented shape of the block-scale decode.  A constant
   bound to an expression (``_G = _NH // _KV``) or bound twice stays UNPROVEN.
   COMPUTE-DTYPE BINDING (DSL-COMPUTE-DTYPE-BINDING-1, rounds 2-3): a MAC
   statement's operands are recorded per OCCURRENCE -- one entry per tensor
   element factor (payload or block scale, by the pairwise block-scale rule
   over the whole product) and one per explicit ``fp32(...)`` conversion
   factor -- never as a deduplicated name list, so ``fp32(X) * fp32(W)`` and
   ``X[m, k] * X[m, k]`` have two factors.  A local alias stores the operand
   record FROZEN AT ITS DEFINING ASSIGNMENT (resolved against the bindings in
   force there, a snapshot component -- the fifth, after issue #61's float-derived set): ``a = Q[m, k]; b = a; a =
   W[m, k]; acc += X[m, k] * b`` contracts X with Q, never with W, and a
   tensor-object alias ``Q = W`` names W from then on; a name with no single
   reaching definition is unproven.  The converter emits the receipt
   (``loop_mac_operands``: positions and declared dtypes) beside an
   independent occurrence channel (``loop_mac_reads``: name, position, index
   pattern, role); the analyzer reconciles the two entry by entry, admits only
   the supported conversions, validates a present receipt even at zero MACs,
   and proves a compute dtype only when every payload occurrence and
   conversion agrees.
   GATHER-INDEX PRODUCTION (manager ruling, an earlier fix round 6): a tensor
   Subscript index that is a proven VALUE -- a tensor element (``IDX[n, k]``,
   ``F[n]``), a VALUE-derived scalar (``int(W[m, k]) - 1``) or a local bound
   to such a value (``e = IDX[n, k]``, ``ti = ROW[p]``) -- is admitted ONLY in
   the index slot of a tensor Subscript, as a declared GATHER (a read or a
   store): it keeps the reviewed gather traffic / accounting rule unchanged,
   its VALUE provenance is never promoted to STATIC and never exported to
   another slot, the statement's class is decided by the other operands as
   today, and the kernel records ``gather_index_statements`` (executed trips
   of statements that address through a gather) so the row exposes that it
   relies on the gather rule.  Lite rows 58 (store through ``F[n]``), 108
   (reads through ``ti`` / ``tj``) and 119 (read through ``e = IDX[n, k]``)
   are its production forms.  A VALUE as a ``Tensor`` size, a ``Dim`` size, an
   ``if`` / ``IfExp`` test, a ``.shape`` index or a loop bound, and a VALUE
   index on a non-tensor receiver, refuse; an UNPROVEN index stays refused.
   A Subscript whose indices are EACH a STATIC, a loop index or a proven
   VALUE is admitted by these rules whether or not a loop variable is
   mentioned (an earlier fix round 7, Lite backlog PLAN 2b): a pure gather through
   VALUE-derived locals -- ``acc = bias[row]`` with ``row = off + n * hd +
   d`` (170 / 171 ``qkv_part_proj``), ``W[t, ge]`` with ``t = SLOT[n, 0]``,
   ``ge = SLOT[n, 1]`` (229 ``slot_routing_weight``) -- is a declared
   gather; a literal-only index set (``W[0, 0]``) stays UNPROVEN unless it is
   the SINGLETON-SCALE-READ PRODUCTION below (``GM[0]`` on a declared
   one-element input).
   Gather / scatter addressing is an accounted, reviewed feature of the DSL
   (sparse / MoE / attention-index programs); what the grammar forbids is
   guessing, and a declared gather is not a guess.  The RECEIPT is per
   statement (an earlier fix round 7): each record carries its own gather flag,
   across the branches of an ``if`` the gather-bearing count is the
   per-branch maximum carried by that branch's records (never OR-ed onto
   unrelated records), and a statement-``if`` TEST that reads through a
   gather is one gather EVALUATION per trip of the ``if``
   (``loop_gather_controls``), counted into ``gather_index_statements``
   without touching the statement counts.  The recompute billing convention
   (a per-element recomputed reduction billed as SM work) changes how work
   is billed, never whether the classes are proven: ``class_evidence_uncertain``
   and the grammar violations stay on the node and reach layer, total and
   perf status (recompute-refusal-30).
   LITERAL-CONSTANT PRODUCTION (manager ruling, issue #61 round 1): a
   module-level ``Name`` that :func:`module_bindings` proves bound EXACTLY
   ONCE, UNCONDITIONALLY, at the top level of the module, by a plain
   single-target ``Assign(targets=[Name], value=Constant(int | float))`` --
   ANY int or float ``Constant`` value, including a non-finite float written
   as a literal (``_INF = 1e309`` parses to +inf; the manager's addition for
   the row 228 rewrite), ``UnaryOp(USub | UAdd, Constant)`` allowed
   (``_NEG = -1.0e30``, ``-1e309``); a ``float("inf")`` stays a Call -- never
   rebound, never deleted, not a kernel parameter, not bound anywhere in the
   kernel body (a kernel that binds the name anywhere refuses its module
   value; :func:`body_bound_names` collects the names a target BINDS, a
   subscript store binds none), and not the name of a whitelisted callable,
   is STATIC in every STATIC role: subscript index arithmetic (``V[h * _HD +
   j]``), the SIZE slot of ``Tensor(...)`` / ``Dim(...)`` (``Tensor(E=_E)``,
   ``Dim("K", size=_TOPK)``, size arithmetic ``P=nb * _C``), loop bounds,
   the TEST of an ``if`` / ``IfExp`` (``if v > _EPS``), the constant keywords
   of ``Tensor`` (``fill=_INF * -1.0`` with ``_INF = 1e30``: ``fill``,
   ``dtype``, ``role``, ``sparsity`` are VALUE-slot constants, not sizes) and
   arithmetic composed only of STATIC operands.  In a VALUE slot the Name
   itself is STATIC and arithmetic with a VALUE operand is that VALUE: ``x *
   _SCALE`` is VALUE arithmetic whose class is unchanged (one value factor),
   ``acc += X[m, k] * W[n, k] * _SCALE`` is still one MAC with ``_SCALE`` a
   scale.  The INT / FLOAT boundary: an INT literal name is admitted in every
   role; a FLOAT literal name is admitted only in VALUE roles (a factor, an
   assigned value, a test, a ``fill``) and REFUSES anywhere inside an INDEX,
   DIVISOR or SIZE slot -- also through a cast, arithmetic or a STATIC ALIAS
   (``W[m, int(k // _BLK)]`` with ``_BLK = 2.0`` refuses; so does ``f = _F;
   W[m, int(k * f)]`` and ``g = f * 2; Tensor(M=int(g))``: a local whose
   STATIC value derives from a float literal is FLOAT-DERIVED, tracked
   flow-ordered as a snapshot component -- ``float_static`` -- so it takes
   part in the whole-state fixpoint key, is re-derived by every rebinding
   FROM THE INCOMING VALUE (``f = f``, ``f = f * 1``, ``f = int(f)`` keep the
   mark, ``f = 2`` clears it: the value is evaluated before the destination
   is rebound, ISSUE61-FLOAT-SELF-TRANSFER-5) and is the union at a
   control-flow join; an index and an extent are integral).  Outside the production and UNPROVEN as before: an
   expression-bound name (``_G = _NH // _KV``, ``_CUT1 = 2 * _MROPE_T``), a
   call-bound name (``_INF = float("inf")``), a name bound more than once,
   conditionally (under ``if`` / ``for`` / ``try`` / ``with``), by a chained
   or unpacking target, an annotated or augmented assignment, a walrus, an
   import or a ``def``, a deleted name, a name a star import may bind, a name
   the kernel rebinds or takes as a parameter, a literal name used as a
   tensor receiver (``_E[k]``) or as a callee, and a literal name that
   shadows a whitelisted builtin (``abs = 2``).
   SINGLETON-SCALE-READ PRODUCTION (manager ruling, issue #61 round 1): a
   tensor ``Subscript(value=Name(<declared kernel INPUT>), slice=Constant(0)
   | Tuple(Constant(0), ...))`` -- the receiver is a parameter annotated
   ``Tensor("<axes>", ...)`` that is NEVER REBOUND anywhere in the kernel
   body -- any binding form, before or after the read, conditional or not
   (``GM[0]`` reads followed by ``GM = GM`` refuse; the SIZE production's
   ``body_bound_names`` census, ISSUE61-SINGLETON-REBIND-3) -- EVERY index is
   the int ``Constant`` ``0`` and the number of indices equals the declared
   rank -- is a VALUE of SCALAR
   origin (a runtime scalar: a value factor as a multiplier, a scale as a
   divisor), never STATIC, PROVIDED the bound input's extent along every
   indexed axis is exactly 1.  The classifier proves the shape of the read
   (receiver, rank, literal indices) and records the candidate
   (``StatementContext.singleton_reads``, per statement
   ``singleton_statements``; the candidate set takes part in the fixpoint
   key); the EXTENT is proven at the call site by the converter from the
   DECLARED SHAPE BINDING -- the recorded input tensor in the parameter's
   position (signature input order, a clean tensor prefix) must have exactly
   that rank with every dim size 1 -- never from the index; a call whose
   binding does not prove it (a non-singleton tensor, a rank mismatch, a
   legacy call-order trace) makes THAT layer ``class_evidence_unproven``
   with the reason in ``class_grammar_violations``.  A kernel that reads
   through the production records ``singleton_scale_reads`` on its layer,
   beside ``gather_index_statements``, with the same receipt algebra as the
   gather receipt (ISSUE61-SINGLETON-RECEIPT-4): per record (a statement's
   flag is its own; across the branches of an ``if`` the per-branch maximum
   carried by that branch's records), a statement-``if`` TEST that reads a
   singleton is one control EVALUATION per trip of the ``if``
   (``loop_singleton_controls``), and a read OUTSIDE every loop -- an outer
   ``scale = GM[0]`` or an outer ``if GM[0] > 0:`` test -- counts once per
   kernel call (``loop_singleton_outer_reads``).  It is the NVFP4 global-scale form of the
   Lite decode / encode / epilogue kernels (``ges = 2688.0 / GM[0]`` with
   ``GM: Tensor("ONE")`` bound to a one-element tensor; ``GA[0]``, ``GW[0]``,
   ``GQ[0]``, ``GK[0]``, ``GV[0]``, ``XG[0]``).  Everything else stays
   UNPROVEN: a literal-index read of a non-singleton input is refused at the
   binding (``W[0, 0]`` on an (M, K) input, ``ST[0]`` / ``ST[1]`` state reads
   on a (2,) or (3,) input), a rank mismatch (``G[0]`` on a ``Tensor("A,B")``
   declared (1, 1)) and any non-zero or non-literal index without a loop
   index are refused by the classifier, a local allocation or a local alias
   read by a literal index is not a declared input (``T = Tensor(ONE=1);
   T[0]``), and a STORE through the singleton form (``GM[0] = x``, a write to
   a declared input) refuses.  Locals derived from a singleton read are VALUE
   by the existing rules; ``GM[k]`` with a loop index is the ordinary
   tensor-element rule.
4. Unsupported constructs refuse the kernel: ``NamedExpr`` (walrus) anywhere,
   lambdas / defs / classes inside the kernel, comprehensions, ``try`` /
   ``with`` / ``while``, starred or double-starred call arguments, attribute
   access or calls on tensor-derived receivers, ``global`` / ``nonlocal``,
   ``del``, tuple / chained / unpacking targets, string formatting, list /
   dict / set displays, any call outside the whitelist.

Every classification entry point (``_classify_rhs``, ``_classify_product``,
non-accumulation assignments, statement-``if`` and ``IfExp`` tests and arms,
subscript and shape indices, ``Tensor`` / ``Dim`` arguments, loop bounds and
``return`` expressions) routes through :func:`provenance`: no path returns a
complete class without consulting it, and UNPROVEN dominates every join.

This module is the single classification code path: the DSL parser records
the classes per executed assignment and the converter sums the trips per
class; the graph analyzer bills MACs and ``other_ops`` from those sums and
never from a whole-kernel flag.
"""
from __future__ import annotations

import ast
from typing import Iterable, Optional, Set, Tuple

MAC = "mac"
OTHER = "other"

# provenance classes (VALUE carries its origin: SCALAR or TENSOR)
STATIC = "static"
SCALAR = "scalar"
TENSOR = "tensor"
TENSOR_OBJ = "tensor_object"      # a name bound to a Tensor (parameter / local allocation): subscriptable, not a value
UNPROVEN = "unproven"
VALUE_CLASSES = (SCALAR, TENSOR)

UNPROVEN_FLAG = "class_evidence_unproven"
BRANCH_FLAG = "branch_class_uncertain"

_SUM_OPS = (ast.Add, ast.Sub)
# pure-static builtins: STATIC on STATIC arguments through an unshadowed name; a cast of a value is a value
_PURE_STATIC_BUILTINS = frozenset({"int", "float", "abs", "min", "max", "round", "floor", "len", "range", "bool"})
# the DSL execution namespace's math functions (einsum_lib.numba_accel.execution_namespace): a value of values
_VALUE_FUNCTIONS = frozenset({"exp", "log", "sqrt", "cos", "sin", "tanh", "floor", "log2", "round", "erf", "fp32",
                              "abs", "max", "min", "int", "float"})
_STATIC_ATTR_CALLS = frozenset({"size", "dim", "numel"})
_MASK_BUILTINS = frozenset({"bool"})
_DSL_CTORS = frozenset({"Dim", "Tensor"})
# the SLOT an expression is classified for: a value (factor, assigned value, test), a tensor Subscript
# index, or a Tensor(...) / Dim(...) size -- the two productions are admitted by slot
SLOT_VALUE, SLOT_INDEX, SLOT_SIZE = "value", "index", "size"
# the right operand of a floor division met in the index / size slot: the only slot that admits the BLOCK-INDEX
# PRODUCTION's once-bound module int literal (every expression node is visited through provenance, this one included)
SLOT_DIVISOR = "divisor"
# every callee the grammar admits (through an unshadowed plain name)
ALLOWED_CALLEES = _PURE_STATIC_BUILTINS | _VALUE_FUNCTIONS | _DSL_CTORS
# the constant keywords of Tensor(...): VALUE-slot constants (a fill, a dtype, a role), never sizes
_TENSOR_CONSTANT_KEYWORDS = frozenset({"dtype", "role", "sparsity", "fill"})
LITERAL_INT, LITERAL_FLOAT = "int", "float"


def _is_tensor_ctor(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == "Tensor")
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == "Tensor")))


class StatementContext:
    """What the classifier knows about the names of one kernel body: which
    names are Tensor objects, which are runtime scalars (scalar parameters),
    every name bound anywhere in the body (shadowing), the names bound to
    differing classes (unproven at every use), and -- per statement -- the
    flow-ordered bindings in force before that statement."""

    def __init__(self, tensor_names: Iterable[str] = (), scalar_names: Iterable[str] = (),
                 aliases: Optional[dict] = None):
        self.tensor_names = frozenset(tensor_names)
        self.scalar_names = frozenset(scalar_names)
        self.aliases = dict(aliases or {})        # the bindings in force at the END of the body (final state)
        self.index_names: Set[str] = set()        # names bound to a loop index at the end of the body
        self.bound_names: Set[str] = set()        # every name bound anywhere in the body
        self.conflicting: Set[str] = set()        # names bound to differing provenance classes
        self.module_bound: Set[str] = set()       # names bound anywhere at module scope ("*" for a star import)
        self.literal_names: dict = {}             # LITERAL-CONSTANT PRODUCTION: module name -> "int" | "float" (once-bound literal, not kernel-bound)
        self.block_divisors: Set[str] = set()     # module-level names bound once to an INT literal: the BLOCK-INDEX PRODUCTION's divisor
        self.input_ranks: dict = {}               # declared kernel INPUT tensors: parameter -> declared rank (the Tensor("A,B") annotation)
        self.singleton_reads: dict = {}           # SINGLETON-SCALE-READ candidates: parameter -> rank (the extent is proven at the call site)
        self.singleton_statements: Set[int] = set()  # id(stmt) of statements reading through a singleton candidate
        self.body_rebound: Set[str] = set()       # every name the body binds anywhere (body_bound_names): a rebound input is no singleton
        self._singleton_hit: bool = False         # a singleton read was met while classifying the selected statement
        self.violations: list = []                # supported-grammar violations (kernel-level UNPROVEN)
        self.converged: bool = True               # the binding fixpoint converged within its bound
        self.gather_statements: Set[int] = set()  # id(stmt) of statements addressing through a GATHER index
        self._gather_hit: bool = False            # a gather index was met while classifying the selected statement
        self.module_info: dict = {}               # module_bindings(module): name -> {count, kinds, deleted, conditional}
        self.int_params: Set[str] = set()         # kernel parameters annotated `int`: the DSL's size inputs (SIZE slot only)
        self.sizes: dict = {}                     # name -> its provenance in the SIZE slot at the end of the body
        self._sizes: Optional[dict] = None        # the active view (selected statement)
        self.alias_specs: dict = {}               # alias name -> the operand spec FROZEN at its defining assignment (final state)
        self._at: dict = {}                       # id(stmt) -> (bindings, index_names, sizes, alias_specs) before that statement
        self._bindings: Optional[dict] = None     # the active view (selected statement)
        self._index: Optional[Set[str]] = None
        self._extra: tuple = ()                   # the selected statement's further snapshot components (snapshot_extra)
        self._alias_specs: Optional[dict] = None  # the active view (selected statement)
        self.float_static: Set[str] = set()       # FLOAT-DERIVED STATIC locals at the end of the body (snapshot component 3 per statement)

    # -- per-statement view -------------------------------------------------
    def select(self, stmt: Optional[ast.stmt]) -> None:
        snap = self._at.get(id(stmt)) if stmt is not None else None
        self._bindings, self._index, self._sizes = snap[:3] if snap is not None else (None, None, None)
        self._extra = tuple(snap[3:]) if snap is not None else ()        # later state components, frozen per statement
        self._alias_specs = snap[4] if snap is not None and len(snap) >= 5 else None   # component 4: the alias records (3 is #61's float-derived set)
        self._gather_hit = False
        self._singleton_hit = False

    def snapshot_extra(self) -> tuple:
        """The state components beyond (bindings, index names, sizes) recorded before the selected statement; a
        later change that captures e.g. the reaching operands of an alias AT ITS DEFINING ASSIGNMENT appends them
        here and they take part in the fixpoint key automatically (freeze_state over the whole table)."""
        return self._extra

    def size_provenance(self, name: str) -> Optional[str]:
        """The provenance a local Name has in the SIZE slot (its reaching value classified as a size)."""
        d = self._sizes if self._sizes is not None else self.sizes
        return d.get(name)

    def note_gather(self) -> None:
        """A tensor Subscript index that is a proven VALUE (the GATHER-INDEX production)."""
        self._gather_hit = True

    def alias_spec(self, name: str):
        """The operand spec the alias *name* captured AT ITS DEFINING ASSIGNMENT, as it reaches the selected
        statement (a frozen, value-comparable tuple; None: not an alias / no single reaching definition).  Later
        rebinding of the names the alias read never changes a captured value (DTYPE-ALIAS-CAPTURE-4)."""
        d = self._alias_specs if self._alias_specs is not None else self.alias_specs
        return d.get(name)
    def note_singleton(self, name: str, rank: int) -> None:
        """A literal-0 read of a declared input (the SINGLETON-SCALE-READ production): the
        candidate is recorded for the converter's extent proof."""
        self._singleton_hit = True
        self.singleton_reads[name] = rank

    def _view(self) -> Tuple[dict, Set[str]]:
        return (self._bindings if self._bindings is not None else self.aliases,
                self._index if self._index is not None else self.index_names)

    # -- queries --------------------------------------------------------------
    def name_provenance(self, name: str) -> str:
        if name in self.conflicting:
            return UNPROVEN
        bindings, _ = self._view()
        if name in bindings:
            return bindings[name]
        if name in self.tensor_names:
            return TENSOR_OBJ
        if name in self.scalar_names:
            return SCALAR
        if name in self.literal_names:
            return STATIC                                 # LITERAL-CONSTANT PRODUCTION: a once-bound module literal
        return UNPROVEN

    def is_bound(self, name: str) -> bool:
        bindings, _ = self._view()
        return name in bindings or name in self.tensor_names or name in self.scalar_names or name in self.literal_names

    def is_float_literal(self, name: str) -> bool:
        return self.literal_names.get(name) == LITERAL_FLOAT

    def is_float_static(self, name: str) -> bool:
        """A local whose STATIC value derives from a FLOAT literal constant (before the selected statement)."""
        extra = self._extra
        return name in (extra[0] if extra else self.float_static)

    def is_locally_bound(self, name: str) -> bool:
        """The name was (re)bound by a statement of the body before the selected statement."""
        bindings, _ = self._view()
        return name in bindings

    def is_index(self, name: str) -> bool:
        _, index = self._view()
        return name in index and name not in self.conflicting

    def is_shadowed(self, name: str) -> bool:
        """A builtin / DSL function name that the kernel binds anywhere
        (parameter, local, lambda, def, import) or that the MODULE binds
        (def / assign / import / star import) does not prove the builtin."""
        return (name in self.bound_names or name in self.tensor_names or name in self.scalar_names
                or name in self.module_bound or "*" in self.module_bound)


def _combine(provs: Iterable[str]) -> str:
    """The join of provenances: UNPROVEN dominates, then tensor > scalar > static."""
    provs = list(provs)
    if any(p in (UNPROVEN, TENSOR_OBJ) for p in provs):
        return UNPROVEN
    if any(p == TENSOR for p in provs):
        return TENSOR
    if any(p == SCALAR for p in provs):
        return SCALAR
    return STATIC


def _slice_elements(node: ast.expr) -> list:
    return list(node.elts) if isinstance(node, ast.Tuple) else [node]


def _mentions_index(node: ast.expr, loop_vars: Set[str], ctx: Optional[StatementContext]) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if ctx is not None:
                if ctx.is_index(n.id) or (n.id in loop_vars and not ctx.is_bound(n.id) and n.id not in ctx.conflicting):
                    return True
            elif n.id in loop_vars:
                return True
    return False


def _float_literal_inside(node: ast.expr, ctx: StatementContext) -> bool:
    """A FLOAT literal constant appears anywhere in *node* (an INDEX / DIVISOR / SIZE slot refuses it, also
    through a cast or arithmetic: an index and an extent are integral)."""
    return any(isinstance(n, ast.Name) and (ctx.is_float_literal(n.id) or ctx.is_float_static(n.id)) for n in ast.walk(node))


def _is_literal_zero(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) and node.value == 0


def _is_singleton_read(name: str, elts: list, ctx: StatementContext) -> bool:
    """The SHAPE of the SINGLETON-SCALE-READ production: a declared kernel INPUT (a parameter with a
    ``Tensor("<axes>")`` annotation, not rebound) read by the int Constant ``0`` on every index, with
    as many indices as declared axes.  The extent along those axes is proven by the converter from the
    call site's declared shape binding, never here."""
    rank = ctx.input_ranks.get(name)
    return (rank is not None and name not in ctx.conflicting and name not in ctx.body_rebound
            and len(elts) == rank and all(_is_literal_zero(e) for e in elts))


def _is_singleton_store(name: str, elts: list, ctx: StatementContext) -> bool:
    """The singleton form as a STORE target (``GM[0] = x`` on a declared input): refused whatever the
    rebinding census says -- the store itself is what makes the input a written, non-singleton tensor."""
    rank = ctx.input_ranks.get(name)
    return rank is not None and len(elts) == rank and all(_is_literal_zero(e) for e in elts)


def _name_provenance(name: str, loop_vars: Set[str], ctx: Optional[StatementContext]) -> str:
    if ctx is None:
        return STATIC if name in loop_vars else SCALAR
    if name in ctx.conflicting:
        return UNPROVEN
    if ctx.is_bound(name):
        return ctx.name_provenance(name)
    if name in loop_vars:
        return STATIC
    return UNPROVEN


def _callee_is(node: ast.Call, names: frozenset, ctx: Optional[StatementContext]) -> bool:
    """The call goes through an unshadowed plain name in *names*."""
    return (isinstance(node.func, ast.Name) and node.func.id in names
            and (ctx is None or not ctx.is_shadowed(node.func.id)))


def provenance(node: ast.expr, loop_vars: Set[str], ctx: Optional[StatementContext], slot: str = SLOT_VALUE) -> str:
    """The value provenance of an expression (see the module docstring),
    classified for a SLOT: ``value`` (a factor, an assigned value, a test),
    ``index`` (a tensor Subscript index) or ``size`` (a ``Tensor`` / ``Dim``
    argument).  The block-index production is STATIC only in the index and
    size slots; a proven VALUE index is a declared gather (noted on the
    context) only in the index slot.  ``ctx`` None (a bare statement with no
    kernel context) treats every non-loop name as a Tensor when subscripted
    and as a runtime scalar otherwise -- the legacy behaviour of the
    classifier tables."""
    if slot != SLOT_VALUE and ctx is not None and _float_literal_inside(node, ctx):
        return UNPROVEN                                               # LITERAL-CONSTANT: a FLOAT literal name is not an index / divisor / extent
    if isinstance(node, ast.Constant):
        return STATIC
    if isinstance(node, ast.Compare):
        provs = [provenance(node.left, loop_vars, ctx)] + [provenance(c, loop_vars, ctx) for c in node.comparators]
        return UNPROVEN if _combine(provs) == UNPROVEN else STATIC   # a mask only gates
    if isinstance(node, ast.BoolOp):
        return _combine(provenance(v, loop_vars, ctx) for v in node.values)   # or / and return an operand
    if isinstance(node, ast.Name):
        if slot == SLOT_DIVISOR and ctx is not None and node.id in ctx.block_divisors and not ctx.is_bound(node.id):
            return STATIC                                             # BLOCK-INDEX PRODUCTION: the once-bound module int literal divisor
        if slot == SLOT_SIZE and ctx is not None and node.id not in ctx.conflicting:
            if node.id in ctx.int_params:
                return STATIC                                         # an `int` parameter never rebound anywhere in the body (SIZE-BINDING-23)
            if ctx.size_provenance(node.id) == STATIC and ctx.name_provenance(node.id) != UNPROVEN:
                return STATIC                                         # a local bound to STATIC size arithmetic
        return _name_provenance(node.id, loop_vars, ctx)
    if isinstance(node, ast.Attribute):
        if node.attr == "shape" and isinstance(node.value, ast.Name):
            base = _name_provenance(node.value.id, loop_vars, ctx) if ctx is not None else TENSOR_OBJ
            return STATIC if base == TENSOR_OBJ else UNPROVEN         # X.shape of a tensor NAME
        return UNPROVEN
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr == "shape":
            idx = provenance(node.slice, loop_vars, ctx, SLOT_INDEX)
            if idx != STATIC or not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int) and not isinstance(node.slice.value, bool)):
                return UNPROVEN                                       # W.shape[<index>]: only a literal index is proven
            return provenance(base, loop_vars, ctx)                   # W.shape[1]
        if isinstance(base, ast.Name):
            base_prov = TENSOR_OBJ if ctx is None and base.id not in loop_vars else _name_provenance(base.id, loop_vars, ctx)
            if base_prov == TENSOR_OBJ:                               # ONLY a tensor OBJECT is a receiver: a tensor-origin VALUE is a number (GATHER-RECEIVER-28)
                elts = _slice_elements(node.slice)
                if ctx is not None and _is_singleton_read(base.id, elts, ctx):
                    for e in elts:
                        provenance(e, loop_vars, ctx, SLOT_INDEX)        # the literal indices are visited like any index
                    ctx.note_singleton(base.id, len(elts))            # SINGLETON-SCALE-READ PRODUCTION: a runtime scalar
                    return SCALAR
                provs = [provenance(e, loop_vars, ctx, SLOT_INDEX) for e in elts]
                if any(pe == UNPROVEN for pe in provs):
                    return UNPROVEN                                   # an unproven index
                if any(pe in VALUE_CLASSES for pe in provs):
                    if ctx is not None:
                        ctx.note_gather()                             # GATHER-INDEX PRODUCTION: a proven VALUE index
                    return TENSOR                                     # ... admitted whether or not a loop variable is mentioned (W[t, ge])
                # every index STATIC: a loop-indexed element; a literal-only / STATIC-only index set (W[0, 0]) stays
                # unproven -- the classifier does not know the tensor's extent (the reviewer's W[0,0] rule); the one
                # admitted literal-only form is the SINGLETON-SCALE-READ production above (GM[0] on a declared input)
                return TENSOR if any(_mentions_index(e, loop_vars, ctx) for e in elts) else UNPROVEN
            return UNPROVEN
        return UNPROVEN
    if isinstance(node, ast.Call):
        if any(isinstance(a, ast.Starred) for a in node.args) or any(kw.arg is None for kw in node.keywords):
            return UNPROVEN                                           # *args / **kwargs: the call is not proven
        args = list(node.args) + [kw.value for kw in node.keywords]
        if isinstance(node.func, ast.Attribute) and node.func.attr in _STATIC_ATTR_CALLS and isinstance(node.func.value, ast.Name):
            base = _name_provenance(node.func.value.id, loop_vars, ctx) if ctx is not None else TENSOR_OBJ
            if base == TENSOR_OBJ and all(provenance(a, loop_vars, ctx) == STATIC for a in args):
                return STATIC                                         # X.size(1) of a tensor NAME
            return UNPROVEN
        if isinstance(node.func, ast.Name) and node.func.id in ("Dim", "Tensor") and (ctx is None or not ctx.is_shadowed(node.func.id)):
            # every constructor argument must be STATIC: the dim sizes are SIZE slots (a literal, Dim, .shape[<literal>],
            # STATIC size arithmetic, an int literal constant, the block-index production); Tensor's constant keywords
            # (dtype / role / sparsity / fill) are VALUE-slot constants (a float literal constant is a fill, not a size);
            # anything else (a bare module name, a value) leaves the object unproven
            if not all(provenance(a, loop_vars, ctx, SLOT_SIZE) == STATIC for a in node.args):
                return UNPROVEN
            for kw in node.keywords:
                kw_slot = SLOT_VALUE if (node.func.id == "Tensor" and kw.arg in _TENSOR_CONSTANT_KEYWORDS) else SLOT_SIZE
                if provenance(kw.value, loop_vars, ctx, kw_slot) != STATIC:
                    return UNPROVEN
            return TENSOR_OBJ if node.func.id == "Tensor" else STATIC
        if _callee_is(node, frozenset({"len"}), ctx) and len(args) == 1 and isinstance(args[0], ast.Name):
            base = provenance(args[0], loop_vars, ctx) if ctx is not None else TENSOR_OBJ
            if base == TENSOR_OBJ:
                return STATIC                                         # len(X) of a tensor NAME
        if _callee_is(node, _MASK_BUILTINS, ctx):
            inner = _combine(provenance(a, loop_vars, ctx) for a in args)
            return UNPROVEN if inner == UNPROVEN else STATIC          # an explicit boolean is a mask
        if _callee_is(node, _PURE_STATIC_BUILTINS | _VALUE_FUNCTIONS, ctx):
            if not args:
                return UNPROVEN
            sub = slot if (slot == SLOT_SIZE and node.func.id in _PURE_STATIC_BUILTINS) else SLOT_VALUE
            return _combine(provenance(a, loop_vars, ctx, sub) for a in args)   # static of statics; a cast / function of a value is that value
        return UNPROVEN                                               # unknown, shadowed, lambda, method, module call
    if isinstance(node, ast.BinOp):
        sub = slot if slot == SLOT_SIZE else SLOT_VALUE                  # size arithmetic stays in the size slot
        if slot in (SLOT_INDEX, SLOT_SIZE) and isinstance(node.op, ast.FloorDiv) and isinstance(node.right, ast.Name) and ctx is not None:
            # BLOCK-INDEX PRODUCTION: k // _BLK in the index / size slot -- a STATIC dividend (classified in this slot, so
            # the production nests) floor-divided by the once-bound module int literal, which only the divisor slot
            # admits; anything else is ordinary arithmetic of the sub-slot (both operands are visited either way)
            if (provenance(node.left, loop_vars, ctx, slot) == STATIC
                    and provenance(node.right, loop_vars, ctx, SLOT_DIVISOR) == STATIC):
                return STATIC
        return _combine((provenance(node.left, loop_vars, ctx, sub), provenance(node.right, loop_vars, ctx, sub)))
    if isinstance(node, ast.UnaryOp):
        inner = provenance(node.operand, loop_vars, ctx, slot if slot == SLOT_SIZE else SLOT_VALUE)
        if isinstance(node.op, ast.Not):
            return UNPROVEN if inner == UNPROVEN else STATIC           # an explicit boolean is a mask
        return inner
    if isinstance(node, ast.NamedExpr):
        return UNPROVEN                                               # walrus: outside the grammar
    if isinstance(node, ast.IfExp):
        test = provenance(node.test, loop_vars, ctx)
        body = provenance(node.body, loop_vars, ctx)
        orelse = provenance(node.orelse, loop_vars, ctx)
        if test != STATIC:
            return UNPROVEN                                           # a TEST must be a proven mask at every depth
        if body == orelse:
            return body                                               # equal arms keep their class (two statics: a mask; two tensor objects: that object)
        return _combine((body, orelse))
    return UNPROVEN


def _product_factors(node: ast.expr, denominator: bool = False) -> list:
    """Flatten a chain of ``*`` / ``/`` into ``(factor, is_denominator)``
    pairs; a non-product is one factor."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _product_factors(node.left, denominator) + _product_factors(node.right, denominator)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _product_factors(node.left, denominator) + _product_factors(node.right, True)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _product_factors(node.operand, denominator)
    return [(node, denominator)]


def _is_value_factor(prov: str, denominator: bool) -> bool:
    if prov in (TENSOR, UNPROVEN, TENSOR_OBJ):
        return True
    if prov == SCALAR:
        return not denominator                            # FMA as a multiplier, a scale as a divisor
    return False                                          # static: a scale either side


def value_factor_count(node: ast.expr, loop_vars: Optional[Iterable[str]] = None,
                       ctx: Optional[StatementContext] = None) -> int:
    """Number of value factors in the product *node*."""
    lv = set(loop_vars or ())
    return sum(1 for f, den in _product_factors(node) if _is_value_factor(provenance(f, lv, ctx), den))


def _classify_product(product: ast.expr, lv: Set[str], ctx: Optional[StatementContext]) -> Tuple[str, int, Set[str]]:
    flags: Set[str] = set()
    while isinstance(product, ast.UnaryOp) and isinstance(product.op, (ast.USub, ast.UAdd)):
        product = product.operand
    if not (isinstance(product, ast.BinOp) and isinstance(product.op, (ast.Mult, ast.Div))):
        if provenance(product, lv, ctx) == UNPROVEN:
            flags.add(UNPROVEN_FLAG)                                  # an unproven accumulated value (Y += f(k))
        return OTHER, 0, flags
    n_value = 0
    for factor, den in _product_factors(product):
        prov = provenance(factor, lv, ctx)
        if prov in (UNPROVEN, TENSOR_OBJ):
            flags.add(UNPROVEN_FLAG)
        if _is_value_factor(prov, den):
            n_value += 1
    if n_value < 2:
        return OTHER, 0, flags
    return MAC, n_value - 2, flags


def _merge_arms(body: list, orelse: list) -> list:
    """The per-class maximum over two arms of differing composition (the
    statement-``if`` rule): MACs and others each at their larger count."""
    n_mac = max(sum(1 for k, _ in body if k == MAC), sum(1 for k, _ in orelse if k == MAC))
    n_other = max(sum(1 for k, _ in body if k == OTHER), sum(1 for k, _ in orelse if k == OTHER))
    extra = max(sum(e for k, e in body if k == MAC), sum(e for k, e in orelse if k == MAC))
    return [(MAC, extra if i == 0 else 0) for i in range(n_mac)] + [(OTHER, 0)] * n_other


def _classify_rhs(rhs: ast.expr, lv: Set[str], ctx: Optional[StatementContext]) -> Tuple[list, Set[str]]:
    while isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, (ast.USub, ast.UAdd)):
        rhs = rhs.operand
    entry: Set[str] = {UNPROVEN_FLAG} if provenance(rhs, lv, ctx) == UNPROVEN else set()
    if isinstance(rhs, ast.IfExp):
        if provenance(rhs.test, lv, ctx) != STATIC:
            entry.add(UNPROVEN_FLAG)                                  # a TEST must be a proven mask
        body, fb = _classify_rhs(rhs.body, lv, ctx)
        orelse, fo = _classify_rhs(rhs.orelse, lv, ctx)
        flags = fb | fo | entry
        if sorted(body) == sorted(orelse):
            return body, flags
        flags.add(BRANCH_FLAG)
        return _merge_arms(body, orelse), flags
    kind, extra, flags = _classify_product(rhs, lv, ctx)
    return [(kind, extra)], flags | entry


def _same_target(a: ast.expr, b: ast.expr) -> bool:
    try:
        return ast.unparse(a) == ast.unparse(b)
    except Exception:  # pragma: no cover - defensive
        return False


def _assign_view(stmt: ast.stmt) -> ast.stmt:
    """A VALUED annotated assignment executes as the plain assignment of its value (a local annotation is never
    evaluated): the equivalent ``Assign`` node, located like the original; any other statement unchanged."""
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        new = ast.copy_location(ast.Assign(targets=[stmt.target], value=stmt.value, type_comment=None), stmt)
        new.end_lineno, new.end_col_offset = getattr(stmt, "end_lineno", None), getattr(stmt, "end_col_offset", None)
        new.annotation = stmt.annotation                                  # kept for the record; not a classified slot
        return new
    return stmt


def normalize_statements(stmts: list) -> list:
    """AnnAssign-is-Assign (manager ruling, an earlier fix round 6; TARGET-KB-ANNASSIGN-CLASS-27): every valued ``AnnAssign``
    in *stmts* and in every nested statement suite is rewritten IN PLACE to its :func:`_assign_view`, so that
    accumulator detection, binding, flow, counting and classification have ONE code path.  Returns *stmts*."""
    for i, s in enumerate(stmts):
        view = _assign_view(s)
        if view is not s:
            stmts[i] = s = view
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(s, field, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                normalize_statements(sub)
        for h in getattr(s, "handlers", None) or []:
            normalize_statements(h.body)
        for c in getattr(s, "cases", None) or []:
            normalize_statements(c.body)
    return stmts


def _accumulated_rhs(stmt: ast.stmt) -> Optional[ast.expr]:
    stmt = _assign_view(stmt)
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, _SUM_OPS):
        return stmt.value
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.value, ast.BinOp) and isinstance(stmt.value.op, _SUM_OPS)):
        # ``T = T + a * b``  /  ``T = a * b + T``
        if _same_target(stmt.value.left, stmt.targets[0]):
            return stmt.value.right
        if isinstance(stmt.value.op, ast.Add) and _same_target(stmt.value.right, stmt.targets[0]):
            return stmt.value.left
    return None


def classify_records(stmt: ast.stmt, loop_vars: Optional[Iterable[str]] = None,
                     ctx: Optional[StatementContext] = None) -> Tuple[list, Set[str]]:
    """``([(kind, extra_other), ...], flags)`` for one executed assignment.

    One record per statement, except an accumulated inline conditional whose
    arms differ in class: then the per-class maximum over the arms (several
    records) with ``branch_class_uncertain``.  ``class_evidence_unproven`` is
    set when a factor's provenance could not be proven.  Non-assignment
    statements are one ``OTHER`` record so callers can count them uniformly.
    With a context, the bindings in force BEFORE *stmt* are consulted
    (flow order).
    """
    lv = set(loop_vars or ())
    if ctx is not None:
        ctx.select(stmt)
    original = stmt
    stmt = _assign_view(stmt)                                         # AnnAssign is Assign (one classification path)
    try:
        rhs = _accumulated_rhs(stmt)
        if rhs is None:
            flags: Set[str] = set()
            value = getattr(stmt, "value", None)
            if value is not None and provenance(value, lv, ctx) == UNPROVEN:
                flags.add(UNPROVEN_FLAG)                              # an unproven assigned value (t = f(k))
            targets = list(getattr(stmt, "targets", [])) or [getattr(stmt, "target", None)]
            for target in targets:
                if isinstance(target, ast.Subscript):
                    if _store_index_unproven(target, lv, ctx):
                        flags.add(UNPROVEN_FLAG)                      # an unproven store index (Y[f(k)] = ...)
                elif target is not None and not isinstance(target, ast.Name):
                    flags.add(UNPROVEN_FLAG)                          # unsupported target form
            if ctx is not None and ctx._gather_hit:
                ctx.gather_statements.add(id(original))
            if ctx is not None and ctx._singleton_hit:
                ctx.singleton_statements.add(id(original))
            return [(OTHER, 0)], flags
        records, flags = _classify_rhs(rhs, lv, ctx)
        target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
        if isinstance(target, ast.Subscript) and _store_index_unproven(target, lv, ctx):
            flags = set(flags) | {UNPROVEN_FLAG}
        if ctx is not None and ctx._gather_hit:
            ctx.gather_statements.add(id(original))                   # the statement addresses through a declared gather
        if ctx is not None and ctx._singleton_hit:
            ctx.singleton_statements.add(id(original))                # the statement reads through a singleton scale
        return records, flags
    finally:
        if ctx is not None:
            ctx.select(None)


def _store_index_unproven(target: ast.Subscript, lv: Set[str], ctx: Optional[StatementContext]) -> bool:
    """A store ``T[i, j] = ...`` through the INDEX slot: STATIC indices, or a proven VALUE
    index as a declared gather (scatter) on a tensor receiver; anything else is unproven."""
    base = target.value
    base_prov = (TENSOR_OBJ if ctx is None else _name_provenance(base.id, lv, ctx)) if isinstance(base, ast.Name) else UNPROVEN
    if base_prov != TENSOR_OBJ:                                       # a tensor-origin VALUE is not a receiver (GATHER-RECEIVER-28)
        return True
    if ctx is not None and _is_singleton_store(base.id, _slice_elements(target.slice), ctx):
        return True                                       # a STORE through the singleton form: a write to a declared input refuses
    provs = [provenance(e, lv, ctx, SLOT_INDEX) for e in _slice_elements(target.slice)]
    if any(pe == UNPROVEN for pe in provs):
        return True
    if any(pe in VALUE_CLASSES for pe in provs) and ctx is not None:
        ctx.note_gather()
    return False


def classify_statement(stmt: ast.stmt, loop_vars: Optional[Iterable[str]] = None,
                       ctx: Optional[StatementContext] = None):
    """Return ``(kind, extra_other)`` for one executed assignment statement
    (the single-record view: a per-class maximum reports its MAC record)."""
    records, _flags = classify_records(stmt, loop_vars, ctx)
    macs = [r for r in records if r[0] == MAC]
    return macs[0] if macs else records[0]


def _target_names(target: ast.expr) -> list:
    """The names an assignment TARGET binds: a Name, or the Names of tuple / list / starred unpacking.  A
    subscript or attribute store (``Y[i + n] = ...``, ``obj.attr = ...``) mutates an object and binds NOTHING:
    its receiver and index expressions are READS (size-read-as-binding-32)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list = []
        for elt in target.elts:
            out.extend(_target_names(elt))
        return out
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


_DSL_MODULES = frozenset({"solar.einsum_lib", "einsum_lib"})


def _is_dsl_import(s: ast.ImportFrom) -> bool:
    """The DSL's own vocabulary import: exactly ``from solar.einsum_lib import <names>`` (or the
    relative ``from .einsum_lib`` / ``from einsum_lib``), plain names, no alias, no star."""
    return (s.module in _DSL_MODULES or (s.level and s.module == "einsum_lib")) and not any(a.name == "*" or a.asname for a in s.names)


def scope_walk(node: ast.AST):
    """Yield the nodes of *node* that execute in ITS scope: every child is descended except a lambda's BODY (its
    defaults are yielded, they execute where the lambda is defined) and the bodies of a nested def / class (their
    definition-time expressions are yielded).  A comprehension is walked in full: an assignment expression inside
    it binds in the CONTAINING scope (PEP 572), so ``[(abs := f) for _ in xs]`` does shadow ``abs``
    (module-lambda-scope-33)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, ast.Lambda):
            stack.extend(n.args.defaults)
            stack.extend(d for d in n.args.kw_defaults if d is not None)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(definition_time_expressions(n))
        else:
            stack.extend(ast.iter_child_nodes(n))


def enclosing_scope_walrus_targets(node: ast.AST) -> list:
    """The Names that assignment expressions reachable from *node* bind IN THE ENCLOSING SCOPE (see
    :func:`scope_walk`), in source order."""
    out: list = []
    for sub in scope_walk(node):
        if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
            out.append(sub.target.id)
    return out


def definition_time_expressions(node: ast.stmt) -> list:
    """The expressions of a ``def`` / ``class`` statement that execute in the ENCLOSING scope when the statement
    runs: decorators; positional, keyword-only, vararg and kwarg annotations; defaults and keyword defaults; the
    return annotation; class bases and keyword values.  The body is the nested scope and is not included."""
    out = list(getattr(node, "decorator_list", None) or [])
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x is not None]:
            if arg.annotation is not None:
                out.append(arg.annotation)
        out.extend(a.defaults)
        out.extend(d for d in a.kw_defaults if d is not None)
        if node.returns is not None:
            out.append(node.returns)
    elif isinstance(node, ast.ClassDef):
        out.extend(node.bases)
        out.extend(kw.value for kw in node.keywords)
    return out


def module_bindings(module: Optional[ast.Module]) -> dict:
    """One scope-aware collector over the WHOLE module tree: every statement
    suite (if / else, for / while bodies, try, with, match) is visited; nested
    def / class / lambda bodies are not (their locals are not module names)
    except for ``global`` declarations anywhere, which bind at module scope.
    Returns ``name -> {"count", "kinds", "deleted", "conditional", "int_literal"}``;
    the key ``"*"`` records a star import."""
    info: dict = {}
    if module is None:
        return info

    def rec(name: str, kind: str, conditional: bool, int_literal: bool = False) -> None:
        e = info.setdefault(name, {"count": 0, "kinds": set(), "deleted": False, "conditional": False, "int_literal": False})
        e["count"] += 1
        e["kinds"].add(kind)
        e["conditional"] = e["conditional"] or conditional
        e["int_literal"] = e["int_literal"] or int_literal

    def visit(stmts, conditional: bool) -> None:
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rec(s.name, "def", conditional)
                for inner in ast.walk(s):
                    if isinstance(inner, ast.Global):
                        for n in inner.names:
                            rec(n, "global", True)
                # decorators, argument defaults, annotations, the return annotation and class bases / keywords
                # EXECUTE IN THE ENCLOSING SCOPE at definition time: a walrus in them binds a module name
                # (module-defaults-26); only the def / class BODY is the nested scope
                for expr in definition_time_expressions(s):
                    for n in enclosing_scope_walrus_targets(expr):             # a lambda BODY inside them is nested scope
                        rec(n, "walrus", conditional)
                continue
            if isinstance(s, ast.ImportFrom) and _is_dsl_import(s):
                continue                                              # the DSL vocabulary
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    rec("*" if a.name == "*" else (a.asname or a.name).split(".")[0], "import", conditional)
            elif isinstance(s, ast.Assign):
                v = s.value
                if isinstance(v, ast.UnaryOp) and isinstance(v.op, (ast.USub, ast.UAdd)):
                    v = v.operand
                lit = isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool)
                single = len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                for tgt in s.targets:
                    for n in _target_names(tgt):
                        rec(n, "assign" if single else "chained_or_unpacking", conditional, lit and single)
            elif isinstance(s, ast.AnnAssign):
                for n in _target_names(s.target):
                    rec(n, "annassign", conditional)
            elif isinstance(s, ast.AugAssign):
                for n in _target_names(s.target):
                    rec(n, "augassign", conditional)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                for n in _target_names(s.target):
                    rec(n, "for", conditional)
                visit(s.body, True); visit(s.orelse, True)
            elif isinstance(s, ast.While):
                visit(s.body, True); visit(s.orelse, True)
            elif isinstance(s, ast.If):
                visit(s.body, True); visit(s.orelse, True)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if item.optional_vars is not None:
                        for n in _target_names(item.optional_vars):
                            rec(n, "with", conditional)
                visit(s.body, True)
            elif isinstance(s, ast.Try):
                visit(s.body, True)
                for h in s.handlers:
                    if h.name:
                        rec(h.name, "except", True)
                    visit(h.body, True)
                visit(s.orelse, True); visit(s.finalbody, True)
            elif isinstance(s, ast.Delete):
                for tgt in s.targets:
                    for n in _target_names(tgt):
                        e = info.setdefault(n, {"count": 0, "kinds": set(), "deleted": False, "conditional": False, "int_literal": False})
                        e["deleted"] = True
            elif hasattr(ast, "Match") and isinstance(s, ast.Match):
                for case in s.cases:
                    for inner in ast.walk(case.pattern):
                        for attr in ("name", "rest"):
                            n = getattr(inner, attr, None)
                            if isinstance(n, str):
                                rec(n, "match", True)
                    visit(case.body, True)
            # walrus targets in the statement's expressions bind at module scope -- except inside a lambda BODY,
            # which is nested scope (scope_walk)
            for sub in scope_walk(s):
                if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                    rec(sub.target.id, "walrus", conditional)
                elif isinstance(sub, ast.Global) and not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for n in sub.names:
                        rec(n, "global", True)
    visit(module.body, False)
    return info


def module_bound_names(module: Optional[ast.Module]) -> Set[str]:
    """Every name the MODULE binds anywhere (see :func:`module_bindings`); ``"*"`` for a star import."""
    return set(module_bindings(module))


def module_literal_kinds(module: Optional[ast.Module]) -> dict:
    """Module-level names bound EXACTLY ONCE, unconditionally, at the top
    level, by a plain single-target assignment to an int or float literal
    (unary minus / plus on the literal allowed), never rebound and never
    deleted (``_BLK = 16``, ``_NEG = -1.0e30``, ``_INF = 1e309`` -- any int or
    float Constant value, a non-finite literal included): ``name -> "int" |
    "float"`` (the LITERAL-CONSTANT PRODUCTION's binding proof).  A bool, a complex, an
    expression (``1 + 1``), a call (``float("inf")``) and every other binding
    form are not literals; with a star import nothing is proven."""
    out: dict = {}
    if module is None:
        return out
    info = module_bindings(module)
    if "*" in info:
        return out
    for s in module.body:                                             # unconditional, top-level only
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            v = s.value
            if isinstance(v, ast.UnaryOp) and isinstance(v.op, (ast.USub, ast.UAdd)):
                v = v.operand
            if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
                name = s.targets[0].id
                e = info.get(name, {})
                if e.get("count") == 1 and not e.get("deleted") and not e.get("conditional") and e.get("kinds") == {"assign"}:
                    out[name] = LITERAL_INT if isinstance(v.value, int) else LITERAL_FLOAT
    return out


def module_literal_names(module: Optional[ast.Module], ints_only: bool = False) -> Set[str]:
    """The names of :func:`module_literal_kinds` (the int literals only when *ints_only*)."""
    kinds = module_literal_kinds(module)
    return {n for n, k in kinds.items() if k == LITERAL_INT or not ints_only}



def grammar_violations(stmts: Iterable[ast.stmt], ctx: StatementContext) -> list:
    """The supported-grammar violations of a kernel body (see the module
    docstring); non-empty means the WHOLE kernel is unproven."""
    out: list = []

    def note(node, what: str) -> None:
        msg = f"{what} at line {getattr(node, 'lineno', '?')}"
        if msg not in out:
            out.append(msg)

    def callee_ok(call: ast.Call) -> bool:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id in ALLOWED_CALLEES and not ctx.is_shadowed(f.id)
        if isinstance(f, ast.Attribute) and f.attr in _STATIC_ATTR_CALLS and isinstance(f.value, ast.Name):
            return True                                              # X.size(1) of a tensor NAME
        return False

    def check_stmt(s: ast.stmt) -> None:
        if isinstance(s, ast.For):
            if not isinstance(s.target, ast.Name):
                note(s, "for target is not a Name")
            if not (isinstance(s.iter, ast.Call) and isinstance(s.iter.func, ast.Name) and s.iter.func.id == "Dim"):
                note(s, "for loop is not over Dim(...)")
            if s.orelse:
                note(s, "for/else")
            for b in s.body:
                check_stmt(b)
        elif isinstance(s, ast.Assign):
            if len(s.targets) != 1:
                note(s, "chained assignment")
            for tgt in s.targets:
                if not (isinstance(tgt, ast.Name) or (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name))):
                    note(s, "unsupported assignment target")
        elif isinstance(s, ast.AugAssign):
            if not (isinstance(s.target, ast.Name) or (isinstance(s.target, ast.Subscript) and isinstance(s.target.value, ast.Name))):
                note(s, "unsupported augmented-assignment target")
        elif isinstance(s, ast.AnnAssign):
            if not isinstance(s.target, ast.Name) or s.value is None:
                note(s, "unsupported annotated assignment")
        elif isinstance(s, ast.If):
            for b in s.body + s.orelse:
                check_stmt(b)
        elif isinstance(s, (ast.Return, ast.Pass)):
            pass
        elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            pass                                                     # a docstring
        else:
            note(s, f"unsupported statement {type(s).__name__}")

    stmts = list(stmts)
    normalize_statements(stmts)                                       # AnnAssign is Assign
    for s in stmts:
        check_stmt(s)
    for s in stmts:
        for node in ast.walk(s):
            if isinstance(node, ast.NamedExpr):
                note(node, "walrus (NamedExpr)")
            elif isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                note(node, f"nested {type(node).__name__}")
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                note(node, "comprehension")
            elif isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
                note(node, "string formatting")
            elif isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple)) and not isinstance(getattr(node, "ctx", None), ast.Store):
                parent_ok = False
                note(node, f"{type(node).__name__} display") if not _is_subscript_index(node, stmts) else None
            elif isinstance(node, ast.Starred):
                note(node, "starred expression")
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                note(node, type(node).__name__.lower())
            elif isinstance(node, ast.Call):
                if not callee_ok(node):
                    note(node, f"call outside the whitelist: {ast.unparse(node.func)}")
                if any(isinstance(a, ast.Starred) for a in node.args) or any(kw.arg is None for kw in node.keywords):
                    note(node, "starred / double-starred call arguments")
            elif isinstance(node, ast.Attribute):
                if not (node.attr == "shape" and isinstance(node.value, ast.Name)) and not (
                        node.attr in _STATIC_ATTR_CALLS and isinstance(node.value, ast.Name)):
                    note(node, f"attribute access {ast.unparse(node)}")
            elif isinstance(node, ast.Subscript):
                base = node.value
                if isinstance(base, ast.Attribute) and base.attr == "shape":
                    if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
                        note(node, "shape index is not a literal")
                elif not isinstance(base, ast.Name):
                    note(node, f"subscript of a non-Name {ast.unparse(base)}")
    return out


def _is_subscript_index(node: ast.expr, stmts: list) -> bool:
    """A Tuple that is the slice of a Subscript (``X[m, k]``) is index syntax, not a display."""
    for s in stmts:
        for sub in ast.walk(s):
            if isinstance(sub, ast.Subscript) and sub.slice is node:
                return True
    return False


def body_bound_names(stmts: Iterable[ast.stmt]) -> Set[str]:
    """Every name bound ANYWHERE in a statement list by any binding form: plain / annotated / augmented assignment
    targets, walrus targets, ``for`` / ``with`` targets, ``except ... as``, ``del``, ``def`` / ``class`` names,
    ``import`` names, ``global`` / ``nonlocal`` declarations (the SIZE production's "never rebound" condition)."""
    out: Set[str] = set()
    for s in stmts:
        for sub in scope_walk(s):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    out.update(_target_names(tgt))
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
                out.update(_target_names(sub.target))
            elif isinstance(sub, (ast.With, ast.AsyncWith)):
                for item in sub.items:
                    if item.optional_vars is not None:
                        out.update(_target_names(item.optional_vars))
            elif isinstance(sub, ast.Delete):
                for tgt in sub.targets:
                    out.update(_target_names(tgt))
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name).split(".")[0] for a in sub.names)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                out.add(sub.name)
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                out.update(sub.names)
    return out


def freeze_state(value):
    """A hashable, VALUE-based image of an abstract-state component (dicts, sets, lists, tuples, scalars, nested
    arbitrarily): the fixpoint's convergence key is the frozen image of the WHOLE per-statement snapshot table, so
    every state component -- bindings, index names, sizes and anything a later change adds -- takes part in
    convergence (size-fixpoint-24); nothing is compared by identity.  A component that is none of these is
    frozen by ``repr`` as a last resort -- for an AST object that is its identity, NOT structural equality: any
    future expression state (e.g. the compute-dtype binding's alias operands) must be encoded structurally
    (tuples of names / unparsed texts / roles) BEFORE it enters a snapshot, so that two passes reaching the same
    value freeze identically (the review round-7 note)."""
    if isinstance(value, dict):
        return ("dict",) + tuple(sorted(((str(k), freeze_state(v)) for k, v in value.items()), key=repr))
    if isinstance(value, (set, frozenset)):
        return ("set",) + tuple(sorted((freeze_state(v) for v in value), key=repr))
    if isinstance(value, (list, tuple)):
        return ("seq",) + tuple(freeze_state(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, ast.AST):
        return ("ast", ast.dump(value))
    return repr(value)


def _validate_statement_slots(stmts: Iterable[ast.stmt], ctx: StatementContext, lv: Set[str]) -> list:
    """The post-fixpoint SLOT VALIDATION of EVERY statement of the body -- inside or outside loops, used or unused --
    against the bindings in force before it (outer-slots-25): a loop bound and an ``if`` / ``while`` test must
    be STATIC, an assigned value and a ``return`` expression must not be UNPROVEN, a subscript store goes through the
    index slot, an expression statement must be a literal (a docstring).  Returns the violations; non-empty means the
    whole kernel is unproven.  Runs on the CONVERGED snapshots, so a loop-carried name proven by the fixpoint is
    not refused by an early pass."""
    out: list = []

    def note(s: ast.stmt, what: str) -> None:
        msg = f"{what} at line {getattr(s, 'lineno', '?')}"
        if msg not in out:
            out.append(msg)

    def check(s: ast.stmt) -> None:
        ctx.select(s)
        try:
            if isinstance(s, ast.For):
                if provenance(s.iter, lv, ctx) != STATIC:
                    note(s, f"loop bound not proven static: {ast.unparse(s.iter)}")
            elif isinstance(s, (ast.If, ast.While)):
                if provenance(s.test, lv, ctx) != STATIC:
                    note(s, f"if test not proven static: {ast.unparse(s.test)}")
            elif isinstance(s, (ast.Assign, ast.AugAssign)):
                if provenance(s.value, lv, ctx) == UNPROVEN:
                    note(s, f"assigned value not proven: {ast.unparse(s.value)}")
                for tgt in (s.targets if isinstance(s, ast.Assign) else [s.target]):
                    if isinstance(tgt, ast.Subscript) and _store_index_unproven(tgt, lv, ctx):
                        note(s, f"store index not proven: {ast.unparse(tgt)}")
            elif isinstance(s, ast.Return):
                if s.value is not None and provenance(s.value, lv, ctx) == UNPROVEN:
                    note(s, f"return expression not proven: {ast.unparse(s.value)}")
            elif isinstance(s, ast.Expr):
                if provenance(s.value, lv, ctx) != STATIC:
                    note(s, f"expression statement not proven: {ast.unparse(s.value)}")
        finally:
            ctx.select(None)
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(s, field, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                for b in sub:
                    check(b)
        for h in getattr(s, "handlers", None) or []:
            for b in h.body:
                check(b)

    for s in stmts:
        check(s)
    return out


def build_context(stmts: Iterable[ast.stmt], tensor_names: Iterable[str] = (), scalar_names: Iterable[str] = (),
                  loop_vars: Optional[Iterable[str]] = None, module: Optional[ast.Module] = None,
                  int_params: Iterable[str] = (), input_ranks: Optional[dict] = None) -> StatementContext:
    """A :class:`StatementContext` for a statement list: the given Tensor /
    scalar names plus a FLOW-ORDERED walk of every binding form, recording
    the bindings in force before each statement (:meth:`StatementContext.select`).

    Rules: a plain / annotated / chained target takes the provenance of its
    value; a tuple / list / starred (unpacking) target re-derives every name
    as unproven; an augmented assignment makes its name a
    runtime scalar unless it already holds a tensor value; ``for`` targets
    are loop indices on every iteration (the body is walked twice so a body
    statement sees the previous trip's bindings); ``with`` targets, ``def`` /
    ``class`` / ``import`` names and deleted names are unproven; control-flow
    joins take the lattice join with UNPROVEN dominating, and a name bound on
    one side only is unproven.  A name bound to differing classes anywhere in
    the body is UNPROVEN at every use.

    *input_ranks* (``parameter -> declared rank``) names the kernel's declared
    INPUT tensors for the SINGLETON-SCALE-READ production; without it no
    literal-index read is a candidate."""
    if isinstance(stmts, list):
        normalize_statements(stmts)                                   # AnnAssign is Assign, in place, before anything reads the body
    ctx = StatementContext(tensor_names, scalar_names)
    ctx.module_info = module_bindings(module)
    ctx.module_bound = set(ctx.module_info)
    # SIZE-BINDING-23: an `int` parameter is a size source only if it is NEVER rebound anywhere in the body (any binding
    # form, before or after the use) and its annotation resolves to the builtin -- a module that binds `int` (or
    # star-imports) leaves every `int` annotation unproven
    stmts = list(stmts)
    rebound = body_bound_names(stmts)
    ctx.body_rebound = set(rebound)                                   # a singleton INPUT is never rebound anywhere (SINGLETON-REBIND-3)
    ctx.int_params = {p for p in int_params if p in ctx.scalar_names and p not in rebound}
    if "int" in ctx.module_bound or "*" in ctx.module_bound:
        ctx.int_params = set()
    seed_lv = set(loop_vars or ())
    # LITERAL-CONSTANT PRODUCTION (issue #61): once-bound module literals that are not parameters, not bound anywhere
    # in the kernel body (body_bound_names: the names a target BINDS -- a subscript store binds none, an earlier fix round 8),
    # not a seeded loop variable and not the name of a whitelisted callable
    shadowed = ctx.tensor_names | ctx.scalar_names | set(rebound) | seed_lv | ALLOWED_CALLEES
    ctx.literal_names = {n: k for n, k in module_literal_kinds(module).items() if n not in shadowed}
    ctx.block_divisors = module_literal_names(module, ints_only=True) - ctx.tensor_names - ctx.scalar_names
    # SINGLETON-SCALE-READ PRODUCTION (issue #61): the declared input tensors and their declared ranks
    ctx.input_ranks = {n: int(r) for n, r in (input_ranks or {}).items() if n in ctx.tensor_names}
    bindings: dict = {name: STATIC for name in seed_lv}
    index: Set[str] = set(seed_lv)
    # every binding class a name has taken: loop indices are STATIC, parameters are bindings too
    seen_classes: dict = {name: {STATIC} for name in seed_lv}
    for name in ctx.tensor_names:
        seen_classes.setdefault(name, set()).add(TENSOR_OBJ)
    for name in ctx.scalar_names:
        seen_classes.setdefault(name, set()).add(SCALAR)
    loop_end: dict = {}              # id(for) -> the state at the end of its body in the previous pass

    sizes: dict = {}
    alias_specs: dict = {}           # alias name -> frozen operand spec captured when the alias was bound

    fstat: Set[str] = set()          # FLOAT-DERIVED STATIC locals (issue #61): snapshot component 3, in the fixpoint key

    def ctx_view() -> StatementContext:
        ctx._bindings, ctx._index, ctx._sizes, ctx._alias_specs = bindings, index, sizes, alias_specs
        ctx._extra = (fstat, alias_specs)                                     # snapshot components 3 (#61 float-derived) and 4 (dtype alias records)
        return ctx

    def bind(name: str, prov: str, is_index: bool = False, assign_form: bool = True) -> None:
        bindings[name] = prov
        sizes.pop(name, None)                                             # every binding form invalidates the size view
        alias_specs.pop(name, None)               # every binding form invalidates the alias record
        fstat.discard(name)                                               # ... and the float-derived mark (re-derived below)
        if is_index:
            index.add(name)
        else:
            index.discard(name)
        ctx.bound_names.add(name)
        if assign_form:
            seen_classes.setdefault(name, set()).add(prov)

    def value_is_index(value: ast.expr, prov: str) -> bool:
        return prov == STATIC and any(isinstance(n, ast.Name) and n.id in index for n in ast.walk(value))

    def bind_target(target: ast.expr, value: Optional[ast.expr], prov: str) -> None:
        if isinstance(target, ast.Name):
            # EVALUATE the value against the INCOMING environment first -- its provenance (done by the caller), its SIZE
            # view, its float-derived state and its captured operand record -- THEN publish the new binding (which kills
            # the destination's previous size view, float-derived mark and alias record): a self-transfer `f = f`,
            # `f = f * 1`, `f = int(f)` keeps what the incoming value carries; a rebinding to an independent value does
            # not (ISSUE61-FLOAT-SELF-TRANSFER-5).  DTYPE-ALIAS-CAPTURE-4: the operands the value reads are resolved NOW,
            # against the environment in force at this definition (earlier aliases through their own frozen records),
            # and frozen; a later rebinding of a name the value read cannot reach a value captured here
            size_prov = provenance(value, index | seed_lv, ctx_view(), SLOT_SIZE) if value is not None else None
            float_derived = value is not None and prov == STATIC and _float_literal_inside(value, ctx_view())
            captured = _factor_specs(value, index | seed_lv, ctx_view()).frozen() if value is not None else None
            bind(target.id, prov, is_index=value is not None and value_is_index(value, prov))
            if value is not None:
                sizes[target.id] = size_prov                              # the value as a size
                if float_derived:
                    fstat.add(target.id)                                  # a STATIC alias of a float literal is float-derived
                alias_specs[target.id] = captured
        elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
            # unpacking is not a proven binding form: every target name is re-derived as unproven
            # (default-deny; a stale earlier binding never survives the write)
            for name in _target_names(target):
                bind(name, UNPROVEN)
            for name in _target_names(target):
                bind(name, UNPROVEN)
        # a subscript / attribute store binds no name

    def walrus_bindings(node: ast.AST) -> None:
        # a walrus is outside the grammar (the kernel refuses); its target is never a proven binding
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                bind(sub.target.id, UNPROVEN)

    def join_binding(a: str, b: str) -> str:
        """The join of two BINDING classes at a control-flow merge: equal classes stay (a tensor object
        reaching a merge from both sides is still that tensor object); value classes join through the
        value lattice; anything else is unproven."""
        if a == b:
            return a
        if a in (STATIC, SCALAR, TENSOR) and b in (STATIC, SCALAR, TENSOR):
            return _combine((a, b))
        return UNPROVEN

    def join_into(other_b: dict, other_i: Set[str], other_s: Optional[dict] = None, other_f: Optional[Set[str]] = None,
                  other_as: Optional[dict] = None) -> None:
        nonlocal bindings, index, sizes, fstat, alias_specs
        fstat = fstat | (other_f if other_f is not None else set())       # float-derived on either side: float-derived (fail-closed)
        merged = {}
        for name in set(bindings) | set(other_b):
            if name in bindings and name in other_b:
                merged[name] = join_binding(bindings[name], other_b[name])
            else:
                merged[name] = UNPROVEN
        bindings = merged
        index = index & other_i
        other_s = sizes if other_s is None else other_s
        sizes = {n: v for n, v in sizes.items() if other_s.get(n) == v}
        other_as = alias_specs if other_as is None else other_as
        alias_specs = {n: v for n, v in alias_specs.items() if other_as.get(n) == v}      # frozen records compare by value

    def snapshot(s: ast.stmt) -> None:
        ctx._at[id(s)] = (dict(bindings), set(index), dict(sizes), set(fstat), dict(alias_specs))

    def state() -> tuple:
        return (dict(bindings), set(index), dict(sizes), set(fstat), dict(alias_specs))

    def restore(st: tuple) -> None:
        nonlocal bindings, index, sizes, fstat, alias_specs
        bindings, index, sizes, fstat, alias_specs = dict(st[0]), set(st[1]), dict(st[2]), set(st[3]), dict(st[4])

    def visit(nodes) -> None:
        nonlocal bindings, index, sizes, fstat, alias_specs
        for s in nodes:
            snapshot(s)
            if isinstance(s, ast.Assign):
                prov = provenance(s.value, index | seed_lv, ctx_view())
                walrus_bindings(s.value)
                if (len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and _accumulated_rhs(s) is not None):
                    # ``acc = acc + ...``: the accumulator idiom written out (the unit contract's own form)
                    name = s.targets[0].id
                    prev = bindings.get(name)
                    val = provenance(_accumulated_rhs(s), index | seed_lv, ctx_view())
                    if prev in (None, STATIC, SCALAR) and val != UNPROVEN:
                        bind(name, SCALAR, assign_form=False)
                    elif prev == TENSOR and val != UNPROVEN:
                        bind(name, TENSOR, assign_form=False)
                    else:
                        bind(name, UNPROVEN, assign_form=False)
                    continue
                for t in s.targets:
                    bind_target(t, s.value, prov)
            elif isinstance(s, ast.AnnAssign):
                if s.value is not None:
                    walrus_bindings(s.value)
                    bind_target(s.target, s.value, provenance(s.value, index | seed_lv, ctx_view()))
                elif isinstance(s.target, ast.Name):
                    bind(s.target.id, UNPROVEN)
            elif isinstance(s, ast.AugAssign):
                walrus_bindings(s.value)
                if isinstance(s.target, ast.Name):
                    name = s.target.id
                    prev = bindings.get(name)
                    val = provenance(s.value, index | seed_lv, ctx_view())
                    if prev in (None, STATIC, SCALAR) and val != UNPROVEN:
                        bind(name, SCALAR, assign_form=False)               # an accumulator is a runtime scalar
                    elif prev == TENSOR and val != UNPROVEN:
                        bind(name, TENSOR, assign_form=False)
                    else:
                        bind(name, UNPROVEN, assign_form=False)
            elif isinstance(s, ast.For):
                walrus_bindings(s.iter)
                before = state()
                prev_end = loop_end.get(id(s))
                if prev_end is not None:
                    join_into(*prev_end)                                      # the previous trip's bindings (fixpoint)
                for name in _target_names(s.target):
                    bind(name, STATIC, is_index=True, assign_form=prev_end is None)
                visit(s.body)
                end = state()
                loop_end[id(s)] = end
                restore(before)
                join_into(*end)
                visit(s.orelse)
            elif isinstance(s, ast.If):
                walrus_bindings(s.test)
                before = state()
                visit(s.body)
                body_end = state()
                restore(before)
                visit(s.orelse)
                join_into(*body_end)
            elif isinstance(s, ast.While):
                walrus_bindings(s.test)
                before = state()
                visit(s.body)
                join_into(*before)
                visit(s.orelse)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            bind(name, UNPROVEN)
                visit(s.body)
            elif isinstance(s, ast.Try):
                before = state()
                visit(s.body)
                for h in s.handlers:
                    if h.name:
                        bind(h.name, UNPROVEN)
                    visit(h.body)
                join_into(*before)
                visit(s.orelse)
                visit(s.finalbody)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(s.name, UNPROVEN)                                        # a local callable shadows
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for alias in s.names:
                    bind((alias.asname or alias.name).split(".")[0], UNPROVEN)
            elif isinstance(s, ast.Delete):
                for t in s.targets:
                    for name in _target_names(t):
                        bind(name, UNPROVEN)
            elif isinstance(s, ast.Return):
                pass                                                          # validated by slot after the fixpoint
            else:
                for child in ast.iter_child_nodes(s):
                    if isinstance(child, ast.expr):
                        walrus_bindings(child)
                for f in ("body", "orelse"):
                    sub = getattr(s, f, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        visit(sub)

    # the REAL fixpoint: iterate the flow-ordered pass until the per-statement snapshots stop changing,
    # bounded by the number of bindings + 2; a run that does not converge is unproven.  The key is the frozen
    # image of the WHOLE snapshot table (every state component, by value): nothing is hand-picked.
    def snapshot_key() -> tuple:
        # the whole per-statement snapshot table AND the production candidate set found so far (issue #61: the
        # SINGLETON-SCALE-READ candidates are state the passes accumulate; the LITERAL-CONSTANT set is fixed above)
        return freeze_state((ctx._at, ctx.singleton_reads))

    n_bindings = sum(1 for s in stmts for sub in ast.walk(s)
                     if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.NamedExpr)))
    bound = n_bindings + 2
    previous = None
    ctx.converged = False
    for _pass in range(bound):
        bindings = {name: STATIC for name in seed_lv}
        index = set(seed_lv)
        sizes = {}
        alias_specs = {}
        fstat = set()
        ctx._at = {}
        visit(stmts)
        key = snapshot_key()
        if key == previous:
            ctx.converged = True
            break
        previous = key
    ctx.aliases = dict(bindings)
    ctx.index_names = set(index)
    ctx.sizes = dict(sizes)
    ctx.alias_specs = dict(alias_specs)
    ctx.float_static = set(fstat)
    ctx.conflicting = {name for name, classes in seen_classes.items() if len(classes) > 1}
    ctx._bindings, ctx._index, ctx._sizes, ctx._extra, ctx._alias_specs = None, None, None, (), None
    if not ctx.converged:
        ctx.violations.append(f"binding analysis did not converge within {bound} passes")
    ctx.violations.extend(_validate_statement_slots(stmts, ctx, set(seed_lv)))   # every statement, every slot
    return ctx


def _tensor_names_in(node: ast.expr, lv: Set[str], ctx: Optional[StatementContext]) -> list:
    """The tensor NAMES *node* reads -- a subscripted tensor parameter, the object a tensor-object alias names, or
    the names an alias captured when it was defined -- in source order without repeats (the operand listing of
    the compute-dtype binding)."""
    out: list = []

    def add(name: str) -> None:
        if name not in out:
            out.append(name)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id not in lv:
            spec = ctx.alias_spec(sub.id) if ctx is not None else None
            if spec is not None:
                for name in _Specs.from_frozen(spec).names:
                    add(name)
            elif ctx is not None and ctx.name_provenance(sub.id) == TENSOR_OBJ and sub.id in ctx.tensor_names:
                add(sub.id)
    if ctx is None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id not in lv:
                add(sub.value.id)
    return out


_SUPPORTED_CONVERSIONS = frozenset({"fp32"})     # the DSL's explicit conversion builtins (einsum_lib.fp32)


class _Specs:
    """The operand OCCURRENCES of one MAC product and their roles (DSL-COMPUTE-DTYPE-BINDING-1 rounds 2-3).

    ``accesses`` is one entry per tensor-element factor occurrence, in source order, as ``(tensor name, index facts)``
    where the index facts are ``(text, block_left, plain_loop_name)`` per index element, resolved where the factor
    was met (for an alias: at its defining assignment); ``conversions`` one entry per explicit conversion factor
    occurrence; ``names`` every tensor name read (deduplicated); ``object`` the tensor a tensor-object alias names.
    :meth:`assign_roles` turns the accesses into ``occurrences`` (name, index text, role) and the deduplicated
    ``payloads`` / ``scales`` summaries.  Occurrences, never the deduplicated names, carry the factor cardinality
    (DTYPE-ROLE-MULTIPLICITY-7); :meth:`frozen` is the value-comparable record an alias stores
    (DTYPE-ALIAS-CAPTURE-4)."""

    def __init__(self) -> None:
        self.accesses: list = []       # [(name, facts)] per tensor-element factor occurrence
        self.conversions: list = []    # one target per explicit conversion factor occurrence
        self.names: list = []          # every tensor name read (operand listing, deduplicated)
        self.object: Optional[str] = None   # the tensor object a tensor-object alias names (``Q = W``)
        self.unproven = False          # a payload whose declared dtype cannot be established
        self.arms_differ = False       # a conditional whose arms contract different operands
        self.occurrences: list = []    # after assign_roles: [{"name", "index", "role"}] per access
        self.payloads: list = []       # deduplicated payload names (descriptive summary)
        self.scales: list = []         # deduplicated scale names (descriptive summary)

    def add_name(self, name: str) -> None:
        if name not in self.names:
            self.names.append(name)

    def merge(self, other: "_Specs") -> None:
        self.accesses.extend(other.accesses)
        self.conversions.extend(other.conversions)
        for n in other.names:
            self.add_name(n)
        if other.object is not None:
            if self.object is not None and self.object != other.object:
                self.unproven = True
            self.object = other.object
        self.unproven |= other.unproven
        self.arms_differ |= other.arms_differ

    def frozen(self) -> tuple:
        return ("spec", tuple((n, tuple(f)) for n, f in self.accesses), tuple(self.conversions), tuple(self.names),
                self.object, self.unproven, self.arms_differ)

    @classmethod
    def from_frozen(cls, rec: tuple) -> "_Specs":
        s = cls()
        _tag, accesses, conversions, names, obj, unproven, arms = rec
        s.accesses = [(n, tuple(f)) for n, f in accesses]
        s.conversions = list(conversions)
        s.names = list(names)
        s.object = obj
        s.unproven = bool(unproven)
        s.arms_differ = bool(arms)
        return s

    def assign_roles(self) -> None:
        """The block-scale rule over the product's accesses: an access S is a SCALE of a payload P when S has at
        least one index ``i // <static>`` with ``i`` a loop index that P indexes plainly, and every other index of
        S is identical to one of P's.  A floor-divided index without such a payload (the GQA head-group index
        ``h // _G`` of a K/V operand) is a payload access, not a scale.  Decided from the facts recorded where each
        access was met, one role per OCCURRENCE."""
        self.occurrences, self.payloads, self.scales = [], [], []
        for name, facts in self.accesses:
            is_scale = False
            if any(left for _t, left, _p in facts):
                for pname, pfacts in self.accesses:
                    if pname == name:
                        continue
                    ptexts = {t for t, _l, _p in pfacts}
                    pplain = {p for _t, _l, p in pfacts if p}
                    if (any(left and left in pplain and t not in ptexts for t, left, _p in facts)
                            and all((t in ptexts) or (left and left in pplain) for t, left, _p in facts)):
                        is_scale = True
                        break
            role = "scale" if is_scale else "payload"
            self.occurrences.append({"name": name, "index": ", ".join(t for t, _l, _p in facts), "role": role})
            lst = self.scales if is_scale else self.payloads
            if name not in lst:
                lst.append(name)

    def key(self) -> tuple:
        return (tuple((o["name"], o["index"], o["role"]) for o in self.occurrences), tuple(self.conversions),
                self.unproven, self.arms_differ)

    def as_dict(self) -> dict:
        return {"occurrences": [dict(o) for o in self.occurrences], "payloads": list(self.payloads), "scales": list(self.scales),
                "conversions": list(self.conversions), "names": list(self.names), "unproven": self.unproven,
                "arms_differ": self.arms_differ}


def _index_facts(slice_node: ast.expr, lv: Set[str], ctx: Optional[StatementContext]) -> tuple:
    """Per index element of a tensor access: ``(text, block_left, plain_loop_name)`` -- the loop index a block
    division ``i // c`` floor-divides (when the element IS the block-index production in the index slot) and
    the loop index a plain Name element is; resolved against the environment where the access is met."""
    facts = []
    for e in _slice_elements(slice_node):
        block_left = None
        if (isinstance(e, ast.BinOp) and isinstance(e.op, ast.FloorDiv) and isinstance(e.left, ast.Name) and e.left.id in lv
                and provenance(e, lv, ctx, SLOT_INDEX) == STATIC):
            block_left = e.left.id
        plain = e.id if isinstance(e, ast.Name) and e.id in lv else None
        facts.append((ast.unparse(e), block_left, plain))
    return tuple(facts)


def _factor_specs(node: ast.expr, lv: Set[str], ctx: Optional[StatementContext], depth: int = 0) -> _Specs:
    """The operand occurrences of one value factor: a tensor NAME subscripted is one ACCESS (its role -- payload or
    block scale -- is decided over the whole product); an explicit ``fp32(...)`` is one CONVERSION occurrence (its
    operands only listed); a local alias contributes the record it CAPTURED AT ITS DEFINITION; a tensor-object
    alias (``Q = W``) names its object; a product alias contributes its factors' occurrences.  Anything else that
    reaches the contraction (another call or cast, arithmetic other than a product, an operand selection, a
    runtime scalar, an unresolvable name, a subscript of a value alias) leaves the payload dtype UNPROVEN."""
    s = _Specs()
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    if depth > 16:
        s.unproven = True
        return s
    for name in _tensor_names_in(node, lv, ctx):
        s.add_name(name)
    if isinstance(node, ast.Call) and _callee_is(node, _SUPPORTED_CONVERSIONS, ctx) and isinstance(node.func, ast.Name):
        s.conversions.append(node.func.id)
        return s
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        base = node.value.id
        spec = ctx.alias_spec(base) if ctx is not None else None
        if spec is not None:
            obj = _Specs.from_frozen(spec).object
            if obj is None:
                s.unproven = True                                       # a subscript of a VALUE alias: not a tensor object
                return s
            source = obj
        elif ctx is None or (ctx.name_provenance(base) == TENSOR_OBJ and base in ctx.tensor_names):
            source = base
        else:
            s.unproven = True
            return s
        s.accesses.append((source, _index_facts(node.slice, lv, ctx)))
        s.add_name(source)
        return s
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        for factor, _den in _product_factors(node):
            prov = provenance(factor, lv, ctx)
            if prov == STATIC:
                continue
            s.merge(_factor_specs(factor, lv, ctx, depth + 1))
        return s
    if isinstance(node, ast.Name) and ctx is not None:
        spec = ctx.alias_spec(node.id)
        if spec is not None and node.id not in ctx.conflicting:
            s.merge(_Specs.from_frozen(spec))                           # the record frozen when the alias was defined
            return s
        if ctx.name_provenance(node.id) == TENSOR_OBJ and node.id in ctx.tensor_names and node.id not in ctx.conflicting:
            s.object = node.id                                          # a tensor-object alias names its object
            s.add_name(node.id)
            return s
    if isinstance(node, ast.IfExp):
        body = _factor_specs(node.body, lv, ctx, depth + 1)
        orelse = _factor_specs(node.orelse, lv, ctx, depth + 1)
        s.merge(body)
        s.merge(orelse)
        if body.frozen() != orelse.frozen():
            s.arms_differ = True
        return s
    s.unproven = True
    return s


def mac_operand_specs(stmt: ast.stmt, loop_vars: Optional[Iterable[str]] = None,
                      ctx: Optional[StatementContext] = None) -> Optional[dict]:
    """The operands a MAC statement contracts: ``{"occurrences": [{name, index, role}] (one per tensor-element
    factor occurrence), "conversions": [target per conversion occurrence], "payloads", "scales" (deduplicated
    summaries), "names", "unproven", "arms_differ"}`` (see :class:`_Specs`, :func:`_factor_specs`).  The converter
    binds every occurrence to the kernel input's position and declared dtype and emits the receipt beside an
    independent occurrence channel; the analyzer reconciles the two and proves the compute dtype only when every
    payload occurrence and conversion agrees.  ``None`` for a statement that is not a MAC."""
    lv = set(loop_vars or ())
    rhs = _accumulated_rhs(stmt)
    if rhs is None:
        return None
    if ctx is not None:
        ctx.select(stmt)
    try:
        arms: list = []

        def visit(node: ast.expr) -> None:
            while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                node = node.operand
            if isinstance(node, ast.IfExp):
                visit(node.body)
                visit(node.orelse)
                return
            kind, _extra, _flags = _classify_product(node, lv, ctx)
            if kind != MAC:
                return
            s = _Specs()
            for factor, den in _product_factors(node):
                prov = provenance(factor, lv, ctx)
                if not _is_value_factor(prov, den):
                    continue
                if prov == SCALAR:
                    s.unproven = True                       # a runtime scalar has no declared dtype
                    continue
                s.merge(_factor_specs(factor, lv, ctx))
            s.assign_roles()                                 # payload / block-scale roles over the whole product
            arms.append(s)

        visit(rhs)
        if not arms:
            return None
        out = arms[0]
        for s in arms[1:]:
            out.merge(s)
            out.occurrences = out.occurrences + s.occurrences
            for n in s.payloads:
                if n not in out.payloads:
                    out.payloads.append(n)
            for n in s.scales:
                if n not in out.scales:
                    out.scales.append(n)
        if len({s.key() for s in arms}) > 1:
            out.arms_differ = True                          # conditional arms contract different operands: never union
        return out.as_dict()
    finally:
        if ctx is not None:
            ctx.select(None)


def mac_operand_names(stmt: ast.stmt, loop_vars: Optional[Iterable[str]] = None,
                      ctx: Optional[StatementContext] = None) -> list:
    """The tensor names a MAC statement multiplies (payloads and scales, in
    source order without repeats; both arms of a conditional); ``[]`` for a
    statement that is not a MAC."""
    specs = mac_operand_specs(stmt, loop_vars, ctx)
    if not specs:
        return []
    names: list = []
    for n in specs["payloads"] + specs["scales"] + specs["names"]:
        if n not in names:
            names.append(n)
    return names


def classify_source_statement(text: str, loop_vars: Optional[Iterable[str]] = None):
    """Classify one statement given as source text (``SyntaxError`` -> OTHER)."""
    try:
        mod = ast.parse(text.strip(), mode="exec")
    except (SyntaxError, ValueError):
        return OTHER, 0
    if len(mod.body) != 1:
        return OTHER, 0
    return classify_statement(mod.body[0], loop_vars)


def classify_source_records(text: str, loop_vars: Optional[Iterable[str]] = None,
                            tensor_names: Optional[Iterable[str]] = None,
                            scalar_names: Iterable[str] = (), module: Optional[ast.Module] = None) -> Tuple[list, Set[str]]:
    """Classify every assignment of a statement list given as source text,
    with a context built from *tensor_names* / *scalar_names* and the text's
    own local aliases.  Returns ``(records, flags)``."""
    mod = ast.parse(text.strip(), mode="exec")
    ctx = build_context(mod.body, tensor_names or (), scalar_names, loop_vars, module=module) if tensor_names is not None else None
    records: list = []
    flags: Set[str] = set()
    if ctx is not None:
        ctx.violations.extend(grammar_violations(mod.body, ctx))
        if ctx.violations:
            flags.add(UNPROVEN_FLAG)                                  # outside the supported grammar: the kernel is unproven
    for stmt in mod.body:
        if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            r, f = classify_records(stmt, loop_vars, ctx)
            records.extend(r)
            flags |= f
        elif isinstance(stmt, ast.If) and ctx is not None:
            ctx.select(stmt)
            if provenance(stmt.test, set(loop_vars or ()), ctx) != STATIC:
                flags.add(UNPROVEN_FLAG)
            ctx.select(None)
            for sub in stmt.body + stmt.orelse:
                if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    r, f = classify_records(sub, loop_vars, ctx)
                    records.extend(r)
                    flags |= f
    return records, flags


__all__ = ["MAC", "OTHER", "STATIC", "SCALAR", "TENSOR", "UNPROVEN", "VALUE_CLASSES", "UNPROVEN_FLAG", "BRANCH_FLAG",
           "StatementContext", "build_context", "classify_records", "classify_statement",
           "classify_source_statement", "classify_source_records", "grammar_violations", "module_bindings", "module_bound_names",
           "module_literal_names", "module_literal_kinds", "ALLOWED_CALLEES", "SLOT_VALUE", "SLOT_INDEX", "SLOT_SIZE", "SLOT_DIVISOR",
           "LITERAL_INT", "LITERAL_FLOAT", "provenance", "value_factor_count", "normalize_statements", "freeze_state", "body_bound_names",
           "definition_time_expressions", "scope_walk", "enclosing_scope_walrus_targets", "mac_operand_names", "mac_operand_specs"]
