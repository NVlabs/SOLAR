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

"""Base classes for einsum operation handlers.

This module defines the core data structures and abstract base class
for all einsum operation handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, ClassVar
import logging

import re

from solar.common.types import TensorShape, TensorShapes
from solar.common.utils import validate_einsum_ranks_match_shapes

logger = logging.getLogger(__name__)


@dataclass
class EinsumOperand:
    """Represents an operand in an einsum operation."""
    name: str
    dims: List[str]
    is_output: bool = False
    stride: Optional[Dict[str, int]] = None
    dilation: Optional[Dict[str, int]] = None

    def to_timeloop_dataspace(self) -> Dict[str, Any]:
        """Convert to timeloop dataspace format."""
        dataspace = {
            'name': self.name,
            'projection': self.dims,
        }
        if self.is_output:
            dataspace['read_write'] = 'true'
        return dataspace


@dataclass
class EinsumOp:
    """Represents an einsum operation.

    The extended einsum representation supports different elementwise and reduction
    operations beyond the standard multiply-add semantics:

    - elementwise_op: The operation applied element-wise (default: 'mul')
      Examples: 'mul' for matmul, 'add' for element-wise add, 'max' for max pooling
    - reduction_op: The operation used to reduce/aggregate (default: 'add')
      Examples: 'add' for sum, 'max' for max reduction, 'none' for no reduction
    - is_real_einsum: True if this is a standard tensor contraction (mul+add)
    - is_einsum_supportable: True if the operation can be expressed with extended einsum

    Standard einsum (matmul): elementwise_op='mul', reduction_op='add'
    Element-wise add:        elementwise_op='add', reduction_op='none'
    Max pooling:             elementwise_op='copy', reduction_op='max'
    """
    operands: List[EinsumOperand]
    equation: str
    name: str
    is_real_einsum: bool = True
    elementwise_op: str = "mul"  # 'mul', 'add', 'sub', 'div', 'max', 'min', 'copy'
    reduction_op: str = "add"    # 'add', 'max', 'min', 'mul', 'none'
    is_einsum_supportable: bool = True  # Can this op be expressed with extended einsum?
    # Exact MAC count supplied by a handler whose single equation is only a
    # summary of a multi-GEMM operation (recurrent layers: the time loop, the
    # layer stack and both directions are not expressible as one contraction
    # over the recorded operands).  None means: derive the cost from the
    # equation as usual.
    compute_cost_override: Optional[int] = None

    @property
    def input_operands(self) -> List[EinsumOperand]:
        """Get input operands."""
        return [op for op in self.operands if not op.is_output]

    @property
    def output_operands(self) -> List[EinsumOperand]:
        """Get output operands."""
        return [op for op in self.operands if op.is_output]

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        """Calculate compute cost from einsum rank dimensions.

        Collects unique rank dimension sizes from ALL operands (input + output).
        Compound dims like 'P+R' are split into atoms and resolved from other
        operands. No op-specific special cases — purely driven by einsum equation.

        Total cost = product of all unique resolved rank dimension sizes.
        A handler-supplied ``compute_cost_override`` takes precedence (see
        the recurrent handlers in misc_ops).
        """
        if self.compute_cost_override is not None:
            return int(self.compute_cost_override)
        return compute_cost_from_equation(self.equation, tensor_shapes)

    def to_torch_einsum(self, tensor_names: Optional[List[str]] = None) -> str:
        """Convert to torch.einsum format."""
        input_operands = self.input_operands

        if tensor_names is None:
            tensor_names = [op.name for op in input_operands]
        elif len(tensor_names) != len(input_operands):
            raise ValueError(
                f"Number of tensor names ({len(tensor_names)}) must match "
                f"number of input operands ({len(input_operands)})"
            )

        equation_str = f"'{self.equation}'"
        tensor_args = ', '.join(tensor_names)
        return f"torch.einsum({equation_str}, {tensor_args})"


def _parse_dim_atoms(dim: str) -> List[str]:
    """Parse a possibly compound dim into atomic rank names.

    'P+R' -> ['P', 'R']
    'B'   -> ['B']
    'P+R0' -> ['P', 'R0']
    """
    return [d.strip() for d in re.split(r'[+\-]', dim) if d.strip()]


def _parse_equation_operand(operand: str) -> List[str]:
    """Parse one einsum operand into rank tokens.

    Supports single-letter ranks with optional digits and parenthesized
    compound ranks, e.g. ``BGI(P+R)`` -> ``["B", "G", "I", "P+R"]``.
    """
    tokens: List[str] = []
    i = 0
    while i < len(operand):
        if operand[i] == "(":
            j = operand.index(")", i)
            tokens.append(operand[i + 1 : j])
            i = j + 1
        elif operand[i].isalpha():
            token = operand[i]
            i += 1
            while i < len(operand) and operand[i].isdigit():
                token += operand[i]
                i += 1
            tokens.append(token)
        else:
            i += 1
    return tokens


def _output_element_count(tensor_shapes: TensorShapes) -> Optional[int]:
    """Total element count of the output shapes, or None if any dim is unresolved."""
    outputs = list(tensor_shapes.outputs or [])
    if not outputs:
        return None
    total = 0
    for shape in outputs:
        numel = _resolved_numel(shape)
        if numel is None:
            return None
        total += numel
    return int(total)


def _resolved_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, (list, tuple)):
        return None
    numel = 1
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            return None
        numel *= dim
    return int(numel)


def _broadcast_aligns(in_shape: Any, out_shape: Any) -> bool:
    """NumPy/PyTorch right-aligned broadcast of ``in_shape`` onto ``out_shape``."""
    if not isinstance(in_shape, (list, tuple)) or not isinstance(out_shape, (list, tuple)):
        return False
    if len(in_shape) > len(out_shape):
        return False
    for i_dim, o_dim in zip(reversed(list(in_shape)), reversed(list(out_shape))):
        if i_dim != o_dim and i_dim != 1:
            return False
    return True


# Factory / fill ops: the output is created from a size argument, so the
# "input" (``self`` for ``x.new_ones(size)``) does not align with the output
# at all; the work is one fill per output element.
_FACTORY_FILL_OPS = frozenset({
    "new_ones", "new_zeros", "new_full", "new_empty", "zeros_like", "ones_like",
    "full_like", "empty_like", "zeros", "ones", "full", "empty", "arange",
    "linspace", "eye", "rand", "randn", "rand_like", "randn_like", "randint",
    "randint_like",
})


class ElementwiseSelectEinsumOp(EinsumOp):
    """Generic fallback op for an unhandled op with no reduction (copy / select / view).

    The analyzer emits it when no handler is registered: ``is_real_einsum``
    False, elementwise ``copy``, reduction ``none`` and, when the rank
    changes, distinct output labels (``AB->O0O1O2O3O4``).  Its cost is a
    defined SELECTION cost: one operation per OUTPUT element, provided every
    input is a right-aligned broadcast of the output (a [64, 64] mask selected
    into [2048, 8, 2, 64, 64]) or has the same element count (a view such as
    an in-place ``unsqueeze_``).  The rank product used by handled ops would
    multiply the input labels into the output here: Mamba2's ``where`` was
    billed 64*64 = 4096x its 134,217,728 outputs (549,755,813,888 other_ops).
    Two further selection shapes are defined the same way (BigBird sidecar):
    a MULTI-OUTPUT PARTITION of one input (``unbind`` / ``chunk`` / ``split``:
    the outputs' element counts sum to the input's) costs one op per output
    element, and a FACTORY FILL (``new_ones``, ``zeros_like``, ...; the input
    is only the size/dtype donor) costs one op per output element.  When the
    inputs neither align nor match the output's element count, or an output
    dim is unresolved, the rank product is kept unchanged.

    This is a fallback, not a handler-provided MAC model, so the analyzer's
    handler precedence does not apply to it (``COST_TAKES_PRECEDENCE``).
    """

    COST_TAKES_PRECEDENCE: ClassVar[bool] = False

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        if self.is_real_einsum or self.reduction_op != "none":
            return compute_cost_from_equation(self.equation, tensor_shapes)
        outputs = list(tensor_shapes.outputs or [])
        output_numel = _output_element_count(tensor_shapes)
        inputs = list(tensor_shapes.inputs or [])
        if output_numel is None or not outputs or not inputs:
            return compute_cost_from_equation(self.equation, tensor_shapes)
        if str(self.name).lower() in _FACTORY_FILL_OPS:
            return output_numel
        if len(outputs) != 1:
            # multi-output partition of a single input (unbind/chunk/split)
            if len(inputs) == 1 and _resolved_numel(inputs[0]) == output_numel:
                return output_numel
            return compute_cost_from_equation(self.equation, tensor_shapes)
        out_shape = outputs[0]
        for in_shape in inputs:
            if in_shape is None:
                return compute_cost_from_equation(self.equation, tensor_shapes)
            if _broadcast_aligns(in_shape, out_shape):
                continue
            if _resolved_numel(in_shape) == output_numel:
                continue
            return compute_cost_from_equation(self.equation, tensor_shapes)
        return output_numel


class EinsumOperandEvidenceError(ValueError):
    """The equation's operands cannot be bound to the recorded shapes."""


def _equation_input_tokens(equation: str) -> Tuple[List[List[str]], List[str]]:
    lhs, rhs = equation.split("->", 1)
    return [_parse_equation_operand(o) for o in lhs.split(",")], _parse_equation_operand(rhs)


def resolve_equation_extents(equation: str, tensor_shapes: TensorShapes) -> Tuple[Dict[str, int], List[str]]:
    """Bind every single-atom rank of ``equation`` to its RESOLVED extent.

    Returns ``(extents, problems)``.  Binding rules (P2-2 of that review):

    * across operands a size-1 dim broadcasts against a non-singleton extent;
      the resolved extent is the non-singleton one and every later operand is
      compared to THAT, so ``i,i,i->i`` with [1],[2],[3] reports 2 vs 3;
    * a rank repeated WITHIN one operand (``ii``, a diagonal) requires equal
      extents, not broadcasting (``[1, 3]`` is a problem);
    * the output's extent must equal the resolved extent (a size-1 output of
      a resolved non-singleton rank is a problem too);
    * operand cardinality (fewer shapes than operands; extra trailing shapes
      such as a bias slot are allowed) and rank-count mismatches are problems.
    """
    extents: Dict[str, int] = {}
    problems: List[str] = []
    if not equation or "->" not in equation:
        return extents, problems
    input_tokens, output_tokens = _equation_input_tokens(equation)
    shapes = list(tensor_shapes.inputs or [])
    if len(shapes) < len(input_tokens):
        problems.append(
            f"operand cardinality: {len(shapes)} input shape(s) for {len(input_tokens)} equation operands")
    bound_at: Dict[str, str] = {}

    def _merge(atom: str, dim: int, where: str) -> None:
        prev = extents.get(atom)
        if prev is None:
            extents[atom] = dim
            bound_at[atom] = where
        elif prev == dim:
            return
        elif prev == 1:
            extents[atom] = dim          # broadcast 1 resolved to the non-singleton extent
            bound_at[atom] = where
        elif dim == 1:
            return                       # this operand broadcasts
        else:
            problems.append(f"rank {atom}: {prev} ({bound_at[atom]}) vs {dim} ({where})")

    for idx, tokens in enumerate(input_tokens):
        if idx >= len(shapes) or shapes[idx] is None:
            continue
        shape = shapes[idx]
        where = f"operand {idx}"
        if len(tokens) != len(shape):
            problems.append(f"{where}: {len(tokens)} label(s) {''.join(tokens)} vs shape {list(shape)}")
            continue
        seen_here: Dict[str, int] = {}
        for tok, dim in zip(tokens, shape):
            atoms = _parse_dim_atoms(tok)
            if len(atoms) != 1 or not isinstance(dim, int) or isinstance(dim, bool):
                continue
            atom = atoms[0]
            dim = int(dim)
            if atom in seen_here:
                if seen_here[atom] != dim:   # diagonal: equal extents required, no broadcasting
                    problems.append(
                        f"rank {atom}: diagonal extents {seen_here[atom]} vs {dim} within {where}")
                continue
            seen_here[atom] = dim
            _merge(atom, dim, where)
    outputs = list(tensor_shapes.outputs or [])
    if outputs and outputs[0] is not None:
        shape = outputs[0]
        if len(output_tokens) != len(shape):
            problems.append(f"output: {len(output_tokens)} label(s) {''.join(output_tokens)} vs shape {list(shape)}")
        else:
            for tok, dim in zip(output_tokens, shape):
                atoms = _parse_dim_atoms(tok)
                if len(atoms) != 1 or not isinstance(dim, int) or isinstance(dim, bool):
                    continue
                atom = atoms[0]
                prev = extents.get(atom)
                if prev is None:
                    extents[atom] = int(dim)
                    bound_at[atom] = "output"
                elif prev != int(dim):
                    problems.append(f"output rank {atom}: {int(dim)} vs resolved {prev} ({bound_at[atom]})")
    return extents, problems


def validate_equation_operands(equation: str, tensor_shapes: TensorShapes) -> List[str]:
    """Problems binding an equation's operands to positional shapes (empty = consistent).

    See :func:`resolve_equation_extents` for the rules.  The saved Mamba2 SSD
    einsum (F6) carried only the last two of its four operand shapes, bound
    positionally to the first two terms: 2 shapes for 4 operands, and C = 8 vs
    2.  ``compute_cost_from_equation`` binds first-wins and silently returns a
    number for such evidence; callers that need the evidence to be complete
    check this list first.
    """
    return resolve_equation_extents(equation, tensor_shapes)[1]


def optimal_dense_binary_schedule(
    equation: str,
    tensor_shapes: TensorShapes,
    max_operands: int = 8,
) -> Optional[Dict[str, Any]]:
    """Cheapest ordinary dense binary contraction tree for a multi-operand einsum.

    Exhaustive subset dynamic programme over every binary tree of the input
    operands (no reassociation beyond pairing, no sparsity, no reuse across
    operations).  Each pairwise step forms the product over the union of the
    two operands' indices and keeps the indices still needed by the output or
    by a later operand; a step that sums out at least one index costs that
    many MACs, a step that sums nothing is a scalar (Hadamard / broadcast)
    multiply of that many elements.  The objective is ``2 * MACs + scalar
    multiplies`` (one MAC = two flops).  Returns None when the equation has
    fewer than three operands, a compound rank, an unbound index or more than
    ``max_operands`` operands.  This is an OPTIMIZED-WORK ESTIMATE within a
    stated class, not a universal lower bound (maintainers' ruling); the as-written
    single-contraction estimate is reported beside it, and neither replaces
    the other in code.
    """
    if not equation or "->" not in equation:
        return None
    input_tokens, output_tokens = _equation_input_tokens(equation)
    n = len(input_tokens)
    if n < 3 or n > max_operands:
        return None
    extents, problems = resolve_equation_extents(equation, tensor_shapes)
    if problems:
        return None   # unsupported / contradictory evidence: no schedule is labelled valid
    shapes = list(tensor_shapes.inputs)
    index_sets: List[frozenset] = []
    for idx, tokens in enumerate(input_tokens):
        labels: List[str] = []
        for tok, dim in zip(tokens, shapes[idx]):
            atoms = _parse_dim_atoms(tok)
            if len(atoms) != 1:
                return None
            labels.append(atoms[0])
        index_sets.append(frozenset(labels))
    out_labels: List[str] = []
    for tok in output_tokens:
        atoms = _parse_dim_atoms(tok)
        if len(atoms) != 1:
            return None
        out_labels.append(atoms[0])
    if any(a not in extents for a in out_labels):
        return None
    out_set = frozenset(out_labels)
    # Physical per-operand extents: a leaf's own shape (singletons stay 1).
    # The GLOBAL resolved extents above validate compatibility; each binary
    # step resolves broadcasting from its two operands' own extents, so an
    # intermediate of two singletons is 1 element and [1],[1],[3] costs
    # 1 + 3 = 4 scalar multiplies, not 6 (round-2 P2-2).
    leaf_extents: List[Dict[str, int]] = []
    for idx, tokens in enumerate(input_tokens):
        ext: Dict[str, int] = {}
        for tok, dim in zip(tokens, shapes[idx]):
            ext[_parse_dim_atoms(tok)[0]] = int(dim)
        leaf_extents.append(ext)

    def _broadcast(ea: Dict[str, int], eb: Dict[str, int]) -> Dict[str, int]:
        return {a: max(ea.get(a, 1), eb.get(a, 1)) for a in set(ea) | set(eb)}

    def _size(ext: Dict[str, int]) -> int:
        total = 1
        for v in ext.values():
            total *= v
        return total

    full = (1 << n) - 1
    # best[mask] = (objective, macs, scalar, kept_extents, description)
    best: Dict[int, Tuple[int, int, int, Dict[str, int], str]] = {}
    for i in range(n):
        best[1 << i] = (0, 0, 0, leaf_extents[i], f"In{i}")
    for mask in range(1, full + 1):
        if mask & (mask - 1) == 0:
            continue
        outside = frozenset().union(*(index_sets[i] for i in range(n) if not mask & (1 << i))) if mask != full else frozenset()
        needed = out_set | outside
        sub = (mask - 1) & mask
        cand: Optional[Tuple[int, int, int, Dict[str, int], str]] = None
        while sub:
            rest = mask ^ sub
            if sub < rest:  # unordered split
                a, b = best[sub], best[rest]
                product = _broadcast(a[3], b[3])        # the step's own broadcast extents
                kept = {k: v for k, v in product.items() if k in needed}
                size = _size(product)
                summed = frozenset(product) - frozenset(kept)
                macs = a[1] + b[1] + (size if summed else 0)
                scalar = a[2] + b[2] + (0 if summed else size)
                obj = 2 * macs + scalar
                step = (f"({a[4]} x {b[4]}: " + (f"sum over {''.join(sorted(summed))}, {size} MACs" if summed else f"scale, {size} mults") + ")")
                if cand is None or obj < cand[0]:
                    cand = (obj, macs, scalar, kept, step)
            sub = (sub - 1) & mask
        assert cand is not None
        best[mask] = cand
    obj, macs, scalar, kept, desc = best[full]
    return {
        "macs": int(macs),
        "scalar_multiplies": int(scalar),
        "objective_2macs_plus_scalar": int(obj),
        "schedule": desc,
        "num_operands": n,
    }


def compute_cost_from_equation(equation: str, tensor_shapes: TensorShapes) -> int:
    """Calculate compute cost from an einsum equation and tensor shapes.

    The cost model is the product of every unique rank used by the equation.
    Compound ranks like ``P+R`` are split into atoms; kernel atoms such as
    ``R`` are resolved from concrete input operands, and output-position atoms
    such as ``P`` are resolved from the output shapes.  A rank that is size 1
    in one operand and larger in another is a broadcast rank and takes the
    larger extent regardless of operand order.
    """
    if not equation or "->" not in equation:
        return 0

    lhs, rhs = equation.split("->", 1)
    input_tokens = [_parse_equation_operand(operand) for operand in lhs.split(",")]
    output_tokens = _parse_equation_operand(rhs)

    all_ranks: Dict[str, Optional[int]] = {}
    for tokens in input_tokens:
        for token in tokens:
            for atom in _parse_dim_atoms(token):
                all_ranks.setdefault(atom, None)
    for token in output_tokens:
        for atom in _parse_dim_atoms(token):
            all_ranks.setdefault(atom, None)

    def _resolve(tokens_by_operand: List[List[str]], shapes: List[TensorShape]) -> None:
        for idx, tokens in enumerate(tokens_by_operand):
            if idx >= len(shapes):
                break
            shape = shapes[idx]
            for dim_offset, token in enumerate(tokens):
                atoms = _parse_dim_atoms(token)
                if len(atoms) == 1 and dim_offset < len(shape):
                    atom = atoms[0]
                    extent = int(shape[dim_offset])
                    current = all_ranks.get(atom)
                    # First occurrence binds; a later, larger extent replaces a
                    # size-1 one (broadcast semantics), never the reverse.
                    if current is None or (current == 1 and extent > 1):
                        all_ranks[atom] = extent

    _resolve(input_tokens, tensor_shapes.inputs)
    _resolve([output_tokens], tensor_shapes.outputs)

    total_ops = 1
    for value in all_ranks.values():
        if value is not None and value > 0:
            total_ops *= value
    return int(total_ops)


_CONV_POSITIONAL = ("stride", "padding", "dilation", "groups")


def _as_int_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (tuple, list)) and value and all(isinstance(v, int) for v in value):
        return tuple(int(v) for v in value)
    return None


def conv_call_kwargs(module_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convolution call arguments: ``groups`` (int) and ``stride``/``padding``/``dilation`` (tuples).

    ``nn.Conv*`` modules expose these directly in ``module_args``; functional
    ``F.conv*d`` calls only carry them inside torchview's ``raw_attributes``
    string, either as kwargs (``{groups: 2048, padding: 1}``) or positionally
    after the tensor arguments (``conv1d(input, weight, bias, stride,
    padding, dilation, groups)``).  Explicit ``module_args`` entries win.
    Missing values are simply absent from the returned dict.
    """
    out: Dict[str, Any] = {}
    module_args = module_args or {}
    raw = module_args.get("raw_attributes", "") if isinstance(module_args, dict) else ""
    if isinstance(raw, str) and raw:
        # kwargs: bare keys, values are ints or int tuples
        for m in re.finditer(r"\b(groups|stride|padding|dilation)\s*:\s*(\([^)]*\)|\[[^\]]*\]|-?\d+)", raw):
            key, val = m.group(1), m.group(2).strip()
            nums = [int(x) for x in re.findall(r"-?\d+", val)]
            if not nums:
                continue
            out[key] = nums[0] if key == "groups" else tuple(nums)
        # positional: scalars after the last Tensor(...) inside the args list
        args_part = raw.split("], {", 1)[0] if "], {" in raw else raw
        tail = args_part[args_part.rfind(")") + 1:] if "Tensor(" in args_part else ""
        pos = [int(x) for x in re.findall(r"-?\d+", tail)] if tail else []
        # stride, padding, dilation, groups may each be an int (tuples are rare positionally)
        for key, val in zip(_CONV_POSITIONAL, pos):
            out.setdefault(key, val if key == "groups" else (val,))
    for key in _CONV_POSITIONAL:
        if key in module_args and module_args[key] is not None:
            v = module_args[key]
            if key == "groups":
                try:
                    out[key] = int(v)
                except (TypeError, ValueError):
                    pass
            else:
                t = _as_int_tuple(v)
                if t is not None:
                    out[key] = t
    return out


class EinsumOpHandler(ABC):
    """Abstract base class for einsum operation handlers.

    Each handler is responsible for converting one or more related operation
    types to einsum notation. Handlers receive TensorShapes (positional)
    and should access inputs/outputs by index, not by name.
    """

    supported_ops: List[str] = []

    def __init__(self, debug: bool = False):
        """Initialize the handler.

        Args:
            debug: Enable debug output.
        """
        self.debug = debug

    @abstractmethod
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate an einsum operation for the given operation.

        Args:
            op_name: Normalized operation name.
            tensor_shapes: Positional input/output shapes.
            **kwargs: Additional operation-specific parameters.

        Returns:
            EinsumOp representing the operation.
        """
        pass

    def can_handle(self, op_name: str) -> bool:
        """Check if this handler can process the given operation.

        Args:
            op_name: Normalized operation name.

        Returns:
            True if this handler supports the operation.
        """
        return op_name.lower() in [op.lower() for op in self.supported_ops]

    def _validate_einsum(
        self,
        einsum_op: "EinsumOp",
        tensor_shapes: Dict[str, List[List[int]]]
    ) -> "EinsumOp":
        """Validate that einsum ranks match tensor shapes.

        If validation fails, logs a warning and attempts to fix the equation
        by regenerating it based on actual shapes.

        Args:
            einsum_op: The generated EinsumOp to validate.
            tensor_shapes: Dictionary with "inputs" and "outputs" keys containing shape lists.
                          Format: {"inputs": [[shape1], [shape2]], "outputs": [[output_shape]]}

        Returns:
            The validated (and possibly corrected) EinsumOp.
        """
        is_valid, error_msg = validate_einsum_ranks_match_shapes(
            einsum_op.equation, tensor_shapes
        )

        if not is_valid:
            logger.warning(
                f"Einsum rank mismatch for {einsum_op.name}: {error_msg}. "
                f"Equation: {einsum_op.equation}, tensor_shapes: {tensor_shapes}"
            )
            # Try to fix by regenerating equation from shapes
            corrected_op = self._try_fix_einsum_ranks(einsum_op, tensor_shapes)
            if corrected_op is not None:
                return corrected_op

        return einsum_op

    def _try_fix_einsum_ranks(
        self,
        einsum_op: "EinsumOp",
        tensor_shapes: Dict[str, List[List[int]]]
    ) -> Optional["EinsumOp"]:
        """Attempt to fix einsum equation to match actual tensor shapes.

        This is a best-effort fix that regenerates the equation based on
        actual tensor ranks.

        Args:
            einsum_op: The EinsumOp with mismatched ranks.
            tensor_shapes: Dictionary with "inputs" and "outputs" keys containing shape lists.

        Returns:
            Corrected EinsumOp if fix was possible, None otherwise.
        """
        import string

        # Get actual shapes from tensor_shapes
        input_shapes = tensor_shapes.get("inputs", [])
        output_shapes = tensor_shapes.get("outputs", [])

        if not input_shapes or not output_shapes:
            return None

        input_shape = input_shapes[0] if input_shapes else None
        input_1_shape = input_shapes[1] if len(input_shapes) > 1 else None
        output_shape = output_shapes[0] if output_shapes else None

        if input_shape is None or output_shape is None:
            return None

        input_rank = len(input_shape)
        output_rank = len(output_shape)

        # Generate labels based on actual ranks
        input_labels = string.ascii_uppercase[:input_rank]
        output_labels = string.ascii_uppercase[:output_rank]

        # For binary ops, handle second input
        if input_1_shape is not None:
            input_1_rank = len(input_1_shape)

            # Handle broadcasting: use output labels for the larger tensor
            if input_1_rank < input_rank:
                # Second input is smaller, use suffix of output labels (broadcast from right)
                input_1_labels = output_labels[-input_1_rank:] if input_1_rank > 0 else ""
            elif input_1_rank > input_rank:
                # First input is smaller, use suffix of output labels
                input_labels = output_labels[-input_rank:] if input_rank > 0 else ""
                input_1_labels = output_labels
            else:
                input_1_labels = input_labels

            new_equation = f"{input_labels},{input_1_labels}->{output_labels}"

            # Update operands
            new_operands = [
                EinsumOperand("Input", list(input_labels), is_output=False),
                EinsumOperand("Input_1", list(input_1_labels), is_output=False),
                EinsumOperand("Output", list(output_labels), is_output=True),
            ]
        else:
            new_equation = f"{input_labels}->{output_labels}"
            new_operands = [
                EinsumOperand("Input", list(input_labels), is_output=False),
                EinsumOperand("Output", list(output_labels), is_output=True),
            ]

        logger.info(f"Fixed einsum equation: {einsum_op.equation} -> {new_equation}")

        return EinsumOp(
            operands=new_operands,
            equation=new_equation,
            name=einsum_op.name,
            is_real_einsum=einsum_op.is_real_einsum,
            elementwise_op=einsum_op.elementwise_op,
            reduction_op=einsum_op.reduction_op,
            is_einsum_supportable=einsum_op.is_einsum_supportable,
        )

@dataclass
class AFOperand:
    """One tensor access in AccelForge (AF) einsum YAML.

    - dims_lowercase: Labels in the current (renamed) einsum equation. The projection
      targets AccelForge uses for this access.
    - dims_uppercase: Optional parallel list of labels from the input graph node's
      tensor notation. If None, don't do renaming and use the dims_lowercase list as is.
    """
    name: str
    dims_lowercase: List[str]
    dims_uppercase: Optional[List[str]] = None
    is_output: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict to finally dump to AF yaml."""
        operand = dict()
        operand['name'] = self.name
        operand['projection'] = (
            [d.lower() for d in self.dims_lowercase]
            if self.dims_uppercase is None
            else {d.upper(): pr.lower() for d, pr in zip(self.dims_uppercase, self.dims_lowercase)}
        )
        if self.is_output:
            operand['output'] = True
        return operand

@dataclass
class AFOp:
    """Represents an operation in AccelForge (AF) einsum format.

    - tensor_accesses: List of tensor accesses.
    - is_copy_operation: Used to copy input tensors into memory.
    """
    name: str
    tensor_accesses: List[AFOperand]
    is_copy_operation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict to finally dump to AF yaml."""
        op_dict = dict()
        op_dict['name'] = self.name
        if self.is_copy_operation:
            op_dict['is_copy_operation'] = True
        op_dict['tensor_accesses'] = [operand.to_dict() for operand in self.tensor_accesses]
        return op_dict


__all__ = [
    "EinsumOperand",
    "EinsumOp",
    "EinsumOpHandler",
    "compute_cost_from_equation",
    "ElementwiseSelectEinsumOp",
    "EinsumOperandEvidenceError",
    "validate_equation_operands",
    "resolve_equation_extents",
    "optimal_dense_binary_schedule",
]
