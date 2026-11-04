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

"""Handler for explicit ``torch.einsum`` operations.

Cost convention (issue #32 / an earlier decision)
----------------------------------
An N-operand einsum is costed as the **left-to-right chain of pairwise
contractions** ``((op0 * op1) * op2) * ...``, which is the order
``torch.einsum`` documents and uses when ``opt_einsum`` is not available or
is disabled.  Each pairwise step performs

    work(step) = prod(extent(i) for i in indices(A) | indices(B))

scalar multiplications, and the intermediate produced by the step keeps
exactly the indices that are still needed, i.e. those in the output or in
any later operand.  Indices repeated inside one operand (``ii``, a
diagonal) count once.  The unit of a step is decided by what it sums away:
a step that drops at least one index is a reduction and its work counts
as multiply-accumulates (``macs``); a step that drops no index (an outer
product or a broadcast mask) is elementwise and its work counts as
``scalar_multiplies`` (billed as other_ops, never as MACs).  For two
operands the chain is a single step: when it reduces, its MACs are the
product of all unique indices -- bit-identical to the single-contraction
formula used for DSL kernels (``compute_cost_from_equation``) and to the
matmul / bmm handlers; when it does not (``i,j->ij`` at 3 x 4: 0 MACs +
12 scalar multiplies, where the generic formula reports 12 "MACs"), the
two differ by exactly that unit.  A one-operand einsum (transpose, trace,
sum) keeps the previous convention: the product of its unique indices as
reduction work.

The order torch actually executes may differ when ``opt_einsum`` is
installed and enabled (strategy ``auto``); that choice is made at run time
and is not recorded in the captured graph, so SOLAR uses the documented
left-to-right order deterministically.

Before this handler existed, ``einsum`` had no handler at all: the analyzer
fell back to a generic elementwise ``ABC->ABC`` op whose cost is
``numel(first captured operand)``, which truncated every multi-operand
einsum (and every 2-operand ``torch.einsum``) to the size of one input.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import TensorShape, TensorShapes

logger = logging.getLogger(__name__)

# One token = "..." or a letter optionally followed by digits.  This accepts
# torch equations ("bij,bjk->bik", case-sensitive single letters) and Solar /
# renamed equations ("B1I1J1,B1J1K1->B1I1K1").
_TOKEN_RE = re.compile(r"\.\.\.|[A-Za-z][0-9]*")
_ELLIPSIS = "..."
_ELLIPSIS_PREFIX = "__ell"

# raw_attributes of a torch.einsum FunctionNode:
#   "[['bclhn,bcshn,bhcls,bcshp->bclhp', Tensor(shape=(2048, 2, 64, 8, 16), dtype=torch.float32), ...], {}]"
_RAW_EQUATION_RE = re.compile(r"['\"]([A-Za-z0-9,.\s]*?(?:->[A-Za-z0-9.\s]*)?)['\"]")
_RAW_TENSOR_RE = re.compile(r"Tensor\(shape=\(([^)]*)\)(?:,\s*dtype=([\w.]+))?")


def parse_einsum_operands_from_raw_attributes(raw_attributes: Any) -> List[Tuple[List[int], Optional[str]]]:
    """Return ``[(shape, dtype_or_None), ...]`` for every tensor argument in raw_attributes."""
    text = str(raw_attributes or "")
    out: List[Tuple[List[int], Optional[str]]] = []
    for m in _RAW_TENSOR_RE.finditer(text):
        dims = [int(x) for x in m.group(1).split(",") if x.strip()]
        out.append((dims, m.group(2)))
    return out


def parse_einsum_equation_from_raw_attributes(raw_attributes: Any) -> Optional[str]:
    """Return the first quoted einsum equation in raw_attributes (torch spelling), or None."""
    text = str(raw_attributes or "")
    for m in _RAW_EQUATION_RE.finditer(text):
        eq = m.group(1).strip()
        if eq and ("," in eq or "->" in eq) and any(ch.isalpha() for ch in eq):
            return eq
    return None


def tokenize_einsum(
    equation: str,
    operand_shapes: Optional[Sequence[Optional[Sequence[int]]]] = None,
) -> Tuple[List[List[str]], List[str]]:
    """Split an einsum equation into per-operand token lists and output tokens.

    Handles torch-style single letters, Solar tokens with digit suffixes,
    ``...`` broadcasting (expanded to synthetic tokens, right-aligned, using
    the operand ranks when shapes are given) and implicit output (no ``->``:
    broadcast dims followed by the labels that occur exactly once, sorted).
    """
    eq = (equation or "").replace(" ", "")
    if not eq:
        return [], []
    if "->" in eq:
        lhs, rhs = eq.split("->", 1)
        explicit = True
    else:
        lhs, rhs, explicit = eq, "", False
    inputs = [_TOKEN_RE.findall(op) for op in lhs.split(",")] if lhs else []

    ell_max = 0
    for i, toks in enumerate(inputs):
        if _ELLIPSIS in toks:
            named = len(toks) - 1
            shape = None
            if operand_shapes is not None and i < len(operand_shapes):
                shape = operand_shapes[i]
            if shape is not None:
                ell_max = max(ell_max, max(0, len(shape) - named))
    ell_tokens = [f"{_ELLIPSIS_PREFIX}{k}" for k in range(ell_max)]
    for i, toks in enumerate(inputs):
        if _ELLIPSIS in toks:
            named = len(toks) - 1
            shape = None
            if operand_shapes is not None and i < len(operand_shapes):
                shape = operand_shapes[i]
            n_ell = max(0, len(shape) - named) if shape is not None else ell_max
            j = toks.index(_ELLIPSIS)
            inputs[i] = toks[:j] + ell_tokens[ell_max - n_ell:] + toks[j + 1:]

    if explicit:
        out_toks = _TOKEN_RE.findall(rhs)
        if _ELLIPSIS in out_toks:
            j = out_toks.index(_ELLIPSIS)
            out_toks = out_toks[:j] + ell_tokens + out_toks[j + 1:]
    else:
        counts = Counter(t for toks in inputs for t in toks)
        out_toks = list(ell_tokens) + sorted(
            t for t, c in counts.items() if c == 1 and not t.startswith(_ELLIPSIS_PREFIX)
        )
    return inputs, out_toks


def resolve_index_extents(
    inputs: Sequence[Sequence[str]],
    output: Sequence[str],
    operand_shapes: Sequence[Optional[Sequence[int]]],
    output_shape: Optional[Sequence[int]] = None,
) -> Dict[str, int]:
    """Map every index token to its extent from positional operand shapes (then the output shape)."""
    extents: Dict[str, int] = {}
    for toks, shape in zip(inputs, operand_shapes):
        if shape is None or len(shape) != len(toks):
            continue
        for t, d in zip(toks, shape):
            d = int(d)
            if t not in extents or extents[t] == 1:  # a broadcast 1 may be widened
                extents[t] = d
    if output_shape is not None and len(output_shape) == len(output):
        for t, d in zip(output, output_shape):
            if t not in extents:
                extents[t] = int(d)
    return extents


def pairwise_chain_cost(
    inputs: Sequence[Sequence[str]],
    output: Sequence[str],
    extents: Dict[str, int],
) -> Tuple[int, int]:
    """Left-to-right pairwise contraction cost split by unit (see module docstring).

    Returns ``(reduction_macs, scalar_multiplies)``.  Each step forms the
    product of the two operands' index extents; it is a reduction
    (multiply-accumulate) only when at least one index of the step is summed
    away, i.e. does not survive into the output or a later operand.  A step
    that drops no index is an elementwise (broadcast) multiply: one scalar
    multiply per element, no accumulation.  Mixing the two in one ``macs``
    figure overstated the Mamba2 SSD chain by its three mask/decay steps
    (independent review P2-2).  A single-operand equation is costed as
    before (its index product as reduction work).
    """
    index_sets: List[Set[str]] = [set(toks) for toks in inputs]
    if not index_sets:
        return 0, 0

    def ext(t: str) -> int:
        if t not in extents:
            logger.warning("einsum cost: extent of index %r unknown; treated as 1", t)
            return 1
        return int(extents[t])

    if len(index_sets) == 1:
        return int(math.prod(ext(t) for t in index_sets[0])), 0

    out_set = set(output)
    current = index_sets[0]
    macs = 0
    scalar = 0
    for k in range(1, len(index_sets)):
        nxt = index_sets[k]
        union = current | nxt
        later: Set[str] = set()
        for s in index_sets[k + 1:]:
            later |= s
        keep = union & (out_set | later)
        work = math.prod(ext(t) for t in union)
        if union - keep:          # at least one index summed away: multiply-accumulate
            macs += work
        else:                     # nothing summed: elementwise multiply
            scalar += work
        current = keep
    return int(macs), int(scalar)


def pairwise_chain_macs(
    inputs: Sequence[Sequence[str]],
    output: Sequence[str],
    extents: Dict[str, int],
) -> int:
    """Reduction MACs of the left-to-right pairwise chain (scalar multiplies excluded)."""
    return pairwise_chain_cost(inputs, output, extents)[0]


def pairwise_chain_scalar_multiplies(
    inputs: Sequence[Sequence[str]],
    output: Sequence[str],
    extents: Dict[str, int],
) -> int:
    """Elementwise (non-reducing) multiplies of the left-to-right pairwise chain."""
    return pairwise_chain_cost(inputs, output, extents)[1]


@dataclass
class TorchEinsumOp(EinsumOp):
    """EinsumOp for an explicit torch.einsum with the pairwise-chain cost.

    ``cost_equation`` is the equation used for costing (the torch spelling
    when known, otherwise the Solar equation); ``operand_shapes`` are the
    per-operand shapes recorded when the op was built (from raw_attributes
    or the positional shapes) and are used for any operand whose positional
    shape is missing or of a different rank at cost time.
    """

    cost_equation: str = ""
    operand_shapes: Optional[List[Optional[List[int]]]] = None

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        """Reduction MACs of the chain (scalar multiplies are reported separately)."""
        chain = self._chain(tensor_shapes)
        if chain is None:
            return super().get_compute_cost(tensor_shapes)
        return pairwise_chain_cost(*chain)[0]

    def scalar_multiplies(self, tensor_shapes: TensorShapes) -> int:
        """Elementwise multiplies of the chain steps that sum nothing away."""
        chain = self._chain(tensor_shapes)
        if chain is None:
            return 0
        return pairwise_chain_cost(*chain)[1]

    def _chain(self, tensor_shapes: TensorShapes):
        equation = self.cost_equation or self.equation
        if not equation:
            return None
        positional = list(tensor_shapes.inputs)
        probe_inputs, _ = tokenize_einsum(equation, positional or self.operand_shapes)
        n = len(probe_inputs)
        shapes: List[Optional[List[int]]] = []
        for i in range(n):
            shape = list(positional[i]) if i < len(positional) and positional[i] is not None else None
            if (shape is None or len(shape) != len(probe_inputs[i])) and self.operand_shapes and i < len(self.operand_shapes):
                shape = self.operand_shapes[i]
            shapes.append(shape)
        inputs, output = tokenize_einsum(equation, shapes)
        output_shape = tensor_shapes.outputs[0] if tensor_shapes.outputs else None
        extents = resolve_index_extents(inputs, output, shapes, output_shape)
        return inputs, output, extents


class TorchEinsumHandler(EinsumOpHandler):
    """Handler for explicit ``torch.einsum`` nodes.

    The equation is taken from ``equation=`` (the graph analyzer passes the
    layer's equation) or parsed from ``module_args["raw_attributes"]`` (the
    model analyzer path).  Without any equation the previous behaviour is
    kept: a generic elementwise copy op that is not a real einsum.
    """

    supported_ops = ["einsum"]

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        equation: Optional[str] = kwargs.get("equation")
        module_args = kwargs.get("module_args") or {}
        raw_attributes = kwargs.get("raw_attributes") or module_args.get("raw_attributes")
        raw_operands = parse_einsum_operands_from_raw_attributes(raw_attributes) if raw_attributes else []
        if not equation and raw_attributes:
            equation = parse_einsum_equation_from_raw_attributes(raw_attributes)

        if not equation:
            return self._generic_fallback(op_name, tensor_shapes)

        positional = [list(s) if s is not None else None for s in tensor_shapes.inputs]
        inputs_tok, _ = tokenize_einsum(equation, positional)
        n = len(inputs_tok)
        recorded: List[Optional[List[int]]] = []
        for i in range(n):
            shape = positional[i] if i < len(positional) else None
            if (shape is None or len(shape) != len(inputs_tok[i])) and i < len(raw_operands):
                shape = list(raw_operands[i][0])
            recorded.append(shape)
        inputs_tok, output_tok = tokenize_einsum(equation, recorded)

        operands = [
            EinsumOperand(f"In{i}", list(toks), is_output=False) for i, toks in enumerate(inputs_tok)
        ]
        operands.append(EinsumOperand("Output", list(output_tok), is_output=True))
        return TorchEinsumOp(
            operands=operands,
            equation=equation,
            name="einsum",
            elementwise_op="mul",
            reduction_op="add",
            is_real_einsum=True,
            cost_equation=equation,
            operand_shapes=recorded,
        )

    @staticmethod
    def _generic_fallback(op_name: str, tensor_shapes: TensorShapes) -> EinsumOp:
        """Identical to EinsumAnalyzer's no-handler fallback (kept for compatibility)."""
        import string
        input_shape = tensor_shapes.inputs[0] if tensor_shapes.inputs else None
        if input_shape is None:
            raise ValueError(f"Unsupported operation: {op_name} (no shapes, no equation)")
        labels = string.ascii_uppercase[: len(input_shape)]
        return EinsumOp(
            operands=[
                EinsumOperand("Input", list(labels), is_output=False),
                EinsumOperand("Output", list(labels), is_output=True),
            ],
            equation=f"{labels}->{labels}",
            name="einsum",
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )


_registry = get_global_registry(load_handlers=False)
_registry.register_handler(TorchEinsumHandler)


__all__ = [
    "TorchEinsumHandler",
    "TorchEinsumOp",
    "tokenize_einsum",
    "resolve_index_extents",
    "pairwise_chain_macs",
    "parse_einsum_operands_from_raw_attributes",
    "parse_einsum_equation_from_raw_attributes",
]
