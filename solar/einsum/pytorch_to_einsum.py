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

"""Convert PyTorch computation graphs to einsum representation.

This module implements the first stage of the Solar pipeline:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

The output follows the einsum graph schema:

    layers:
      <layer_id>:
        type: <operation_type>
        einsum_equation: <equation_string>
        elementwise_op: <op>
        reduction_op: <op>
        is_real_einsum: <bool>
        is_einsum_supportable: <bool>
        tensor_names: {inputs: [...], outputs: [...]}
        tensor_shapes: {inputs: [...], outputs: [...]}
        connections: {inputs: [...], outputs: [...]}

Example:
    >>> from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    >>> converter = PyTorchToEinsum()
    >>> result = converter.convert("input/pytorch_graph.yaml", "output/")
"""

from __future__ import annotations

import json
import re
import string
import copy
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import yaml

from solar.common.types import TensorShapes
from solar.common.utils import (
    ensure_directory,
    NoAliasDumper,
    validate_tensor_names_match_shapes,
    LocalDumper,
    flowify,
)
from solar.einsum.af_graph_builder import build_af_graph_from_dict
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.einsum_rank_renamer import EinsumRankRenamer
from solar.einsum.einsum_to_taco import add_taco_expressions
from solar.einsum.ops.base import EinsumOp, EinsumOperand
from solar.einsum.ops.registry import get_global_registry


PathLike = Union[str, Path]

# Operation categories for einsum supportability classification
_REAL_EINSUM_OPS = frozenset({
    "matmul", "mm", "bmm", "linear",
    "conv1d", "conv2d", "conv3d",
    "convtranspose1d", "convtranspose2d", "convtranspose3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "attention", "sdpa",
    "einsum",
})

_BINARY_ELEMENTWISE_OPS = frozenset({
    "add", "sub", "mul", "div", "pow",
    "add_", "sub_", "mul_", "div_",
    "__add__", "__sub__", "__mul__", "__truediv__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
})

_UNARY_ELEMENTWISE_OPS = frozenset({
    "relu", "sigmoid", "tanh", "gelu", "selu", "elu", "mish",
    "softmax", "log_softmax", "softplus", "hardswish", "hardsigmoid",
    "abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos",
    "clamp", "clamp_", "relu_",
    "dropout", "dropout_",
})

_REDUCTION_OPS = frozenset({
    "sum", "mean", "prod", "max", "min", "amax", "amin",
    "argmax", "argmin", "logsumexp", "norm",
})

_NORM_OPS = frozenset({
    "batch_norm", "batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d",
    "layer_norm", "layernorm", "group_norm", "groupnorm",
    "instance_norm", "instancenorm", "normalize",
})

_POOLING_OPS = frozenset({
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
})

_SHAPE_OPS = frozenset({
    "view", "reshape", "flatten", "unflatten",
    "squeeze", "unsqueeze", "expand", "repeat",
    "transpose", "permute", "t", "contiguous",
    "cat", "concat", "stack", "split", "chunk",
    "__getitem__", "getitem", "select", "index_select",
})

_MATRIX_OPS = frozenset({"diag", "diagonal", "tril", "triu"})

_EMBEDDING_OPS = frozenset({"embedding"})

_RNN_OPS = frozenset({"gru", "lstm", "rnn"})

_TRIVIAL_OPS = frozenset({
    "clone", "detach", "copy_", "to", "type", "float", "half",
    "hidden-tensor", "output-tensor", "auxiliary-tensor",
    "roll", "pad", "unfold", "fold",
})

_ATTENTION_OPS = frozenset({
    "multi_head_attention_forward", "multihead_attention",
    "flex_attention",
})

_ALL_SUPPORTABLE_OPS = (
    _REAL_EINSUM_OPS | _BINARY_ELEMENTWISE_OPS | _UNARY_ELEMENTWISE_OPS |
    _REDUCTION_OPS | _NORM_OPS | _POOLING_OPS | _SHAPE_OPS |
    _MATRIX_OPS | _EMBEDDING_OPS | _RNN_OPS | _TRIVIAL_OPS | _ATTENTION_OPS
)

_UNSUPPORTABLE_OPS = frozenset({
    "if", "while", "for", "return", "raise",
    "print", "assert", "pass",
})


def _product(shape: List[int]) -> int:
    """Compute product of dimensions in a shape.

    Args:
        shape: List of dimension sizes.

    Returns:
        Product of all dimensions (1 for empty shape).
    """
    result = 1
    for dim in shape:
        result *= int(dim)
    return int(result)


class PyTorchToEinsum:
    """Convert PyTorch computation graphs to einsum representation.

    This converter transforms pytorch_graph.yaml files into einsum_graph.yaml
    files, translating PyTorch operations into einsum notation where possible.

    Attributes:
        debug: Whether to print debug information.
        enable_agent: Whether to use LLM agent for unknown operations.
        api_key: API key for LLM agent.
        cache_dir: Directory for caching generated handlers.
    """

    def __init__(
        self,
        debug: bool = False,
        enable_agent: bool = False,
        api_key: Optional[str] = None,
        cache_dir: str = "./solar_handlers_cache",
    ) -> None:
        """Initialize the converter.

        Args:
            debug: Enable debug output.
            enable_agent: Enable LLM agent for unknown node types.
            api_key: OpenAI API key for LLM agent.
            cache_dir: Directory for caching generated handlers.
        """
        self._debug = debug
        self._enable_agent = enable_agent
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._einsum_analyzer = EinsumAnalyzer(debug=debug)

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    @property
    def einsum_analyzer(self) -> EinsumAnalyzer:
        """The einsum analyzer instance."""
        return self._einsum_analyzer

    def _parse_einsum_from_raw_attributes(
        self,
        module_args: Dict[str, Any],
    ) -> Optional[str]:
        """Parse einsum equation from raw_attributes in module_args.

        For torch.einsum operations, the raw_attributes field contains the
        einsum equation string as the first argument.

        Example raw_attributes:
            "[[\'bijl,lk->bijk\', Tensor(...), Tensor(...)], {}]"

        Args:
            module_args: Dictionary containing module arguments.

        Returns:
            Solar-compatible einsum equation (uppercase) or None if not found.
        """
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None

        # Try to extract the einsum equation string from raw_attributes
        # Pattern: first string argument in the list, e.g., 'bijl,lk->bijk'
        import re

        # Match quoted string that looks like an einsum equation (contains -> and comma)
        # Handles both single and double quotes
        pattern = r"['\"]([a-zA-Z0-9,\s]+->[\s]*[a-zA-Z0-9]+)['\"]"
        match = re.search(pattern, raw_attrs)

        if match:
            equation = match.group(1).strip()
            # Convert to Solar format (uppercase)
            return self._convert_einsum_to_solar_format(equation)

        return None

    def _convert_einsum_to_solar_format(self, equation: str) -> str:
        """Convert a lowercase einsum equation to Solar's uppercase format.

        Solar uses uppercase letters for dimension labels, with optional
        numeric suffixes for batch dimensions (e.g., B0, B1).

        Example:
            'bijl,lk->bijk' -> 'B0IJL,LK->B0IJK'

        Args:
            equation: Lowercase einsum equation string.

        Returns:
            Uppercase einsum equation string.
        """
        if not equation or "->" not in equation:
            return equation

        # Split into inputs and output
        parts = equation.split("->")
        if len(parts) != 2:
            return equation.upper()

        lhs, rhs = parts[0].strip(), parts[1].strip()

        # Collect all unique dimension letters
        all_dims = set()
        for char in lhs + rhs:
            if char.isalpha():
                all_dims.add(char.lower())

        # Create mapping: lowercase letter -> uppercase with optional number
        # We'll use simple uppercase for now, but could add batch numbering
        dim_map = {d: d.upper() for d in sorted(all_dims)}

        # Apply mapping to equation
        result_lhs = ""
        for char in lhs:
            if char.isalpha():
                result_lhs += dim_map.get(char.lower(), char.upper())
            else:
                result_lhs += char

        result_rhs = ""
        for char in rhs:
            if char.isalpha():
                result_rhs += dim_map.get(char.lower(), char.upper())
            else:
                result_rhs += char

        return f"{result_lhs}->{result_rhs}"

    def _parse_reduction_args_from_raw_attributes(
        self,
        module_args: Dict[str, Any],
    ) -> Tuple[Optional[List[int]], bool]:
        """Parse reduction arguments (dim, keepdim) from raw_attributes.

        For reduction operations like sum/mean/max/min, the raw_attributes field
        contains the dim and keepdim arguments.

        Example raw_attributes:
            "[[Tensor(...)], {dim: 1}]"
            "[[Tensor(...)], {dim: 1, keepdim: True}]"
            "[[Tensor(...)], {dim: [1, 2]}]"

        Args:
            module_args: Dictionary containing module arguments.

        Returns:
            Tuple of (reduction_dims, keepdim). reduction_dims is a list of ints or None.
        """
        # First check parsed dim/keepdim fields (from _parse_torchview_attributes)
        if "dim" in module_args:
            dim_val = module_args["dim"]
            reduce_dims = [dim_val] if isinstance(dim_val, int) else list(dim_val)
            keepdim = bool(module_args.get("keepdim", False))
            return reduce_dims, keepdim

        # Then try raw_attributes string (regex parsing)
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None, False

        reduce_dims = None
        keepdim = False

        # Match dim: [<numbers>] first (list case)
        list_dim_pattern = r"dim:\s*\[([^\]]+)\]"
        match = re.search(list_dim_pattern, raw_attrs)
        if match:
            dims_str = match.group(1)
            reduce_dims = [int(d.strip()) for d in dims_str.split(",")]
        else:
            # Pattern for single dim: dim: 1 or dim: -1
            single_dim_pattern = r"dim:\s*(-?\d+)"
            match = re.search(single_dim_pattern, raw_attrs)
            if match:
                reduce_dims = [int(match.group(1))]

        # Match keepdim: True or keepdim: False
        keepdim_pattern = r"keepdim:\s*(True|False)"
        match = re.search(keepdim_pattern, raw_attrs)
        if match:
            keepdim = match.group(1) == "True"

        return reduce_dims, keepdim

    def _tensor_arg_shapes_from_raw(
        self,
        module_args: Dict[str, Any],
    ) -> List[Optional[Tuple[int, ...]]]:
        """Shapes of the positional ``Tensor`` arguments recorded by torchview.

        torchview stores the real call signature in ``raw_attributes``, e.g.::

            [[Tensor(shape=(112, 64, 512, 512), dtype=torch.float32),
              Tensor(shape=(), dtype=torch.float32)], {p: 'fro'}]

        Returns the shape tuple of every ``Tensor(shape=(...))`` occurrence in
        order; a scalar tensor yields ``()`` and an unparseable shape yields
        ``None``. This is ground truth for an op's true tensor arity — used by
        the dropped-edge repair in ``_build_op_graph`` to detect ops whose
        recorded ``input_shapes`` undercount their actual tensor inputs
        (torchview can drop a scalar-tensor edge, e.g. ``x / x.norm()``).
        """
        raw = module_args.get("raw_attributes", "") if module_args else ""
        if not raw:
            return []
        shapes: List[Optional[Tuple[int, ...]]] = []
        for m in re.finditer(r"Tensor\(shape=\(([^)]*)\)", raw):
            body = m.group(1).strip().rstrip(",").strip()
            if not body:
                shapes.append(())
                continue
            try:
                shapes.append(tuple(int(d.strip()) for d in body.split(",") if d.strip()))
            except ValueError:
                shapes.append(None)
        return shapes

    # ----- torchview-quirk constants ---------------------------------------
    # Layer types whose output dtype passes through unchanged from the first
    # input (no real compute, no precision change). Used by the dtype repair
    # in ``_repair_torchview_quirks``.
    _SHAPE_OP_TYPES_FOR_DTYPE: "set[str]" = {
        "view", "reshape", "flatten", "unflatten", "squeeze", "unsqueeze",
        "expand", "repeat", "transpose", "permute", "t", "contiguous",
        "cat", "concat", "stack", "split", "chunk",
        "__getitem__", "getitem", "select", "index_select",
        "narrow", "slice", "movedim", "swapaxes", "swapdims",
        "view_as", "reshape_as", "broadcast_to", "expand_as",
        "detach", "alias", "ravel", "unbind", "diagonal",
    }
    _DTYPE_BITS: "dict[str, int]" = {
        "float64": 64, "double": 64, "complex128": 128, "complex64": 64,
        "float32": 32, "tf32": 32,
        "bfloat16": 16, "float16": 16, "half": 16,
        "int64": 64, "long": 64, "int32": 32, "int": 32,
        "int16": 16, "short": 16, "int8": 8, "uint8": 8, "byte": 8,
        "bool": 1,
    }

    @classmethod
    def _bits_of_dtype(cls, dtype_str: Optional[str]) -> int:
        if not dtype_str:
            return 32
        return cls._DTYPE_BITS.get(
            str(dtype_str).replace("torch.", "").lower(), 32
        )

    def _repair_torchview_quirks(
        self,
        layers: Dict[str, Any],
        op_ids: List[str],
        tensor_ids: List[str],
    ) -> None:
        """Single pass repairing every known torchview tracing quirk.

        torchview is great at recording shapes and types but consistently
        drops a small number of well-known patterns. Three repairs run
        here in dependency order so every downstream pass — per-op
        handlers, op-graph build, AF emission — sees a clean ``layers``
        dict and never has to second-guess torchview:

        **(A) Dropped scalar-tensor edges** (FrobeniusNorm pattern).
        torchview can omit a tensor edge entirely when one operand is a
        scalar tensor produced by a reduction (canonical: ``x / x.norm()``).
        Detected by comparing each op's ``raw_attributes`` tensor-arg count
        against its recorded ``input_shapes``; missing slots are wired in
        from a matching dangling ``hidden-tensor`` dead-end (one producer,
        no consumer) of the same shape.

        **(B) Split tensor-node pairs** (cumsum_exclusive zeros_like).
        torchview occasionally splits a single physical tensor flowing
        producer→consumer into two disconnected nodes — an orphan (no
        producer, has consumer) and a dead-end (has producer, no consumer)
        of matching ``(shape, dtype)``. We rewire the orphan's
        ``connections.inputs`` to the dead-end's producer so the normal
        edge build picks up the producer→consumer edge, and record the
        producer in ``self._tensor_to_producer_op`` for the converter's
        hidden-tensor resolution path.

        **(C) Wrong output dtypes** (fp32 override on bf16/fp16 outputs).
        torchview reports ``torch.float32`` on FunctionNode outputs even
        when all inputs are narrower. Walk the layers in insertion (topo)
        order; for each non-shape op set the output dtype to the widest
        input dtype and propagate forward through downstream tensor nodes
        and the consumer ops' ``input_dtypes``. Shape ops pass through the
        first input's dtype.

        Mutates ``layers`` in place. Initializes
        ``self._tensor_to_producer_op``. Conservative: only unambiguous
        1:1 matches are applied for (A) and (B); ambiguous candidates are
        left untouched.
        """
        self._tensor_to_producer_op = {}
        op_id_set = set(op_ids)

        # --- (A+B prep) Index orphan / dead-end tensor nodes --------------
        # The shape/dtype of a tensor node is normally its own
        # ``output_shapes[0]`` / ``output_dtypes[0]``. But torchview's
        # ``hidden-tensor`` placeholder records empty lists for both, so we
        # fall back to the producer's ``output_shapes[0]`` when the tensor
        # itself has no shape recorded.
        orphans_by_key: Dict[Tuple[Tuple[int, ...], str], List[str]] = defaultdict(list)
        dangling_by_key: Dict[Tuple[Tuple[int, ...], str], List[Tuple[str, str]]] = defaultdict(list)
        hidden_dangling_by_shape: Dict[Tuple[int, ...], List[Tuple[str, str, str]]] = defaultdict(list)
        for tensor_id in tensor_ids:
            tdata = layers.get(tensor_id) or {}
            conns = tdata.get("connections") or {}
            producers_ = [p for p in (conns.get("inputs") or []) if p in op_id_set]
            consumers_ = [c for c in (conns.get("outputs") or []) if c in op_id_set]
            shapes = tdata.get("output_shapes") or tdata.get("input_shapes") or []
            dtypes = tdata.get("output_dtypes") or tdata.get("input_dtypes") or []
            if not shapes and len(producers_) == 1:
                pdata = layers.get(producers_[0]) or {}
                shapes = pdata.get("output_shapes") or []
                dtypes = pdata.get("output_dtypes") or []
            if not shapes:
                continue
            sh = tuple(shapes[0]) if shapes[0] is not None else ()
            dt = str(dtypes[0]) if dtypes else ""
            key = (sh, dt)
            if not producers_ and consumers_:
                orphans_by_key[key].append(tensor_id)
            elif len(producers_) == 1 and not consumers_:
                dangling_by_key[key].append((tensor_id, producers_[0]))
                if (tdata.get("type") or "").lower() == "hidden-tensor":
                    hidden_dangling_by_shape[sh].append((tensor_id, producers_[0], dt))

        # --- (A) Dropped scalar-tensor edges ------------------------------
        consumed: Set[Tuple[str, str]] = set()
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            arg_shapes = [s for s in self._tensor_arg_shapes_from_raw(
                odata.get("module_args") or {}) if s is not None]
            if not arg_shapes:
                continue
            recorded = [tuple(s) for s in (odata.get("input_shapes") or [])
                        if s is not None]
            missing = Counter(arg_shapes) - Counter(recorded)
            if not missing:
                continue
            in_dtypes = odata.get("input_dtypes") or []
            default_dt = str(in_dtypes[0]) if in_dtypes else "torch.float32"
            for sh, cnt in missing.items():
                for _ in range(cnt):
                    candidates = [(t, p, d) for (t, p, d)
                                  in hidden_dangling_by_shape.get(sh, [])
                                  if (t, p) not in consumed]
                    if len(candidates) != 1:
                        continue  # ambiguous or none
                    tensor_id, producer_op, dt = candidates[0]
                    if producer_op == op_id:
                        continue
                    consumed.add((tensor_id, producer_op))
                    odata.setdefault("input_shapes", []).append(list(sh))
                    odata.setdefault("input_dtypes", []).append(dt or default_dt)
                    odata.setdefault("input_types", []).append("input")
                    oc = odata.setdefault("connections", {}).setdefault("inputs", [])
                    if tensor_id not in oc:
                        oc.append(tensor_id)
                    tdata = layers.get(tensor_id) or {}
                    tout = tdata.setdefault("connections", {}).setdefault("outputs", [])
                    if op_id not in tout:
                        tout.append(op_id)

        # --- (B) Split tensor-node pairs ----------------------------------
        for key, orphan_ids in orphans_by_key.items():
            de_list = [(t, p) for (t, p) in dangling_by_key.get(key, [])
                       if (t, p) not in consumed]
            if len(orphan_ids) != 1 or len(de_list) != 1:
                continue
            orphan_id = orphan_ids[0]
            producer_op = de_list[0][1]
            self._tensor_to_producer_op[orphan_id] = producer_op
            # Rewire the orphan's ``connections.inputs`` to the producer so
            # the normal edge build picks up producer→consumer naturally.
            orphan_data = layers.get(orphan_id) or {}
            ocon_in = orphan_data.setdefault("connections", {}).setdefault("inputs", [])
            if producer_op not in ocon_in:
                ocon_in.append(producer_op)

        # --- (C) Output-dtype correction ----------------------------------
        # Seed the corrected-dtype map from every NON-OP node's declared
        # dtype: regular tensor nodes (intermediates), auxiliary-tensor
        # nodes (model inputs), and parameter-tensor nodes (weights). These
        # are ground truth, recorded by torchview at trace time. Op-output
        # dtypes get overwritten below as we walk the graph in topo order.
        # ``_partition_nodes`` splits these into three lists; iterate every
        # non-op layer (anything in ``layers`` that's not in ``op_id_set``)
        # so we don't miss auxiliary/parameter tensors.
        corrected_dtype: Dict[str, str] = {}
        for layer_id, ldata in layers.items():
            if layer_id in op_id_set:
                continue
            outd = (ldata.get("output_dtypes") or ldata.get("input_dtypes")
                    or [])
            if outd:
                corrected_dtype[layer_id] = outd[0]

        for layer_id, odata in layers.items():
            if layer_id not in op_id_set:
                continue
            in_tensors = (odata.get("connections") or {}).get("inputs") or []
            in_dtypes = list(odata.get("input_dtypes") or [])
            for i, tid in enumerate(in_tensors):
                if tid in corrected_dtype and i < len(in_dtypes):
                    in_dtypes[i] = corrected_dtype[tid]
            if in_dtypes:
                odata["input_dtypes"] = in_dtypes
            layer_type = (odata.get("type") or "").lower()
            if layer_type in self._SHAPE_OP_TYPES_FOR_DTYPE:
                widest = (in_dtypes[0] if in_dtypes
                          else (odata.get("output_dtypes") or ["torch.float32"])[0])
            elif in_dtypes:
                widest = max(in_dtypes, key=self._bits_of_dtype)
            else:
                widest = (odata.get("output_dtypes") or ["torch.float32"])[0]
            n_out = len(odata.get("output_dtypes") or []) or 1
            odata["output_dtypes"] = [widest] * n_out
            for tid in (odata.get("connections") or {}).get("outputs") or []:
                if tid in layers:
                    tdata = layers[tid]
                    n = len(tdata.get("output_dtypes") or []) or 1
                    tdata["output_dtypes"] = [widest] * n
                corrected_dtype[tid] = widest

    def _validate_tensor_shape_consistency(
        self,
        einsum_graph: Dict[str, Any],
    ) -> None:
        """Assert that every tensor name reused across einsums has a single shape.

        Walks every einsum's tensor_names + tensor_shapes (inputs and outputs)
        and builds a (name → set of shape tuples) map. Any name with >1 distinct
        shape indicates an emit bug — for example, the cumsum_exclusive case
        where ``Model.cat`` mis-attributed input 0 to ``Model.cumsum``, causing
        the name ``Model.cumsum.Output`` to claim both ``[32768, 1]`` and
        ``[32768, 32767]`` shapes.

        Surfaces the violation at the solar boundary (with a list of every
        einsum that referenced the conflicting tensor) rather than letting
        the inconsistency propagate to AccelForge.
        """
        shapes_by_name: Dict[str, Dict[Tuple[int, ...], List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for layer_name, m in (einsum_graph.get("layers") or {}).items():
            tnames = m.get("tensor_names") or {}
            tshapes = m.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                names = tnames.get(side) or []
                shapes = tshapes.get(side) or []
                n = min(len(names), len(shapes))
                for i in range(n):
                    if shapes[i] is None:
                        continue
                    key = tuple(shapes[i])
                    shapes_by_name[names[i]][key].append(
                        f"{layer_name}.{side}[{i}]"
                    )
        conflicts: List[str] = []
        for name, by_shape in shapes_by_name.items():
            if len(by_shape) > 1:
                listed = "; ".join(
                    f"{list(shape)} via {refs}" for shape, refs in by_shape.items()
                )
                conflicts.append(f"  '{name}': {listed}")
        if conflicts:
            raise ValueError(
                "Pre-AF tensor-shape inconsistency detected: a tensor name "
                "is referenced with multiple distinct shapes across einsums. "
                "This usually means a producer was mis-attributed in stage-1. "
                "Conflicts:\n" + "\n".join(conflicts)
            )

    def convert(
        self,
        pytorch_graph_path: PathLike,
        output_dir: PathLike,
        *,
        copy_graph: bool = True,
        expand_complex_ops: bool = True,
        enable_rename: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Convert a PyTorch graph to einsum representation.

        This method:
        1. Loads the PyTorch graph
        2. Builds an operation-only graph (collapsing tensor nodes)
        3. Converts operations to einsum notation
        4. Writes einsum_graph.yaml
        5. Optionally renames ranks using BFS and writes einsum_graph_renamed.yaml

        Args:
            pytorch_graph_path: Path to pytorch_graph.yaml (or legacy JSON).
            output_dir: Directory to write output files.
            copy_graph: If True, copy input graph to output directory.
            expand_complex_ops: If True, attempt to expand complex operations.
            enable_rename: If True, run BFS rank renaming; otherwise copy einsum_graph.yaml as renamed.

        Returns:
            The einsum graph dictionary, or None on failure.
        """
        src = Path(pytorch_graph_path)
        out_dir = ensure_directory(output_dir)

        if not src.exists():
            if self._debug:
                print(f"Debug: PyTorch graph not found: {src}")
            return None

        pytorch_graph = self._load_pytorch_graph(src)
        if not pytorch_graph:
            return None

        if copy_graph:
            self._copy_input_graph(src, out_dir, pytorch_graph)

        # Build operation-only graph (collapse tensor nodes)
        op_graph, start_nodes_info, param_nodes_info = self._build_op_graph(pytorch_graph)

        # Optional complex operation expansion
        if expand_complex_ops:
            op_graph = self._expand_complex_ops(op_graph)

        # Build einsum graph dictionary
        einsum_graph = self._build_einsum_graph(
            pytorch_graph, op_graph, start_nodes_info, param_nodes_info
        )

        # Add TACO expressions to all layers
        einsum_graph = add_taco_expressions(einsum_graph)

        # Pre-AF validation: every tensor name reused across einsums must
        # have a single, consistent shape. Catches emit-side inconsistencies
        # (e.g. the cat-input mis-attribution that mapped two distinct
        # shapes onto one tensor name in kernbench L1/92 cumsum_exclusive)
        # before they reach AccelForge's pydantic schema, where the error
        # message points only at the symptom rather than the cause.
        self._validate_tensor_shape_consistency(einsum_graph)

        # Write einsum_graph.yaml
        out_path = out_dir / "einsum_graph.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                einsum_graph, f,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False
            )

        if self._debug:
            print(f"✅ Wrote einsum graph: {out_path}")

        renamed_path = out_dir / "einsum_graph_renamed.yaml"
        if enable_rename:
            renamer = EinsumRankRenamer(debug=self._debug)
            renamer.rename(einsum_graph, renamed_path)
            if self._debug:
                print(f"✅ Wrote renamed einsum graph: {renamed_path}")
        else:
            import shutil
            shutil.copy2(out_path, renamed_path)
            if self._debug:
                print(f"✅ Copied einsum graph as renamed (rename disabled): {renamed_path}")

        # Principled AccelForge graph emission via one union-find pass
        # over the stage-2 einsum graph. Replaces the historical
        # ``_build_af_einsum_graph_renamed`` + ``_build_accelforge_graph``
        # pipeline, which suffered from a phase-skew bug where the conflict
        # mint mutated ``dims`` in-place without persisting back into
        # ``ops["Output"]`` (causing the "rank-size collision / Rk reuse"
        # failures on DenseNet/Mamba2/RNN/LSTM/GRU/etc.).
        af_graph = build_af_graph_from_dict(einsum_graph)
        # Strip internal diagnostics before serializing — AF rejects unknown
        # top-level fields.
        af_to_write = {k: v for k, v in af_graph.items() if not k.startswith("_")}
        out_path = out_dir / "af_einsum_graph.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                af_to_write, f,
                sort_keys=False,
                default_flow_style=False
            )

        if self._debug:
            print(f"✅ Wrote AccelForge graph: {out_path}")
            diagnostics = af_graph.get("_diagnostics") or []
            for d in diagnostics:
                print(f"  af_graph diagnostic: {d}")

        return einsum_graph

    # Backward compatibility alias
    convert_graph = convert

    def _copy_input_graph(
        self,
        src: Path,
        out_dir: Path,
        pytorch_graph: Dict[str, Any],
    ) -> None:
        """Copy input graph to output directory."""
        try:
            dst = out_dir / "pytorch_graph.yaml"
            if src.suffix.lower() in {".yaml", ".yml"}:
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            elif not dst.exists():
                with open(dst, "w") as f:
                    yaml.dump(
                        pytorch_graph, f,
                        Dumper=NoAliasDumper,
                        sort_keys=False,
                        default_flow_style=False
                    )
        except Exception:
            if self._debug:
                print("Debug: Failed to copy/write canonical pytorch_graph.yaml")

    def _load_pytorch_graph(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load PyTorch graph from YAML or JSON file.

        Args:
            path: Path to the graph file.

        Returns:
            The graph dictionary, or None on failure.
        """
        try:
            suffix = path.suffix.lower()

            if suffix in {".yaml", ".yml"}:
                with open(path) as f:
                    data = yaml.safe_load(f)
            elif suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
            else:
                if self._debug:
                    print(f"Debug: Unsupported file extension: {path.suffix}")
                return None

            if isinstance(data, dict) and "layers" in data:
                return data
            if isinstance(data, list):
                return self._convert_node_list(data, model_name=path.stem)

            if self._debug:
                print(f"Debug: Unexpected structure in {path}")
            return None

        except Exception as exc:
            if self._debug:
                print(f"Debug: Failed to load PyTorch graph: {exc}")
            return None

    def _convert_node_list(
        self,
        nodes: List[Dict[str, Any]],
        *,
        model_name: str,
    ) -> Dict[str, Any]:
        """Convert legacy node list format to structured graph dictionary."""
        layers: Dict[str, Any] = {}
        for node in nodes:
            node_id = node.get("node_id") or node.get("name") or "unknown"
            layers[node_id] = {
                "type": node.get("node_type", node.get("type", "unknown")),
                "node_class": node.get("node_class", "UnknownNode"),
                "input_shapes": node.get("input_shapes", []) or [],
                "output_shapes": node.get("output_shapes", []) or [],
                "weight_nodes": node.get("weight_nodes", []) or [],
                "weight_shapes": node.get("weight_shapes", []) or [],
                "module_args": node.get("module_args", {}) or {},
                "connections": {
                    "inputs": node.get("input_nodes", []) or [],
                    "outputs": node.get("output_nodes", []) or [],
                },
            }
        return {"model_name": model_name, "layers": layers}

    def _build_op_graph(
        self,
        pytorch_graph: Dict[str, Any],
    ) -> Tuple[nx.DiGraph, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build operation-only graph by collapsing tensor nodes.

        The input PyTorch graph is typically bipartite (TensorNodes and
        Function/Module nodes). This method collapses tensors and connects
        producer operations to consumer operations.

        Args:
            pytorch_graph: The PyTorch graph dictionary.

        Returns:
            Tuple of (operation graph, start node information, parameter node info).
        """
        layers = pytorch_graph.get("layers") or {}
        tensor_ids, op_ids, auxiliary_ids, parameter_ids = self._partition_nodes(layers)

        # Repair every known torchview tracing quirk in one place: dropped
        # scalar edges, orphan/dead-end tensor pairs, and fp32-overridden
        # output dtypes. After this call, ``layers`` is the cleaned source
        # of truth for downstream graph construction and handlers.
        self._repair_torchview_quirks(layers, op_ids, tensor_ids)

        graph = nx.DiGraph()
        for op_id in op_ids:
            graph.add_node(op_id, **(layers.get(op_id) or {}))

        # Collect auxiliary tensor info for start nodes (model inputs only)
        start_nodes_info = self._collect_start_node_info(
            layers, auxiliary_ids, op_ids
        )

        # Collect parameter tensor info separately (model weights)
        param_nodes_info = self._collect_start_node_info(
            layers, parameter_ids, op_ids
        )

        for tensor_id in tensor_ids:
            tensor_data = layers.get(tensor_id) or {}
            conns = tensor_data.get("connections") or {}
            producers = list(conns.get("inputs") or [])
            consumers = list(conns.get("outputs") or [])

            if len(producers) == 1 and producers[0] in op_ids:
                self._tensor_to_producer_op[tensor_id] = producers[0]

            for producer in producers:
                for consumer in consumers:
                    if producer in op_ids and consumer in op_ids:
                        if producer != consumer:
                            graph.add_edge(producer, consumer)

        # Fallback: use direct connections if no tensor nodes
        if not tensor_ids:
            for op_id in op_ids:
                conns = (layers.get(op_id) or {}).get("connections") or {}
                outputs = list(conns.get("outputs") or [])
                for out_id in outputs:
                    if out_id in op_ids and out_id != op_id:
                        graph.add_edge(op_id, out_id)

        return graph, start_nodes_info, param_nodes_info

    def _partition_nodes(
        self,
        layers: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Partition nodes into tensor, operation, and auxiliary categories.

        Args:
            layers: The layers dictionary from the PyTorch graph.

        Returns:
            Tuple of (tensor_ids, op_ids, auxiliary_tensor_ids, parameter_tensor_ids).
        """
        tensor_ids: List[str] = []
        op_ids: List[str] = []
        auxiliary_ids: List[str] = []

        parameter_ids: List[str] = []

        for node_id, data in (layers or {}).items():
            node_class = (data.get("node_class") or "").lower()
            node_type = (data.get("type") or "").lower()

            # Any *-tensor/TensorNode should be treated as a tensor-side node,
            # never an operation node.
            if "tensornode" in node_class or "tensor" in node_type:
                if node_type == "auxiliary-tensor":
                    auxiliary_ids.append(node_id)
                elif node_type == "parameter-tensor":
                    parameter_ids.append(node_id)
                else:
                    tensor_ids.append(node_id)
            else:
                op_ids.append(node_id)

        return tensor_ids, op_ids, auxiliary_ids, parameter_ids

    def _collect_start_node_info(
        self,
        layers: Dict[str, Any],
        auxiliary_ids: List[str],
        op_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Collect information about auxiliary tensors to create start nodes."""
        start_nodes_info: List[Dict[str, Any]] = []

        for idx, aux_id in enumerate(auxiliary_ids):
            aux_data = layers.get(aux_id) or {}
            conns = aux_data.get("connections") or {}
            output_shapes = aux_data.get("output_shapes") or []
            consumers = list(conns.get("outputs") or [])
            # Filter to only include operation nodes
            valid_consumers = [c for c in consumers if c in op_ids]

            output_dtypes = aux_data.get("output_dtypes") or []

            start_nodes_info.append({
                "original_id": aux_id,
                "index": idx,
                "output_shapes": output_shapes,
                "output_dtypes": output_dtypes,
                "consumers": valid_consumers,
            })

        return start_nodes_info

    def _expand_complex_ops(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Expand complex operations using GraphExpander (best-effort)."""
        if not graph.nodes:
            return graph

        try:
            from solar.einsum.graph_expander import GraphExpander

            expander = GraphExpander(
                debug=self._debug,
                enable_agent=self._enable_agent,
                api_key=self._api_key,
                cache_dir=self._cache_dir,
            )
            return expander.expand(graph)
        except Exception:
            return graph

    def _build_einsum_graph(
        self,
        pytorch_graph: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        param_nodes_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build einsum graph dictionary from operation graph."""
        result: Dict[str, Any] = {
            "model_name": pytorch_graph.get("model_name", "pytorch_model"),
            "layers": {},
        }

        # Add start nodes from auxiliary tensors (model inputs only)
        start_node_id_map = self._add_start_nodes(result, start_nodes_info)

        # Combine start + param info for ID mapping in _convert_operation.
        # Parameter nodes don't get their own einsum layers.
        all_source_nodes_info = list(start_nodes_info) + list(param_nodes_info or [])

        # Map parameter node original IDs so _convert_operation can find them
        for info in (param_nodes_info or []):
            original_id = info["original_id"]
            start_node_id_map[original_id] = original_id

        # Map hidden-tensor IDs to their producer op so all downstream
        # code (connections, tensor_names) resolves them automatically.
        for tensor_id, producer_op in self._tensor_to_producer_op.items():
            if tensor_id not in start_node_id_map:
                start_node_id_map[tensor_id] = producer_op

        # Track node ID remapping for split/expanded operations
        # Maps original node_id -> final output node_id
        node_id_remap: Dict[str, str] = {}

        # Track expanded nodes' input mappings
        # Maps original node_id -> {input_index -> subgraph_node_id}
        expanded_input_map: Dict[str, Dict[int, str]] = {}

        # Convert each operation to einsum representation
        for node_id in op_graph.nodes():
            node_data = dict(op_graph.nodes[node_id] or {})
            self._validate_input_types_alignment(node_id, node_data)

            # Check if this is a linear layer with bias that should be split
            if self._should_split_linear_with_bias(node_data):
                matmul_layer, add_layer = self._split_linear_with_bias(
                    node_id, node_data, op_graph, all_source_nodes_info, start_node_id_map
                )
                result["layers"][node_id] = matmul_layer
                add_node_id = f"{node_id}.bias_add"
                result["layers"][add_node_id] = add_layer
                # Remap: original node_id outputs now come from add_node_id
                node_id_remap[node_id] = add_node_id

            # Check if this is a group-wise conv that needs reshape expansion
            elif self._should_expand_groupwise_conv(node_data):
                subgraph_layers, final_node_id, input_mapping = self._expand_groupwise_conv(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is MHA that should be expanded
            elif self._should_expand_mha(node_data):
                subgraph_layers, final_node_id, input_mapping = self._expand_mha(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is LSTM that should be expanded
            elif self._should_expand_lstm(node_data):
                subgraph_layers, final_node_id, input_mapping = self._expand_lstm(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is GRU that should be expanded
            elif self._should_expand_gru(node_data):
                subgraph_layers, final_node_id, input_mapping = self._expand_gru(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                node_id_remap[node_id] = final_node_id
                expanded_input_map[node_id] = input_mapping

            # Check if this is SDPA that should be expanded
            elif self._should_expand_sdpa(node_data):
                subgraph_layers, final_node_id, input_mapping = self._expand_sdpa(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                for sub_id, sub_layer in subgraph_layers.items():
                    result["layers"][sub_id] = sub_layer
                # Remap: original node_id outputs now come from final subgraph node
                node_id_remap[node_id] = final_node_id
                # Store input mapping for predecessor updates
                expanded_input_map[node_id] = input_mapping

            else:
                layer_dict = self._convert_operation(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                result["layers"][node_id] = layer_dict

        # Fix connections for split/expanded operations
        self._fix_split_connections(result, node_id_remap, expanded_input_map)

        return result

    def _should_expand_mha(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a multi_head_attention_forward that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"multi_head_attention_forward", "multihead_attention"}

    def _should_expand_sdpa(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a scaled_dot_product_attention that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"scaled_dot_product_attention", "sdpa", "attention"}

    def _should_expand_lstm(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is an LSTM that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"lstm"}

    def _should_expand_gru(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a GRU that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        return node_type in {"gru"}

    def _expand_sdpa(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand scaled_dot_product_attention into a subgraph of operations.

        Based on PyTorch's reference implementation:
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight = torch.softmax(attn_weight, dim=-1)
            return attn_weight @ value

        Returns:
            Tuple of (subgraph_layers_dict, final_node_id, input_mapping)
            input_mapping maps input index -> subgraph node that receives it
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        module_args = node_data.get("module_args", {})
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        if len(input_shapes) < 3:
            raise ValueError(f"SDPA requires 3 inputs (Q, K, V). Got: {input_shapes}")

        query_shape = list(input_shapes[0])  # [B, H, Q, D]
        key_shape = list(input_shapes[1])    # [B, H, K, D]
        value_shape = list(input_shapes[2])  # [B, H, K, V]
        output_shape = list(output_shapes[0]) if output_shapes else None

        # Infer dimensions
        B = query_shape[0]  # batch
        H = query_shape[1]  # heads
        Q_len = query_shape[2]  # query sequence length
        D = query_shape[3]  # embedding dim
        K_len = key_shape[2]    # key sequence length
        V_dim = value_shape[3]  # value embedding dim

        # Intermediate shapes
        scores_shape = [B, H, Q_len, K_len]  # Q @ K^T
        final_output_shape = output_shape if output_shape else [B, H, Q_len, V_dim]

        # Build input connections
        input_connections = sorted(list(op_graph.predecessors(node_id)))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)

        output_connections = sorted(list(op_graph.successors(node_id)))

        subgraph: Dict[str, Dict[str, Any]] = {}

        # Node IDs for subgraph
        qk_node_id = f"{node_id}.qk_matmul"
        scale_node_id = f"{node_id}.scale"
        softmax_node_id = f"{node_id}.softmax"
        av_node_id = f"{node_id}.av_matmul"

        # Build input mapping: which predecessor input goes to which subgraph node
        # Q (input 0) -> qk_matmul
        # K (input 1) -> qk_matmul
        # V (input 2) -> av_matmul
        input_mapping: Dict[int, str] = {
            0: qk_node_id,  # Q -> qk_matmul
            1: qk_node_id,  # K -> qk_matmul
            2: av_node_id,  # V -> av_matmul
        }

        # 1. Q @ K^T -> attention scores
        # Einsum: BHQD,BHKD->BHQK (D is contracted)
        subgraph[qk_node_id] = {
            "type": "matmul",
            "einsum_equation": "BHQD,BHKD->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    f"{input_connections[0]}.Output" if input_connections else f"{node_id}.Query",
                    f"{input_connections[1]}.Output" if len(input_connections) > 1 else f"{node_id}.Key",
                ],
                "outputs": [f"{qk_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [query_shape, key_shape],
                "outputs": [scores_shape],
            },
            # Operands drive the AF graph builder; without them the
            # layer is silently dropped (cf. commit 8162f29 for linear).
            "operands": {
                "Input":  ["B", "H", "Q", "D"],
                "Weight": ["B", "H", "K", "D"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": input_connections[:2] if len(input_connections) >= 2 else input_connections,
                "outputs": [scale_node_id],
            },
        }

        # 2. Scale by 1/sqrt(d_k)
        subgraph[scale_node_id] = {
            "type": "mul",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{qk_node_id}.Output"],
                "outputs": [f"{scale_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "operands": {
                "Input":  ["B", "H", "Q", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [qk_node_id],
                "outputs": [softmax_node_id],
            },
            "additional_info": {
                "scale_factor": f"1/sqrt({D})",
            },
        }

        # 3. Softmax over K dimension (dim=-1)
        subgraph[softmax_node_id] = {
            "type": "softmax",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "softmax",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{scale_node_id}.Output"],
                "outputs": [f"{softmax_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "operands": {
                "Input":  ["B", "H", "Q", "K"],
                "Output": ["B", "H", "Q", "K"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [scale_node_id],
                "outputs": [av_node_id],
            },
            "additional_info": {
                "dim": -1,
            },
        }

        # 4. Attention weights @ V -> output
        # Einsum: BHQK,BHKV->BHQV (K is contracted)
        subgraph[av_node_id] = {
            "type": "matmul",
            "einsum_equation": "BHQK,BHKV->BHQV",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [
                    f"{softmax_node_id}.Output",
                    f"{input_connections[2]}.Output" if len(input_connections) > 2 else f"{node_id}.Value",
                ],
                "outputs": [f"{av_node_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape, value_shape],
                "outputs": [final_output_shape],
            },
            "operands": {
                "Input":  ["B", "H", "Q", "K"],
                "Weight": ["B", "H", "K", "V"],
                "Output": ["B", "H", "Q", "V"],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": [softmax_node_id] + (input_connections[2:3] if len(input_connections) > 2 else []),
                "outputs": output_connections,
            },
        }

        return subgraph, av_node_id, input_mapping

    def _should_expand_groupwise_conv(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a group-wise convolution that needs reshape expansion.

        Currently only conv1d / conv2d are expanded via the reshape pass.
        Conv3d and conv-transpose variants rely on the AF graph builder's
        union-find canonicalization to handle the C_out vs C_out/groups
        split, since the reshape path doesn't survive the parameter-tensor
        weight (still 4-/5-dim original) <-> 5-/6-dim grouped einsum mismatch.
        """
        node_type = str(node_data.get("type", "")).lower()
        if node_type not in ("conv1d", "conv2d"):
            return False

        module_args = node_data.get("module_args") or {}
        groups = int(module_args.get("groups", 1))
        if groups <= 1:
            return False

        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        in_channels = int(
            module_args.get(
                "in_channels",
                input_shapes[0][1] if input_shapes and len(input_shapes[0]) > 1 else 0,
            )
        )
        out_channels = int(
            module_args.get(
                "out_channels",
                output_shapes[0][1] if output_shapes and len(output_shapes[0]) > 1 else 0,
            )
        )

        # Depthwise conv is handled directly by the conv handler.
        if groups == in_channels and groups == out_channels:
            return False
        return True

    def _expand_groupwise_conv(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: "nx.DiGraph",
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand group-wise conv into input view, grouped conv, and output view."""
        module_args = node_data.get("module_args") or {}
        groups = int(module_args.get("groups", 1))
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        node_type = str(node_data.get("type", "conv2d")).lower()
        is_2d = node_type == "conv2d"

        input_shape = list(input_shapes[0]) if len(input_shapes) > 0 else []
        weight_shape = list(input_shapes[1]) if len(input_shapes) > 1 else []
        output_shape = list(output_shapes[0]) if output_shapes else []

        B = input_shape[0]
        C_in = input_shape[1]
        O_total = weight_shape[0] if weight_shape else output_shape[1]
        I = C_in // groups
        O_pg = O_total // groups

        if is_2d:
            H, W = input_shape[2], input_shape[3]
            KH, KW = weight_shape[2], weight_shape[3]
            H_out, W_out = output_shape[2], output_shape[3]
            reshaped_input = [B, groups, I, H, W]
            reshaped_weight = [groups, O_pg, I, KH, KW]
            reshaped_output = [B, groups, O_pg, H_out, W_out]
            reshape_in_eq = "ABCD->AE0E1CD"
            reshape_in_operands = {
                "Input": ["A", "B", "C", "D"],
                "Output": ["A", "E0", "E1", "C", "D"],
            }
            reshape_out_eq = "ABCDE->AF0DE"
            reshape_out_operands = {
                "Input": ["A", "B", "C", "D", "E"],
                "Output": ["A", "F0", "D", "E"],
            }
            fallback_conv_equation = "BGI(P+R)(Q+S),GOIRS->BGOPQ"
        else:
            L = input_shape[2]
            KL = weight_shape[2]
            L_out = output_shape[2]
            reshaped_input = [B, groups, I, L]
            reshaped_weight = [groups, O_pg, I, KL]
            reshaped_output = [B, groups, O_pg, L_out]
            reshape_in_eq = "ABC->ADE0C"
            reshape_in_operands = {
                "Input": ["A", "B", "C"],
                "Output": ["A", "D", "E0", "C"],
            }
            reshape_out_eq = "ABCD->AE0D"
            reshape_out_operands = {
                "Input": ["A", "B", "C", "D"],
                "Output": ["A", "E0", "D"],
            }
            fallback_conv_equation = "BGI(P+R),GOIR->BGOP"

        raw_input_connections = list((node_data.get("connections") or {}).get("inputs") or [])
        if not raw_input_connections:
            raw_input_connections = list(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                original_id = info["original_id"]
                if original_id not in raw_input_connections:
                    raw_input_connections.append(original_id)

        # Strict tensor→producer resolution (see notes in the twin block of
        # `_convert_operation`). Fall back to op_graph.predecessors only when
        # exactly enough unmatched predecessors remain to fill all deferred
        # slots — raise on ambiguity rather than silently positional-guess.
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        op_predecessors = list(op_graph.predecessors(node_id))
        input_connections: List[Optional[str]] = []
        assigned_preds: set = set()
        deferred_indices: List[int] = []
        input_types_raw = list(node_data.get("input_types") or [])
        for idx, conn_id in enumerate(raw_input_connections):
            mapped = start_node_id_map.get(conn_id, conn_id)
            itype = str(input_types_raw[idx]).lower() if idx < len(input_types_raw) else "input"
            if itype == "weight":
                input_connections.append(mapped)
                continue
            if mapped in start_node_id_map.values() or mapped in op_graph.nodes:
                input_connections.append(mapped)
                assigned_preds.add(mapped)
                continue
            producer = tensor_to_producer.get(conn_id)
            if producer is not None and producer in op_graph.nodes:
                input_connections.append(producer)
                assigned_preds.add(producer)
                continue
            input_connections.append(None)
            deferred_indices.append(idx)
        if deferred_indices:
            unmatched_preds = [p for p in op_predecessors if p not in assigned_preds]
            if len(unmatched_preds) == len(deferred_indices):
                for d_idx, pred in zip(deferred_indices, unmatched_preds):
                    input_connections[d_idx] = pred
            elif len(unmatched_preds) == 0:
                # Genuinely producerless input — fall through to literal name.
                for d_idx in deferred_indices:
                    input_connections[d_idx] = start_node_id_map.get(
                        raw_input_connections[d_idx], raw_input_connections[d_idx]
                    )
            else:
                raise ValueError(
                    f"_convert_operation({node_id!r}, conv path): ambiguous "
                    f"predecessor resolution; deferred={len(deferred_indices)}, "
                    f"unmatched_preds={len(unmatched_preds)}."
                )

        output_connections = list(op_graph.successors(node_id))
        if not output_connections:
            raw_outs = list((node_data.get("connections") or {}).get("outputs") or [])
            output_connections = [c for c in raw_outs if c in op_graph.nodes]

        input_types = node_data.get("input_types") or []
        activation_conn = None
        weight_conn = None
        for idx, conn in enumerate(input_connections):
            itype = input_types[idx] if idx < len(input_types) else "input"
            if str(itype).lower() == "weight" or "parameter-tensor" in conn:
                if weight_conn is None:
                    weight_conn = conn
            elif activation_conn is None:
                activation_conn = conn
        if activation_conn is None and input_connections:
            activation_conn = input_connections[0]
        if weight_conn is None and len(input_connections) > 1:
            weight_conn = input_connections[1]

        reshape_in_id = f"{node_id}.reshape_input"
        conv_id = f"{node_id}.groupwise_conv"
        reshape_out_id = f"{node_id}.reshape_output"

        activation_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = input_dtypes[1] if len(input_dtypes) > 1 else activation_dtype
        output_dtype = output_dtypes[0] if output_dtypes else activation_dtype
        weight_tensor_name = f"{weight_conn}.Output" if weight_conn else f"{conv_id}.Weight"

        reshape_in_layer = {
            "type": "view",
            "einsum_equation": reshape_in_eq,
            "elementwise_op": "copy",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{activation_conn}.Output" if activation_conn else f"{reshape_in_id}.Input"],
                "outputs": [f"{reshape_in_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(input_shape)],
                "outputs": [reshaped_input],
            },
            "tensor_dtypes": {
                "inputs": [activation_dtype],
                "outputs": [activation_dtype],
            },
            "operands": reshape_in_operands,
            "connections": {
                "inputs": [activation_conn] if activation_conn else [],
                "outputs": [conv_id],
            },
        }

        stride = self._as_list(module_args.get("stride"), [1, 1] if is_2d else [1])
        padding = self._as_list(module_args.get("padding"), [0, 0] if is_2d else [0])
        dilation = self._as_list(module_args.get("dilation"), [1, 1] if is_2d else [1])

        conv_ts = TensorShapes(
            inputs=[reshaped_input, reshaped_weight],
            outputs=[reshaped_output],
        )
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op(
                node_type,
                conv_ts,
                module_args=module_args,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            conv_equation = einsum_op.equation
            conv_operands = {operand.name: operand.dims for operand in einsum_op.operands}
        except Exception:
            conv_equation = fallback_conv_equation
            if is_2d:
                conv_operands = {
                    "Input": ["B", "G", "I", "P+R", "Q+S"],
                    "Weight": ["G", "O", "I", "R", "S"],
                    "Output": ["B", "G", "O", "P", "Q"],
                }
            else:
                conv_operands = {
                    "Input": ["B", "G", "I", "P+R"],
                    "Weight": ["G", "O", "I", "R"],
                    "Output": ["B", "G", "O", "P"],
                }

        conv_layer = {
            "type": node_type,
            "einsum_equation": conv_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{reshape_in_id}.Output", weight_tensor_name],
                "outputs": [f"{conv_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [reshaped_input, reshaped_weight],
                "outputs": [reshaped_output],
            },
            "tensor_dtypes": {
                "inputs": [activation_dtype, weight_dtype],
                "outputs": [output_dtype],
            },
            "operands": conv_operands,
            "connections": {
                "inputs": [reshape_in_id, weight_conn] if weight_conn else [reshape_in_id],
                "outputs": [reshape_out_id],
            },
        }

        reshape_out_layer = {
            "type": "view",
            "einsum_equation": reshape_out_eq,
            "elementwise_op": "copy",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{conv_id}.Output"],
                "outputs": [f"{reshape_out_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [reshaped_output],
                "outputs": [list(output_shape)],
            },
            "tensor_dtypes": {
                "inputs": [output_dtype],
                "outputs": [output_dtype],
            },
            "operands": reshape_out_operands,
            "connections": {
                "inputs": [conv_id],
                "outputs": output_connections,
            },
        }

        subgraph = {
            reshape_in_id: reshape_in_layer,
            conv_id: conv_layer,
            reshape_out_id: reshape_out_layer,
        }

        input_mapping = {}
        if activation_conn:
            input_mapping[0] = reshape_in_id
        if weight_conn:
            input_mapping[1] = conv_id

        return subgraph, reshape_out_id, input_mapping

    @staticmethod
    def _as_list(value: Any, default: List[int]) -> List[Any]:
        """Normalize scalar/list convolution args to a list."""
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _expand_mha(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: "nx.DiGraph",
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand multi_head_attention_forward into a subgraph.

        MHA decomposes into:
          1. in_proj (linear): input @ in_proj_weight^T  [S*B,D] @ [D,3D] -> [S*B,3D]
          2. qk_matmul: Q @ K^T  [B,H,S,D/H] x [B,H,D/H,S] -> [B,H,S,S]
          3. scale: * 1/sqrt(d_k)
          4. softmax
          5. av_matmul: attn @ V  [B,H,S,S] x [B,H,S,D/H] -> [B,H,S,D/H]
          6. out_proj (linear): concat @ out_proj_weight^T  [S*B,D] @ [D,D] -> [S*B,D]

        The head count cancels for MACs: B*H*S*S*(D/H) = B*S*S*D.
        We use num_heads=1 equivalent shapes so standard equations work.
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_types = [str(t).lower() for t in (node_data.get("input_types") or [])]
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (input_dtypes[i] for i, t in enumerate(input_types)
             if t == "weight" and i < len(input_dtypes)),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        # Parse activation shape: [S, B, D]
        act_shape = input_shapes[0] if input_shapes else []
        if len(act_shape) < 3:
            raise ValueError(f"MHA requires [S,B,D] input. Got: {act_shape}")
        S, B, D = act_shape[0], act_shape[1], act_shape[2]

        # Find weight shapes by type
        in_proj_w_shape = None  # [3D, D]
        out_proj_w_shape = None  # [D, D]
        for i, t in enumerate(input_types):
            if t == "weight" and i < len(input_shapes):
                ws = input_shapes[i]
                if isinstance(ws, list) and len(ws) == 2:
                    if ws[0] == 3 * D and ws[1] == D:
                        in_proj_w_shape = ws
                    elif ws[0] == D and ws[1] == D:
                        out_proj_w_shape = ws

        # Derived shapes for sub-nodes
        proj_shape = [S, B, 3 * D]  # after in_proj
        # Use single-head equivalent: [B, 1, S, D] so H cancels in cost
        # Use Q/K labels for sequence dims to avoid repeated dim in equations
        q_shape = [B, 1, S, D]
        k_transposed_shape = [B, 1, D, S]  # K^T for matmul handler convention
        v_shape = [B, 1, S, D]
        scores_shape = [B, 1, S, S]  # shapes still use S,S (same value)
        attn_out_shape = [B, 1, S, D]
        final_shape = list(output_shapes[0]) if output_shapes else [S, B, D]

        # Build connections
        input_connections = sorted(list(op_graph.predecessors(node_id)))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        output_connections = sorted(list(op_graph.successors(node_id)))

        subgraph: Dict[str, Dict[str, Any]] = {}

        in_proj_id = f"{node_id}.in_proj"
        qk_id = f"{node_id}.qk_matmul"
        scale_id = f"{node_id}.scale"
        softmax_id = f"{node_id}.softmax"
        av_id = f"{node_id}.av_matmul"
        out_proj_id = f"{node_id}.out_proj"

        input_mapping: Dict[int, str] = {0: in_proj_id}
        if len(input_connections) > 1:
            input_mapping[1] = in_proj_id
        if len(input_connections) > 2:
            input_mapping[2] = in_proj_id

        act_input_name = (
            f"{input_connections[0]}.Output" if input_connections
            else f"{node_id}.Input"
        )
        in_proj_w_name = f"{node_id}.in_proj.Weight"
        out_proj_w_name = f"{node_id}.out_proj.Weight"

        # 1. in_proj: input @ in_proj_weight^T
        #    Weight is [N, K] = [3D, D]. Equation MK,NK->MN matches handler convention.
        in_proj_input_shape = [S * B, D]
        in_proj_output_shape = [S * B, 3 * D]
        if in_proj_w_shape:
            subgraph[in_proj_id] = {
                "type": "linear",
                "einsum_equation": "MK,NK->MN",
                "elementwise_op": "mul",
                "reduction_op": "add",
                "is_real_einsum": True,
                "is_einsum_supportable": True,
                "operands": {"Input": ["M", "K"], "Weight": ["N", "K"], "Output": ["M", "N"]},
                "tensor_names": {
                    "inputs": [act_input_name, in_proj_w_name],
                    "outputs": [f"{in_proj_id}.Output"],
                },
                "tensor_types": {
                    "inputs": ["input", "weight"],
                    "outputs": ["output"],
                },
                "tensor_shapes": {
                    "inputs": [in_proj_input_shape, in_proj_w_shape],
                    "outputs": [in_proj_output_shape],
                },
                "tensor_dtypes": {
                    "inputs": [act_dtype, weight_dtype],
                    "outputs": [act_dtype],
                },
                "connections": {
                    "inputs": input_connections[:1],
                    "outputs": [qk_id],
                },
            }

        # 2. qk_matmul: Q @ K^T
        subgraph[qk_id] = {
            "type": "matmul",
            "einsum_equation": "BHQD,BHDK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "operands": {"Input": ["B", "H", "Q", "D"], "Input_1": ["B", "H", "D", "K"], "Output": ["B", "H", "Q", "K"]},
            "tensor_names": {
                "inputs": [f"{in_proj_id}.Output" if in_proj_w_shape else act_input_name,
                           f"{in_proj_id}.Output" if in_proj_w_shape else act_input_name],
                "outputs": [f"{qk_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [q_shape, k_transposed_shape],
                "outputs": [scores_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [in_proj_id] if in_proj_w_shape else input_connections[:1],
                "outputs": [scale_id],
            },
        }

        # 3. scale
        subgraph[scale_id] = {
            "type": "mul",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "mul",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": {"Input": ["B", "H", "Q", "K"], "Output": ["B", "H", "Q", "K"]},
            "tensor_names": {
                "inputs": [f"{qk_id}.Output"],
                "outputs": [f"{scale_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [qk_id],
                "outputs": [softmax_id],
            },
        }

        # 4. softmax
        subgraph[softmax_id] = {
            "type": "softmax",
            "einsum_equation": "BHQK->BHQK",
            "elementwise_op": "softmax",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": {"Input": ["B", "H", "Q", "K"], "Output": ["B", "H", "Q", "K"]},
            "tensor_names": {
                "inputs": [f"{scale_id}.Output"],
                "outputs": [f"{softmax_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [scale_id],
                "outputs": [av_id],
            },
        }

        # 5. av_matmul: attn @ V
        subgraph[av_id] = {
            "type": "matmul",
            "einsum_equation": "BHQK,BHKV->BHQV",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "operands": {"Input": ["B", "H", "Q", "K"], "Input_1": ["B", "H", "K", "V"], "Output": ["B", "H", "Q", "V"]},
            "tensor_names": {
                "inputs": [f"{softmax_id}.Output",
                           f"{in_proj_id}.Output" if in_proj_w_shape else act_input_name],
                "outputs": [f"{av_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape, v_shape],
                "outputs": [attn_out_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, act_dtype],
                "outputs": [act_dtype],
            },
            "connections": {
                "inputs": [softmax_id],
                "outputs": [out_proj_id] if out_proj_w_shape else output_connections,
            },
        }

        # 6. out_proj: Weight is [N, K] = [D, D]. Equation MK,NK->MN.
        final_node_id = av_id
        if out_proj_w_shape:
            out_proj_input_shape = [S * B, D]
            out_proj_output_shape = [S * B, D]
            subgraph[out_proj_id] = {
                "type": "linear",
                "einsum_equation": "MK,NK->MN",
                "elementwise_op": "mul",
                "reduction_op": "add",
                "is_real_einsum": True,
                "is_einsum_supportable": True,
                "operands": {"Input": ["M", "K"], "Weight": ["N", "K"], "Output": ["M", "N"]},
                "tensor_names": {
                    "inputs": [f"{av_id}.Output", out_proj_w_name],
                    "outputs": [f"{out_proj_id}.Output"],
                },
                "tensor_types": {
                    "inputs": ["input", "weight"],
                    "outputs": ["output"],
                },
                "tensor_shapes": {
                    "inputs": [out_proj_input_shape, out_proj_w_shape],
                    "outputs": [out_proj_output_shape],
                },
                "tensor_dtypes": {
                    "inputs": [act_dtype, weight_dtype],
                    "outputs": [out_dtype],
                },
                "connections": {
                    "inputs": [av_id],
                    "outputs": output_connections,
                },
            }
            final_node_id = out_proj_id

        return subgraph, final_node_id, input_mapping

    def _expand_lstm(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: "nx.DiGraph",
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand LSTM into a subgraph of linear operations.

        LSTM decomposes into (per timestep, summed over S steps):
          1. ih_linear: input @ W_ih^T   [S*B, I] @ [4H, I] -> [S*B, 4H]
          2. hh_linear: hidden @ W_hh^T  [S*B, H] @ [4H, H] -> [S*B, 4H]
          3. gate ops (sigmoid, tanh) — elementwise, not real einsums
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_types = [str(t).lower() for t in (node_data.get("input_types") or [])]
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (input_dtypes[i] for i, t in enumerate(input_types)
             if t == "weight" and i < len(input_dtypes)),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        act_shape = input_shapes[0] if input_shapes else []
        if len(act_shape) < 3:
            raise ValueError(f"LSTM requires [S,B,I] input. Got: {act_shape}")
        S, B, I = act_shape[0], act_shape[1], act_shape[2]

        # Find weight shapes
        w_ih_shape = None  # [4H, I]
        w_hh_shape = None  # [4H, H]
        for i, t in enumerate(input_types):
            if t == "weight" and i < len(input_shapes):
                ws = input_shapes[i]
                if isinstance(ws, list) and len(ws) == 2:
                    if ws[1] == I and w_ih_shape is None:
                        w_ih_shape = ws
                    elif w_hh_shape is None:
                        w_hh_shape = ws

        H = w_hh_shape[1] if w_hh_shape else (w_ih_shape[0] // 4 if w_ih_shape else I)

        input_connections = sorted(list(op_graph.predecessors(node_id)))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        output_connections = sorted(list(op_graph.successors(node_id)))

        subgraph: Dict[str, Dict[str, Any]] = {}

        ih_id = f"{node_id}.ih_linear"
        hh_id = f"{node_id}.hh_linear"
        gates_id = f"{node_id}.gates"

        input_mapping: Dict[int, str] = {0: ih_id}
        if len(input_connections) > 1:
            input_mapping[1] = hh_id

        act_input_name = (
            f"{input_connections[0]}.Output" if input_connections
            else f"{node_id}.Input"
        )
        hidden_input_name = (
            f"{input_connections[1]}.Output" if len(input_connections) > 1
            else f"{node_id}.Hidden"
        )

        # Hidden state shape — h0 is [num_layers*num_directions, B, H].
        # Use the shape from input_shapes[1] if available.
        hidden_shape = input_shapes[1] if len(input_shapes) > 1 else [1, B, H]

        G = 4 * H  # gate dimension

        # 1. ih_linear: input @ W_ih^T — [S, B, I] @ [G, I] -> [S, B, G]
        ih_output_shape = [S, B, G]
        subgraph[ih_id] = {
            "type": "linear",
            "einsum_equation": "SBI,GI->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [act_input_name, f"{node_id}.Weight_ih"],
                "outputs": [f"{ih_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(act_shape), w_ih_shape or [G, I]],
                "outputs": [ih_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input":  ["S", "B", "I"],
                "Weight": ["G", "I"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[:1],
                "outputs": [gates_id],
            },
        }

        # 2. hh_linear: hidden @ W_hh^T
        # The hidden projection runs once per timestep but we represent the
        # total work as [S, B, H] @ [G, H] -> [S, B, G] so MACs reflect
        # S steps of B×H×G multiplies.
        hh_input_shape = [S, B, H]
        hh_output_shape = [S, B, G]
        subgraph[hh_id] = {
            "type": "linear",
            "einsum_equation": "SBH,GH->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [hidden_input_name, f"{node_id}.Weight_hh"],
                "outputs": [f"{hh_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [hh_input_shape, w_hh_shape or [G, H]],
                "outputs": [hh_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input":  ["S", "B", "H"],
                "Weight": ["G", "H"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[1:2] if len(input_connections) > 1 else [],
                "outputs": [gates_id],
            },
        }

        # 3. Gate ops (sigmoid/tanh) — elementwise, combines ih + hh results
        final_shape = list(output_shapes[0]) if output_shapes else [S, B, H]
        subgraph[gates_id] = {
            "type": "sigmoid",
            "einsum_equation": "SBH->SBH",
            "elementwise_op": "sigmoid",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{ih_id}.Output", f"{hh_id}.Output"],
                "outputs": [f"{gates_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [ih_output_shape, hh_output_shape],
                "outputs": [final_shape],
            },
            "tensor_dtypes": {
                "inputs": [out_dtype, out_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input":  ["S", "B", "H"],
                "Output": ["S", "B", "H"],
            },
            "connections": {
                "inputs": [ih_id, hh_id],
                "outputs": output_connections,
            },
        }

        final_node_id = gates_id
        return subgraph, final_node_id, input_mapping

    def _expand_gru(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: "nx.DiGraph",
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[int, str]]:
        """Expand GRU into a subgraph of linear operations.

        GRU decomposes into (per timestep, summed over S steps):
          1. ih_linear: input @ W_ih^T   [S*B, I] @ [3H, I] -> [S*B, 3H]
          2. hh_linear: hidden @ W_hh^T  [S*B, H] @ [3H, H] -> [S*B, 3H]
          3. gate ops (sigmoid, tanh) — elementwise, not real einsums
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        input_types = [str(t).lower() for t in (node_data.get("input_types") or [])]
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (input_dtypes[i] for i, t in enumerate(input_types)
             if t == "weight" and i < len(input_dtypes)),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        act_shape = input_shapes[0] if input_shapes else []
        if len(act_shape) < 3:
            raise ValueError(f"GRU requires [S,B,I] input. Got: {act_shape}")
        S, B, I = act_shape[0], act_shape[1], act_shape[2]

        w_ih_shape = None  # [3H, I]
        w_hh_shape = None  # [3H, H]
        for i, t in enumerate(input_types):
            if t == "weight" and i < len(input_shapes):
                ws = input_shapes[i]
                if isinstance(ws, list) and len(ws) == 2:
                    if ws[1] == I and w_ih_shape is None:
                        w_ih_shape = ws
                    elif w_hh_shape is None:
                        w_hh_shape = ws

        H = w_hh_shape[1] if w_hh_shape else (w_ih_shape[0] // 3 if w_ih_shape else I)

        input_connections = sorted(list(op_graph.predecessors(node_id)))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        output_connections = sorted(list(op_graph.successors(node_id)))

        subgraph: Dict[str, Dict[str, Any]] = {}

        ih_id = f"{node_id}.ih_linear"
        hh_id = f"{node_id}.hh_linear"
        gates_id = f"{node_id}.gates"

        input_mapping: Dict[int, str] = {0: ih_id}
        if len(input_connections) > 1:
            input_mapping[1] = hh_id

        act_input_name = (
            f"{input_connections[0]}.Output" if input_connections
            else f"{node_id}.Input"
        )
        hidden_input_name = (
            f"{input_connections[1]}.Output" if len(input_connections) > 1
            else f"{node_id}.Hidden"
        )

        hidden_shape = input_shapes[1] if len(input_shapes) > 1 else [1, B, H]

        G = 3 * H  # gate dimension

        # 1. ih_linear: input @ W_ih^T — [S, B, I] @ [G, I] -> [S, B, G]
        ih_output_shape = [S, B, G]
        subgraph[ih_id] = {
            "type": "linear",
            "einsum_equation": "SBI,GI->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [act_input_name, f"{node_id}.Weight_ih"],
                "outputs": [f"{ih_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [list(act_shape), w_ih_shape or [G, I]],
                "outputs": [ih_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input":  ["S", "B", "I"],
                "Weight": ["G", "I"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[:1],
                "outputs": [gates_id],
            },
        }

        # 2. hh_linear: hidden @ W_hh^T
        # The hidden projection runs once per timestep but we represent the
        # total work as [S, B, H] @ [G, H] -> [S, B, G] so MACs reflect
        # S steps of B×H×G multiplies.
        hh_input_shape = [S, B, H]
        hh_output_shape = [S, B, G]
        subgraph[hh_id] = {
            "type": "linear",
            "einsum_equation": "SBH,GH->SBG",
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [hidden_input_name, f"{node_id}.Weight_hh"],
                "outputs": [f"{hh_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "weight"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [hh_input_shape, w_hh_shape or [G, H]],
                "outputs": [hh_output_shape],
            },
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input":  ["S", "B", "H"],
                "Weight": ["G", "H"],
                "Output": ["S", "B", "G"],
            },
            "connections": {
                "inputs": input_connections[1:2] if len(input_connections) > 1 else [],
                "outputs": [gates_id],
            },
        }

        # 3. Gate ops (sigmoid/tanh) — elementwise, combines ih + hh results
        final_shape = list(output_shapes[0]) if output_shapes else [S, B, H]
        subgraph[gates_id] = {
            "type": "sigmoid",
            "einsum_equation": "SBH->SBH",
            "elementwise_op": "sigmoid",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "tensor_names": {
                "inputs": [f"{ih_id}.Output", f"{hh_id}.Output"],
                "outputs": [f"{gates_id}.Output"],
            },
            "tensor_types": {
                "inputs": ["input", "input"],
                "outputs": ["output"],
            },
            "tensor_shapes": {
                "inputs": [ih_output_shape, hh_output_shape],
                "outputs": [final_shape],
            },
            "tensor_dtypes": {
                "inputs": [out_dtype, out_dtype],
                "outputs": [out_dtype],
            },
            "operands": {
                "Input":  ["S", "B", "H"],
                "Output": ["S", "B", "H"],
            },
            "connections": {
                "inputs": [ih_id, hh_id],
                "outputs": output_connections,
            },
        }

        final_node_id = gates_id
        return subgraph, final_node_id, input_mapping

    def _should_split_linear_with_bias(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a linear layer with bias that should be split."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()

        if node_type != "linear":
            return False

        input_shapes = node_data.get("input_shapes") or []
        input_types = [str(t).lower() for t in (node_data.get("input_types") or [])]

        # Prefer explicit tensor typing: one activation + at least two weight inputs,
        # with at least one rank-1 weight as bias.
        if input_types:
            weight_indices = [i for i, t in enumerate(input_types) if t == "weight"]
            input_indices = [i for i, t in enumerate(input_types) if t == "input"]
            has_rank1_weight = any(
                i < len(input_shapes) and isinstance(input_shapes[i], list) and len(input_shapes[i]) == 1
                for i in weight_indices
            )
            if len(input_indices) >= 1 and len(weight_indices) >= 2 and has_rank1_weight:
                return True
            # Don't early-return False here; input_types can be incomplete
            # in some traced graphs. Fall through to fallback checks.

        # Fallback without input_types: x, weight, bias by shape rank pattern.
        if len(input_shapes) >= 3:
            has_rank1 = any(isinstance(s, list) and len(s) == 1 for s in input_shapes)
            has_rank2_or_more = any(isinstance(s, list) and len(s) >= 2 for s in input_shapes)
            if has_rank1 and has_rank2_or_more:
                return True

        # Fallback: infer from metadata/notes text when shape info is incomplete.
        module_args = node_data.get("module_args") or {}
        if bool(module_args.get("bias", False)):
            return True

        notes_blob = " ".join(
            str(v)
            for v in (
                node_data.get("notes"),
                module_args.get("raw_attributes"),
                module_args.get("function_name"),
            )
            if v is not None
        ).lower()
        if "bias" in notes_blob:
            return True

        return False

    def _validate_input_types_alignment(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Ensure input_types aligns 1:1 with input_shapes for op nodes.

        When the torchview graph collapses multiple tensor inputs into
        fewer connection nodes (e.g. ``cat`` receiving two tensors via a
        single hidden-tensor node), ``input_types`` will be shorter than
        ``input_shapes``.  Pad with ``'input'`` to restore alignment,
        since the missing entries are always activation (non-weight)
        tensors.

        If ``input_types`` is *longer* than ``input_shapes``, that
        indicates a graph construction bug and we raise immediately.
        """
        input_shapes = node_data.get("input_shapes") or []
        input_types = node_data.get("input_types") or []
        if len(input_types) < len(input_shapes):
            input_types = list(input_types) + ["input"] * (len(input_shapes) - len(input_types))
            node_data["input_types"] = input_types
        elif len(input_types) > len(input_shapes):
            node_type = node_data.get("type", "unknown")
            raise ValueError(
                f"Node '{node_id}' (type={node_type}) has more input_types "
                f"({len(input_types)}) than input_shapes ({len(input_shapes)}). "
                "This indicates a graph construction bug."
            )

    def _split_linear_with_bias(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split a linear layer with bias into matmul + add operations.

        Returns:
            Tuple of (matmul_layer_dict, add_layer_dict)
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []

        # Keep original input order from PyTorch graph; don't sort.
        node_connections = (node_data.get("connections") or {}).get("inputs") or []
        input_connections = list(node_connections)
        for pred in op_graph.predecessors(node_id):
            if pred not in input_connections:
                input_connections.append(pred)
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)

        # Use collapsed op-graph successors so tensor nodes (e.g. hidden-tensor)
        # are not emitted in einsum connections.
        output_connections = list(op_graph.successors(node_id))
        if not output_connections:
            raw_output_connections = list((node_data.get("connections") or {}).get("outputs") or [])
            output_connections = [c for c in raw_output_connections if c in op_graph.nodes]

        # Extract dtypes from the original node for propagation to sub-nodes.
        input_types = node_data.get("input_types") or []
        input_dtypes = node_data.get("input_dtypes") or []
        output_dtypes = node_data.get("output_dtypes") or []
        act_dtype = input_dtypes[0] if input_dtypes else "torch.float32"
        weight_dtype = next(
            (input_dtypes[i] for i, t in enumerate(input_types)
             if str(t).lower() == "weight" and i < len(input_dtypes)),
            act_dtype,
        )
        out_dtype = output_dtypes[0] if output_dtypes else act_dtype

        # Infer x/weight/bias from ordered inputs + input_shapes.
        typed_inputs: List[Tuple[int, str, Any, str]] = []
        for idx, conn in enumerate(input_connections):
            ishape = input_shapes[idx] if idx < len(input_shapes) else None
            itype = input_types[idx] if idx < len(input_types) else "input"
            typed_inputs.append((idx, conn, ishape, str(itype)))

        activation_entry: Optional[Tuple[int, str, Any, str]] = None
        weight_entries: List[Tuple[int, str, Any, str]] = []
        for entry in typed_inputs:
            _, conn, _, itype = entry
            if itype == "weight" or "parameter-tensor" in conn:
                weight_entries.append(entry)
            elif activation_entry is None:
                activation_entry = entry

        if activation_entry is None and typed_inputs:
            activation_entry = typed_inputs[0]

        # Bias is normally rank-1 among weight inputs.
        bias_entry: Optional[Tuple[int, str, Any, str]] = None
        for entry in weight_entries:
            ishape = entry[2]
            if isinstance(ishape, list) and len(ishape) == 1:
                bias_entry = entry
                break

        # Fallback when rank-based inference fails: last weight is bias.
        if bias_entry is None and len(weight_entries) >= 2:
            bias_entry = weight_entries[-1]

        # Weight matrix is a non-bias weight, preferring rank-2.
        weight_entry: Optional[Tuple[int, str, Any, str]] = None
        for entry in weight_entries:
            if bias_entry is not None and entry[1] == bias_entry[1]:
                continue
            ishape = entry[2]
            if isinstance(ishape, list) and len(ishape) >= 2:
                weight_entry = entry
                break
        if weight_entry is None:
            for entry in weight_entries:
                if bias_entry is None or entry[1] != bias_entry[1]:
                    weight_entry = entry
                    break

        weight_shape = list(weight_entry[2]) if (weight_entry and isinstance(weight_entry[2], list)) else None
        bias_shape = list(bias_entry[2]) if (bias_entry and isinstance(bias_entry[2], list)) else None

        # Intermediate shape (output of matmul, input to add)
        matmul_output_shape = output_shapes[0] if output_shapes else []

        # === MATMUL LAYER ===
        # Get einsum equation for matmul
        matmul_input_shapes_for_equation: List[List[Any]] = []
        if activation_entry and isinstance(activation_entry[2], list):
            matmul_input_shapes_for_equation.append(list(activation_entry[2]))
        if weight_entry and isinstance(weight_entry[2], list):
            matmul_input_shapes_for_equation.append(list(weight_entry[2]))
        matmul_ts = TensorShapes(
            inputs=matmul_input_shapes_for_equation,
            outputs=list(node_data.get("output_shapes") or []),
        )
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op("linear", matmul_ts)
            matmul_equation = einsum_op.equation
            matmul_operands = {o.name: list(o.dims) for o in einsum_op.operands}
        except Exception:
            # Fallback equation
            batch_dims = len(input_shapes[0]) - 1 if input_shapes else 0
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            input_str = ''.join(batch_letters + ["K"])
            weight_str = "NK"
            output_str = ''.join(batch_letters + ["N"])
            matmul_equation = f"{input_str},{weight_str}->{output_str}"
            # Parse the fallback equation back into the operand structure.
            # Tokens are an uppercase letter optionally followed by digits (e.g. B0).
            _toks = lambda s: re.findall(r"[A-Z]\d*", s)
            matmul_operands = {
                "Input": _toks(input_str),
                "Weight": _toks(weight_str),
                "Output": _toks(output_str),
            }

        add_node_id = f"{node_id}.bias_add"

        matmul_input_names: List[str] = []
        matmul_input_shapes_list: List[List[Any]] = []
        matmul_connection_inputs: List[str] = []

        if activation_entry:
            activation_conn_id = activation_entry[1]
            # Activation tensors should reference the canonical start node IDs
            # (e.g. start/start_1) after tensor-node collapse.
            activation_einsum_id = start_node_id_map.get(activation_conn_id, activation_conn_id)
            matmul_input_names.append(f"{activation_einsum_id}.Output")
            if isinstance(activation_entry[2], list):
                matmul_input_shapes_list.append(list(activation_entry[2]))
            matmul_connection_inputs.append(activation_einsum_id)
        if weight_entry:
            matmul_input_names.append(f"{weight_entry[1]}.Output")
            if isinstance(weight_entry[2], list):
                matmul_input_shapes_list.append(list(weight_entry[2]))
            matmul_connection_inputs.append(weight_entry[1])

        matmul_tensor_names = {
            "inputs": matmul_input_names,
            "outputs": [f"{node_id}.Output"],
        }
        matmul_tensor_types = {
            "inputs": ["input" if i == 0 else "weight" for i in range(len(matmul_input_names))],
            "outputs": ["output"],
        }
        matmul_tensor_shapes = {
            "inputs": matmul_input_shapes_list,
            "outputs": [list(matmul_output_shape)] if matmul_output_shape else [],
        }

        matmul_layer: Dict[str, Any] = {
            # Keep type as linear so MACs are computed by LinearHandler.
            "type": "linear",
            "einsum_equation": matmul_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            # Operands drive the AF graph builder; without them the layer
            # is silently skipped from the AF einsums list.
            "operands": matmul_operands,
            "tensor_names": matmul_tensor_names,
            "tensor_types": matmul_tensor_types,
            "tensor_shapes": matmul_tensor_shapes,
            "tensor_dtypes": {
                "inputs": [act_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": matmul_connection_inputs,
                "outputs": [add_node_id],  # Output goes to bias_add
            },
        }

        if weight_shape:
            matmul_layer["additional_info"] = {
                "weights": [{"name": "Weight", "shape": list(weight_shape)}]
            }

        # === ADD (BIAS) LAYER ===
        # Generate einsum equation for bias add (broadcast add).
        if matmul_output_shape and bias_shape and len(matmul_output_shape) >= 1 and len(bias_shape) == 1:
            labels = string.ascii_uppercase[:len(matmul_output_shape)]
            add_equation = f"{labels},{labels[-1]}->{labels}"
            add_operands = {
                "Input": list(labels),
                "Weight": [labels[-1]],
                "Output": list(labels),
            }
        elif matmul_output_shape:
            labels = string.ascii_uppercase[:len(matmul_output_shape)]
            add_equation = f"{labels}->{labels}"
            add_operands = {
                "Input": list(labels),
                "Output": list(labels),
            }
        else:
            add_equation = ""
            add_operands = None

        add_input_names = [f"{node_id}.Output"]
        add_input_shapes_list = [list(matmul_output_shape)] if matmul_output_shape else []
        add_connection_inputs = [node_id]

        if bias_entry and bias_shape:
            add_input_names.append(f"{bias_entry[1]}.Output")
            add_input_shapes_list.append(list(bias_shape))
            add_connection_inputs.append(bias_entry[1])

        add_tensor_names = {
            "inputs": add_input_names,
            "outputs": [f"{add_node_id}.Output"],
        }
        add_tensor_types = {
            "inputs": ["input"] + (["weight"] if len(add_input_names) > 1 else []),
            "outputs": ["output"],
        }
        add_tensor_shapes = {
            "inputs": add_input_shapes_list,
            "outputs": [list(output_shapes[0])] if output_shapes else [],
        }

        add_layer: Dict[str, Any] = {
            "type": "add",
            "einsum_equation": add_equation,
            "elementwise_op": "add",
            "reduction_op": "none",
            "is_real_einsum": False,
            "is_einsum_supportable": True,
            "operands": add_operands,
            "tensor_names": add_tensor_names,
            "tensor_types": add_tensor_types,
            "tensor_shapes": add_tensor_shapes,
            "tensor_dtypes": {
                "inputs": [out_dtype, weight_dtype],
                "outputs": [out_dtype],
            },
            "connections": {
                "inputs": add_connection_inputs,
                "outputs": output_connections,  # Original outputs
            },
        }

        # Add bias info
        if bias_shape:
            add_layer["additional_info"] = {
                "weights": [{"name": "Bias", "shape": list(bias_shape)}]
            }

        return matmul_layer, add_layer

    def _fix_split_connections(
        self,
        result: Dict[str, Any],
        node_id_remap: Dict[str, str],
        expanded_input_map: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """Fix connections for layers that reference split/expanded operations.

        When an operation is split/expanded:
        1. Downstream layers that consume the output should reference the final node
        2. Upstream layers (predecessors) should have their outputs updated to
           reference the correct subgraph entry node

        Args:
            result: The einsum graph dictionary being built.
            node_id_remap: Maps original node_id -> final output node_id.
            expanded_input_map: Maps original node_id -> {input_index -> subgraph_node_id}.
        """
        if expanded_input_map is None:
            expanded_input_map = {}

        if not node_id_remap and not expanded_input_map:
            return

        # First pass: Update predecessor outputs for expanded operations
        for original_node_id, input_mapping in expanded_input_map.items():
            # Find all layers that have the original_node_id in their outputs
            for layer_id, layer_data in result["layers"].items():
                connections = layer_data.get("connections", {})
                outputs = connections.get("outputs", [])

                if original_node_id in outputs:
                    # This layer was a predecessor to the expanded node
                    # Find which input index this layer corresponds to
                    # by looking at the subgraph's inputs
                    new_outputs = []
                    for out in outputs:
                        if out == original_node_id:
                            # Determine which subgraph node this layer feeds into
                            # based on which input it provides
                            # We need to find the correct entry node
                            target_node = self._find_entry_node_for_predecessor(
                                result, layer_id, original_node_id, input_mapping
                            )
                            new_outputs.append(target_node)
                        else:
                            new_outputs.append(out)
                    connections["outputs"] = new_outputs

        # Second pass: Update downstream references
        for layer_id, layer_data in result["layers"].items():
            connections = layer_data.get("connections", {})
            inputs = connections.get("inputs", [])

            # Update input connections to reference final output node
            new_inputs = []
            for inp in inputs:
                # BUGFIX: Don't remap if the current layer is itself the target of the remapping
                # (e.g., don't replace Model.linear -> Model.linear.bias_add in Model.linear.bias_add's own inputs)
                # This prevents creating self-loops in split layers like bias_add
                if inp in node_id_remap and node_id_remap[inp] != layer_id:
                    new_inputs.append(node_id_remap[inp])
                else:
                    new_inputs.append(inp)
            connections["inputs"] = new_inputs

            # Update tensor_names inputs
            tensor_names = layer_data.get("tensor_names", {})
            if tensor_names:
                input_names = tensor_names.get("inputs", [])
                new_input_names = []
                for name in input_names:
                    for old_id, new_id in node_id_remap.items():
                        # Keep split node self-inputs stable (e.g. bias_add should
                        # consume Model.linear.Output, not its own output).
                        if new_id == layer_id:
                            continue
                        if name == f"{old_id}.Output" or name.startswith(f"{old_id}.Output_"):
                            name = name.replace(f"{old_id}.", f"{new_id}.", 1)
                            break
                    new_input_names.append(name)
                tensor_names["inputs"] = new_input_names

    def _find_entry_node_for_predecessor(
        self,
        result: Dict[str, Any],
        predecessor_id: str,
        original_node_id: str,
        input_mapping: Dict[int, str],
    ) -> str:
        """Find which subgraph entry node a predecessor should connect to.

        Args:
            result: The einsum graph dictionary.
            predecessor_id: ID of the predecessor layer.
            original_node_id: ID of the original (expanded) node.
            input_mapping: Maps input index -> subgraph node that receives it.

        Returns:
            The subgraph node ID that this predecessor should connect to.
        """
        # Look at the subgraph nodes to find which one has this predecessor in its inputs
        for subgraph_node_id in input_mapping.values():
            if subgraph_node_id in result["layers"]:
                subgraph_layer = result["layers"][subgraph_node_id]
                subgraph_inputs = subgraph_layer.get("connections", {}).get("inputs", [])
                if predecessor_id in subgraph_inputs:
                    return subgraph_node_id

        # Default: return the first entry node (qk_matmul for SDPA)
        if input_mapping:
            return input_mapping.get(0, list(input_mapping.values())[0])

        return original_node_id

    def _add_start_nodes(
        self,
        result: Dict[str, Any],
        start_nodes_info: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Add start nodes to the einsum graph."""
        start_node_id_map: Dict[str, str] = {}

        for info in start_nodes_info:
            idx = info["index"]
            start_id = "start" if idx == 0 else f"start_{idx}"
            original_id = info["original_id"]
            start_node_id_map[original_id] = start_id

            output_shapes = info.get("output_shapes") or []
            consumers = info.get("consumers", [])

            # Build tensor_names
            output_names = [f"{start_id}.Output"]
            for i in range(1, len(output_shapes)):
                output_names.append(f"{start_id}.Output_{i}")

            tensor_names = {
                "inputs": [],  # Start nodes have no inputs
                "outputs": output_names,
            }

            # Build tensor_shapes
            tensor_shapes = {
                "inputs": [],  # Start nodes have no inputs
                "outputs": [list(s) for s in output_shapes],
            }

            # Generate einsum equation
            equation = ""
            operands = {}
            if output_shapes and len(output_shapes[0]) > 0:
                dims = len(output_shapes[0])
                labels = [f"{c}{idx}" for c in string.ascii_uppercase[:dims]]
                equation = f"->{''.join(labels)}"
                operands = {start_id: labels}

            layer_dict: Dict[str, Any] = {
                "type": "start",
                "einsum_equation": equation,
                "elementwise_op": "copy",
                "reduction_op": "none",
                "is_real_einsum": False,
                "is_einsum_supportable": False,
                "tensor_names": tensor_names,
                "tensor_types": {
                    "inputs": [],
                    "outputs": ["input" for _ in output_names],
                },
                "tensor_shapes": tensor_shapes,
                "operands": operands,
                "connections": {
                    "inputs": [],
                    "outputs": consumers,
                },
            }

            # Propagate dtype info for start nodes
            output_dtypes = info.get("output_dtypes") or []
            if output_dtypes:
                layer_dict["tensor_dtypes"] = {
                    "inputs": [],
                    "outputs": output_dtypes,
                }

            result["layers"][start_id] = layer_dict

        return start_node_id_map


    def _convert_operation(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Convert a single operation to einsum representation."""
        node_type_raw = node_data.get("type", "unknown")
        node_type = self._einsum_analyzer._get_operation_from_name(str(node_type_raw))

        ts = TensorShapes(
            inputs=list(node_data.get("input_shapes") or []),
            outputs=list(node_data.get("output_shapes") or []),
        )

        # Get module_args for operations like transpose/permute
        module_args = node_data.get("module_args", {})

        # Try to get einsum representation
        equation = ""
        operands = {}
        elementwise_op = "mul"
        reduction_op = "add"
        is_real_einsum = True
        is_einsum_supportable = True

        # Special handling for torch.einsum operations - parse the equation from raw_attributes
        if node_type == "einsum":
            parsed_equation = self._parse_einsum_from_raw_attributes(module_args)
            if parsed_equation:
                equation = parsed_equation
                elementwise_op = "mul"
                reduction_op = "add"
                is_real_einsum = True
                is_einsum_supportable = True
                lhs, rhs = parsed_equation.split("->")
                lhs_parts = lhs.split(",")
                operands = {"Input": list(lhs_parts[0]), "Output": list(rhs)}
                if len(lhs_parts) == 2:
                    operands["Weight"] = list(lhs_parts[1])
            else:
                # Fallback: try to get from analyzer
                try:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, ts, module_args=module_args
                    )
                    equation = einsum_op.equation
                    elementwise_op = einsum_op.elementwise_op
                    reduction_op = einsum_op.reduction_op
                    is_real_einsum = einsum_op.is_real_einsum
                    is_einsum_supportable = einsum_op.is_einsum_supportable
                    operands = {operand.name: operand.dims for operand in einsum_op.operands}
                except Exception:
                    equation = ""
                    is_einsum_supportable = True
        else:
            # For reduction operations, parse dim and keepdim from raw_attributes
            # Based on PyTorch docs: https://docs.pytorch.org/docs/stable/nn.functional.html
            # These operations support dim and keepdim parameters:
            # - sum, mean, prod: standard reductions
            # - max, min, amax, amin: value reductions
            # - argmax, argmin: index reductions
            # - logsumexp, norm: special reductions
            # - std, var: statistical reductions
            # - all, any: boolean reductions
            reduce_dims = None
            keepdim = False
            reduction_ops_with_dim = {
                "sum", "mean", "prod",
                "max", "min", "amax", "amin",
                "argmax", "argmin",
                "logsumexp", "norm",
                "std", "var",
                "all", "any",
                "nansum", "nanmean",
            }
            if node_type in reduction_ops_with_dim:
                reduce_dims, keepdim = self._parse_reduction_args_from_raw_attributes(module_args)

            try:
                # Pass module_args, reduce_dims, and keepdim to the analyzer
                if reduce_dims is not None:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, ts, module_args=module_args, dims=reduce_dims, keepdim=keepdim
                    )
                else:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, ts, module_args=module_args, keepdim=keepdim
                    )
                equation = einsum_op.equation
                elementwise_op = einsum_op.elementwise_op
                reduction_op = einsum_op.reduction_op
                is_real_einsum = einsum_op.is_real_einsum
                is_einsum_supportable = einsum_op.is_einsum_supportable
                operands = {operand.name: operand.dims for operand in einsum_op.operands}
            except Exception:
                equation = ""
                is_einsum_supportable = self._is_operation_supportable(node_type)

                # Set default ops based on node type
                if node_type in {"add", "sub", "mul", "div"}:
                    elementwise_op = node_type
                    reduction_op = "none"
                    is_real_einsum = False
                elif node_type in {"sum", "mean"}:
                    elementwise_op = "copy"
                    reduction_op = "add"
                    is_real_einsum = False
                elif node_type == "prod":
                    elementwise_op = "copy"
                    reduction_op = "mul"
                    is_real_einsum = False
                elif node_type in {"max", "min"}:
                    elementwise_op = "copy"
                    reduction_op = node_type
                    is_real_einsum = False

        # Build input connections preserving the original PyTorch argument order.
        # Then map tensor-node IDs to start-node IDs where applicable.
        raw_input_connections = list((node_data.get("connections") or {}).get("inputs") or [])
        if not raw_input_connections:
            raw_input_connections = list(op_graph.predecessors(node_id))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                original_id = info["original_id"]
                if original_id not in raw_input_connections:
                    raw_input_connections.append(original_id)

        # Resolve each input tensor to its producer op. Priority:
        #   1. weight slots → keep the mapped value as-is
        #   2. tensors that map directly to a start node or op node
        #   3. hidden tensors with a known producer in _tensor_to_producer_op
        #      (populated by _build_op_graph, including torchview-orphan
        #      reconciliation for shape-donor patterns like zeros_like)
        # Any remaining unresolved inputs are filled from op_graph.predecessors
        # only if the count matches exactly — i.e. there is a unique 1-to-1
        # assignment. Mismatched counts raise, replacing the legacy positional
        # fallback that silently mis-attributed the cat inputs in kernbench
        # L1/92 cumsum_exclusive.
        tensor_to_producer = getattr(self, "_tensor_to_producer_op", {})
        op_predecessors = list(op_graph.predecessors(node_id))
        input_connections: List[Optional[str]] = []
        assigned_preds: set = set()
        deferred_indices: List[int] = []
        input_types_raw = list(node_data.get("input_types") or [])
        for i, conn_id in enumerate(raw_input_connections):
            mapped = start_node_id_map.get(conn_id, conn_id)
            itype = str(input_types_raw[i]).lower() if i < len(input_types_raw) else "input"
            if itype == "weight":
                input_connections.append(mapped)
                continue
            if mapped in start_node_id_map.values() or mapped in op_graph.nodes:
                input_connections.append(mapped)
                assigned_preds.add(mapped)
                continue
            producer = tensor_to_producer.get(conn_id)
            if producer is not None and producer in op_graph.nodes:
                input_connections.append(producer)
                assigned_preds.add(producer)
                continue
            input_connections.append(None)
            deferred_indices.append(i)
        if deferred_indices:
            unmatched_preds = [p for p in op_predecessors if p not in assigned_preds]
            if len(unmatched_preds) == len(deferred_indices):
                # Unambiguous positional match.
                for idx, pred in zip(deferred_indices, unmatched_preds):
                    input_connections[idx] = pred
            elif len(unmatched_preds) == 0:
                # No op-graph candidates — the input is genuinely producerless
                # (e.g. BERT's position-ids tensor created by an untraced
                # `torch.arange`). Fall through to the literal tensor id, which
                # downstream code handles as a phantom name. NOT the L1/92
                # pattern, which has unmatched_preds != 0.
                for idx in deferred_indices:
                    input_connections[idx] = start_node_id_map.get(
                        raw_input_connections[idx], raw_input_connections[idx]
                    )
            else:
                # Truly ambiguous: there are candidate predecessors but the
                # count does not match. This was the kernbench L1/92
                # cumsum_exclusive failure: 2 orphan inputs mapped to 1
                # predecessor. Refuse to guess.
                unresolved = [raw_input_connections[i] for i in deferred_indices]
                raise ValueError(
                    f"_convert_operation({node_id!r}): cannot uniquely "
                    f"resolve inputs {unresolved}. deferred="
                    f"{len(deferred_indices)}, unmatched_preds="
                    f"{len(unmatched_preds)}. Producer attribution is "
                    f"ambiguous — usually a torchview tracing gap that "
                    f"_build_op_graph's reconciliation pass couldn't "
                    f"shape-match uniquely."
                )

        # Take tensor input/output types directly from PyTorch graph by index.
        pytorch_input_types = list(node_data.get("input_types") or [])
        if len(pytorch_input_types) < len(input_connections):
            pytorch_input_types.extend(["input"] * (len(input_connections) - len(pytorch_input_types)))

        # Inject input_types into node_data so downstream functions can use it.
        node_data_with_types = dict(node_data)
        node_data_with_types["input_types"] = pytorch_input_types

        output_connections = sorted(list(op_graph.successors(node_id)))

        # Build tensor_names using input_types
        tensor_names = self._build_tensor_names(
            node_id, node_data_with_types, input_connections, output_connections
        )
        pytorch_output_types = list(node_data.get("output_types") or [])
        if len(pytorch_output_types) < len(tensor_names.get("outputs", [])):
            pytorch_output_types.extend(
                ["output"] * (len(tensor_names.get("outputs", [])) - len(pytorch_output_types))
            )
        tensor_types = {
            "inputs": list(pytorch_input_types[: len(tensor_names.get("inputs", []))]),
            "outputs": list(pytorch_output_types[: len(tensor_names.get("outputs", []))]),
        }

        # Build tensor_shapes: shapes matching tensor_names order
        tensor_shapes = self._build_tensor_shapes(node_data)

        # Validate tensor_names and tensor_shapes match
        is_valid, error_msg = validate_tensor_names_match_shapes(tensor_names, tensor_shapes)
        if not is_valid:
            # Fix mismatch by aligning counts
            tensor_names, tensor_shapes = self._align_tensor_names_and_shapes(
                tensor_names, tensor_shapes, node_data
            )
            tensor_types["inputs"] = tensor_types["inputs"][: len(tensor_names.get("inputs", []))]
            tensor_types["outputs"] = tensor_types["outputs"][: len(tensor_names.get("outputs", []))]

        # Build additional_info for weight/bias metadata
        additional_info = self._build_additional_info(node_data)

        # Filter out weight connections from connections.inputs
        # (parameter nodes don't exist as layers in the einsum graph)
        activation_connections = [
            c for i, c in enumerate(input_connections)
            if not (
                i < len(pytorch_input_types)
                and str(pytorch_input_types[i]).lower() == "weight"
            )
        ]

        # Propagate dtype info from pytorch graph so downstream stages
        # (graph_analyzer) can detect non-standard dtypes like torch.bool.
        input_dtypes = list(node_data.get("input_dtypes") or [])
        output_dtypes = list(node_data.get("output_dtypes") or [])
        tensor_dtypes: Dict[str, Any] = {}
        if input_dtypes:
            tensor_dtypes["inputs"] = input_dtypes
        if output_dtypes:
            tensor_dtypes["outputs"] = output_dtypes

        result: Dict[str, Any] = {
            "type": node_type,
            "einsum_equation": equation,
            "elementwise_op": elementwise_op,
            "reduction_op": reduction_op,
            "is_real_einsum": is_real_einsum,
            "is_einsum_supportable": is_einsum_supportable,
            "tensor_names": tensor_names,
            "tensor_types": tensor_types,
            "tensor_shapes": tensor_shapes,
            "operands": operands,
            "connections": {
                "inputs": activation_connections,
                "outputs": output_connections,
            },
        }

        if tensor_dtypes:
            result["tensor_dtypes"] = tensor_dtypes

        if additional_info:
            result["additional_info"] = additional_info

        # Pass through raw_attributes from module_args if present
        raw_attributes = module_args.get("raw_attributes")
        if raw_attributes:
            result["raw_attributes"] = raw_attributes

        return result

    def _build_tensor_names(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        input_connections: List[str],
        output_connections: List[str],
    ) -> Dict[str, List[str]]:
        """Build tensor names matching input_shapes/output_shapes order.

        Uses input_types to name weight inputs as <node_id>.Weight
        and activation inputs as <predecessor_id>.Output.
        """
        input_names: List[str] = []
        output_names: List[str] = []
        input_types = node_data.get("input_types") or []

        weight_idx = 0
        for i, pred_id in enumerate(input_connections):
            itype = input_types[i] if i < len(input_types) else 'input'
            if itype == 'weight':
                name = f"{node_id}.Weight" if weight_idx == 0 else f"{node_id}.Weight_{weight_idx}"
                input_names.append(name)
                weight_idx += 1
            else:
                input_names.append(f"{pred_id}.Output")

        # Output tensors
        output_names.append(f"{node_id}.Output")
        output_shapes = node_data.get("output_shapes") or []
        for i in range(1, len(output_shapes)):
            output_names.append(f"{node_id}.Output_{i}")

        return {
            "inputs": input_names,
            "outputs": output_names,
        }

    def _build_tensor_shapes(
        self,
        node_data: Dict[str, Any],
    ) -> Dict[str, List[List[int]]]:
        """Build tensor shapes matching input_shapes/output_shapes order.

        All inputs (activation + weight) are already in input_shapes in arg order.
        """
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []

        return {
            "inputs": [list(s) for s in input_shapes],
            "outputs": [list(s) for s in output_shapes],
        }

    def _align_tensor_names_and_shapes(
        self,
        tensor_names: Dict[str, List[str]],
        tensor_shapes: Dict[str, List[List[int]]],
        node_data: Dict[str, Any],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[List[int]]]]:
        """Align tensor_names and tensor_shapes to have matching counts.

        When there's a mismatch (e.g., weight_nodes vs weight_shapes have different lengths),
        this method aligns them by using the shapes as the source of truth and generating
        placeholder names if needed, or trimming excess names.
        """
        input_names = tensor_names.get("inputs", [])
        output_names = tensor_names.get("outputs", [])
        input_shapes = tensor_shapes.get("inputs", [])
        output_shapes = tensor_shapes.get("outputs", [])

        # Align inputs
        if len(input_names) != len(input_shapes):
            # Use shapes as source of truth
            if len(input_shapes) > len(input_names):
                # Add placeholder names for missing entries
                node_id = node_data.get("id", "unknown")
                for i in range(len(input_names), len(input_shapes)):
                    input_names.append(f"{node_id}.Input_{i}")
            else:
                # Trim excess names
                input_names = input_names[:len(input_shapes)]

        # Align outputs
        if len(output_names) != len(output_shapes):
            if len(output_shapes) > len(output_names):
                node_id = node_data.get("id", "unknown")
                for i in range(len(output_names), len(output_shapes)):
                    output_names.append(f"{node_id}.Output_{i}")
            else:
                output_names = output_names[:len(output_shapes)]

        return (
            {"inputs": input_names, "outputs": output_names},
            {"inputs": input_shapes, "outputs": output_shapes},
        )

    def _build_additional_info(
        self,
        node_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build additional_info metadata (weight info is in tensor_names/tensor_shapes)."""
        return {}

    def _is_operation_supportable(self, op_type: str) -> bool:
        """Check if an operation can be expressed with extended einsum."""
        op = op_type.lower()

        # Check against known supportable operations
        if op in _ALL_SUPPORTABLE_OPS:
            return True

        # Check for suffixed matches
        for supported_op in _ALL_SUPPORTABLE_OPS:
            if op.endswith(f".{supported_op}"):
                return True

        # Check prefixed patterns
        if any(op.startswith(prefix) for prefix in ["torch.", "nn.", "functional."]):
            stripped = op.split(".")[-1]
            return stripped in _ALL_SUPPORTABLE_OPS

        # Default: supportable unless explicitly unsupportable
        return op not in _UNSUPPORTABLE_OPS







# Backward compatibility alias
PyTorchEinsumConverter = PyTorchToEinsum


__all__ = [
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
]
