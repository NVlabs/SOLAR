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
        shapes: {<operand>: <shape>, ...}
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import yaml

from solar.common.utils import (
    ensure_directory, 
    NoAliasDumper,
    validate_tensor_names_match_shapes,
)
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

_RNN_OPS = frozenset[str]({"gru", "lstm", "rnn"})

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
    ) -> Tuple[Optional[int], bool]:
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
            Tuple of (reduction_dim, keepdim).
        """
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None, False
        
        reduce_dim = None
        keepdim = False
        
        # Match dim: <number> or dim: [<numbers>]
        # Pattern for single dim: dim: 1 or dim: -1
        single_dim_pattern = r"dim:\s*(-?\d+)"
        match = re.search(single_dim_pattern, raw_attrs)
        if match:
            reduce_dim = int(match.group(1))
        else:
            # Pattern for list of dims: dim: [1, 2]
            list_dim_pattern = r"dim:\s*\[([^\]]+)\]"
            match = re.search(list_dim_pattern, raw_attrs)
            if match:
                dims_str = match.group(1)
                # Return first dim for now (could return list for multi-dim reduction)
                dims = [int(d.strip()) for d in dims_str.split(",")]
                reduce_dim = dims[0] if dims else None
        
        # Match keepdim: True or keepdim: False
        keepdim_pattern = r"keepdim:\s*(True|False)"
        match = re.search(keepdim_pattern, raw_attrs)
        if match:
            keepdim = match.group(1) == "True"
        
        return reduce_dim, keepdim

    def convert(
        self,
        pytorch_graph_path: PathLike,
        output_dir: PathLike,
        *,
        copy_graph: bool = True,
        expand_complex_ops: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Convert a PyTorch graph to einsum representation.
        
        This method:
        1. Loads the PyTorch graph
        2. Builds an operation-only graph (collapsing tensor nodes)
        3. Converts operations to einsum notation
        4. Writes einsum_graph.yaml
        5. Renames ranks using BFS and writes einsum_graph_renamed.yaml

        Args:
            pytorch_graph_path: Path to pytorch_graph.yaml (or legacy JSON).
            output_dir: Directory to write output files.
            copy_graph: If True, copy input graph to output directory.
            expand_complex_ops: If True, attempt to expand complex operations.

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
        op_graph, start_nodes_info = self._build_op_graph(pytorch_graph)

        # Optional complex operation expansion
        if expand_complex_ops:
            op_graph = self._expand_complex_ops(op_graph)

        # Build einsum graph dictionary
        einsum_graph = self._build_einsum_graph(
            pytorch_graph, op_graph, start_nodes_info
        )

        # Add TACO expressions to all layers
        einsum_graph = add_taco_expressions(einsum_graph)

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

        # Rename ranks using BFS traversal
        renamer = EinsumRankRenamer(debug=self._debug)
        renamed_path = out_dir / "einsum_graph_renamed.yaml"
        renamer.rename(einsum_graph, renamed_path)

        if self._debug:
            print(f"✅ Wrote renamed einsum graph: {renamed_path}")

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
    ) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
        """Build operation-only graph by collapsing tensor nodes.
        
        The input PyTorch graph is typically bipartite (TensorNodes and
        Function/Module nodes). This method collapses tensors and connects
        producer operations to consumer operations.
        
        Args:
            pytorch_graph: The PyTorch graph dictionary.
            
        Returns:
            Tuple of (operation graph, start node information).
        """
        layers = pytorch_graph.get("layers") or {}
        tensor_ids, op_ids, auxiliary_ids = self._partition_nodes(layers)

        graph = nx.DiGraph()
        for op_id in op_ids:
            graph.add_node(op_id, **(layers.get(op_id) or {}))

        # Collect auxiliary tensor info for start nodes
        start_nodes_info = self._collect_start_node_info(
            layers, auxiliary_ids, op_ids
        )

        # Connect operations via tensor producers/consumers
        for tensor_id in tensor_ids:
            tensor_data = layers.get(tensor_id) or {}
            conns = tensor_data.get("connections") or {}
            producers = list(conns.get("inputs") or [])
            consumers = list(conns.get("outputs") or [])
            
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

        return graph, start_nodes_info

    def _partition_nodes(
        self,
        layers: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Partition nodes into tensor, operation, and auxiliary categories.
        
        Args:
            layers: The layers dictionary from the PyTorch graph.
            
        Returns:
            Tuple of (tensor_ids, op_ids, auxiliary_tensor_ids).
        """
        tensor_ids: List[str] = []
        op_ids: List[str] = []
        auxiliary_ids: List[str] = []
        
        for node_id, data in (layers or {}).items():
            node_class = (data.get("node_class") or "").lower()
            node_type = (data.get("type") or "").lower()
            
            if node_type == "auxiliary-tensor":
                auxiliary_ids.append(node_id)
            elif "tensornode" in node_class or "tensor" in node_type:
                tensor_ids.append(node_id)
            else:
                op_ids.append(node_id)
                
        return tensor_ids, op_ids, auxiliary_ids

    def _collect_start_node_info(
        self,
        layers: Dict[str, Any],
        auxiliary_ids: List[str],
        op_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Collect information about auxiliary tensors to create start nodes."""
        start_nodes_info: List[Dict[str, Any]] = []
        
        for idx, aux_id in enumerate(sorted(auxiliary_ids)):
            aux_data = layers.get(aux_id) or {}
            conns = aux_data.get("connections") or {}
            output_shapes = aux_data.get("output_shapes") or []
            consumers = list(conns.get("outputs") or [])
            
            # Filter to only include operation nodes
            valid_consumers = [c for c in consumers if c in op_ids]
            
            start_nodes_info.append({
                "original_id": aux_id,
                "index": idx,
                "output_shapes": output_shapes,
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
    ) -> Dict[str, Any]:
        """Build einsum graph dictionary from operation graph."""
        result: Dict[str, Any] = {
            "model_name": pytorch_graph.get("model_name", "pytorch_model"),
            "layers": {},
        }

        # Add start nodes from auxiliary tensors
        start_node_id_map = self._add_start_nodes(result, start_nodes_info)

        # Track node ID remapping for split/expanded operations
        # Maps original node_id -> final output node_id
        node_id_remap: Dict[str, str] = {}
        
        # Track expanded nodes' input mappings
        # Maps original node_id -> {input_index -> subgraph_node_id}
        expanded_input_map: Dict[str, Dict[int, str]] = {}

        # Convert each operation to einsum representation
        for node_id in op_graph.nodes():
            node_data = dict(op_graph.nodes[node_id] or {})
            
            # Check if this is a linear layer with bias that should be split
            if self._should_split_linear_with_bias(node_data):
                matmul_layer, add_layer = self._split_linear_with_bias(
                    node_id, node_data, op_graph, start_nodes_info, start_node_id_map
                )
                result["layers"][node_id] = matmul_layer
                add_node_id = f"{node_id}.bias_add"
                result["layers"][add_node_id] = add_layer
                # Remap: original node_id outputs now come from add_node_id
                node_id_remap[node_id] = add_node_id
            
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
    
    def _should_expand_sdpa(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a scaled_dot_product_attention that should be expanded."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()
        
        return node_type in {"scaled_dot_product_attention", "sdpa", "attention"}
    
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
            "shapes": {
                "Input": query_shape,
                "Input_1": key_shape,
                "Output": scores_shape,
            },
            "tensor_names": {
                "inputs": [
                    f"{input_connections[0]}.Output" if input_connections else f"{node_id}.Query",
                    f"{input_connections[1]}.Output" if len(input_connections) > 1 else f"{node_id}.Key",
                ],
                "outputs": [f"{qk_node_id}.Output"],
            },
            "tensor_shapes": {
                "inputs": [query_shape, key_shape],
                "outputs": [scores_shape],
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
            "shapes": {
                "Input": scores_shape,
                "Output": scores_shape,
            },
            "tensor_names": {
                "inputs": [f"{qk_node_id}.Output"],
                "outputs": [f"{scale_node_id}.Output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
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
            "shapes": {
                "Input": scores_shape,
                "Output": scores_shape,
            },
            "tensor_names": {
                "inputs": [f"{scale_node_id}.Output"],
                "outputs": [f"{softmax_node_id}.Output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape],
                "outputs": [scores_shape],
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
            "shapes": {
                "Input": scores_shape,
                "Input_1": value_shape,
                "Output": final_output_shape,
            },
            "tensor_names": {
                "inputs": [
                    f"{softmax_node_id}.Output",
                    f"{input_connections[2]}.Output" if len(input_connections) > 2 else f"{node_id}.Value",
                ],
                "outputs": [f"{av_node_id}.Output"],
            },
            "tensor_shapes": {
                "inputs": [scores_shape, value_shape],
                "outputs": [final_output_shape],
            },
            "connections": {
                "inputs": [softmax_node_id] + (input_connections[2:3] if len(input_connections) > 2 else []),
                "outputs": output_connections,
            },
        }
        
        return subgraph, av_node_id, input_mapping
    
    def _should_split_linear_with_bias(self, node_data: Dict[str, Any]) -> bool:
        """Check if this is a linear layer with bias that should be split."""
        node_type = node_data.get("type", "")
        if isinstance(node_type, str):
            node_type = node_type.lower()
        else:
            node_type = str(node_type).lower()
        
        if node_type != "linear":
            return False
        
        # Check if bias is present
        weight_nodes = node_data.get("weight_nodes") or []
        weight_shapes = node_data.get("weight_shapes") or []
        
        # Linear with bias has 2 weight nodes: weight and bias
        if len(weight_shapes) >= 2:
            return True
        
        # Check weight node names for bias
        for wn in weight_nodes:
            if "bias" in str(wn).lower():
                return True
        
        return False
    
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
        weight_shapes = node_data.get("weight_shapes") or []
        weight_nodes = node_data.get("weight_nodes") or []
        
        # Separate weight and bias
        weight_shape = weight_shapes[0] if weight_shapes else None
        bias_shape = weight_shapes[1] if len(weight_shapes) > 1 else None
        
        # If bias_shape not found by index, look for it by name
        if bias_shape is None:
            for idx, wn in enumerate(weight_nodes):
                if "bias" in str(wn).lower() and idx < len(weight_shapes):
                    bias_shape = weight_shapes[idx]
                    break
        
        # Build input connections
        input_connections = sorted(list(op_graph.predecessors(node_id)))
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        
        output_connections = sorted(list(op_graph.successors(node_id)))
        
        # Intermediate shape (output of matmul, input to add)
        matmul_output_shape = output_shapes[0] if output_shapes else []
        
        # === MATMUL LAYER ===
        # Generate einsum for matmul (linear without bias)
        matmul_shapes: Dict[str, List[int]] = {}
        if input_shapes:
            matmul_shapes["Input"] = list(input_shapes[0])
        if weight_shape:
            matmul_shapes["Weight"] = list(weight_shape)
        if matmul_output_shape:
            matmul_shapes["Output"] = list(matmul_output_shape)
        
        # Get einsum equation for matmul
        try:
            einsum_op = self._einsum_analyzer.get_einsum_op("linear", matmul_shapes)
            matmul_equation = einsum_op.equation
        except Exception:
            # Fallback equation
            batch_dims = len(input_shapes[0]) - 1 if input_shapes else 0
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            input_str = ''.join(batch_letters + ["K"])
            weight_str = "NK"
            output_str = ''.join(batch_letters + ["N"])
            matmul_equation = f"{input_str},{weight_str}->{output_str}"
        
        add_node_id = f"{node_id}.bias_add"
        
        matmul_input_names = [f"{pred}.Output" for pred in input_connections]
        matmul_input_shapes_list = [list(s) for s in input_shapes]

        if weight_shape:
            matmul_input_names.append(f"{node_id}.Weight")
            matmul_input_shapes_list.append(list(weight_shape))

        matmul_tensor_names = {
            "inputs": matmul_input_names,
            "outputs": [f"{node_id}.Output"],
        }
        matmul_tensor_shapes = {
            "inputs": matmul_input_shapes_list,
            "outputs": [list(matmul_output_shape)] if matmul_output_shape else [],
        }
        
        matmul_layer: Dict[str, Any] = {
            "type": "matmul",
            "einsum_equation": matmul_equation,
            "elementwise_op": "mul",
            "reduction_op": "add",
            "is_real_einsum": True,
            "is_einsum_supportable": True,
            "shapes": matmul_shapes,
            "tensor_names": matmul_tensor_names,
            "tensor_shapes": matmul_tensor_shapes,
            "connections": {
                "inputs": input_connections,
                "outputs": [add_node_id],  # Output goes to bias_add
            },
        }
        
        if weight_shape:
            matmul_layer["additional_info"] = {
                "weights": [{"name": "Weight", "shape": list(weight_shape)}]
            }
        
        # === ADD (BIAS) LAYER ===
        add_shapes: Dict[str, List[int]] = {}
        if matmul_output_shape:
            add_shapes["Input"] = list(matmul_output_shape)
        if bias_shape:
            add_shapes["Weight"] = list(bias_shape)
            add_shapes["Bias"] = list(bias_shape)
        if output_shapes:
            add_shapes["Output"] = list(output_shapes[0])
        
        # Generate einsum equation for add (elementwise)
        if matmul_output_shape:
            dims = len(matmul_output_shape)
            labels = string.ascii_uppercase[:dims]
            add_equation = f"{labels}->{labels}"
        else:
            add_equation = ""
        
        add_input_names = [f"{node_id}.Output"]
        add_input_shapes_list = [list(matmul_output_shape)] if matmul_output_shape else []

        if bias_shape:
            add_input_names.append(f"{node_id}.Bias")
            add_input_shapes_list.append(list(bias_shape))

        add_tensor_names = {
            "inputs": add_input_names,
            "outputs": [f"{add_node_id}.Output"],
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
            "shapes": add_shapes,
            "tensor_names": add_tensor_names,
            "tensor_shapes": add_tensor_shapes,
            "connections": {
                "inputs": [node_id],  # Input from matmul layer
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
            
            # Build shapes dictionary (legacy format for backward compatibility)
            shapes: Dict[str, List[int]] = {}
            if output_shapes:
                shapes["Output"] = list(output_shapes[0])
                for i, shape in enumerate(output_shapes[1:], start=1):
                    shapes[f"Output_{i}"] = list(shape)
            
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
            if output_shapes and len(output_shapes[0]) > 0:
                dims = len(output_shapes[0])
                labels = string.ascii_uppercase[:dims]
                equation = f"->{labels}"
            
            result["layers"][start_id] = {
                "type": "start",
                "einsum_equation": equation,
                "elementwise_op": "copy",
                "reduction_op": "none",
                "is_real_einsum": False,
                "is_einsum_supportable": False,
                "shapes": shapes,
                "tensor_names": tensor_names,
                "tensor_shapes": tensor_shapes,
                "connections": {
                    "inputs": [],
                    "outputs": consumers,
                },
            }
            
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

        shapes = self._extract_operand_shapes(node_data)
        
        # Get module_args for operations like transpose/permute
        module_args = node_data.get("module_args", {})
        
        # Try to get einsum representation
        equation = ""
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
            else:
                # Fallback: try to get from analyzer
                try:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, shapes, module_args=module_args
                    )
                    equation = einsum_op.equation
                    elementwise_op = einsum_op.elementwise_op
                    reduction_op = einsum_op.reduction_op
                    is_real_einsum = einsum_op.is_real_einsum
                    is_einsum_supportable = einsum_op.is_einsum_supportable
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
            reduce_dim = None
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
                reduce_dim, keepdim = self._parse_reduction_args_from_raw_attributes(module_args)
            
            try:
                # Pass module_args, reduce_dim, and keepdim to the analyzer
                if reduce_dim is not None:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, shapes, module_args=module_args, dims=[reduce_dim], keepdim=keepdim
                    )
                else:
                    einsum_op = self._einsum_analyzer.get_einsum_op(
                        node_type, shapes, module_args=module_args, keepdim=keepdim
                    )
                equation = einsum_op.equation
                elementwise_op = einsum_op.elementwise_op
                reduction_op = einsum_op.reduction_op
                is_real_einsum = einsum_op.is_real_einsum
                is_einsum_supportable = einsum_op.is_einsum_supportable
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

        # Build input connections
        input_connections = sorted(list(op_graph.predecessors(node_id)))
        
        # Add start nodes that feed into this operation
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        
        output_connections = sorted(list(op_graph.successors(node_id)))
        
        # Build tensor_names: input names from predecessors, weight names, output name
        tensor_names = self._build_tensor_names(
            node_id, node_data, input_connections, output_connections
        )
        
        # Build tensor_shapes: shapes matching tensor_names order
        tensor_shapes = self._build_tensor_shapes(node_data)
        
        # Validate tensor_names and tensor_shapes match
        is_valid, error_msg = validate_tensor_names_match_shapes(tensor_names, tensor_shapes)
        if not is_valid:
            # Fix mismatch by aligning counts
            tensor_names, tensor_shapes = self._align_tensor_names_and_shapes(
                tensor_names, tensor_shapes, node_data
            )
        
        # Build additional_info for weight/bias metadata
        additional_info = self._build_additional_info(node_data)
        
        result: Dict[str, Any] = {
            "type": node_type,
            "einsum_equation": equation,
            "elementwise_op": elementwise_op,
            "reduction_op": reduction_op,
            "is_real_einsum": is_real_einsum,
            "is_einsum_supportable": is_einsum_supportable,
            "shapes": shapes,
            "tensor_names": tensor_names,
            "tensor_shapes": tensor_shapes,
            "connections": {
                "inputs": input_connections,
                "outputs": output_connections,
            },
        }
        
        if additional_info:
            result["additional_info"] = additional_info
        
        # Pass through raw_attributes from module_args if present
        raw_attributes = module_args.get("raw_attributes")
        if raw_attributes:
            result["raw_attributes"] = raw_attributes
        
        return result

    def _extract_operand_shapes(
        self,
        node_data: Dict[str, Any],
    ) -> Dict[str, List[int]]:
        """Extract operand shapes from node data.
        
        Returns a dictionary with legacy keys (Input, Output, Weight, etc.)
        for backward compatibility with graph_analyzer.
        """
        shapes: Dict[str, List[int]] = {}

        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        weight_shapes = node_data.get("weight_shapes") or []
        weight_nodes = node_data.get("weight_nodes") or []

        # Add input shapes with legacy keys
        if input_shapes:
            shapes["Input"] = list(input_shapes[0])
            for i, shape in enumerate(input_shapes[1:], start=1):
                shapes[f"Input_{i}"] = list(shape)

        # Add output shapes with legacy keys
        if output_shapes:
            shapes["Output"] = list(output_shapes[0])
            for i, shape in enumerate(output_shapes[1:], start=1):
                shapes[f"Output_{i}"] = list(shape)

        # Add weight shapes with legacy keys
        for idx, w_shape in enumerate(weight_shapes):
            name = weight_nodes[idx] if idx < len(weight_nodes) else f"w{idx}"
            key = self._canonical_weight_key(str(name), idx)
            if key in shapes:
                key = f"{key}_{idx}"
            shapes[key] = list(w_shape)

        # Ensure canonical "Weight" entry exists
        if weight_shapes and "Weight" not in shapes:
            first_key = next(
                (k for k in shapes if k.lower().startswith("weight")),
                None
            )
            if first_key:
                shapes["Weight"] = shapes[first_key]

        return shapes

    def _build_tensor_names(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        input_connections: List[str],
        output_connections: List[str],
    ) -> Dict[str, List[str]]:
        """Build tensor names that match einsum equation operand order.
        
        Input tensor names: <predecessor_node_id>.Output
        Weight tensor names: <node_id>.Weight (or .Bias, etc.)
        Output tensor names: <node_id>.Output
        
        Order matches einsum equation: inputs (activations), weights -> outputs
        """
        input_names: List[str] = []
        output_names: List[str] = []
        
        # Input activation tensors from predecessor nodes
        for pred_id in input_connections:
            input_names.append(f"{pred_id}.Output")
        
        # Weight tensors (no source node, belong to this node)
        weight_nodes = node_data.get("weight_nodes") or []
        for w_name in weight_nodes:
            # Normalize weight name
            w_norm = str(w_name).strip()
            if w_norm.lower() == "weight":
                input_names.append(f"{node_id}.Weight")
            elif w_norm.lower() == "bias":
                input_names.append(f"{node_id}.Bias")
            else:
                # Use the original name
                safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", w_norm)
                input_names.append(f"{node_id}.{safe_name}")
        
        # Output tensor
        output_names.append(f"{node_id}.Output")
        
        # Add additional outputs if there are multiple
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
        """Build tensor shapes that match einsum equation operand order.
        
        Order matches einsum equation: inputs (activations), weights -> outputs
        """
        input_shapes_list: List[List[int]] = []
        output_shapes_list: List[List[int]] = []
        
        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        weight_shapes = node_data.get("weight_shapes") or []
        
        # Input activation shapes
        for shape in input_shapes:
            input_shapes_list.append(list(shape))
        
        # Weight shapes (come after activation inputs in einsum)
        for shape in weight_shapes:
            input_shapes_list.append(list(shape))
        
        # Output shapes
        for shape in output_shapes:
            output_shapes_list.append(list(shape))
        
        return {
            "inputs": input_shapes_list,
            "outputs": output_shapes_list,
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
        """Build additional_info with weight/bias metadata."""
        additional_info: Dict[str, Any] = {}
        
        weight_nodes = node_data.get("weight_nodes") or []
        weight_shapes = node_data.get("weight_shapes") or []
        
        if weight_nodes or weight_shapes:
            weights_info: List[Dict[str, Any]] = []
            for idx, w_name in enumerate(weight_nodes):
                w_info: Dict[str, Any] = {"name": str(w_name)}
                if idx < len(weight_shapes):
                    w_info["shape"] = list(weight_shapes[idx])
                weights_info.append(w_info)
            
            # Handle case where we have shapes but no names
            for idx in range(len(weight_nodes), len(weight_shapes)):
                weights_info.append({
                    "name": f"weight_{idx}",
                    "shape": list(weight_shapes[idx]),
                })
            
            if weights_info:
                additional_info["weights"] = weights_info
        
        return additional_info

    def _canonical_weight_key(self, name: str, idx: int) -> str:
        """Map parameter/buffer names to stable operand keys."""
        raw = name.strip()
        norm = raw.lower()
        
        if norm == "weight":
            return "Weight"
        if norm == "bias":
            return "Bias"
            
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", raw)
        if safe:
            return f"Weight_{safe}"
        return f"Weight_{idx}"

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

