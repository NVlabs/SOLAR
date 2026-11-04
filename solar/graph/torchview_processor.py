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

"""TorchView graph processor for extracting layer information from PyTorch models.

This module provides functionality to process torchview ComputationGraph objects
and extract detailed information about model layers, following Google's Python
style guide.

The output format matches the original process_torchview_graph.py output:
- node_id: Hierarchical node identifier (e.g., "Model.linear_0")
- node_type: Operation type (e.g., "linear", "conv2d", "matmul")
- node_class: Actual node class (e.g., "FunctionNode", "TensorNode", "ModuleNode")
- input_nodes: List of input node IDs (connections from predecessors)
- output_nodes: List of output node IDs (connections to successors)
- input_shapes: List of input tensor shapes
- output_shapes: List of output tensor shapes
- input_dtypes: List of input tensor data types
- output_dtypes: List of output tensor data types
- input_types: List of input tensor types
- output_types: List of output tensor types
- module_args: Dictionary of module configuration arguments
"""

import json
import ast
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch import nn

from solar.common.constants import (
    BOOLEAN_ATTRS,
    GEOMETRIC_ATTRS,
    MODULE_ATTR_NAMES,
)
from solar.common.types import GraphInfo, NodeInfo, TensorShape
from solar.graph import tensor_layouts as _tensor_layouts
from solar.common.utils import ensure_directory
import logging

logger = logging.getLogger(__name__)


class TorchviewProcessor:
    """Processes torchview computation graphs to extract layer information.

    This class provides methods to extract detailed information from torchview
    ComputationGraph objects, including node hierarchies, shapes, connections,
    and PyTorch nn.Module parameters.
    """

    def __init__(self, debug: bool = False):
        """Initialize the TorchviewProcessor.

        Args:
            debug: Enable debug output for troubleshooting.
        """
        self.debug = debug
        self._processed_nodes: set = set()
        self._matched_modules: set = set()
        self._node_counter: Dict[str, int] = {}
        # Mapping from original node object id to clean node_id
        self._original_to_clean_id: Dict[str, str] = {}
        self._cached_default_dtype: Optional[str] = None

    def process_graph(self,
                     computation_graph: Any,
                     output_dir: str,
                     kernel_name: str,
                     original_model: Optional[nn.Module] = None,
                     capture_capability: Optional[Dict[str, Any]] = None) -> List[NodeInfo]:
        """Process a torchview ComputationGraph and save extracted layer nodes.

        Args:
            computation_graph: torchview ComputationGraph object.
            output_dir: Directory to save outputs.
            kernel_name: Name of the kernel for file naming.
            original_model: Original PyTorch model for parameter extraction.

        Returns:
            List of NodeInfo objects containing extracted layer information.
        """
        if self.debug:
            logger.debug(f"Processing torchview graph for {kernel_name}...")

        # Reset state for new graph
        self._reset_state()

        # Extract layer nodes
        # Parameter layout records bind through the traced model object (an earlier decision round 4).
        self._param_layouts = _tensor_layouts.parameter_layouts(original_model)
        layer_nodes = self._extract_layer_nodes(computation_graph, original_model)

        # Save canonical YAML graph (and remove any legacy JSON artifacts).
        output_path = Path(output_dir)
        ensure_directory(output_path)
        yaml_filename = output_path / "pytorch_graph.yaml"

        self._save_pytorch_graph_yaml(layer_nodes, yaml_filename, model_name=kernel_name,
                                      capture_capability=capture_capability)

        if self.debug:
            self._print_layer_summary(layer_nodes)

        return layer_nodes

    def _reset_state(self) -> None:
        """Reset internal state for processing a new graph."""
        self._processed_nodes.clear()
        self._matched_modules.clear()
        self._node_counter.clear()
        self._original_to_clean_id.clear()
        # Reset module index tracker for hierarchical naming
        self._module_index_tracker: Dict[Tuple[str, str], Dict[int, int]] = {}
        self._module_has_duplicates: set = set()
        self._names_repeated_in_any_path: set = set()
        self._hierarchical_counter: Dict[str, int] = {}
        # Mapping from original node id to hierarchical name
        self._original_to_hierarchical: Dict[str, str] = {}

    def _extract_layer_nodes(self,
                           computation_graph: Any,
                           original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
        """Extract layer nodes from the computation graph.

        Args:
            computation_graph: torchview ComputationGraph object.
            original_model: Original PyTorch model for parameter extraction.

        Returns:
            List of NodeInfo objects.
        """
        # Try different extraction methods in order of preference
        layer_nodes = []

        # Note: node_hierarchy contains hierarchical ModuleNode info (e.g., nn.Linear
        # with in_features/out_features), but the computation graph is extracted from
        # the flattened edge_list which contains FunctionNodes (e.g., F.linear).
        # The hierarchy is useful for understanding module structure but edge_list
        # provides the actual computation graph with tensor flow.
        #
        #  Extract from node hierarchy if available (hierarchical module info)
        # if hasattr(computation_graph, 'node_hierarchy') and computation_graph.node_hierarchy:
        #     if self._is_hierarchy_useful(computation_graph.node_hierarchy):
        #         if self.debug:
        #             print("Extracting from node_hierarchy...")
        #         return self._extract_from_hierarchy(
        #             computation_graph.node_hierarchy, 'Model'
        #         )

        # Extract from edge_list (flattened computation graph)
        if hasattr(computation_graph, 'edge_list') and computation_graph.edge_list:
            if self.debug:
                logger.debug(f"Extracting from edge_list ({len(computation_graph.edge_list)} edges)...")
            layer_nodes = self._extract_from_edge_list(computation_graph, original_model)
            return layer_nodes

        # Parse visual graph as fallback
        if hasattr(computation_graph, 'visual_graph'):
            if self.debug:
                logger.debug("Parsing visual graph...")
            return self._extract_from_visual_graph(computation_graph.visual_graph)

        return layer_nodes

    def _is_hierarchy_useful(self, node_hierarchy: Dict[str, Any]) -> bool:
        """Check if node hierarchy contains useful computation nodes.

        Args:
            node_hierarchy: Node hierarchy dictionary.

        Returns:
            True if hierarchy contains useful nodes, False otherwise.
        """
        for key, node in node_hierarchy.items():
            node_class = type(key).__name__
            if node_class in ['TensorNode', 'ModuleNode', 'FunctionNode']:
                return True
        return False

    def _extract_from_hierarchy(self,
                              node_hierarchy: Dict[Any, Any],
                              parent_name: str = '',
                              original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
        """Recursively extract nodes from hierarchy.

        The hierarchy structure from torchview is:
        {ModuleNode: [TensorNode, FunctionNode, {ChildModuleNode: [...]}, ...]}

        Keys are ModuleNode objects, values are lists containing:
        - TensorNode objects (inputs/outputs/intermediates)
        - FunctionNode objects (operations)
        - Nested dicts for child modules

        Args:
            node_hierarchy: Node hierarchy dictionary.
            parent_name: Parent node name for building hierarchical IDs.
            original_model: Original PyTorch model for dtype inference fallback.

        Returns:
            List of NodeInfo objects.
        """
        layer_nodes = []

        for module_node, children_list in node_hierarchy.items():
            try:
                # The key is the ModuleNode itself
                module_name = getattr(module_node, 'name', str(module_node))
                full_module_id = f"{parent_name}.{module_name}" if parent_name else module_name

                # Extract the ModuleNode info
                node_info = self._extract_node_info(module_node, full_module_id, original_model)
                layer_nodes.append(node_info)

                # Process the children list
                if isinstance(children_list, list):
                    for child in children_list:
                        child_class = type(child).__name__

                        if child_class == 'dict' or isinstance(child, dict):
                            # Nested module hierarchy - recurse
                            child_nodes = self._extract_from_hierarchy(
                                child, full_module_id, original_model
                            )
                            layer_nodes.extend(child_nodes)
                        elif child_class in ['TensorNode', 'FunctionNode']:
                            # Extract tensor or function node
                            child_name = getattr(child, 'name', str(child))
                            child_id = f"{full_module_id}.{child_name}"
                            child_info = self._extract_node_info(child, child_id, original_model)
                            layer_nodes.append(child_info)

            except Exception as e:
                if self.debug:
                    logger.debug(f"Warning: Could not extract module {module_node}: {e}")
                continue

        return layer_nodes

    def _extract_node_info(self, node: Any, node_id: str, original_model: Optional[nn.Module] = None) -> NodeInfo:
        """Extract information from a single node.

        Args:
            node: Node object from the computation graph.
            node_id: Unique identifier for the node.
            original_model: Original PyTorch model for dtype inference fallback.

        Returns:
            NodeInfo object containing extracted information.
        """
        node_class_name = type(node).__name__

        # Extract node type
        node_type = self._get_node_type(node, node_class_name)

        # Extract shapes
        input_shapes, output_shapes = self._extract_shapes(node, node_type)

        # Extract dtypes (pass shapes and original_model for dtype inference fallback)
        input_dtypes, output_dtypes = self._extract_dtypes(node, node_type, original_model, input_shapes, output_shapes)

        # Extract module args (no longer extracting weight_nodes/weight_shapes)
        module_info = self._extract_module_info(node)

        # Layout record bound to this tensor node's identity (inputs: the
        # traced RecorderTensor; parameters: the nn.Parameter object).
        layout = _tensor_layouts.layout_for_node(node, node_type, getattr(self, "_param_layouts", None))
        if layout:
            module_info['module_args']['layout'] = layout

        recorded_out = getattr(node, 'recorded_output_dtypes', None)
        has_recorded_dtypes = (
            isinstance(recorded_out, (list, tuple)) and bool(recorded_out))

        return NodeInfo(
            node_id=node_id,
            type=node_type,
            node_class=node_class_name,
            input_nodes=[],
            output_nodes=[],
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            input_types=[],   # populated later from connection node types
            output_types=[],  # populated later from connection node types
            module_args=module_info['module_args'],
            # Actual executed output dtypes were recorded at capture —
            # mark them so the converter's dtype repair consumes the
            # evidence instead of reconstructing heuristically.
            output_dtypes_recorded=has_recorded_dtypes
        )

    def _get_node_type(self, node: Any, node_class: str) -> str:
        """Determine the node type from the node object.

        Args:
            node: Node object.
            node_class: Class name of the node.

        Returns:
            String representing the node type.
        """
        return (
            getattr(node, 'operation', None) or
            getattr(node, 'op_name', None) or
            getattr(node, 'name', None) or
            getattr(node, '_op', None) or
            node_class.lower().replace('node', '')
        )

    def _extract_shapes(self, node: Any, node_type: str) -> Tuple[List[TensorShape], List[TensorShape]]:
        """Extract input and output shapes from a node.

        Args:
            node: Node object.
            node_type: Type of the node for special handling.

        Returns:
            Tuple of (input_shapes, output_shapes).
        """
        input_shapes = []
        output_shapes = []
        node_class = type(node).__name__

        # Special handling for TensorNode - check tensor_shape first
        if node_class == 'TensorNode':
            tensor_shape = None
            if hasattr(node, 'tensor_shape') and node.tensor_shape is not None:
                if hasattr(node.tensor_shape, '__iter__'):
                    tensor_shape = list(node.tensor_shape)
                else:
                    tensor_shape = [node.tensor_shape]

            if tensor_shape:
                node_name = node_type.lower() if node_type else ''
                if node_name in ['input-tensor', 'auxiliary-tensor']:
                    output_shapes = [tensor_shape]
                elif node_name == 'output-tensor':
                    input_shapes = [tensor_shape]
                elif node_name == 'parameter-tensor':
                    # Parameter tensors (weights/biases): shape is output (what they provide)
                    output_shapes = [tensor_shape]
                elif node_name == 'hidden-tensor':
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                else:
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                return input_shapes, output_shapes

        # Extract input shapes from inputs attribute
        if hasattr(node, 'inputs') and node.inputs:
            for inp in node.inputs:
                if hasattr(inp, 'tensor_shape') and inp.tensor_shape is not None:
                    input_shapes.append(list(inp.tensor_shape))
                elif hasattr(inp, 'shape'):
                    input_shapes.append(list(inp.shape))
        elif hasattr(node, 'input_shape') and node.input_shape:
            if isinstance(node.input_shape, (list, tuple)):
                input_shapes = [list(s) for s in node.input_shape]
            else:
                input_shapes = [list(node.input_shape)]

        # Extract output shapes from outputs attribute
        if hasattr(node, 'outputs') and node.outputs:
            for out in node.outputs:
                if hasattr(out, 'tensor_shape') and out.tensor_shape is not None:
                    output_shapes.append(list(out.tensor_shape))
                elif hasattr(out, 'shape'):
                    output_shapes.append(list(out.shape))
        elif hasattr(node, 'output_shape') and node.output_shape:
            if isinstance(node.output_shape, (list, tuple)):
                output_shapes = [list(s) for s in node.output_shape]
            else:
                output_shapes = [list(node.output_shape)]

        return input_shapes, output_shapes

    def _extract_dtypes(self, node: Any, node_type: str, original_model: Optional[nn.Module] = None,
                       input_shapes: Optional[List[TensorShape]] = None,
                       output_shapes: Optional[List[TensorShape]] = None) -> Tuple[List[str], List[str]]:
        """Extract input and output dtypes from a node.

        Args:
            node: Node object.
            node_type: Type of the node for special handling.

        Returns:
            Tuple of (input_dtypes, output_dtypes).
        """
        input_dtypes = []
        output_dtypes = []
        node_class = type(node).__name__

        # Helper function to extract dtype from a tensor-like object
        def get_dtype(obj: Any) -> Optional[str]:
            """Extract dtype string from a tensor-like object."""
            if obj is None:
                return None

            # Check for tensor_dtype attribute (torchview convention)
            if hasattr(obj, 'tensor_dtype') and obj.tensor_dtype is not None:
                dtype = obj.tensor_dtype
                if isinstance(dtype, torch.dtype):
                    return str(dtype)
                elif dtype is not None:
                    return str(dtype)

            # Check for dtype attribute directly
            if hasattr(obj, 'dtype'):
                dtype = obj.dtype
                if isinstance(dtype, torch.dtype):
                    return str(dtype)
                elif isinstance(dtype, str):
                    return dtype

            # Check for tensor attribute that might have dtype
            if hasattr(obj, 'tensor') and obj.tensor is not None:
                tensor_obj = obj.tensor
                if hasattr(tensor_obj, 'dtype'):
                    return str(tensor_obj.dtype)

            # Check if it's a torch.Tensor directly
            if isinstance(obj, torch.Tensor):
                return str(obj.dtype)

            return None

        # Special handling for TensorNode - mirror shape extraction logic
        if node_class == 'TensorNode':
            tensor_dtype = None

            # Try tensor_dtype attribute first (torchview convention)
            if hasattr(node, 'tensor_dtype') and node.tensor_dtype is not None:
                tensor_dtype = str(node.tensor_dtype)
            # Try dtype attribute
            elif hasattr(node, 'dtype') and node.dtype is not None:
                tensor_dtype = str(node.dtype)
            # Try tensor attribute
            elif hasattr(node, 'tensor') and node.tensor is not None:
                if hasattr(node.tensor, 'dtype'):
                    tensor_dtype = str(node.tensor.dtype)
            # Try getting from inputs/outputs if available
            elif hasattr(node, 'outputs') and node.outputs:
                for out in node.outputs:
                    dtype = get_dtype(out)
                    if dtype:
                        tensor_dtype = dtype
                        break

            # If we found a dtype, assign it based on node type (mirroring shape logic)
            if tensor_dtype:
                node_name = node_type.lower() if node_type else ''
                if node_name in ['input-tensor', 'auxiliary-tensor']:
                    # Input tensors: dtype is output (what they produce)
                    output_dtypes = [tensor_dtype]
                elif node_name == 'output-tensor':
                    # Final output tensors: dtype is input (what they receive)
                    input_dtypes = [tensor_dtype]
                elif node_name == 'hidden-tensor':
                    # Hidden tensors: intermediate nodes have both input and output dtypes
                    input_dtypes = [tensor_dtype]
                    output_dtypes = [tensor_dtype]
                else:
                    # Unknown tensor type: add to both
                    input_dtypes = [tensor_dtype]
                    output_dtypes = [tensor_dtype]
                return input_dtypes, output_dtypes

        # Extract input dtypes from inputs attribute (mirroring shape extraction)
        if hasattr(node, 'inputs') and node.inputs:
            for inp in node.inputs:
                dtype = get_dtype(inp)
                if dtype:
                    input_dtypes.append(dtype)
                else:
                    # If we can't get dtype from the input object, try to infer from shape
                    # by checking if it has tensor_shape and we can get dtype from that
                    if hasattr(inp, 'tensor_shape') and hasattr(inp, 'tensor_dtype'):
                        dtype = get_dtype(inp)
                        if dtype:
                            input_dtypes.append(dtype)

        # Extract output dtypes from outputs attribute (mirroring shape extraction)
        if hasattr(node, 'outputs') and node.outputs:
            for out in node.outputs:
                dtype = get_dtype(out)
                if dtype:
                    output_dtypes.append(dtype)
                else:
                    # Try to get dtype from tensor_shape's source
                    if hasattr(out, 'tensor_shape') and hasattr(out, 'tensor_dtype'):
                        dtype = get_dtype(out)
                        if dtype:
                            output_dtypes.append(dtype)

        # Recorded ACTUAL output dtypes from the executed call results
        # (recorder-shim evidence, per output slot, in order).  Consumed
        # BEFORE any attribute fallback and before the model-default fill:
        # a cast's target dtype exists in no other trace evidence, and the
        # executed result is strictly better evidence than attribute
        # scraping (review follow-up — BF16 cast billed at FP32).
        recorded_out = getattr(node, 'recorded_output_dtypes', None)
        if isinstance(recorded_out, (list, tuple)) and recorded_out:
            output_dtypes = [str(d) for d in recorded_out]

        # Fallback: try input_shape/output_shape attributes (some nodes might have these)
        if not input_dtypes and hasattr(node, 'input_dtype') and node.input_dtype:
            if isinstance(node.input_dtype, (list, tuple)):
                input_dtypes = [str(d) for d in node.input_dtype]
            else:
                input_dtypes = [str(node.input_dtype)]

        if not output_dtypes and hasattr(node, 'output_dtype') and node.output_dtype:
            if isinstance(node.output_dtype, (list, tuple)):
                output_dtypes = [str(d) for d in node.output_dtype]
            else:
                output_dtypes = [str(node.output_dtype)]

        # Fallback: If we have shapes but no dtypes, try to infer from original model
        # This ensures dtype counts match shape counts
        if input_shapes is None:
            input_shapes = []
        if output_shapes is None:
            output_shapes = []

        if original_model is not None:
            if self._cached_default_dtype is None:
                for param in original_model.parameters():
                    if param.dtype is not None:
                        self._cached_default_dtype = str(param.dtype)
                        break
                if self._cached_default_dtype is None:
                    self._cached_default_dtype = "torch.float32"

            default_dtype = self._cached_default_dtype

            while len(input_dtypes) < len(input_shapes):
                input_dtypes.append(default_dtype)

            while len(output_dtypes) < len(output_shapes):
                output_dtypes.append(default_dtype)

        return input_dtypes, output_dtypes

    def _extract_module_info(self, node: Any) -> Dict[str, Any]:
        """Extract module argument information from a node.

        Args:
            node: Node object.

        Returns:
            Dictionary with module_args.
        """
        module_info = {
            'module_args': {}
        }

        node_class = type(node).__name__

        if node_class == 'ModuleNode':
            self._extract_module_node_info(node, module_info)
        elif node_class == 'FunctionNode':
            self._extract_function_node_info(node, module_info)

        return module_info

    def _extract_module_node_info(self, node: Any, module_info: Dict[str, Any]) -> None:
        """Extract information from a ModuleNode.

        Args:
            node: ModuleNode object.
            module_info: Dictionary to populate with extracted information.
        """
        module = self._get_pytorch_module(node)

        if module is not None:
            module_info['module_args'] = self._extract_module_arguments(module)
        else:
            # Fallback: parse the 'attributes' string from torchview
            # Format: "Linear(training=False, in_features=64, out_features=64)"
            if hasattr(node, 'attributes') and node.attributes:
                parsed = self._parse_module_attributes_string(node.attributes)
                if parsed:
                    module_info['module_args'] = parsed

    def _get_pytorch_module(self, node: Any) -> Optional[nn.Module]:
        """Get the PyTorch module from a node object.

        Args:
            node: Node object that may contain a PyTorch module.

        Returns:
            PyTorch module if found, None otherwise.
        """
        # Try common attribute names
        for attr_name in MODULE_ATTR_NAMES:
            if hasattr(node, attr_name):
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, nn.Module):
                    return attr_value
        return None

    def _extract_module_arguments(self, module: nn.Module) -> Dict[str, Any]:
        """Extract module configuration arguments.

        Args:
            module: PyTorch module.

        Returns:
            Dictionary of module arguments.
        """
        args = {'module_type': type(module).__name__}

        # Extract common attributes based on module type
        for attr_name in dir(module):
            if attr_name.startswith('_') or callable(getattr(module, attr_name)):
                continue

            try:
                value = getattr(module, attr_name)

                # Handle special attribute types
                if attr_name in GEOMETRIC_ATTRS and hasattr(value, '__iter__'):
                    args[attr_name] = list(value)
                elif attr_name in BOOLEAN_ATTRS:
                    args[attr_name] = bool(value)
                elif isinstance(value, (int, float, str, bool)):
                    args[attr_name] = value
                elif attr_name == 'bias':
                    args[attr_name] = value is not None

            except Exception:
                continue

        return args

    def _parse_module_attributes_string(self, attributes: str) -> Dict[str, Any]:
        """Parse torchview ModuleNode attributes string.

        Format: "Linear(training=False, in_features=64, out_features=64)"

        Args:
            attributes: Stringified module attributes from torchview.

        Returns:
            Dictionary of parsed module arguments.
        """
        result: Dict[str, Any] = {}

        if not attributes:
            return result

        # Extract module type from the beginning
        # Format: "ModuleType(key=value, ...)"
        import re
        match = re.match(r'(\w+)\((.*)\)', attributes)
        if not match:
            return result

        module_type = match.group(1)
        args_str = match.group(2)

        result['module_type'] = module_type

        # Parse key=value pairs
        # Handle nested parentheses for tuples like kernel_size=(3, 3)
        for kv_match in re.finditer(r'(\w+)=([^,]+(?:\([^)]*\))?)', args_str):
            key = kv_match.group(1)
            value_str = kv_match.group(2).strip()

            # Parse the value
            try:
                if value_str == 'True':
                    result[key] = True
                elif value_str == 'False':
                    result[key] = False
                elif value_str == 'None':
                    result[key] = None
                elif value_str.startswith('(') and value_str.endswith(')'):
                    # Tuple like (3, 3)
                    result[key] = eval(value_str)
                elif '.' in value_str and not value_str.replace('.', '').replace('-', '').isdigit():
                    # String with dots (like torch.float32)
                    result[key] = value_str
                elif value_str.replace('.', '').replace('-', '').isdigit():
                    # Number
                    if '.' in value_str:
                        result[key] = float(value_str)
                    else:
                        result[key] = int(value_str)
                else:
                    result[key] = value_str
            except Exception:
                result[key] = value_str

        return result

    def _extract_function_node_info(self, node: Any, module_info: Dict[str, Any]) -> None:
        """Extract information from a FunctionNode.

        Uses input_types from torchview (if available) to identify which
        input tensors are activations ('input') vs parameters ('weight').
        Falls back to raw_attributes parsing for older torchview versions.

        Args:
            node: FunctionNode object.
            module_info: Dictionary to populate with extracted information.
        """
        node_name = getattr(node, 'name', '').lower()
        module_info['module_args']['function_name'] = node_name

        # Extract from torchview 'attributes' field (contains stringified args/kwargs)
        # This is populated when collect_attributes=True in draw_graph()
        if hasattr(node, 'attributes') and node.attributes:
            parsed_args = self._parse_torchview_attributes(node.attributes, node_name)
            if parsed_args:
                module_info['module_args'].update(parsed_args)

        # Scalar keywords the binder could not place in the callable's
        # (testing-override) signature and left out of the operand bind
        # (ACC-140-SDPA-KWARGS); their values are still in raw_attributes.
        unbound = getattr(node, 'unbound_scalar_kwargs', None)
        if unbound:
            module_info['module_args']['unbound_scalar_kwargs'] = list(unbound)
        # Structured capture-time record of F.interpolate's scalar arguments
        # (issue #50): the converter's identity-copy rule reads only this.
        interp_ev = getattr(node, 'interpolate_evidence', None)
        if isinstance(interp_ev, dict):
            module_info['module_args']['interpolate_evidence'] = dict(interp_ev)
        interp_err = getattr(node, 'interpolate_evidence_error', None)
        if isinstance(interp_err, dict):
            module_info['module_args']['interpolate_evidence_error'] = dict(interp_err)


        # Capture-time operand binding evidence for the normalisation ops (issue
        # #34): the parameter-operand shim's mirror of (args, kwargs) with every
        # tensor leaf replaced by how the capture bound it ('activation' /
        # 'parameter' / 'detached'); consumed by _attach_detached_parameters.
        kinds = getattr(node, 'operand_kinds', None)
        if kinds is not None and node_name in self._DETACHED_PARAM_OPS:
            module_info['module_args']['operand_kinds'] = kinds
        # ... and the identity of the executed callee (functional wrapper vs
        # native ATen entry point decides the argument order).
        callee = getattr(node, 'callee', None)
        if callee and node_name in self._DETACHED_PARAM_OPS:
            module_info['module_args']['callee'] = str(callee)          # diagnostic only
        if node_name in self._DETACHED_PARAM_OPS:
            # the executed call bound to the callable's ACTUAL signature
            # (parameter name per positional / keyword argument), the
            # signature's source and the callable's type: the role evidence
            ctype = getattr(node, 'callable_type', None)
            if ctype:
                module_info['module_args']['callable_type'] = str(ctype)
            bindings = getattr(node, 'operand_bindings', None)
            if bindings is not None:
                module_info['module_args']['operand_bindings'] = bindings
                module_info['module_args']['binding_source'] = getattr(node, 'binding_source', None)
                module_info['module_args']['binding_param_order'] = list(getattr(node, 'binding_param_order', []) or [])
                module_info['module_args']['binding_wrapped_differs'] = bool(getattr(node, 'binding_wrapped_differs', False))
                module_info['module_args']['binding_defaults'] = dict(getattr(node, 'binding_defaults', {}) or {})
        # Identity binding of the call's tensor arguments (an earlier decision round 5)
        call_operands = getattr(node, 'call_operands', None)
        if isinstance(call_operands, dict):
            module_info['module_args']['call_operands'] = dict(call_operands)

        # Extract scalar kwargs as module_args
        if hasattr(node, 'kwargs') and node.kwargs:
            for key, value in node.kwargs.items():
                if not hasattr(value, 'shape'):
                    module_info['module_args'][key] = value

    def _parse_torchview_attributes(
        self,
        attributes: str,
        node_name: str,
    ) -> Dict[str, Any]:
        """Parse torchview stringified attributes to extract function arguments.

        torchview's stringify_attributes() produces strings like:
        - For functions: "[[Tensor(shape=(2, 32, 64), dtype=torch.float32), 1, 2], {}]"
        - This represents [args_list, kwargs_dict]

        We parse this by replacing Tensor(...) with a placeholder and using eval.

        Args:
            attributes: Stringified attributes from torchview.
            node_name: Name of the function (e.g., 'transpose', 'permute').

        Returns:
            Dictionary of parsed arguments.
        """
        result: Dict[str, Any] = {}

        if not attributes:
            return result

        # Store raw attributes for debugging
        result['raw_attributes'] = attributes

        try:
            # Parse the attributes string by replacing Tensor(...) with a placeholder
            args_list, kwargs_dict = self._eval_attributes_string(attributes)

            if args_list is None:
                return result

            # Extract non-tensor arguments (skip index 0 which is usually the input tensor)
            non_tensor_args = [arg for arg in args_list if not isinstance(arg, dict) or 'tensor_placeholder' not in arg]
            # Filter out tensor placeholders
            scalar_args = [arg for arg in non_tensor_args if not (isinstance(arg, dict) and 'tensor_placeholder' in arg)]

            if node_name == 'transpose':
                # transpose(input, dim0, dim1) - extract dim0 and dim1
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if len(int_args) >= 2:
                    result['dim0'] = int_args[0]
                    result['dim1'] = int_args[1]
                    result['transpose_dims'] = [int_args[0], int_args[1]]
                # Also check kwargs
                if kwargs_dict:
                    if 'dim0' in kwargs_dict:
                        result['dim0'] = kwargs_dict['dim0']
                    if 'dim1' in kwargs_dict:
                        result['dim1'] = kwargs_dict['dim1']
                    if 'dim0' in result and 'dim1' in result:
                        result['transpose_dims'] = [result['dim0'], result['dim1']]

            elif node_name == 'permute':
                # permute(input, dims) or permute(input, *dims)
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if int_args:
                    result['permute_dims'] = int_args
                # Check for tuple/list arg
                for arg in scalar_args:
                    if isinstance(arg, (list, tuple)) and all(isinstance(d, int) for d in arg):
                        result['permute_dims'] = list(arg)
                        break
                # Check kwargs
                if kwargs_dict and 'dims' in kwargs_dict:
                    result['permute_dims'] = list(kwargs_dict['dims'])

            elif node_name == 't':
                # t() is always transpose(0, 1) for 2D tensors
                result['dim0'] = 0
                result['dim1'] = 1
                result['transpose_dims'] = [1, 0]

            elif node_name in ('view', 'reshape'):
                # view(input, *sizes) or reshape(input, shape)
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if int_args:
                    result['target_shape'] = int_args
                # Check for tuple/list arg
                for arg in scalar_args:
                    if isinstance(arg, (list, tuple)) and all(isinstance(d, int) for d in arg):
                        result['target_shape'] = list(arg)
                        break

            elif node_name in ('mean', 'sum', 'logsumexp', 'prod', 'amax', 'amin',
                               'any', 'all', 'norm', 'std', 'var'):
                # Reduction ops: func(input, dim, keepdim=False)
                # scalar_args may contain dim as int or list of ints
                if kwargs_dict:
                    if 'dim' in kwargs_dict:
                        dim_val = kwargs_dict['dim']
                        result['dim'] = [dim_val] if isinstance(dim_val, int) else list(dim_val)
                    if 'keepdim' in kwargs_dict:
                        result['keepdim'] = kwargs_dict['keepdim']
                # dim can also be a positional arg
                if 'dim' not in result:
                    for arg in scalar_args:
                        if isinstance(arg, int):
                            result['dim'] = [arg]
                            break
                        elif isinstance(arg, (list, tuple)) and all(isinstance(d, int) for d in arg):
                            result['dim'] = list(arg)
                            break

        except Exception as e:
            if self.debug:
                logger.debug(f"Warning: Failed to parse attributes for {node_name}: {e}")

        return result

    @staticmethod
    def _replace_balanced_calls(text: str, func_name: str, replacement: str) -> str:
        """Replace func_name(...) using balanced parenthesis matching."""
        result = []
        i = 0
        tag = func_name + "("
        while i < len(text):
            if text[i:i + len(tag)] == tag:
                depth = 1
                j = i + len(tag)
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                result.append(replacement)
                i = j
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    def _eval_attributes_string(
        self,
        attributes: str,
    ) -> Tuple[Optional[List[Any]], Optional[Dict[str, Any]]]:
        """Safely evaluate torchview attributes string.

        Replaces Tensor(...) and slice(...) with placeholders, quotes bare
        keyword dict keys, then evaluates the string.

        Args:
            attributes: Stringified attributes from torchview.

        Returns:
            Tuple of (args_list, kwargs_dict) or (None, None) on failure.
        """
        import re

        try:
            processed = attributes

            processed = self._replace_balanced_calls(
                processed, "Tensor", "{'tensor_placeholder': True}"
            )
            processed = self._replace_balanced_calls(
                processed, "slice", "None"
            )

            processed = re.sub(r'torch\.\w+', 'None', processed)
            processed = processed.replace("Ellipsis", "None")
            processed = processed.replace("...", "None")

            # Quote bare keyword keys: {tensor: v, a: v} -> {"tensor": v, "a": v}
            processed = re.sub(
                r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', processed
            )

            parsed = eval(processed, {"__builtins__": {}}, {})

            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                args_list = parsed[0] if isinstance(parsed[0], (list, tuple)) else []
                kwargs_dict = parsed[1] if isinstance(parsed[1], dict) else {}
                return list(args_list), kwargs_dict

            return None, None

        except Exception as e:
            if self.debug:
                logger.debug(f"Warning: Failed to eval attributes: {e}")
            return None, None

    def _parse_tensor_shapes_from_attributes(
        self,
        attributes: str,
    ) -> List[TensorShape]:
        """Parse all tensor shapes from a torchview raw_attributes string.

        The string looks like:
          ``[[Tensor(shape=(2048, 8192), dtype=torch.float32), ...], {}]``

        This method extracts every ``Tensor(shape=(...))`` occurrence **in
        order**, returning a list of shape lists.  Non-tensor arguments are
        skipped so the returned indices correspond to *tensor* positions only.

        Args:
            attributes: Raw attributes string from torchview.

        Returns:
            Ordered list of tensor shapes (each a ``List[int]``).
        """
        if not attributes:
            return []

        tensor_shapes: List[TensorShape] = []
        for match in re.finditer(
            r'Tensor\(shape=\(([^)]*)\)', attributes
        ):
            dims_str = match.group(1).strip()
            if not dims_str:
                continue
            try:
                shape = [
                    int(d.strip()) for d in dims_str.split(',') if d.strip()
                ]
                tensor_shapes.append(shape)
            except ValueError:
                continue

        return tensor_shapes

    def _extract_weights_from_attributes(
        self,
        attributes: str,
        node_name: str,
    ) -> Tuple[List[str], List[TensorShape]]:
        """Extract weight / bias tensor info from a raw_attributes string.

        Uses :pyattr:`_WEIGHT_TENSOR_INDICES` to map known ops (``linear``,
        ``conv2d``, …) to the positional tensor arguments that represent
        learnable parameters (weight, bias, running statistics, etc.).

        Args:
            attributes: Raw attributes string from torchview.
            node_name: Lowercased operation name (e.g. ``'linear'``).

        Returns:
            ``(weight_nodes, weight_shapes)`` – parallel lists of parameter
            names and their shapes.  Empty lists when the op is unknown or the
            attributes could not be parsed.
        """
        positions = self._WEIGHT_TENSOR_INDICES.get(node_name)
        if not positions:
            return [], []

        tensor_shapes = self._parse_tensor_shapes_from_attributes(attributes)
        if not tensor_shapes:
            return [], []

        weight_nodes: List[str] = []
        weight_shapes: List[TensorShape] = []

        for tensor_idx, param_name in positions:
            if tensor_idx < len(tensor_shapes):
                weight_nodes.append(param_name)
                weight_shapes.append(tensor_shapes[tensor_idx])

        return weight_nodes, weight_shapes

    def _infer_parameter_name(self,
                             op_name: str,
                             arg_index: int,
                             shape: TensorShape) -> str:
        """Infer parameter name from operation type and argument position.

        Args:
            op_name: Operation name.
            arg_index: Position of the argument.
            shape: Shape of the tensor.

        Returns:
            Inferred parameter name.
        """
        op_lower = op_name.lower()

        if 'conv' in op_lower:
            if arg_index == 0:
                return 'input'
            elif arg_index == 1:
                return 'weight'
            elif arg_index == 2:
                return 'bias'
        elif 'linear' in op_lower:
            if arg_index == 0:
                return 'input'
            elif arg_index == 1:
                return 'weight'
            elif arg_index == 2:
                return 'bias'
        elif 'batch_norm' in op_lower or 'batchnorm' in op_lower:
            param_names = ['input', 'weight', 'bias', 'running_mean', 'running_var']
            return param_names[arg_index] if arg_index < len(param_names) else f'bn_arg_{arg_index}'

        return f'arg_{arg_index}' if arg_index > 0 else 'input'

    def _infer_module_arguments(self,
                               op_name: str,
                               param_name: str,
                               param_shape: TensorShape,
                               module_args: Dict[str, Any]) -> None:
        """Infer module configuration arguments from parameter shapes.

        Args:
            op_name: Operation name.
            param_name: Parameter name.
            param_shape: Shape of the parameter tensor.
            module_args: Dictionary to update with inferred arguments.
        """
        op_lower = op_name.lower()

        if param_name == 'weight':
            # Convolution operations
            if 'conv' in op_lower and len(param_shape) >= 3:
                if 'transpose' in op_lower:
                    # ConvTranspose weight: [in_channels, out_channels, *kernel_size]
                    module_args['in_channels'] = param_shape[0]
                    module_args['out_channels'] = param_shape[1]
                    module_args['kernel_size'] = param_shape[2:]
                else:
                    # Conv weight: [out_channels, in_channels, *kernel_size]
                    module_args['out_channels'] = param_shape[0]
                    module_args['in_channels'] = param_shape[1]
                    module_args['kernel_size'] = param_shape[2:]

            # Linear operations
            elif 'linear' in op_lower and len(param_shape) >= 2:
                module_args['out_features'] = param_shape[0]
                module_args['in_features'] = param_shape[1]

        elif param_name == 'bias':
            module_args['bias'] = True

    def _extract_from_edge_list(self,
                               computation_graph: Any,
                               original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
        """Extract nodes from the edge_list of the computation graph.

        This method properly tracks node relationships (input_nodes, output_nodes)
        by processing the edge list to build connection information.

        Args:
            computation_graph: ComputationGraph with edge_list.
            original_model: Original PyTorch model for parameter extraction.

        Returns:
            List of NodeInfo objects with proper connection information.
        """
        computation_nodes = {}  # original_id -> node object
        node_order = []  # Preserve order of discovery

        # Step 1: Collect all unique nodes from edges
        for i, edge in enumerate(computation_graph.edge_list):
            if len(edge) < 2:
                raise ValueError(
                    f"Edge at index {i} has fewer than 2 nodes: {edge}. "
                    f"Expected format: (source_node, target_node)."
                )
            if len(edge) > 2:
                raise ValueError(
                    f"Edge at index {i} has more than 2 nodes: {len(edge)} nodes found. "
                    f"Expected exactly 2 nodes per edge: (source_node, target_node)."
                )

            source_node, target_node = edge[0], edge[1]

            # Add nodes to computation_nodes dict
            for node in (source_node, target_node):
                self._validate_node_type(node)
                original_id = str(getattr(node, 'node_id', id(node)))
                if original_id not in computation_nodes:
                    computation_nodes[original_id] = node
                    node_order.append(original_id)

        if self.debug:
            logger.debug(f"Found {len(computation_nodes)} unique computation nodes")

        # Step 2: Pre-scan all nodes to discover which module names have duplicates
        # This allows us to add indices consistently (Linear_0, Linear_1, Linear_2)
        self._prescan_module_hierarchy(computation_nodes, node_order)

        # Step 3: Generate clean IDs and hierarchical names for all nodes
        result = []
        for original_id in node_order:
            node = computation_nodes[original_id]
            clean_id = self._generate_clean_id(node)
            hierarchical_name = self._generate_hierarchical_name(node)

            self._original_to_clean_id[original_id] = clean_id
            self._original_to_hierarchical[original_id] = hierarchical_name

            node_info = self._extract_node_info(node, clean_id, original_model)
            # Add hierarchical_name to module_args
            node_info.module_args['hierarchical_name'] = hierarchical_name
            # Provenance of the call: id() of the nn.Module whose forward made it
            # (torchview ModuleNode.compute_unit_id), resolved against the traced
            # model object when needed (issue #34 round 4, identity-3).  The
            # display name in hierarchical_name is NOT provenance.
            owner = self._innermost_parent_module_node(node)
            if owner is not None and getattr(owner, 'compute_unit_id', None) is not None:
                node_info.module_args['owner_module_id'] = int(owner.compute_unit_id)
            result.append(node_info)

        # Step 3: Build relationships from edge list using the ID mapping
        id_to_node_info = {node.node_id: node for node in result}

        for edge in computation_graph.edge_list:
            if len(edge) >= 2:
                source_node, target_node = edge[0], edge[1]

                source_original_id = str(getattr(source_node, 'node_id', id(source_node)))
                target_original_id = str(getattr(target_node, 'node_id', id(target_node)))

                source_clean_id = self._original_to_clean_id.get(source_original_id)
                target_clean_id = self._original_to_clean_id.get(target_original_id)

                if source_clean_id and target_clean_id:
                    source_info = id_to_node_info.get(source_clean_id)
                    target_info = id_to_node_info.get(target_clean_id)

                    if source_info and target_info:
                        # Add connection if not already present
                        if target_clean_id not in source_info.output_nodes:
                            source_info.output_nodes.append(target_clean_id)
                        if source_clean_id not in target_info.input_nodes:
                            target_info.input_nodes.append(source_clean_id)

        # Reorder input_nodes to match positional arg order using ordered_input_nodes
        # from the patched torchview FunctionNode
        for original_id in node_order:
            node = computation_nodes[original_id]
            ordered_inputs = getattr(node, 'ordered_input_nodes', None)
            if ordered_inputs:
                clean_id = self._original_to_clean_id.get(original_id)
                if clean_id:
                    node_info = id_to_node_info.get(clean_id)
                    if node_info and node_info.input_nodes:
                        # Build ordered input_nodes from ordered_input_nodes
                        ordered_clean_ids = []
                        for input_node in ordered_inputs:
                            inp_orig_id = str(getattr(input_node, 'node_id', id(input_node)))
                            inp_clean_id = self._original_to_clean_id.get(inp_orig_id)
                            if inp_clean_id and inp_clean_id in node_info.input_nodes:
                                ordered_clean_ids.append(inp_clean_id)
                        # Append any remaining input_nodes not in ordered list
                        for inp_id in node_info.input_nodes:
                            if inp_id not in ordered_clean_ids:
                                ordered_clean_ids.append(inp_id)
                        node_info.input_nodes = ordered_clean_ids

        # Populate input_types and output_types based on connected node types.
        # Input classification:
        #   'input-tensor', 'auxiliary-tensor', 'hidden-tensor' -> 'input'
        #   'parameter-tensor' -> 'weight'
        #   FunctionNode/ModuleNode predecessor -> 'input'
        # Output classification:
        #   'output-tensor', 'auxiliary-tensor', 'hidden-tensor' -> 'output'
        #   FunctionNode/ModuleNode successor -> 'output'
        #   'parameter-tensor' -> ASSERT FAIL (invalid)
        _WEIGHT_TENSOR_TYPES = {'parameter-tensor'}

        for node_info in result:
            # input_types
            input_types = []
            for inp_id in node_info.input_nodes:
                inp_node = id_to_node_info.get(inp_id)
                if inp_node and inp_node.type.lower() in _WEIGHT_TENSOR_TYPES:
                    input_types.append('weight')
                else:
                    input_types.append('input')
            node_info.input_types = input_types

            # output_types
            output_types = []
            for out_id in node_info.output_nodes:
                out_node = id_to_node_info.get(out_id)
                if out_node:
                    assert out_node.type.lower() != 'parameter-tensor', (
                        f"Output node {out_id} of {node_info.node_id} is parameter-tensor, "
                        f"which should only appear as input."
                    )
                output_types.append('output')
            node_info.output_types = output_types

        if self.debug:
            nodes_with_inputs = sum(1 for n in result if n.input_nodes)
            nodes_with_outputs = sum(1 for n in result if n.output_nodes)
            logger.debug(f"Nodes with input connections: {nodes_with_inputs}")
            logger.debug(f"Nodes with output connections: {nodes_with_outputs}")

        # Step 3b: attach normalisation parameters/buffers that torchview did
        # not connect (keyword-argument params, running statistics) as
        # parameter-tensor nodes + weight inputs (issue #34 / an earlier decision).
        self._attach_detached_parameters(result, id_to_node_info)

        # Step 4: Apply parameters from original model if provided
        # This handles both FunctionNode and ModuleNode cases
        if original_model:
            self._apply_model_parameters(result, original_model, computation_nodes)

        return result

    _VALID_NODE_TYPES = ('TensorNode', 'ModuleNode', 'FunctionNode')

    # Mapping from op name to (tensor_index, param_name) tuples.
    # tensor_index is the position among tensor arguments only (non-tensor
    # args like scalars are skipped).  Index 0 is always the activation
    # input and is therefore excluded from this mapping.
    _WEIGHT_TENSOR_INDICES: Dict[str, List[Tuple[int, str]]] = {
        'linear':           [(1, 'weight'), (2, 'bias')],
        'conv1d':           [(1, 'weight'), (2, 'bias')],
        'conv2d':           [(1, 'weight'), (2, 'bias')],
        'conv3d':           [(1, 'weight'), (2, 'bias')],
        'conv_transpose1d': [(1, 'weight'), (2, 'bias')],
        'conv_transpose2d': [(1, 'weight'), (2, 'bias')],
        'conv_transpose3d': [(1, 'weight'), (2, 'bias')],
        'batch_norm':       [(1, 'running_mean'), (2, 'running_var'),
                             (3, 'weight'), (4, 'bias')],
        'layer_norm':       [(1, 'weight'), (2, 'bias')],
        'group_norm':       [(1, 'weight'), (2, 'bias')],
        'instance_norm':    [(1, 'running_mean'), (2, 'running_var'),
                             (3, 'weight'), (4, 'bias')],
        'rms_norm':         [(1, 'weight')],
        'embedding':        [(1, 'weight')],
    }

    # Ops whose affine parameters (and, in eval mode, running statistics) reach
    # the torch function as keyword arguments or as module buffers.  torchview
    # connects only traced parameters (e.g. a conv weight passed positionally)
    # as parameter-tensor nodes, so these are absent from the graph although
    # the recorded call attributes list them with their shapes.  They are
    # attached as weight inputs from those attributes (issue #34 / an earlier decision).
    _DETACHED_PARAM_OPS = frozenset({
        'layer_norm', 'group_norm', 'batch_norm', 'instance_norm', 'rms_norm',
    })

    def _validate_node_type(self, node: Any) -> None:
        """Validate that a node is one of the expected computation node types.

        Args:
            node: Node object to validate.

        Raises:
            TypeError: If node is not one of the valid node types.
        """
        node_class = type(node).__name__
        if node_class not in self._VALID_NODE_TYPES:
            raise TypeError(
                f"Invalid node type: {node_class}. "
                f"Expected one of {self._VALID_NODE_TYPES}."
            )

    def _prescan_module_hierarchy(
        self,
        computation_nodes: Dict[str, Any],
        node_order: List[str]
    ) -> None:
        """Pre-scan all nodes to discover which module names have duplicates.

        This allows consistent indexing where all instances of a duplicated
        module name get indices (Linear_0, Linear_1, Linear_2) instead of
        (Linear, Linear_1, Linear_2).

        Also tracks module names that appear multiple times in ANY hierarchy path,
        so they get consistent indexing across all nodes.

        Args:
            computation_nodes: Dict mapping original_id to node objects.
            node_order: List of original_ids in discovery order.
        """
        # Temporary tracker to count unique instances at each level
        temp_tracker: Dict[Tuple[str, str], set] = {}

        # Track module names that appear multiple times in any single hierarchy
        # (e.g., EncoderLayer appears 3 times in EncoderLayer.EncoderLayer.EncoderLayer)
        if not hasattr(self, '_names_repeated_in_any_path'):
            self._names_repeated_in_any_path: set = set()

        for original_id in node_order:
            node = computation_nodes[original_id]
            hierarchy_with_ids = self._get_module_hierarchy_with_ids(node)

            # Check if any module name appears multiple times in this hierarchy
            name_counts: Dict[str, int] = {}
            for module_name, _ in hierarchy_with_ids:
                name_counts[module_name] = name_counts.get(module_name, 0) + 1

            # Track names that repeat in any path
            for name, count in name_counts.items():
                if count > 1:
                    self._names_repeated_in_any_path.add(name)

            parent_path = 'root'
            for module_name, obj_id in hierarchy_with_ids:
                key = (parent_path, module_name)

                if key not in temp_tracker:
                    temp_tracker[key] = set()

                temp_tracker[key].add(obj_id)

                # Build parent_path for next level (use module_name without index for now)
                parent_path = f"{parent_path}.{module_name}"

        # Mark keys that have duplicates (multiple different obj_ids at same level)
        for key, obj_ids in temp_tracker.items():
            if len(obj_ids) > 1:
                self._module_has_duplicates.add(key)

    # Argument layouts of the normalisation calls as torchview records them.
    # ``torch.nn.functional.<op>`` is recorded through ``handle_torch_function``
    # with the wrapper's own layout: the activation (and, for batch_norm, the
    # running statistics) positional, weight/bias/flags as keywords with their
    # defaults filled in.  The native ``torch.<op>`` records the ATen layout,
    # which for batch_norm and instance_norm places ``weight, bias`` BEFORE
    # ``running_mean, running_var`` (an earlier decision review train-2);
    # layer_norm, group_norm and rms_norm share one order in both spellings.
    _NORM_SIGNATURES: Dict[str, Tuple[str, ...]] = {
        'layer_norm':    ('input', 'normalized_shape', 'weight', 'bias', 'eps', 'cudnn_enable'),
        'rms_norm':      ('input', 'normalized_shape', 'weight', 'eps'),
        'group_norm':    ('input', 'num_groups', 'weight', 'bias', 'eps', 'cudnn_enabled'),
        'batch_norm':    ('input', 'running_mean', 'running_var', 'weight', 'bias', 'training', 'momentum', 'eps'),
        'instance_norm': ('input', 'running_mean', 'running_var', 'weight', 'bias', 'use_input_stats', 'momentum', 'eps'),
    }
    _NORM_NATIVE_SIGNATURES: Dict[str, Tuple[str, ...]] = {
        'batch_norm':    ('input', 'weight', 'bias', 'running_mean', 'running_var', 'training', 'momentum', 'eps', 'cudnn_enabled'),
        'instance_norm': ('input', 'weight', 'bias', 'running_mean', 'running_var', 'use_input_stats', 'momentum', 'eps', 'cudnn_enabled'),
    }
    _NORM_PARAM_ROLES = ('running_mean', 'running_var', 'weight', 'bias')
    _NORM_STAT_ROLES = ('running_mean', 'running_var')
    _TENSOR_RE = re.compile(r'Tensor\(shape=\(([^)]*)\)(?:,\s*dtype=([\w.]+))?\)')

    _BUILTIN_CALLABLE_TYPES = ('builtin_function_or_method',)
    _NORM_VOCABULARY = ('input', 'running_mean', 'running_var', 'weight', 'bias')

    def _norm_role_names(self, op: str, margs: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """How each recorded argument slot maps to a parameter NAME, from
        evidence tied to the executed callable — never from its ``__name__`` /
        ``__qualname__`` (copied by ``functools.wraps``), a positional count or
        a shape.  Returns ``({"positional": [name|None, ...], "keyword":
        {kw: name}, "spelling": ...}, None)`` or ``(None, refusal_reason)``.

        * a C builtin (``callable_type == 'builtin_function_or_method'``;
          ``torch.batch_norm`` & co.; a type no wrapper can forge): positional
          arguments follow the op's ATen order (``_NORM_NATIVE_SIGNATURES``,
          the dispatcher's declaration), keywords their own names.  When torch's
          testing-override stub, which the shim used to order keyword tensors,
          disagrees with that order AND the call spelled tensors by keyword,
          the mapping is refused (torch.instance_norm's stub has the functional
          order — a documented torch limitation, surfaced instead of guessed);
        * any other callable: the shim's binding of the executed call to the
          callable's ACTUAL ``inspect.signature`` (``operand_bindings``, source
          'signature', honouring ``__signature__`` / ``__wrapped__``).  A
          binding that only exists through a testing-override stub (a Python
          wrapper around a builtin without a signature of its own) is refused;
          so is a missing binding.
        """
        ctype = str(margs.get('callable_type') or '')
        bindings = margs.get('operand_bindings')
        source = margs.get('binding_source')
        pos = list(bindings[0]) if isinstance(bindings, (list, tuple)) and len(bindings) == 2 and isinstance(bindings[0], (list, tuple)) else None
        kw = dict(bindings[1]) if isinstance(bindings, (list, tuple)) and len(bindings) == 2 and isinstance(bindings[1], dict) else None
        kinds = margs.get('operand_kinds')
        kw_kinds = kinds[1] if isinstance(kinds, (list, tuple)) and len(kinds) == 2 and isinstance(kinds[1], dict) else {}
        tensor_keywords = sorted(k for k, v in kw_kinds.items() if isinstance(v, str))   # keywords that carried a tensor
        if ctype in self._BUILTIN_CALLABLE_TYPES:
            table = self._NORM_NATIVE_SIGNATURES.get(op) or self._NORM_SIGNATURES[op]
            order = list(margs.get('binding_param_order') or [])
            # the stub-order hazard concerns the ORDER of tensor operands; scalar
            # keywords (flags, momentum, eps) reorder nothing (scalar-kw-8)
            if tensor_keywords and source == 'testing_override' and order and [n for n in order if n in self._NORM_VOCABULARY] != [n for n in table if n in self._NORM_VOCABULARY]:
                return None, (f"torch's testing-override signature for the builtin {op} ({order[:5]}...) disagrees with its ATen "
                              f"argument order ({list(table[:5])}...), so the keyword-spelled call cannot be bound safely")
            n_pos = len(pos) if pos is not None else len(table)
            return {'positional': [table[i] if i < len(table) else f'arg_{i}' for i in range(n_pos)], 'keyword': kw or {}, 'spelling': 'native-builtin'}, None
        if pos is None or kw is None:
            return None, "the capture recorded no signature binding for the executed callable (no inspectable signature or the call did not bind)"
        if source != 'signature':
            return None, (f"the executed callable ({ctype}) has no signature of its own; its binding came from torch's testing-override "
                          "stub of the callable it wraps, which does not establish the wrapper's argument order")
        if margs.get('binding_wrapped_differs') and tensor_keywords:
            return None, ("the callable's own signature differs from the wrapped signature its functools.wraps metadata points to, "
                          "and the capture ordered the keyword tensor operands by the latter; the keyword-spelled call cannot be bound safely")
        return {'positional': pos, 'keyword': kw, 'spelling': 'signature', 'order': list(margs.get('binding_param_order') or [])}, None

    def _norm_call_roles(self, attributes: str, op: str, kinds: Any = None, callee: Optional[str] = None,
                         margs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Resolve the recorded call of a normalisation op into argument slots.

        Returns ``{"spelling": "functional"|"native", "activation":
        {"positional": bool, "shape": [...], "kind": ...}|None, "slots":
        [(role, shape, dtype, kind), ...] (the tensor-valued parameter roles
        in signature order, ``None`` arguments omitted), "flags": {name:
        value}, "bound": bool}`` or None when the attributes cannot be
        parsed.  Roles come from the per-spelling signature by position and
        from keyword names; the spelling (functional wrapper vs native ATen
        entry point, i.e. which signature table applies) is taken from the
        RECORDED CALLEE (``callee``: the module-qualified name of the function
        object torchview executed), never from the positional count or from
        shapes; an unrecorded or unrecognised callee gives ``spelling: None``
        and the caller refuses the binding.  ``kind`` is the capture-time
        binding of that very argument (``operand_kinds`` recorded by the
        parameter-operand shim: ``'activation'`` / ``'parameter'`` /
        ``'detached'``) joined by the same position / keyword, or None when
        no binding was recorded for it (``bound`` is False when the capture
        recorded none at all).
        """
        if op not in self._NORM_SIGNATURES or not attributes:
            return None
        pos_kinds = kinds[0] if isinstance(kinds, (list, tuple)) and len(kinds) == 2 \
            and isinstance(kinds[0], (list, tuple)) else None
        kw_kinds = kinds[1] if isinstance(kinds, (list, tuple)) and len(kinds) == 2 \
            and isinstance(kinds[1], dict) else None
        def pos_kind(i: int) -> Optional[str]:
            k = pos_kinds[i] if pos_kinds is not None and i < len(pos_kinds) else None
            return k if isinstance(k, str) else None
        def kw_kind(name: str) -> Optional[str]:
            k = kw_kinds.get(name) if kw_kinds is not None else None
            return k if isinstance(k, str) else None
        def scalar_flag(kind: Any) -> Tuple[str, Any]:
            """An update-mode flag from the STRUCTURED per-slot record the shim
            wrote for the executed call (flagtext-10): ``('bool', value)``
            when the actual argument was a Python Boolean; otherwise
            ``('unestablished', <what it was>)`` — None, a number, a string or
            any other opaque object, a tensor, a container (a mapping or a
            list, whose leaves are never read), or no structured record at all
            (a capture older than this record / an unrecorded slot).  The
            stringified attributes are never consulted for a flag."""
            if isinstance(kind, dict) and kind.get('scalar') == 'bool' and isinstance(kind.get('value'), bool):
                return 'bool', kind['value']
            if isinstance(kind, dict) and 'scalar' in kind:
                return 'unestablished', (f"{kind['scalar']} {kind.get('value')!r}" if 'value' in kind else f"opaque {kind.get('type')}")
            if isinstance(kind, str):
                return 'unestablished', f'a tensor ({kind})'
            if isinstance(kind, (list, tuple, dict)):
                return 'unestablished', f'a {type(kind).__name__} (container; its leaves are not a flag)'
            return 'unestablished', 'no structured scalar record for the slot'
        def flag_from(flags: Dict[str, Any], name: str, kind: Any) -> None:
            state, val = scalar_flag(kind)
            if state == 'bool':
                flags[name] = val
            else:
                flags.setdefault('unestablished', {})[name] = val
        text = str(attributes).strip()
        split = text.rfind('], {')
        positional_text, keyword_text = (text[:split + 1] + ']', text[split + 3:]) if split >= 0 else (text, '')
        tensors: List[Tuple[List[int], Optional[str]]] = []
        def repl(m: 're.Match') -> str:
            tensors.append(([int(d) for d in m.group(1).split(',') if d.strip()], m.group(2)))
            return f'("__T__", {len(tensors) - 1})'
        positional_text = self._TENSOR_RE.sub(repl, positional_text)
        positional_text = re.sub(r'torch\.\w+', 'None', positional_text)
        try:
            positional = ast.literal_eval(positional_text)
            if isinstance(positional, (list, tuple)) and len(positional) == 1 and isinstance(positional[0], (list, tuple)):
                positional = positional[0]
            positional = list(positional)
        except Exception:
            return None
        def is_tensor(v: Any) -> bool:
            return isinstance(v, tuple) and len(v) == 2 and v[0] == '__T__'
        names, refusal = self._norm_role_names(op, margs or {})
        if names is None:
            # The roles cannot be bound; the UPDATE flags can still be read
            # where they are unambiguous (a keyword-spelled flag, or a builtin's
            # positional flag at its ATen position) so that a training /
            # use_input_stats call is refused for THAT reason — the exclusion of
            # statistics updates does not depend on which tensor is which.
            flags: Dict[str, Any] = {}
            stats_mentioned = False
            ctype = str((margs or {}).get('callable_type') or '')
            table = (self._NORM_NATIVE_SIGNATURES.get(op) or self._NORM_SIGNATURES[op]) if ctype in self._BUILTIN_CALLABLE_TYPES else ()
            for i, v in enumerate(positional):
                name = table[i] if i < len(table) else None
                if name in ('training', 'use_input_stats'):
                    flag_from(flags, name, pos_kinds[i] if pos_kinds is not None and i < len(pos_kinds) else None)
                elif name in self._NORM_STAT_ROLES and is_tensor(v):
                    stats_mentioned = True
            for kwname, kind in (kw_kinds or {}).items():
                if kwname in ('training', 'use_input_stats'):
                    flag_from(flags, kwname, kind)
            for m in re.finditer(r'(\w+):\s*Tensor\(shape=\(([^)]*)\)', keyword_text):
                if m.group(1) in self._NORM_STAT_ROLES:
                    stats_mentioned = True
            return {'spelling': None, 'callee': callee, 'refusal': refusal, 'activation': None, 'slots': [], 'flags': flags,
                    'stats_mentioned': stats_mentioned, 'bound': False}
        spelling = names['spelling']
        # slot ORDER (messages and the pairing with the connected inputs, which
        # the shim ordered by the same signature): the builtin's ATen order, or
        # the callable's own parameter order
        sig = self._NORM_NATIVE_SIGNATURES.get(op) or self._NORM_SIGNATURES[op]
        if spelling == 'signature':
            own = [n for n in names.get('order') or [] if n in self._NORM_VOCABULARY]
            sig = tuple(own) if own else tuple(self._NORM_SIGNATURES[op])
        pos_names = list(names['positional']); kw_names = dict(names['keyword'])
        by_role: Dict[str, Tuple[List[int], Optional[str], Optional[str]]] = {}
        activation = None
        flags: Dict[str, Any] = {}
        unknown: List[str] = []
        for i, v in enumerate(positional):
            name = pos_names[i] if i < len(pos_names) and pos_names[i] else f'arg_{i}'
            if is_tensor(v) and name not in self._NORM_VOCABULARY:
                unknown.append(f'positional[{i}] -> {name!r}')
            elif is_tensor(v):
                shape, dtype = tensors[v[1]]
                if name == 'input':
                    activation = {'positional': True, 'shape': shape, 'kind': pos_kind(i)}
                elif name in self._NORM_PARAM_ROLES:
                    by_role[name] = (shape, dtype, pos_kind(i))
            elif name in ('training', 'use_input_stats'):
                # the flag comes from the structured record of the executed
                # argument (none-9 / FLAGTEXT-10), never from the text
                flag_from(flags, name, pos_kinds[i] if pos_kinds is not None and i < len(pos_kinds) else None)
        for m in re.finditer(r'(\w+):\s*(Tensor\(shape=\(([^)]*)\)(?:,\s*dtype=([\w.]+))?\)|True|False|None)', keyword_text):
            kwname, val = m.group(1), m.group(2)
            name = kw_names.get(kwname, kwname)
            if val.startswith('Tensor('):
                shape = [int(d) for d in m.group(3).split(',') if d.strip()]
                if name not in self._NORM_VOCABULARY:
                    unknown.append(f'keyword {kwname!r} -> {name!r}')
                elif name == 'input':
                    activation = {'positional': False, 'shape': shape, 'kind': kw_kind(kwname)}
                elif name in self._NORM_PARAM_ROLES:
                    by_role[name] = (shape, m.group(4), kw_kind(kwname))
        # Keyword-spelled update flags: from the structured record of each
        # TOP-LEVEL keyword argument (flagtext-10) — a Boolean establishes
        # the flag; None / a number / a string / a container / a tensor / no
        # record leave it unestablished (never coerced, never read out of a
        # nested value or a string, never replaced by the omitted default: the
        # shim records defaults only for parameters the call did not supply).
        for kwname, kind in (kw_kinds or {}).items():
            name = kw_names.get(kwname, kwname)
            if name in ('training', 'use_input_stats'):
                flag_from(flags, name, kind)
        if unknown:
            return {'spelling': None, 'callee': callee, 'activation': None, 'slots': [], 'flags': {}, 'bound': False,
                    'refusal': f"tensor arguments bound to parameters outside the {op} vocabulary {list(self._NORM_VOCABULARY)}: {unknown}"}
        # update-mode flags the call OMITTED take the callable's own resolved
        # defaults (default-8); a flag that is neither supplied nor known
        # leaves the update mode unestablished and the caller refuses
        for fname in ('training', 'use_input_stats'):
            if fname not in flags and fname in ((margs or {}).get('binding_defaults') or {}):
                dv = (margs or {}).get('binding_defaults')[fname]
                if isinstance(dv, bool):
                    flags[fname] = dv
                    flags[f'{fname}_from_default'] = True
        slots = [(role,) + by_role[role] for role in sig if role in by_role]
        return {'spelling': spelling, 'callee': callee, 'activation': activation, 'slots': slots,
                'flags': flags, 'bound': pos_kinds is not None and kw_kinds is not None}

    def _mark_norm_unsupported(self, node_info: NodeInfo, reason: str, op: str, detail: str) -> None:
        """Record an explicit, machine-readable boundary instead of guessing."""
        node_info.module_args['norm_params_unattached'] = reason
        key = f'_warned_{reason}'
        if not getattr(self, key, False):
            setattr(self, key, True)
            warnings.warn(
                f"{node_info.node_id} ({op}): {detail}  The layer is marked "
                f"norm_params_unattached={reason!r}; its parameters/statistics are NOT billed and "
                "the converter marks the operation not supportable.",
                RuntimeWarning, stacklevel=3)

    @staticmethod
    def _binding_mismatch(
        extras: List[Tuple[List[int], Optional[str]]],
        extra_kinds: List[str],
        bound_slots: List[Tuple[str, List[int], Optional[str], str]],
    ) -> Optional[str]:
        """Check the connected extra inputs against the slots the recorded call
        bound to connected tensors, pairwise in signature order.

        Returns None when the binding is consistent (same count; each pair
        agrees in shape, in dtype when both are recorded, and in identity: a
        slot bound to an nn.Parameter is a parameter-tensor node, a slot bound
        to an activation is not), else the reason.  Nothing here chooses a
        role: the roles are the recorded call's, this only refuses a record
        that does not describe the connected inputs.
        """
        roles = [r for r, _, _, _ in bound_slots]
        if len(extras) != len(bound_slots):
            return (f"{len(extras)} connected inputs follow the activation but the recorded call binds "
                    f"{len(bound_slots)} tensor-valued parameter slots to connected tensors {roles}.")
        for (shape, dtype), nkind, (role, sshape, sdtype, kind) in zip(extras, extra_kinds, bound_slots):
            if list(sshape) != list(shape):
                return f"the connected input bound to {role!r} has shape {list(shape)}, the recorded call {list(sshape)}."
            if dtype and sdtype and str(dtype) != str(sdtype):
                return f"the connected input bound to {role!r} has dtype {dtype}, the recorded call {sdtype}."
            if kind == 'parameter' and nkind != 'parameter-tensor':
                return f"the recorded call binds {role!r} to an nn.Parameter but the connected input is a {nkind or 'unknown'} node."
            if kind == 'activation' and nkind in ('parameter-tensor', 'unknown', ''):
                return f"the recorded call binds {role!r} to an activation but the connected input is a {nkind or 'unknown'} node."
        return None

    def _attach_detached_parameters(
        self,
        result: List[NodeInfo],
        id_to_node_info: Dict[str, NodeInfo],
    ) -> None:
        """Attach the normalisation parameters torchview did not connect (issue #34).

        Argument identity is reconciled from the recorded call
        (:meth:`_norm_call_roles`: per-spelling signature tables, so native
        ``torch.batch_norm/instance_norm`` and the functional wrappers each get
        their own argument order) against the inputs the capture connected, in
        argument order:

        * the activation is the first connected input of the recorded
          activation shape; a keyword ``input=`` activation is accepted when
          the capture connected it (the parameter-operand shim binds keyword
          tensors semantically) and marked
          ``norm_params_unattached='keyword_activation'`` when its edge is
          missing;
        * every connected tensor after the activation (an ``nn.Parameter``
          recorded as a parameter-tensor node, a model input or a computed
          tensor passed positionally or by keyword) KEEPS its edge and is
          never re-attached; it is assigned to a parameter slot by order, a
          shape/dtype mismatch ruling an assignment out;
        * which slot each connected tensor occupies comes ONLY from the
          executed call: the parameter-operand shim records, per argument of
          the call it executed, whether that argument was an nn.Parameter, a
          connected activation or a detached tensor (``operand_kinds``), and
          that record is joined to the signature slots by position / keyword
          (:meth:`_norm_call_roles`); the connected inputs are then checked
          against the slots so bound (:meth:`_binding_mismatch`).  Only the
          slots the call bound to DETACHED tensors are attached, each once,
          as a ``parameter-tensor`` node + ``weight`` input named by its role
          (buffers such as running statistics, plain tensors);
        * a call without a recorded binding (capture without the shim's
          operand order, unknown callee, tensors in a ``**kwargs`` catch-all)
          or whose binding does not describe the connected inputs is marked
          ``norm_params_unattached='argument_mapping_unresolved'``: no role is
          ever inferred from a shape, a module class or a forward object
          (identity-3/4/5); ``owner_module_id`` stays a diagnostic;
        * ``batch_norm(training=True)`` / ``instance_norm(use_input_stats=True)``
          with running statistics present UPDATE those statistics in either
          spelling; this inference accounting excludes them explicitly
          (``norm_params_unattached='running_stats_update'``).
        Nodes without recorded call attributes are marked
        ``norm_params_unattached='no_call_attributes'``.  The converter turns
        every marker into ``is_einsum_supportable=False`` and the analyzer
        reports the layer as incomplete.
        """
        new_nodes: List[NodeInfo] = []
        for node_info in result:
            op = str(node_info.type).lower()
            if node_info.node_class != 'FunctionNode' or op not in self._DETACHED_PARAM_OPS:
                continue
            attrs = (node_info.module_args or {}).get('raw_attributes')
            if not attrs:
                # torchview recorded no call attributes (releases before upstream
                # #128 without solar/patches/torchview-collect-attributes-inputs.patch).
                self._mark_norm_unsupported(
                    node_info, 'no_call_attributes', op,
                    "torchview recorded no call attributes; normalisation weight/bias/running "
                    "statistics cannot be attached and are NOT billed as traffic.  Requires "
                    "torchview with collect_attributes propagation (upstream #128) or "
                    "solar/patches/torchview-collect-attributes-inputs.patch.")
                continue
            margs = node_info.module_args or {}
            roles = self._norm_call_roles(str(attrs), op, margs.get('operand_kinds'), margs.get('callee'), margs)
            if roles is None:
                self._mark_norm_unsupported(node_info, 'argument_mapping_unresolved', op,
                                            "the recorded call attributes could not be parsed.")
                continue
            if roles['spelling'] is None:
                _upd = (op == 'batch_norm' and roles['flags'].get('training') is True) or \
                       (op == 'instance_norm' and roles['flags'].get('use_input_stats') is True)
                if _upd and roles.get('stats_mentioned'):
                    # refused either way; the specific reason is the statistics update
                    self._mark_norm_unsupported(
                        node_info, 'running_stats_update', op,
                        "training/use_input_stats mode updates the running statistics; this inference accounting "
                        f"path does not represent those writes (the argument roles were also not bindable: {roles.get('refusal')}).")
                    continue
                self._mark_norm_unsupported(
                    node_info, 'argument_mapping_unresolved', op,
                    f"{roles.get('refusal')}; the argument roles are not inferred from the callee's name, a positional "
                    "count or shapes.")
                continue
            connected = [list(sh) for sh in node_info.input_shapes]
            kinds = [str(getattr(id_to_node_info.get(nid), 'type', '')) for nid in node_info.input_nodes]
            kinds += ['unknown'] * (len(connected) - len(kinds))
            act = roles['activation']
            act_connected = bool(act) and bool(connected) and list(connected[0]) == list(act['shape']) \
                and kinds[0] != 'parameter-tensor' and act.get('kind') in (None, 'activation')
            if act is None or (not act['positional'] and not act_connected):
                self._mark_norm_unsupported(
                    node_info, 'keyword_activation', op,
                    "the activation is passed as keyword input= and the capture recorded no "
                    "producer edge for it, so the dependency cannot be restored.")
                continue
            if not act_connected:
                self._mark_norm_unsupported(node_info, 'argument_mapping_unresolved', op,
                                            "the first connected input does not match the recorded activation.")
                continue
            slots = list(roles['slots'])
            # running-statistics update modes are excluded from this inference accounting:
            # an ESTABLISHED updating flag refuses whatever the tensor binding says
            has_stats = any(role in self._NORM_STAT_ROLES for role, _, _, _ in slots)
            _flag = {'batch_norm': 'training', 'instance_norm': 'use_input_stats'}.get(op)
            updating = (op == 'batch_norm' and roles['flags'].get('training') is True) or \
                       (op == 'instance_norm' and roles['flags'].get('use_input_stats') is True)
            if updating and has_stats:
                self._mark_norm_unsupported(
                    node_info, 'running_stats_update', op,
                    f"{roles['spelling']} training/use_input_stats mode updates the running statistics; "
                    "this inference accounting path does not represent those writes.")
                continue
            extras = [(connected[i], (node_info.input_dtypes[i] if i < len(node_info.input_dtypes) else None))
                      for i in range(1, len(connected))]
            extra_kinds = kinds[1:]
            if not roles['bound'] or any(kind is None for _, _, _, kind in slots):
                self._mark_norm_unsupported(
                    node_info, 'argument_mapping_unresolved', op,
                    f"the capture recorded no operand binding for the {roles['spelling']} call; which of the "
                    f"recorded argument slots {[r for r, _, _, _ in slots]} the connected inputs occupy is not "
                    "established and is not guessed.")
                continue
            bound_slots = [s for s in slots if s[3] != 'detached']
            mismatch = self._binding_mismatch(extras, extra_kinds, bound_slots)
            if mismatch:
                self._mark_norm_unsupported(
                    node_info, 'argument_mapping_unresolved', op,
                    f"the recorded {roles['spelling']} call's binding does not describe the connected inputs: {mismatch}")
                continue
            # the binding describes the connected inputs; the update mode must now be
            # ESTABLISHED by a Boolean (supplied, or the callable's own Boolean
            # default) — None / a number / a string / a container / no structured
            # record is refused, never coerced (none-9, flagtext-10)
            if _flag and has_stats and _flag not in roles['flags']:
                _raw = (roles['flags'].get('unestablished') or {}).get(_flag)
                self._mark_norm_unsupported(
                    node_info, 'argument_mapping_unresolved', op,
                    (f"the recorded {roles['spelling']} call supplies {_flag}={_raw}, which is not a Boolean and does not "
                     "establish the update mode (an explicit None is not eval mode; none-9)" if _raw is not None else
                     f"the recorded {roles['spelling']} call neither supplies {_flag!r} nor records its default from the "
                     "executed callable's signature") +
                    "; whether the running statistics are updated cannot be established.")
                continue
            to_attach = [(role, shape, dtype) for role, shape, dtype, kind in slots if kind == 'detached']
            if not to_attach:
                continue
            act_dtype = node_info.input_dtypes[0] if node_info.input_dtypes else None
            attached_records: List[Dict[str, Any]] = []
            for name, shape, dtype in to_attach:
                pid = f"{node_info.node_id}.{name}"
                while pid in id_to_node_info:
                    pid += "_"
                # Provenance of the attached parameter (consumed by the converter's
                # ``parameter_sources`` slot identity, an earlier change): a capture-style
                # parameter-tensor TensorNode whose id is the slot's connection,
                # plus an explicit record of where it came from — the executed
                # call's detached argument slot ``name``, bound as described in
                # ``module_args`` (binding_source / callable_type / callee).
                source = f"recorded_call:{node_info.node_id}:{name}"
                pnode = NodeInfo(
                    node_id=pid, type='parameter-tensor', node_class='TensorNode',
                    input_nodes=[], output_nodes=[node_info.node_id],
                    input_shapes=[], output_shapes=[list(shape)],
                    input_dtypes=[], output_dtypes=[dtype or act_dtype],
                    input_types=[], output_types=['output'],
                    module_args={'hierarchical_name': pid, 'parameter_of': node_info.node_id, 'parameter_name': name,
                                 'parameter_source': source},
                )
                id_to_node_info[pid] = pnode
                new_nodes.append(pnode)
                node_info.input_nodes.append(pid)
                node_info.input_shapes.append(list(shape))
                node_info.input_dtypes.append(dtype or act_dtype)
                node_info.input_types.append('weight')
                attached_records.append({'role': name, 'node_id': pid, 'slot': len(node_info.input_nodes) - 1,
                                         'shape': list(shape), 'dtype': dtype or act_dtype, 'source': source})
            node_info.module_args['attached_parameters'] = attached_records
            if self.debug:
                logger.debug(f"Attached {len(to_attach)} detached parameter tensor(s) to {node_info.node_id}: "
                      f"{[n for n, _, _ in to_attach]}")
        result.extend(new_nodes)

    @staticmethod
    def _innermost_parent_module_node(node: Any, visited: Optional[set] = None) -> Any:
        """The torchview ModuleNode whose forward produced ``node`` (walking the
        same parent chain as :meth:`_get_module_hierarchy_with_ids`), or None."""
        if visited is None:
            visited = set()
        if id(node) in visited:
            return None
        visited.add(id(node))
        parents = list(getattr(node, 'parents', []) or [])
        for parent in parents:
            if type(parent).__name__ == 'ModuleNode':
                return parent
        for parent in parents:
            if type(parent).__name__ == 'TensorNode':
                found = TorchviewProcessor._innermost_parent_module_node(parent, visited)
                if found is not None:
                    return found
        return None

    def _generate_clean_id(self, node: Any) -> str:
        """Generate a flat node ID with Model prefix.

        Format: Model.<opname>_<count>

        Args:
            node: Node object.

        Returns:
            Flat node ID string.
        """
        node_name = getattr(node, 'name', type(node).__name__.lower())

        # Use flat naming: Model.<op_name>_<count>
        op_key = f"Model.{node_name}"
        if op_key not in self._node_counter:
            self._node_counter[op_key] = 0
            count = 0
        else:
            self._node_counter[op_key] += 1
            count = self._node_counter[op_key]

        return f"Model.{node_name}_{count}" if count > 0 else f"Model.{node_name}"

    def _generate_hierarchical_name(self, node: Any) -> str:
        """Generate a hierarchical name showing the full module path.

        Format: Model.<level0name>_<idx>.<level1name>_<idx>.<opname>

        Args:
            node: Node object.

        Returns:
            Hierarchical name string.
        """
        node_name = getattr(node, 'name', type(node).__name__.lower())

        # Build hierarchical path from parent ModuleNodes (with their object IDs)
        hierarchy_with_ids = self._get_module_hierarchy_with_ids(node)

        # Convert hierarchy to indexed names
        indexed_hierarchy = self._index_hierarchy(hierarchy_with_ids)

        # Build full path: Model.<indexed_hierarchy>.<node_name>
        if indexed_hierarchy:
            base_path = 'Model.' + '.'.join(indexed_hierarchy)
        else:
            base_path = 'Model'

        # Add counter for the operation name uniqueness within this hierarchy
        if not hasattr(self, '_hierarchical_counter'):
            self._hierarchical_counter: Dict[str, int] = {}

        op_key = f"{base_path}.{node_name}"
        if op_key not in self._hierarchical_counter:
            self._hierarchical_counter[op_key] = 0
            count = 0
        else:
            self._hierarchical_counter[op_key] += 1
            count = self._hierarchical_counter[op_key]

        op_name_indexed = f"{node_name}_{count}" if count > 0 else node_name
        return f"{base_path}.{op_name_indexed}"

    def _get_module_hierarchy_with_ids(self, node: Any, visited: Optional[set] = None) -> List[Tuple[str, int]]:
        """Trace up parent chain to find ModuleNode hierarchy with object IDs.

        Args:
            node: Node object.
            visited: Set of visited node IDs to prevent cycles.

        Returns:
            List of (module_name, object_id) tuples from root to immediate parent.
        """
        if visited is None:
            visited = set()

        node_id = id(node)
        if node_id in visited:
            return []
        visited.add(node_id)

        node_class = type(node).__name__
        parents = list(getattr(node, 'parents', []))

        # Look for ModuleNode parent first
        for parent in parents:
            parent_class = type(parent).__name__
            if parent_class == 'ModuleNode':
                parent_name = getattr(parent, 'name', 'unknown')
                parent_obj_id = id(parent)
                # Recurse to get the ModuleNode's containment hierarchy
                parent_hierarchy = self._get_module_hierarchy_with_ids(parent, visited)
                return parent_hierarchy + [(parent_name, parent_obj_id)]

        # No direct ModuleNode parent - trace through TensorNode parents only
        # (TensorNodes are part of module containment, FunctionNodes are data flow)
        for parent in parents:
            parent_class = type(parent).__name__
            if parent_class == 'TensorNode':
                # Continue tracing through TensorNode to find containing ModuleNode
                parent_hierarchy = self._get_module_hierarchy_with_ids(parent, visited)
                if parent_hierarchy:
                    return parent_hierarchy

        # No module container found
        return []

    def _index_hierarchy(self, hierarchy_with_ids: List[Tuple[str, int]]) -> List[str]:
        """Convert hierarchy with IDs to indexed names.

        Tracks module names at each level and assigns indices when names repeat.
        Always adds index suffix when there are duplicates at that level, OR when
        the same module name appears multiple times in ANY hierarchy path.

        Args:
            hierarchy_with_ids: List of (module_name, object_id) tuples.

        Returns:
            List of indexed module names (e.g., ['MultiHeadAttention', 'Linear_0']).
        """
        if not hierarchy_with_ids:
            return []

        # Track seen (parent_path, name) -> {obj_id: index}
        # This allows us to assign consistent indices based on first-seen order
        if not hasattr(self, '_module_index_tracker'):
            self._module_index_tracker: Dict[Tuple[str, str], Dict[int, int]] = {}

        # Track which (parent_path, name) keys have duplicates
        if not hasattr(self, '_module_has_duplicates'):
            self._module_has_duplicates: set = set()

        # Track names that repeat in any path (set by prescan)
        if not hasattr(self, '_names_repeated_in_any_path'):
            self._names_repeated_in_any_path: set = set()

        result = []
        parent_path = 'root'

        # Track indices for names that repeat across any hierarchy path
        # This ensures consistent indexing even for nodes at different depths
        path_name_indices: Dict[str, int] = {}

        for module_name, obj_id in hierarchy_with_ids:
            key = (parent_path, module_name)

            if key not in self._module_index_tracker:
                self._module_index_tracker[key] = {}

            tracker = self._module_index_tracker[key]

            if obj_id not in tracker:
                # Assign next index for this module name at this level
                tracker[obj_id] = len(tracker)
                # If this is the second or later instance, mark as having duplicates
                if len(tracker) > 1:
                    self._module_has_duplicates.add(key)

            idx = tracker[obj_id]

            # Check if this name repeats in any hierarchy path (from prescan)
            if module_name in self._names_repeated_in_any_path:
                # Use path-local index for names that can repeat in hierarchies
                if module_name not in path_name_indices:
                    path_name_indices[module_name] = 0
                else:
                    path_name_indices[module_name] += 1
                indexed_name = f"{module_name}_{path_name_indices[module_name]}"
            elif key in self._module_has_duplicates:
                # Use global index for duplicates at the same level
                indexed_name = f"{module_name}_{idx}"
            else:
                indexed_name = module_name

            result.append(indexed_name)
            parent_path = f"{parent_path}.{indexed_name}"

        return result

    def _get_module_hierarchy(self, node: Any, visited: Optional[set] = None) -> List[str]:
        """Trace up parent chain to find ModuleNode hierarchy.

        Args:
            node: Node object.
            visited: Set of visited node IDs to prevent cycles.

        Returns:
            List of module names from root to immediate parent.
        """
        hierarchy_with_ids = self._get_module_hierarchy_with_ids(node, visited)
        return [name for name, _ in hierarchy_with_ids]

    def _extract_from_visual_graph(self, visual_graph: Any) -> List[NodeInfo]:
        """Extract nodes from the visual graph representation.

        Args:
            visual_graph: graphviz.Digraph object.

        Returns:
            List of NodeInfo objects.
        """
        if not hasattr(visual_graph, 'source'):
            return []

        nodes = {}
        edges = []

        # Parse graphviz source
        self._parse_graphviz_source(visual_graph.source, nodes, edges)

        # Build relationships
        for source_id, target_id in edges:
            if source_id in nodes and target_id in nodes:
                nodes[source_id].output_nodes.append(nodes[target_id].node_id)
                nodes[target_id].input_nodes.append(nodes[source_id].node_id)

        return list(nodes.values())

    def _parse_graphviz_source(self,
                              source: str,
                              nodes: Dict[str, NodeInfo],
                              edges: List[Tuple[str, str]]) -> None:
        """Parse graphviz source to extract nodes and edges.

        Args:
            source: Graphviz source string.
            nodes: Dictionary to populate with nodes.
            edges: List to populate with edges.
        """
        lines = source.split('\n')
        current_node_def = ""
        in_node_definition = False

        for line in lines:
            line = line.strip()

            # Check for node definition
            if re.match(r'^\d+\s+\[label=<', line):
                in_node_definition = True
                current_node_def = line
                node_id = re.match(r'^(\d+)', line).group(1)
            elif in_node_definition and line.endswith(']'):
                current_node_def += " " + line
                in_node_definition = False

                # Parse node definition
                node_info = self._parse_node_definition(node_id, current_node_def)
                if node_info:
                    nodes[node_id] = node_info
                current_node_def = ""
            elif in_node_definition:
                current_node_def += " " + line

            # Check for edge definition
            elif '->' in line and '[' not in line:
                edge_match = re.match(r'^(\d+)\s*->\s*(\d+)', line)
                if edge_match:
                    edges.append((edge_match.group(1), edge_match.group(2)))

    def _parse_node_definition(self, node_id: str, node_def: str) -> Optional[NodeInfo]:
        """Parse a node definition from graphviz source.

        Args:
            node_id: Node ID from graphviz.
            node_def: Node definition string.

        Returns:
            NodeInfo object if successfully parsed, None otherwise.
        """
        try:
            # Extract node type
            node_type = "Unknown"
            type_match = re.search(r'<TD[^>]*>([^<]*)<BR/>depth:\d+</TD>', node_def)
            if type_match:
                node_type = type_match.group(1).strip()

            # Extract shapes
            input_shapes = []
            output_shapes = []

            if node_type in ["input-tensor", "output-tensor"]:
                shape_match = re.search(r'<TD>\(([^)]+)\)</TD>', node_def)
                if shape_match:
                    shape_str = shape_match.group(1)
                    shape = [int(x.strip()) for x in shape_str.split(',')]
                    if node_type == "input-tensor":
                        output_shapes.append(shape)
                    else:
                        input_shapes.append(shape)

            # Create node ID
            hierarchical_id = f"Model.{node_type}_{node_id}"

            # Extract dtypes (empty for visual graph extraction)
            input_dtypes = []
            output_dtypes = []

            return NodeInfo(
                node_id=hierarchical_id,
                node_type=node_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes
            )

        except Exception as e:
            if self.debug:
                logger.debug(f"Error parsing node {node_id}: {e}")
            return None

    def _apply_model_parameters(self,
                               layer_nodes: List[NodeInfo],
                               model: nn.Module,
                               computation_nodes: Optional[Dict[str, Any]] = None) -> None:
        """Apply parameters from the original model to extracted nodes.

        This uses shape-based matching to correctly associate PyTorch modules
        with their corresponding function nodes. Also handles ModuleNode cases
        where the node directly references a PyTorch module.

        Args:
            layer_nodes: List of NodeInfo objects to update.
            model: Original PyTorch model.
            computation_nodes: Optional dict mapping original IDs to node objects.
        """
        # First, try to extract from ModuleNode objects directly
        if computation_nodes:
            for original_id, node in computation_nodes.items():
                if type(node).__name__ == 'ModuleNode':
                    clean_id = self._original_to_clean_id.get(original_id)
                    if clean_id:
                        node_info = next((n for n in layer_nodes if n.node_id == clean_id), None)
                        if node_info and node_info.node_id not in self._processed_nodes:
                            pytorch_module = self._get_pytorch_module(node)
                            if pytorch_module:
                                module_type = type(pytorch_module).__name__
                                self._apply_module_to_node(node_info, pytorch_module, module_type)
                                self._processed_nodes.add(node_info.node_id)
                                if self.debug:
                                    logger.debug(f"Applied module from ModuleNode: {node_info.node_id}")

        # Collect all modules from the model
        modules_by_type: Dict[str, List[Tuple[str, nn.Module]]] = {}
        for name, module in model.named_modules():
            if name == '':  # Skip root
                continue
            module_type = type(module).__name__
            if module_type not in modules_by_type:
                modules_by_type[module_type] = []
            modules_by_type[module_type].append((name, module))

        if self.debug:
            logger.debug(f"Found {sum(len(v) for v in modules_by_type.values())} PyTorch modules")

        # Match modules to nodes by type and shape
        for module_type, modules_list in modules_by_type.items():
            # Find candidate nodes for this module type (both FunctionNode and ModuleNode)
            candidate_nodes = [
                node for node in layer_nodes
                if node.node_class in ('FunctionNode', 'ModuleNode')
                and module_type.lower() in node.type.lower()
                and node.node_id not in self._processed_nodes
            ]

            if self.debug and modules_list:
                logger.debug(f"Matching {len(modules_list)} {module_type} modules to {len(candidate_nodes)} nodes")

            # Special handling for Linear layers with shape matching
            if module_type == 'Linear' and len(modules_list) > len(candidate_nodes) > 0:
                self._match_linear_modules_by_shape(modules_list, candidate_nodes)
            else:
                # Standard sequential matching
                for i, (module_name, module) in enumerate(modules_list):
                    if module_name in self._matched_modules:
                        continue

                    target_node = None
                    for node in candidate_nodes:
                        if node.node_id not in self._processed_nodes:
                            target_node = node
                            break

                    if target_node:
                        self._apply_module_to_node(target_node, module, module_type)
                        self._processed_nodes.add(target_node.node_id)
                        self._matched_modules.add(module_name)

    def _match_linear_modules_by_shape(self,
                                       modules_list: List[Tuple[str, nn.Module]],
                                       candidate_nodes: List[NodeInfo]) -> None:
        """Match Linear modules to nodes using shape-based matching.

        Args:
            modules_list: List of (name, module) tuples.
            candidate_nodes: List of candidate NodeInfo objects.
        """
        for node in candidate_nodes:
            if node.node_id in self._processed_nodes:
                continue

            # Get expected dimensions from node shapes
            if node.input_shapes and node.output_shapes:
                input_shape = node.input_shapes[0]
                output_shape = node.output_shapes[0]

                if len(input_shape) > 0 and len(output_shape) > 0:
                    expected_in_features = input_shape[-1]
                    expected_out_features = output_shape[-1]

                    # Find matching Linear module
                    for module_name, module in modules_list:
                        if module_name in self._matched_modules:
                            continue

                        if (hasattr(module, 'in_features') and
                            hasattr(module, 'out_features')):
                            if (module.in_features == expected_in_features and
                                module.out_features == expected_out_features):
                                self._apply_module_to_node(node, module, 'Linear')
                                self._processed_nodes.add(node.node_id)
                                self._matched_modules.add(module_name)
                                if self.debug:
                                    logger.debug(f"Shape match: {node.node_id} <-> {module_name}")
                                break

    def _apply_module_to_node(self,
                            node: NodeInfo,
                            module: nn.Module,
                            module_type: str) -> None:
        """Apply module arguments to a node.

        Args:
            node: NodeInfo to update.
            module: PyTorch module with parameters.
            module_type: Type name of the module.
        """
        node.module_args['module_type'] = module_type
        node.module_args.update(self._extract_module_arguments(module))

    def _save_pytorch_graph_yaml(
        self,
        layer_nodes: List[NodeInfo],
        filename: Path,
        *,
        model_name: str,
        capture_capability: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save extracted nodes to a structured YAML graph.

        The YAML format:

          model_name: <str>
          layers:
            <node_id>:
              type: <str>
              node_class: <str>
              input_shapes: [...]
              output_shapes: [...]
              input_types: [...]
              output_types: [...]
              module_args: {...}
              connections:
                inputs: [...]
                outputs: [...]

        Args:
            layer_nodes: Extracted nodes.
            filename: Output YAML path.
            model_name: Human-readable model name.
        """
        graph_dict: Dict[str, Any] = {
            "model_name": model_name,
            "layers": {},
        }
        if capture_capability is not None:
            # torchview capability record (issue #45): what the capture could
            # see; `degraded` captures are only produced under
            # --allow-degraded-capture and are marked not supportable downstream.
            graph_dict["capture_capability"] = dict(capture_capability)
            graph_dict["capture_status"] = "degraded" if capture_capability.get("degraded") else "ok"

        for node in layer_nodes:
            graph_dict["layers"][node.node_id] = node.to_dict()

        with open(filename, "w") as f:
            from solar.common.utils import NoAliasDumper
            yaml.dump(graph_dict, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            logger.debug(f"PyTorch graph YAML saved to {filename}")

    def _print_layer_summary(self, layer_nodes: List[NodeInfo]) -> None:
        """Print summary of extracted layer nodes.

        Args:
            layer_nodes: List of NodeInfo objects.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"EXTRACTED LAYER NODES ({len(layer_nodes)} nodes)")
        logger.info(f"{'='*80}")

        for i, node in enumerate(layer_nodes[:5], 1):  # Show first 5
            logger.info(f"\n[{i}] Node ID: {node.node_id}")
            logger.info(f"Type: {node.type} ({node.node_class})")
            logger.info(f"Input Nodes: {node.input_nodes}")
            logger.info(f"Output Nodes: {node.output_nodes}")
            logger.info(f"Input Shapes: {node.input_shapes}")
            logger.info(f"Output Shapes: {node.output_shapes}")
            logger.info(f"Input Dtypes: {node.input_dtypes}")
            logger.info(f"Output Dtypes: {node.output_dtypes}")
            if node.input_types:
                logger.info(f"Input Types: {node.input_types}")

        if len(layer_nodes) > 5:
            logger.info(f"\n... and {len(layer_nodes) - 5} more nodes")
