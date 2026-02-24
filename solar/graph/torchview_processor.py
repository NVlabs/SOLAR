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
- weight_nodes: List of parameter names (e.g., ["weight", "bias"])
- weight_shapes: List of parameter shapes
- module_args: Dictionary of module configuration arguments
"""

import json
import re
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
from solar.common.types import GraphInfo, NodeInfo, ShapeDict, TensorShape
from solar.common.utils import ensure_directory


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
    
    def process_graph(self,
                     computation_graph: Any,
                     output_dir: str,
                     kernel_name: str,
                     original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
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
            print(f"Processing torchview graph for {kernel_name}...")
        
        # Reset state for new graph
        self._reset_state()
        
        # Extract layer nodes
        layer_nodes = self._extract_layer_nodes(computation_graph, original_model)
        
        # Save canonical YAML graph (and remove any legacy JSON artifacts).
        output_path = Path(output_dir)
        ensure_directory(output_path)
        yaml_filename = output_path / "pytorch_graph.yaml"

        self._save_pytorch_graph_yaml(layer_nodes, yaml_filename, model_name=kernel_name)
        
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
                print(f"Extracting from edge_list ({len(computation_graph.edge_list)} edges)...")
            layer_nodes = self._extract_from_edge_list(computation_graph, original_model)
            return layer_nodes
        
        # Parse visual graph as fallback
        if hasattr(computation_graph, 'visual_graph'):
            if self.debug:
                print("Parsing visual graph...")
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
                    print(f"Warning: Could not extract module {module_node}: {e}")
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
        
        # Extract module parameters
        module_info = self._extract_module_info(node)
        
        return NodeInfo(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class_name,
            input_nodes=[],  # Will be filled in by relationship building
            output_nodes=[],  # Will be filled in by relationship building
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            weight_nodes=module_info['weight_nodes'],
            weight_shapes=module_info['weight_shapes'],
            module_args=module_info['module_args']
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
                    # Input tensors: shape is output (what they produce)
                    output_shapes = [tensor_shape]
                elif node_name == 'output-tensor':
                    # Final output tensors: shape is input (what they receive)
                    input_shapes = [tensor_shape]
                elif node_name == 'hidden-tensor':
                    # Hidden tensors: intermediate nodes have both input and output shapes
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                else:
                    # Unknown tensor type: add to both
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
            # Try to get dtype from model parameters or inputs
            default_dtype = None
            
            # Check model parameters for dtype
            for param in original_model.parameters():
                if param.dtype is not None:
                    default_dtype = str(param.dtype)
                    break
            
            # If no parameter dtype found, use default float32
            if default_dtype is None:
                default_dtype = "torch.float32"
            
            # Fill in missing input dtypes to match input_shapes count
            while len(input_dtypes) < len(input_shapes):
                input_dtypes.append(default_dtype)
            
            # Fill in missing output dtypes to match output_shapes count
            while len(output_dtypes) < len(output_shapes):
                output_dtypes.append(default_dtype)
        
        return input_dtypes, output_dtypes
    
    def _extract_module_info(self, node: Any) -> Dict[str, Any]:
        """Extract module parameter information from a node.
        
        Args:
            node: Node object.
            
        Returns:
            Dictionary with weight_nodes, weight_shapes, and module_args.
        """
        module_info = {
            'weight_nodes': [],
            'weight_shapes': [],
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
        # Try to access the underlying PyTorch module
        module = self._get_pytorch_module(node)
        
        if module is not None:
            # Extract parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param is not None:
                    module_info['weight_nodes'].append(param_name)
                    module_info['weight_shapes'].append(list(param.shape))
            
            # Extract buffers
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer is not None:
                    module_info['weight_nodes'].append(buffer_name)
                    module_info['weight_shapes'].append(list(buffer.shape))
            
            # Extract module arguments
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
        
        # Extract from args attribute (fallback)
        if hasattr(node, 'args') and node.args:
            for i, arg in enumerate(node.args):
                if hasattr(arg, 'shape') and hasattr(arg, 'numel'):
                    param_name = self._infer_parameter_name(node_name, i, list(arg.shape))
                    if param_name != 'input':
                        module_info['weight_nodes'].append(param_name)
                        module_info['weight_shapes'].append(list(arg.shape))
                        # Infer module arguments from parameter shapes
                        self._infer_module_arguments(node_name, param_name, list(arg.shape), module_info['module_args'])
        
        # Extract from kwargs
        if hasattr(node, 'kwargs') and node.kwargs:
            for key, value in node.kwargs.items():
                if hasattr(value, 'shape'):
                    module_info['weight_nodes'].append(key)
                    module_info['weight_shapes'].append(list(value.shape))
                else:
                    module_info['module_args'][key] = value

        # ------------------------------------------------------------------
        # Fallback: extract weight info from raw_attributes for known ops
        # (linear, conv*, batch_norm, layer_norm, …).  This covers the
        # common case where torchview FunctionNodes don't expose `args`
        # with actual tensor data but DO record tensor shapes in the
        # stringified `attributes` field.
        # ------------------------------------------------------------------
        if (
            not module_info['weight_nodes']
            and hasattr(node, 'attributes')
            and node.attributes
        ):
            w_nodes, w_shapes = self._extract_weights_from_attributes(
                node.attributes, node_name
            )
            if w_nodes:
                module_info['weight_nodes'] = w_nodes
                module_info['weight_shapes'] = w_shapes
                # Also infer module_args (in_features, out_features, …)
                for param_name, param_shape in zip(w_nodes, w_shapes):
                    self._infer_module_arguments(
                        node_name, param_name, param_shape,
                        module_info['module_args'],
                    )
    
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
                        
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to parse attributes for {node_name}: {e}")
        
        return result
    
    def _eval_attributes_string(
        self,
        attributes: str,
    ) -> Tuple[Optional[List[Any]], Optional[Dict[str, Any]]]:
        """Safely evaluate torchview attributes string.
        
        Replaces Tensor(...) with a placeholder dict and evaluates the string.
        
        Args:
            attributes: Stringified attributes from torchview.
            
        Returns:
            Tuple of (args_list, kwargs_dict) or (None, None) on failure.
        """
        import re
        
        try:
            # Replace Tensor(shape=(...), dtype=...) with a placeholder dict
            # Pattern matches: Tensor(shape=(1, 2, 3), dtype=torch.float32)
            def replace_tensor(match: re.Match) -> str:
                return "{'tensor_placeholder': True}"
            
            # Replace all Tensor(...) occurrences
            # Handle nested parentheses by matching Tensor( then everything until matching )
            processed = attributes
            
            # Simple approach: replace Tensor(...) patterns
            # This regex handles nested parens by being greedy within the Tensor() call
            tensor_pattern = r'Tensor\([^)]*(?:\([^)]*\)[^)]*)*\)'
            processed = re.sub(tensor_pattern, "{'tensor_placeholder': True}", processed)
            
            # Replace torch.dtype references
            processed = re.sub(r'torch\.\w+', 'None', processed)
            
            # Now safely evaluate
            parsed = eval(processed, {"__builtins__": {}}, {})
            
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                args_list = parsed[0] if isinstance(parsed[0], (list, tuple)) else []
                kwargs_dict = parsed[1] if isinstance(parsed[1], dict) else {}
                return list(args_list), kwargs_dict
            
            return None, None
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to eval attributes: {e}")
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
            print(f"  Found {len(computation_nodes)} unique computation nodes")
        
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
        
        if self.debug:
            # Count nodes with connections
            nodes_with_inputs = sum(1 for n in result if n.input_nodes)
            nodes_with_outputs = sum(1 for n in result if n.output_nodes)
            print(f"  Nodes with input connections: {nodes_with_inputs}")
            print(f"  Nodes with output connections: {nodes_with_outputs}")
        
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
        'embedding':        [(1, 'weight')],
    }
    
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
                print(f"Error parsing node {node_id}: {e}")
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
                                    print(f"  Applied module from ModuleNode: {node_info.node_id}")
        
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
            print(f"  Found {sum(len(v) for v in modules_by_type.values())} PyTorch modules")
        
        # Match modules to nodes by type and shape
        for module_type, modules_list in modules_by_type.items():
            # Find candidate nodes for this module type (both FunctionNode and ModuleNode)
            candidate_nodes = [
                node for node in layer_nodes
                if node.node_class in ('FunctionNode', 'ModuleNode')
                and module_type.lower() in node.node_type.lower()
                and node.node_id not in self._processed_nodes
            ]
            
            if self.debug and modules_list:
                print(f"  Matching {len(modules_list)} {module_type} modules to {len(candidate_nodes)} nodes")
            
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
                                    print(f"    Shape match: {node.node_id} <-> {module_name}")
                                break
    
    def _apply_module_to_node(self,
                            node: NodeInfo,
                            module: nn.Module,
                            module_type: str) -> None:
        """Apply module parameters to a node.
        
        Args:
            node: NodeInfo to update.
            module: PyTorch module with parameters.
            module_type: Type name of the module.
        """
        # Clear existing parameters to avoid duplicates
        node.weight_nodes.clear()
        node.weight_shapes.clear()
        
        # Extract parameters
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None:
                node.weight_nodes.append(param_name)
                node.weight_shapes.append(list(param.shape))
        
        # Extract buffers
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                node.weight_nodes.append(buffer_name)
                node.weight_shapes.append(list(buffer.shape))
        
        # Update module arguments
        node.module_args['module_type'] = module_type
        node.module_args.update(self._extract_module_arguments(module))
    
    def _save_pytorch_graph_yaml(
        self,
        layer_nodes: List[NodeInfo],
        filename: Path,
        *,
        model_name: str,
    ) -> None:
        """Save extracted nodes to a structured YAML graph.

        The YAML format matches the original process_torchview_graph.py output:

          model_name: <str>
          layers:
            <node_id>:
              type: <str>
              node_class: <str>
              input_shapes: [...]
              output_shapes: [...]
              weight_nodes: [...]
              weight_shapes: [...]
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

        for node in layer_nodes:
            graph_dict["layers"][node.node_id] = {
                "type": node.node_type,
                "node_class": node.node_class,
                "input_shapes": node.input_shapes,
                "output_shapes": node.output_shapes,
                "input_dtypes": node.input_dtypes,
                "output_dtypes": node.output_dtypes,
                "weight_nodes": node.weight_nodes,
                "weight_shapes": node.weight_shapes,
                "module_args": node.module_args,
                "connections": {
                    "inputs": node.input_nodes,
                    "outputs": node.output_nodes,
                },
            }

        with open(filename, "w") as f:
            from solar.common.utils import NoAliasDumper
            yaml.dump(graph_dict, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            print(f"PyTorch graph YAML saved to {filename}")
    
    def _print_layer_summary(self, layer_nodes: List[NodeInfo]) -> None:
        """Print summary of extracted layer nodes.
        
        Args:
            layer_nodes: List of NodeInfo objects.
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTED LAYER NODES ({len(layer_nodes)} nodes)")
        print(f"{'='*80}")
        
        for i, node in enumerate(layer_nodes[:5], 1):  # Show first 5
            print(f"\n[{i}] Node ID: {node.node_id}")
            print(f"    Type: {node.node_type} ({node.node_class})")
            print(f"    Input Nodes: {node.input_nodes}")
            print(f"    Output Nodes: {node.output_nodes}")
            print(f"    Input Shapes: {node.input_shapes}")
            print(f"    Output Shapes: {node.output_shapes}")
            print(f"    Input Dtypes: {node.input_dtypes}")
            print(f"    Output Dtypes: {node.output_dtypes}")
            if node.weight_nodes:
                print(f"    Weights: {node.weight_nodes}")
        
        if len(layer_nodes) > 5:
            print(f"\n... and {len(layer_nodes) - 5} more nodes")
