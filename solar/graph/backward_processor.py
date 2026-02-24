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

"""Backward pass graph extraction for Solar.

This module extracts backward computation graphs from PyTorch models using
autograd and torchview/fx to capture gradient computation operations.

Based on:
- torchviz: https://github.com/szagoruyko/pytorchviz
- torch.fx: https://pytorch.org/docs/stable/fx.html
- functorch aot_autograd: https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn

try:
    import torchview
except ImportError:
    torchview = None

from solar.common.utils import ensure_directory
from solar.common.types import ProcessingConfig
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.graph.torchview_processor import TorchviewProcessor


class BackwardProcessor:
    """Extract backward computation graphs from PyTorch models.
    
    This class captures the backward pass computation graph by:
    1. Running forward pass
    2. Computing loss
    3. Calling backward() to build the computation graph
    4. Extracting the backward graph using torchview or fx
    
    Assumption: No recompute (intermediate activations are stored during forward).
    """
    
    def __init__(self, debug: bool = False):
        """Initialize BackwardProcessor.
        
        Args:
            debug: Enable debug output.
        """
        self.debug = debug
        config = ProcessingConfig(debug=debug)
        self._forward_processor = PyTorchProcessor(config=config)
        self._torchview_processor = TorchviewProcessor(debug=debug)
    
    def extract_backward_graph(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target: torch.Tensor,
        output_dir: str,
        model_name: str = "model",
    ) -> Optional[Dict[str, Any]]:
        """Extract backward computation graph.
        
        Args:
            model: PyTorch model.
            inputs: List of input tensors for forward pass.
            loss_fn: Loss function that takes (output, target) -> scalar loss.
            target: Target tensor for loss computation.
            output_dir: Directory to save backward graph.
            model_name: Name for the model (used in file naming).
            
        Returns:
            Dictionary containing backward graph data, or None if failed.
        """
        if self.debug:
            print(f"Extracting backward graph for {model_name}...")
        
        try:
            # Ensure model is in eval mode and requires grad
            model.eval()
            for param in model.parameters():
                param.requires_grad = True
            
            # Ensure inputs require grad
            inputs_with_grad = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    inp_grad = inp.clone().detach().requires_grad_(True)
                    inputs_with_grad.append(inp_grad)
                else:
                    inputs_with_grad.append(inp)
            
            # Run forward pass
            with torch.set_grad_enabled(True):
                output = model(*inputs_with_grad)
                
                # Compute loss
                loss = loss_fn(output, target)
                
                # Backward pass - this builds the computation graph
                loss.backward()
            
            # Try grad_fn extraction first (better for backward operation-level info)
            # This gives us the actual backward function types (AddmmBackward, GELUBackward, etc.)
            # and can extract torch function information from saved_tensors
            backward_graph = self._extract_backward_from_grad_fn(
                model, inputs_with_grad, output, loss, target, loss_fn, output_dir, model_name
            )
            
            # FX extraction as fallback (captures low-level ATen ops, less useful for backward)
            if backward_graph is None:
                backward_graph = self._extract_backward_with_fx(
                    model, inputs_with_grad, output, loss, target, output_dir, model_name, loss_fn
                )
            
            # Final fallback to simplified manual extraction
            if backward_graph is None:
                backward_graph = self._extract_backward_manual(
                    model, inputs_with_grad, output, loss, target, output_dir, model_name, loss_fn
                )
            
            return backward_graph
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting backward graph: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_backward_graph_torchview(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        output: torch.Tensor,
        loss: torch.Tensor,
        target: torch.Tensor,
        output_dir: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract backward graph using torchview.
        
        This method wraps the forward+backward computation in a function
        and uses torchview to visualize the backward graph.
        """
        if torchview is None:
            if self.debug:
                print("torchview not available, skipping torchview extraction")
            return None
        
        try:
            # Note: We'll use the manual extraction method instead
            # as torchview is primarily designed for forward graphs
            
            # Use manual extraction method
            # torchview is primarily designed for forward graphs
            return self._extract_backward_manual(
                model, inputs, output, loss, target, output_dir, model_name, loss_fn
            )
            
        except Exception as e:
            if self.debug:
                print(f"Error in torchview backward extraction: {e}")
            return None
    
    
    def _extract_backward_manual(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        output: torch.Tensor,
        loss: torch.Tensor,
        target: torch.Tensor,
        output_dir: str,
        model_name: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """Extract backward graph by tracing autograd graph and using torchview.
        
        This method:
        1. Runs forward and backward
        2. Uses torchview to capture backward operations
        3. Extracts backward operations from grad_fn chain
        4. Builds a backward graph with proper connections
        """
        try:
            # Reset model
            model.train()  # Use train mode to enable gradient computation
            for param in model.parameters():
                param.requires_grad = True
                if param.grad is not None:
                    param.grad.zero_()
            
            # Prepare inputs with gradients
            inputs_with_grad = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    inp_grad = inp.clone().detach().requires_grad_(True)
                    inputs_with_grad.append(inp_grad)
                else:
                    inputs_with_grad.append(inp)
            
            # Create a wrapper function that includes backward for torchview
            def forward_backward_fn(*args):
                """Wrapper that runs forward and backward."""
                # Reset gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                
                # Forward
                out = model(*args)
                loss_val = loss_fn(out, target)
                
                # Backward
                loss_val.backward()
                
                # Return loss and gradients for visualization
                return loss_val, out
            
            # Use torchview to capture the backward graph
            # We'll trace the forward+backward computation
            backward_graph = None
            
            if torchview is not None:
                try:
                    # Create a wrapper module for torchview (it expects nn.Module, not function)
                    class BackwardWrapper(nn.Module):
                        def __init__(self, model, loss_fn, target):
                            super().__init__()
                            self.model = model
                            self.loss_fn = loss_fn
                            self.target = target
                        
                        def forward(self, *args):
                            out = self.model(*args)
                            loss = self.loss_fn(out, self.target)
                            loss.backward()
                            return loss, out
                    
                    wrapper = BackwardWrapper(model, loss_fn, target)
                    wrapper.eval()
                    
                    # Try to capture graph with torchview
                    graph = torchview.draw_graph(
                        wrapper,
                        input_data=inputs_with_grad,
                        device="cpu",
                        save_graph=True,
                        expand_nested=True,
                        depth=float('inf'),
                        hide_module_functions=False,
                        hide_inner_tensors=False,
                        collect_attributes=True,
                        directory=str(Path(output_dir).parent),
                        filename=f"{model_name}_backward_graph",
                    )
                    
                    # Process the graph to extract backward operations
                    backward_graph = self._extract_backward_from_torchview_graph(
                        graph, model, inputs_with_grad, output_dir, model_name
                    )
                    
                    # Save PDF visualization
                    if hasattr(graph, 'visual_graph'):
                        pdf_path = Path(output_dir) / "torchview_graph.pdf"
                        try:
                            graph.visual_graph.render(
                                format='pdf',
                                filename=str(pdf_path.with_suffix('')),
                                cleanup=True,
                                view=False
                            )
                            if self.debug:
                                print(f"Saved backward graph PDF to {pdf_path}")
                        except Exception as e:
                            if self.debug:
                                print(f"Could not save PDF: {e}")
                    
                    # Also try to save using the forward processor's graph saving
                    try:
                        # Use torchview processor to save the graph
                        self._torchview_processor.process_graph(
                            graph, str(output_dir), f"{model_name}_backward", model
                        )
                    except Exception as e:
                        if self.debug:
                            print(f"Could not process graph with torchview processor: {e}")
                    
                except Exception as e:
                    if self.debug:
                        print(f"torchview backward extraction failed: {e}")
            
            # Fallback: Extract from grad_fn chain
            if backward_graph is None:
                backward_graph = self._extract_backward_from_grad_fn(
                    model, inputs_with_grad, output, loss, target, loss_fn, output_dir, model_name
                )
            
            return backward_graph
            
        except Exception as e:
            if self.debug:
                print(f"Error in manual backward extraction: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_backward_with_fx(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        output: torch.Tensor,
        loss: torch.Tensor,
        target: torch.Tensor,
        output_dir: str,
        model_name: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """Extract backward graph using torch.fx to expose actual torch function calls.
        
        This method uses functorch's aot_autograd or torch.fx to trace backward
        operations and extract the actual PyTorch function calls.
        """
        try:
            # Method 1: Try functorch aot_autograd (best for backward extraction)
            # Note: This captures the full computation including backward, but shows
            # low-level ATen operations. For better backward operation extraction,
            # we'll use grad_fn traversal which gives us the actual backward function types.
            try:
                try:
                    from functorch.compile import aot_function
                    from functorch import make_fx
                except ImportError:
                    # Try alternative import path
                    try:
                        from torch._functorch.compile import aot_function
                        from torch._functorch import make_fx
                    except ImportError:
                        raise ImportError("functorch not available")
                
                # Use aot_function to get forward and backward graphs separately
                def forward_fn(*args):
                    out = model(*args)
                    loss_val = loss_fn(out, target)
                    return loss_val
                
                # aot_function provides forward and backward graphs
                try:
                    # This gives us the compiled forward and backward graphs
                    compiled_fn = aot_function(forward_fn)
                    # Run it to get the graphs
                    _ = compiled_fn(*inputs)
                    
                    if self.debug:
                        print("aot_function compiled successfully")
                        print("Note: Extracting backward from grad_fn chain provides better operation-level info")
                    
                except Exception as e:
                    if self.debug:
                        print(f"aot_function failed: {e}, falling back to grad_fn extraction")
                
            except ImportError:
                if self.debug:
                    print("functorch not available, using grad_fn extraction (better for backward ops)")
            except Exception as e:
                if self.debug:
                    print(f"functorch extraction failed: {e}, using grad_fn extraction")
            
            # Method 2: Use torch.fx.symbolic_trace (limited for backward)
            try:
                import torch.fx as fx
                
                # Create a wrapper module that includes backward
                class BackwardTracer(nn.Module):
                    def __init__(self, model, loss_fn, target):
                        super().__init__()
                        self.model = model
                        self.loss_fn = loss_fn
                        self.target = target
                    
                    def forward(self, *args):
                        out = self.model(*args)
                        loss = self.loss_fn(out, self.target)
                        # Note: FX can't directly trace backward, but we can
                        # trace the forward operations and infer backward
                        return loss, out
                
                tracer = BackwardTracer(model, loss_fn, target)
                fx_graph = fx.symbolic_trace(tracer)
                
                if self.debug:
                    print(f"FX symbolic trace: {len(fx_graph.graph.nodes)} nodes")
                
                # Extract operations from FX graph
                return self._extract_backward_from_fx_graph(
                    fx_graph, model, inputs, output_dir, model_name
                )
                
            except Exception as e:
                if self.debug:
                    print(f"torch.fx extraction failed: {e}")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"Error in FX backward extraction: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_backward_from_fx_graph(
        self,
        fx_graph: Any,
        model: nn.Module,
        inputs: List[torch.Tensor],
        output_dir: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract backward operations from an FX graph.
        
        This extracts the actual torch function calls from the FX graph.
        """
        try:
            backward_graph = {
                "model_name": f"{model_name}_backward",
                "layers": {}
            }
            
            # Extract nodes from FX graph
            node_counter = 0
            node_id_map = {}
            
            # Traverse FX graph nodes
            import torch.fx as fx
            for fx_node in list(fx_graph.graph.nodes):
                node_id = f"backward_fx_{node_counter}"
                node_counter += 1
                node_id_map[fx_node] = node_id
                
                # Get operation type
                op_type = fx_node.op
                target = fx_node.target if hasattr(fx_node, 'target') else None
                
                # Map FX operation to operation type
                if op_type == "call_function":
                    # Extract actual torch function name
                    if target is not None:
                        func_name = str(target)
                        if hasattr(target, '__name__'):
                            func_name = target.__name__
                        elif hasattr(target, '__qualname__'):
                            func_name = target.__qualname__
                        
                        # Map to operation type
                        backward_op_type = self._map_torch_function_to_op_type(func_name)
                    else:
                        backward_op_type = "unknown_backward"
                elif op_type == "call_method":
                    method_name = str(target) if target else "unknown"
                    backward_op_type = self._map_torch_method_to_op_type(method_name)
                elif op_type == "call_module":
                    backward_op_type = "module_backward"
                else:
                    backward_op_type = f"{op_type}_backward"
                
                # Get input/output shapes
                input_shapes = []
                output_shape = []
                
                # Try to get shapes from args
                if hasattr(fx_node, 'args'):
                    for arg in fx_node.args:
                        if isinstance(arg, torch.Tensor):
                            input_shapes.append(list(arg.shape))
                        elif isinstance(arg, fx.Node):
                            # Reference to another node
                            pass
                
                # Get output shape from users
                if hasattr(fx_node, 'users'):
                    for user in fx_node.users:
                        if isinstance(user, fx.Node) and hasattr(user, 'args'):
                            for arg in user.args:
                                if isinstance(arg, torch.Tensor):
                                    output_shape = list(arg.shape)
                                    break
                
                # Get input connections
                input_nodes = []
                if hasattr(fx_node, 'args'):
                    for arg in fx_node.args:
                        if isinstance(arg, fx.Node) and arg in node_id_map:
                            input_nodes.append(node_id_map[arg])
                
                # Get output connections
                output_nodes = []
                if hasattr(fx_node, 'users'):
                    for user in fx_node.users:
                        if isinstance(user, fx.Node):
                            if user in node_id_map:
                                output_nodes.append(node_id_map[user])
                            else:
                                # Create node ID for user
                                user_id = f"backward_fx_{node_counter}"
                                node_counter += 1
                                node_id_map[user] = user_id
                                output_nodes.append(user_id)
                
                # Create layer entry
                backward_graph["layers"][node_id] = {
                    "type": backward_op_type,
                    "node_class": "FunctionNode",
                    "input_shapes": input_shapes if input_shapes else [],
                    "output_shapes": [output_shape] if output_shape else [],
                    "weight_nodes": [],
                    "weight_shapes": [],
                    "module_args": {
                        "function_name": str(target) if target else op_type,
                        "hierarchical_name": f"{model_name}.{node_id}",
                        "fx_op": op_type,
                        "fx_target": str(target) if target else None,
                        "fx_args": str(fx_node.args) if hasattr(fx_node, 'args') else None,
                    },
                    "connections": {
                        "inputs": input_nodes,
                        "outputs": output_nodes
                    }
                }
            
            # Save backward graph
            output_path = Path(output_dir)
            ensure_directory(output_path)
            
            import yaml
            from solar.common.utils import NoAliasDumper
            
            yaml_path = output_path / "pytorch_graph.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    backward_graph,
                    f,
                    Dumper=NoAliasDumper,
                    sort_keys=False,
                    default_flow_style=False
                )
            
            if self.debug:
                print(f"Saved FX-extracted backward graph to {yaml_path}")
                print(f"Extracted {len(backward_graph['layers'])} backward operations")
            
            return backward_graph
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting from FX graph: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _map_torch_function_to_op_type(self, func_name: str) -> str:
        """Map torch function name to operation type."""
        func_lower = func_name.lower()
        
        # Map common torch functions
        func_map = {
            'addmm': 'addmm_backward',
            'mm': 'mm_backward',
            'matmul': 'matmul_backward',
            'linear': 'linear_backward',
            'gelu': 'gelu_backward',
            'relu': 'relu_backward',
            'add': 'add_backward',
            'mul': 'mul_backward',
            'div': 'div_backward',
            'sub': 'sub_backward',
            'sum': 'sum_backward',
            'mean': 'mean_backward',
            'view': 'view_backward',
            'reshape': 'reshape_backward',
            'transpose': 'transpose_backward',
            't': 't_backward',
            'permute': 'permute_backward',
            'cat': 'cat_backward',
            'stack': 'stack_backward',
            'mse_loss': 'mse_loss_backward',
            'cross_entropy': 'cross_entropy_backward',
        }
        
        for key, value in func_map.items():
            if key in func_lower:
                return value
        
        return f"{func_lower}_backward"
    
    def _map_torch_method_to_op_type(self, method_name: str) -> str:
        """Map torch method name to operation type."""
        return self._map_torch_function_to_op_type(method_name)
    
    def _extract_backward_from_torchview_graph(
        self,
        graph: Any,
        model: nn.Module,
        inputs: List[torch.Tensor],
        output_dir: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract backward operations from torchview graph.
        
        Note: torchview primarily captures forward graphs, so this is a best-effort
        extraction. We'll use the forward graph structure and infer backward operations.
        """
        # For now, fall back to grad_fn extraction
        # A full implementation would traverse the torchview graph nodes
        # and identify backward operations
        return None
    
    def _extract_backward_from_grad_fn(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        output: torch.Tensor,
        loss: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        output_dir: str,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract backward graph by traversing grad_fn chain.
        
        This method walks the autograd graph starting from the loss
        and extracts backward operations and their connections.
        """
        try:
            # Run forward and backward to build the graph
            output = model(*inputs)
            loss = loss_fn(output, target)
            loss.backward()
            
            # Build backward graph structure
            backward_graph = {
                "model_name": f"{model_name}_backward",
                "layers": {}
            }
            
            # Traverse the grad_fn chain from loss to extract backward operations
            visited_grad_fns = {}
            backward_op_nodes = []
            
            def traverse_grad_fn(grad_fn, node_name_prefix="backward", depth=0):
                """Recursively traverse grad_fn chain to extract backward operations."""
                if grad_fn is None or depth > 50:  # Limit depth to avoid infinite recursion
                    return None
                
                # Use grad_fn object id as key to avoid duplicates
                grad_fn_id = id(grad_fn)
                if grad_fn_id in visited_grad_fns:
                    return visited_grad_fns[grad_fn_id]
                
                grad_fn_name = type(grad_fn).__name__
                node_id = f"{node_name_prefix}_{len(backward_op_nodes)}"
                
                # Get shapes from saved tensors if available
                input_shapes = []
                saved_tensors_info = []
                if hasattr(grad_fn, 'saved_tensors'):
                    for saved_tensor in grad_fn.saved_tensors:
                        if hasattr(saved_tensor, 'shape'):
                            input_shapes.append(list(saved_tensor.shape))
                            saved_tensors_info.append({
                                "shape": list(saved_tensor.shape),
                                "dtype": str(saved_tensor.dtype) if hasattr(saved_tensor, 'dtype') else None
                            })
                
                # Get output shape from next_functions (backward flow)
                output_shape = []
                if hasattr(grad_fn, 'next_functions') and grad_fn.next_functions:
                    # Try to infer output shape from the operation
                    # This is approximate - a full implementation would track tensor shapes
                    if input_shapes:
                        output_shape = input_shapes[0]  # Default: same as first input
                
                # Extract actual torch function calls from grad_fn
                # Many backward operations expose their forward operations through saved_tensors
                torch_function_calls = self._extract_torch_functions_from_grad_fn(grad_fn)
                
                # Create backward operation node
                backward_op_type = self._map_grad_fn_to_op_type(grad_fn_name)
                
                # Get next nodes (inputs to this backward op)
                next_nodes = []
                if hasattr(grad_fn, 'next_functions'):
                    for i, (next_fn, _) in enumerate(grad_fn.next_functions):
                        if next_fn is not None:
                            next_node_id = traverse_grad_fn(next_fn, f"{node_id}_input_{i}", depth + 1)
                            if next_node_id:
                                next_nodes.append(next_node_id)
                
                backward_graph["layers"][node_id] = {
                    "type": backward_op_type,
                    "node_class": "FunctionNode",
                    "input_shapes": input_shapes if input_shapes else [output_shape] if output_shape else [],
                    "output_shapes": [output_shape] if output_shape else input_shapes[:1] if input_shapes else [],
                    "weight_nodes": [],
                    "weight_shapes": [],
                    "module_args": {
                        "function_name": grad_fn_name.lower(),
                        "hierarchical_name": f"{model_name}.{node_id}",
                        "raw_attributes": str(grad_fn),
                        "torch_functions": torch_function_calls,
                        "saved_tensors": saved_tensors_info
                    },
                    "connections": {
                        "inputs": next_nodes,
                        "outputs": []
                    }
                }
                
                visited_grad_fns[grad_fn_id] = node_id
                backward_op_nodes.append(node_id)
                return node_id
            
            # Start traversal from loss's grad_fn
            loss_grad_fn = loss.grad_fn if hasattr(loss, 'grad_fn') else None
            if loss_grad_fn:
                loss_node_id = traverse_grad_fn(loss_grad_fn, "loss_backward")
            else:
                loss_node_id = None
            
            # Add input gradient nodes
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor) and inp.grad is not None:
                    node_id = f"input_grad_{i}"
                    backward_graph["layers"][node_id] = {
                        "type": "auxiliary-tensor",
                        "node_class": "TensorNode",
                        "input_shapes": [],
                        "output_shapes": [list(inp.grad.shape)],
                        "weight_nodes": [],
                        "weight_shapes": [],
                        "module_args": {
                            "hierarchical_name": f"{model_name}.input_grad_{i}"
                        },
                        "connections": {
                            "inputs": [loss_node_id] if loss_node_id else [],
                            "outputs": []
                        }
                    }
            
            # Add parameter gradient nodes
            param_idx = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    node_id = f"param_grad_{param_idx}"
                    backward_graph["layers"][node_id] = {
                        "type": "auxiliary-tensor",
                        "node_class": "TensorNode",
                        "input_shapes": [],
                        "output_shapes": [list(param.grad.shape)],
                        "weight_nodes": [],
                        "weight_shapes": [],
                        "module_args": {
                            "hierarchical_name": f"{model_name}.{name}.grad"
                        },
                        "connections": {
                            "inputs": [loss_node_id] if loss_node_id else [],
                            "outputs": []
                        }
                    }
                    param_idx += 1
            
            # Connect backward operations - update outputs for each node based on inputs
            # For each backward operation, find which nodes use it as input
            node_to_outputs = {}
            for node_id, node_data in backward_graph["layers"].items():
                inputs = node_data["connections"]["inputs"]
                for input_node in inputs:
                    if input_node not in node_to_outputs:
                        node_to_outputs[input_node] = []
                    if node_id not in node_to_outputs[input_node]:
                        node_to_outputs[input_node].append(node_id)
            
            # Update outputs for all nodes
            for node_id, outputs in node_to_outputs.items():
                if node_id in backward_graph["layers"]:
                    # Merge with existing outputs
                    existing_outputs = backward_graph["layers"][node_id]["connections"]["outputs"]
                    backward_graph["layers"][node_id]["connections"]["outputs"] = list(set(existing_outputs + outputs))
            
            # Connect backward operations to gradient outputs
            # AccumulateGrad operations output to parameter gradients
            # Find AccumulateGrad nodes and connect them to param gradients
            accumulate_grad_nodes = [
                node_id for node_id in backward_graph["layers"]
                if backward_graph["layers"][node_id]["type"] == "accumulategrad_backward"
            ]
            
            # Connect AccumulateGrad nodes to parameter gradients
            param_grad_idx = 0
            for node_id in accumulate_grad_nodes:
                if param_grad_idx < param_idx:
                    grad_node_id = f"param_grad_{param_grad_idx}"
                    if grad_node_id in backward_graph["layers"]:
                        backward_graph["layers"][node_id]["connections"]["outputs"].append(grad_node_id)
                        backward_graph["layers"][grad_node_id]["connections"]["inputs"].append(node_id)
                    param_grad_idx += 1
            
            # Connect input gradients - find the backward op that produces input gradients
            # This is typically the last operation before AccumulateGrad for inputs
            if backward_op_nodes:
                # The first backward op (closest to loss) may produce input gradients
                first_backward_op = backward_op_nodes[0]
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor) and inp.grad is not None:
                        grad_node_id = f"input_grad_{i}"
                        if grad_node_id in backward_graph["layers"]:
                            # Find the backward op that produces this input gradient
                            # For now, connect to the first backward op
                            if first_backward_op not in backward_graph["layers"][grad_node_id]["connections"]["inputs"]:
                                backward_graph["layers"][grad_node_id]["connections"]["inputs"].append(first_backward_op)
                            if grad_node_id not in backward_graph["layers"][first_backward_op]["connections"]["outputs"]:
                                backward_graph["layers"][first_backward_op]["connections"]["outputs"].append(grad_node_id)
            
            # Save backward graph
            output_path = Path(output_dir)
            ensure_directory(output_path)
            
            import yaml
            from solar.common.utils import NoAliasDumper
            
            yaml_path = output_path / "pytorch_graph.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    backward_graph,
                    f,
                    Dumper=NoAliasDumper,
                    sort_keys=False,
                    default_flow_style=False
                )
            
            if self.debug:
                print(f"Saved backward graph to {yaml_path}")
            
            return backward_graph
            
        except Exception as e:
            if self.debug:
                print(f"Error in grad_fn extraction: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_torch_functions_from_grad_fn(self, grad_fn: Any) -> List[Dict[str, Any]]:
        """Extract actual torch function calls from a grad_fn.
        
        This method inspects the grad_fn to find what torch operations
        are being performed in the backward pass.
        
        Args:
            grad_fn: The autograd function node
            
        Returns:
            List of torch function call information
        """
        torch_functions = []
        
        try:
            grad_fn_name = type(grad_fn).__name__
            
            # Extract information from grad_fn attributes
            # Many backward operations store forward operation info
            
            # Check for saved tensors (these often contain forward operation results)
            if hasattr(grad_fn, 'saved_tensors'):
                for i, tensor in enumerate(grad_fn.saved_tensors):
                    if isinstance(tensor, torch.Tensor):
                        torch_functions.append({
                            "type": "saved_tensor",
                            "index": i,
                            "shape": list(tensor.shape) if hasattr(tensor, 'shape') else None,
                            "dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else None,
                        })
            
            # Check for specific backward operation patterns
            # AddmmBackward typically uses torch.addmm in forward
            if 'Addmm' in grad_fn_name:
                torch_functions.append({
                    "type": "torch_function",
                    "name": "torch.addmm",
                    "description": "Matrix multiplication with bias addition"
                })
            
            # GELUBackward uses torch.gelu in forward
            elif 'Gelu' in grad_fn_name:
                torch_functions.append({
                    "type": "torch_function",
                    "name": "torch.gelu",
                    "description": "GELU activation"
                })
            
            # ViewBackward uses view/reshape in forward
            elif 'View' in grad_fn_name:
                torch_functions.append({
                    "type": "torch_function",
                    "name": "torch.view",
                    "description": "Tensor view/reshape"
                })
            
            # TBackward uses transpose in forward
            elif 'T' in grad_fn_name and 'Backward' in grad_fn_name:
                torch_functions.append({
                    "type": "torch_function",
                    "name": "torch.t",
                    "description": "Tensor transpose"
                })
            
            # MSELossBackward uses mse_loss in forward
            elif 'MseLoss' in grad_fn_name:
                torch_functions.append({
                    "type": "torch_function",
                    "name": "torch.nn.functional.mse_loss",
                    "description": "Mean squared error loss"
                })
            
            # AccumulateGrad just accumulates gradients
            elif 'AccumulateGrad' in grad_fn_name:
                torch_functions.append({
                    "type": "torch_function",
                    "name": "torch.autograd.accumulate_grad",
                    "description": "Gradient accumulation"
                })
            
            # Try to extract from grad_fn's __dict__ for more info
            if hasattr(grad_fn, '__dict__'):
                for key, value in grad_fn.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        torch_functions.append({
                            "type": "grad_fn_attribute",
                            "name": key,
                            "shape": list(value.shape) if hasattr(value, 'shape') else None,
                        })
                    elif isinstance(value, (int, float)):
                        torch_functions.append({
                            "type": "grad_fn_attribute",
                            "name": key,
                            "value": value,
                        })
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting torch functions from grad_fn: {e}")
        
        return torch_functions
    
    def _map_grad_fn_to_op_type(self, grad_fn_name: str) -> str:
        """Map autograd function name to operation type.
        
        Args:
            grad_fn_name: Name of the grad_fn class (e.g., 'AddBackward', 'MulBackward')
            
        Returns:
            Operation type string (e.g., 'add_backward', 'mul_backward')
        """
        # Remove 'Backward' suffix and convert to lowercase
        if grad_fn_name.endswith('Backward'):
            op_name = grad_fn_name[:-8].lower()
        else:
            op_name = grad_fn_name.lower()
        
        # Map common backward operations
        backward_op_map = {
            'add': 'add_backward',
            'mul': 'mul_backward',
            'mm': 'matmul_backward',
            'addmm': 'addmm_backward',
            'linear': 'linear_backward',
            'gelu': 'gelu_backward',
            'relu': 'relu_backward',
            'sum': 'sum_backward',
            'mean': 'mean_backward',
            'view': 'view_backward',
            'transpose': 'transpose_backward',
            'permute': 'permute_backward',
            'cat': 'cat_backward',
            'stack': 'stack_backward',
            'unsqueeze': 'unsqueeze_backward',
            'squeeze': 'squeeze_backward',
            'mseloss': 'mse_loss_backward',
            'accumulategrad': 'accumulate_grad_backward',
        }
        
        return backward_op_map.get(op_name, f"{op_name}_backward")
