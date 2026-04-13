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

"""PyTorch model processor for extracting and analyzing PyTorch models.

This module processes PyTorch model files (including kernelbench) 
to extract computation graphs and model information.

Supports both 'Model' and 'ReferenceModel' class names, and uses
launch_reference_implementation to infer how to call the model.
"""

import ast
import gc
import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
import torchview
from torch import nn

from solar.common.types import ProcessingConfig
from solar.common.utils import (
    ensure_directory,
    load_module_from_file,
    setup_safe_environment,
)
from solar.graph.torchview_processor import TorchviewProcessor


class PyTorchProcessor:
    """Process PyTorch model into a saved torch graph."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the PyTorchProcessor.
        
        Args:
            config: Processing configuration. If None, uses defaults.
        """
        self.config = config or ProcessingConfig()
        self.torchview_processor = TorchviewProcessor(debug=self.config.debug)
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set up safe execution environment."""
        if self.config.safe_mode:
            setup_safe_environment()
    
    def process_model_file(self, file_path: str, output_dir: str) -> bool:
        """Process PyTorch model file into `output_dir`.

        Args:
            file_path: Path to a Python file containing `Model`/`ReferenceModel` and `get_inputs()`.
            output_dir: Output directory to write graph artifacts to.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.config.debug:
                print(f"Processing {file_path}...")

            output_path = Path(output_dir)
            ensure_directory(output_path)

            # Check if already processed
            if self._is_already_processed(output_path) and not self.config.force_rerun:
                print(f"Skipping {file_path} - already processed")
                return True

            # Copy source file for reproducibility (best effort).
            try:
                import shutil
                
                filename = Path(file_path).name
                # Avoid double-prefixing if already starts with "source_"
                if filename.startswith("source_"):
                    source_copy = output_path / filename
                else:
                    source_copy = output_path / f"source_{filename}"
                
                # Clean up any old source_* files to avoid accumulation
                for old_source in output_path.glob("source_*.py"):
                    old_source.unlink()
                
                # Always copy the latest source file
                shutil.copy2(file_path, source_copy)
            except Exception:
                pass

            # Load and process model
            model, inputs, module = self._load_model(file_path)
            if model is None:
                return False

            # Generate torchview graph (pass output_path for saving visualization)
            graph = self._generate_torchview_graph(model, inputs, str(output_path), module=module)
            if graph is None:
                return False
            
            # Process and save graph
            kernel_name = Path(file_path).stem
            self.torchview_processor.process_graph(
                graph, str(output_path), kernel_name, original_model=model
            )

            # Patch input tensor dtypes into pytorch_graph.yaml.
            # The torchview tracer loses non-float dtypes (e.g. torch.bool)
            # because the fallback infers dtype from model parameters.
            # We fix this by reading the actual dtypes from get_inputs().
            self._patch_input_dtypes(output_path, inputs)

            # Clean up
            self._cleanup(model, inputs, graph)
            
            print(f"✓ Successfully processed {file_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _is_already_processed(self, output_dir: Path) -> bool:
        """Check if a model has already been processed.
        
        Args:
            output_dir: Output directory to check.
            
        Returns:
            True if already processed, False otherwise.
        """
        # Canonical artifact.
        return (output_dir / "pytorch_graph.yaml").exists()
    
    def _get_model_class(self, module: Any) -> Tuple[Optional[type], str]:
        """Get the model class from a module.
        
        Supports both 'Model' and 'ReferenceModel' class names.
        
        Args:
            module: Loaded Python module.
            
        Returns:
            Tuple of (model_class, class_name) or (None, "") if not found.
        """
        # Try Model first (kernelbench convention)
        if hasattr(module, 'Model'):
            return module.Model, 'Model'
        
        # Try ReferenceModel (SolBench V1 convention)
        if hasattr(module, 'ReferenceModel'):
            return module.ReferenceModel, 'ReferenceModel'
        
        return None, ""
    
    def _infer_model_call_from_launch(self, module: Any, model: nn.Module, inputs: Any) -> Callable:
        """Infer how to call the model from launch_reference_implementation.
        
        Analyzes launch_reference_implementation to determine if the model
        should be called with *inputs (unpacked) or with inputs directly.
        
        Args:
            module: Loaded Python module.
            model: Model instance.
            inputs: Inputs from get_inputs().
            
        Returns:
            A callable that takes (model, inputs) and returns model output.
        """
        if not hasattr(module, 'launch_reference_implementation'):
            # Default: call model with unpacked inputs if tuple/list, else directly
            def default_call(m, i):
                if isinstance(i, (tuple, list)):
                    return m(*i)
                return m(i)
            return default_call
        
        # Try to analyze the source code of launch_reference_implementation
        try:
            source = inspect.getsource(module.launch_reference_implementation)
            
            # Parse the source to understand how model is called
            # Common patterns:
            # 1. return model(*inputs) - unpack inputs
            # 2. return model(inputs) - pass inputs directly (rare)
            # 3. return model(input1, input2, ...) - specific args
            
            if '*inputs' in source or '*args' in source:
                # Unpacking pattern
                def unpack_call(m, i):
                    if isinstance(i, (tuple, list)):
                        return m(*i)
                    return m(i)
                return unpack_call
            else:
                # Try to detect single argument pattern
                def single_call(m, i):
                    if isinstance(i, (tuple, list)):
                        return m(*i)
                    return m(i)
                return single_call
                
        except Exception:
            pass
        
        # Fallback: try to call launch_reference_implementation signature
        def fallback_call(m, i):
            if isinstance(i, (tuple, list)):
                return m(*i)
            return m(i)
        return fallback_call
    
    def _infer_init_args_from_module(self, module: Any, model_class: type) -> Tuple[tuple, dict]:
        """Infer model init args from module globals.
        
        Looks for global constants that match the model's __init__ signature.
        Common patterns:
        - HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS, etc.
        
        Args:
            module: Loaded Python module.
            model_class: The model class to instantiate.
            
        Returns:
            Tuple of (args, kwargs) for model instantiation.
        """
        try:
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.items())[1:]  # Skip 'self'
            
            args = []
            kwargs = {}
            
            # Common name mappings from parameter names to global constants
            name_mappings = {
                'hidden_size': ['HIDDEN_SIZE', 'hidden_size', 'D_MODEL', 'd_model'],
                'intermediate_size': ['INTERMEDIATE_SIZE', 'intermediate_size', 'FFN_DIM', 'ffn_dim'],
                'num_heads': ['NUM_HEADS', 'num_heads', 'N_HEADS', 'n_heads'],
                'num_attention_heads': ['NUM_ATTENTION_HEADS', 'num_attention_heads', 'NUM_HEADS'],
                'num_key_value_heads': ['NUM_KEY_VALUE_HEADS', 'num_key_value_heads', 'N_KV_HEADS'],
                'head_dim': ['HEAD_DIM', 'head_dim'],
                'num_layers': ['NUM_LAYERS', 'num_layers', 'N_LAYERS'],
                'vocab_size': ['VOCAB_SIZE', 'vocab_size'],
                'max_seq_len': ['MAX_SEQ_LEN', 'max_seq_len', 'SEQ_LEN'],
                'dropout': ['DROPOUT', 'dropout'],
                'eps': ['EPS', 'eps', 'RMS_NORM_EPS', 'LAYER_NORM_EPS'],
                'batch_size': ['BATCH_SIZE', 'batch_size'],
                'seq_len': ['SEQ_LEN', 'seq_len'],
            }
            
            for param_name, param in params:
                # Skip if has default
                if param.default != inspect.Parameter.empty:
                    continue
                
                # Try to find matching global
                found = False
                candidates = name_mappings.get(param_name, [param_name.upper(), param_name])
                
                for candidate in candidates:
                    if hasattr(module, candidate):
                        value = getattr(module, candidate)
                        if isinstance(value, (int, float, str, bool)):
                            args.append(value)
                            found = True
                            if self.config.debug:
                                print(f"    Inferred {param_name}={value} from {candidate}")
                            break
                
                if not found:
                    # Can't infer this required argument
                    if self.config.debug:
                        print(f"    Could not infer required arg: {param_name}")
                    return (), {}
            
            return tuple(args), kwargs
            
        except Exception as e:
            if self.config.debug:
                print(f"    Error inferring init args: {e}")
            return (), {}
    
    def _load_model(self,
                   file_path: str) -> Tuple[Optional[nn.Module], Optional[Any], Optional[Any]]:
        """Load a model from a Python file.

        Supports both 'Model' and 'ReferenceModel' class names.
        Uses launch_reference_implementation to infer how to call the model.
        Falls back to inferring init args from module globals.
        
        Args:
            file_path: Path to the model file.
            
        Returns:
            Tuple of (model, inputs) or (None, None) if failed.
        """
        try:
            module = load_module_from_file(file_path)
            
            # Get model class (Model or ReferenceModel)
            model_class, class_name = self._get_model_class(module)
            if model_class is None:
                print(f"Warning: No Model or ReferenceModel class found in {file_path}")
                return None, None, None
            
            if self.config.debug:
                print(f"  Found model class: {class_name}")
            
            # Check for get_inputs
            if not hasattr(module, 'get_inputs'):
                print(f"Warning: No get_inputs function found in {file_path}")
                return None, None, None
            
            # Create model instance - try multiple strategies
            model = None
            
            # Strategy 1: Use get_init_inputs() if available
            if hasattr(module, 'get_init_inputs'):
                try:
                    init_inputs = module.get_init_inputs()
                    model = model_class(*init_inputs) if init_inputs else model_class()
                    if self.config.debug:
                        print(f"  Created model using get_init_inputs()")
                except Exception as e:
                    if self.config.debug:
                        print(f"  get_init_inputs() failed: {e}")
            
            # Strategy 2: Try no-arg constructor
            if model is None:
                try:
                    model = model_class()
                    if self.config.debug:
                        print(f"  Created model using no-arg constructor")
                except TypeError:
                    pass  # Needs arguments
            
            # Strategy 3: Infer init args from module globals
            if model is None:
                args, kwargs = self._infer_init_args_from_module(module, model_class)
                if args or kwargs:
                    try:
                        model = model_class(*args, **kwargs)
                        if self.config.debug:
                            print(f"  Created model using inferred args: {args}")
                    except Exception as e:
                        if self.config.debug:
                            print(f"  Inferred args failed: {e}")
            
            if model is None:
                print(f"Error: Could not instantiate model class {class_name} in {file_path}")
                return None, None, None
            
            # Get inputs.  Try to allocate on meta device first to avoid
            # OOM on large workloads.  Fall back to CPU if meta fails
            # (some get_inputs use ops incompatible with meta device).
            inputs = None

            # Strategy 1: SolBench v3 model with custom _ref_get_inputs
            if inputs is None and hasattr(module, '_ref_get_inputs'):
                try:
                    import re as _re
                    src = Path(file_path).read_text()
                    axes_match = _re.search(r"_axes\s*=\s*(\{[^}]+\})", src)
                    if axes_match:
                        _axes = eval(axes_match.group(1))  # noqa: S307
                        inp_dict = module._ref_get_inputs(_axes, torch.device("meta"))
                        order_match = _re.search(r"_param_order\s*=\s*(\[[^\]]+\])", src)
                        if order_match:
                            _param_order = eval(order_match.group(1))  # noqa: S307
                            inputs = [inp_dict[k] for k in _param_order if k in inp_dict]
                        if self.config.debug:
                            print("  Allocated inputs on meta device (via _ref_get_inputs)")
                except Exception:
                    inputs = None

            # Strategy 2: Monkey-patch torch.randn/zeros/ones/empty to allocate
            # on meta device, then call get_inputs() normally.
            if inputs is None:
                try:
                    _orig_randn = torch.randn
                    _orig_zeros = torch.zeros
                    _orig_ones = torch.ones
                    _orig_empty = torch.empty

                    def _meta_factory(orig_fn):
                        def wrapper(*args, **kwargs):
                            kwargs.pop("device", None)
                            return orig_fn(*args, device="meta", **kwargs)
                        return wrapper

                    torch.randn = _meta_factory(_orig_randn)
                    torch.zeros = _meta_factory(_orig_zeros)
                    torch.ones = _meta_factory(_orig_ones)
                    torch.empty = _meta_factory(_orig_empty)
                    try:
                        inputs = module.get_inputs()
                        if self.config.debug:
                            print("  Allocated inputs on meta device (via patched factories)")
                    finally:
                        torch.randn = _orig_randn
                        torch.zeros = _orig_zeros
                        torch.ones = _orig_ones
                        torch.empty = _orig_empty
                except Exception:
                    inputs = None

            # Strategy 3: Fall back to normal CPU allocation
            if inputs is None:
                inputs = module.get_inputs()

            # Store the inferred call pattern for later use
            self._model_caller = self._infer_model_call_from_launch(module, model, inputs)

            return model, inputs, module

        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return None, None, None
    
    def _generate_torchview_graph(
        self,
        model: nn.Module,
        inputs: Any,
        output_dir: Optional[str] = None,
        module: Any = None,
    ) -> Optional[Any]:
        """Generate a torchview computation graph.

        Args:
            model: PyTorch model.
            inputs: Model inputs (may be on meta device).
            output_dir: Directory to save graph visualization (if save_graph enabled).
            module: Original loaded module (for re-generating CPU inputs on fallback).

        Returns:
            Computation graph or None if failed.
        """
        devices_to_try = ["meta", "cpu"]

        for device in devices_to_try:
            try:
                # Prepare model
                model = model.to_empty(device=device)
                model.eval()

                # Check for RNN-like models that need CPU
                if device == "meta" and self._is_rnn_model(model):
                    continue

                # For CPU fallback: if inputs are on meta device, we must
                # re-generate real CPU inputs (meta tensors have no data,
                # causing garbage values for .item(), indexing, etc.).
                if device == "cpu" and self._inputs_on_meta(inputs) and module is not None:
                    if self.config.debug:
                        print("  Re-generating CPU inputs via get_inputs()")
                    device_inputs = module.get_inputs()
                else:
                    device_inputs = self._move_inputs_to_device(inputs, device)

                # Generate graph (don't let torchview save - we'll do it ourselves)
                graph = torchview.draw_graph(
                    model,
                    input_data=device_inputs,
                    device=device,
                    save_graph=False,  # We handle saving separately
                    expand_nested=True,
                    depth=float('inf'),
                    hide_module_functions=False,
                    hide_inner_tensors=False,
                    roll=False,
                    strict=False,
                    collect_attributes=True,  # Capture function/module args
                )

                if self.config.debug:
                    print(f"✅ Generated graph using {device} device")

                # Save graph visualization if requested
                if self.config.save_graph and output_dir:
                    self._save_torchview_graph(graph, output_dir)

                return graph

            except (NotImplementedError, RuntimeError) as e:
                if device == "meta":
                    if self.config.debug:
                        print(f"⚠️ Meta device failed: {e}")
                    continue
                else:
                    raise
            except Exception as e:
                if device == "meta":
                    continue
                raise

        return None

    @staticmethod
    def _inputs_on_meta(inputs: Any) -> bool:
        """Check if any input tensor is on meta device."""
        if isinstance(inputs, torch.Tensor):
            return inputs.device.type == "meta"
        if isinstance(inputs, (list, tuple)):
            return any(
                isinstance(x, torch.Tensor) and x.device.type == "meta"
                for x in inputs
            )
        if isinstance(inputs, dict):
            return any(
                isinstance(v, torch.Tensor) and v.device.type == "meta"
                for v in inputs.values()
            )
        return False

    @staticmethod
    def _move_inputs_to_device(inputs: Any, device: str) -> Any:
        """Move input tensors to the given device.

        For ``meta`` device this replaces real tensors with zero-memory
        meta tensors that preserve shape and dtype, avoiding large CPU
        allocations when only graph structure is needed.
        """
        import torch

        def _move(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                if device == "meta":
                    return torch.empty(obj.shape, dtype=obj.dtype, device="meta")
                # Handle meta->cpu: can't .to() a meta tensor, recreate instead
                if obj.device.type == "meta" and device != "meta":
                    return torch.empty(obj.shape, dtype=obj.dtype, device=device)
                return obj.to(device)
            if isinstance(obj, (list, tuple)):
                moved = [_move(x) for x in obj]
                return type(obj)(moved)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            return obj

        return _move(inputs)

    def _save_torchview_graph(self, graph: Any, output_dir: str) -> None:
        """Save torchview graph visualization to the output directory.
        
        Args:
            graph: Torchview graph object.
            output_dir: Directory to save the graph visualization.
        """
        try:
            output_path = Path(output_dir)
            graph_filename = output_path / "torchview_graph"
            
            # torchview's visual_graph is a graphviz.Digraph object
            if hasattr(graph, 'visual_graph'):
                # Render to PDF (default) and PNG
                graph.visual_graph.render(
                    filename=str(graph_filename),
                    format='pdf',
                    cleanup=True,  # Remove the intermediate .gv file
                )
                if self.config.debug:
                    print(f"📊 Saved torchview graph: {graph_filename}.pdf")
            else:
                if self.config.debug:
                    print("⚠️ Graph object does not have visual_graph attribute")
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ Failed to save torchview graph: {e}")
    
    def _is_rnn_model(self, model: nn.Module) -> bool:
        """Check if a model is RNN-like.
        
        Args:
            model: PyTorch model.
            
        Returns:
            True if RNN-like, False otherwise.
        """
        # Check for RNN attributes
        if hasattr(model, 'hidden'):
            return True
        
        # Check module names
        for name, _ in model.named_modules():
            if any(rnn_type in name.lower() for rnn_type in ['rnn', 'gru', 'lstm']):
                return True
        
        return False
    
    def _patch_input_dtypes(self, output_path: Path, inputs: Any) -> None:
        """Patch actual input tensor dtypes into pytorch_graph.yaml.

        The torchview tracer falls back to model-parameter dtypes for tensor
        nodes, losing non-float dtypes (e.g. ``torch.bool``).  This method
        reads the actual dtypes from the ``get_inputs()`` return value and
        overwrites the corresponding ``auxiliary-tensor`` / ``input-tensor``
        nodes in the saved graph.
        """
        import yaml
        graph_path = output_path / "pytorch_graph.yaml"
        if not graph_path.exists():
            return

        # Collect actual dtypes from inputs
        actual_dtypes: list[str] = []
        if isinstance(inputs, (list, tuple)):
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    actual_dtypes.append(str(inp.dtype))
        elif isinstance(inputs, dict):
            for inp in inputs.values():
                if isinstance(inp, torch.Tensor):
                    actual_dtypes.append(str(inp.dtype))

        if not actual_dtypes:
            return

        try:
            with open(graph_path) as f:
                graph = yaml.safe_load(f) or {}
        except Exception:
            return

        layers = graph.get("layers", {})
        # Find auxiliary-tensor / input-tensor nodes (model inputs) in order
        input_nodes = [
            (nid, node) for nid, node in layers.items()
            if str(node.get("type", "")).lower() in ("auxiliary-tensor", "input-tensor")
        ]

        changed = False
        for idx, (nid, node) in enumerate(input_nodes):
            if idx >= len(actual_dtypes):
                break
            actual = actual_dtypes[idx]
            existing = node.get("output_dtypes", [])
            if existing != [actual]:
                node["output_dtypes"] = [actual]
                node["input_dtypes"] = [actual]
                changed = True

        if changed:
            from solar.common.utils import NoAliasDumper
            with open(graph_path, "w") as f:
                yaml.dump(graph, f, Dumper=NoAliasDumper, sort_keys=False,
                          default_flow_style=False)

    def _cleanup(self, *objects: Any) -> None:
        """Clean up objects and run garbage collection.

        Args:
            *objects: Objects to delete.
        """
        for obj in objects:
            del obj
        gc.collect()
