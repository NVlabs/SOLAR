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

"""CLI command to extract backward computation graphs.

This module provides a command-line interface for extracting backward
computation graphs from PyTorch models and running the Solar backward analysis pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# Add solar to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solar.graph.backward_processor import BackwardProcessor
from solar.graph.pytorch_processor import PyTorchProcessor


def load_model_and_functions(model_file: str):
    """Load model and helper functions from a Python file.
    
    Args:
        model_file: Path to Python file containing Model class and helper functions.
        
    Returns:
        Tuple of (model, get_inputs, get_loss_fn, get_target) functions.
    """
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {model_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get Model class
    if not hasattr(module, "Model"):
        raise ValueError(f"Module {model_file} must define a 'Model' class")
    Model = module.Model
    
    # Get helper functions
    get_inputs = getattr(module, "get_inputs", None)
    if get_inputs is None:
        raise ValueError(f"Module {model_file} must define 'get_inputs()' function")
    
    get_loss_fn = getattr(module, "get_loss_fn", None)
    if get_loss_fn is None:
        raise ValueError(f"Module {model_file} must define 'get_loss_fn()' function")
    
    get_target = getattr(module, "get_target", None)
    if get_target is None:
        raise ValueError(f"Module {model_file} must define 'get_target()' function")
    
    # Instantiate model
    model = Model()
    
    return model, get_inputs, get_loss_fn, get_target


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract backward computation graph from PyTorch model"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Path to Python file containing Model class and helper functions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for backward graph (will create <output_dir>_backward)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (default: inferred from model file)"
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save graph visualization (PDF)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if output exists"
    )
    
    args = parser.parse_args()
    
    # Determine model name
    model_name = args.model_name
    if model_name is None:
        model_name = Path(args.model_file).stem
    
    # Determine backward output directory
    output_dir = Path(args.output_dir)
    backward_output_dir = output_dir.parent / f"{output_dir.name}_backward"
    
    # Check if already exists
    if not args.force_rerun and (backward_output_dir / "pytorch_graph.yaml").exists():
        print(f"Backward graph already exists at {backward_output_dir}/pytorch_graph.yaml")
        print("Use --force-rerun to regenerate")
        return 0
    
    # Create output directory
    backward_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_file}...")
    try:
        model, get_inputs, get_loss_fn, get_target = load_model_and_functions(args.model_file)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1
    
    print("Getting inputs and target...")
    inputs = get_inputs()
    loss_fn = get_loss_fn()
    target = get_target(inputs)
    
    print("Extracting backward graph...")
    processor = BackwardProcessor(debug=args.debug)
    
    backward_graph = processor.extract_backward_graph(
        model=model,
        inputs=inputs,
        loss_fn=loss_fn,
        target=target,
        output_dir=str(backward_output_dir / "graph"),
        model_name=model_name,
    )
    
    if backward_graph is None:
        print("Failed to extract backward graph", file=sys.stderr)
        return 1
    
    print(f"✅ Backward graph extracted to {backward_output_dir}/graph/pytorch_graph.yaml")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
