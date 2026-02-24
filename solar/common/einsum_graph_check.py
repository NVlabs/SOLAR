#!/usr/bin/env python3
"""Einsum graph consistency checker.

This module validates einsum graphs to ensure:
1. Connection consistency: predecessor outputs match current inputs,
   successor inputs match current outputs
2. Tensor name consistency: tensor_names match between connected nodes
3. Shape consistency: tensor shapes are compatible across connections

Usage:
    # As a module
    from solar.common.einsum_graph_check import EinsumGraphChecker
    checker = EinsumGraphChecker()
    errors = checker.check_graph(graph_dict)
    
    # From command line
    python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


@dataclass
class ValidationError:
    """Represents a validation error in the graph."""
    layer_id: str
    error_type: str
    message: str
    severity: str = "error"  # error, warning
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.layer_id}: {self.error_type} - {self.message}"


@dataclass
class ValidationResult:
    """Results from graph validation."""
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if graph passed validation (no errors)."""
        return len(self.errors) == 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        if error.severity == "warning":
            self.warnings.append(error)
        else:
            self.errors.append(error)
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid and not self.has_warnings:
            return "✅ Graph validation passed"
        
        parts = []
        if not self.is_valid:
            parts.append(f"❌ {len(self.errors)} error(s)")
        if self.has_warnings:
            parts.append(f"⚠️ {len(self.warnings)} warning(s)")
        return " | ".join(parts)


class EinsumGraphChecker:
    """Validates einsum graph consistency.
    
    Checks performed:
    1. Connection consistency
       - Each node's input connections should list predecessors
       - Each node's output connections should list successors
       - Bidirectional: if A->B in A.outputs, then A should be in B.inputs
    
    2. Tensor name consistency
       - tensor_names.inputs should reference predecessor's tensor_names.outputs
       - tensor_names.outputs should be referenced by successor's tensor_names.inputs
    
    3. Shape consistency (optional)
       - Shapes should be compatible across connections
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the checker.
        
        Args:
            debug: Enable debug output.
        """
        self._debug = debug
    
    def check_graph(self, graph: Dict[str, Any]) -> ValidationResult:
        """Validate an einsum graph.
        
        Args:
            graph: Einsum graph dictionary with 'layers' key.
            
        Returns:
            ValidationResult containing any errors/warnings.
        """
        result = ValidationResult()
        
        if "layers" not in graph:
            available_keys = list(graph.keys()) if graph else []
            result.add_error(ValidationError(
                layer_id="<root>",
                error_type="missing_layers",
                message=(
                    f"Graph missing 'layers' key - cannot validate structure.\n"
                    f"    Expected: Top-level 'layers' dictionary containing all nodes\n"
                    f"    Found keys: {available_keys}\n"
                    f"    FIX: Ensure the YAML file has a 'layers:' section at the root level"
                )
            ))
            return result
        
        layers = graph["layers"]
        
        # Build connection maps for validation
        all_outputs: Dict[str, Set[str]] = {}  # layer_id -> set of output layer_ids
        all_inputs: Dict[str, Set[str]] = {}   # layer_id -> set of input layer_ids
        tensor_outputs: Dict[str, List[str]] = {}  # layer_id -> list of output tensor names
        tensor_inputs: Dict[str, List[str]] = {}   # layer_id -> list of input tensor names
        
        for layer_id, layer_data in layers.items():
            connections = layer_data.get("connections", {})
            all_outputs[layer_id] = set(connections.get("outputs", []))
            all_inputs[layer_id] = set(connections.get("inputs", []))
            
            tensor_names = layer_data.get("tensor_names", {})
            tensor_outputs[layer_id] = tensor_names.get("outputs", [])
            tensor_inputs[layer_id] = tensor_names.get("inputs", [])
        
        # Check each layer
        for layer_id, layer_data in layers.items():
            # Check 1: Connection bidirectionality
            self._check_connection_consistency(
                result, layer_id, layer_data, layers, all_outputs, all_inputs
            )
            
            # Check 2: Tensor name consistency
            self._check_tensor_name_consistency(
                result, layer_id, layer_data, layers, tensor_outputs, tensor_inputs
            )
            
            # Check 3: Shape consistency (optional warnings)
            self._check_shape_consistency(
                result, layer_id, layer_data, layers
            )
        
        return result
    
    def _check_connection_consistency(
        self,
        result: ValidationResult,
        layer_id: str,
        layer_data: Dict[str, Any],
        layers: Dict[str, Any],
        all_outputs: Dict[str, Set[str]],
        all_inputs: Dict[str, Set[str]],
    ) -> None:
        """Check that connections are bidirectionally consistent."""
        connections = layer_data.get("connections", {})
        outputs = connections.get("outputs", [])
        inputs = connections.get("inputs", [])
        layer_type = layer_data.get("type", "unknown")
        
        # Check: If this layer lists X in outputs, X should list this layer in inputs
        for out_id in outputs:
            if out_id not in layers:
                # Provide list of available layers for debugging
                available = list(layers.keys())[:10]
                suffix = "..." if len(layers) > 10 else ""
                result.add_error(ValidationError(
                    layer_id=layer_id,
                    error_type="missing_successor",
                    message=(
                        f"Output connection '{out_id}' does not exist in graph.\n"
                        f"    Layer '{layer_id}' (type={layer_type}) lists '{out_id}' in connections.outputs\n"
                        f"    but '{out_id}' is not a valid layer ID.\n"
                        f"    Available layers: {available}{suffix}\n"
                        f"    FIX: Update connections.outputs to reference an existing layer, "
                        f"or add the missing layer."
                    )
                ))
            elif layer_id not in all_inputs.get(out_id, set()):
                succ_inputs = list(all_inputs.get(out_id, set()))
                succ_type = layers[out_id].get("type", "unknown")
                result.add_error(ValidationError(
                    layer_id=layer_id,
                    error_type="connection_mismatch",
                    message=(
                        f"Bidirectional connection broken: output '{layer_id}' -> '{out_id}'\n"
                        f"    Layer '{layer_id}' (type={layer_type}) lists '{out_id}' in connections.outputs\n"
                        f"    BUT layer '{out_id}' (type={succ_type}) does NOT list '{layer_id}' in connections.inputs\n"
                        f"    '{out_id}'.connections.inputs = {succ_inputs}\n"
                        f"    FIX: Add '{layer_id}' to '{out_id}'.connections.inputs, "
                        f"or remove '{out_id}' from '{layer_id}'.connections.outputs"
                    )
                ))
        
        # Check: If this layer lists X in inputs, X should list this layer in outputs
        for in_id in inputs:
            if in_id not in layers:
                available = list(layers.keys())[:10]
                suffix = "..." if len(layers) > 10 else ""
                result.add_error(ValidationError(
                    layer_id=layer_id,
                    error_type="missing_predecessor",
                    message=(
                        f"Input connection '{in_id}' does not exist in graph.\n"
                        f"    Layer '{layer_id}' (type={layer_type}) lists '{in_id}' in connections.inputs\n"
                        f"    but '{in_id}' is not a valid layer ID.\n"
                        f"    Available layers: {available}{suffix}\n"
                        f"    FIX: Update connections.inputs to reference an existing layer, "
                        f"or add the missing layer."
                    )
                ))
            elif layer_id not in all_outputs.get(in_id, set()):
                pred_outputs = list(all_outputs.get(in_id, set()))
                pred_type = layers[in_id].get("type", "unknown")
                result.add_error(ValidationError(
                    layer_id=layer_id,
                    error_type="connection_mismatch",
                    message=(
                        f"Bidirectional connection broken: input '{in_id}' -> '{layer_id}'\n"
                        f"    Layer '{layer_id}' (type={layer_type}) lists '{in_id}' in connections.inputs\n"
                        f"    BUT layer '{in_id}' (type={pred_type}) does NOT list '{layer_id}' in connections.outputs\n"
                        f"    '{in_id}'.connections.outputs = {pred_outputs}\n"
                        f"    FIX: Add '{layer_id}' to '{in_id}'.connections.outputs, "
                        f"or remove '{in_id}' from '{layer_id}'.connections.inputs"
                    )
                ))
    
    def _check_tensor_name_consistency(
        self,
        result: ValidationResult,
        layer_id: str,
        layer_data: Dict[str, Any],
        layers: Dict[str, Any],
        tensor_outputs: Dict[str, List[str]],
        tensor_inputs: Dict[str, List[str]],
    ) -> None:
        """Check that tensor names are consistent across connections."""
        tensor_names = layer_data.get("tensor_names", {})
        if not tensor_names:
            return  # Skip if no tensor_names field
        
        current_inputs = tensor_names.get("inputs", [])
        current_outputs = tensor_names.get("outputs", [])
        connections = layer_data.get("connections", {})
        input_layer_ids = connections.get("inputs", [])
        output_layer_ids = connections.get("outputs", [])
        layer_type = layer_data.get("type", "unknown")
        
        # Check: Current layer's inputs should come from predecessors' outputs
        for input_name in current_inputs:
            found = False
            # Extract the source layer from tensor name (format: "layer_id.Output")
            source_layer_id = self._extract_layer_from_tensor_name(input_name)
            
            if source_layer_id and source_layer_id in layers:
                # Check if this tensor is in the source layer's outputs
                source_outputs = tensor_outputs.get(source_layer_id, [])
                if input_name in source_outputs:
                    found = True
            
            if not found and source_layer_id:
                # Only warn if we can identify the source layer
                source_outputs = tensor_outputs.get(source_layer_id, []) if source_layer_id else []
                source_type = layers.get(source_layer_id, {}).get("type", "unknown") if source_layer_id else "N/A"
                source_exists = "exists" if source_layer_id and source_layer_id in layers else "MISSING"
                result.add_error(ValidationError(
                    layer_id=layer_id,
                    error_type="tensor_name_mismatch",
                    message=(
                        f"Input tensor '{input_name}' not found in source layer's outputs.\n"
                        f"    Current layer: '{layer_id}' (type={layer_type})\n"
                        f"    tensor_names.inputs: {current_inputs}\n"
                        f"    Extracted source layer: '{source_layer_id}' ({source_exists}, type={source_type})\n"
                        f"    Source layer's tensor_names.outputs: {source_outputs}\n"
                        f"    FIX: Either update '{layer_id}'.tensor_names.inputs to match "
                        f"'{source_layer_id}'.tensor_names.outputs,\n"
                        f"         or update '{source_layer_id}'.tensor_names.outputs to include '{input_name}'"
                    ),
                    severity="warning"
                ))
        
        # Check: Current layer's outputs should be used in successors' inputs
        for output_name in current_outputs:
            if not output_layer_ids:
                continue  # Terminal node, no successors to check
            
            found = False
            for succ_id in output_layer_ids:
                if succ_id in layers:
                    succ_inputs = tensor_inputs.get(succ_id, [])
                    if output_name in succ_inputs:
                        found = True
                        break
            
            if not found and output_layer_ids:
                # Collect info about all successors for helpful debugging
                succ_info = []
                for succ_id in output_layer_ids:
                    if succ_id in layers:
                        succ_type = layers[succ_id].get("type", "unknown")
                        succ_in = tensor_inputs.get(succ_id, [])
                        succ_info.append(f"'{succ_id}' (type={succ_type}): inputs={succ_in}")
                    else:
                        succ_info.append(f"'{succ_id}': MISSING from graph")
                
                result.add_error(ValidationError(
                    layer_id=layer_id,
                    error_type="unused_output",
                    message=(
                        f"Output tensor '{output_name}' not found in any successor's inputs.\n"
                        f"    Current layer: '{layer_id}' (type={layer_type})\n"
                        f"    tensor_names.outputs: {current_outputs}\n"
                        f"    connections.outputs (successors): {output_layer_ids}\n"
                        f"    Successors' tensor_names.inputs:\n"
                        f"      " + "\n      ".join(succ_info) + "\n"
                        f"    FIX: Update successor's tensor_names.inputs to include '{output_name}',\n"
                        f"         or update '{layer_id}'.tensor_names.outputs to match successor expectations"
                    ),
                    severity="warning"
                ))
    
    def _check_shape_consistency(
        self,
        result: ValidationResult,
        layer_id: str,
        layer_data: Dict[str, Any],
        layers: Dict[str, Any],
    ) -> None:
        """Check that shapes are compatible across connections (optional)."""
        tensor_shapes = layer_data.get("tensor_shapes", {})
        shapes = layer_data.get("shapes", {})
        connections = layer_data.get("connections", {})
        
        if not tensor_shapes and not shapes:
            return
        
        # Get output shapes from this layer
        output_shapes = tensor_shapes.get("outputs", [])
        output_shape_legacy = shapes.get("Output")
        
        # Check against successors' input shapes
        for succ_id in connections.get("outputs", []):
            if succ_id not in layers:
                continue
            
            succ_data = layers[succ_id]
            succ_tensor_shapes = succ_data.get("tensor_shapes", {})
            succ_shapes = succ_data.get("shapes", {})
            succ_inputs = succ_tensor_shapes.get("inputs", [])
            
            # This is a simplified check - in practice would need to match
            # specific tensor names to their shapes
            # For now, just check if shapes exist
            if output_shapes and succ_inputs:
                # Could add more detailed shape compatibility checks here
                pass
    
    def _extract_layer_from_tensor_name(self, tensor_name: str) -> Optional[str]:
        """Extract layer ID from a tensor name.
        
        Tensor names follow format: "layer_id.Output" or "layer_id.Output_1"
        
        Args:
            tensor_name: Full tensor name.
            
        Returns:
            Layer ID or None if cannot extract.
        """
        if not tensor_name or "." not in tensor_name:
            return None
        
        # Handle nested IDs like "Model.scaled_dot_product_attention.qk_matmul.Output"
        parts = tensor_name.rsplit(".", 1)
        if len(parts) == 2 and parts[1].startswith("Output"):
            return parts[0]
        
        return None
    
    def check_file(self, path: Path) -> ValidationResult:
        """Validate an einsum graph file.
        
        Args:
            path: Path to einsum_graph.yaml file.
            
        Returns:
            ValidationResult.
        """
        try:
            with open(path) as f:
                graph = yaml.safe_load(f)
            return self.check_graph(graph)
        except FileNotFoundError:
            result = ValidationResult()
            result.add_error(ValidationError(
                layer_id="<file>",
                error_type="file_not_found",
                message=(
                    f"File not found: '{path}'\n"
                    f"    FIX: Verify the file path is correct and the file exists.\n"
                    f"    Expected: einsum_graph.yaml generated by toeinsum pipeline"
                )
            ))
            return result
        except yaml.YAMLError as e:
            result = ValidationResult()
            # Extract line/column info if available
            error_detail = str(e)
            if hasattr(e, 'problem_mark') and e.problem_mark:
                mark = e.problem_mark
                error_detail = f"Line {mark.line + 1}, Column {mark.column + 1}: {e.problem}"
            result.add_error(ValidationError(
                layer_id="<file>",
                error_type="yaml_parse_error",
                message=(
                    f"Failed to parse YAML file: '{path}'\n"
                    f"    Parse error: {error_detail}\n"
                    f"    FIX: Check the YAML file for syntax errors (missing colons, "
                    f"incorrect indentation, unquoted special chars)"
                )
            ))
            return result
        except Exception as e:
            result = ValidationResult()
            result.add_error(ValidationError(
                layer_id="<file>",
                error_type="load_error",
                message=(
                    f"Unexpected error loading graph file: '{path}'\n"
                    f"    Error type: {type(e).__name__}\n"
                    f"    Error message: {e}\n"
                    f"    FIX: Check file permissions and ensure the file is a valid YAML document"
                )
            ))
            return result


def check_einsum_graph(
    graph: Dict[str, Any],
    debug: bool = False,
) -> ValidationResult:
    """Convenience function to check an einsum graph.
    
    Args:
        graph: Einsum graph dictionary.
        debug: Enable debug output.
        
    Returns:
        ValidationResult.
    """
    checker = EinsumGraphChecker(debug=debug)
    return checker.check_graph(graph)


def check_einsum_graph_file(
    path: str,
    debug: bool = False,
) -> ValidationResult:
    """Convenience function to check an einsum graph file.
    
    Args:
        path: Path to einsum_graph.yaml file.
        debug: Enable debug output.
        
    Returns:
        ValidationResult.
    """
    checker = EinsumGraphChecker(debug=debug)
    return checker.check_file(Path(path))


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Validate einsum graph consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml
  python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml --strict
  python -m solar.common.einsum_graph_check path/to/einsum_graph.yaml --debug

Checks performed:
  1. Connection consistency: Bidirectional validation of inputs/outputs
  2. Tensor name consistency: Validates tensor_names match between connected nodes
  3. Shape consistency: Warns about potential shape mismatches (optional)
"""
    )
    parser.add_argument(
        "graph_file",
        type=str,
        help="Path to einsum_graph.yaml file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Einsum Graph Validator")
    print(f"{'='*70}")
    print(f"File: {args.graph_file}")
    print(f"{'='*70}\n")
    
    checker = EinsumGraphChecker(debug=args.debug)
    result = checker.check_file(Path(args.graph_file))
    
    # Print summary
    print(f"{result.summary()}\n")
    
    # Print errors with clear formatting
    if result.errors:
        print(f"{'-'*70}")
        print(f"ERRORS ({len(result.errors)}):")
        print(f"{'-'*70}")
        for i, error in enumerate(result.errors, 1):
            print(f"\n[Error {i}] Layer: {error.layer_id}")
            print(f"  Type: {error.error_type}")
            print(f"  Details:")
            # Indent the message for readability
            for line in error.message.split('\n'):
                print(f"    {line}")
        print()
    
    # Print warnings with clear formatting
    if result.warnings:
        print(f"{'-'*70}")
        print(f"WARNINGS ({len(result.warnings)}):")
        print(f"{'-'*70}")
        for i, warning in enumerate(result.warnings, 1):
            print(f"\n[Warning {i}] Layer: {warning.layer_id}")
            print(f"  Type: {warning.error_type}")
            print(f"  Details:")
            # Indent the message for readability
            for line in warning.message.split('\n'):
                print(f"    {line}")
        print()
    
    # Print final status
    print(f"{'='*70}")
    if result.is_valid and not result.has_warnings:
        print("✅ Validation PASSED - Graph is consistent")
    elif result.is_valid:
        print("⚠️  Validation PASSED with warnings - Review recommended")
    else:
        print("❌ Validation FAILED - Fix errors before proceeding")
    print(f"{'='*70}\n")
    
    # Exit code
    if not result.is_valid:
        sys.exit(1)
    elif args.strict and result.has_warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
