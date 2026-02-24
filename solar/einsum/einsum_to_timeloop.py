"""Convert einsum graphs to Timeloop workload format.

This module converts Solar's einsum graph representation into Timeloop's
workload YAML format, which can be used for architectural exploration.

The default input is `einsum_graph_renamed.yaml` which has consistent
dimension labels propagated via BFS traversal from start nodes.

The output format follows Timeloop's workload specification:
    - version: "0.5"
    - shape: dimension bounds
    - einsums: list of einsum operations with tensor accesses
    - renames: tensor renaming rules for constraints

Example:
    >>> from solar.einsum.einsum_to_timeloop import EinsumToTimeloop
    >>> converter = EinsumToTimeloop()
    >>> result = converter.convert("input/einsum_graph_renamed.yaml", "output/")
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
import yaml

from solar.common.utils import (
    ensure_directory,
    load_einsum_graph_to_networkx,
    NoAliasDumper,
    parse_dim_tokens,
)


PathLike = Union[str, Path]


@dataclass
class TensorAccess:
    """Represents a tensor access in Timeloop format.
    
    Attributes:
        name: Tensor name.
        projection: List of dimension names accessed.
        is_output: Whether this is an output tensor.
    """
    name: str
    projection: List[str]
    is_output: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Timeloop tensor_access dictionary."""
        result: Dict[str, Any] = {
            'name': self.name,
            'projection': self.projection.copy(),
        }
        if self.is_output:
            result['output'] = True
        return result


@dataclass
class TimeloopEinsum:
    """Represents an einsum operation in Timeloop format.
    
    Attributes:
        name: Operation name.
        tensor_accesses: List of tensor accesses.
        is_copy_operation: Whether this is a copy operation.
        renames: Optional rename rules.
    """
    name: str
    tensor_accesses: List[TensorAccess]
    is_copy_operation: bool = False
    renames: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Timeloop einsum dictionary."""
        result: Dict[str, Any] = {'name': self.name}
        
        if self.is_copy_operation:
            result['is_copy_operation'] = True
        
        result['tensor_accesses'] = [ta.to_dict() for ta in self.tensor_accesses]
        
        if self.renames:
            result['renames'] = self.renames
        
        return result


@dataclass
class TimeloopWorkload:
    """Represents a complete Timeloop workload specification.
    
    Attributes:
        version: Timeloop format version.
        shape: Dimension bounds mapping.
        einsums: List of einsum operations.
        default_renames: Optional default rename rules.
    """
    version: str = "0.5"
    shape: Dict[str, str] = field(default_factory=dict)
    einsums: List[TimeloopEinsum] = field(default_factory=list)
    default_renames: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Timeloop workload dictionary."""
        result: Dict[str, Any] = {
            'workload': {
                'version': self.version,
                'shape': self.shape.copy(),
                'einsums': [e.to_dict() for e in self.einsums],
            }
        }
        
        if self.default_renames:
            result['renames'] = self.default_renames
        
        return result


# Operations that never have weight parameters (activation-to-activation)
_NO_WEIGHT_OPS = frozenset({
    'matmul', 'mm', 'bmm',
    'add', 'sub', 'mul', 'div',
    'cat', 'concat', 'stack',
})

# Operations that typically have weight parameters
_WEIGHT_OPS = frozenset({
    'linear', 'conv1d', 'conv2d', 'conv3d',
    'convtranspose1d', 'convtranspose2d', 'convtranspose3d',
    'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
    'embedding',
})


class EinsumToTimeloop:
    """Convert einsum graphs to Timeloop workload format.
    
    Uses NetworkX to load and traverse the einsum graph, matching
    tensor names to the graph's connection structure.
    
    Attributes:
        debug: Whether to print debug information.
    """
    
    def __init__(self, debug: bool = False) -> None:
        """Initialize the converter.
        
        Args:
            debug: Enable debug output.
        """
        self._debug = debug
        self._dim_counter = 0
        self._dim_cache: Dict[int, str] = {}
        self._global_dims: Dict[str, int] = {}
        self._tensor_dims: Dict[str, List[str]] = {}
        self._graph: Optional[nx.DiGraph] = None
        self._layers: Dict[str, Any] = {}

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    def convert(
        self,
        einsum_graph_path: PathLike,
        output_path: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """Convert einsum graph to Timeloop workload format.
        
        Args:
            einsum_graph_path: Path to einsum_graph.yaml.
            output_path: Optional output path for Timeloop YAML.
            
        Returns:
            Timeloop workload dictionary.
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
        """
        graph_path = Path(einsum_graph_path)
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Einsum graph not found: {graph_path}")
        
        with open(graph_path) as f:
            einsum_graph = yaml.safe_load(f)
        
        # Reset state
        self._reset_state()
        
        # Build NetworkX graph
        self._layers = einsum_graph.get('layers', {})
        self._graph = load_einsum_graph_to_networkx(self._layers)
        
        # Convert to Timeloop format
        workload = self._convert_graph(einsum_graph)
        result = workload.to_dict()
        
        # Write output if path provided
        if output_path:
            out_path = Path(output_path)
            ensure_directory(out_path.parent)
            with open(out_path, 'w') as f:
                yaml.dump(
                    result, f,
                    Dumper=NoAliasDumper,
                    default_flow_style=False,
                    sort_keys=False
                )
            if self._debug:
                print(f"✅ Wrote Timeloop workload: {out_path}")
        
        return result
    
    def _reset_state(self) -> None:
        """Reset internal state for a new conversion."""
        self._dim_counter = 0
        self._dim_cache.clear()
        self._global_dims.clear()
        self._tensor_dims.clear()
        self._graph = None
        self._layers = {}
    
    def _convert_graph(self, einsum_graph: Dict[str, Any]) -> TimeloopWorkload:
        """Convert einsum graph to Timeloop workload."""
        layers = einsum_graph.get('layers', {})
        
        # First pass: collect dimensions and bounds
        self._collect_dimensions(layers)
        
        # Second pass: create Timeloop einsums (topological order)
        timeloop_einsums = []
        try:
            layer_order = list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            layer_order = list(layers.keys())
        
        for layer_id in layer_order:
            if layer_id in layers:
                layer_data = layers[layer_id]
                einsum = self._convert_layer(layer_id, layer_data)
                if einsum:
                    timeloop_einsums.append(einsum)
        
        # Build shape specification
        shape = {
            dim_name: f"0 <= {dim_name} < {bound}"
            for dim_name, bound in sorted(self._global_dims.items())
        }
        
        # Build default renames
        default_renames = self._build_default_renames()
        
        return TimeloopWorkload(
            version="0.5",
            shape=shape,
            einsums=timeloop_einsums,
            default_renames=default_renames,
        )
    
    def _collect_dimensions(self, layers: Dict[str, Any]) -> None:
        """Collect all dimensions and their bounds from the graph."""
        for layer_id, layer_data in layers.items():
            shapes = layer_data.get('shapes', {})
            equation = layer_data.get('einsum_equation', '')
            
            if not equation:
                continue
            
            dim_info = self._parse_equation_dimensions(equation, shapes)
            
            for dim_name, bound in dim_info.items():
                if dim_name in self._global_dims:
                    self._global_dims[dim_name] = max(
                        self._global_dims[dim_name], bound
                    )
                else:
                    self._global_dims[dim_name] = bound
    
    def _parse_equation_dimensions(
        self,
        equation: str,
        shapes: Dict[str, List[int]],
    ) -> Dict[str, int]:
        """Parse einsum equation and extract dimension bounds."""
        dim_info: Dict[str, int] = {}
        
        if not equation or '->' not in equation:
            return dim_info
        
        parts = equation.split('->')
        if len(parts) != 2:
            return dim_info
        
        input_part, _ = parts
        input_operands = input_part.split(',')
        
        # Get shape lists in order (inputs first, matching einsum equation order)
        shape_list = []
        inputs = shapes.get("inputs", [])
        for i, shape in enumerate(inputs):
            shape_list.append((f"input_{i}", shape))
        
        # Map dimensions to bounds
        shape_idx = 0
        for operand_str in input_operands:
            operand_str = operand_str.strip()
            dim_tokens = parse_dim_tokens(operand_str)
            
            if shape_idx < len(shape_list):
                _, shape = shape_list[shape_idx]
                for i, dim_token in enumerate(dim_tokens):
                    if i < len(shape):
                        dim_name = dim_token.lower()
                        if dim_name not in dim_info:
                            dim_info[dim_name] = shape[i]
                        else:
                            dim_info[dim_name] = max(dim_info[dim_name], shape[i])
                shape_idx += 1
        
        return dim_info
    
    def _convert_layer(
        self,
        layer_id: str,
        layer_data: Dict[str, Any],
    ) -> Optional[TimeloopEinsum]:
        """Convert a single layer to Timeloop einsum format."""
        equation = layer_data.get('einsum_equation', '')
        shapes = layer_data.get('shapes', {})
        layer_type = layer_data.get('type', 'unknown')
        elementwise_op = layer_data.get('elementwise_op', 'mul')
        reduction_op = layer_data.get('reduction_op', 'add')
        connections = layer_data.get('connections', {})
        
        if not equation:
            if self._debug:
                print(f"  Skipping {layer_id}: no einsum equation")
            return None
        
        input_nodes = connections.get('inputs', [])
        
        # Parse equation with graph-based tensor naming
        tensor_accesses = self._parse_tensor_accesses(
            equation, shapes, layer_id, input_nodes, layer_type
        )
        
        if not tensor_accesses:
            return None
        
        # Determine if copy operation
        is_copy = (
            (elementwise_op == 'copy' and reduction_op == 'none') or
            layer_type in {'copy', 'clone', 'contiguous'}
        )
        
        # Build renames
        renames = self._build_layer_renames(tensor_accesses, layer_type)
        
        return TimeloopEinsum(
            name=self._sanitize_name(layer_id),
            tensor_accesses=tensor_accesses,
            is_copy_operation=is_copy,
            renames=renames if renames else None,
        )
    
    def _parse_tensor_accesses(
        self,
        equation: str,
        shapes: Dict[str, List[int]],
        layer_id: str,
        input_nodes: List[str],
        layer_type: str,
    ) -> List[TensorAccess]:
        """Parse einsum equation into tensor accesses."""
        tensor_accesses: List[TensorAccess] = []
        
        if '->' not in equation:
            return tensor_accesses
        
        parts = equation.split('->')
        if len(parts) != 2:
            return tensor_accesses
        
        input_part, output_part = parts
        input_operands = [op.strip() for op in input_part.split(',')]
        output_dims_str = output_part.strip()
        
        output_name = self._sanitize_name(layer_id)
        
        # Determine if operation has weight
        has_weight = self._has_weight_operand(layer_type, shapes, len(input_nodes))
        
        # Generate tensor names
        tensor_names = self._generate_tensor_names(
            layer_id, input_nodes, len(input_operands), has_weight
        )
        
        # Create input tensor accesses
        for i, operand_str in enumerate(input_operands):
            dim_tokens = parse_dim_tokens(operand_str)
            dims = [d.lower() for d in dim_tokens]
            tensor_name = tensor_names[i] if i < len(tensor_names) else f"T{i}"
            tensor_accesses.append(TensorAccess(
                name=tensor_name,
                projection=dims,
                is_output=False,
            ))
        
        # Create output tensor access
        output_dim_tokens = parse_dim_tokens(output_dims_str)
        output_dims = [d.lower() for d in output_dim_tokens]
        tensor_accesses.append(TensorAccess(
            name=output_name,
            projection=output_dims,
            is_output=True,
        ))
        
        return tensor_accesses
    
    def _has_weight_operand(
        self,
        layer_type: str,
        shapes: Dict[str, List[int]],
        num_input_nodes: int,
    ) -> bool:
        """Determine if an operation has a weight operand."""
        layer_type_lower = layer_type.lower()
        
        if layer_type_lower in _NO_WEIGHT_OPS:
            return False
        
        if layer_type_lower in _WEIGHT_OPS:
            return True
        
        return 'Weight' in shapes and num_input_nodes < 2
    
    def _generate_tensor_names(
        self,
        layer_id: str,
        input_nodes: List[str],
        num_operands: int,
        has_weight: bool,
    ) -> List[str]:
        """Generate tensor names based on graph connections."""
        output_name = self._sanitize_name(layer_id)
        weight_name = f"{output_name}_w"
        
        if num_operands == 1:
            if input_nodes:
                return [self._sanitize_name(input_nodes[0])]
            return [f"{output_name}_in"]
        
        elif num_operands == 2:
            if has_weight:
                if input_nodes:
                    return [self._sanitize_name(input_nodes[0]), weight_name]
                return [f"{output_name}_in", weight_name]
            else:
                names = []
                for i in range(2):
                    if i < len(input_nodes):
                        names.append(self._sanitize_name(input_nodes[i]))
                    else:
                        names.append(f"{output_name}_in{i}")
                return names
        
        else:
            names = []
            weight_added = False
            for i in range(num_operands):
                if i < len(input_nodes):
                    names.append(self._sanitize_name(input_nodes[i]))
                elif has_weight and not weight_added:
                    names.append(weight_name)
                    weight_added = True
                else:
                    names.append(f"{output_name}_in{i}")
            return names
    
    def _build_layer_renames(
        self,
        tensor_accesses: List[TensorAccess],
        layer_type: str,
    ) -> Optional[Dict[str, str]]:
        """Build rename rules for a layer."""
        if len(tensor_accesses) < 2:
            return None
        
        inputs = [ta for ta in tensor_accesses if not ta.is_output]
        outputs = [ta for ta in tensor_accesses if ta.is_output]
        
        if not inputs or not outputs:
            return None
        
        renames: Dict[str, str] = {'output': outputs[0].name}
        
        if len(inputs) == 1:
            renames['input'] = inputs[0].name
            renames['weight'] = 'Nothing()'
        elif len(inputs) == 2:
            if layer_type in {'matmul', 'linear', 'mm', 'bmm', 
                             'conv1d', 'conv2d', 'conv3d'}:
                renames['input'] = inputs[0].name
                renames['weight'] = inputs[1].name
            else:
                renames['input'] = inputs[0].name
                renames['weight'] = inputs[1].name
        else:
            renames['input'] = inputs[0].name
            renames['weight'] = inputs[1].name if len(inputs) > 1 else 'Nothing()'
        
        return renames
    
    def _build_default_renames(self) -> Dict[str, Any]:
        """Build default rename rules for the workload."""
        return {
            'einsums': [
                {
                    'name': 'default',
                    'tensor_accesses': [
                        {
                            'name': 'input',
                            'source': 'Inputs() & Intermediates()',
                            'expected_count': 1,
                        },
                        {
                            'name': 'output',
                            'source': 'Outputs()',
                            'expected_count': 1,
                        },
                        {
                            'name': 'weight',
                            'source': '~(input | output)',
                            'expected_count': 1,
                        },
                    ],
                },
            ],
        }
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in Timeloop."""
        sanitized = name.replace('.', '_').replace('-', '_').replace('/', '_')
        sanitized = ''.join(
            c if c.isalnum() or c == '_' else '_'
            for c in sanitized
        )
        if sanitized and sanitized[0].isdigit():
            sanitized = 'L' + sanitized
        return sanitized


# Backward compatibility alias
TimeloopFormatter = EinsumToTimeloop


def convert_to_timeloop(
    einsum_graph_path: PathLike,
    output_path: Optional[PathLike] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Convenience function to convert einsum graph to Timeloop format.
    
    Args:
        einsum_graph_path: Path to einsum_graph.yaml.
        output_path: Optional output path for Timeloop YAML.
        debug: Enable debug output.
        
    Returns:
        Timeloop workload dictionary.
    """
    converter = EinsumToTimeloop(debug=debug)
    return converter.convert(einsum_graph_path, output_path)


def main() -> None:
    """CLI entry point for einsum to Timeloop conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert einsum_graph.yaml to Timeloop workload format",
    )
    parser.add_argument(
        "einsum_graph",
        help="Path to einsum_graph.yaml",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for Timeloop YAML (default: timeloop_graph.yaml)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.einsum_graph)
    output_path = Path(args.output) if args.output else input_path.parent / "timeloop_graph.yaml"
    
    result = convert_to_timeloop(input_path, output_path, debug=args.debug)
    
    workload = result.get('workload', {})
    num_dims = len(workload.get('shape', {}))
    num_einsums = len(workload.get('einsums', []))
    
    print(f"✅ Converted to Timeloop format:")
    print(f"   Dimensions: {num_dims}")
    print(f"   Einsums: {num_einsums}")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "EinsumToTimeloop",
    "TimeloopFormatter",  # Backward compatibility
    "TensorAccess",
    "TimeloopEinsum",
    "TimeloopWorkload",
    "convert_to_timeloop",
]

