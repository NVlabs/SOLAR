"""Analyze an einsum graph into hardware-independent metrics.

This module implements the **second stage** of the Solar pipeline:

  `einsum_graph.yaml`  ->  `analysis.yaml`

The output `analysis.yaml` is intended to be hardware-independent and includes:
- per-layer: macs, flops (= 2 * macs), orojenesis_elements, fused_elements
- totals across the graph

Memory Access Models (in elements, multiply by bytes_per_element for bytes):
- orojenesis_elements (unfused): All tensor accesses (inputs + outputs) per op
- fused_elements: Model I/O only (intermediate tensors excluded per op)
- fused_prefetched_elements: Total model I/O across entire graph (single roofline)

Note: input_elements includes all inputs to an operation (including weights/biases).
Weights are treated as inputs since they are just another operand to the computation.

Note: "start" nodes are filtered out before analysis as they represent model inputs,
not actual computation. Their outputs are treated as external inputs to the graph.

See SOL_GUIDE.md for detailed explanation of the three SOL models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from solar.einsum import EinsumAnalyzer
from solar.common.constants import BYTES_PER_ELEMENT, DEFAULT_PRECISION
from solar.common.utils import ensure_directory, NoAliasDumper


PathLike = Union[str, Path]


def _product(shape: List[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return int(out)


class EinsumGraphAnalyzer:
    """Analyze `einsum_graph.yaml` and write `analysis.yaml`."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.einsum_analyzer = EinsumAnalyzer(debug=debug)

    def analyze_graph(
        self,
        einsum_graph_path: PathLike,
        output_dir: PathLike,
        *,
        precision: str = DEFAULT_PRECISION,
        copy_graph: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Analyze an einsum graph and write `analysis.yaml`.

        Args:
            einsum_graph_path: Path to `einsum_graph.yaml`.
            output_dir: Directory to write `analysis.yaml` into.
            precision: Tensor precision for byte calculations (e.g., fp32, bf16).
            copy_graph: If True, copy the einsum graph into output dir using the
                canonical name `einsum_graph.yaml`.

        Returns:
            Analysis dict, or None on failure.
        """
        src = Path(einsum_graph_path)
        out_dir = ensure_directory(output_dir)

        if not src.exists():
            if self.debug:
                print(f"Debug: einsum graph not found: {src}")
            return None

        try:
            with open(src) as f:
                graph = yaml.safe_load(f) or {}
        except Exception as exc:
            if self.debug:
                print(f"Debug: failed reading einsum graph: {exc}")
            return None

        if copy_graph:
            try:
                dst = out_dir / "einsum_graph.yaml"
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            except Exception:
                if self.debug:
                    print("Debug: failed to copy einsum_graph.yaml")

        all_layers: Dict[str, Any] = graph.get("layers") or {}
        element_size = int(BYTES_PER_ELEMENT.get(precision, 4))

        # Filter out "start" nodes - they represent model inputs, not computation
        # Keep track of start node IDs for reference
        start_node_ids: Set[str] = set()
        layers_in: Dict[str, Any] = {}
        
        for layer_id, layer in all_layers.items():
            op_type = str(layer.get("type", "")).lower()
            if op_type == "start":
                start_node_ids.add(layer_id)
            else:
                layers_in[layer_id] = layer
        
        if self.debug:
            print(f"Debug: Filtered out {len(start_node_ids)} start nodes")
            print(f"Debug: Analyzing {len(layers_in)} computation nodes")

        # Build tensor producer/consumer maps to identify intermediate tensors
        # A tensor is intermediate if it's produced by one op AND consumed by another op
        # Tensor naming: <layer_id>.Output is the output tensor of layer_id
        all_layer_ids: Set[str] = set(layers_in.keys())
        
        # Track which tensors are produced (outputs) and consumed (inputs)
        tensor_producers: Dict[str, str] = {}  # tensor_name -> producer_layer_id
        tensor_consumers: Dict[str, Set[str]] = {}  # tensor_name -> set of consumer_layer_ids
        
        for layer_id, layer in layers_in.items():
            connections = layer.get("connections") or {}
            input_layer_ids = list(connections.get("inputs") or [])
            
            # This layer produces its output tensor
            output_tensor = f"{layer_id}.Output"
            tensor_producers[output_tensor] = layer_id
            
            # This layer consumes tensors from its input layers (excluding start nodes)
            for input_layer_id in input_layer_ids:
                if input_layer_id in all_layer_ids:  # Only count non-start nodes
                    input_tensor = f"{input_layer_id}.Output"
                    if input_tensor not in tensor_consumers:
                        tensor_consumers[input_tensor] = set()
                    tensor_consumers[input_tensor].add(layer_id)
        
        # Identify intermediate tensors: produced by one op AND consumed by another
        intermediate_tensors: Set[str] = set()
        for tensor_name in tensor_producers:
            if tensor_name in tensor_consumers and len(tensor_consumers[tensor_name]) > 0:
                intermediate_tensors.add(tensor_name)
        
        if self.debug:
            print(f"Debug: Found {len(intermediate_tensors)} intermediate tensors")
            for t in sorted(intermediate_tensors)[:10]:
                print(f"  - {t}")

        layers_out: Dict[str, Any] = {}
        total_macs = 0
        total_flops = 0
        total_other_ops = 0
        total_orojenesis_elems = 0
        total_fused_elems = 0
        total_intermediate_elems = 0

        for layer_id, layer in layers_in.items():
            op_type = str(layer.get("type", "unknown"))
            equation = str(layer.get("einsum_equation", "") or "")
            if "is_real_einsum" not in layer:
                raise ValueError(
                    f"Layer '{layer_id}' (type={op_type}) is missing 'is_real_einsum' field. "
                    f"All layers in the einsum graph must specify is_real_einsum: true/false."
                )
            is_real_einsum = bool(layer["is_real_einsum"])
            shapes: Dict[str, List[int]] = layer.get("shapes") or {}
            tensor_shapes: Dict[str, Any] = layer.get("tensor_shapes") or {}
            connections: Dict[str, Any] = layer.get("connections") or {}
            input_layer_ids = list(connections.get("inputs") or [])
            output_layer_ids = list(connections.get("outputs") or [])

            # Compute ops cost from einsum analyzer
            # For real einsum: this goes into macs; for non-real: this goes into other_ops
            ops_cost = 0
            try:
                ops_cost = int(self.einsum_analyzer.get_compute_cost(op_type, shapes))
            except Exception:
                ops_cost = 0

            if is_real_einsum:
                macs = ops_cost
                other_ops = 0
            else:
                macs = 0
                other_ops = ops_cost

            flops = int(2 * macs)

            # Memory elements: use tensor_shapes to avoid double counting
            # tensor_shapes has: inputs (list of shapes), outputs (list of shapes)
            # This is more accurate than shapes which may have duplicate entries
            # (e.g., Input_1 and Weight can be the same tensor)
            input_shapes = tensor_shapes.get("inputs") or []
            output_shapes = tensor_shapes.get("outputs") or []
            
            # Calculate input elements from tensor_shapes
            input_elems = 0
            for shp in input_shapes:
                if isinstance(shp, list):
                    input_elems += _product(shp)
            input_elems = int(input_elems)
            
            # Calculate output elements from tensor_shapes
            output_elems = 0
            for shp in output_shapes:
                if isinstance(shp, list):
                    output_elems += _product(shp)
            output_elems = int(output_elems)
            
            # Total memory elements = inputs + outputs (no double counting)
            orojenesis_elems = int(input_elems + output_elems)
            
            # Build memory_elements dict for reporting (use tensor_shapes structure)
            mem_elems: Dict[str, int] = {}
            for i, shp in enumerate(input_shapes):
                if isinstance(shp, list):
                    mem_elems[f"Input_{i}" if i > 0 else "Input"] = _product(shp)
            for i, shp in enumerate(output_shapes):
                if isinstance(shp, list):
                    mem_elems[f"Output_{i}" if i > 0 else "Output"] = _product(shp)
            mem_elems["total"] = orojenesis_elems
            
            # Check if inputs come from intermediate tensors (outputs of other ops in graph)
            # Inputs from start nodes are NOT intermediate - they are model inputs (external)
            inputs_from_graph = [
                inp for inp in input_layer_ids 
                if inp in all_layer_ids  # Only non-start nodes count
            ]
            input_is_intermediate = len(inputs_from_graph) > 0
            
            # Check if output is intermediate (consumed by another op in graph)
            output_tensor = f"{layer_id}.Output"
            output_is_intermediate = output_tensor in intermediate_tensors
            
            # Calculate intermediate elements for this layer
            intermediate_input_elems = input_elems if input_is_intermediate else 0
            intermediate_output_elems = output_elems if output_is_intermediate else 0
            layer_intermediate_elems = intermediate_input_elems + intermediate_output_elems
            
            # Model I/O elements: inputs not from graph + outputs not consumed by graph
            model_input_elems = input_elems if not input_is_intermediate else 0
            model_output_elems = output_elems if not output_is_intermediate else 0
            model_io_elems = model_input_elems + model_output_elems

            # Fused elements = non-intermediate I/O (inputs include weights)
            # Intermediate tensors are excluded because they stay in cache/registers
            fused_elems = int(model_io_elems)

            layers_out[layer_id] = {
                "type": op_type,
                "einsum_equation": equation,
                "is_real_einsum": is_real_einsum,
                "macs": macs,
                "other_ops": other_ops,
                "flops": flops,
                "orojenesis_elements": orojenesis_elems,
                "fused_elements": fused_elems,
                "memory_elements": mem_elems,
                "input_elements": input_elems,
                "output_elements": output_elems,
                "intermediate_elements": layer_intermediate_elems,
                "model_io_elements": model_io_elems,
                "input_is_intermediate": input_is_intermediate,
                "output_is_intermediate": output_is_intermediate,
                "connections": {"inputs": input_layer_ids, "outputs": output_layer_ids},
            }

            total_macs += macs
            total_other_ops += other_ops
            total_flops += flops
            total_orojenesis_elems += orojenesis_elems
            total_fused_elems += fused_elems
            total_intermediate_elems += layer_intermediate_elems

        # Calculate fused_prefetched_elements: total model I/O across entire graph
        # This is the memory footprint when all intermediate tensors are perfectly fused
        # Note: input_elements already includes weights, so model_io_elements captures
        # all non-intermediate memory accesses
        total_model_io_elems = sum(
            layer.get("model_io_elements", 0)
            for layer in layers_out.values()
        )
        total_fused_prefetched_elems = int(total_model_io_elems)

        analysis: Dict[str, Any] = {
            "layers": layers_out,
            "total": {
                "num_layers": len(layers_out),
                "num_start_nodes_filtered": len(start_node_ids),
                "macs": int(total_macs),
                "other_ops": int(total_other_ops),
                "flops": int(total_flops),
                "orojenesis_elements": int(total_orojenesis_elems),
                "fused_elements": int(total_fused_elems),
                "fused_prefetched_elements": total_fused_prefetched_elems,
                "model_io_elements": int(total_model_io_elems),
                "intermediate_elements": int(total_intermediate_elems),
                "num_intermediate_tensors": len(intermediate_tensors),
            },
            "metadata": {
                "precision": precision,
                "bytes_per_element": element_size,
                "source_graph": str(src),
            },
        }

        out_path = out_dir / "analysis.yaml"
        with open(out_path, "w") as f:
            yaml.dump(analysis, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            print(f"✅ Wrote analysis: {out_path}")

        return analysis


__all__ = ["EinsumGraphAnalyzer"]
