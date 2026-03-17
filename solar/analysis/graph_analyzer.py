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

"""Analyze an einsum graph into hardware-independent metrics.

This module implements the **second stage** of the Solar pipeline:

  `einsum_graph.yaml`  ->  `analysis.yaml`

The output `analysis.yaml` is intended to be hardware-independent and includes:
- per-layer: macs, flops (= 2 * macs), unfused_elements, fused_elements
- totals across the graph

Memory Access Models (in elements, multiply by bytes_per_element for bytes):
- unfused_elements: All tensor accesses (inputs + outputs) per op
- orojenesis_elements: Set to None (orojenesis runs not enabled)
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
from solar.common.types import TensorShapes
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
        element_size = BYTES_PER_ELEMENT.get(precision, 4)

        # Override precision/element_size from quant metadata if available
        quant_precision = self._resolve_quant_precision(src)
        if quant_precision:
            element_size = BYTES_PER_ELEMENT.get(quant_precision, element_size)
            precision = quant_precision
            if self.debug:
                print(f"  Quant override: precision={precision}, bytes_per_element={element_size}")

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

        # Build tensor producer/consumer maps using tensor_names from the
        # einsum graph.  A tensor is intermediate if it is produced by one op
        # AND consumed by another op.
        all_layer_ids: Set[str] = set(layers_in.keys())

        tensor_producers: Dict[str, str] = {}   # tensor_name -> producer_layer_id
        tensor_consumers: Dict[str, Set[str]] = {}  # tensor_name -> set of consumer_layer_ids

        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}

            for oname in (t_names.get("outputs") or []):
                tensor_producers[oname] = layer_id

            for iname in (t_names.get("inputs") or []):
                if iname in tensor_producers:
                    tensor_consumers.setdefault(iname, set()).add(layer_id)
        
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
        total_unfused_elems = 0
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
            tensor_shapes: Dict[str, Any] = layer.get("tensor_shapes") or {}
            tensor_types: Dict[str, Any] = layer.get("tensor_types") or {}
            tensor_names: Dict[str, Any] = layer.get("tensor_names") or {}
            connections: Dict[str, Any] = layer.get("connections") or {}
            input_layer_ids = list(connections.get("inputs") or [])
            output_layer_ids = list(connections.get("outputs") or [])

            ts = TensorShapes(
                inputs=tensor_shapes.get("inputs", []),
                outputs=tensor_shapes.get("outputs", []),
            )

            ops_cost = 0
            try:
                ops_cost = int(self.einsum_analyzer.get_compute_cost(op_type, ts))
            except Exception:
                ops_cost = 0

            # Zero-compute operations: no ALU work, only pointer/metadata
            # manipulation or pure memory copies.
            #
            # View/reshape ops: pointer manipulation, zero cost
            # Slice/select ops: pointer offset, zero cost
            # Scatter/index ops: in-place writes, zero compute
            # Embedding: table lookup, zero MACs
            # Memory ops (cat, repeat, stack, chunk, split): move data but
            #   have zero *compute* cost — bounded by memory bandwidth, not
            #   SM throughput.  Their memory cost is already captured by
            #   input_elems/output_elems; assigning them other_ops would
            #   double-count as both memory AND compute.
            _ZERO_COMPUTE_OPS = {
                # Embedding
                "embedding", "embedding_bag",
                # View / reshape (pointer manipulation)
                "expand", "expand_as",
                "view", "reshape", "contiguous",
                "transpose", "permute", "t",
                "unsqueeze", "squeeze", "flatten",
                "unfold", "unflatten",
                # Slice / select (pointer offset)
                "__getitem__", "narrow", "slice", "select",
                # Scatter / in-place write
                "__setitem__", "scatter", "scatter_",
                "index_copy", "index_copy_",
                "index_put", "index_put_",
                # Memory-only ops (data movement, zero ALU compute)
                "cat", "concat", "stack",
                "chunk", "split", "tensor_split",
                "repeat", "repeat_interleave", "tile",
                "roll", "flip",
                "pad", "constant_pad_nd",
                "clone", "copy_",
                # Type conversion (zero compute)
                "to", "type", "type_as", "float", "half", "bfloat16", "int",
            }
            if op_type in _ZERO_COMPUTE_OPS:
                ops_cost = 0
                is_real_einsum = False

            if is_real_einsum:
                macs = ops_cost
                other_ops = 0
            else:
                macs = 0
                other_ops = ops_cost

            flops = int(2 * macs)

            input_shapes = tensor_shapes.get("inputs") or []
            output_shapes = tensor_shapes.get("outputs") or []
            input_type_list = tensor_types.get("inputs") or []
            output_type_list = tensor_types.get("outputs") or []

            # Per-tensor element counts and sizes
            input_sizes: List[int] = []
            output_sizes: List[int] = []
            input_elems = 0
            output_elems = 0

            for shp in input_shapes:
                if isinstance(shp, list):
                    e = _product(shp)
                    input_sizes.append(int(e))
                    input_elems += e
            input_elems = int(input_elems)

            for shp in output_shapes:
                if isinstance(shp, list):
                    e = _product(shp)
                    output_sizes.append(int(e))
                    output_elems += e
            output_elems = int(output_elems)

            # View/reshape ops produce zero-copy aliases — they never
            # materialize data to DRAM.  The downstream consumer accounts
            # for the actual read, so these ops contribute 0 memory.
            _ZERO_COPY_VIEW_OPS = {
                "expand", "expand_as",
                "view", "reshape", "contiguous",
                "transpose", "permute", "t",
                "unsqueeze", "squeeze", "flatten",
                "unfold", "unflatten",
                # chunk/split return views into the source tensor
                "chunk", "split", "tensor_split",
            }
            # Slicing/selection ops return a view into the source tensor.
            # The actual memory read is the output slice size, not the
            # full source.  Set input = output size, output = 0 so the
            # downstream consumer accounts for reading the slice.
            _SLICE_VIEW_OPS = {
                "__getitem__", "narrow", "slice", "select",
            }
            # Scatter/index-write ops (__setitem__, scatter, index_copy)
            # write a slice into a large target tensor.  Memory cost is
            # the values being written, not the full target.  The smallest
            # input shape is typically the values/indices; use that as
            # the write cost and set output to the same (in-place update).
            _SCATTER_OPS = {
                "__setitem__", "scatter", "scatter_",
                "index_copy", "index_copy_",
                "index_put", "index_put_",
            }
            if op_type in _ZERO_COPY_VIEW_OPS:
                input_elems = 0
                output_elems = 0
            elif op_type in _SLICE_VIEW_OPS:
                input_elems = output_elems
                output_elems = 0
            elif op_type in _SCATTER_OPS:
                # Scatter inputs are typically [target, indices, source] or
                # [target, source].  The write size is the source/values
                # tensor — exclude the largest (target) and take the max
                # of the remaining (to skip tiny index tensors).
                if len(input_sizes) >= 2:
                    without_target = sorted(input_sizes)[:-1]
                    slice_elems = max(without_target)
                elif input_sizes:
                    slice_elems = min(input_sizes)
                elif output_sizes:
                    slice_elems = min(output_sizes)
                else:
                    slice_elems = 0
                input_elems = 0
                output_elems = slice_elems

            # Unfused elements = all inputs + outputs (no fusion)
            unfused_elems = int(input_elems + output_elems)

            # Per-input-tensor classification using tensor_types from the
            # einsum graph.  Weight/bias tensors (type="weight") have no
            # producer node in the graph and always require DRAM access.
            # Only activation inputs (type="input") flowing between
            # graph-internal ops are intermediate (fusable).
            #
            # When input_elems was overridden to 0 (zero-copy/scatter ops),
            # skip classification — all input memory is already accounted for.
            input_name_list = tensor_names.get("inputs") or []
            graph_internal_input_elems = 0
            external_input_elems = 0

            if input_elems > 0:
                for i, shp in enumerate(input_shapes):
                    if not isinstance(shp, list):
                        continue
                    elems = _product(shp)
                    itype = input_type_list[i] if i < len(input_type_list) else "weight"
                    iname = input_name_list[i] if i < len(input_name_list) else ""
                    if itype == "weight":
                        external_input_elems += elems
                    elif iname in tensor_producers:
                        graph_internal_input_elems += elems
                    else:
                        external_input_elems += elems
            else:
                external_input_elems = input_elems

            intermediate_input_elems = int(graph_internal_input_elems)
            model_input_elems = int(external_input_elems)
            input_is_intermediate = graph_internal_input_elems > 0

            output_name_list = tensor_names.get("outputs") or []
            output_is_intermediate = any(
                oname in tensor_consumers for oname in output_name_list
            )

            intermediate_output_elems = output_elems if output_is_intermediate else 0
            layer_intermediate_elems = intermediate_input_elems + intermediate_output_elems

            model_output_elems = output_elems if not output_is_intermediate else 0
            model_io_elems = model_input_elems + model_output_elems

            fused_elems = int(model_io_elems)

            layers_out[layer_id] = {
                "type": op_type,
                "einsum_equation": equation,
                "is_real_einsum": is_real_einsum,
                "macs": macs,
                "other_ops": other_ops,
                "flops": flops,
                "unfused_elements": unfused_elems,
                "orojenesis_elements": None,
                "fused_elements": fused_elems,
                "tensor_shapes": {
                    "inputs": [s for s in input_shapes if isinstance(s, list)],
                    "outputs": [s for s in output_shapes if isinstance(s, list)],
                },
                "tensor_sizes": {
                    "inputs": input_sizes,
                    "outputs": output_sizes,
                },
                "tensor_types": {
                    "inputs": list(input_type_list),
                    "outputs": list(output_type_list),
                },
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
            total_unfused_elems += unfused_elems
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
                "unfused_elements": int(total_unfused_elems),
                "orojenesis_elements": None,
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

    # Maps metadata orig_dtypes keywords to Solar precision names
    _QUANT_DTYPE_MAP = {
        "nvfp4": "nvfp4",
        "float4_e2m1fn_x2": "nvfp4",
        "fp8": "fp8",
        "float8_e4m3fn": "fp8",
        "float8_e5m2": "fp8",
    }

    def _resolve_quant_precision(self, einsum_graph_path: Path) -> Optional[str]:
        """Search for metadata.yaml near the einsum graph and return quant precision.

        Walks up from the einsum_graph_path looking for metadata.yaml
        (max 3 levels). Picks highest-throughput quant dtype (nvfp4 > fp8).
        """
        search_dir = einsum_graph_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        meta = yaml.safe_load(f) or {}
                except Exception:
                    return None

                best = None
                for conv in meta.get("dtype_conversions") or []:
                    orig = str(conv.get("orig_dtypes", "")).lower()
                    for keyword, prec in self._QUANT_DTYPE_MAP.items():
                        if keyword in orig:
                            if best is None or BYTES_PER_ELEMENT.get(prec, 99) < BYTES_PER_ELEMENT.get(best, 99):
                                best = prec
                            break
                return best
            search_dir = search_dir.parent
        return None


__all__ = ["EinsumGraphAnalyzer"]
