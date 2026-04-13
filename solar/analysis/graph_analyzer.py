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
- unfused_elements: All tensor accesses (inputs + outputs) per op, summed
- orojenesis_elements: Set to None (orojenesis runs not enabled)
- fused_elements: Deduplicated external I/O (weights + model inputs/outputs),
    same tensor read by multiple ops counted once. Equal to fused_prefetched.
- fused_prefetched_elements: Same as fused_elements (deduplicated external I/O)

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
        _BOOL_DTYPES = {"torch.bool", "bool"}
        start_node_ids: Set[str] = set()
        bool_start_node_ids: Set[str] = set()
        layers_in: Dict[str, Any] = {}

        for layer_id, layer in all_layers.items():
            op_type = str(layer.get("type", "")).lower()
            if op_type == "start":
                start_node_ids.add(layer_id)
                # Detect bool-typed start nodes from tensor_dtypes
                out_dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
                if out_dtypes and all(str(d) in _BOOL_DTYPES for d in out_dtypes):
                    bool_start_node_ids.add(layer_id)
            else:
                layers_in[layer_id] = layer

        if self.debug:
            print(f"Debug: Filtered out {len(start_node_ids)} start nodes")
            if bool_start_node_ids:
                print(f"Debug: Found {len(bool_start_node_ids)} bool-typed start nodes: {bool_start_node_ids}")
            print(f"Debug: Analyzing {len(layers_in)} computation nodes")

        # Build tensor producer/consumer maps using tensor_names from the
        # einsum graph.  A tensor is intermediate if it is produced by one op
        # AND consumed by another op.
        all_layer_ids: Set[str] = set(layers_in.keys())

        tensor_producers: Dict[str, str] = {}   # tensor_name -> producer_layer_id
        tensor_consumers: Dict[str, Set[str]] = {}  # tensor_name -> set of consumer_layer_ids

        # Pass 1: gather all produced tensor names (order-independent).
        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}
            for oname in (t_names.get("outputs") or []):
                tensor_producers[oname] = layer_id

        # Pass 2: gather consumers for tensors produced somewhere in graph.
        for layer_id, layer in layers_in.items():
            t_names = layer.get("tensor_names") or {}
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

        # TEMPORARY FIX: Propagate bool-ness from start nodes through graph.
        # A computation layer is "bool" if ALL its inputs come from bool
        # sources (bool start nodes or other bool layers).
        _bool_layers: Set[str] = set()
        if bool_start_node_ids:
            # Process layers in topological-ish order (inputs before outputs)
            # by iterating until no more layers are added.
            changed = True
            while changed:
                changed = False
                for layer_id, layer in layers_in.items():
                    if layer_id in _bool_layers:
                        continue
                    conns = layer.get("connections") or {}
                    inp_ids = list(conns.get("inputs") or [])
                    if not inp_ids:
                        continue
                    # All inputs must be bool (start or layer)
                    if all(
                        inp in bool_start_node_ids or inp in _bool_layers
                        for inp in inp_ids
                    ):
                        _bool_layers.add(layer_id)
                        changed = True

            if self.debug and _bool_layers:
                print(f"Debug: Skipping memory for {len(_bool_layers)} bool-derived layers: {sorted(_bool_layers)}")

        layers_out: Dict[str, Any] = {}
        total_macs = 0          # tensor-core MACs (matmul, conv)
        total_flops = 0         # 2 * total_macs
        total_other_ops = 0     # CUDA-core elementwise/reduction ops
        total_unfused_elems = 0       # Σ (all input + output elems) per op
        total_intermediate_elems = 0  # Σ intermediate activation elems

        # Deduplicated external (non-intermediate) tensor tracking for the
        # fused / fused_prefetched model.  When the same external tensor
        # (e.g. model input x) fans out to multiple ops, it is read from
        # DRAM once.  We track by tensor_name → max element count.
        unique_external_inputs: Dict[str, int] = {}
        unique_external_outputs: Dict[str, int] = {}

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

            # ── Step 1: Compute per-tensor sizes from shapes ──
            input_sizes: List[int] = []
            output_sizes: List[int] = []
            for shp in input_shapes:
                input_sizes.append(_product(shp) if isinstance(shp, list) else 0)
            for shp in output_shapes:
                output_sizes.append(_product(shp) if isinstance(shp, list) else 0)

            # memory_reads[i]  = DRAM read elements for input tensor i
            # memory_writes[i] = DRAM write elements for output tensor i
            # Initialised to raw tensor sizes; special-case ops override below.
            memory_reads: List[int] = list(input_sizes)
            memory_writes: List[int] = list(output_sizes)

            # ── Step 2: Override memory_reads/writes for special-case ops ──

            _ZERO_COPY_VIEW_OPS = {
                "expand", "expand_as",
                "view", "reshape", "contiguous",
                "transpose", "permute", "t",
                "unsqueeze", "squeeze", "flatten",
                "unfold", "unflatten",
                # chunk/split return views into the source tensor
                "chunk", "split", "tensor_split",
            }
            _SLICE_VIEW_OPS = {
                "__getitem__", "narrow", "slice", "select",
            }
            _SCATTER_OPS = {
                "__setitem__", "scatter", "scatter_",
                "index_copy", "index_copy_",
                "index_put", "index_put_",
            }

            # For embedding (table lookup), only the gathered rows are read
            # from the weight matrix, not the entire vocabulary table.
            # Input shapes are [indices_shape, weight_shape] where weight is
            # [vocab_size, embedding_dim].  The actual DRAM read is just the
            # rows selected by indices — which equals the output shape
            # [batch, seq, embedding_dim].  Use min(input, output) to handle
            # both small token counts (gathered rows << full table) and large
            # token counts (most/all rows accessed, full table is the bound).
            if op_type in ("embedding", "embedding_bag"):
                total_output = sum(output_sizes)
                gathered = min(sum(input_sizes), total_output)
                memory_reads = [0] * len(input_sizes)
                if input_sizes:
                    memory_reads[-1] = gathered
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # TEMPORARY FIX: Skip memory for bool-typed tensors.
            # Bool tensors (masks, attention patterns) are 1 byte each but
            # SOLAR uses a global bytes_per_element (2 for fp16). Rather than
            # counting them at the wrong byte width, zero them out — masks are
            # negligible compared to compute/activation tensors and should not
            # dominate the SOL estimate.
            if layer_id in _bool_layers:
                memory_reads = [0] * len(input_sizes)
                memory_writes = [0] * len(output_sizes)

            # View/reshape ops produce zero-copy aliases — they never
            # materialize data to DRAM.  The downstream consumer accounts
            # for the actual read, so these ops contribute 0 memory.
            elif op_type in _ZERO_COPY_VIEW_OPS:
                memory_reads = [0] * len(input_sizes)
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # Slicing/selection ops return a view into the source tensor.
            # The actual memory read is the output slice size, not the
            # full source.  Set read = output size so the downstream
            # consumer accounts for reading the slice.
            elif op_type in _SLICE_VIEW_OPS:
                out_total = sum(output_sizes)
                memory_reads = [out_total] if input_sizes else []
                memory_reads += [0] * max(0, len(input_sizes) - 1)
                memory_writes = [0] * len(output_sizes)
                other_ops = 0

            # Scatter/index-write ops (__setitem__, scatter, index_copy)
            # write a slice into a large target tensor.  Memory cost is
            # the values being written, not the full target.  The smallest
            # input shape is typically the values/indices; use that as
            # the write cost and set output to the same (in-place update).
            elif op_type in _SCATTER_OPS:
                if len(input_sizes) >= 2:
                    slice_elems = max(sorted(input_sizes)[:-1])
                elif input_sizes:
                    slice_elems = min(input_sizes)
                elif output_sizes:
                    slice_elems = min(output_sizes)
                else:
                    slice_elems = 0
                memory_reads = [0] * len(input_sizes)
                memory_writes = [slice_elems] if output_sizes else []
                memory_writes += [0] * max(0, len(output_sizes) - 1)
                other_ops = 0

            # ── Step 3: Derive totals from corrected per-tensor counts ──
            input_elems = int(sum(memory_reads))
            output_elems = int(sum(memory_writes))
            unfused_elems = input_elems + output_elems

            # ── Step 4: Classify inputs as external vs graph-internal ──
            # Uses memory_reads (already corrected) so no re-scanning needed.
            # Classify each input tensor:
            #   - "weight"        → always external (DRAM read every time)
            #   - graph-internal  → intermediate activation (fusable, skip in fused model)
            #   - other           → external model input (DRAM read)
            #
            # graph-internal = not a weight AND produced by another op in the graph.
            # When input_elems was zeroed (zero-copy/scatter ops), skip classification.        
            input_name_list = tensor_names.get("inputs") or []
            graph_internal_input_elems = 0   # intermediate activations from other ops
            external_input_elems = 0         # weights + model-level inputs (always DRAM)

            for i, mem_read in enumerate(memory_reads):
                if mem_read <= 0:
                    continue
                itype = input_type_list[i] if i < len(input_type_list) else "weight"
                iname = input_name_list[i] if i < len(input_name_list) else ""
                is_graph_internal = (itype != "weight" and iname in tensor_producers)

                if is_graph_internal:
                    graph_internal_input_elems += mem_read
                else:
                    external_input_elems += mem_read
                    if iname:
                        unique_external_inputs[iname] = max(
                            unique_external_inputs.get(iname, 0), mem_read
                        )

            intermediate_input_elems = int(graph_internal_input_elems)
            model_input_elems = int(external_input_elems)
            input_is_intermediate = graph_internal_input_elems > 0

            # Classify outputs: intermediate if consumed by another op, else model output.
            output_name_list = tensor_names.get("outputs") or []
            output_is_intermediate = any(
                oname in tensor_consumers for oname in output_name_list
            )

            # Intermediate output elems: written to cache (fused) not DRAM
            intermediate_output_elems = output_elems if output_is_intermediate else 0
            # Total intermediate elems for this layer (inputs + outputs)
            layer_intermediate_elems = intermediate_input_elems + intermediate_output_elems

            # Model output elems: final graph outputs that must go to DRAM
            model_output_elems = output_elems if not output_is_intermediate else 0
            # Per-op model I/O: external inputs + model outputs (no intermediates)
            model_io_elems = model_input_elems + model_output_elems

            # Track unique external outputs for deduplication.
            if not output_is_intermediate:
                for oname in output_name_list:
                    unique_external_outputs[oname] = max(
                        unique_external_outputs.get(oname, 0), int(output_elems)
                    )

            # Per-op fused elements: only non-intermediate DRAM traffic
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
                "memory_elements": {
                    "inputs": memory_reads,
                    "outputs": memory_writes,
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
            total_intermediate_elems += layer_intermediate_elems

        # Deduplicated graph-level external I/O: when the same tensor
        # (e.g. model input x) fans out to multiple ops, count it once.
        # Used for both fused and fused_prefetched totals.
        total_fused_prefetched_elems = int(
            sum(unique_external_inputs.values())
            + sum(unique_external_outputs.values())
        )
        # fused_elements == fused_prefetched_elements (same dedup logic)
        total_fused_elems = total_fused_prefetched_elems

        # model_io_elements: raw per-op sum (may double-count shared inputs).
        # Kept for diagnostic / per-layer inspection.
        total_model_io_elems = sum(
            layer.get("model_io_elements", 0)
            for layer in layers_out.values()
        )

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
