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

"""Export DSLGraph to einsum_graph.yaml format."""

from pathlib import Path
from copy import deepcopy
import yaml

from .model import DSLGraph, DSLKernel


def _kernel_connection_inputs(kernel: DSLKernel) -> list:
    """Return the list of predecessor node IDs for connections.inputs.

    Rules:
    - Weight inputs (role=='weight') are not included in connections — they are
      standalone tensors with no data-flow predecessor node.
    - Model inputs (role=='input', from_kernel=='input') map to the 'start' node.
    - Intermediate tensors (from_kernel == some kernel ID) map to that kernel ID.
    """
    seen = []
    for inp in kernel.inputs:
        if inp.tensor.is_weight:
            continue
        if inp.from_kernel == "input":
            node = "start"
        else:
            node = inp.from_kernel
        if node not in seen:
            seen.append(node)
    return seen


def dsl_graph_to_dict(graph: DSLGraph) -> dict:
    """Convert DSLGraph to the einsum_graph.yaml dict structure.

    The output mirrors the format produced by the solar PyTorch-to-einsum pipeline:
      model_name: <str>
      layers:
        start: {...}
        <kernel_id>: {...}
        ...

    Notable fields per node:
      einsum_equation: null  (to be filled in by a future converter)
      loop_expr: <str>       (innermost loop body from the DSL Python file)
      elementwise_op: null
      reduction_op: null
    """
    model_inputs = graph.model_inputs
    start_consumers = graph.kernels_consuming_model_inputs()

    # Only emit tensor_sparsity when at least one tensor carries sparsity
    _has_sparsity = any(t.sparsity is not None for t in graph.tensors.values())

    # Build start node
    start_node = {
        "type": "start",
        "einsum_equation": None,
        "loop_expr": None,
        "elementwise_op": None,
        "reduction_op": None,
        "tensor_names": {
            "inputs": [],
            "outputs": [t.name for t in model_inputs],
        },
        "tensor_types": {
            "inputs": [],
            "outputs": [t.tensor_type for t in model_inputs],
        },
        "tensor_shapes": {
            "inputs": [],
            "outputs": [t.shape for t in model_inputs],
        },
        "tensor_dtypes": {"inputs": [], "outputs": [t.dtype for t in model_inputs]},
        "connections": {
            "inputs": [],
            "outputs": [k.id for k in start_consumers],
        },
    }
    if _has_sparsity:
        start_node["tensor_sparsity"] = {
            "inputs": [],
            "outputs": [t.sparsity for t in model_inputs],
        }

    layers = {"start": start_node}

    for kernel in graph.kernels:
        all_input_names = [inp.tensor.name for inp in kernel.inputs]
        all_input_types = [inp.tensor.tensor_type for inp in kernel.inputs]
        all_input_shapes = [inp.tensor.shape for inp in kernel.inputs]

        conn_inputs = _kernel_connection_inputs(kernel)
        conn_outputs = graph.successors(kernel.id)

        layer_dict = {
            "type": "kernel",
            "einsum_equation": None,
            "loop_expr": kernel.loop_expr,
            "is_real_einsum": True,
            "elementwise_op": None,
            "reduction_op": None,
            "tensor_names": {
                "inputs": all_input_names,
                "outputs": [kernel.output.name],
            },
            "tensor_types": {
                "inputs": all_input_types,
                "outputs": [kernel.output.tensor_type],
            },
            "tensor_shapes": {
                "inputs": all_input_shapes,
                "outputs": [kernel.output.shape],
            },
            "tensor_dtypes": {
                "inputs": [inp.tensor.dtype for inp in kernel.inputs],
                "outputs": [kernel.output.dtype],
            },
            "connections": {
                "inputs": conn_inputs,
                "outputs": conn_outputs,
            },
        }
        if _has_sparsity:
            layer_dict["tensor_sparsity"] = {
                "inputs": [inp.tensor.sparsity for inp in kernel.inputs],
                "outputs": [kernel.output.sparsity],
            }
            # Positional param names let the analyzer decide which sparse
            # inputs participate MULTIPLICATIVELY in loop_expr (zeros-free
            # compute discount applies only to those).
            if kernel.param_names:
                layer_dict["param_names"] = list(kernel.param_names)
            if kernel.mult_params:
                layer_dict["multiplicative_params"] = list(kernel.mult_params)
        layers[kernel.id] = layer_dict

    result = {"model_name": graph.model_name, "layers": layers}
    if graph.model_outputs is not None:
        result["model_outputs"] = deepcopy(graph.model_outputs)
        result["model_output_ops"] = list(dict.fromkeys(
            o["op"] for o in graph.model_outputs if o.get("op") is not None))
        result["model_output_provenance"] = graph.model_output_provenance
    if getattr(graph, "returned_outputs_unrecorded", None):
        # return-provenance-1: the sidecar's fail-closed marker reaches
        # the analyzed graph; the analyzer reports the external-output
        # footprint unknown (incomplete) instead of pricing a leaf guess.
        result["returned_outputs_unrecorded"] = str(graph.returned_outputs_unrecorded)
    # Recorded trace provenance per tensor (registered-input vs
    # internal-constant) — the analyzer's only admissible evidence for
    # treating a boundary tensor as internally created; a __<role>_<n>
    # name spelling is NOT provenance (review C19-P1).
    origins = {t.name: t.origin
               for t in graph.tensors.values() if t.origin}
    if origins:
        result["tensor_origins"] = origins
    return result


def write_einsum_yaml(data: dict, path: Path) -> None:
    """Write the einsum_graph dict to YAML without anchors/aliases."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
