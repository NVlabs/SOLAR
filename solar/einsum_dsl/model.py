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

"""Domain model classes for the DSL graph representation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DSLTensor:
    """A tensor defined in the DSL YAML tensors section."""

    name: str
    dims: dict  # ordered: dim_name -> int size, e.g. {"B": 2, "S": 32, "H": 64}
    dtype: str  # e.g. "fp32"
    role: str   # "input" | "weight" | "" (intermediate/output)
    sparsity: Optional[float] = None  # density (fraction nonzero, 0-1); None = dense
    # Recorded trace provenance: "registered-input" (came from the public
    # inputs dict, whatever it is named) | "internal-constant" (created
    # inline by the trace) | "" (unrecorded — e.g. kernel outputs, or
    # metadata predating provenance recording).  Name spellings are never
    # provenance (review C19-P1).
    origin: str = ""

    @property
    def shape(self) -> list:
        return list(self.dims.values())

    @property
    def is_model_input(self) -> bool:
        return self.role == "input"

    @property
    def is_weight(self) -> bool:
        return self.role == "weight"

    @property
    def tensor_type(self) -> str:
        if self.is_model_input:
            return "input"
        if self.is_weight:
            return "weight"
        return "output"


@dataclass
class DSLInput:
    """One input edge into a kernel node."""

    tensor: DSLTensor
    from_kernel: str  # kernel ID (e.g. "linear_q_proj#0") or "input" for external tensors


@dataclass
class DSLKernel:
    """A kernel node in the DSL graph."""

    id: str             # e.g. "linear_q_proj#0"
    function_name: str  # e.g. "linear_q_proj"
    inputs: list        # list[DSLInput]
    output: DSLTensor
    loop_expr: Optional[str] = None  # innermost loop body, e.g. "Y[b, s, h2] += X[b, s, h] * W[h2, h]"
    param_names: Optional[list] = None  # function parameter names in order, e.g. ["X", "W"]
    loop_paths: Optional[list] = None  # [[(dim_name, explicit_size|None), ...], ...] from Dim() loops
    mult_params: Optional[list] = None  # params used multiplicatively anywhere in the body
    tensor_param_count: Optional[int] = None  # leading Tensor-annotated positional params; None = inputs not mappable to names
    inputs_in_signature_order: bool = False  # trace marker: recorded inputs follow parameter order (absent in legacy artifacts)
    loop_paths_complete: bool = False  # parser proof that loop_paths represent EVERY executed loop (False = not proven)
    loop_paths_omitted: Optional[list] = None  # loops the Dim() path walk could not represent (reasons)
    loop_outer_statements: Optional[list] = None  # [[(dim_name, size|None), ...], count] per non-innermost loop level (statement-coverage-1)
    loop_statement_classes: Optional[list] = None
    loop_statement_classes_uncertain: Optional[list] = None  # e.g. ["branch_class_uncertain"]: classes are a per-class maximum over branches  # [[(dim_name, size|None), ...], "mac"|"other", extra_multiplies] per executed assignment (issue #59)
    loop_statement_grammar_violations: Optional[list] = None  # supported-grammar violations: the kernel's classes are unproven (an earlier fix round 5)
    loop_statement_class_gather: Optional[list] = None  # per loop_statement_classes entry: the statement addresses through a declared GATHER index (an earlier fix round 6)
    loop_gather_controls: Optional[list] = None  # [[(dim_name, size|None), ...], ...]: statement-if TESTS evaluated through a gather, one evaluation per trip (an earlier fix round 7)
    loop_statement_class_lines: Optional[list] = None  # per loop_statement_classes entry: the entry line of the loop that produced it (an earlier change round 7: sibling nests share prefixes)
    loop_gather_control_lines: Optional[list] = None  # same, per loop_gather_controls entry
    loop_statement_class_operands: Optional[list] = None  # per loop_statement_classes entry: the tensor names its MAC multiplies ([] for other work)
    source_text: Optional[str] = None  # the kernel's own source segment (DTYPE-OCCURRENCE-CHANNEL-8: bound to the node with its hash)
    source_sha256: Optional[str] = None
    module_source_sha256: Optional[str] = None  # sha256 of the whole DSL module text (graph-level dsl_module_source)
    signature: Optional[list] = None  # [{name, position, tensor, dtype}] of the positional parameters, from the annotations
    loop_statement_class_singleton: Optional[list] = None  # per loop_statement_classes entry: the statement reads a SINGLETON-SCALE candidate (issue #61)
    loop_singleton_reads: Optional[dict] = None  # SINGLETON-SCALE-READ candidates: parameter -> declared rank; the extent is proven per call by the converter (issue #61)
    loop_singleton_controls: Optional[list] = None  # [[(dim_name, size|None), ...], ...]: statement-if TESTS reading a singleton, one evaluation per trip (issue #61)
    loop_singleton_outer_reads: Optional[int] = None  # singleton reads outside every loop, once per kernel call (issue #61)
    loop_singleton_control_lines: Optional[list] = None  # same as loop_gather_control_lines, per loop_singleton_controls entry (an earlier change x #61 reconciliation)
    loop_source_parsed: bool = False  # the kernel's source was parsed (its omissions are observed facts, not the legacy/builtin placeholder)
    loop_path_lines: Optional[list] = None  # per loop_paths entry: the line of EACH Dim loop along the path (registry state at each loop's entry)
    loop_outer_lines: Optional[list] = None  # same, per loop_outer_statements entry
    returned_names: Optional[list] = None  # names the kernel returns (``return Y``); None entries for non-Name returns
    registration_order: Optional[list] = None  # run-time registration order of the recorded inputs (indices), from the trace
    local_tensor_dims: Optional[list] = None  # [{"line": n, "dims": {name: size spec}, "conditional": bool}] per Tensor(...) allocation in the body, source order (runtime registry bindings)
    loop_nest: Optional[list] = None  # the Dim() loop nest as a tree: {"dim", "size", "line", "end", "weight"|"level_assigns", "children"} (round 4: iteration-by-iteration registry replay)
    name_bindings: Optional[list] = None  # every binding of a name in the body, execution order: {"seq", "name", "line", "kind": alloc|alias|other, "source", "alloc", "conditional"}
    returns: Optional[list] = None  # every return statement: {"seq", "line", "name", "conditional"}
    dim_registrations: Optional[list] = None  # explicit Dim("U", size=...) registrations other than a for-header's own call: {"line", "col", "dim", "size", "conditional"} (round 5) + "eseq"/"stmt_line"/"stmt_col" (evaluation order within the statement), "header_of" ([line, col] of the loop whose header nests it), "order_unknown"/"text" (nested where the parser does not model the order: refused) (round 8, nested-dim-eval-25)
    scalar_args: Optional[dict] = None  # the call's int arguments by parameter name, from the trace (Dim("R", size=kernel_size) resolves from them)


@dataclass
class DSLGraph:
    """The complete DSL graph with all tensors and kernels."""

    model_name: str
    tensors: dict       # name -> DSLTensor (all tensors including intermediates)
    kernels: list       # list[DSLKernel] in topological order from YAML
    model_outputs: Optional[list] = None  # recorded return identities, not leaf inference
    model_output_provenance: Optional[str] = None
    returned_outputs_unrecorded: Optional[str] = None  # the trace could not bind the returned arrays (fail-closed marker)

    @property
    def model_inputs(self) -> list:
        """Tensors with role=='input' — the model-level inputs fed to the start node."""
        return [t for t in self.tensors.values() if t.is_model_input]

    def kernels_consuming_model_inputs(self) -> list:
        """Kernels that have at least one input coming directly from a model input tensor."""
        result = []
        for k in self.kernels:
            for inp in k.inputs:
                if inp.from_kernel == "input" and inp.tensor.is_model_input:
                    result.append(k)
                    break
        return result

    def successors(self, kernel_id: str) -> list:
        """Kernel IDs that consume the output of kernel_id."""
        result = []
        for k in self.kernels:
            for inp in k.inputs:
                if inp.from_kernel == kernel_id:
                    result.append(k.id)
                    break
        return result
