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

"""Assemble DSLGraph from raw YAML data and AST-parsed loop expressions."""

from .model import DSLTensor, DSLInput, DSLKernel, DSLGraph


class UnsupportedKernelReturn(ValueError):
    """A traced kernel call recorded no output Tensor: the kernel returned a tuple (or another non-Tensor value),
    which the single-output graph node cannot represent (TARGET-DSL-TUPLE-13).  ``code`` is
    ``unsupported_tuple_return`` for a tuple, ``unsupported_return:<kind>`` otherwise (``unsupported_return`` for a
    legacy trace that did not record the kind)."""

    def __init__(self, kernel_id, function_name, info=None):
        info = info if isinstance(info, dict) else {}
        kind = info.get("kind")
        if kind == "tuple":
            self.code = "unsupported_tuple_return"
            what = f"a tuple of {info.get('length', '?')} values ({info.get('tensors', '?')} Tensor(s))"
        elif kind:
            self.code = f"unsupported_return:{kind}"
            what = f"a {kind} value"
        else:
            self.code = "unsupported_return"
            what = "a non-Tensor value (legacy trace without the return kind)"
        self.kernel_id = kernel_id
        super().__init__(
            f"{self.code}: kernel {function_name!r} ({kernel_id}) returned {what}; the einsum graph represents one "
            f"output Tensor per kernel call -- return a single Tensor (split the kernel, or return the tuple from a "
            f"plain Python function that calls one kernel per output)")



def _rename_numeric_dims(tensors: dict) -> None:
    """Rename numeric dim names (e.g. '1') to unique letter-based names in place.

    The einsum pipeline requires dim names to start with a letter so that
    ``parse_dim_tokens`` can tokenise them correctly.  YAML dims like ``'1': 1``
    (common in depthwise conv weights) are renamed to an unused letter.
    """
    # Collect all existing dim names across all tensors
    all_dim_names: set = set()
    for tensor in tensors.values():
        all_dim_names.update(tensor.dims.keys())

    # Find numeric dim names that need renaming
    numeric_names = [d for d in all_dim_names if d.isdigit() or (isinstance(d, int))]

    if not numeric_names:
        return

    # Assign replacement names (try G for "groups", then backwards through alphabet)
    taken = {d[0].upper() for d in all_dim_names if isinstance(d, str) and d and d[0].isalpha()}
    rename_map: dict = {}
    for num_name in numeric_names:
        for c in "GZYXVUTIAJLNM":
            if c not in taken:
                rename_map[num_name] = c.lower()  # lowercase to match DSL convention
                taken.add(c)
                break

    if not rename_map:
        return

    # Apply renames to all tensors
    for tensor in tensors.values():
        new_dims = {}
        for dim_name, dim_size in tensor.dims.items():
            key = str(dim_name) if isinstance(dim_name, int) else dim_name
            if key in rename_map:
                new_dims[rename_map[key]] = dim_size
            else:
                new_dims[dim_name] = dim_size
        tensor.dims = new_dims


def build_dsl_graph(raw_yaml: dict, loop_exprs: dict, model_name: str,
                    kernel_info: dict = None) -> DSLGraph:
    """Build a DSLGraph from the raw DSL YAML dict and loop expression map.

    Args:
        raw_yaml:    Output of load_dsl_yaml — dict with 'tensors' and 'graph' keys.
        loop_exprs:  Output of extract_loop_exprs — {function_name: loop_expr_str | None}.
                     Ignored if kernel_info is provided.
        model_name:  Model name string (derived from filename stem).
        kernel_info: Output of extract_kernel_info — {function_name: {"loop_expr": ..., "param_names": [...]}}.
                     If provided, takes precedence over loop_exprs.

    Returns:
        DSLGraph with all tensors and kernels populated.
    """
    # Build DSLTensor objects from the tensors section
    tensors: dict = {}
    for name, tdata in raw_yaml["tensors"].items():
        # Convert dim keys to strings (YAML may parse '1' as int 1)
        raw_dims = tdata.get("dims", {})
        dims = {str(k): v for k, v in raw_dims.items()}
        raw_sparsity = tdata.get("sparsity")
        sparsity = float(raw_sparsity) if raw_sparsity is not None else None
        tensors[name] = DSLTensor(
            name=name,
            dims=dims,
            dtype=tdata.get("dtype", "fp32"),
            role=tdata.get("role", ""),
            sparsity=sparsity,
            origin=tdata.get("origin", ""),
        )

    # Rename numeric dim names (e.g. '1') to letters for parse_dim_tokens compat
    _rename_numeric_dims(tensors)

    # Build DSLKernel objects from the graph section (preserves YAML insertion order)
    kernels: list = []
    for kernel_id, kdata in raw_yaml["graph"].items():
        fn_name = kdata["function"]
        if kdata["output"] not in tensors:
            raise UnsupportedKernelReturn(kernel_id, fn_name, kdata.get("unsupported_return"))
        output_tensor = tensors[kdata["output"]]

        inputs: list = []
        for inp_entry in kdata.get("inputs", []):
            tensor_name = inp_entry["tensor"]
            from_k = inp_entry["from"]
            if tensor_name not in tensors:
                continue
            inputs.append(DSLInput(
                tensor=tensors[tensor_name],
                from_kernel=from_k,
            ))

        if kernel_info and fn_name in kernel_info:
            info = kernel_info[fn_name]
            loop_expr = info.get("loop_expr")
            param_names = info.get("param_names")
            loop_paths = info.get("loop_paths")
            loop_outer_statements = info.get("loop_outer_statements")
            loop_statement_classes = info.get("loop_statement_classes")
            loop_statement_classes_uncertain = info.get("loop_statement_classes_uncertain")
            loop_statement_grammar_violations = info.get("loop_statement_grammar_violations")
            loop_statement_class_gather = info.get("loop_statement_class_gather")
            loop_gather_controls = info.get("loop_gather_controls")
            loop_statement_class_lines = info.get("loop_statement_class_lines")
            loop_gather_control_lines = info.get("loop_gather_control_lines")
            loop_statement_class_operands = info.get("loop_statement_class_operands")
            source_text, source_sha256 = info.get("source_text"), info.get("source_sha256")
            module_source_sha256, signature = info.get("module_source_sha256"), info.get("signature")
            loop_statement_class_singleton = info.get("loop_statement_class_singleton")
            loop_singleton_reads = info.get("loop_singleton_reads")
            loop_singleton_controls = info.get("loop_singleton_controls")
            loop_singleton_outer_reads = info.get("loop_singleton_outer_reads")
            loop_singleton_control_lines = info.get("loop_singleton_control_lines")
            mult_params = info.get("mult_params")
            tensor_param_count = info.get("tensor_param_count")
            loop_paths_complete = bool(info.get("loop_paths_complete", False))
            loop_paths_omitted = info.get("loop_paths_omitted")
            loop_source_parsed = True
            local_tensor_dims = info.get("local_tensor_dims")
            loop_path_lines = info.get("loop_path_lines")
            loop_outer_lines = info.get("loop_outer_lines")
            returned_names = info.get("returned_names")
            loop_nest = info.get("loop_nest")
            name_bindings = info.get("name_bindings")
            returns = info.get("returns")
            dim_registrations = info.get("dim_registrations")
        else:
            loop_expr = loop_exprs.get(fn_name) if loop_exprs else None
            param_names = None
            loop_paths = None
            loop_outer_statements = None
            loop_statement_classes = None
            loop_statement_classes_uncertain = None
            loop_statement_grammar_violations = None
            loop_statement_class_gather = None
            loop_gather_controls = None
            loop_statement_class_lines = loop_gather_control_lines = loop_singleton_control_lines = None
            loop_statement_class_operands = None
            source_text = source_sha256 = module_source_sha256 = signature = None
            loop_statement_class_singleton = None
            loop_singleton_reads = None
            loop_singleton_controls = None
            loop_singleton_outer_reads = None
            mult_params = None
            tensor_param_count = None
            loop_paths_complete = False
            loop_paths_omitted = ["no parsed loop structure (legacy artifact)"]
            loop_source_parsed = False   # a builtin (gather/slice_at/...) or a legacy artifact: nothing was parsed, nothing is claimed
            local_tensor_dims = None
            loop_nest = name_bindings = returns = dim_registrations = None
            loop_path_lines = None
            loop_outer_lines = None
            returned_names = None

        kernels.append(DSLKernel(
            id=kernel_id,
            function_name=fn_name,
            inputs=inputs,
            output=output_tensor,
            loop_expr=loop_expr,
            param_names=param_names,
            loop_paths=loop_paths,
            loop_outer_statements=loop_outer_statements,
            loop_statement_classes=loop_statement_classes,
            loop_statement_classes_uncertain=loop_statement_classes_uncertain,
            loop_statement_grammar_violations=loop_statement_grammar_violations,
            loop_statement_class_gather=loop_statement_class_gather,
            loop_gather_controls=loop_gather_controls,
            loop_statement_class_lines=loop_statement_class_lines,
            loop_gather_control_lines=loop_gather_control_lines,
            loop_statement_class_operands=loop_statement_class_operands,
            source_text=source_text,
            source_sha256=source_sha256,
            module_source_sha256=module_source_sha256,
            signature=signature,
            loop_statement_class_singleton=loop_statement_class_singleton,
            loop_singleton_reads=loop_singleton_reads,
            loop_singleton_controls=loop_singleton_controls,
            loop_singleton_outer_reads=loop_singleton_outer_reads,
            loop_singleton_control_lines=loop_singleton_control_lines,
            mult_params=mult_params,
            tensor_param_count=tensor_param_count,
            inputs_in_signature_order=(kdata.get("input_order") == "signature"),
            loop_paths_complete=loop_paths_complete,
            loop_paths_omitted=loop_paths_omitted,
            loop_source_parsed=loop_source_parsed,
            local_tensor_dims=local_tensor_dims,
            loop_path_lines=loop_path_lines,
            loop_outer_lines=loop_outer_lines,
            returned_names=returned_names,
            loop_nest=loop_nest,
            name_bindings=name_bindings,
            returns=returns,
            dim_registrations=dim_registrations,
            registration_order=kdata.get("registration_order"),
            scalar_args=kdata.get("scalar_args") if isinstance(kdata.get("scalar_args"), dict) else None,
        ))

    return DSLGraph(model_name=model_name, tensors=tensors, kernels=kernels,
                    model_outputs=raw_yaml.get("model_outputs"),
                    model_output_provenance=raw_yaml.get("model_output_provenance"),
                    returned_outputs_unrecorded=raw_yaml.get("returned_outputs_unrecorded"))
