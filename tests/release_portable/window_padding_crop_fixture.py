# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static helpers for the portable window padding/crop controls.

Nothing here executes the DSL.  The fixture program is parsed with the public
``ast_parser`` and converter stand-ins are assembled from hand-derived
call-site shapes, mirroring what a trace would record (declared signature
dimension names bound to the shapes of that call) without tracing.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from solar.einsum_dsl import einsum_converter as ec
from solar.einsum_dsl.ast_parser import _int_size_spec, extract_kernel_info

HERE = Path(__file__).resolve().parent
DSL_PATH = HERE / "fixtures" / "window_padding_crop_einsum_lib.py"

# Hand-chosen fixture constants (must match the DSL module constants).
WS = 3
C = 6
F = 12


def geometry(b: int, h: int, w: int, ws: int = WS) -> dict:
    """Hand-derived window geometry: ceil-divided counts and padded extents."""
    nh = -(-h // ws)
    nw = -(-w // ws)
    return {
        "nh": nh,
        "nw": nw,
        "windows": b * nh * nw,          # padded windows across the batch
        "padded_h": nh * ws,
        "padded_w": nw * ws,
        "tokens": ws * ws,
        "partial": (h % ws != 0) or (w % ws != 0),
    }


def _function(source: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(source).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def _tensor_dim_names(annotation) -> list | None:
    """``Tensor("B,H,W,C", ...)`` -> ["B", "H", "W", "C"]; None for non-tensors."""
    if (isinstance(annotation, ast.Call) and isinstance(annotation.func, ast.Name)
            and annotation.func.id == "Tensor" and annotation.args
            and isinstance(annotation.args[0], ast.Constant)):
        return [d.strip() for d in str(annotation.args[0].value).split(",")]
    return None


def signature_dims(name: str, source: Path = DSL_PATH) -> tuple:
    """(ordered {tensor_param: dim names}, output dim names) from the annotations."""
    func = _function(source, name)
    params = {}
    for arg in func.args.args:
        dims = _tensor_dim_names(arg.annotation)
        if dims is not None:
            params[arg.arg] = dims
    return params, _tensor_dim_names(func.returns)


def _local_consts(source: Path, func: ast.FunctionDef) -> dict:
    """Module int constants plus function-local size bindings (as ast_parser does)."""
    tree = ast.parse(Path(source).read_text())
    consts = {}
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, int)):
            consts[stmt.targets[0].id] = stmt.value.value
    for stmt in ast.walk(func):
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            spec = _int_size_spec(stmt.value, consts)
            if spec is not None:
                consts[stmt.targets[0].id] = spec
    return consts


def declared_output_shape(name: str, param_shapes: dict, source: Path = DSL_PATH) -> list:
    """Evaluate the kernel's ``Tensor(DIM=..., ...)`` constructor at a call site.

    Each dimension keyword is resolved with the public size-spec machinery
    (``_int_size_spec`` then ``_eval_size_spec``); an unresolvable dimension
    yields ``None`` in that position.
    """
    func = _function(source, name)
    consts = _local_consts(source, func)
    for stmt in ast.walk(func):
        if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "Tensor" and stmt.value.keywords):
            shape = []
            for kw in stmt.value.keywords:
                if kw.arg in ("dtype", "role", "sparsity"):
                    continue
                spec = _int_size_spec(kw.value, consts)
                # The hand-built shapes are keyed by parameter name, i.e. the
                # input/parameter correspondence the converter requires before
                # it evaluates arithmetic specs (the internal converter); pass them as
                # ``binop_shapes`` as the converter does for a certified call.
                shape.append(ec._eval_size_spec(spec, param_shapes, param_shapes) if spec is not None else None)
            return shape
    raise ValueError(f"{name}: no Tensor(...) output constructor found")


def kernel_stand_in(name: str, param_shapes: dict, *, info: dict | None = None,
                    source: Path = DSL_PATH, output_shape: list | None = None):
    """Converter kernel stand-in for one call site of a fixture kernel.

    ``param_shapes`` maps each tensor parameter to its call-site shape; the
    declared signature dimension names are bound to those shapes exactly as a
    trace binds them, and the output is the statically declared shape.
    """
    info = info if info is not None else extract_kernel_info(source)[name]
    params, out_dims = signature_dims(name, source)
    inputs = []
    for pname, dims in params.items():
        shape = list(param_shapes[pname])
        if len(shape) != len(dims):
            raise ValueError(f"{name}.{pname}: shape {shape} does not match dims {dims}")
        inputs.append(SimpleNamespace(tensor=SimpleNamespace(shape=shape, dims=dict(zip(dims, shape)))))
    shape = list(output_shape) if output_shape is not None else declared_output_shape(name, param_shapes, source)
    out = SimpleNamespace(shape=shape, dims={d: s for d, s in zip(out_dims or [], shape) if isinstance(s, int)})
    # A trace records tensor inputs in signature order and the parser
    # certifies the leading Tensor-parameter count; the converter evaluates
    # arithmetic size specs only under that correspondence (the internal converter), so the
    # stand-in carries both markers exactly as the assembler sets them.
    return SimpleNamespace(id=f"{name}#0", loop_paths=info["loop_paths"], loop_expr=info["loop_expr"],
                           param_names=info["param_names"], inputs=inputs, output=out,
                           tensor_param_count=info.get("tensor_param_count", len(params)),
                           inputs_in_signature_order=True)
