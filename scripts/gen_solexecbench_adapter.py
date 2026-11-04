#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate a SOLAR-compatible PyTorch adapter from a problem definition.

Scope: adapter generation for CAPPED verification workloads only (the
``capping-weights`` skill's ``--const-override`` path).  SOLAR ships no
benchmark runner, dataset, reference data or benchmark configuration; the
problem directory is user-supplied.

Given a problem directory containing ``definition.json`` (and optionally
``workload.jsonl``), this script emits a standalone ``.py`` file that follows
the SOLAR PyTorch contract (``get_model``, ``get_inputs``, and an
``nn.Module.forward`` that calls the reference ``run()``).

Usage::

    python gen_solexecbench_adapter.py <problem_dir> <output.py> [--workload-idx N]
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # import the checkout this script lives in
if str(REPO_ROOT) in sys.path:  # the checkout this script lives in must take precedence
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
if os.environ.get("SOLAR_ENGINE_SOURCE_PROBE"):  # provenance self-binding probe (tests/diagnostics)
    from solar.engine_source import probe_hook  # noqa: E402
    probe_hook()
from typing import Optional

_DTYPE_MAP = {
    "float64": "torch.float64",
    "float32": "torch.float32",
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float8_e4m3fn": "torch.float8_e4m3fn",
    "float8_e5m2": "torch.float8_e5m2",
    "int64": "torch.int64",
    "int32": "torch.int32",
    "int16": "torch.int16",
    "int8": "torch.int8",
    "bool": "torch.bool",
}

_VAR_AXIS_DEFAULTS = {
    "batch_size": 2,
    "batch": 2,
    "seq_len": 64,
    "seq_len_q": 64,
    "seq_len_kv": 64,
    "M": 64,
    "N": 64,
    "K": 64,
}

# Upper bound for var axes not listed in _VAR_AXIS_DEFAULTS.  When a
# workload provides a large concrete value (e.g. num_pages=9316), we
# clamp it so the adapter always generates verification-friendly sizes.
_VAR_AXIS_MAX = 64


def _resolve_axes(defn: dict, workload: Optional[dict] = None,
                  const_overrides: Optional[dict] = None) -> dict:
    """Resolve all axes to concrete integer values.

    *const_overrides* ({axis: value}) shrinks ``const`` (architectural/weight)
    axes for verification-scale runs — e.g. hidden 16384 -> 2048, or 512
    experts -> 4.  Sound for VERIFICATION because the reference ``run()`` is
    shape-generic and adapter weights + reference shrink consistently; the
    resulting SOL/trace numbers reflect the capped model, so re-trace at true
    shapes (shape-only mode) for analysis.  Keep divisibility constraints
    (e.g. hidden = heads x head_dim) when choosing values.
    """
    const_overrides = const_overrides or {}
    axes: dict[str, int] = {}
    for name, spec in defn["axes"].items():
        if spec["type"] == "const":
            if name in const_overrides:
                axes[name] = int(const_overrides[name])
                print(f"  [const-override] {name}: {spec['value']} -> {axes[name]}",
                      file=sys.stderr)
            else:
                axes[name] = spec["value"]
        elif spec["type"] == "var":
            if workload and name in workload.get("axes", {}):
                # Prefer workload values — they are internally consistent
                # with each other.  Only clamp truly large ones.
                axes[name] = min(workload["axes"][name], _VAR_AXIS_MAX)
            else:
                axes[name] = _VAR_AXIS_DEFAULTS.get(name, 4)
    for name, spec in defn["axes"].items():
        if spec["type"] == "expr":
            axes[name] = int(eval(spec["expression"], {"__builtins__": {}}, axes))
    return axes


def _custom_input_names(workload: Optional[dict]) -> set:
    """Return input names marked as ``"type": "custom"`` in the workload."""
    if not workload:
        return set()
    result = set()
    for name, spec in workload.get("inputs", {}).items():
        if isinstance(spec, dict) and spec.get("type") == "custom":
            result.add(name)
    return result


def _llm_generate_get_inputs(
    ref_code: str,
    activation_specs: list[tuple[str, dict]],
    axes: dict,
) -> Optional[str]:
    """Ask the LLM to write the body of get_inputs() that satisfies constraints.

    Returns the function body lines (indented with 4 spaces) or None on failure.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from solar.einsum_lib.agent import TranslationAgent
        agent = TranslationAgent(max_retries=1, verify=False, thinking=False)
    except Exception:
        return None

    inputs_desc = []
    for name, spec in activation_specs:
        shape_expr = spec["shape"]
        resolved = []
        for d in shape_expr:
            if d in axes:
                resolved.append(f"{d}={axes[d]}")
            else:
                resolved.append(str(d))
        inputs_desc.append(
            f"  - {name}: dtype={spec['dtype']}, shape=({', '.join(resolved)})"
        )

    act_names = [n for n, _ in activation_specs]
    return_line = f"return [{', '.join(act_names)}]"

    prompt = (
        "You are generating a `get_inputs()` function for a PyTorch model.\n\n"
        "The reference code (which will consume these inputs) is:\n"
        f"```python\n{ref_code}\n```\n\n"
        f"Resolved axis values: {json.dumps(axes)}\n\n"
        f"Return EXACTLY these {len(act_names)} tensors in this order:\n"
        + "\n".join(inputs_desc) + "\n\n"
        "DO NOT include any other inputs (scalars like sm_scale, or model "
        "attributes are handled separately — only return the tensors above).\n\n"
        "IMPORTANT: Read the reference code carefully. Look for:\n"
        "- assert statements that constrain input values\n"
        "- indexing operations (tensor[indices]) — indices must be in range\n"
        "- indptr/indices patterns (CSR format) — indptr must be sorted, "
        "indptr[-1] must equal len(indices)\n"
        "- any value-dependent behavior\n\n"
        "Generate ONLY the function body (no `def` line). The `axes` dict is "
        "already defined. Use `device = torch.device('cpu')`. "
        f"End with: `{return_line}`\n\n"
        "Example format:\n"
        "```python\n"
        "    device = torch.device('cpu')\n"
        "    x = torch.randn((axes['batch'], axes['dim']), device=device)\n"
        "    indices = torch.randint(0, axes['num_pages'], (axes['n'],), "
        "dtype=torch.int32, device=device)\n"
        "    return [x, indices]\n"
        "```\n\n"
        "Return ONLY the Python code block, nothing else."
    )

    try:
        msg = agent._complete([{"role": "user", "content": prompt}])
        text = msg.content if hasattr(msg, "content") else str(msg)
        # Extract code from markdown block
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1)
        lines = text.strip().split("\n")
        # Ensure proper indentation
        body = []
        for line in lines:
            stripped = line.rstrip()
            if stripped:
                if not stripped.startswith("    "):
                    stripped = "    " + stripped
                body.append(stripped)
            else:
                body.append("")
        result = "\n".join(body)

        # Validate: the return statement must have exactly the right names.
        expected = f"return [{', '.join(n for n, _ in activation_specs)}]"
        if expected not in result.replace(" ", "").replace("\n", ""):
            # Force correct return line
            body = [l for l in body if not l.strip().startswith("return ")]
            body.append(f"    {expected}")
            result = "\n".join(body)
        return result
    except Exception as exc:
        print(f"  [WARN] LLM get_inputs generation failed: {exc}",
              file=sys.stderr)
        return None


def generate_adapter(problem_dir: str, output_path: str,
                     workload_idx: int = 0,
                     const_overrides: Optional[dict] = None) -> None:
    problem_dir = Path(problem_dir)
    defn = json.loads((problem_dir / "definition.json").read_text())

    workload = None
    wkl_path = problem_dir / "workload.jsonl"
    if wkl_path.exists():
        lines = wkl_path.read_text().strip().split("\n")
        if workload_idx < len(lines):
            workload = json.loads(lines[workload_idx])

    axes = _resolve_axes(defn, workload, const_overrides)
    ref_code = defn["reference"]
    custom_ep = defn.get("custom_inputs_entrypoint")

    if custom_ep:
        ref_code = re.sub(
            rf"\bdef\s+{re.escape(custom_ep)}\s*\(",
            "def _custom_gen_inputs(",
            ref_code,
        )

    input_names = list(defn["inputs"].keys())
    output_names = list(defn["outputs"].keys())
    custom_names = _custom_input_names(workload)

    # Classify axes as var or const for deciding which inputs are weights.
    var_axes = {name for name, spec in defn["axes"].items()
                if spec["type"] == "var"}
    # expr axes that depend on a var axis are also variable.
    for name, spec in defn["axes"].items():
        if spec["type"] == "expr":
            if any(v in spec["expression"] for v in var_axes):
                var_axes.add(name)

    def _resolve_dim(d):
        if d in axes:
            return axes[d]
        try:
            return int(d)
        except ValueError:
            return int(eval(d, {"__builtins__": {}}, axes))

    def _is_weight(spec):
        """An input is a weight if it has a shape and NONE of its dims are var."""
        shape = spec["shape"]
        if shape is None or len(shape) == 0:
            return False
        return not any(d in var_axes for d in shape)

    def _tensor_init(name, spec):
        """Return a Python expression that creates a tensor for *spec*."""
        shape = spec["shape"]
        dtype = spec["dtype"]
        resolved = tuple(_resolve_dim(d) for d in shape)
        td = _DTYPE_MAP.get(dtype, "torch.float32")
        if dtype == "bool":
            return (f"torch.randint(0, 2, {resolved},"
                    f" dtype={td}).bool()")
        if dtype.startswith("int"):
            return (f"torch.randint(0, 10, {resolved},"
                    f" dtype={td})")
        if dtype.startswith("float8") or dtype.startswith("float4"):
            return (f"torch.randn({resolved},"
                    f" dtype=torch.float32).to({td})")
        return f"torch.randn({resolved}, dtype={td})"

    # Partition inputs into weights (all-const dims → nn.Parameter) and
    # activations (has var dims → get_inputs).
    weight_names = []
    activation_names = []
    scalar_names = []
    bool_weight_names = []
    for name in input_names:
        spec = defn["inputs"][name]
        if spec["shape"] is None:
            scalar_names.append(name)
        elif spec["dtype"] == "bool" and _is_weight(spec):
            # Bool tensors can't be nn.Parameter; store as buffer.
            bool_weight_names.append(name)
        elif _is_weight(spec):
            weight_names.append(name)
        else:
            activation_names.append(name)

    # ----- build Model class -----
    init_lines = []
    for name in weight_names:
        spec = defn["inputs"][name]
        init_lines.append(
            f"        self.{name} = torch.nn.Parameter("
            f"{_tensor_init(name, spec)}, requires_grad=False)"
        )
    # Bool tensors: register as buffers (not nn.Parameter, which rejects bool).
    # Prefix with '_b_' to avoid clashing with nn.Module built-in attrs
    # like 'training'.
    _RESERVED_MODULE_ATTRS = {"training"}
    for name in bool_weight_names:
        spec = defn["inputs"][name]
        shape = spec["shape"]
        resolved = tuple(_resolve_dim(d) for d in shape)
        buf_name = f"_b_{name}" if name in _RESERVED_MODULE_ATTRS else name
        init_lines.append(
            f"        self.register_buffer('{buf_name}', torch.randint(0, 2,"
            f" {resolved}, dtype=torch.bool))"
        )
    # Scalars stored as plain attributes
    for name in scalar_names:
        spec = defn["inputs"][name]
        dtype = spec["dtype"]
        if dtype in ("float32", "float64", "float16", "bfloat16"):
            init_lines.append(f"        self.{name} = 0.1")
        elif dtype.startswith("int"):
            init_lines.append(f"        self.{name} = 1")
        elif dtype == "bool":
            init_lines.append(f"        self.{name} = True")

    forward_params = ", ".join(activation_names)

    def _self_ref(name):
        if name in activation_names:
            return name
        if name in bool_weight_names and name in _RESERVED_MODULE_ATTRS:
            return f"self._b_{name}"
        return f"self.{name}"

    run_args = ", ".join(_self_ref(name) for name in input_names)

    model_lines = []
    model_lines.append("class Model(torch.nn.Module):")
    if init_lines:
        model_lines.append("    def __init__(self):")
        model_lines.append("        super().__init__()")
        model_lines.extend(init_lines)
        model_lines.append("")
    model_lines.append(f"    def forward(self, {forward_params}):")
    model_lines.append(f"        return run({run_args})")

    # ----- build get_inputs() body (activations only) -----
    # Always ask the LLM to generate get_inputs() so that structural
    # constraints (sorted indptr, valid indices, etc.) are respected.
    # For float-only problems the LLM will produce simple torch.randn()
    # calls; for integer/structured inputs it will add the necessary
    # constraint logic.  Falls back to mechanical generation on failure.
    llm_body = None
    if not custom_ep and activation_names:
        act_specs = [(n, defn["inputs"][n]) for n in activation_names]
        llm_body = _llm_generate_get_inputs(ref_code, act_specs, axes)

    axes_init = f"    axes = {axes}"

    if llm_body is not None:
        gi = [axes_init, llm_body]
    else:
        gi = []
        gi.append(axes_init)
        gi.append("    device = torch.device('cpu')")

        if custom_ep:
            gi.append("    _custom = _custom_gen_inputs(axes, device)")

        for name in activation_names:
            spec = defn["inputs"][name]
            shape = spec["shape"]
            dtype = spec["dtype"]
            td = _DTYPE_MAP.get(dtype, "torch.float32")
            resolved = tuple(_resolve_dim(d) for d in shape)
            shape_expr = str(resolved)

            if custom_ep:
                gi.append(f"    {name} = _custom['{name}']")
                continue

            if dtype == "bool":
                gi.append(
                    f"    {name} = torch.randint(0, 2, {shape_expr},"
                    f" dtype={td}, device=device).bool()"
                )
            elif dtype.startswith("int"):
                gi.append(
                    f"    {name} = torch.randint(0, 10, {shape_expr},"
                    f" dtype={td}, device=device)"
                )
            elif dtype.startswith("float8") or dtype.startswith("float4"):
                gi.append(
                    f"    {name} = torch.randn({shape_expr},"
                    f" dtype=torch.float32, device=device).to({td})"
                )
            else:
                gi.append(
                    f"    {name} = torch.randn({shape_expr},"
                    f" dtype={td}, device=device)"
                )

        gi.append(f"    return [{', '.join(activation_names)}]")

    adapter = (
        f"# Auto-generated SOLAR adapter for SOL-ExecBench: {defn['name']}\n"
        f"# Source: {problem_dir / 'definition.json'}\n"
        f"\n"
        f"{ref_code}\n"
        f"\n"
        f"\n"
        f"# ---- SOLAR PyTorch contract ----\n"
        + "\n".join(model_lines)
        + "\n\n\n"
        f"def get_model():\n"
        f"    return Model().eval()\n"
        f"\n"
        f"\n"
        f"def get_inputs():\n"
        + "\n".join(gi)
        + "\n"
    )

    Path(output_path).write_text(adapter, encoding="utf-8")

    has_custom = bool(custom_ep) or (llm_body is not None)
    return has_custom


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_dir", help="Path to the problem directory")
    parser.add_argument("output", help="Path for the generated .py adapter")
    parser.add_argument(
        "--workload-idx",
        type=int,
        default=0,
        help="Which line of workload.jsonl to use for var axis values (default: 0)",
    )
    parser.add_argument(
        "--const-override",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help="Shrink a const (architectural/weight) axis for verification-scale "
             "runs, e.g. --const-override hidden_size=2048 --const-override "
             "num_experts=4. Repeatable. Reference and weights shrink "
             "consistently; keep divisibility (hidden = heads x head_dim). "
             "SOL/trace numbers then reflect the CAPPED model.",
    )
    args = parser.parse_args()
    overrides = {}
    for item in args.const_override:
        k, _, v = item.partition("=")
        if not v:
            parser.error(f"--const-override needs AXIS=VALUE, got: {item}")
        overrides[k.strip()] = int(v)
    has_custom = generate_adapter(args.problem_dir, args.output, args.workload_idx,
                                  const_overrides=overrides or None)
    if has_custom:
        print("HAS_CUSTOM_INPUTS")
    else:
        print("STANDARD_INPUTS")
