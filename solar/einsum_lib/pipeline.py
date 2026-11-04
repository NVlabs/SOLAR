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

"""Translation pipeline: PyTorch model → solar.einsum_lib code.

Shared by scripts/to_einsum_lib.py and solar-toDSL-model CLI.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from solar.einsum_lib.agent import TranslationAgent, TranslationResult
import logging

logger = logging.getLogger(__name__)

_TORCH_DTYPE_TO_SOLAR = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.bfloat16": "bf16",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.bool": "bool",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e5m2": "fp8",
}

_INTEGER_DTYPES = frozenset({"int8", "int16", "int32", "int64", "bool"})
_SKIP_INT_ATTRS = frozenset({"training", "T_destination", "dump_patches"})
_CAPTURE_SEED = 42

SUPPORTED_SOURCE_TYPES = ["pytorch", "raw", "jax", "triton", "cuda"]


def _torch_dtype_str(dtype) -> str:
    return _TORCH_DTYPE_TO_SOLAR.get(str(dtype), "fp32")


def _is_integer_dtype(dtype: str) -> bool:
    return dtype in _INTEGER_DTYPES


_BYTES_PER_DTYPE = {
    np.dtype(np.float64): 8, np.dtype(np.float32): 4,
    np.dtype(np.int64): 8, np.dtype(np.int32): 4, np.dtype(np.bool_): 1,
}


def _estimate_allocation_bytes(shapes: dict, dtypes: dict, use_fp64: bool) -> int:
    """Exact byte count for all tensors that will be allocated."""
    float_dt = np.dtype(np.float64 if use_fp64 else np.float32)
    total = 0
    for key, shape in shapes.items():
        elems = 1
        for d in shape:
            elems *= d
        solar_dt = dtypes.get(key, "")
        if solar_dt in _INTEGER_DTYPES:
            bpe = 8  # int64
        else:
            bpe = _BYTES_PER_DTYPE.get(float_dt, 4)
        total += elems * bpe
    return total


_MEM_SAFETY_FACTOR = 0.8


def _get_available_memory_bytes() -> Optional[int]:
    """Return available system RAM * safety factor, or ``None`` if psutil is missing."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available * _MEM_SAFETY_FACTOR)


def _check_memory(shapes: dict, dtypes: dict, use_fp64: bool,
                   mem_budget_bytes: Optional[int] = None):
    """Raise ``MemoryError`` if estimated allocation exceeds the budget.

    Uses ``min(mem_budget_bytes, system_available)`` so the user can
    tighten the limit but never exceed real system RAM.
    """
    system = _get_available_memory_bytes()
    if mem_budget_bytes is not None:
        limit = min(mem_budget_bytes, system) if system is not None else mem_budget_bytes
    elif system is not None:
        limit = system
    else:
        return
    needed = _estimate_allocation_bytes(shapes, dtypes, use_fp64)
    if needed > limit:
        raise MemoryError(
            f"Estimated {needed / 1e9:.1f} GB needed for tensor allocation, "
            f"but budget is {limit / 1e9:.1f} GB"
        )


def _get_effective_budget_bytes(mem_budget_bytes: Optional[int]) -> Optional[int]:
    """Return the effective capping budget in bytes.

    Takes the tighter of the user-supplied *mem_budget_bytes* and the
    actual available system RAM (via psutil).  Returns ``None`` only
    when psutil is unavailable and no explicit budget was given.
    """
    system = _get_available_memory_bytes()
    if mem_budget_bytes is not None:
        if system is not None:
            return min(mem_budget_bytes, system)
        return mem_budget_bytes
    return system


def shape_derivable_config_fields(config_dict: dict, input_shapes: dict) -> dict:
    """Return the config fields that are safe to shrink for verification.

    A model that reads its dimensions from a module-level config rather than
    from the shapes of its arguments can only be shrunk through that config.
    Shrinking it is safe only for fields the translated IR does not have to
    hard-code: the IR is written against the *original* source and then run on
    capped tensors, so any dimension it could not recover from an input shape
    stays baked in at full size and stops matching.

    A field's value appearing as a whole dimension of some input is the
    available proxy for "recoverable".  It excludes dimensions that appear only
    folded into a product -- a head count inside ``H * head_dim``, which the IR
    must therefore hard-code -- and it excludes non-dimensional numbers such as
    an RoPE base or an epsilon, which match no extent at all.
    """
    dims = {d for shape in input_shapes.values() for d in shape}
    return {
        name: value
        for name, value in config_dict.items()
        if isinstance(value, int) and not isinstance(value, bool) and value in dims
    }


def sanitize_config_overrides(config_dict: dict, overrides: dict,
                             cappable: dict, debug: bool = False) -> dict:
    """Drop proposed config overrides that are unsafe to apply.

    Keeps only shrinks of fields :func:`shape_derivable_config_fields` cleared,
    so a stray proposal degrades to less capping rather than to a translation
    that cannot verify.
    """
    clean = {}
    for name, value in overrides.items():
        if name not in cappable:
            if debug:
                reason = ("not a shape-derivable dimension"
                          if name in config_dict else "not a config field")
                logger.debug(f"Ignoring config override {name}={value} ({reason})")
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            if debug:
                logger.debug(f"Ignoring config override {name}={value} "
                      f"(not a positive integer)")
            continue
        if value >= cappable[name]:
            if debug:
                logger.debug(f"Ignoring config override {name}={value} "
                      f"(not smaller than {cappable[name]})")
            continue
        clean[name] = value
    return clean


def _cap_with_llm(
    source_code: str,
    input_shapes: dict,
    config_dict: Optional[dict] = None,
    api_key: Optional[str] = None,
    api_key_file: Optional[str] = None,
    debug: bool = False,
    error_context: Optional[str] = None,
    mem_budget_bytes: Optional[int] = None,
    use_fp64: bool = False,
    config_kind: str = "huggingface",
    cappable_fields: Optional[list] = None,
):
    """Ask the LLM to propose small sizes for fast verification.

    Returns ``(config_overrides, verify_shapes)`` where:

    *config_overrides* is a dict of HuggingFace config field overrides
    (empty when the model has no HF config or the LLM chose not to
    change any fields).

    *verify_shapes* maps every key in *input_shapes* to a (possibly
    unchanged) shape tuple.
    """
    from solar.einsum_lib.agent import TranslationAgent

    agent = TranslationAgent(
        api_key=api_key, api_key_file=api_key_file,
        max_retries=1, verify=False, thinking=False,
    )

    shapes_str = json.dumps({k: list(v) for k, v in input_shapes.items()}, indent=2)

    bpe = 8 if use_fp64 else 4
    weight_bytes = 0
    for name, shape in input_shapes.items():
        if not name.startswith("x"):
            elems = 1
            for d in shape:
                elems *= d
            weight_bytes += elems * bpe

    budget = _get_effective_budget_bytes(mem_budget_bytes)
    budget_line = ""
    if budget is not None:
        activation_budget = max(budget - weight_bytes, 0)
        def _fmt_bytes(b: int) -> str:
            if b >= 1e9:
                return f"{b / 1e9:.2f} GB"
            if b >= 1e6:
                return f"{b / 1e6:.1f} MB"
            return f"{b:,} bytes"
        budget_line = (
            f"- Memory budget: weights/buffers consume ~{_fmt_bytes(weight_bytes)}. "
            f"You have ~{_fmt_bytes(activation_budget)} for activation tensors "
            f"(each element is {bpe} bytes). Keep total activation memory under "
            f"{_fmt_bytes(activation_budget)}.\n"
        )
        if debug:
            logger.debug(f"Memory budget: total={_fmt_bytes(budget)}, "
                  f"weights={_fmt_bytes(weight_bytes)}, "
                  f"activations={_fmt_bytes(activation_budget)}")

    if config_dict is not None and config_kind == "module":
        config_str = json.dumps(config_dict, indent=2, default=str)
        allowed = ", ".join(cappable_fields or [])
        config_section = (
            f"Module-level config dict (the model reads its dimensions from "
            f"this, not from its arguments):\n```json\n{config_str}\n```\n\n"
        )
        config_instructions = (
            '"config_overrides": Shrinking the input tensors alone cannot '
            "shrink this model, because it takes its dimensions from the config "
            "dict above. Propose smaller values for these fields, and these "
            f"only: {allowed}. Every other field must be left alone -- the "
            "remaining dimensions are baked into the translation and would stop "
            "matching. Each value must be strictly smaller than the current "
            "one. Aim for batch 1, sequence length <= 128, and embedding, "
            "hidden, and rank dimensions in the 64-256 range. Respect any "
            "divisibility the code relies on (a dimension the model splits "
            "across heads must stay a multiple of the head count, which you are "
            "not shrinking). Only include fields you are changing.\n\n"
        )
    elif config_dict is not None:
        config_str = json.dumps(config_dict, indent=2, default=str)
        config_section = (
            f"HuggingFace model config:\n```json\n{config_str}\n```\n\n"
        )
        config_instructions = (
            '"config_overrides": If the HuggingFace config is present above, '
            "propose a miniature config for fast verification. Shrink dimensional "
            "fields (hidden_size, intermediate_size, vocab_size, "
            "num_attention_heads, max_position_embeddings, and any "
            "model-specific equivalents). Do NOT change layer count fields "
            "(num_hidden_layers, num_layers, n_layer). Do NOT disable "
            "architectural features (e.g. do not change axial_pos_embds, "
            "attn_layers, or similar structural flags). "
            "Keep values mutually "
            "consistent (e.g. hidden_size must be divisible by "
            "num_attention_heads). Typical targets: hidden ~32-64, "
            "intermediate ~64-128, vocab ~128-256, heads 2-4, positions "
            "~32-64. Only include fields you are changing. "
            "If not a HuggingFace model, set to {}.\n\n"
        )
    else:
        config_section = ""
        config_instructions = '"config_overrides": Set to {} (not a HuggingFace model).\n\n'

    prompt = (
        f"I have a model. I need to run it at SMALL sizes for "
        f"fast verification.\n\n"
        f"Source code:\n```python\n{source_code}\n```\n\n"
        f"{config_section}"
        f"Input tensor shapes (activations start with 'x', rest are "
        f"weights/buffers):\n```json\n{shapes_str}\n```\n\n"
        f"Return a JSON object with two keys:\n\n"
        f"{config_instructions}"
        f'"input_shapes": For activation inputs (keys starting with \'x\'), '
        f"propose small shapes for fast verification. "
        f"If config_overrides will rebuild the model with smaller weights, "
        f"make sure inputs are valid for the new config "
        f"(e.g. seq_len <= new max_position_embeddings, "
        f"token IDs < new vocab_size). "
        f"If config_overrides is empty, only shrink free dimensions "
        f"(batch, sequence, spatial) and keep architectural dims tied to "
        f"weights unchanged. "
        f"Include only tensors whose shapes change.\n\n"
        f"Rules:\n"
        f"- Keep batch size at 1\n"
        f"- Keep the same number of dimensions per tensor\n"
        f"{budget_line}"
        f"- Keep dimensions small enough that verification completes in "
        f"seconds\n\n"
    )

    if error_context:
        prompt += (
            f"IMPORTANT: A previous proposal caused this error:\n"
            f"{error_context}\n"
            f"Fix the proposal to avoid this error.\n\n"
        )

    prompt += "Return ONLY the JSON object. No explanation."

    messages = [
        {"role": "system", "content": "You produce valid JSON. No markdown fences."},
        {"role": "user", "content": prompt},
    ]

    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0].strip()
        idx = raw.find("{")
        if idx > 0:
            raw = raw[idx:]
        ridx = raw.rfind("}")
        if ridx >= 0:
            raw = raw[: ridx + 1]
        return json.loads(raw)

    max_attempts = 2
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            raw = agent._call_llm(messages)
            if debug:
                logger.debug(f"Capping LLM raw response:\n{raw}")
            result = _parse_json(raw)

            config_overrides = result.get("config_overrides", {})
            if not isinstance(config_overrides, dict):
                config_overrides = {}

            shape_changes = result.get("input_shapes", {})
            if not isinstance(shape_changes, dict):
                shape_changes = {}

            verify_shapes = {}
            for name, shape in input_shapes.items():
                if name in shape_changes:
                    proposed = tuple(shape_changes[name])
                    if (len(proposed) == len(shape)
                            and all(isinstance(d, int) and d > 0 for d in proposed)):
                        verify_shapes[name] = proposed
                        continue
                verify_shapes[name] = shape

            # Propagate dimension changes to tensors the LLM didn't
            # mention, keyed by (position, original_value) so that only
            # dimensions with the exact same original size at the same
            # rank position are updated (e.g. batch_size at dim 0).
            dim_map: dict = {}
            for name, new_shape in verify_shapes.items():
                old_shape = input_shapes[name]
                for pos, (old_d, new_d) in enumerate(
                        zip(old_shape, new_shape)):
                    if old_d != new_d:
                        dim_map[(pos, old_d)] = new_d
            if dim_map:
                for name in list(verify_shapes):
                    old_shape = input_shapes[name]
                    cur = verify_shapes[name]
                    if cur != old_shape:
                        continue
                    patched = list(cur)
                    changed = False
                    for pos, d in enumerate(cur):
                        if (pos, d) in dim_map:
                            patched[pos] = dim_map[(pos, d)]
                            changed = True
                    if changed:
                        verify_shapes[name] = tuple(patched)

            if config_overrides and debug:
                logger.debug(f"LLM config overrides: {config_overrides}")
            changed_shapes = {k: v for k, v in verify_shapes.items()
                              if v != input_shapes[k]}
            if changed_shapes:
                logger.info(f"LLM-proposed shape changes: {changed_shapes}")

            return config_overrides, verify_shapes, changed_shapes

        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                print(f"  [WARN] LLM capping attempt {attempt + 1} failed ({exc}), retrying...")
                messages.append({"role": "assistant", "content": raw if 'raw' in dir() else ""})
                messages.append({"role": "user", "content":
                    "That was not valid JSON. Return ONLY a JSON object with "
                    "\"config_overrides\" and \"input_shapes\" keys."
                })

    raise RuntimeError(
        f"LLM capping failed after {max_attempts} attempts: {last_exc}"
    )


def scale_shapes_with_llm(source_code, input_shapes, **kwargs):
    """Backward-compatible wrapper around :func:`_cap_with_llm`.

    Returns only the verify_shapes dict (ignores config_overrides).
    """
    _, verify_shapes, _ = _cap_with_llm(source_code, input_shapes, **kwargs)
    return verify_shapes


def _try_build_capped_model(module, config_overrides: dict,
                            debug: bool = False,
                            input_shape_overrides: Optional[dict] = None):
    """Build a miniature HuggingFace model using LLM-proposed config overrides.

    Applies *config_overrides* to the module's HuggingFace config, patches
    ``from_pretrained`` to call ``from_config`` instead, and constructs
    both a capped model (for verification) and an original-config model
    (for shape extraction, with random weights).

    *input_shape_overrides* — LLM-proposed activation shape changes (e.g.
    ``{'x': (1, 64)}``).  If the first dimension of ``'x'`` differs from
    ``module.batch_size``, the module attribute is updated so that
    ``get_inputs()`` produces inputs with the correct batch size.

    Returns ``(capped_model, capped_inputs, original_model)`` on success,
    or ``None`` on failure.
    """
    import copy

    config = getattr(module, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        return None

    original_config = copy.deepcopy(config)

    capped_config = copy.deepcopy(config)
    for field, value in config_overrides.items():
        orig = getattr(config, field, None)
        if isinstance(orig, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(capped_config, field, value)
        if debug:
            old = getattr(original_config, field, "N/A")
            logger.debug(f"Config cap {field}: {old} -> {value}")

    try:
        import transformers
    except ImportError:
        return None

    _pretrained_classes = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForMaskedLM",
                     "AutoModelForSequenceClassification", "AutoModel",
                     "ReformerModelWithLMHead"):
        cls = getattr(transformers, cls_name, None)
        if cls is not None and hasattr(cls, "from_pretrained"):
            _pretrained_classes.append(cls)

    originals = {}
    for cls in _pretrained_classes:
        originals[cls] = cls.from_pretrained

    def _make_shim(target_cls, cfg):
        def _shim(*args, **kwargs):
            return target_cls.from_config(cfg)
        return _shim

    def _build_model_with_config(cfg):
        for cls in _pretrained_classes:
            cls.from_pretrained = _make_shim(cls, cfg)
        try:
            model, inputs = get_pytorch_model_and_inputs(module)
        finally:
            for cls in _pretrained_classes:
                cls.from_pretrained = originals[cls]
        return model, inputs

    saved_attrs = {}
    for attr in ("vocab_size", "sequence_length", "batch_size"):
        if hasattr(module, attr):
            saved_attrs[attr] = getattr(module, attr)

    def _restore_module():
        module.config = original_config
        for attr, val in saved_attrs.items():
            setattr(module, attr, val)

    # Adjust module-level variables for get_inputs() validity
    module.config = capped_config
    if hasattr(module, "vocab_size") and hasattr(capped_config, "vocab_size"):
        module.vocab_size = capped_config.vocab_size
    max_pos = None
    for f in ("max_position_embeddings", "n_positions"):
        if hasattr(capped_config, f):
            max_pos = getattr(capped_config, f)
            break
    if max_pos is not None and hasattr(module, "sequence_length"):
        if getattr(module, "sequence_length") > max_pos:
            module.sequence_length = max_pos

    if input_shape_overrides and hasattr(module, "batch_size"):
        x_shape = input_shape_overrides.get("x")
        if x_shape and len(x_shape) > 0:
            module.batch_size = x_shape[0]

    try:
        capped_model, capped_inputs = _build_model_with_config(capped_config)
    except Exception as exc:
        if debug:
            logger.debug(f"Config capping failed: {exc}")
        _restore_module()
        return None

    _restore_module()

    try:
        original_model, _ = _build_model_with_config(original_config)
    except Exception:
        original_model = None

    return capped_model, capped_inputs, original_model


def load_pytorch_module(file_path: str):
    """Dynamically import a .py file expected to contain a PyTorch model.

    Loaded from the CURRENT source bytes (bytecode caches bypassed) so the
    verification runner can bind the executed model code to the hashed
    source — see solar.einsum_lib.source_load.
    """
    from solar.einsum_lib.source_load import load_source_module

    return load_source_module(file_path, "_model_module")


def get_pytorch_model_and_inputs(module):
    """Extract the Model class, instance, and inputs from a PyTorch module."""
    model_class = getattr(module, "Model", None) or getattr(module, "ReferenceModel", None)
    if model_class is None:
        raise RuntimeError("Module must define a Model or ReferenceModel class")

    init_inputs = []
    if hasattr(module, "get_init_inputs"):
        init_inputs = module.get_init_inputs()

    model = model_class(*init_inputs) if init_inputs else model_class()
    model.eval()

    if not hasattr(module, "get_inputs"):
        raise RuntimeError("Module must define get_inputs()")
    inputs = module.get_inputs()

    return model, inputs


def _to_numpy(t, use_fp64: bool = False) -> np.ndarray:
    """Convert a torch tensor (sparse or dense) to a numpy array.

    Meta tensors (no data) are materialized with random values on CPU.
    Boolean and integer tensors keep their dtype; floating-point tensors are
    promoted to float32 (default) or float64 (when *use_fp64* is True).
    """
    import torch
    if t.is_meta:
        t = torch.empty(t.shape, dtype=t.dtype, device="cpu").uniform_(-0.1, 0.1)
    if t.is_sparse:
        t = t.to_dense()
    t = t.detach().cpu()
    if t.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.float()
    arr = t.numpy()
    if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
        return arr
    return arr.astype(np.float64 if use_fp64 else np.float32)


def build_all_shapes_and_dtypes(model, inputs, use_fp64: bool = False):
    """Build unified shapes and dtypes dicts for activation inputs and model parameters.

    Returns (shapes, dtypes, fixed_inputs, buffer_keys).

    *fixed_inputs* preserves original values for activations (from
    ``get_inputs()``, which may enforce constraints like positivity) and
    buffers (pre-computed state).  Parameters get random values during
    verification and are NOT stored here.

    *buffer_keys* is the set of input names from ``named_buffers()``.
    """
    shapes = {}
    dtypes = {}
    fixed_inputs: dict = {}
    buffer_keys: set = set()

    # Activations — always fixed (preserves get_inputs() constraints)
    import torch
    for i, t in enumerate(inputs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        key = "x" if len(inputs) == 1 else f"x_{i}"
        arr = _to_numpy(t, use_fp64=use_fp64)
        shapes[key] = arr.shape
        dtypes[key] = _torch_dtype_str(t.dtype)
        fixed_inputs[key] = arr

    # Parameters — shape/dtype only (random values used during verification)
    for name, param in model.named_parameters():
        safe = name.replace(".", "_")
        shapes[safe] = tuple(param.shape)
        dtypes[safe] = _torch_dtype_str(param.dtype)

    # Buffers — always fixed (pre-computed state, use as-is)
    for name, buf in model.named_buffers():
        safe = name.replace(".", "_")
        if safe in shapes:
            continue
        arr = _to_numpy(buf, use_fp64=use_fp64)
        shapes[safe] = arr.shape
        dtypes[safe] = _torch_dtype_str(buf.dtype)
        if buf.numel() > 0:
            fixed_inputs[safe] = arr
        buffer_keys.add(safe)

    # Integer constants (num_experts, top_k, num_heads, groups, etc.)
    for module_prefix, module in [("", model)] + list(model.named_modules()):
        for attr_name, val in vars(module).items():
            if (not isinstance(val, int)
                    or attr_name.startswith("_")
                    or attr_name in _SKIP_INT_ATTRS):
                continue
            key = (f"{module_prefix}_{attr_name}".replace(".", "_").lstrip("_")
                   if module_prefix else attr_name)
            if key not in fixed_inputs and key not in shapes:
                fixed_inputs[key] = val

    return shapes, dtypes, fixed_inputs, buffer_keys


def _run_reference(model, verify_shapes, fixed_inputs, input_dtypes,
                   cap_input_size, use_fp64, debug,
                   mem_budget_bytes=None):
    """Build numpy/torch inputs from *verify_shapes* and run the PyTorch model.

    The reference model always runs at **native precision** (parameters and
    inputs keep their original dtypes).  The numpy copies stored in
    *numpy_inputs* are fp64 so the IR has full precision headroom.

    Returns ``(numpy_inputs, captured)`` where *captured* is the result of
    ``capture_model_run``.  Raises ``RuntimeError`` if the model crashes
    (e.g. broadcasting mismatch from bad capped shapes).
    """
    import torch
    from solar.einsum_lib.verify import capture_model_run

    # Memory check at native precision (no fp64 inflation)
    _check_memory(verify_shapes, input_dtypes, False,
                  mem_budget_bytes=mem_budget_bytes)

    rng = np.random.RandomState(_CAPTURE_SEED)
    numpy_inputs: dict = {}
    for name, shape in verify_shapes.items():
        if name in fixed_inputs:
            arr = fixed_inputs[name]
            if cap_input_size:
                slices = tuple(slice(0, s) for s in shape)
                numpy_inputs[name] = arr[slices]
            else:
                numpy_inputs[name] = arr
        elif input_dtypes.get(name, "") == "bool":
            numpy_inputs[name] = rng.randint(0, 2, size=shape).astype(np.bool_)
        elif _is_integer_dtype(input_dtypes.get(name, "")):
            high = max(shape) if shape else 100
            numpy_inputs[name] = rng.randint(0, high, size=shape).astype(np.int64)
        else:
            numpy_inputs[name] = rng.uniform(-0.1, 0.1, size=shape).astype(np.float32)

    # Integer constants (not in verify_shapes since they aren't tensors)
    for name, val in fixed_inputs.items():
        if isinstance(val, int) and name not in numpy_inputs:
            numpy_inputs[name] = val

    if debug:
        logger.debug("Built verification inputs")

    # --- Reference runs at native precision (no upcasting) ---
    # Load random values into parameters at their ORIGINAL dtype.
    with torch.no_grad():
        for name, param in model.named_parameters():
            safe = name.replace(".", "_")
            if safe in numpy_inputs:
                arr = numpy_inputs[safe]
                if not isinstance(arr, np.ndarray) or arr.ndim == 0:
                    continue
                if np.issubdtype(arr.dtype, np.floating):
                    param.data = torch.from_numpy(arr).to(param.dtype)
                else:
                    param.data = torch.from_numpy(arr)
        for name, buf in model.named_buffers():
            safe = name.replace(".", "_")
            if safe in numpy_inputs:
                arr = numpy_inputs[safe]
                if not isinstance(arr, np.ndarray) or arr.ndim == 0:
                    continue
                if np.issubdtype(arr.dtype, np.floating):
                    dense_t = torch.from_numpy(arr).to(buf.dtype)
                else:
                    dense_t = torch.from_numpy(arr)
                if buf.is_sparse:
                    dense_t = dense_t.to_sparse()
                parts = name.split(".")
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], dense_t)

    # Build torch activation inputs at native precision.
    # input_dtypes tells us the original dtype; we cast back to it.
    _SOLAR_TO_TORCH = {
        "bf16": torch.bfloat16, "fp16": torch.float16,
        "fp32": torch.float32, "fp64": torch.float64,
        "fp8": torch.float8_e4m3fn,
    }
    torch_inputs = []
    for key in sorted(k for k in numpy_inputs if k.startswith("x")):
        arr = numpy_inputs[key]
        if isinstance(arr, (int, float)) or not isinstance(arr, np.ndarray) or arr.ndim == 0:
            torch_inputs.append(torch.tensor(arr))
        elif arr.dtype == np.bool_:
            torch_inputs.append(torch.from_numpy(arr))
        elif np.issubdtype(arr.dtype, np.integer):
            torch_inputs.append(torch.from_numpy(arr).to(torch.long))
        else:
            native_dt = _SOLAR_TO_TORCH.get(input_dtypes.get(key, ""), torch.float32)
            torch_inputs.append(torch.from_numpy(arr).to(native_dt))

    if debug:
        logger.debug("Running PyTorch reference model (native precision)...")
    captured = capture_model_run(model, torch_inputs, numpy_inputs, seed=_CAPTURE_SEED)
    if debug:
        logger.debug("Got reference output (golden answer)")

    # numpy_inputs stay at fp32 (natural precision for numpy storage).
    # The IR runs its own kernels in fp32; tolerance accounts for
    # summation-order differences via _dtype_precision_floor.

    return numpy_inputs, captured




_MAX_CAP_RETRIES = 2


def prepare_pytorch(file_path: str, verify: bool, cap_input_size: bool = False,
                    use_fp64: bool = False, debug: bool = False,
                    cap_error_context: Optional[str] = None,
                    mem_budget_bytes: Optional[int] = None):
    """Load a PyTorch model, generate inputs, and run it once with hooks.

    Returns (source_code, input_shapes, input_dtypes, reference_outputs,
    all_inputs, input_sparsity, mask_fill_values, buffer_keys, verify_shapes).
    reference_outputs, all_inputs, and mask_fill_values are None when verify=False.
    buffer_keys is a set of input names from ``named_buffers()``.

    cap_error_context: optional error message from a previous capping round
    (either a crash from bad shapes or a translation failure).  Forwarded to
    ``scale_shapes_with_llm`` so the capping LLM can learn from the error.
    """
    from solar.einsum_lib.verify import compute_sparsity

    module = load_pytorch_module(file_path)
    source_code = Path(file_path).read_text(encoding="utf-8")

    # --- No verification: just extract shapes ----------------------------
    if not verify:
        model, inputs = get_pytorch_model_and_inputs(module)
        if debug:
            logger.debug("Loaded model and inputs")
        input_shapes, input_dtypes, fixed_inputs, buffer_keys = (
            build_all_shapes_and_dtypes(model, inputs, use_fp64=False)
        )
        input_sparsity: dict = {}
        for key, arr in fixed_inputs.items():
            density = compute_sparsity(arr)
            if density < 1.0:
                input_sparsity[key] = round(density, 4)
        return (source_code, input_shapes, input_dtypes, None, None,
                input_sparsity, None, buffer_keys, input_shapes, model)

    # Model object actually used to produce the reference outputs; threaded to
    # the translation agent so it can inspect the exact model under test.
    verify_model = None

    # --- Unified capping (config + input shapes) via single LLM call -----
    if cap_input_size:
        config_dict = None
        hf_config = getattr(module, "config", None)
        if hf_config is not None and hasattr(hf_config, "to_dict"):
            config_dict = hf_config.to_dict()

        if config_dict is not None:
            # HF model: build a lightweight from_config model for shape
            # extraction — avoids downloading the full pretrained checkpoint.
            if debug:
                logger.debug("Extracting shapes via from_config "
                      "(skipping pretrained download)")
            shape_result = _try_build_capped_model(
                load_pytorch_module(file_path), {}, debug=False,
            )
            if shape_result is not None:
                model, inputs, _ = shape_result
            else:
                model, inputs = get_pytorch_model_and_inputs(module)
        else:
            model, inputs = get_pytorch_model_and_inputs(module)
        if debug:
            logger.debug("Loaded model and inputs")
        input_shapes, input_dtypes, fixed_inputs, buffer_keys = (
            build_all_shapes_and_dtypes(model, inputs, use_fp64=False)
        )
        if debug:
            logger.debug("Built shapes and dtypes")

    if cap_input_size:
        cap_error = cap_error_context
        exc: Optional[Exception] = None
        for cap_try in range(_MAX_CAP_RETRIES):
            config_overrides, verify_shapes, llm_shape_changes = _cap_with_llm(
                source_code, input_shapes, config_dict=config_dict,
                debug=debug, error_context=cap_error,
                mem_budget_bytes=mem_budget_bytes, use_fp64=use_fp64,
            )

            if config_overrides:
                # HF path: rebuild with miniature config
                if debug:
                    logger.debug("Rebuilding model with "
                          "LLM-proposed config overrides")
                cap_module = load_pytorch_module(file_path)
                hf_result = _try_build_capped_model(
                    cap_module, config_overrides, debug=debug,
                    input_shape_overrides=llm_shape_changes,
                )
                if hf_result is None:
                    cap_error = ("Config capping failed to build model. "
                                 "Try different config values.")
                    if cap_try < _MAX_CAP_RETRIES - 1:
                        print(f"  [WARN] Config cap failed, retrying "
                              f"(attempt {cap_try + 2}/{_MAX_CAP_RETRIES})...")
                        continue
                    raise RuntimeError(
                        f"Config capping failed after {_MAX_CAP_RETRIES} "
                        f"attempts"
                    )

                capped_model, capped_inputs, original_model = hf_result
                verify_model = capped_model

                # Original shapes for the LLM prompt
                if original_model is not None:
                    orig_module = load_pytorch_module(file_path)
                    orig_inputs = orig_module.get_inputs()
                    input_shapes, input_dtypes, _, buffer_keys = (
                        build_all_shapes_and_dtypes(
                            original_model, orig_inputs,
                            use_fp64=False)
                    )

                # Capped shapes for verification
                verify_shapes, verify_dtypes, fixed_inputs, buffer_keys = (
                    build_all_shapes_and_dtypes(
                        capped_model, capped_inputs,
                        use_fp64=False)
                )
                # The LLM may also propose activation shape changes (e.g.
                # smaller batch) that aren't captured by the config rebuild.
                # Apply only explicitly changed shapes (typically activations
                # like 'x') — not weights, which are already correct from
                # the config-rebuilt model.
                for k, v in llm_shape_changes.items():
                    if k in verify_shapes:
                        verify_shapes[k] = v
                input_dtypes.update(verify_dtypes)

                try:
                    numpy_inputs, captured = _run_reference(
                        capped_model, verify_shapes, fixed_inputs,
                        verify_dtypes, True, use_fp64, debug,
                        mem_budget_bytes=mem_budget_bytes,
                    )
                    break
                except Exception as _exc:
                    exc = _exc
                    cap_error = str(exc)
            else:
                # Non-HF path: original model with capped input shapes
                verify_model = model
                if debug and verify_shapes != input_shapes:
                    for k in input_shapes:
                        if input_shapes[k] != verify_shapes.get(k, input_shapes[k]):
                            logger.debug(f"Capped {k}: "
                                  f"{input_shapes[k]} -> "
                                  f"{verify_shapes[k]}")
                try:
                    numpy_inputs, captured = _run_reference(
                        model, verify_shapes, fixed_inputs, input_dtypes,
                        cap_input_size, use_fp64, debug,
                        mem_budget_bytes=mem_budget_bytes,
                    )
                    break
                except Exception as _exc:
                    exc = _exc
                    cap_error = str(exc)

            if exc is not None and isinstance(exc, MemoryError):
                budget = _get_effective_budget_bytes(mem_budget_bytes)
                budget_gb = (f"{budget / 1e9:.2f}" if budget
                             else "unknown")
                cap_error += (
                    f"\nActivation shapes require too much memory. "
                    f"Target total under {budget_gb} GB."
                )
            if cap_try < _MAX_CAP_RETRIES - 1:
                print(f"  [WARN] Capped run failed "
                      f"({cap_error[:120]}), asking LLM to fix "
                      f"(attempt {cap_try + 2}/{_MAX_CAP_RETRIES})...")
            else:
                raise RuntimeError(
                    f"Capped shapes crashed the model after "
                    f"{_MAX_CAP_RETRIES} attempts: {cap_error}"
                ) from exc
    else:
        # --- No capping: use original model as-is -----------------------
        model, inputs = get_pytorch_model_and_inputs(module)
        verify_model = model
        if debug:
            logger.debug("Loaded model and inputs")
        input_shapes, input_dtypes, fixed_inputs, buffer_keys = (
            build_all_shapes_and_dtypes(model, inputs, use_fp64=False)
        )
        verify_shapes = input_shapes
        numpy_inputs, captured = _run_reference(
            model, verify_shapes, fixed_inputs, input_dtypes,
            cap_input_size, use_fp64, debug,
            mem_budget_bytes=mem_budget_bytes,
        )

    input_sparsity: dict = {}
    for key, arr in fixed_inputs.items():
        density = compute_sparsity(arr)
        if density < 1.0:
            input_sparsity[key] = round(density, 4)

    for mask_name, mask_arr in captured.captured_masks.items():
        verify_shapes[mask_name] = tuple(mask_arr.shape)
        input_shapes[mask_name] = tuple(mask_arr.shape)
        input_dtypes[mask_name] = "fp32"
        numpy_inputs[mask_name] = mask_arr
        density = compute_sparsity(mask_arr)
        if density < 1.0:
            input_sparsity[mask_name] = round(density, 4)

    for rnd_name, rnd_arr in captured.captured_randn.items():
        arr = rnd_arr.astype(np.float64) if use_fp64 else rnd_arr.astype(np.float32)
        verify_shapes[rnd_name] = tuple(arr.shape)
        input_shapes[rnd_name] = tuple(arr.shape)
        input_dtypes[rnd_name] = "fp32"
        numpy_inputs[rnd_name] = arr

    return (source_code, input_shapes, input_dtypes,
            captured.reference_outputs, numpy_inputs, input_sparsity,
            captured.mask_fill_values, buffer_keys, verify_shapes, verify_model)


def _prepare_raw(file_path, input_shapes_json, verify):
    """Load raw source code with manually specified input shapes."""
    source_code = Path(file_path).read_text(encoding="utf-8")
    if input_shapes_json is None:
        raise RuntimeError("--input-shapes is required for --source-type raw")
    input_shapes = {k: tuple(v) for k, v in json.loads(input_shapes_json).items()}
    return source_code, input_shapes, None, None, None, {}, None, None


_DEFAULT_MAX_CAP_ROUNDS = 2


def _build_tool_context(file_path, input_shapes, verify_model):
    """Assemble the namespace seeded into the agent's run_python tool.

    Always frontend-neutral: `source_path` and `input_shapes`.  When an
    instantiated model is available (PyTorch), the exact object under
    verification (and its config) is included so the agent inspects THAT rather
    than re-downloading a different (uncapped) model.
    """
    ctx = {
        "source_path": str(file_path),
        "input_shapes": input_shapes,
    }
    if verify_model is not None:
        ctx["model"] = verify_model
        ctx["config"] = getattr(verify_model, "config", None)
    return ctx


_DTYPE_TO_NP: dict = {
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
}
_BF16_EPS = 2.0 ** -7                   # 7.8125e-3, IEEE 754 bfloat16
# Headroom multipliers: the reference runs at native precision while the IR
# runs in fp64.  Summation-order differences in the reference accumulate
# proportionally to eps of the native dtype.  bf16 needs little headroom
# (tolerance is already huge).  fp32 needs more because its tolerance is
# tight and large reductions can differ by ~100x eps.
_DTYPE_HEADROOM = {
    "bf16": 4.0,
    "fp16": 4.0,
    "fp32": 256.0,    # 256 * 1.19e-7 ≈ 3e-5
}
_DEFAULT_HEADROOM = 4.0


def _dtype_precision_floor(input_dtypes: Optional[dict]) -> float:
    """Tolerance floor implied by the narrowest float dtype in the inputs.

    The reference model runs at native precision while the IR runs in fp64.
    Summation-order differences in the reference produce noise proportional
    to eps of the native dtype.  This function returns a tolerance floor
    that absorbs that noise.
    """
    if not input_dtypes:
        return 0.0
    floor = 0.0
    for dtype_str in input_dtypes.values():
        if dtype_str == "bf16":
            eps = _BF16_EPS
        elif dtype_str in _DTYPE_TO_NP:
            eps = float(np.finfo(_DTYPE_TO_NP[dtype_str]).eps)
        else:
            continue
        headroom = _DTYPE_HEADROOM.get(dtype_str, _DEFAULT_HEADROOM)
        floor = max(floor, headroom * eps)
    return floor



def translate_file(
    file_path: str,
    source_type: str = "pytorch",
    output_dir: Optional[str] = None,
    max_retries: int = 5,
    max_tokens: int = 65536,
    rate_limit_retries: int = 3,
    rate_limit_delay: int = 60,
    verify: bool = True,
    api_key_file: Optional[str] = None,
    api_key: Optional[str] = None,
    input_shapes_json: Optional[str] = None,
    thinking: bool = True,
    thinking_budget: int = 10000,
    save_on_fail: bool = False,
    fast: bool = True,
    cap_input_size: bool = False,
    use_fp64: bool = False,
    debug: bool = False,
    verify_timeout: Optional[int] = None,
    max_cap_rounds: int = _DEFAULT_MAX_CAP_ROUNDS,
    mem_budget_bytes: Optional[int] = None,
    llm_model: Optional[str] = None,
) -> TranslationResult:
    """Full pipeline: load source, translate to einsum_lib, verify, save.

    Supports multiple source types: 'pytorch' (default) and 'raw'.

    When *cap_input_size* is True and *verify* is True, the pipeline wraps
    the prepare + translate sequence in an outer loop of up to
    *max_cap_rounds*.  If the translation LLM exhausts its retries, the
    error is fed back to the capping LLM which proposes new shapes and a
    fresh reference is computed.  If all capped rounds fail, the pipeline
    falls back to one final attempt with the original uncapped shapes.
    """
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unknown source type '{source_type}'. "
            f"Supported: {SUPPORTED_SOURCE_TYPES}"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    use_cap_rounds = cap_input_size and verify
    n_cap_rounds = max_cap_rounds if use_cap_rounds else 1
    cap_error: Optional[str] = None
    best_result: Optional[TranslationResult] = None

    for cap_round in range(n_cap_rounds):
        if use_cap_rounds and cap_round > 0:
            logger.info(f"\n  [CAP ROUND {cap_round + 1}/{n_cap_rounds}] "
                  f"Re-capping shapes (previous error: {cap_error[:120]})")

        # The instantiated model under verification (PyTorch only); seeded into
        # the agent's run_python namespace so it inspects the exact object under
        # test instead of re-downloading. Stays None for jax/raw.
        verify_model = None

        # Floor on the output comparison tolerance.  CUDA derives it from the
        # reference output dtypes.  JAX/Triton derive it from the declared input
        # dtypes, because the reference runs in fp32/fp64 but the model's actual
        # precision is limited by the narrowest float it operates on (e.g. bf16).
        compare_min_tol = 0.0

        try:
            if source_type == "pytorch":
                logger.info(f"Loading {file_path.name}...")
                prepared = prepare_pytorch(
                    str(file_path), verify, cap_input_size=cap_input_size,
                    use_fp64=use_fp64, debug=debug,
                    cap_error_context=cap_error,
                    mem_budget_bytes=mem_budget_bytes,
                )
                (source_code, input_shapes, input_dtypes, reference_outputs,
                 all_inputs, input_sparsity, mask_fill_values,
                 buffer_keys, verify_shapes, verify_model) = prepared
                compare_min_tol = _dtype_precision_floor(input_dtypes)
                if debug and compare_min_tol:
                    logger.debug(f"Narrowest float dtype implies tolerance "
                          f"floor {compare_min_tol:.3e}")
                logger.info(f"Input shapes: {input_shapes}")
                if verify and cap_input_size and verify_shapes != input_shapes:
                    logger.info(f"Verify shapes: {verify_shapes}")
                if input_dtypes:
                    logger.info(f"Input dtypes: {input_dtypes}")
                if input_sparsity:
                    logger.info(f"Input sparsity (density): {input_sparsity}")
            elif source_type == "jax":
                logger.info(f"Loading {file_path.name} (JAX/Flax)...")
                from solar.einsum_lib.jax_frontend import (
                    jax_reference_is_fp64, prepare_jax,
                )
                prepared = prepare_jax(str(file_path), verify,
                                       cap_input_size=cap_input_size,
                                       use_fp64=use_fp64, debug=debug,
                                       mem_budget_bytes=mem_budget_bytes,
                                       cap_error_context=cap_error)
                use_fp64 = jax_reference_is_fp64()
                if verify:
                    (source_code, input_shapes, input_dtypes, reference_outputs,
                     all_inputs, input_sparsity, mask_fill_values,
                     buffer_keys, verify_shapes) = prepared
                else:
                    (source_code, input_shapes, input_dtypes, reference_outputs,
                     all_inputs, input_sparsity, mask_fill_values,
                     buffer_keys) = prepared
                    verify_shapes = input_shapes
                compare_min_tol = _dtype_precision_floor(input_dtypes)
                if debug and compare_min_tol:
                    logger.debug(f"Narrowest float dtype implies tolerance "
                          f"floor {compare_min_tol:.3e}")
                logger.info(f"Input shapes: {input_shapes}")
                if verify and cap_input_size and verify_shapes != input_shapes:
                    logger.info(f"Verify shapes: {verify_shapes}")
                if input_dtypes:
                    logger.info(f"Input dtypes: {input_dtypes}")
            elif source_type == "triton":
                logger.info(f"Loading {file_path.name} (Triton kernel)...")
                from solar.einsum_lib.triton_frontend import prepare_triton
                prepared = prepare_triton(str(file_path), verify,
                                          use_fp64=use_fp64, debug=debug,
                                          cap_input_size=cap_input_size)
                if verify:
                    (source_code, input_shapes, input_dtypes, reference_outputs,
                     all_inputs, input_sparsity, mask_fill_values,
                     buffer_keys, verify_shapes) = prepared
                else:
                    (source_code, input_shapes, input_dtypes, reference_outputs,
                     all_inputs, input_sparsity, mask_fill_values,
                     buffer_keys) = prepared
                    verify_shapes = input_shapes
                verify_model = None
                compare_min_tol = _dtype_precision_floor(input_dtypes)
                if debug and compare_min_tol:
                    logger.debug(f"Narrowest float dtype implies tolerance "
                          f"floor {compare_min_tol:.3e}")
                logger.info(f"Input shapes: {input_shapes}")
                if verify and cap_input_size and verify_shapes != input_shapes:
                    logger.info(f"Verify shapes: {verify_shapes}")
                if input_dtypes:
                    logger.info(f"Input dtypes: {input_dtypes}")
            elif source_type == "cuda":
                logger.info(f"Loading {file_path.name} (CUDA kernel)...")
                from solar.einsum_lib.cuda_frontend import prepare_cuda
                prepared = prepare_cuda(str(file_path), verify,
                                        use_fp64=use_fp64, debug=debug,
                                        cap_input_size=cap_input_size)
                if verify:
                    (source_code, input_shapes, input_dtypes, reference_outputs,
                     all_inputs, input_sparsity, mask_fill_values,
                     buffer_keys, verify_shapes, compare_min_tol) = prepared
                else:
                    (source_code, input_shapes, input_dtypes, reference_outputs,
                     all_inputs, input_sparsity, mask_fill_values,
                     buffer_keys) = prepared
                    verify_shapes = input_shapes
                verify_model = None
                logger.info(f"Input shapes: {input_shapes}")
                if verify and cap_input_size and verify_shapes != input_shapes:
                    logger.info(f"Verify shapes: {verify_shapes}")
                if input_dtypes:
                    logger.info(f"Input dtypes: {input_dtypes}")
            else:
                logger.info(f"Loading {file_path.name} (source_type=raw)...")
                (source_code, input_shapes, input_dtypes, reference_outputs,
                 all_inputs, input_sparsity, mask_fill_values,
                 buffer_keys) = _prepare_raw(
                    str(file_path), input_shapes_json, verify
                )
                verify_shapes = input_shapes
                logger.info(f"Input shapes: {input_shapes}")
        except Exception as prepare_exc:
            if use_cap_rounds and cap_round < n_cap_rounds - 1:
                cap_error = (
                    f"Model preparation/capping failed: {prepare_exc}"
                )
                print(f"  [WARN] Preparation failed ({str(prepare_exc)[:120]}), "
                      f"will re-cap in round {cap_round + 2}/{n_cap_rounds}...")
                continue
            raise

        full_inputs: Optional[dict] = None
        if verify:
            float_dtype = np.float64 if use_fp64 else np.float32
            _dt = input_dtypes or {}
            full_inputs = {}
            for name, shape in input_shapes.items():
                dt = np.int64 if _is_integer_dtype(_dt.get(name, "")) else float_dtype
                if shape == ():
                    stub = np.zeros((), dtype=dt)
                else:
                    stub = np.broadcast_to(np.zeros(1, dtype=dt), shape)
                    stub.flags.writeable = False
                full_inputs[name] = stub
            if all_inputs:
                for name, val in all_inputs.items():
                    if isinstance(val, int):
                        full_inputs[name] = val

        agent_kwargs = {
            "max_retries": max_retries,
            "max_tokens": max_tokens,
            "rate_limit_retries": rate_limit_retries,
            "rate_limit_delay": rate_limit_delay,
            "verify": verify,
            "thinking": thinking,
            "thinking_budget": thinking_budget,
            "fast": fast,
            "source_type": source_type,
            "use_fp64": use_fp64,
            "debug": debug,
            "verify_timeout": verify_timeout,
            "mem_budget_bytes": _get_effective_budget_bytes(mem_budget_bytes),
            "compare_min_tol": compare_min_tol,
            "python_tool": True,
            "tool_context": _build_tool_context(
                file_path, verify_shapes, verify_model
            ),
        }
        if api_key_file:
            agent_kwargs["api_key_file"] = api_key_file
        if api_key:
            # explicit key handed in by an in-process caller (the fallback
            # dispatcher's Pythonic provider); no env/key-file lookup happens
            # when it is given
            agent_kwargs["api_key"] = api_key
        if llm_model:
            agent_kwargs["model"] = llm_model
        agent = TranslationAgent(**agent_kwargs)

        int_constants = (
            {k: v for k, v in all_inputs.items() if isinstance(v, int)}
            if all_inputs else None
        ) or None

        logger.info(f"Translating (max {max_retries} attempts)...")
        t0 = time.time()
        result = agent.translate(
            source_code=source_code,
            input_shapes=input_shapes,
            reference_outputs=reference_outputs,
            input_dtypes=input_dtypes,
            all_inputs=all_inputs,
            input_sparsity=input_sparsity if input_sparsity else None,
            mask_fill_values=mask_fill_values,
            full_inputs=full_inputs,
            buffer_keys=buffer_keys,
            int_constants=int_constants,
        )
        elapsed = time.time() - t0

        status = "VERIFIED" if result.verified else "FAILED"
        timing_str = result.timing.summary()
        logger.info(f"[{status}] attempts={result.attempts}, time={elapsed:.1f}s ({timing_str})")
        if result.verification_error:
            print(f"  Error: {result.verification_error}")

        if best_result is None or result.verified:
            best_result = result

        if result.verified:
            break

        if use_cap_rounds and cap_round < n_cap_rounds - 1:
            cap_error = (
                f"Translation failed after {result.attempts} attempts with "
                f"these capped shapes: {verify_shapes}. "
                f"Last error: {result.verification_error}"
            )

    if use_cap_rounds and not best_result.verified:
        print(f"\n  [FAIL] All {n_cap_rounds} capped rounds failed.")

    result = best_result

    should_save = output_dir and result.code and (result.verified or save_on_fail)
    if should_save:
        from solar.einsum_lib.metadata import write_metadata_yaml

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        code_file = out_path / f"{file_path.stem}_einsum_lib.py"
        code_file.write_text(result.code, encoding="utf-8")
        logger.info(f"Saved: {code_file}")

        if result.metadata:
            yaml_file = out_path / f"{file_path.stem}_einsum_lib.yaml"
            write_metadata_yaml(result.metadata, str(yaml_file))
            logger.info(f"Saved: {yaml_file}")

        if result.numba_code:
            numba_file = out_path / f"{file_path.stem}_einsum_lib_numba.py"
            numba_file.write_text(result.numba_code, encoding="utf-8")
            logger.info(f"Saved: {numba_file}")

    return result


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def discover_source_files(directory: str) -> list:
    """Find all source files in a directory.

    Handles two layouts:
      - Flat:   directory/*                    (files directly in directory)
      - Nested: directory/<name>/<name>.py     (subdirectory with matching .py)
    """
    dirpath = Path(directory)
    files = []

    for subdir in sorted(dirpath.iterdir()):
        if subdir.is_dir():
            model_file = subdir / f"{subdir.name}.py"
            if model_file.exists():
                files.append(model_file)

    for f in sorted(dirpath.glob("*")):
        if f.is_file() and not f.name.startswith("__"):
            files.append(f)

    seen = set()
    unique = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)

    return unique


def run_batch(args):
    """Translate all files in a directory."""
    files = discover_source_files(args.path)
    if not files:
        logger.info(f"No source files found in {args.path}")
        sys.exit(1)

    logger.info(f"Found {len(files)} source file(s) (source_type={args.source_type}):")
    for f in files:
        logger.info(f"{f}")
    logger.info("")

    results = {}
    t_total = time.time()

    for i, f in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] {f.name}")
        try:
            result = translate_file(
                str(f),
                source_type=args.source_type,
                output_dir=str(Path(args.output_dir) / f.stem) if args.output_dir else None,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                rate_limit_retries=args.rate_limit_retries,
                rate_limit_delay=args.rate_limit_delay,
                verify=not args.no_verify,
                api_key_file=args.api_key_file,
                thinking=not args.no_thinking,
                thinking_budget=args.thinking_budget,
                save_on_fail=args.save_on_fail,
            )
            results[f.name] = {
                "verified": result.verified,
                "attempts": result.attempts,
                "error": result.verification_error,
            }
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results[f.name] = {
                "verified": False,
                "attempts": 0,
                "error": str(e),
            }
        logger.info("")

    elapsed = time.time() - t_total

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    total = len(results)
    verified = sum(1 for r in results.values() if r["verified"])
    failed = total - verified

    for name, r in results.items():
        status = "PASS" if r["verified"] else "FAIL"
        detail = f"attempts={r['attempts']}"
        if r["error"]:
            detail += f", error={r['error'][:80]}"
        logger.info(f"[{status}] {name}: {detail}")

    print(f"\nTotal: {total} | Verified: {verified} | Failed: {failed}")
    logger.info(f"Time: {elapsed:.1f}s")

    if args.output_dir:
        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Summary saved: {summary_path}")

    sys.exit(0 if failed == 0 else 1)


def run_single(args):
    """Translate a single file."""
    result = translate_file(
        args.path,
        source_type=args.source_type,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        rate_limit_retries=args.rate_limit_retries,
        rate_limit_delay=args.rate_limit_delay,
        verify=not args.no_verify,
        api_key_file=args.api_key_file,
        input_shapes_json=args.input_shapes,
        thinking=not args.no_thinking,
        thinking_budget=args.thinking_budget,
        save_on_fail=args.save_on_fail,
        cap_input_size=getattr(args, "cap_input_size", False),
        debug=getattr(args, "debug", False),
        verify_timeout=getattr(args, "verify_timeout", None),
        max_cap_rounds=getattr(args, "max_cap_rounds", _DEFAULT_MAX_CAP_ROUNDS),
    )

    show_code = result.verified or args.save_on_fail
    if result.code and show_code:
        tag = "VERIFIED" if result.verified else "UNVERIFIED"
        logger.info("\n" + "=" * 60)
        logger.info(f"GENERATED CODE ({tag}):")
        logger.info("=" * 60)
        logger.info(result.code)
        logger.info("=" * 60)

    if not result.verified:
        print("\nFAIL")

    sys.exit(0 if result.verified else 1)
