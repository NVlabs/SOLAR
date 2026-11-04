# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translation agent: converts PyTorch source code to einsum_lib programs.

Uses an LLM to translate, then verifies the result by executing both the
original and translated programs and comparing outputs.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from solar.einsum_lib.verify import (
    VerificationResult,
    compare_outputs,
    execute_translated,
)
from solar.einsum_lib.core import get_trace
from solar.einsum_lib.metadata import build_metadata
import logging

logger = logging.getLogger(__name__)

try:
    import openai as _openai_module  # type: ignore
except ImportError:
    _openai_module = None  # type: ignore[assignment]

# Public configuration: SOLAR ships NO built-in inference endpoint.  The
# OpenAI-compatible base URL must be supplied explicitly (``base_url=`` or the
# SOLAR_API_BASE_URL environment variable).  The agent refuses to construct a
# client without one, so no request can ever go to an unintended provider.
_BASE_URL_ENV = "SOLAR_API_BASE_URL"
_DEFAULT_BASE_URL = None

# SOLAR ships no default model either: the model name must be one the
# configured endpoint serves and is given explicitly (``model=``, the CLI's
# ``--llm-model``) or through the SOLAR_LLM_MODEL environment variable.
# Per-model request settings (max_tokens, thinking) may be registered here by
# a deployment; an unregistered model uses the generic settings below.
_MODEL_ENV = "SOLAR_LLM_MODEL"
_MODEL_REGISTRY: dict = {}
_FALLBACK_MODELS: list = []
_DEFAULT_MODEL = None

# Max number of python-tool rounds allowed within a single LLM turn before we
# force the model to produce a final answer.  Deliberately very large: this is
# only a last-resort guard against pathological non-termination, not a limit on
# how much the agent is meant to inspect.
_MAX_TOOL_ROUNDS = 30

# OpenAI-style tool schema for a generic Python execution capability.  Only
# offered to the LLM when the python tool is enabled on the agent.  Deliberately
# framework-agnostic: it is a plain Python REPL, not tied to any model object.
_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run arbitrary Python and get captured stdout back. The namespace "
            "persists across calls within this translation, so you can import "
            "libraries, load/instantiate the model yourself, read its real "
            "source, evaluate config- or shape-dependent branches, and print "
            "whatever you want to see. Use it to ground yourself in what the "
            "code ACTUALLY does instead of relying on memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source to execute. Use print(...) to emit "
                        "output you want returned."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

# Appended to the system prompt when the python tool is enabled.  Deliberately
# generic and framework-agnostic: it nudges the agent to ground itself against
# the real implementation rather than relying on memory.
_PYTHON_INSTRUCTION = """

INSPECTION TOOL - run_python (READ-ONLY purpose):
You have a `run_python(code)` tool for INSPECTING the source model ONLY. Its
sole purpose is to help you understand what the model does before you write your
translation. It is NOT a REPL for developing or testing your einsum_lib code.

Pre-set namespace:
- `source_path`: path to the source file being translated.
- `input_shapes`: dict of input name -> shape for the model under test.
- `model` (when available): the already-instantiated model (do NOT re-download).
- `config` (when available): its config object.

WHAT TO USE IT FOR (inspection only):
- Read source code: `import inspect; print(inspect.getsource(type(model)))`
- Check model structure: `model.named_modules()`, `model.named_parameters()`
- Resolve shape-dependent branches: check which code path executes for these
  `input_shapes` (e.g. `if seq_len <= chunk_length`).
- Confirm parameter shapes and which parameters are consumed in the forward pass.

WHAT NOT TO DO (strictly forbidden):
- Do NOT execute, test, or debug your generated einsum_lib code via run_python.
  The framework verifies it automatically — just submit your final code.
- Do NOT build manual reference implementations or numerical comparisons.
- Do NOT import or call solar.einsum_lib inside run_python.
- Do NOT inspect solar.* internals (kernel, Dim, Tensor, verify, pipeline, etc.)

WORKFLOW: Inspect the model as needed to understand it, then write and submit
your translation. Do not iterate or debug via run_python.

EXPECTED SIZE MISMATCH: the provided model may be a CAPPED miniature. This is
normal. Your code MUST be shape-general (derive sizes from `.shape`, never
hardcode). Do NOT spend tool calls comparing prompt sizes vs model sizes.

If the run_python tool is not available, emit a fenced block ```python_tool```
containing the code and you will receive its output. When you have gathered what
you need, return ONLY the final code block.
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------



def _profile_measured_density(code, inputs_used):
    """Isolated-child entry (module-level, picklable for ``verify._run_isolated``):
    a full-value traced pass whose density measurements come back as
    metadata.  The child is a fresh process; the parent's density-profiling
    switch is re-applied there by ``_run_isolated`` itself."""
    execute_translated(
        code, {k: v.shape for k, v in inputs_used.items() if hasattr(v, "shape")},
        all_inputs=inputs_used, trace=True, shape_only=False, fast=False,
    )
    return build_metadata(get_trace())


@dataclass
class TimingInfo:
    """Per-phase timing breakdown for a translation run."""

    llm_seconds: float = 0.0
    verify_seconds: float = 0.0
    trace_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.llm_seconds + self.verify_seconds + self.trace_seconds

    def summary(self) -> str:
        parts = [f"verify={self.verify_seconds:.2f}s"]
        parts.append(f"trace={self.trace_seconds:.2f}s")
        parts.append(f"llm={self.llm_seconds:.2f}s")
        return ", ".join(parts)


@dataclass
class TranslationResult:
    """Outcome of a translate() call.

    ``verified`` is True only when the generated code executed AND its
    outputs were compared against ``reference_outputs`` within the applied
    tolerance.  ``verification_status`` names the state precisely:

    * ``verified``              — compared, within tolerance
    * ``verified_vacuous``      — compared and passed, but every output listed
                                  in ``vacuous_outputs`` had an all-zero
                                  reference (weak evidence; legacy policy keeps
                                  ``verified=True`` — callers must show it)
    * ``executed_no_reference`` — ran structurally; NO numerical comparison
                                  happened (``verified=False``)
    * ``mismatch`` / ``execution_failed`` — the last attempt failed that way
    * ``unverified``            — ``verify=False`` was requested
    """

    code: str
    verified: bool
    verification_error: Optional[str] = None
    attempts: int = 1
    metadata: Optional[dict] = None
    numba_code: Optional[str] = None
    timing: TimingInfo = field(default_factory=TimingInfo)
    verification_status: str = "unverified"
    vacuous_outputs: tuple = ()
    # The tolerance compare_outputs actually applied (rtol/atol/min_tol), or
    # None when no comparison happened.
    applied_tolerance: Optional[dict] = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Shared by every preamble (substituted for _WORK_MINIMAL_TOKEN below).  The
# loop nest the model writes is what Solar counts and prices, so "do not
# optimize" must not be read as "translate naively".
_WORK_MINIMAL_TOKEN = "@@WORK_MINIMAL_RULES@@\n"
_WORK_MINIMAL_RULES = """\
- WORK-MINIMAL TRANSLATION: "Do NOT optimize" forbids tricks that do not \
change the work performed (unrolling, constant folding, tiling). It is NOT a \
request for a naive translation. The loop nest you write is what Solar counts \
and prices, so write the most efficient translation that is still \
functionally correct: it performs exactly the work the source semantically \
requires, no more and no less. Correctness wins: efficiency is not a licence \
to change numerics, and if an efficient form cannot be made to match the \
source output, use the plain form that does.
  * Loop each Dim over the extent the source really computes. A projection \
that uses half of a fused weight loops that half, not the whole axis \
(`Dim("I", size=W.shape[0] // 2)`); a window loops the window, not the \
sequence. Never loop a full axis and discard part of the result.
  * Give every sub-range loop an explicit, statically resolvable extent: an \
integer expression over the `.shape` of the kernel's own inputs and integer \
constants. An extent that cannot be resolved statically is refused or \
mis-counted by the converter (it falls back to index inference).
  * Compute each value once: hoist loop-invariant work out of inner loops, \
reuse shared intermediates, and read a single copy instead of materializing \
repeats (GQA, broadcast, expand). Do not add passes the source does not \
perform.
- STATIC MASKS: a mask is static when shapes and constants in the source fully \
determine it (causal/triangular, sliding window or band, block-diagonal, \
fixed padding or valid length, strides and dilations). Use the most efficient \
form that still matches the numerics, in this order:
  1. Compact loops: iterate only the unmasked range with a resolvable extent \
(a window of W keys loops `Dim("W", size=W)`; a padded tail is left out of \
the extent) and drop the mask.
  2. If the live region is not a rectangular loop range (causal, irregular \
band), build the mask in `run_model` as a real 0/1 tensor from shapes and \
constants and give it its ANALYTIC density at the ORIGINAL (uncapped) shapes: \
`Tensor.from_numpy(mask, "S,T", sparsity=<density>)` and multiply by it \
INSIDE the kernel body (`acc * M[s, t]`): a multiplicative input's density \
discounts that kernel's MACs and bytes. `apply_mask(..., masked_value=0.0)` \
on a kernel's OUTPUT does not discount the producing kernel; it only gives \
the result (and downstream consumers) the mask's density. Compute the \
density in closed form at the ORIGINAL S (square causal: (S+1)/(2*S); \
causal window W: sum_i min(i+1, W) / S^2; symmetric band `abs(s-t) < W` on \
an S-by-S mask with 1 <= W <= S: ((2*W-1)*S - W*(W-1)) / S^2, which is 1 \
for W >= S; note the plain (2*W-1)/S ignores the truncated edges and is \
wrong); do not measure it on capped inputs, whose density differs from the \
original-scale one.
  3. Never hide a static mask in an arithmetic guard inside the kernel body \
(`max(0.0, min(1.0, s - t))`): the guard's sparsity is invisible to the \
analyzer, so the work is billed dense. Hoist it into the mask tensor of \
rule 2.
  A data-dependent mask (a runtime attention or padding mask input, a learned \
or routed selection) is NOT static: never claim sparsity for it. Keep the \
dense form and the mask as an input.
"""

_PREAMBLE = """\
You are a code translator. Translate PyTorch source code into equivalent \
Python code using the solar.einsum_lib library.

CRITICAL RULES:
- Do NOT optimize: no loop unrolling, constant folding, or performance tricks.
@@WORK_MINIMAL_RULES@@
- The code must capture the GENERAL structure of each operation, not be \
specialized to specific input values.
- Treat all inputs as symbolic — do NOT inline literal values from the source.
- Every loop nest must be perfectly nested — computation only in the \
innermost body. A kernel may contain multiple sequential loop nests.
- For softmax: decompose into 4 perfectly nested loop nests \
(max-reduction, exp-shift, sum-reduction, normalize). Do NOT use a \
softmax() function call.
- Create output tensors with sizes from input `.shape`: \
`C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype=...)`. \
Do NOT hardcode literal dimension sizes.
- Every Tensor MUST include `dtype=` matching the source program's precision. \
If not specified in the source, use `dtype="fp32"`. \
Supported: fp64, fp32, tf32, fp16, bf16, fp8, int8, int32, int64, nvfp4. \
Use int64 for integer index tensors (e.g. relative_position_index for gather).
- Scalar inputs have shape `()`. Use `Tensor("", dtype=...)` for the \
annotation and index with `S[()]` in the kernel body. \
In `run_model`, load with `Tensor.from_numpy(inputs["key"], "", dtype=...)`.

OUTPUT FORMAT:
Produce a single Python code block containing:
1. `from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask`
2. One `@kernel` function per operation in the model.
3. A `model()` function that composes the kernels.
4. A `run_model(inputs: dict) -> dict` function that:
   - Calls `clear_dims()`
   - The `inputs` dict contains ALL arrays: both activation inputs \
(keys starting with "x") AND model parameters/weights (all other keys).
   - Wrap each array as a Tensor via `Tensor.from_numpy()`.  Use \
`role="weight"` for parameter tensors.
   - Calls `model()` and returns `{"output": result.data}`
   - Do NOT create weights with `Tensor.from_random()` — use the \
arrays from `inputs`.
   - **Dim names in `Tensor.from_numpy()` must match the kernel signatures \
that consume those tensors.** The `@kernel` decorator registers dims from \
the actual tensors passed in. If a kernel expects `Tensor("B,C,H,W")`, the \
tensor must use dims `"B,C,H,W"` — not renamed variants like `"BB,CIN,IH,IW"`. \
Mismatched names cause `Dimension 'X' is not registered` errors at runtime.
   - If the inputs dict contains captured mask tensors (keys starting with \
"captured_mask_"), load each as a Tensor and call the builtin \
`apply_mask(tensor, mask, masked_value=...)` in the `model()` function. \
Each mask uses the convention 1 = keep, 0 = masked out. \
The `masked_value` for each mask will be specified alongside its shape. \
Do NOT define `apply_mask` as a \
`@kernel` — it is already provided by the library import.

Return ONLY the code block, no explanations.

REFERENCE:

"""


_PREAMBLE_JAX = """\
You are a code translator. Translate JAX/Flax source code into equivalent \
Python code using the solar.einsum_lib library.

CRITICAL RULES:
- Do NOT optimize: no loop unrolling, constant folding, or performance tricks.
@@WORK_MINIMAL_RULES@@
- The code must capture the GENERAL structure of each operation, not be \
specialized to specific input values.
- Treat all inputs as symbolic — do NOT inline literal values from the source.
- Every loop nest must be perfectly nested — computation only in the \
innermost body. A kernel may contain multiple sequential loop nests.
- For softmax (including jax.nn.softmax): decompose into 4 perfectly nested \
loop nests (max-reduction, exp-shift, sum-reduction, normalize). Do NOT \
use a softmax() function call.
- For jnp.einsum: expand into explicit Dim loops. Do NOT copy einsum strings.
- Create output tensors with sizes from input `.shape`: \
`C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype=...)`. \
Do NOT hardcode literal dimension sizes.
- Every Tensor MUST include `dtype=` matching the source program's precision. \
If not specified in the source, use `dtype="fp32"`. \
Supported: fp64, fp32, tf32, fp16, bf16, fp8, int8, int32, int64, nvfp4. \
Use int64 for integer index tensors (e.g. relative_position_index for gather).
- Scalar inputs have shape `()`. Use `Tensor("", dtype=...)` for the \
annotation and index with `S[()]` in the kernel body. \
In `run_model`, load with `Tensor.from_numpy(inputs["key"], "", dtype=...)`.

OUTPUT FORMAT:
Produce a single Python code block containing:
1. `from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask`
2. One `@kernel` function per operation in the model.
3. A `model()` function that composes the kernels.
4. A `run_model(inputs: dict) -> dict` function that:
   - Calls `clear_dims()`
   - The `inputs` dict contains ALL arrays: both activation inputs \
(keys starting with "x") AND model parameters/weights (all other keys).
   - Wrap each array as a Tensor via `Tensor.from_numpy()`.  Use \
`role="weight"` for parameter tensors.
   - Calls `model()` and returns `{"output": result.data}`
   - Do NOT create weights with `Tensor.from_random()` — use the \
arrays from `inputs`.
   - **Dim names in `Tensor.from_numpy()` must match the kernel signatures \
that consume those tensors.** If a kernel expects `Tensor("B,C,H,W")`, pass \
dims `"B,C,H,W"` — not renamed variants. Mismatched names cause runtime errors.

Return ONLY the code block, no explanations.

REFERENCE:

"""


_PREAMBLE_TRITON = """\
You are a code translator. Translate an OpenAI Triton kernel into equivalent \
Python code using the solar.einsum_lib library.

The source defines a `@triton.jit` kernel plus a `kernel_fn(inputs)` launcher \
and `get_inputs()`. Translate the MATHEMATICAL computation the kernel performs \
into named-dimension loops. Recover the algorithm from the SPMD kernel body:

- Each program instance (`tl.program_id(axis)`) is one iteration of an OUTER \
loop over that grid axis (typically rows / blocks). Reconstruct the full \
iteration space; do NOT special-case a single program id.
- `tl.arange(0, BLOCK_SIZE)` with `mask = offs < N` is a loop over a real \
dimension of size `N`. `BLOCK_SIZE` is a hardware tiling detail — it is NOT a \
tensor dimension. Never emit `BLOCK_SIZE`; loop over the true size (from \
`.shape`) and drop the mask.
- `tl.load(ptr + offs, mask=...)` / `tl.store(...)` are reads/writes into \
tensors at affine index expressions. Convert the flat pointer arithmetic \
(`row * stride + col`) into multi-dimensional indexing with named Dims.
- `tl.sum(x, axis=0)` (and max/other reductions) become explicit reduction \
loop nests, exactly like the softmax decomposition rules below.
- Intra-kernel casts (e.g. `.to(tl.float32)`) are precision details; keep the \
tensor `dtype=` matching the kernel's inputs/output, ignore transient casts.

CRITICAL RULES:
- Do NOT optimize: no loop unrolling, constant folding, or performance tricks.
@@WORK_MINIMAL_RULES@@
- The code must capture the GENERAL structure of the computation, not be \
specialized to specific input values or a specific grid launch.
- Treat all inputs as symbolic — do NOT inline literal values. Numerical \
constants that are genuine hyperparameters (e.g. an epsilon) may be used.
- Every loop nest must be perfectly nested — computation only in the \
innermost body. A kernel may contain multiple sequential loop nests.
- For softmax: decompose into 4 perfectly nested loop nests \
(max-reduction, exp-shift, sum-reduction, normalize). Do NOT use a \
softmax() function call.
- Create output tensors with sizes from input `.shape`: \
`C = Tensor(B=A.shape[0], M=A.shape[1], N=W.shape[1], dtype=...)`. \
Do NOT hardcode literal dimension sizes.
- Every Tensor MUST include `dtype=` matching the kernel's precision. \
If unclear, use `dtype="fp32"`. \
Supported: fp64, fp32, tf32, fp16, bf16, fp8, int8, int32, int64, nvfp4. \
Use int64 for integer index tensors.
- Scalar inputs have shape `()`. Use `Tensor("", dtype=...)` for the \
annotation and index with `S[()]` in the kernel body. \
In `run_model`, load with `Tensor.from_numpy(inputs["key"], "", dtype=...)`.

OUTPUT FORMAT:
Produce a single Python code block containing:
1. `from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask`
2. One `@kernel` function per operation the kernel performs.
3. A `model()` function that composes the kernels.
4. A `run_model(inputs: dict) -> dict` function that:
   - Calls `clear_dims()`
   - The `inputs` dict contains ALL arrays returned by `get_inputs()` \
(activations and weights). Wrap each as a Tensor via `Tensor.from_numpy()`; \
use `role="weight"` for parameter/weight tensors.
   - Calls `model()` and returns `{"output": result.data}`
   - Do NOT create weights with `Tensor.from_random()` — use the \
arrays from `inputs`.
   - **Dim names in `Tensor.from_numpy()` must match the kernel signatures \
that consume those tensors.** Mismatched names cause \
`Dimension 'X' is not registered` errors at runtime.

Return ONLY the code block, no explanations.

REFERENCE:

"""


_PREAMBLE, _PREAMBLE_JAX, _PREAMBLE_TRITON = (
    p.replace(_WORK_MINIMAL_TOKEN, _WORK_MINIMAL_RULES)
    for p in (_PREAMBLE, _PREAMBLE_JAX, _PREAMBLE_TRITON)
)


def _load_reference(filename: str = "einsum_lib_prompt.md") -> str:
    """Load the einsum_lib prompt reference from docs/ (checkout or packaged copy)."""
    from solar.resources import find_resource
    path = find_resource("docs", filename)
    if path is not None:
        return path.read_text(encoding="utf-8")
    return f"({filename} reference file not found)"


def _assistant_tool_msg(msg) -> dict:
    """Serialize an assistant message that carries tool_calls into a dict the
    chat API accepts on the next turn."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ],
    }


def _extract_inspect_block(content: str) -> Optional[str]:
    """Pull a ```python_tool ...``` code block out of a response (text-protocol
    fallback used when the endpoint does not support native tool-calling)."""
    if content and "```python_tool" in content:
        return content.split("```python_tool", 1)[1].split("```", 1)[0].strip()
    return None


def _is_tools_unsupported(exc: Exception) -> bool:
    """Heuristic: did this error come from the endpoint rejecting the `tools`
    request (rather than a transient/server error)?  Any 400 while tools are in
    play is treated as unsupported so we can fall back to the text protocol."""
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    s = str(exc).lower()
    return "400" in s or "tool" in s


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TranslationAgent:
    """Translates PyTorch code to solar.einsum_lib, with verification + retry.

    Usage::

        agent = TranslationAgent()          # reads key from default file
        result = agent.translate(
            source_code=open("model.py").read(),
            input_shapes={"x": (2, 4, 64)},
            reference_outputs={"output": expected_array},
            all_inputs={"x": input_array},
        )
        print(result.code)
        print(result.verified)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_file: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = _DEFAULT_MODEL,
        temperature: float = 0.2,
        max_retries: int = 5,
        max_tokens: Optional[int] = None,
        rate_limit_retries: int = 3,
        rate_limit_delay: int = 60,
        verify: bool = True,
        thinking: bool = True,
        thinking_budget: int = 10000,
        fast: bool = True,
        source_type: str = "pytorch",
        use_fp64: bool = False,
        debug: bool = False,
        verify_timeout: Optional[int] = None,
        mem_budget_bytes: Optional[int] = None,
        python_tool: bool = False,
        tool_context: Optional[dict] = None,
        compare_min_tol: float = 0.0,
        *,
        # round 10 (agent-positional-30): keyword-only and LAST -- the positional slots that were valid before this
        # flag existed (api_key ... compare_min_tol) keep their meaning
        reprompt_on_numba_decline: bool = False,
    ):
        self.model = model or os.environ.get(_MODEL_ENV) or _DEFAULT_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens or _MODEL_REGISTRY.get(self.model, {}).get("max_tokens", 64000)
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_delay = rate_limit_delay
        self.verify = verify
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.fast = fast
        # Typed-decline routing (the default policy; round 10 adjudication agent-routing-scope-31:
        # TYPED-DECLINE-ONLY): a numba TYPED DECLINE of a generated program (``kernel_<code>:...``) is an infrastructure
        # refusal, not a model answer -- by default the fast agent verifies that program on the interpreter path
        # (execute_translated's own fallback) and spends no paid re-prompt on it.  An UNTYPED fast-path failure -- the
        # generated code crashing in the lowered module, a lowered module that does not import, no typed reason -- is the
        # code's own failure and keeps the previous ``execution_failed`` -> re-prompt behaviour, recorded with its untyped
        # class.  ``reprompt_on_numba_decline=True`` keeps the previous policy for typed declines as well.
        self.reprompt_on_numba_decline = reprompt_on_numba_decline
        self.use_fp64 = use_fp64
        self.debug = debug
        self.verify_timeout = verify_timeout
        self.mem_budget_bytes = mem_budget_bytes
        self.compare_min_tol = compare_min_tol
        self.python_tool = python_tool
        self._tool_context = dict(tool_context) if tool_context else {}
        # Persistent namespace for the run_python tool, seeded lazily on first
        # use with the (framework-neutral) tool context.
        self._py_ns: Optional[dict] = None
        self._py_calls = 0
        # Assume native tool-calling works until the endpoint rejects it, then
        # fall back to the text protocol.
        self._tools_supported = python_tool
        self._base_url = base_url or os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL
        if not self._base_url:
            raise ValueError(
                "No inference endpoint configured: pass base_url= or set "
                f"{_BASE_URL_ENV}. SOLAR ships no default endpoint and never "
                "falls back to a public provider."
            )

        if not self.model:
            raise ValueError(
                "No model configured: pass model= (CLI: --llm-model) or set "
                f"{_MODEL_ENV} to a model name your {_BASE_URL_ENV} endpoint "
                "serves. SOLAR ships no default model."
            )

        self._api_key = api_key or self._resolve_api_key(api_key_file)
        if not self._api_key:
            raise ValueError(
                "API key required: pass api_key=, api_key_file= or set "
                "OPENAI_API_KEY. SOLAR reads no implicit key file."
            )

        self._init_client()
        if source_type == "jax":
            self._system_prompt = _PREAMBLE_JAX + _load_reference("einsum_lib_prompt_jax.md")
        elif source_type == "triton":
            self._system_prompt = _PREAMBLE_TRITON + _load_reference()
        else:
            self._system_prompt = _PREAMBLE + _load_reference()
        if self.python_tool:
            self._system_prompt += _PYTHON_INSTRUCTION

    @staticmethod
    def _resolve_api_key(api_key_file: Optional[str] = None) -> Optional[str]:
        """Try to load the API key from file or environment."""
        # Explicit file path
        if api_key_file:
            path = Path(api_key_file)
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()

        # Environment variable
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key

        return None

    # -- OpenAI client -------------------------------------------------------

    def _init_client(self) -> None:
        if _openai_module is None:
            raise ImportError(
                "openai package required: pip install openai"
            )
        self.client = _openai_module.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=_openai_module.Timeout(connect=60, read=900, write=60, pool=60),
        )

    # -- Prompt helpers ------------------------------------------------------

    def _build_user_prompt(
        self,
        source_code: str,
        input_shapes: Dict[str, tuple],
        input_dtypes: Optional[Dict[str, str]] = None,
        input_sparsity: Optional[Dict[str, float]] = None,
        mask_fill_values: Optional[Dict[str, float]] = None,
        buffer_keys: Optional[set] = None,
        int_constants: Optional[Dict[str, int]] = None,
    ) -> str:
        activation_keys = sorted(k for k in input_shapes if k.startswith("x"))
        mask_keys = sorted(k for k in input_shapes if k.startswith("captured_mask_"))
        _buf = buffer_keys or set()
        learned_keys = sorted(
            k for k in input_shapes
            if not k.startswith("x") and not k.startswith("captured_mask_")
            and k not in _buf
        )
        buf_keys_sorted = sorted(
            k for k in input_shapes
            if k in _buf
        )
        sp = input_sparsity or {}
        fv = mask_fill_values or {}

        def _fmt(key):
            shape = input_shapes[key]
            dt = (input_dtypes or {}).get(key, "fp32")
            parts = f'shape: {shape}, dtype: "{dt}"'
            if key in sp:
                parts += f", sparsity: {sp[key]}"
            if key in fv:
                parts += f", masked_value: {fv[key]}"
            return f'"{key}": {{{parts}}}'

        lines = []
        lines.append(f"Source code:\n```python\n{source_code}\n```\n")

        act_str = ", ".join(_fmt(k) for k in activation_keys)
        lines.append(f"Activation inputs: {{{act_str}}}")

        if learned_keys:
            par_str = ", ".join(_fmt(k) for k in learned_keys)
            lines.append(f"Model parameters (learned weights, passed in inputs dict): {{{par_str}}}")

        if buf_keys_sorted:
            buf_str = ", ".join(_fmt(k) for k in buf_keys_sorted)
            lines.append(
                f"\nModel buffers (pre-computed, passed in inputs dict): {{{buf_str}}}\n"
                "Model buffers are pre-computed values captured from the trained model "
                "(e.g. running statistics for BatchNorm). Use them directly in your "
                "computation — do NOT recompute what they represent. Load each via "
                '`Tensor.from_numpy(inputs["<name>"], ...)` with `role="weight"`.'
            )

        if mask_keys:
            mask_str = ", ".join(_fmt(k) for k in mask_keys)
            lines.append(
                f"\nCaptured masks (from instrumented model run): {{{mask_str}}}\n"
                "These masks use the convention: 1 = keep, 0 = masked out.  "
                "Load each via `Tensor.from_numpy(inputs[\"<name>\"], ...)` "
                "and use `apply_mask(tensor, mask, masked_value=...)` in the "
                "`model()` function.  Use the `masked_value` listed above "
                "for each mask."
            )

        _ints = int_constants or {}
        if _ints:
            lines.append(
                f"\nInteger constants (read directly from inputs dict as plain int, "
                f"no Tensor.from_numpy): {_ints}"
            )

        lines.append(
            "\nAll tensors are provided as numpy arrays in the `inputs` dict "
            "passed to `run_model()`. Use `Tensor.from_numpy()` for all of them "
            "(with `role=\"weight\"` for parameter tensors). "
            "Integer constants are plain `int` values — read them directly."
            "\nSet `dtype=` on every Tensor to match the dtypes listed above."
        )
        if sp:
            lines.append(
                "For tensors with a `sparsity` value listed above, set "
                "`sparsity=<value>` on the `Tensor.from_numpy()` call. "
                "This records the density (fraction of non-zero elements) "
                "for downstream FLOPs/bytes analysis."
            )

        lines.append("")
        lines.append("Translate this into solar.einsum_lib code.")
        return "\n".join(lines)

    def _build_retry_prompt(self, error: str, previous_code: str) -> str:
        prompt = (
            f"The previous translation had an error:\n{error}\n\n"
            f"Previous attempt:\n```python\n{previous_code}\n```\n\n"
            "Fix the code and return the corrected version."
        )
        if self.python_tool:
            prompt += (
                "\n\nBefore rewriting, you SHOULD use run_python to investigate "
                "the cause against the actual `model` (e.g. read the relevant "
                "forward source, check which parameters/branches actually run, "
                "or reproduce the sub-computation) rather than guessing."
            )
        return prompt

    # -- LLM call ------------------------------------------------------------

    def _complete(self, messages: list, tools: Optional[list] = None):
        """Make one chat completion, with model fallback + rate-limit retry.

        Returns the raw message object (``choices[0].message``) so callers can
        inspect ``tool_calls`` as well as ``content``.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        _SERVER_ERRORS = {500, 502, 503, 504}
        models_to_try = [self.model] + [m for m in _FALLBACK_MODELS if m != self.model]
        for model_name in models_to_try:
            cfg = _MODEL_REGISTRY.get(model_name, {"max_tokens": self.max_tokens, "thinking_type": "enabled"})
            kwargs["model"] = model_name
            kwargs["max_tokens"] = cfg["max_tokens"]
            if self.thinking:
                kwargs["temperature"] = 1
                thinking_type = cfg.get("thinking_type", "enabled")
                thinking_cfg = {"type": thinking_type}
                if thinking_type == "enabled":
                    thinking_cfg["budget_tokens"] = self.thinking_budget
                kwargs["extra_body"] = {"thinking": thinking_cfg}
            for i in range(self.rate_limit_retries):
                # Transport budget (budget-1): when a dispatcher installed
                # a per-invocation budget, every transport request — including
                # this loop's model-fallback and rate-limit retries — is
                # authorized and preauthorized HERE, immediately before it is
                # sent; an unaffordable request raises before any call.
                from solar.einsum_lib.transport_budget import (charge_transport,
                                                               record_transport_usage)
                charge_transport(model_name)
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    # usage is plumbed out to the active budget (not dropped) so
                    # the invoker can reconcile measured cost (cost-13)
                    record_transport_usage(model_name, getattr(response, "usage", None))
                    return response.choices[0].message
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    is_rate_limit = (
                        status == 429 or "rate" in str(exc).lower()
                    )
                    is_timeout = isinstance(exc, (_openai_module.APITimeoutError,)) if _openai_module else False
                    is_server_error = status in _SERVER_ERRORS or is_timeout
                    if not (is_rate_limit or is_server_error):
                        raise
                    if is_server_error:
                        logger.info(f"{model_name} server error (HTTP {status}), "
                            f"falling back to next model...")
                        break
                    logger.info(f"{model_name} rate-limited (HTTP {status}), sleeping "
                        f"{self.rate_limit_delay}s before retry "
                        f"({i + 1}/{self.rate_limit_retries})...")
                    time.sleep(self.rate_limit_delay)
            else:
                logger.info(f"Exhausted retries for {model_name}, falling back...")
        # Terminal transport call after every fallback model was exhausted:
        # it is a transport request like any other and must be authorized
        # and preauthorized BEFORE it is sent (budget-12).
        from solar.einsum_lib.transport_budget import (charge_transport,
                                                       record_transport_usage)
        charge_transport(kwargs.get("model"))
        response = self.client.chat.completions.create(**kwargs)
        record_transport_usage(kwargs.get("model"), getattr(response, "usage", None))
        return response.choices[0].message

    def _run_python(self, code: str, max_output_chars: int = 8000,
                    max_calls: int = 30) -> str:
        """Execute agent-provided Python in a persistent namespace and return
        captured stdout.

        Framework-agnostic: it is a plain in-process Python REPL. The namespace
        persists across calls within this translation and is seeded once with
        the (neutral) tool context (e.g. ``source_path``, ``input_shapes``) plus
        ``np``. The agent imports/instantiates whatever else it needs itself.

        NOTE: this runs agent-provided Python in-process. It is bounded by an
        output cap and a call budget but is not a hard security sandbox.
        """
        import io
        import contextlib

        if self._py_ns is None:
            self._py_ns = {"np": np}
            self._py_ns.update(self._tool_context)

        self._py_calls += 1
        if self._py_calls > max_calls:
            return (f"ERROR: run_python call budget exhausted ({max_calls}). "
                    "Proceed with what you have.")
        if not code or not code.strip():
            return "ERROR: no code provided."

        if self.debug:
            logger.debug(f"run_python running:\n{code.strip()[:600]}")

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, self._py_ns)  # noqa: S102 - intentional agent-driven exec
        except Exception as exc:
            out = buf.getvalue() + f"\n[EXCEPTION] {type(exc).__name__}: {exc}"
            return out[:max_output_chars]

        out = buf.getvalue()
        if not out.strip():
            out = "(no stdout produced; use print(...) to emit output)"
        if len(out) > max_output_chars:
            out = out[:max_output_chars] + "\n...[truncated]"
        if self.debug:
            logger.debug(f"run_python returned {len(out)} chars")
        return out

    def _call_llm(self, messages: list) -> str:
        """Drive the LLM to a final text answer.

        When the python tool is enabled, the model may call the ``run_python``
        tool (native tool-calling, with a ```python_tool``` text-protocol
        fallback) any number of times up to ``_MAX_TOOL_ROUNDS`` before
        returning code.
        """
        if not self.python_tool:
            return self._complete(messages).content or ""

        import json as _json

        for _ in range(_MAX_TOOL_ROUNDS):
            if self._tools_supported:
                try:
                    msg = self._complete(messages, tools=[_PYTHON_TOOL])
                except Exception as exc:
                    if _is_tools_unsupported(exc):
                        if self.debug:
                            logger.debug("Endpoint rejected tools; "
                                  "switching to ```python_tool``` text protocol")
                        self._tools_supported = False
                        continue
                    raise
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    return msg.content or ""
                messages.append(_assistant_tool_msg(msg))
                for tc in tool_calls:
                    try:
                        args = _json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    out = self._run_python(args.get("code", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": out,
                    })
                continue

            # Text-protocol fallback (no native tool-calling support).
            content = self._complete(messages).content or ""
            block = _extract_inspect_block(content)
            if block is None:
                return content
            messages.append({"role": "assistant", "content": content})
            out = self._run_python(block)
            messages.append({
                "role": "user",
                "content": f"run_python result:\n```\n{out}\n```",
            })

        # Tool-round budget exhausted: force a final answer.
        if self.debug:
            logger.debug(f"run_python round budget "
                  f"({_MAX_TOOL_ROUNDS}) exhausted; forcing final answer")
        # Use a throwaway copy so the "stop calling tools" nudge does NOT leak
        # into the shared history (it would suppress tool use on later retry
        # attempts).
        forced_messages = messages + [{
            "role": "user",
            "content": (
                "Tool-call budget for this turn is exhausted. Do NOT call any "
                "more tools now. Return ONLY the final translated code block."
            ),
        }]
        # If native tool-calling is in use, the message history contains
        # tool_calls; some endpoints (e.g. Bedrock via litellm) reject a request
        # that has tool messages unless `tools=` is still supplied, so keep
        # passing the tool schema even on this final forced completion.
        final_tools = [_PYTHON_TOOL] if self._tools_supported else None
        return self._complete(forced_messages, tools=final_tools).content or ""

    # -- Code extraction -----------------------------------------------------

    @staticmethod
    def _extract_code(response: str) -> str:
        """Pull the first Python code block out of a markdown response."""
        if "```python" in response:
            return response.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in response:
            return response.split("```", 1)[1].split("```", 1)[0].strip()
        return response.strip()

    # -- Main entry ----------------------------------------------------------

    def translate(
        self,
        source_code: str,
        input_shapes: Dict[str, tuple],
        reference_outputs: Optional[Dict[str, "np.ndarray"]] = None,
        input_dtypes: Optional[Dict[str, str]] = None,
        all_inputs: Optional[Dict[str, "np.ndarray"]] = None,
        input_sparsity: Optional[Dict[str, float]] = None,
        mask_fill_values: Optional[Dict[str, float]] = None,
        full_inputs: Optional[Dict[str, "np.ndarray"]] = None,
        buffer_keys: Optional[set] = None,
        int_constants: Optional[Dict[str, int]] = None,
    ) -> TranslationResult:
        """Translate PyTorch source to einsum_lib code.

        Args:
            source_code: PyTorch Python source code.
            input_shapes: Mapping of input tensor names to shape tuples
                (e.g. ``{"x": (2, 4, 64)}``).
            reference_outputs: Pre-captured reference outputs from the
                original model (obtained via :func:`capture_model_run`).
                When provided, translated outputs are compared against these
                directly — no callable reference function is needed.
            input_dtypes: Optional mapping of input names to SOLAR dtype
                strings (e.g. ``{"x": "fp16", "weight": "fp16"}``).
            all_inputs: Complete pre-built numpy inputs dict.  When provided,
                ``execute_translated`` skips random generation and uses these
                arrays directly.  This ensures the translated code receives
                exactly the same inputs that produced *reference_outputs*.
            input_sparsity: Optional mapping of input names to density values
                (fraction of non-zero elements, 0-1).  Included in the LLM
                prompt so the generated code can set ``sparsity=`` on tensors.
            mask_fill_values: Optional mapping of captured mask names to the
                fill value from the original ``masked_fill`` call.  Included
                in the LLM prompt so the generated code uses the correct
                ``masked_value`` argument in ``apply_mask``.
            buffer_keys: Optional set of input names that originated from
                ``named_buffers()`` (pre-computed state, not learned weights).
                Presented separately in the LLM prompt so it knows to use
                them directly rather than recomputing what they represent.

        Returns:
            A ``TranslationResult`` with the generated code and verification
            status.
        """
        from solar.einsum_lib.numba_accel import numba_lower

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._build_user_prompt(
                source_code, input_shapes, input_dtypes,
                input_sparsity=input_sparsity,
                mask_fill_values=mask_fill_values,
                buffer_keys=buffer_keys,
                int_constants=int_constants,
            )},
        ]

        best_code = ""
        last_error: Optional[str] = None
        last_status = "mismatch"
        timing = TimingInfo()

        for attempt in range(1, self.max_retries + 1):
            if self.debug:
                logger.debug(f"Attempt {attempt}/{self.max_retries}: Calling LLM...")
            t_llm = time.time()
            raw_response = self._call_llm(messages)
            llm_elapsed = time.time() - t_llm
            timing.llm_seconds += llm_elapsed

            code = self._extract_code(raw_response)
            best_code = code
            if self.debug:
                logger.debug(f"LLM returned in {llm_elapsed:.1f}s, generated {len(code.splitlines())} lines:")
                logger.debug("-" * 40)
                logger.debug(code)
                logger.debug("-" * 40)

            if not self.verify:
                numba_code = numba_lower(code)
                return TranslationResult(
                    code=code, verified=False, attempts=attempt,
                    numba_code=numba_code, timing=timing,
                    verification_status="unverified",
                )

            # --- execution verification ---
            if self.debug:
                logger.debug(f"Running generated code (fast={self.fast})...")
            t_verify = time.time()
            try:
                translated_outputs, inputs_used = execute_translated(
                    code, input_shapes, input_dtypes=input_dtypes,
                    all_inputs=all_inputs, fast=self.fast,
                    use_fp64=self.use_fp64,
                    timeout=self.verify_timeout,
                    mem_budget_bytes=self.mem_budget_bytes,
                    # a typed numba decline re-prompts only when the old policy is requested; otherwise
                    # execute_translated falls back to the interpreter and the comparison below decides --
                    # for TYPED declines only (round 10): an untyped fast-path failure still raises here
                    fail_on_numba_decline=self.fast and self.reprompt_on_numba_decline,
                    fail_on_untyped_fast_failure=self.fast and not self.reprompt_on_numba_decline,
                )
            except RuntimeError as exc:
                timing.verify_seconds += time.time() - t_verify
                last_error = str(exc)
                last_status = "execution_failed"
                if self.debug:
                    logger.debug(f"Execution failed: {last_error[:200]}")
                if "timed out" in last_error:
                    best_code = code
                    break
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {"role": "user", "content": self._build_retry_prompt(last_error, code)}
                )
                continue
            verify_elapsed = time.time() - t_verify
            timing.verify_seconds += verify_elapsed
            if self.debug:
                logger.debug(f"Generated code executed in {verify_elapsed:.1f}s")

            # --- output comparison ---
            verification_status = "executed_no_reference"
            vacuous_outputs: tuple = ()
            applied_tolerance: Optional[dict] = None
            if reference_outputs is not None:
                if self.debug:
                    logger.debug(f"Comparing outputs...")
                # Tolerance policy (unchanged): fp64 execution applies the
                # documented default rtol=1e-4/atol=1e-5; fp32 execution
                # applies rtol=1e-4/atol=1e-4.  The applied values are
                # recorded on the result so a direct library user can see
                # which one governed the verdict.
                cmp_kwargs = {"rtol": 1e-4, "atol": 1e-5}
                if not self.use_fp64:
                    cmp_kwargs = {"atol": 1e-4, "rtol": 1e-4}
                    if attempt == 1:
                        print("  [WARN] fp32 verification (use_fp64=False) "
                              "applies atol=1e-4; the documented default "
                              "verification tolerance is rtol=1e-4/atol=1e-5 "
                              "(use_fp64=True)")
                if self.compare_min_tol:
                    cmp_kwargs["min_tol"] = self.compare_min_tol
                applied_tolerance = {"rtol": cmp_kwargs["rtol"],
                                     "atol": cmp_kwargs["atol"],
                                     "min_tol": cmp_kwargs.get("min_tol", 0.0)}
                vr: VerificationResult = compare_outputs(
                    translated_outputs, reference_outputs, **cmp_kwargs
                )
                if self.debug:
                    status = "MATCH" if vr.success else "MISMATCH"
                    err_str = f"{vr.max_abs_error:.2e}" if vr.max_abs_error is not None else "N/A"
                    logger.debug(f"Comparison: {status} (max_abs_error={err_str})")
                if not vr.success:
                    last_error = vr.error
                    last_status = "mismatch"
                    if self.debug:
                        logger.debug(f"Mismatch, retrying...")
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append(
                        {"role": "user", "content": self._build_retry_prompt(last_error, code)}
                    )
                    continue
                vacuous_outputs = tuple(vr.vacuous_outputs)
                verification_status = ("verified_vacuous" if vacuous_outputs
                                       else "verified")
            else:
                # Execution-only: the code ran, nothing was compared.  This
                # is NOT a numerical verification and is never reported as
                # one (audit finding A1).  The structural result (code,
                # metadata) is still returned below; there is nothing to
                # retry because nothing failed.
                last_error = ("no reference outputs: execution-only run is "
                              "not a numerical verification (pass "
                              "reference_outputs=/all_inputs=, or construct "
                              "the agent with verify=False)")
                print(f"  [WARN] {last_error}")

            if self.debug:
                logger.debug(f"Running trace pass (shape-only)...")
            t_trace = time.time()
            measured_metadata = {}
            from solar.einsum_lib.core import density_profiling_enabled
            if density_profiling_enabled():
                # Fast verification does not execute the interpreter hooks,
                # and shape-only tracing has no real values. Profile a
                # separate full-value pass and return its trace from the
                # guarded child, rather than losing thread-local trace state.
                from solar.einsum_lib.verify import _run_isolated

                try:
                    measured_metadata = _run_isolated(
                        _profile_measured_density, self.verify_timeout or 300,
                        self.mem_budget_bytes, args=(code, inputs_used))
                except Exception as exc:
                    print(f"[density] measurement failed ({exc}); "
                          "continuing without measured densities")
            try:
                trace_inputs = full_inputs if full_inputs is not None else all_inputs
                execute_translated(
                    code, input_shapes, input_dtypes=input_dtypes,
                    all_inputs=trace_inputs, trace=True, shape_only=True,
                )
                metadata = build_metadata(get_trace())
                from solar.einsum_lib.metadata import merge_measured_density
                merge_measured_density(metadata, measured_metadata)
            except Exception as exc:
                print(f"  [WARN] Trace step failed: {exc}")
                metadata = None
            trace_elapsed = time.time() - t_trace
            timing.trace_seconds += trace_elapsed
            if self.debug:
                logger.debug(f"Trace completed in {trace_elapsed:.1f}s")

            numba_code = numba_lower(code) if self.fast else None
            if self.debug:
                logger.debug(f"Saving outputs...")
            compared = reference_outputs is not None
            return TranslationResult(
                code=code, verified=compared, attempts=attempt,
                verification_error=None if compared else last_error,
                metadata=metadata, numba_code=numba_code,
                timing=timing,
                verification_status=verification_status,
                vacuous_outputs=vacuous_outputs,
                applied_tolerance=applied_tolerance,
            )

        # Exhausted retries
        numba_code = numba_lower(best_code) if self.fast else None
        return TranslationResult(
            code=best_code,
            verified=False,
            verification_error=last_error,
            attempts=self.max_retries,
            numba_code=numba_code,
            timing=timing,
            verification_status=last_status,
        )
