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

"""CLI entry point: translate a single PyTorch model to solar.einsum_lib DSL.

Usage:
    solar-toDSL-model model.py
    solar-toDSL-model model.py -o output/dsl/
    solar-toDSL-model model.py -o output/dsl/ --save-on-fail
    solar-toDSL-model model.py --no-verify
"""

import argparse
import sys
from pathlib import Path

from solar.einsum_lib.pipeline import (
    SUPPORTED_SOURCE_TYPES,
    translate_file,
)


def _run_translate(args, model_path, mem_budget_bytes):
    return translate_file(
        file_path=str(model_path),
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
        fast=not args.no_numba,
        cap_input_size=args.cap_input_size,
        use_fp64=not args.fp32,
        debug=args.debug,
        verify_timeout=args.verify_timeout,
        max_cap_rounds=args.max_cap_rounds,
        mem_budget_bytes=mem_budget_bytes,
        llm_model=args.llm_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate a single PyTorch model to solar.einsum_lib DSL code.",
    )
    parser.add_argument(
        "model",
        help="Path to a PyTorch model .py file (must define Model class and get_inputs()).",
    )
    parser.add_argument(
        "--source-type", "-s",
        choices=SUPPORTED_SOURCE_TYPES,
        default="pytorch",
        help="Source language/format (default: pytorch).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save generated DSL code and metadata YAML.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum LLM translation attempts (default: 5).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens for LLM response (default: auto from model registry).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical verification (translate only).",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help="Path to file containing the API key (alternative to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--input-shapes",
        default=None,
        help='JSON dict of input shapes (required for --source-type raw). '
             'Example: \'{"x": [2, 3, 4], "y": [3, 4]}\'',
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended thinking / reasoning for the LLM.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=10000,
        help="Token budget for extended thinking (default: 10000).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=3,
        help="Max retries on rate-limit (429) errors (default: 3).",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=int,
        default=60,
        help="Seconds to sleep between rate-limit retries (default: 60).",
    )
    parser.add_argument(
        "--save-on-fail",
        action="store_true",
        help="Save and print generated code even when verification fails.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba JIT acceleration; run verification with pure Python loops.",
    )
    parser.add_argument(
        "--cap-input-size",
        action="store_true",
        help=(
            "Use the LLM to intelligently shrink free input dimensions (batch, "
            "sequence length, spatial) for fast verification while preserving "
            "architectural dimensions tied to model weights. The trace pass "
            "still uses original full shapes for accurate SOL analysis."
        ),
    )
    parser.add_argument(
        "--fp64",
        action="store_true",
        default=True,
        help="Upcast to fp64 for verification when model uses only fp32+ "
             "(default). Auto-skipped for bf16/fp16/fp8 models.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Force verification in native precision (no fp64 upcast).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print LLM-generated code before each verification attempt.",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=None,
        help="Maximum seconds per verification attempt. If exceeded, the "
             "attempt counts as failed and the agent retries (default: no limit).",
    )
    parser.add_argument(
        "--max-cap-rounds",
        type=int,
        default=2,
        help="Maximum capping rounds when --cap-input-size is enabled. "
             "If translation fails, the capping LLM re-proposes shapes "
             "and translation restarts with a fresh reference (default: 2).",
    )
    parser.add_argument(
        "--mem-budget",
        type=float,
        default=None,
        help="Memory budget in GB for verification tensor allocation. "
             "Used by both the memory guard and the capping LLM prompt. "
             "Auto-detected from system RAM when omitted.",
    )
    parser.add_argument(
        "--measure-density",
        action="store_true",
        help="Enable density profiling during the full-value verification/trace "
             "runs: eligible inputs and kernel outputs' EXACT density "
             "(count_nonzero/size) is recorded as 'sparsity' in the generated "
             "yaml, feeding the analyzer's zeros-are-free discounts. With "
             "--cap-input-size, changed-shape measurements are refused; use "
             "trace_saved_ir for original-scale revalidation. Attention "
             "probabilities stay dense unless --measure-attention-density is set.",
    )
    parser.add_argument(
        "--measure-attention-density", action="store_true",
        help="With --measure-density, also measure attention probabilities "
             "identified by softmax/probability names or role='attention_probability'. "
             "Changes the dense-attention accounting convention.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model identifier for translation (default: the SOLAR_LLM_MODEL environment "
             "variable; required). Must be a model your SOLAR_API_BASE_URL endpoint serves.",
    )

    args = parser.parse_args()

    from solar.common.cli_logging import configure as _configure_cli_logging

    _configure_cli_logging(bool(getattr(args, "debug", False) or getattr(args, "verbose", False)))
    if args.measure_attention_density and not args.measure_density:
        parser.error("--measure-attention-density requires --measure-density")
    if args.measure_density and args.no_verify:
        parser.error("--measure-density requires verification (remove --no-verify)")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)
    if model_path.is_dir():
        print("Error: expected a single .py file, not a directory.")
        print("Hint: to process a directory use scripts/to_einsum_lib.py")
        sys.exit(1)

    mem_budget_bytes = int(args.mem_budget * 1e9) if args.mem_budget is not None else None

    if args.measure_density:
        # Scoped for the whole translate call: the profiler only acts during
        # REAL-VALUE executions (trace-only/shape-only runs skip measurement),
        # so this instruments exactly the verification/trace passes.
        from solar.einsum_lib import set_density_profiling
        label = "measured_capped" if args.cap_input_size else "measured"
        set_density_profiling(True, source=label,
                              measure_attention=args.measure_attention_density)
        if args.cap_input_size:
            print("[density] profiling ON — densities will be measured at "
                  "CAPPED verification shapes (sparsity_source: "
                  "measured_capped); changed-shape measurements stay dense. "
                  "Use trace_saved_ir for original-scale revalidation.")
        else:
            print("[density] profiling ON (sparsity_source: measured)")

    try:
        result = _run_translate(args, model_path, mem_budget_bytes)
    finally:
        if args.measure_density:
            from solar.einsum_lib import set_density_profiling
            set_density_profiling(False)

    if result.code and (result.verified or args.save_on_fail):
        tag = "VERIFIED" if result.verified else "UNVERIFIED"
        print("\n" + "=" * 60)
        print(f"GENERATED CODE ({tag}):")
        print("=" * 60)
        print(result.code)
        print("=" * 60)

    sys.exit(0 if result.verified else 1)


if __name__ == "__main__":
    main()
