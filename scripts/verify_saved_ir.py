#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify an already-generated einsum_lib program against its source model.

This re-runs only the verification half of the pipeline: it builds the reference
inputs and outputs from the source model, executes a saved
``*_einsum_lib.py`` against them, and compares.  No LLM is involved, so a
translation that timed out during verification can be retried with a longer
budget -- or profiled -- without paying for another round of translation.

Sizing is controlled by ``--config``, which rewrites fields of the source
module's ``CONFIG`` dict before ``get_inputs()`` runs.  That matters for
JAXBench, whose ``workload`` reads its architectural dimensions out of
``CONFIG`` rather than from the shapes of its arguments: shrinking a tensor
without shrinking ``CONFIG`` makes the reference crash, and shrinking ``CONFIG``
keeps every derived shape mutually consistent.  ``--cap-input-size`` is
deliberately *not* used, so no capping LLM call happens either.

The verification core lives in ``solar.translate._runner.run_verification``;
this script is the CLI wrapper (``solar.translate.verify`` is the library
form of the same code path).

Usage:
  python3 scripts/verify_saved_ir.py MODEL_PY GENERATED_IR_PY [options]

Examples:
  # Reproduce what the pipeline verified, with no time limit.
  python3 scripts/verify_saved_ir.py \\
      jaxbench/solar_output/3p_MLA_Attention/solar_model.py \\
      /tmp/3p_MLA/solar_model_einsum_lib.py \\
      --config batch=1 --config seq_len=128

  # Shrink the architecture too, to isolate whether size is the only problem.
  python3 scripts/verify_saved_ir.py MODEL.py IR.py \\
      --config batch=1 --config seq_len=128 --config num_heads=8

  # Time the reference and the IR separately without comparing.
  python3 scripts/verify_saved_ir.py MODEL.py IR.py --config batch=1 --shape-only

  # Emit the numerical-validation receipt the strict SOL gate consumes
  # (same artifact as `solar-translate verify --emit-receipt`).
  python3 scripts/verify_saved_ir.py adapter.py solar_model_einsum_lib.py \\
      --source-type cuda --emit-receipt receipt.json \\
      --definition <problem_dir> --dtype fp16

Exit status:
  0  VERIFIED (also --shape-only, and a vacuous all-zero-reference pass,
     unless --strict is given)
  1  MISMATCH or IR execution failure
  2  usage error / model or IR file not found
  3  --strict asked for numerical proof and the run is shape-only or a
     vacuous pass; or --emit-receipt was requested and the receipt was
     refused (printed as RECEIPT REFUSED: ...)
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) in sys.path:  # the checkout this script lives in must take precedence
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
if os.environ.get("SOLAR_ENGINE_SOURCE_PROBE"):  # provenance self-binding probe (tests/diagnostics)
    from solar.engine_source import probe_hook  # noqa: E402
    probe_hook()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a saved einsum_lib program against its source model, "
                    "with no LLM calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Source model .py (the pipeline's input).")
    parser.add_argument("ir", help="Generated *_einsum_lib.py to verify. Pass the "
                                  "DSL file, not the _numba.py one -- numba "
                                  "conversion happens here.")
    parser.add_argument("--source-type", default="jax",
                        choices=["jax", "triton", "cuda", "pytorch"],
                        help="Frontend that produced the reference (default: jax).")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a CONFIG field before building inputs. "
                             "Repeatable.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Seconds allowed for IR execution (default: no limit).")
    parser.add_argument("--mem-budget", type=float, default=None, metavar="GB",
                        help="RSS budget in GB for IR execution. Runs the IR in a "
                             "forked subprocess killed on breach (MemoryError) "
                             "instead of dying to the system OOM-killer.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--no-fast", action="store_true",
                        help="Interpret the IR directly instead of converting to "
                             "numba. Far slower, but rules the converter out.")
    parser.add_argument("--shape-only", action="store_true",
                        help="Build the reference and execute the IR for shapes "
                             "only, skipping the numeric comparison.")
    parser.add_argument("--fp32", action="store_true",
                        help="Verify at fp32 instead of the default fp64.")
    parser.add_argument("--quiet", action="store_true", help="Suppress frontend debug output.")
    parser.add_argument("--strict", action="store_true",
                        help="Require numerical proof: --shape-only runs and "
                             "vacuous (all-zero reference) passes exit 3 "
                             "instead of 0.")
    parser.add_argument("--emit-receipt", default=None, metavar="PATH",
                        help="Write a numerical-validation receipt bound to this "
                             "run (requires --definition and --dtype). Receipt "
                             "refusal exits 3.")
    parser.add_argument("--definition", default=None, metavar="DIR",
                        help="Problem dir with definition.json [+ workload.jsonl] "
                             "the receipt is bound to.")
    parser.add_argument("--workload-idx", type=int, default=0,
                        help="workload.jsonl line the receipt names (default 0).")
    parser.add_argument("--dtype", default=None,
                        help="Analysis precision the receipt admits (must equal "
                             "the later sol --precision).")
    args = parser.parse_args()

    from solar.translate._runner import (EXIT_USAGE, check_input_files,
                                         coerce_config_value, run_verification)

    model_path = Path(args.model).resolve()
    ir_path = Path(args.ir).resolve()
    err = check_input_files(model_path, ir_path)
    if err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    if args.emit_receipt and (args.definition is None or args.dtype is None):
        parser.error("--emit-receipt requires --definition and --dtype")

    overrides = {}
    for item in args.config:
        if "=" not in item:
            print(f"ERROR: --config expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, _, value = item.partition("=")
        overrides[key.strip()] = coerce_config_value(value)

    result = run_verification(
        model_path, ir_path,
        source_type=args.source_type,
        config_overrides=overrides,
        timeout=args.timeout,
        mem_budget_gb=args.mem_budget,
        rtol=args.rtol,
        atol=args.atol,
        fast=not args.no_fast,
        shape_only=args.shape_only,
        use_fp64=not args.fp32,
        quiet=args.quiet,
    )
    from solar.translate import finalize_verify
    return finalize_verify(
        result, model_path=model_path, ir_path=ir_path, strict=args.strict,
        emit_receipt=args.emit_receipt, definition_dir=args.definition,
        dtype=args.dtype, workload_idx=args.workload_idx)


if __name__ == "__main__":
    sys.exit(main())
