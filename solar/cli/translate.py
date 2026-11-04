# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the dual-mode translation facade (``solar-translate``).

Subcommands:
  scaffold  — agent-mode workspace: adapter stub + NOTES + skill checklist
              (the DEFAULT: ``solar-translate <problem_dir>`` scaffolds)
  verify    — verify a saved DSL against its source (library form of
              scripts/verify_saved_ir.py; same flags, same exit status:
              0 verified, 1 mismatch/failure, 2 usage/missing file,
              3 only with --strict/--emit-receipt when no numerical proof
              can be delivered)
  sol       — original-shape trace -> einsum graph -> analysis -> SOL perf
              (prints a ``validation:`` line; --strict gates on a receipt)
  api       — the LLM pipeline (explicit opt-in; demands --api-key or
              OPENAI_API_KEY; never the default)
"""

from __future__ import annotations
from solar.analysis.mac_presentation import describe_macs

import argparse
import os
import sys
from pathlib import Path

_SUBCOMMANDS = ("scaffold", "verify", "sol", "api")
_SOURCE_TYPES = ["cuda", "pytorch", "jax", "triton"]
_API_SOURCE_TYPES = ["pytorch", "raw", "jax", "triton", "cuda"]


def _add_scaffold(sub):
    p = sub.add_parser("scaffold", help="Agent-mode workspace (DEFAULT mode)")
    p.add_argument("problem_dir", help="Benchmark problem dir (definition.json, "
                                       "reference.py) or source file")
    p.add_argument("--out-dir", default=None,
                   help="Workspace dir (default: <problem_dir>/manual)")
    p.add_argument("--source-type", default="cuda", choices=_SOURCE_TYPES)


def _add_verify(sub):
    p = sub.add_parser("verify", help="Verify a saved DSL against its source")
    p.add_argument("model", help="Source model .py / adapter.py")
    p.add_argument("ir", help="Saved *_einsum_lib.py")
    p.add_argument("--source-type", default="cuda", choices=_SOURCE_TYPES)
    p.add_argument("--mem-budget", type=float, default=None, metavar="GB")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--shape-only", action="store_true",
                   help="Execute for shapes only; no numeric comparison "
                        "(exit 0 unless --strict).")
    p.add_argument("--strict", action="store_true",
                   help="Require numerical proof: shape-only runs and "
                        "vacuous (all-zero reference) passes exit 3.")
    p.add_argument("--emit-receipt", default=None, metavar="PATH",
                   help="Write a numerical-validation receipt (requires "
                        "--definition and --dtype); the strict SOL gate "
                        "consumes it. Receipt refusal exits 3.")
    p.add_argument("--definition", default=None, metavar="DIR",
                   help="Problem dir with definition.json [+ workload.jsonl] "
                        "the receipt is bound to.")
    p.add_argument("--workload-idx", type=int, default=0,
                   help="workload.jsonl line the receipt names (default 0).")
    p.add_argument("--dtype", default=None,
                   help="Analysis precision the receipt admits (must equal "
                        "the later sol --precision).")


def _add_sol(sub):
    p = sub.add_parser("sol", help="Original-shape einsum graph + SOL perf")
    p.add_argument("ir", help="Verified *_einsum_lib.py")
    p.add_argument("--definition", required=True, help="Problem dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arch", default="B200")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--fallback-adapter", default=None)
    p.add_argument("--custom-inputs", default=None)
    p.add_argument("--measure-density", action="store_true")
    p.add_argument("--density-override", action="append", default=[],
                   metavar="NAME=VALUE")
    p.add_argument("--workload-idx", type=int, default=0)
    p.add_argument("--perf-precision", default=None,
                   help="Bill the perf stage at this precision (default: "
                        "--precision).")
    # strict receipt gate (library parity, A7)
    p.add_argument("--strict", action="store_true",
                   help="Refuse to trace unless --receipt hash-matches the "
                        "current inputs and carries a numerical pass.")
    p.add_argument("--receipt", default=None,
                   help="Receipt from `verify --emit-receipt` (with --strict).")
    p.add_argument("--source-path", default=None,
                   help="The source/adapter the receipt validated (with --strict).")
    # forwarded to scripts/trace_saved_ir.py verbatim (A7)
    p.add_argument("--axis", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--mem-budget", type=float, default=None, metavar="GB")
    p.add_argument("--measure-timeout", type=int, default=None)
    p.add_argument("--measure-attention-density", action="store_true")
    p.add_argument("--allow-unbound-constants", action="store_true")


def _add_api(sub):
    p = sub.add_parser(
        "api", help="LLM pipeline (EXPLICIT opt-in; requires an API key)")
    p.add_argument("source", help="Source model .py for toDSL_model")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-type", default="pytorch", choices=_API_SOURCE_TYPES)
    p.add_argument("--api-key", default=None,
                   help="Inference API key (or OPENAI_API_KEY env). Without "
                        "one this subcommand refuses — it never searches for "
                        "key files.")
    p.add_argument("--cap-input-size", action="store_true")
    p.add_argument("--mem-budget", type=float, default=None)
    p.add_argument("--verify-timeout", type=int, default=None)


def main() -> int:
    argv = sys.argv[1:]
    # Default mode is agent: a bare problem dir means `scaffold`.
    if argv and not argv[0].startswith("-") and argv[0] not in _SUBCOMMANDS:
        argv = ["scaffold"] + argv

    parser = argparse.ArgumentParser(
        prog="solar-translate", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_scaffold(sub)
    _add_verify(sub)
    _add_sol(sub)
    _add_api(sub)
    args = parser.parse_args(argv)
    from solar.common.cli_logging import configure as _configure_cli_logging
    _configure_cli_logging(bool(getattr(args, "debug", False) or getattr(args, "verbose", False)))

    from solar import translate as T

    if args.cmd == "scaffold":
        problem = Path(args.problem_dir)
        # A source FILE scaffolds against its parent directory (same rule as
        # translate(mode="agent")) — otherwise the stub points at
        # <file>/reference.py, a path that cannot exist.
        if problem.is_file():
            problem = problem.parent
        out = Path(args.out_dir) if args.out_dir else problem / "manual"
        res = T.scaffold(problem, out, source_type=args.source_type)
        for f in res.files:
            print(f"wrote {f}")
        print()
        print(res.checklist)
        return 0

    if args.cmd == "verify":
        err = T.check_input_files(args.model, args.ir)
        if err:
            print(err, file=sys.stderr)
            return T.EXIT_USAGE
        if args.emit_receipt and (args.definition is None or args.dtype is None):
            parser.error("--emit-receipt requires --definition and --dtype")
        res = T.verify(args.model, args.ir, source_type=args.source_type,
                       mem_budget_gb=args.mem_budget, rtol=args.rtol,
                       atol=args.atol, timeout=args.timeout,
                       shape_only=args.shape_only)
        return T.finalize_verify(
            res, model_path=args.model, ir_path=args.ir, strict=args.strict,
            emit_receipt=args.emit_receipt, definition_dir=args.definition,
            dtype=args.dtype, workload_idx=args.workload_idx)

    if args.cmd == "sol":
        overrides = {}
        for item in args.density_override:
            name, _, value = item.partition("=")
            overrides[name.strip()] = float(value)
        axes = {}
        for item in args.axis:
            name, _, value = item.partition("=")
            axes[name.strip()] = int(value)
        from solar.translate.receipts import ReceiptRefusal
        try:
            res = T.sol(args.ir, args.definition, args.out_dir, arch=args.arch,
                        precision=args.precision,
                        fallback_adapter=args.fallback_adapter,
                        custom_inputs=args.custom_inputs,
                        measure_density=args.measure_density,
                        density_overrides=overrides,
                        workload_idx=args.workload_idx,
                        perf_precision=args.perf_precision,
                        strict=args.strict, receipt=args.receipt,
                        source_path=args.source_path,
                        axes=axes or None, mem_budget_gb=args.mem_budget,
                        measure_timeout=args.measure_timeout,
                        measure_attention_density=args.measure_attention_density,
                        allow_unbound_constants=args.allow_unbound_constants)
        except ReceiptRefusal as e:
            print(f"REFUSED: strict SOL: {e}", file=sys.stderr)
            return T.EXIT_NO_PROOF
        print(f"SOL fused={res.fused_ms:.6g} ms  unfused={res.unfused_ms:.6g} ms  "
              f"macs={describe_macs({'macs': res.macs, 'macs_typed': res.macs_typed})}")   # issue #58: typed, never a bare aggregate
        print(f"perf: {res.perf_path}")
        print(res.validation_line)
        return 0

    if args.cmd == "api":
        key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            print(f"ERROR: {T.API_POLICY}", file=sys.stderr)
            return 2
        # translate() forwards these as toDSL_model flags and skips None/False.
        res = T.translate(args.source, args.out_dir, mode="api",
                          source_type=args.source_type, api_key=key,
                          cap_input_size=args.cap_input_size,
                          mem_budget=args.mem_budget,
                          verify_timeout=args.verify_timeout)
        print(res.log[-4000:])
        print(f"api translation {'VERIFIED' if res.verified else 'NOT VERIFIED'}")
        return 0 if res.verified else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
