#!/usr/bin/env python3
"""CLI for generating FlashInfer benchmark models.

Usage:
    python -m solar.benchmark.flashinfer.cli --trace-dir /path/to/flashinfer-trace --output-dir /path/to/output
    python -m solar.benchmark.flashinfer.cli --op-type gemm --name gemm_n128_k2048
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate kernelbench-style models from FlashInfer trace definitions."
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("flashinfer-trace"),
        help="Path to flashinfer-trace directory (default: flashinfer-trace)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("flashinfer-trace-postprocess"),
        help="Output directory for generated models (default: flashinfer-trace-postprocess)",
    )
    parser.add_argument(
        "--op-type",
        type=str,
        help="Only process specific operation type (e.g., gemm, rmsnorm)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Only process specific definition name (requires --op-type)",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        default=None,
        help="Maximum workloads per definition (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available definitions and exit",
    )

    args = parser.parse_args()

    # Import here to avoid slow imports for --help
    from solar.benchmark.flashinfer.parser import FlashInferParser
    from solar.benchmark.flashinfer.generator import FlashInferModelGenerator

    if not args.trace_dir.exists():
        print(f"Error: Trace directory not found: {args.trace_dir}")
        print("Download from: https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace")
        sys.exit(1)

    parser_obj = FlashInferParser(args.trace_dir)

    if args.list:
        print("Available definitions:")
        for op_type in parser_obj.list_op_types():
            print(f"\n{op_type}/")
            for name in parser_obj.list_definitions(op_type):
                print(f"  - {name}")
        sys.exit(0)

    generator = FlashInferModelGenerator(parser_obj, args.output_dir)

    if args.name:
        if not args.op_type:
            print("Error: --name requires --op-type")
            sys.exit(1)
        print(f"Generating models for {args.op_type}/{args.name}...")
        files = generator.generate_all_for_definition(
            args.op_type, args.name, args.max_workloads
        )
    elif args.op_type:
        print(f"Generating models for all {args.op_type} definitions...")
        files = []
        for name in parser_obj.list_definitions(args.op_type):
            files.extend(
                generator.generate_all_for_definition(
                    args.op_type, name, args.max_workloads
                )
            )
    else:
        print("Generating models for all definitions...")
        files = generator.generate_all(args.max_workloads)

    print(f"\nGenerated {len(files)} files to {args.output_dir}")
    for f in files[:10]:
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")


if __name__ == "__main__":
    main()
