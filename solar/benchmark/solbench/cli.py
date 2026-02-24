#!/usr/bin/env python3
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

"""CLI for SolBench model preparation."""

import argparse
import sys
from pathlib import Path

from .parser import SolBenchParser
from .generator import SolBenchGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SolBench models for Solar analysis."
    )
    
    parser.add_argument(
        "--solbench-dir",
        type=Path,
        required=True,
        help="Path to sol-bench directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated models"
    )
    
    parser.add_argument(
        "--index",
        type=str,
        nargs="+",
        help="Model indices to process (e.g., 0000 0001)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Model name pattern to match"
    )
    
    parser.add_argument(
        "--max-models",
        type=int,
        help="Maximum number of models to generate"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Check solbench directory
    if not args.solbench_dir.exists():
        print(f"Error: SolBench directory not found: {args.solbench_dir}")
        sys.exit(1)
    
    # List mode
    if args.list:
        parser = SolBenchParser(args.solbench_dir)
        models = parser.list_models()
        print(f"Found {len(models)} SolBench model(s):")
        for index, name in models:
            print(f"  [{index}] {name}")
        sys.exit(0)
    
    # Generate models
    generator = SolBenchGenerator(
        args.solbench_dir,
        args.output_dir,
        debug=args.debug
    )
    
    if args.name:
        # Generate by name
        output = generator.generate_by_name(args.name, force=args.force)
        if output:
            print(f"Generated: {output}")
    elif args.index:
        # Generate by indices
        for idx in args.index:
            output = generator.generate_by_index(idx, force=args.force)
            if output:
                print(f"Generated: {output}")
    else:
        # Generate all
        outputs = generator.generate_all(
            max_models=args.max_models,
            force=args.force
        )
        print(f"Generated {len(outputs)} model(s)")
        for output in outputs:
            print(f"  {output.name}")


if __name__ == "__main__":
    main()
