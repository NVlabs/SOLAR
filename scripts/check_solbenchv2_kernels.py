#!/usr/bin/env python3.10
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

"""
Check SolBench v2 kernels for required methods and output status to CSV.

This script parses Python files in sol-bench/data/benchmark and checks for:
- get_inputs() function
- ReferenceModel class OR reference_backward function
- launch_reference_implementation() function

Output: CSV file with kernel status (sol-bench-v2.csv)
"""

import os
import ast
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def check_python_file(filepath: str) -> Dict[str, bool]:
    """
    Parse a Python file and check for required methods/classes.
    
    Args:
        filepath: Path to the Python file
        
    Returns:
        Dictionary with status of each required element
    """
    result = {
        "has_get_inputs": False,
        "has_ReferenceModel": False,
        "has_reference_backward": False,
        "has_launch_reference_implementation": False,
    }
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Check for function definitions
            if isinstance(node, ast.FunctionDef):
                if node.name == "get_inputs":
                    result["has_get_inputs"] = True
                elif node.name == "reference_backward":
                    result["has_reference_backward"] = True
                elif node.name == "launch_reference_implementation":
                    result["has_launch_reference_implementation"] = True
            
            # Check for class definitions
            elif isinstance(node, ast.ClassDef):
                if node.name == "ReferenceModel":
                    result["has_ReferenceModel"] = True
                    
    except SyntaxError as e:
        print(f"  Warning: Syntax error in {filepath}: {e}")
    except Exception as e:
        print(f"  Warning: Error parsing {filepath}: {e}")
    
    return result


def scan_benchmark_directory(benchmark_dir: str) -> List[Dict]:
    """
    Scan the benchmark directory and check all Python files.
    
    Args:
        benchmark_dir: Path to sol-bench/data/benchmark
        
    Returns:
        List of dictionaries with file info and status
    """
    results = []
    
    # Scan L1, L2, Quant subdirectories
    levels = ["L1", "L2", "Quant"]
    
    for level in levels:
        level_dir = Path(benchmark_dir) / level
        if not level_dir.exists():
            print(f"Warning: Directory not found: {level_dir}")
            continue
        
        py_files = sorted(level_dir.glob("*.py"))
        print(f"Found {len(py_files)} files in {level}")
        
        for py_file in py_files:
            print(f"  Checking: {py_file.name}")
            status = check_python_file(str(py_file))
            
            # Determine if kernel is valid (has required methods)
            has_reference = status["has_ReferenceModel"] or status["has_reference_backward"]
            is_valid = (
                status["has_get_inputs"] and 
                has_reference and 
                status["has_launch_reference_implementation"]
            )
            
            results.append({
                "level": level,
                "filename": py_file.name,
                "filepath": str(py_file),
                "has_get_inputs": status["has_get_inputs"],
                "has_ReferenceModel": status["has_ReferenceModel"],
                "has_reference_backward": status["has_reference_backward"],
                "has_launch_reference_implementation": status["has_launch_reference_implementation"],
                "is_valid": is_valid,
            })
    
    return results


def write_csv(results: List[Dict], output_path: str) -> None:
    """Write results to CSV file."""
    if not results:
        print("No results to write")
        return
    
    fieldnames = [
        "level",
        "filename",
        "has_get_inputs",
        "has_ReferenceModel",
        "has_reference_backward",
        "has_launch_reference_implementation",
        "is_valid",
        "filepath",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nWritten results to: {output_path}")


def print_summary(results: List[Dict]) -> None:
    """Print summary statistics."""
    total = len(results)
    valid = sum(1 for r in results if r["is_valid"])
    
    # Per-level stats
    level_stats = {}
    for r in results:
        level = r["level"]
        if level not in level_stats:
            level_stats[level] = {"total": 0, "valid": 0}
        level_stats[level]["total"] += 1
        if r["is_valid"]:
            level_stats[level]["valid"] += 1
    
    print("\n" + "=" * 60)
    print("SOLBENCH V2 KERNEL STATUS SUMMARY")
    print("=" * 60)
    print(f"Total kernels: {total}")
    print(f"Valid kernels: {valid} ({valid/total*100:.1f}%)")
    print(f"Invalid kernels: {total - valid} ({(total-valid)/total*100:.1f}%)")
    print()
    print("Per-level breakdown:")
    for level in ["L1", "L2", "Quant"]:
        if level in level_stats:
            stats = level_stats[level]
            pct = stats["valid"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {level}: {stats['valid']}/{stats['total']} valid ({pct:.1f}%)")
    print("=" * 60)
    
    # Show missing requirements
    missing_get_inputs = sum(1 for r in results if not r["has_get_inputs"])
    missing_reference = sum(1 for r in results if not r["has_ReferenceModel"] and not r["has_reference_backward"])
    missing_launch = sum(1 for r in results if not r["has_launch_reference_implementation"])
    
    print("\nMissing requirements:")
    print(f"  Missing get_inputs(): {missing_get_inputs}")
    print(f"  Missing ReferenceModel/reference_backward: {missing_reference}")
    print(f"  Missing launch_reference_implementation(): {missing_launch}")


def main():
    parser = argparse.ArgumentParser(
        description="Check SolBench v2 kernels for required methods"
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Path to sol-bench/data/benchmark directory"
    )
    parser.add_argument(
        "--output",
        default="sol-bench-v2.csv",
        help="Output CSV file path (default: sol-bench-v2.csv)"
    )
    parser.add_argument(
        "--level",
        choices=["L1", "L2", "Quant", "all"],
        default="all",
        help="Which level to check (default: all)"
    )
    
    args = parser.parse_args()
    
    # Find benchmark directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # solar/scripts -> solar -> repo_root
    
    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir).resolve()
    else:
        # Try multiple possible locations
        possible_paths = [
            repo_root / "sol-bench" / "data" / "benchmark",
            Path.cwd() / "sol-bench" / "data" / "benchmark",
            Path.cwd().parent / "sol-bench" / "data" / "benchmark",
        ]
        benchmark_dir = None
        for path in possible_paths:
            if path.exists():
                benchmark_dir = path
                break
        
        if benchmark_dir is None:
            print(f"Error: Benchmark directory not found")
            print(f"Searched in:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease specify --benchmark-dir or ensure sol-bench/data/benchmark exists")
            return 1
    
    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        print("Please specify --benchmark-dir or ensure sol-bench/data/benchmark exists")
        return 1
    
    print(f"Scanning benchmark directory: {benchmark_dir}")
    
    # Scan and check files
    results = scan_benchmark_directory(str(benchmark_dir))
    
    # Filter by level if specified
    if args.level != "all":
        results = [r for r in results if r["level"] == args.level]
    
    # Write CSV
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = str(repo_root / output_path)
    
    write_csv(results, output_path)
    
    # Print summary
    print_summary(results)
    
    return 0


if __name__ == "__main__":
    exit(main())
