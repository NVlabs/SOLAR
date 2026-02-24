#!/usr/bin/env python3
"""Collect performance results from kernelbench runs.

This script scans the output_kernelbench directory and collects performance
metrics from perf_<arch>.yaml files into a CSV summary.

Usage:
    python collect_perf_results.py [OPTIONS]

Options:
    --arch ARCH         Architecture name (default: H100_PCIe)
    --output FILE       Output CSV file (default: perf_summary.csv)
    --level LEVEL       Only collect from specific level (e.g., level1)
    --output-dir DIR    Output directory for kernelbench (default: ../output_kernelbench)

Example:
    python collect_perf_results.py --arch H100_PCIe --output results.csv
    python collect_perf_results.py --level level1 --output level1_results.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def parse_kernel_name(dirname: str) -> tuple:
    """Parse kernel directory name to extract ID and name.
    
    Args:
        dirname: Directory name like "1_Square_matrix_multiplication_"
        
    Returns:
        Tuple of (kernel_id, kernel_name)
    """
    match = re.match(r'^(\d+)_(.+)$', dirname)
    if match:
        kernel_id = int(match.group(1))
        kernel_name = match.group(2)
        return kernel_id, kernel_name
    return 0, dirname


def load_perf_yaml(perf_path: Path) -> Optional[Dict[str, Any]]:
    """Load performance YAML file.
    
    Args:
        perf_path: Path to perf_<arch>.yaml file
        
    Returns:
        Parsed YAML dict or None if failed
    """
    try:
        with open(perf_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load {perf_path}: {e}", file=sys.stderr)
        return None


def collect_results(
    output_dir: Path,
    arch: str,
    level_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Collect performance results from output directory.
    
    Args:
        output_dir: Path to output_kernelbench directory
        arch: Architecture name (e.g., H100_PCIe)
        level_filter: Optional level to filter (e.g., "level1")
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Find all level directories
    level_dirs = sorted(output_dir.glob("level*"))
    
    for level_dir in level_dirs:
        if not level_dir.is_dir():
            continue
            
        level = level_dir.name
        
        # Apply level filter if specified
        if level_filter and level != level_filter:
            continue
        
        # Find all kernel directories in this level
        kernel_dirs = sorted(level_dir.iterdir())
        
        for kernel_dir in kernel_dirs:
            if not kernel_dir.is_dir():
                continue
            
            kernel_id, kernel_name = parse_kernel_name(kernel_dir.name)
            
            # Look for perf file
            perf_file = kernel_dir / "perf" / f"perf_{arch}.yaml"
            
            if not perf_file.exists():
                # Try to find any perf file
                perf_dir = kernel_dir / "perf"
                if perf_dir.exists():
                    perf_files = list(perf_dir.glob("perf_*.yaml"))
                    if perf_files:
                        perf_file = perf_files[0]
                        actual_arch = perf_file.stem.replace("perf_", "")
                        print(f"Warning: Using {actual_arch} instead of {arch} for {kernel_dir.name}", 
                              file=sys.stderr)
                    else:
                        continue
                else:
                    continue
            
            perf_data = load_perf_yaml(perf_file)
            if not perf_data:
                continue
            
            # Extract metrics
            unfused = perf_data.get("unfused", {})
            fused = perf_data.get("fused", {})
            fused_prefetched = perf_data.get("fused_prefetched", {})
            workload = perf_data.get("workload", {})
            speedup = perf_data.get("speedup", {})
            
            result = {
                "level": level,
                "kernel_id": kernel_id,
                "kernel_name": kernel_name,
                "total_macs": workload.get("total_macs", 0),
                "total_flops": workload.get("total_flops", 0),
                "unfused_memory_bytes": unfused.get("memory_bytes", 0),
                "unfused_runtime_ms": unfused.get("runtime_ms", 0),
                "unfused_bottleneck": unfused.get("bottleneck", ""),
                "unfused_ai": unfused.get("arithmetic_intensity", 0),
                "fused_memory_bytes": fused.get("memory_bytes", 0),
                "fused_runtime_ms": fused.get("runtime_ms", 0),
                "fused_bottleneck": fused.get("bottleneck", ""),
                "fused_ai": fused.get("arithmetic_intensity", 0),
                "fused_prefetched_memory_bytes": fused_prefetched.get("memory_bytes", 0),
                "fused_prefetched_runtime_ms": fused_prefetched.get("runtime_ms", 0),
                "fused_prefetched_bottleneck": fused_prefetched.get("bottleneck", ""),
                "fused_prefetched_ai": fused_prefetched.get("arithmetic_intensity", 0),
                "speedup_fused_vs_unfused": speedup.get("fused_vs_unfused", 1.0),
                "speedup_fused_prefetched_vs_unfused": speedup.get("fused_prefetched_vs_unfused", 1.0),
            }
            
            results.append(result)
    
    # Sort by level and kernel_id
    results.sort(key=lambda x: (x["level"], x["kernel_id"]))
    
    return results


def write_csv(results: List[Dict[str, Any]], output_path: Path, simple: bool = False) -> None:
    """Write results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
        simple: If True, output simplified CSV with fewer columns
    """
    if not results:
        print("No results to write", file=sys.stderr)
        return
    
    if simple:
        # Simplified output as requested
        fieldnames = [
            "level",
            "kernel_id", 
            "kernel_name",
            "sol_time_ms",
            "fused_sol_time_ms",
        ]
        
        # Map to simplified names
        simple_results = []
        for r in results:
            simple_results.append({
                "level": r["level"],
                "kernel_id": r["kernel_id"],
                "kernel_name": r["kernel_name"],
                "sol_time_ms": r["unfused_runtime_ms"],
                "fused_sol_time_ms": r["fused_runtime_ms"],
            })
        results = simple_results
    else:
        fieldnames = list(results[0].keys())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Wrote {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect performance results from kernelbench runs"
    )
    parser.add_argument(
        "--arch",
        default="H100_PCIe",
        help="Architecture name (default: H100_PCIe)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file (default: results_<arch>.csv)",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Only collect from specific level (e.g., level1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for kernelbench (default: ../output_kernelbench)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Output simplified CSV with fewer columns",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Output full CSV with all metrics",
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir.parent / "output_kernelbench"
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Collecting results from: {output_dir}")
    print(f"Architecture: {args.arch}")
    if args.level:
        print(f"Level filter: {args.level}")
    
    results = collect_results(output_dir, args.arch, args.level)
    
    if not results:
        print("No results found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(results)} kernel results")
    
    # Default to simple output unless --full is specified
    simple = not args.full
    
    # Default output filename based on architecture
    output_file = args.output if args.output else f"results_{args.arch}.csv"
    
    write_csv(results, Path(output_file), simple=simple)


if __name__ == "__main__":
    main()
