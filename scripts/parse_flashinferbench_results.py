#!/usr/bin/env python3
"""
Parse FlashInfer benchmark results and create a comparison CSV.

Compares Solar's predicted performance (unfused, fused, fused_prefetched)
with actual measured performance from FlashInfer trace solutions.

Usage:
    python parse_flashinferbench_results.py --arch B200
    python parse_flashinferbench_results.py --arch H100_PCIe --output results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a YAML file."""
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def find_solutions_by_uuid(
    traces_dir: Path, 
    op_type: str, 
    definition: str, 
    uuid: str
) -> List[Dict[str, Any]]:
    """Find all solutions for a given UUID in trace files."""
    trace_file = traces_dir / op_type / f"{definition}.jsonl"
    if not trace_file.exists():
        return []
    
    solutions = []
    entries = load_jsonl(trace_file)
    for entry in entries:
        workload = entry.get("workload", {})
        if workload.get("uuid") == uuid:
            solution_name = entry.get("solution", "unknown")
            evaluation = entry.get("evaluation", {})
            performance = evaluation.get("performance")
            status = evaluation.get("status", "unknown")
            
            if performance is not None:
                latency_ms = performance.get("latency_ms")
                if latency_ms is not None and status == "PASSED":
                    solutions.append({
                        "name": solution_name,
                        "latency_ms": latency_ms,
                        "speedup": performance.get("speedup_factor"),
                    })
    
    return solutions


def parse_results(
    output_dir: Path,
    traces_dir: Path,
    arch: str
) -> List[Dict[str, Any]]:
    """Parse all results from output directory."""
    results = []
    
    # Iterate through op_type/definition/row_id structure
    for op_dir in sorted(output_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        op_type = op_dir.name
        
        for def_dir in sorted(op_dir.iterdir()):
            if not def_dir.is_dir():
                continue
            definition = def_dir.name
            
            for row_dir in sorted(def_dir.iterdir()):
                if not row_dir.is_dir():
                    continue
                try:
                    row_id = int(row_dir.name)
                except ValueError:
                    continue
                
                # Load uuid.yaml
                uuid_data = load_yaml(row_dir / "uuid.yaml")
                if not uuid_data:
                    continue
                
                uuid = uuid_data.get("uuid")
                if not uuid:
                    continue
                
                # Load perf yaml
                perf_file = row_dir / "perf" / f"perf_{arch}.yaml"
                perf_data = load_yaml(perf_file)
                
                # Get Solar predictions
                solar_unfused = None
                solar_fused = None
                solar_prefetched = None
                
                if perf_data:
                    unfused = perf_data.get("unfused", {})
                    fused = perf_data.get("fused", {})
                    prefetched = perf_data.get("fused_prefetched", {})
                    
                    solar_unfused = unfused.get("runtime_ms")
                    solar_fused = fused.get("runtime_ms")
                    solar_prefetched = prefetched.get("runtime_ms")
                
                # Find actual solutions
                solutions = find_solutions_by_uuid(
                    traces_dir, op_type, definition, uuid
                )
                
                # Calculate min latency from solutions
                min_latency_ms = None
                if solutions:
                    min_latency_ms = min(s["latency_ms"] for s in solutions)
                
                # Calculate ratio: min_solution / solar_fused
                runtime_fused_sol_ratio = None
                if min_latency_ms is not None and solar_fused is not None and solar_fused > 0:
                    runtime_fused_sol_ratio = min_latency_ms / solar_fused
                
                # Build result entry
                result = {
                    "op_type": op_type,
                    "definition": definition,
                    "row_id": row_id,
                    "uuid": uuid,
                    "axes": uuid_data.get("resolved_axes", {}),
                    "solar_unfused_ms": solar_unfused,
                    "solar_fused_ms": solar_fused,
                    "solar_prefetched_ms": solar_prefetched,
                    "min_latency_ms": min_latency_ms,
                    "runtime_fused_sol_ratio": runtime_fused_sol_ratio,
                    "solutions": solutions,
                }
                results.append(result)
    
    return results


def write_csv(results: List[Dict[str, Any]], output_path: Path):
    """Write results to CSV file."""
    if not results:
        print("No results to write")
        return
    
    # Find max number of solutions across all results
    max_solutions = max(len(r["solutions"]) for r in results) if results else 0
    
    # Build header
    header = [
        "op_type",
        "definition", 
        "row_id",
        "uuid",
        "axes",
        "solar_unfused_ms",
        "solar_fused_ms",
        "solar_prefetched_ms",
        "min_latency_ms",
        "runtime_fused_sol_ratio",
    ]
    
    # Add solution columns
    for i in range(max_solutions):
        header.extend([
            f"solution_{i+1}_name",
            f"solution_{i+1}_latency_ms",
        ])
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for result in results:
            row = [
                result["op_type"],
                result["definition"],
                result["row_id"],
                result["uuid"],
                str(result["axes"]),
                result["solar_unfused_ms"],
                result["solar_fused_ms"],
                result["solar_prefetched_ms"],
                result["min_latency_ms"],
                result["runtime_fused_sol_ratio"],
            ]
            
            # Add solutions
            for i in range(max_solutions):
                if i < len(result["solutions"]):
                    sol = result["solutions"][i]
                    row.extend([sol["name"], sol["latency_ms"]])
                else:
                    row.extend(["", ""])
            
            writer.writerow(row)
    
    print(f"Wrote {len(results)} results to {output_path}")


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of results."""
    print("\n" + "=" * 80)
    print("FLASHINFER BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Group by op_type
    by_op = {}
    for r in results:
        op = r["op_type"]
        if op not in by_op:
            by_op[op] = []
        by_op[op].append(r)
    
    for op_type, op_results in sorted(by_op.items()):
        print(f"\n{op_type}: {len(op_results)} workloads")
        
        # Show first few with comparison
        for r in op_results[:3]:
            solar_fused = r["solar_fused_ms"]
            solutions = r["solutions"]
            
            if solar_fused and solutions:
                best_sol = min(solutions, key=lambda s: s["latency_ms"])
                ratio = r.get("runtime_fused_sol_ratio")
                ratio_str = f"{ratio:.2f}x" if ratio else "N/A"
                print(f"  {r['definition']} row={r['row_id']}: "
                      f"solar={solar_fused:.6f}ms, "
                      f"best={best_sol['latency_ms']:.6f}ms ({best_sol['name']}), "
                      f"ratio={ratio_str}")
            elif solar_fused:
                print(f"  {r['definition']} row={r['row_id']}: "
                      f"solar={solar_fused:.6f}ms (no solutions found)")


def main():
    parser = argparse.ArgumentParser(
        description="Parse FlashInfer benchmark results and compare with Solar predictions."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="B200",
        help="Architecture config name (default: B200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: flashinferbench_comparison_<arch>.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Solar output directory (default: solar/output_flashinferbench)",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="FlashInfer traces directory (default: flashinfer-trace/traces)",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    solar_root = script_dir.parent
    repo_root = solar_root.parent
    
    output_dir = args.output_dir or solar_root / "output_flashinferbench"
    traces_dir = args.traces_dir or repo_root / "flashinfer-trace" / "traces"
    output_csv = args.output or solar_root / f"flashinferbench_comparison_{args.arch}.csv"
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    if not traces_dir.exists():
        print(f"Warning: Traces directory not found: {traces_dir}")
        print("Solution comparisons will be empty.")
    
    print(f"Parsing results from: {output_dir}")
    print(f"Traces from: {traces_dir}")
    print(f"Architecture: {args.arch}")
    
    # Parse results
    results = parse_results(output_dir, traces_dir, args.arch)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Write CSV
    write_csv(results, output_csv)
    
    print(f"\nOutput: {output_csv}")


if __name__ == "__main__":
    main()
