#!/usr/bin/env python3
"""CLI for verifying generated einsum expressions against PyTorch reference.

This command verifies that einsum_graph.yaml expressions produce correct results
when compared to the original PyTorch implementation.

For kernelbench-specific verification with level and kernel-ids filtering.
For single model verification, use verify_einsum_model.py instead.

Usage:
    solar-verify-einsum --level level1
    solar-verify-einsum --level level1 --kernel-ids 19 20 21
    solar-verify-einsum --kernel-ids 19 --verbose
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Optional

from solar.cli.verify_einsum_model import (
    verify_model_output,
    VerificationResult,
    save_verification_result,
)


class EinsumVerifier:
    """Verifies einsum expressions for kernelbench benchmarks."""
    
    def __init__(self, 
                 debug: bool = False,
                 scale_factor: float = 0.01):
        """Initialize the verifier.
        
        Args:
            debug: Enable debug output.
            scale_factor: Scale factor for tensor dimensions (for faster testing).
        """
        self.debug = debug
        self.scale_factor = scale_factor
    
    def verify_benchmark(self, benchmark_dir: Path) -> VerificationResult:
        """Verify a single benchmark directory.
        
        Args:
            benchmark_dir: Path to benchmark directory (e.g., output_kernelbench/level1/19_ReLU).
            
        Returns:
            VerificationResult with pass/fail status and details.
        """
        benchmark_name = benchmark_dir.name
        
        # Use verify_model_output from verify_einsum_model with details
        success, message, details = verify_model_output(
            str(benchmark_dir),
            verbose=self.debug,
            scale_factor=self.scale_factor,
            return_details=True
        )
        
        return VerificationResult(
            passed=success,
            benchmark_name=benchmark_name,
            error_message=None if success else message,
            expression=details.get('expression'),
            shapes=details.get('shapes'),
            verification_stats=details.get('verification_stats'),
            emulated_code_path=details.get('emulated_code_path')
        )
    
    def get_benchmark_directories(self,
                                  base_dir: Path,
                                  level: Optional[str] = None,
                                  kernel_ids: Optional[List[int]] = None) -> List[Path]:
        """Get list of benchmark directories to verify.
        
        Args:
            base_dir: Base output directory (e.g., output_kernelbench).
            level: Kernel level (e.g., 'level1').
            kernel_ids: Specific kernel IDs to verify.
            
        Returns:
            List of benchmark directory paths.
        """
        benchmark_dirs = []
        
        if level:
            level_dir = base_dir / level
            if not level_dir.exists():
                return []
            
            for kernel_dir in level_dir.iterdir():
                if not kernel_dir.is_dir():
                    continue
                
                # Check if einsum_graph.yaml exists
                if not (kernel_dir / "einsum" / "einsum_graph.yaml").exists():
                    continue
                
                # Filter by kernel_ids if specified
                if kernel_ids:
                    match = re.match(r'(\d+)', kernel_dir.name)
                    if match:
                        kernel_id = int(match.group(1))
                        if kernel_id not in kernel_ids:
                            continue
                
                benchmark_dirs.append(kernel_dir)
        else:
            # Search all levels
            for level_dir in base_dir.iterdir():
                if not level_dir.is_dir() or not level_dir.name.startswith('level'):
                    continue
                
                benchmark_dirs.extend(
                    self.get_benchmark_directories(base_dir, level_dir.name, kernel_ids)
                )
        
        # Sort by kernel ID
        def extract_number(path):
            match = re.match(r'(\d+)', path.name)
            return int(match.group(1)) if match else float('inf')
        
        return sorted(benchmark_dirs, key=extract_number)


def main() -> None:
    """Main entry point for einsum verification."""
    parser = argparse.ArgumentParser(
        description="Verify generated einsum expressions against PyTorch reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all kernels in level1
  python -m solar.cli.verify_einsum --level level1
  
  # Verify specific kernels
  python -m solar.cli.verify_einsum --level level1 --kernel-ids 19 20 21
  
  # Verify with verbose output
  python -m solar.cli.verify_einsum --kernel-ids 19 --verbose
  
  # Custom output directory
  python -m solar.cli.verify_einsum --level level1 --output-dir ./output_kernelbench
  
  # For single model verification, use verify_einsum_model instead:
  python -m solar.cli.verify_einsum_model solar/output_kernelbench/level1/19_ReLU
        """
    )
    
    parser.add_argument(
        "--level",
        help="Kernel level to verify (e.g., level1, level2)"
    )
    parser.add_argument(
        "--kernel-ids",
        nargs="+",
        type=int,
        help="Specific kernel IDs to verify"
    )
    
    # Fix: Use solar root as base, not repo root
    solar_root = Path(__file__).resolve().parents[2]
    default_output_dir = solar_root / "output_kernelbench"
    
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help=f"Output directory containing einsum graphs (default: {default_output_dir})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale factor for tensor dimensions (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = EinsumVerifier(
        debug=args.verbose,
        scale_factor=args.scale,
    )
    
    # Get benchmark directories
    base_dir = Path(args.output_dir)
    if not base_dir.exists():
        print(f"❌ Output directory not found: {base_dir}")
        print(f"   Try specifying --output-dir or check if output_kernelbench exists")
        sys.exit(1)
    
    benchmark_dirs = verifier.get_benchmark_directories(
        base_dir,
        level=args.level,
        kernel_ids=args.kernel_ids
    )
    
    if not benchmark_dirs:
        print(f"❌ No benchmarks found matching criteria")
        print(f"   Base directory: {base_dir}")
        print(f"   Level: {args.level}")
        print(f"   Kernel IDs: {args.kernel_ids}")
        sys.exit(1)
    
    print("=" * 70)
    print("EINSUM VERIFICATION (KernelBench)")
    print("=" * 70)
    print(f"Output directory: {base_dir}")
    print(f"Found {len(benchmark_dirs)} benchmark(s) to verify")
    print(f"Scale factor: {args.scale}")
    print()
    
    # Run verification
    passed = 0
    failed = 0
    results = []
    
    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name
        
        if args.verbose:
            print(f"\n{'='*70}")
            print(f"Verifying: {benchmark_name}")
            print('='*70)
        else:
            print(f"Verifying {benchmark_name}...", end=' ', flush=True)
        
        result = verifier.verify_benchmark(benchmark_dir)
        results.append((benchmark_dir, result))
        
        # Save result
        output_file = save_verification_result(result, benchmark_dir)
        
        if result.passed:
            passed += 1
            if not args.verbose:
                print("✅ PASS")
        else:
            failed += 1
            if not args.verbose:
                print(f"❌ FAIL: {result.error_message}")
        
        if args.verbose:
            print(f"Result saved to: {output_file}")
    
    # Summary
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed benchmarks:")
        for benchmark_dir, result in results:
            if not result.passed:
                print(f"  ❌ {result.benchmark_name}: {result.error_message}")
    
    if passed == len(results):
        print("✅ ALL VERIFICATIONS PASSED")
        sys.exit(0)
    else:
        print(f"❌ {failed} VERIFICATION(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
