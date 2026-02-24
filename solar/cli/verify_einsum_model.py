#!/usr/bin/env python3
"""CLI for verifying a single model's einsum expression against PyTorch reference.

This is a kernelbench-independent verification tool that takes any output folder
containing einsum/einsum_graph.yaml and verifies it.

Usage:
    python -m solar.cli.verify_einsum_model solar/output_kernelbench/level1/19_ReLU
    python -m solar.cli.verify_einsum_model solar/examples/BERT
    python -m solar.cli.verify_einsum_model /path/to/model/output --scale 0.1
    python -m solar.cli.verify_einsum_model /path/to/model/output --quiet  # suppress verbose output
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass, field
from datetime import datetime

import yaml


def setup_verifier_path():
    """Add solar_verifier to Python path."""
    solar_root = Path(__file__).resolve().parents[2]
    verifier_path = solar_root / "solar_verifier"
    if str(verifier_path) not in sys.path:
        sys.path.insert(0, str(verifier_path))


def verify_model_output(output_dir: str, 
                        verbose: bool = False, 
                        scale_factor: float = 0.01,
                        return_details: bool = False) -> Tuple[bool, str]:
    """Verify a single model output directory.
    
    This function is kernelbench-independent and can verify any output folder
    containing einsum/einsum_graph.yaml.
    
    Args:
        output_dir: Path to model output directory (e.g., solar/output_kernelbench/level1/19_ReLU)
        verbose: Print detailed output
        scale_factor: Scale factor for tensor dimensions
        return_details: If True, return (success, message, details_dict) with expression, shapes, etc.
        
    Returns:
        If return_details=False: Tuple of (success: bool, message: str)
        If return_details=True: Tuple of (success: bool, message: str, details: dict)
    """
    setup_verifier_path()
    
    from verify import run_benchmark_test
    
    output_path = Path(output_dir)
    
    # Validate the directory exists
    if not output_path.exists():
        if return_details:
            return False, f"Output directory not found: {output_path}", {}
        return False, f"Output directory not found: {output_path}"
    
    # Validate einsum_graph.yaml exists
    einsum_yaml = output_path / "einsum" / "einsum_graph.yaml"
    if not einsum_yaml.exists():
        if return_details:
            return False, f"No einsum_graph.yaml found at {einsum_yaml}", {}
        return False, f"No einsum_graph.yaml found at {einsum_yaml}"
    
    # Run the verification
    return run_benchmark_test(
        str(output_path),
        verbose=verbose,
        num_runs=1,
        scale_factor=scale_factor,
        return_details=return_details
    )


@dataclass
class VerificationResult:
    """Result of verifying a single einsum expression."""
    passed: bool
    benchmark_name: str
    error_message: str = None
    expression: str = None
    shapes: dict = None
    verification_stats: dict = None
    emulated_code_path: str = None
    
    def to_dict(self):
        """Convert to dictionary for YAML output."""
        result = {
            'status': 'passed' if self.passed else 'failed',
            'benchmark_name': self.benchmark_name,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add expression if available
        if self.expression:
            result['expression'] = self.expression
        
        # Add shapes if available
        if self.shapes:
            result['shapes'] = self.shapes
        
        # Add verification stats if available
        if self.verification_stats:
            result['verification_stats'] = self.verification_stats
        
        # Add emulated code path if available
        if self.emulated_code_path:
            result['emulated_code_path'] = self.emulated_code_path
        
        # Add error info if failed
        if not self.passed and self.error_message:
            result['error'] = {'message': self.error_message}
        
        return result


def save_verification_result(result: VerificationResult, output_dir: Path) -> Path:
    """Save verification result to einsum_verification/einsum_verification.yaml.
    
    Args:
        result: VerificationResult to save.
        output_dir: Base output directory for the benchmark.
        
    Returns:
        Path to the saved einsum_verification.yaml file.
    """
    verification_dir = output_dir / "einsum_verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = verification_dir / "einsum_verification.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(result.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    return output_file


def main() -> None:
    """Main entry point for single model verification."""
    parser = argparse.ArgumentParser(
        description="Verify a single model's einsum expression against PyTorch reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a specific model output (verbose by default)
  python -m solar.cli.verify_einsum_model solar/output_kernelbench/level1/19_ReLU
  
  # Verify with quiet mode (suppress detailed output)
  python -m solar.cli.verify_einsum_model solar/examples/BERT --quiet
  
  # Verify with custom scale factor
  python -m solar.cli.verify_einsum_model /path/to/output --scale 0.1
        """
    )
    
    parser.add_argument(
        "output_dir",
        help="Path to model output directory containing einsum/einsum_graph.yaml"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale factor for tensor dimensions (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # --quiet overrides --verbose
    verbose = args.verbose and not args.quiet
    
    output_path = Path(args.output_dir)
    benchmark_name = output_path.name
    
    print("=" * 70)
    print("EINSUM VERIFICATION")
    print("=" * 70)
    print(f"Output directory: {output_path}")
    print(f"Scale factor: {args.scale}")
    print()
    
    if verbose:
        print(f"Verifying: {benchmark_name}")
        print('='*70)
    else:
        print(f"Verifying {benchmark_name}...", end=' ', flush=True)
    
    # Run verification with details
    success, message, details = verify_model_output(
        str(output_path),
        verbose=verbose,
        scale_factor=args.scale,
        return_details=True
    )
    
    # Create and save result with all details
    result = VerificationResult(
        passed=success,
        benchmark_name=benchmark_name,
        error_message=None if success else message,
        expression=details.get('expression'),
        shapes=details.get('shapes'),
        verification_stats=details.get('verification_stats'),
        emulated_code_path=details.get('emulated_code_path')
    )
    
    output_file = save_verification_result(result, output_path)
    
    if success:
        if not verbose:
            print("✅ PASS")
        print(f"\n✅ VERIFICATION PASSED")
    else:
        if not verbose:
            print(f"❌ FAIL: {message}")
        print(f"\n❌ VERIFICATION FAILED: {message}")
    
    print(f"Result saved to: {output_file}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
