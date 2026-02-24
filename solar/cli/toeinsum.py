"""CLI for converting benchmark models to `einsum_graph.yaml`.

This command targets kernelbench benchmark suites:
- Input graph: `pytorch_graph.yaml` (torchview-derived)
- Output graph: `einsum_graph.yaml`
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from solar.einsum import BenchmarkEinsumConverter
from solar.graph import BenchmarkProcessor
from solar.common.types import ProcessingConfig
from solar.common.utils import ensure_directory


def main() -> None:
    """Main entry point for model to einsum conversion and analysis."""
    parser = argparse.ArgumentParser(
        description="Convert models to einsum representation and analyze performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Kernelbench to einsum conversion and analysis
  solar-toeinsum --level level1
  solar-toeinsum --level level1 --kernel-ids 1 2 3
  solar-toeinsum --kernel-ids 19 --debug
  solar-toeinsum --kernel-status --level level1
  
  # Architecture configuration
  solar-toeinsum --level level1 --kernel-ids 1 --arch-config A6000
  
  # Other options
  solar-toeinsum --list-analyses
  solar-toeinsum --level level1 --collect-stats
  solar-toeinsum --level level1 --kernel-ids 1 --enable-llm-agent
        """
    )
    
    parser.add_argument(
        "--level",
        help="Kernel level to analyze (e.g., level1, level2)"
    )
    parser.add_argument(
        "--kernel-ids",
        nargs="+",
        type=int,
        help="Specific kernel IDs to analyze"
    )
    repo_root = Path(__file__).resolve().parents[3]
    default_kernelbench_dir = repo_root / "kernelbench"
    default_kernelbench_outputs_dir = repo_root / "kernelbench_outputs"
    default_output_dir = repo_root / "solar_outputs" / "kernelbench"

    parser.add_argument(
        "--kernelbench-dir",
        default=str(default_kernelbench_dir),
        help="Directory containing kernelbench source files (default: <repo_root>/kernelbench)",
    )
    parser.add_argument(
        "--kernelbench-outputs-dir",
        default=str(default_kernelbench_outputs_dir),
        help="Directory containing kernelbench outputs (default: <repo_root>/kernelbench_outputs)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Output directory for einsum graphs (default: <repo_root>/solar_outputs/kernelbench)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--list-analyses",
        action="store_true",
        help="List all available analyses and exit"
    )
    parser.add_argument(
        "--collect-stats",
        action="store_true",
        help="Collect comprehensive operation statistics"
    )
    parser.add_argument(
        "--kernel-status",
        action="store_true",
        help="Show status of each kernel"
    )
    parser.add_argument(
        "--arch-config",
        default="H100_PCIe",
        help="Architecture configuration (e.g., H100_PCIe, A6000, H100_fp32)"
    )
    parser.add_argument(
        "--enable-llm-agent",
        action="store_true",
        help="Enable LLM agent for unknown node types"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key for LLM agent (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force regeneration of model graphs even if they exist"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    ensure_directory(output_path)
    
    # Create analyzer
    analyzer = BenchmarkEinsumConverter(
        debug=args.debug,
        enable_agent=args.enable_llm_agent,
        api_key=args.api_key
    )
    
    # List analyses if requested
    if args.list_analyses:
        list_available_analyses(
            analyzer,
            args.kernelbench_outputs_dir,
        )
        return
    
    # Show kernel status if requested
    if args.kernel_status:
        analyzer.print_kernel_status(
            base_dir=args.kernelbench_outputs_dir,
            level=args.level,
            kernel_ids=args.kernel_ids,
        )
        return
    
    # Process model graphs if needed
    process_missing_graphs(
        args.kernelbench_dir,
        args.kernelbench_outputs_dir,
        args.level,
        args.kernel_ids,
        args.force_rerun,
        args.debug
    )
    
    # Get directories to analyze
    kernel_dirs = analyzer.get_output_directories(
        base_dir=args.kernelbench_outputs_dir,
        level=args.level,
        kernel_ids=args.kernel_ids,
    )
    
    if not kernel_dirs:
        print("❌ No directories found matching the criteria!")
        print("\nUse --list-analyses to see available analyses.")
        return
    
    # Collect statistics if requested
    if args.collect_stats:
        print(f"Found {len(kernel_dirs)} kernels to analyze")
        stats = analyzer.collect_operation_statistics(
            kernel_dirs
        )
        print_operation_statistics(stats, output_path)
        return
    
    # Regular analysis
    print(f"Found {len(kernel_dirs)} kernels to convert to einsum")
    print(f"Output directory: {args.output_dir}")
    
    if len(kernel_dirs) == 1:
        # Single kernel analysis
        kernel_dir = kernel_dirs[0]
        
        # Create kernel-specific output directory  
        kernel_name = f"{kernel_dir.parent.name}_{kernel_dir.name}"
        kernel_output_dir = output_path / kernel_name
        ensure_directory(kernel_output_dir)
        
        results = analyzer.analyze_directory(
            kernel_dir,
            kernel_output_dir,
        )
        
        if results:
            print(f"\n✅ Einsum conversion complete for {kernel_name}")
            # Print a tiny summary.
            layers = results.get("layers", {}) if isinstance(results, dict) else {}
            with_eq = sum(1 for v in layers.values() if v.get("einsum_equation"))
            print(f"  Layers: {len(layers)}")
            print(f"  Layers with einsum_equation: {with_eq}/{len(layers) if layers else 0}")
            
            # Show saved files
            print(f"\n📝 Files saved to {kernel_output_dir}:")
            for file_path in kernel_output_dir.iterdir():
                if file_path.is_file():
                    print(f"  - {file_path.name}")
    else:
        # Multiple kernel analysis
        results = analyzer.analyze_kernels(
            level=args.level,
            kernel_ids=args.kernel_ids,
            output_dir=args.output_dir,
            base_dir=args.kernelbench_outputs_dir,
        )
        
        print(f"\n✅ Completed einsum conversion of {len(results)} kernels")
        print(f"📂 Results saved to: {args.output_dir}")


def process_missing_graphs(kernelbench_dir: str,
                          output_dir: str,
                          level: Optional[str],
                          kernel_ids: Optional[List[int]],
                          force_rerun: bool,
                          debug: bool) -> None:
    """Process missing model graphs using BenchmarkProcessor.
    
    Args:
        kernelbench_dir: Directory containing source files.
        output_dir: Directory to save processed graphs.
        level: Kernel level to process.
        kernel_ids: Specific kernel IDs to process.
        force_rerun: Force regeneration even if graphs exist.
        debug: Enable debug output.
    """
    # Check which graphs need to be generated
    target_file = "pytorch_graph.yaml"
    base_path = Path(output_dir)
    source_path = Path(kernelbench_dir)
    
    # Find kernels that need processing
    kernels_to_process = []
    
    if level and kernel_ids:
        # Check specific kernels
        for kernel_id in kernel_ids:
            output_kernel_dir = base_path / level / str(kernel_id)
            graph_file = output_kernel_dir / target_file
            
            if force_rerun or not graph_file.exists():
                # Find matching source file in kernelbench
                source_dir = source_path / level
                if source_dir.exists():
                    for file in source_dir.glob(f"{kernel_id}_*.py"):
                        source_file = file
                        break
                    else:
                        source_file = None
                else:
                    source_file = None
                
                if source_file and source_file.exists():
                    kernels_to_process.append((kernel_id, source_file, output_kernel_dir))
                elif not graph_file.exists():
                    print(f"⚠️ Source file not found for {level} kernel {kernel_id}")
    elif level:
        # Check all kernels in level
        level_dir = source_path / level
        if level_dir.exists():
            for source_file in level_dir.glob("*.py"):
                # Extract kernel ID from filename like "1_name.py"
                parts = source_file.stem.split("_", 1)
                if parts[0].isdigit():
                    kernel_id = int(parts[0])
                else:
                    continue
                
                output_kernel_dir = base_path / level / str(kernel_id)
                graph_file = output_kernel_dir / target_file
                
                if force_rerun or not graph_file.exists():
                    kernels_to_process.append((kernel_id, source_file, output_kernel_dir))
    
    # Process missing graphs
    if kernels_to_process:
        print(f"\n🔄 Generating model graphs for {len(kernels_to_process)} kernels...")
        
        # Create processor
        config = ProcessingConfig(
            output_dir=output_dir,
            save_graph=True,
            force_rerun=force_rerun,
            debug=debug
        )
        processor = BenchmarkProcessor(config)
        
        # Process each kernel
        successful = 0
        failed = 0
        
        for kernel_id, source_file, output_kernel_dir in kernels_to_process:
            kernel_name = f"{source_file.parent.name}_{kernel_id}"
            if debug:
                print(f"  Processing {kernel_name}...")
            
            try:
                success = processor.process_file(str(source_file))
                
                if success:
                    successful += 1
                    if debug:
                        print(f"    ✅ Generated graph for {kernel_name}")
                else:
                    failed += 1
                    print(f"    ❌ Failed to generate graph for {kernel_name}")
            except Exception as e:
                failed += 1
                print(f"    ❌ Error processing {kernel_name}: {e}")
        
        if successful > 0:
            print(f"✅ Generated graphs for {successful} kernels")
        if failed > 0:
            print(f"⚠️ Failed to generate graphs for {failed} kernels")
    elif debug:
        print("ℹ️ All required model graphs already exist")


def list_available_analyses(analyzer: BenchmarkEinsumConverter,
                           base_dir: str) -> None:
    """List all available analyses.
    
    Args:
        analyzer: BenchmarkEinsumConverter instance.
        base_dir: Base directory to scan.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist!")
        return
    
    # Look for level directories
    level_dirs = [d for d in base_path.iterdir() 
                 if d.is_dir() and d.name.startswith("level")]
    
    if not level_dirs:
        print(f"No level directories found in {base_dir}")
        return
    
    # Group by level
    by_level = {}
    target_file = "pytorch_graph.yaml"
    
    for level_dir in level_dirs:
        level_name = level_dir.name
        kernel_dirs = [d for d in level_dir.iterdir() 
                      if d.is_dir() and d.name.isdigit()]
        
        valid_kernels = []
        for kernel_dir in kernel_dirs:
            if (kernel_dir / target_file).exists():
                valid_kernels.append(kernel_dir.name)
        
        if valid_kernels:
            by_level[level_name] = valid_kernels
    
    if not by_level:
        print(f"No directories with {target_file} found in {base_dir}")
        return
    
    print("Available kernelbench graphs for torchview graphs (pytorch_graph.yaml):")
    
    for level, kernels in sorted(by_level.items()):
        kernels.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        print(f"  {level}: {len(kernels)} kernels (IDs: {', '.join(kernels)})")


def print_operation_statistics(stats: dict, output_dir: Path) -> None:
    """Print operation statistics.
    
    Args:
        stats: Statistics dictionary.
        output_dir: Directory to save statistics.
    """
    print("\n" + "=" * 80)
    print("EINSUM OPERATION STATISTICS")
    print("=" * 80)
    
    summary = stats["summary"]
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Kernels: {stats['total_kernels']}")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Unique Operation Types: {summary['unique_operation_types']}")
    
    print(f"\nEINSUM CONVERSION SUPPORT:")
    print(f"  Supported Operations: {summary['supported_einsum_ops_count']} "
          f"({summary['einsum_support_rate']:.1f}%)")
    print(f"  Unsupported Operations: {summary['unsupported_ops_count']}")
    
    print(f"\nTIMELOOP/OROJENESIS SUPPORT:")
    print(f"  Timeloop Runnable: {summary['timeloop_runnable_ops_count']} "
          f"({summary['timeloop_support_rate']:.1f}% of supported)")
    print(f"  Timeloop Failed: {summary['timeloop_failed_ops_count']}")
    
    print(f"\nOPERATION FREQUENCY (Top 10):")
    from collections import Counter
    operation_counts = Counter(stats["operation_counts"])
    for op_type, count in operation_counts.most_common(10):
        print(f"  {op_type:<30} {count:>5} occurrences")
    
    # Save statistics
    ensure_directory(output_dir)
    
    import yaml
    stats_file = output_dir / "operation_statistics.yaml"
    yaml_stats = {
        **stats,
        "supported_einsum_ops": sorted(list(stats["supported_einsum_ops"])),
        "unsupported_ops": sorted(list(stats["unsupported_ops"])),
        "timeloop_runnable_ops": sorted(list(stats["timeloop_runnable_ops"])),
        "timeloop_failed_ops": sorted(list(stats["timeloop_failed_ops"])),
        "operation_counts": dict(stats["operation_counts"])
    }
    
    with open(stats_file, 'w') as f:
        yaml.dump(yaml_stats, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nDetailed statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()