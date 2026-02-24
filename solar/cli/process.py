"""CLI for processing models."""

import argparse
from pathlib import Path
from typing import List, Optional

from solar.graph import BenchmarkProcessor
from solar.common.types import ProcessingConfig


def main() -> None:
    """Main entry point for model processing."""
    parser = argparse.ArgumentParser(
        description="Process kernelbench models"
    )
    
    parser.add_argument(
        "--level",
        default="level1",
        help="Model level to process"
    )
    parser.add_argument(
        "--kernel-ids",
        nargs="+",
        type=int,
        help="Specific kernel IDs to process"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save graph visualizations"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProcessingConfig(
        output_dir=args.output_dir,
        save_graph=args.save_graph,
        force_rerun=args.force,
        debug=args.debug
    )
    
    # Process models
    processor = BenchmarkProcessor(config)
    results = processor.process_directory(
        directory=".",
        level=args.level,
        kernel_ids=args.kernel_ids,
    )
    
    # Print summary
    successful = sum(results.values())
    failed = len(results) - successful
    print(f"\nProcessed {len(results)} models")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
