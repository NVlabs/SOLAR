"""CLI for converting einsum_graph.yaml to Timeloop workload format.

This command is intentionally **single-step**:
- Input: `einsum_graph_renamed.yaml` (or `einsum_graph.yaml`)
- Output: `timeloop_graph.yaml`
"""

import argparse
import sys
from pathlib import Path

from solar.common.utils import ensure_directory
from solar.einsum import EinsumToTimeloop


def main() -> None:
    """Main entry point for `einsum_graph.yaml` -> `timeloop_graph.yaml`."""
    parser = argparse.ArgumentParser(
        description="Convert an einsum graph (einsum_graph.yaml) to Timeloop workload format (timeloop_graph.yaml).",
    )
    parser.add_argument(
        "--einsum-graph-path",
        required=True,
        help="Path to einsum_graph.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for timeloop_graph.yaml.",
    )
    parser.add_argument(
        "--output-name",
        default="timeloop_graph.yaml",
        help="Output filename (default: timeloop_graph.yaml).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )

    args = parser.parse_args()

    graph_path = Path(args.einsum_graph_path)
    if not graph_path.exists():
        print(f"❌ Einsum graph not found: {graph_path}")
        sys.exit(2)

    output_dir = ensure_directory(args.output_dir)
    output_path = output_dir / args.output_name

    converter = EinsumToTimeloop(debug=args.debug)
    
    try:
        result = converter.convert(graph_path, output_path)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

    # Print summary
    workload = result.get('workload', {})
    num_dims = len(workload.get('shape', {}))
    num_einsums = len(workload.get('einsums', []))

    print("✅ Timeloop conversion complete.")
    print(f"   Dimensions: {num_dims}")
    print(f"   Einsums: {num_einsums}")

    print(f"\n📝 Files saved to {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()

