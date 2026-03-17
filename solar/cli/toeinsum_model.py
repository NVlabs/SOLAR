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

"""CLI for converting a single PyTorch graph to an einsum graph.

This command is intentionally **single-step**:
- Input: `pytorch_graph.yaml` (preferred; JSON node lists also accepted)
- Output: `einsum_graph.yaml` and `einsum_graph_renamed.yaml`
- Optionally: `einsum_graph.pdf` (visualization)
"""

import argparse
import sys
from pathlib import Path

from solar.common.utils import ensure_directory
from solar.einsum import PyTorchToEinsum
from solar.einsum.einsum_graph_visualizer import EinsumGraphVisualizer


def _print_conversion_summary(einsum_graph: dict) -> None:
    """Print a compact conversion summary."""
    layers = einsum_graph.get("layers", {}) or {}
    with_eq = sum(1 for v in layers.values() if v.get("einsum_equation"))
    print("\n" + "=" * 60)
    print("TOEINSUM SUMMARY")
    print("=" * 60)
    print(f"Layers: {len(layers)}")
    print(f"Layers with einsum_equation: {with_eq}/{len(layers) if layers else 0}")
    print("=" * 60)


def main() -> None:
    """Main entry point for `pytorch_graph.yaml` -> `einsum_graph.yaml`."""
    parser = argparse.ArgumentParser(
        description="Convert a single PyTorch graph (pytorch_graph.yaml) to an einsum graph (einsum_graph.yaml).",
    )
    parser.add_argument(
        "--graph-path",
        required=True,
        help="Path to a PyTorch graph file (prefer pytorch_graph.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for einsum_graph.yaml.",
    )
    parser.add_argument(
        "--enable-llm-agent",
        action="store_true",
        help="Enable LLM agent for unknown node types.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key for LLM agent (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--cache-dir",
        default="./solar_handlers_cache",
        help="Directory for caching generated handlers (default: ./solar_handlers_cache).",
    )
    parser.add_argument(
        "--no-copy-graph",
        action="store_true",
        help="Do not copy the input graph into the output directory.",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable best-effort complex-op expansion before writing einsum_graph.yaml.",
    )
    parser.add_argument(
        "--enable-rename",
        action="store_true",
        help="Enable BFS rank renaming (disabled by default to save time).",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save a PDF visualization of the einsum graph.",
    )
    parser.add_argument(
        "--graph-pdf-name",
        default="einsum_graph.pdf",
        help="Filename for the graph PDF (default: einsum_graph.pdf).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )

    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        sys.exit(2)

    output_dir = ensure_directory(args.output_dir)

    converter = PyTorchToEinsum(
        debug=args.debug,
        enable_agent=args.enable_llm_agent,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
    )
    results = converter.convert(
        graph_path,
        output_dir,
        copy_graph=not args.no_copy_graph,
        expand_complex_ops=not args.no_expand,
        enable_rename=args.enable_rename,
    )
    if results is None:
        print("❌ Conversion failed.")
        sys.exit(1)

    print("✅ Conversion complete.")
    _print_conversion_summary(results)

    # Save graph visualization if requested
    if args.save_graph:
        visualizer = EinsumGraphVisualizer(debug=args.debug)
        renamed_graph_path = output_dir / "einsum_graph_renamed.yaml"
        pdf_path = output_dir / args.graph_pdf_name
        try:
            visualizer.save_graph_pdf(renamed_graph_path, pdf_path)
            print(f"📊 Graph visualization saved: {pdf_path}")
        except Exception as e:
            print(f"⚠️  Failed to save graph visualization: {e}")

    print(f"\n📝 Files saved to {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()


