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

"""CLI for analyzing a single einsum graph into hardware-independent stats.

This command is intentionally **single-step**:
- Input: `einsum_graph.yaml`
- Output: `analysis.yaml`
"""

import argparse
import sys
from pathlib import Path

from solar.analysis import EinsumGraphAnalyzer
from solar.common.utils import ensure_directory


def main() -> None:
    """Main entry point for `einsum_graph.yaml` -> `analysis.yaml`."""
    parser = argparse.ArgumentParser(
        description="Analyze an einsum graph (einsum_graph.yaml) into analysis.yaml.",
    )
    parser.add_argument(
        "--einsum-graph-path",
        required=True,
        help="Path to einsum_graph.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for analysis.yaml.",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        help="Precision for byte calculations (default: fp16).",
    )
    parser.add_argument(
        "--no-copy-graph",
        action="store_true",
        help="Do not copy einsum_graph.yaml into the output directory.",
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
    analyzer = EinsumGraphAnalyzer(debug=args.debug)
    analysis = analyzer.analyze_graph(
        graph_path,
        output_dir,
        precision=args.precision,
        copy_graph=not args.no_copy_graph,
    )
    if analysis is None:
        print("❌ Analysis failed.")
        sys.exit(1)

    total = analysis.get("total", {}) or {}
    print("✅ Analysis complete.")
    print(f"  Layers: {total.get('num_layers', 0)}")
    print(f"  MACs: {total.get('macs', 0)}")
    print(f"  FLOPs: {total.get('flops', 0)}")
    print(f"  Memory Elements: {total.get('fused_elements', 0)}")
    print(f"\n📝 Files saved to {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()

