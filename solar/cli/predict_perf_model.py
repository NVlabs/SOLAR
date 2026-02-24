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

"""CLI for predicting performance from `analysis.yaml` and an arch YAML.

This command is intentionally **single-step**:
- Input: `analysis.yaml`
- Output: `perf_<arch>.yaml`
"""

import argparse
import sys
from pathlib import Path

from solar.perf import EinsumGraphPerfModel
from solar.common.utils import ensure_directory


def main() -> None:
    """Main entry point for `analysis.yaml` -> `perf_<arch>.yaml`."""
    parser = argparse.ArgumentParser(
        description="Predict performance from analysis.yaml using an architecture YAML.",
    )
    parser.add_argument(
        "--analysis-path",
        required=True,
        help="Path to analysis.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for perf YAML.",
    )
    parser.add_argument(
        "--arch-config",
        default="H100_PCIe",
        help="Architecture name (e.g., H100_PCIe) or path to a YAML file.",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        help="Precision for selecting MAC throughput keys (default: fp16).",
    )
    parser.add_argument(
        "--no-copy-analysis",
        action="store_true",
        help="Do not copy analysis.yaml into the output directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )

    args = parser.parse_args()
    analysis_path = Path(args.analysis_path)
    if not analysis_path.exists():
        print(f"❌ analysis.yaml not found: {analysis_path}")
        sys.exit(2)

    output_dir = ensure_directory(args.output_dir)
    model = EinsumGraphPerfModel(debug=args.debug)
    perf = model.predict(
        analysis_path,
        output_dir,
        arch_config=args.arch_config,
        precision=args.precision,
        copy_analysis=not args.no_copy_analysis,
    )
    if perf is None:
        print("❌ Perf prediction failed.")
        sys.exit(1)

    print("✅ Perf prediction complete.")
    print(f"  Arch: {perf.get('arch', {}).get('name', 'unknown')}")
    print(f"  Unfused runtime (ms): {perf.get('unfused', {}).get('runtime_ms', 0.0):.4f}")
    print(f"  Fused runtime (ms): {perf.get('fused', {}).get('runtime_ms', 0.0):.4f}")
    print(f"\n📝 Files saved to {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()


