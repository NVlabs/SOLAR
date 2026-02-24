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

"""CLI for processing a single standalone PyTorch model file.

This command is intentionally **single-model**:
- Input: a Python file defining `Model` and `get_inputs()`
- Output: a directory where `pytorch_graph.yaml` (and optional artifacts) are written
"""

import argparse
import sys
from pathlib import Path

from solar.common.types import ProcessingConfig
from solar.common.utils import ensure_directory
from solar.graph import PyTorchProcessor


def main() -> None:
    """Main entry point for single-model processing."""
    parser = argparse.ArgumentParser(
        description="Process a single PyTorch model file into a torch graph (pytorch_graph.yaml).",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Path to a Python model file containing Model and get_inputs()",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for this model's extracted graph",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save graph visualizations (torchview)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force regeneration even if output already exists",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Run with safer environment settings (thread limits, CPU-only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for processing (default: 600)",
    )

    args = parser.parse_args()

    model_file = Path(args.model_file)
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        sys.exit(2)

    output_dir = ensure_directory(args.output_dir)

    config = ProcessingConfig(
        save_graph=args.save_graph,
        force_rerun=args.force_rerun,
        timeout=args.timeout,
        output_dir=str(output_dir),  # not used for layout; kept for compatibility/logging
        debug=args.debug,
        safe_mode=args.safe_mode,
    )

    processor = PyTorchProcessor(config)
    ok = processor.process_model_file(str(model_file), str(output_dir))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


