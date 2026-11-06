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

from solar.perf import EinsumGraphPerfModel, PerfProjectionUnavailable
from solar.perf.arch_rates import DEFAULT_SOL_MODE, SOL_MODES
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
        "--dtype-bytes",
        action="store_true",
        help="Price memory traffic at each tensor's own dtype width (bool 1 B, fp32 4 B, fp8 1 B ...) "
             "instead of one bytes_per_element for the whole graph; --precision then only picks the MAC rate.",
    )
    parser.add_argument(
        "--sol-mode",
        choices=list(SOL_MODES),
        default=DEFAULT_SOL_MODE,
        help="Which tensor-core peak set prices the headline SOL (issue #14): 'sparse' (default; the vendor's "
             "structured-sparsity peak, a lower bound) or 'dense' (the more realistic estimate for dense work). "
             "perf_<arch>.yaml always carries both as sol_sparse / sol_dense.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )

    args = parser.parse_args()

    from solar.common.cli_logging import configure as _configure_cli_logging

    _configure_cli_logging(bool(getattr(args, "debug", False) or getattr(args, "verbose", False)))
    analysis_path = Path(args.analysis_path)
    if not analysis_path.exists():
        print(f"❌ analysis.yaml not found: {analysis_path}")
        sys.exit(2)

    output_dir = ensure_directory(args.output_dir)
    model = EinsumGraphPerfModel(debug=args.debug, dtype_bytes=args.dtype_bytes)
    try:
        perf = model.predict(
            analysis_path,
            output_dir,
            arch_config=args.arch_config,
            precision=args.precision,
            copy_analysis=not args.no_copy_analysis,
            sol_mode=args.sol_mode,
        )
    except PerfProjectionUnavailable as exc:
        # Jenny's decision (2026-10-26): a declared compute dtype without a
        # tensor-core rate on this profile is a hard error — no perf_<arch>.yaml,
        # only the REFUSED receipt the model left in the output directory.
        print(f"❌ {exc.message}")
        print(f"   no perf artifact written for {exc.arch}; receipt: {output_dir / f'perf_{exc.arch}.REFUSED.yaml'}")
        sys.exit(3)
    if perf is None:
        print("❌ Perf prediction failed.")
        sys.exit(1)

    print("✅ Perf prediction complete.")
    print(f"  Arch: {perf.get('arch', {}).get('name', 'unknown')}")
    print(f"  Unfused runtime (ms): {perf.get('unfused', {}).get('runtime_ms', 0.0):.4f}")
    print(f"  Fused runtime (ms): {perf.get('fused', {}).get('runtime_ms', 0.0):.4f}")
    sp = perf.get("sol_sparse", {}); dn = perf.get("sol_dense", {})
    print(f"  SOL mode: {perf.get('sol_mode')} (headline; default {perf.get('sol_mode_default')}). Both sets in the record: "
          f"sparse fused {sp.get('fused', {}).get('runtime_ms', 0.0):.4f} ms (lower bound) | "
          f"dense fused {dn.get('fused', {}).get('runtime_ms', 0.0):.4f} ms (more realistic for dense work)")
    conv = perf.get("sol_convention", {}) or {}; term = (perf.get("workload", {}) or {}).get("tensor_core_term", {}) or {}
    other_priced = term.get("non_contraction_other_macs_priced")
    print(f"  Headline: {conv.get('headline', 'unlabelled')} (issue #58 convention B). Tensor-core term prices contraction MACs"
          + (f" + {int(other_priced):,} other non-contraction MACs" if other_priced else "")
          + f"; excluded: {int(term.get('scalar_statement_multiplies_excluded', 0)):,} scalar statement multiplies and "
          f"{int((perf.get('workload', {}) or {}).get('total_other_ops', 0)):,} other_ops (both counts retained; compute_sm_cycles prices other_ops only, unchanged)")
    print(f"\n📝 Files saved to {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
