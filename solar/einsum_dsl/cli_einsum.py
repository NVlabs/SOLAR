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

"""CLI: convert DSL graph/ output into einsum_graph.yaml with einsum equations filled in.

Reads *_einsum_lib.yaml + *_einsum_lib.py from <dsl_dir>, builds the DSLGraph,
fills in einsum_equation for every kernel, then writes to <output_dir>
(default: <dsl_dir>/../einsum/).

This mirrors the output_verified/einsum/ stage of the original solar pipeline.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .loader import load_dsl_yaml
from .ast_parser import extract_loop_exprs, extract_kernel_info, module_source_record
from .assembler import build_dsl_graph
from .exporter import dsl_graph_to_dict, write_einsum_yaml
from .einsum_converter import annotate_einsum_equations


def translate_with_einsum(
    dsl_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Build einsum_graph.yaml with einsum equations derived from tensor dims.

    Args:
        dsl_dir:    Directory containing *_einsum_lib.yaml and *_einsum_lib.py.
        output_dir: Directory to write output. Default: <dsl_dir>/../einsum/

    Returns:
        Path to the written output file.
    """
    dsl_dir = Path(dsl_dir).resolve()

    yaml_files = list(dsl_dir.glob("*_einsum_lib.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No *_einsum_lib.yaml found in {dsl_dir}")
    if len(yaml_files) > 1:
        raise ValueError(f"Multiple *_einsum_lib.yaml files found in {dsl_dir}: {yaml_files}")
    yaml_path = yaml_files[0]

    stem = yaml_path.stem
    py_path = dsl_dir / f"{stem}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"Expected Python DSL file not found: {py_path}")

    model_name = stem.removesuffix("_einsum_lib")

    if output_dir is None:
        output_dir = dsl_dir.parent / "einsum"
    output_dir = Path(output_dir)
    output_path = output_dir / f"{model_name}_einsum_graph.yaml"

    # Pipeline: load → parse → assemble → export → annotate equations
    raw_yaml = load_dsl_yaml(yaml_path)
    kernel_info = extract_kernel_info(py_path)
    dsl_graph = build_dsl_graph(raw_yaml, None, model_name, kernel_info=kernel_info)
    graph_dict = dsl_graph_to_dict(dsl_graph)
    graph_dict = annotate_einsum_equations(dsl_graph, graph_dict)
    # the DSL module text, hashed: the anchor every kernel node's `kernel_source` binds to (the analyzer re-derives
    # the MAC operand occurrences from it; the per-node channels are a validated cache, DTYPE-OCCURRENCE-CHANNEL-8)
    graph_dict["dsl_module_source"] = module_source_record(py_path)
    write_einsum_yaml(graph_dict, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert DSL output to einsum_graph.yaml with einsum equations. "
            "Reads *_einsum_lib.yaml and *_einsum_lib.py from dsl_dir, "
            "derives einsum equations from tensor dims, writes to output_dir."
        )
    )
    parser.add_argument(
        "dsl_dir",
        help="Directory containing *_einsum_lib.yaml and *_einsum_lib.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Output directory (default: <dsl_dir>/../einsum/)",
    )
    args = parser.parse_args()

    try:
        out = translate_with_einsum(args.dsl_dir, args.output_dir)
        print(f"Written: {out}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
