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

"""CLI: translate DSL output directory into einsum_graph.yaml (graph/ stage).

Reads *_einsum_lib.yaml + *_einsum_lib.py from <dsl_dir> and writes
<model_name>_einsum_graph.yaml to <output_dir> (default: ../graph/ relative
to <dsl_dir>, mirroring the output_verified/graph/ convention).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .loader import load_dsl_yaml
from .ast_parser import extract_loop_exprs
from .assembler import build_dsl_graph
from .exporter import dsl_graph_to_dict, write_einsum_yaml


def translate(
    dsl_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Translate DSL output directory to einsum_graph.yaml (graph stage).

    einsum_equation is written as null here; use cli_einsum to produce a
    version with equations filled in.

    Args:
        dsl_dir:    Directory containing *_einsum_lib.yaml and *_einsum_lib.py.
        output_dir: Directory to write <model_name>_einsum_graph.yaml.
                    Defaults to <dsl_dir>/../graph/

    Returns:
        Path to the written output file.
    """
    dsl_dir = Path(dsl_dir).resolve()

    # Detect the *_einsum_lib.yaml file
    yaml_files = list(dsl_dir.glob("*_einsum_lib.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No *_einsum_lib.yaml found in {dsl_dir}")
    if len(yaml_files) > 1:
        raise ValueError(f"Multiple *_einsum_lib.yaml files found in {dsl_dir}: {yaml_files}")
    yaml_path = yaml_files[0]

    # Detect the matching *_einsum_lib.py
    stem = yaml_path.stem  # e.g. "Attention_einsum_lib"
    py_path = dsl_dir / f"{stem}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"Expected Python DSL file not found: {py_path}")

    model_name = stem.removesuffix("_einsum_lib")

    # Default output dir: sibling of dsl_dir named "graph"
    if output_dir is None:
        output_dir = dsl_dir.parent / "graph"
    output_dir = Path(output_dir)
    output_path = output_dir / f"{model_name}_einsum_graph.yaml"

    # Pipeline: load → parse → assemble → export
    raw_yaml = load_dsl_yaml(yaml_path)
    loop_exprs = extract_loop_exprs(py_path)
    dsl_graph = build_dsl_graph(raw_yaml, loop_exprs, model_name)
    graph_dict = dsl_graph_to_dict(dsl_graph)
    write_einsum_yaml(graph_dict, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Translate DSL output (toDSL_model) to einsum_graph.yaml (graph stage). "
            "einsum_equation fields are left null; run solar.einsum_dsl.cli_einsum "
            "to produce a version with equations."
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
        help="Output directory (default: <dsl_dir>/../graph/)",
    )
    args = parser.parse_args()

    try:
        out = translate(args.dsl_dir, args.output_dir)
        print(f"Written: {out}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
