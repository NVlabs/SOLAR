#!/usr/bin/env bash
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

set -euo pipefail

# Run the SOLAR mechanical pipeline for the TwoInputMatmul example and check
# the documented counts.
#
# A [32,64] @ B [64,128] -> [32,128]
#
# Expected results:
#   MACs:           262,144   (32 * 64 * 128)
#   Unfused elems:  14,336    (2048 + 8192 + 4096)
#
# Outputs are written under examples/TwoInputMatmul/output/ (or
# SOLAR_TWOINPUTMATMUL_OUTPUT_DIR): {graph,einsum,analysis,perf}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

MODEL_FILE="${SCRIPT_DIR}/TwoInputMatmul.py"
OUT_BASE="${SOLAR_TWOINPUTMATMUL_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GRAPH_OUT="${OUT_BASE}/graph"
EINSUM_OUT="${OUT_BASE}/einsum"
ANALYSIS_OUT="${OUT_BASE}/analysis"
PERF_OUT="${OUT_BASE}/perf"

mkdir -p "${GRAPH_OUT}" "${EINSUM_OUT}" "${ANALYSIS_OUT}" "${PERF_OUT}"

cd "${SOLAR_ROOT}"

echo "==> Processing model -> ${GRAPH_OUT}"
"${PYTHON}" -m solar.cli.process_model --model-file "${MODEL_FILE}" --output-dir "${GRAPH_OUT}" --force-rerun

echo "==> Route check"
"${PYTHON}" -m solar.frontend.router "${GRAPH_OUT}/pytorch_graph.yaml"

echo "==> Converting pytorch graph -> ${EINSUM_OUT}"
"${PYTHON}" -m solar.cli.toeinsum_model --graph-path "${GRAPH_OUT}/pytorch_graph.yaml" \
  --output-dir "${EINSUM_OUT}" --no-copy-graph --enable-rename

echo "==> Analyzing einsum graph -> ${ANALYSIS_OUT}"
"${PYTHON}" -m solar.cli.analyze_model --einsum-graph-path "${EINSUM_OUT}/einsum_graph_renamed.yaml" \
  --output-dir "${ANALYSIS_OUT}" --no-copy-graph

echo "==> Predicting perf -> ${PERF_OUT}"
"${PYTHON}" -m solar.cli.predict_perf_model --analysis-path "${ANALYSIS_OUT}/analysis.yaml" \
  --output-dir "${PERF_OUT}" --arch-config B200 --precision fp16

echo "==> Checking the documented counts"
"${PYTHON}" - "${ANALYSIS_OUT}/analysis.yaml" "${PERF_OUT}/perf_B200.yaml" <<'PY'
import sys, yaml
analysis = yaml.safe_load(open(sys.argv[1]))["total"]
perf = yaml.safe_load(open(sys.argv[2]))
checks = {"total.macs": (int(analysis["macs"]), 262144),
          "total.unfused_elements": (int(analysis["unfused_elements"]), 14336),
          "unfused.memory_bytes": (int(perf["unfused"]["memory_bytes"]), 28672)}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
for k, (got, want) in checks.items():
    print(f"  {k}: {got} (expected {want}) {'OK' if got == want else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY

echo ""
echo "Done."
echo "Analysis: ${ANALYSIS_OUT}/analysis.yaml"
echo "Perf:     ${PERF_OUT}/perf_B200.yaml"
