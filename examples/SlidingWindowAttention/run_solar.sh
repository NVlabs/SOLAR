#!/usr/bin/env bash
set -euo pipefail

# Run Solar processing + einsum pipeline for the Sliding Window Attention example.
#
# Sliding Window Attention only computes attention within a local window:
#   For each position i, attend to [i - window_size, i + window_size]
#
# This is more efficient than dense attention for long sequences and is used
# in models like Longformer and BigBird.
#
# Outputs are written under:
#   solar/examples/SlidingWindowAttention/output/{graph,einsum,analysis,perf,timeloop}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FILE="${SCRIPT_DIR}/SlidingWindowAttention.py"
OUT_BASE="${SOLAR_SLIDING_WINDOW_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GRAPH_OUT="${OUT_BASE}/graph"
EINSUM_OUT="${OUT_BASE}/einsum"
ANALYSIS_OUT="${OUT_BASE}/analysis"
PERF_OUT="${OUT_BASE}/perf"
TIMELOOP_OUT="${OUT_BASE}/timeloop"

if ! mkdir -p "${GRAPH_OUT}" "${EINSUM_OUT}" "${ANALYSIS_OUT}" "${PERF_OUT}" "${TIMELOOP_OUT}"; then
  echo "❌ Failed to create output directories under: ${OUT_BASE}" >&2
  echo "   Tip: set SOLAR_SLIDING_WINDOW_OUTPUT_DIR to a writable path, e.g.:" >&2
  echo "     SOLAR_SLIDING_WINDOW_OUTPUT_DIR=/tmp/solar_sliding_window_output bash ${SCRIPT_DIR}/run_solar.sh" >&2
  exit 1
fi

cd "${SOLAR_ROOT}"

echo "==> Processing model -> ${GRAPH_OUT}"
python3 -m solar.cli.process_model \
  --model-file "${MODEL_FILE}" \
  --output-dir "${GRAPH_OUT}" \
  --save-graph \
  --force-rerun

echo "==> Converting pytorch graph -> ${EINSUM_OUT}"
python3 -m solar.cli.toeinsum_model \
  --graph-path "${GRAPH_OUT}/pytorch_graph.yaml" \
  --output-dir "${EINSUM_OUT}" \
  --no-copy-graph \
  --save-graph

echo "==> Analyzing einsum graph -> ${ANALYSIS_OUT}"
python3 -m solar.cli.analyze_model \
  --einsum-graph-path "${EINSUM_OUT}/einsum_graph_renamed.yaml" \
  --output-dir "${ANALYSIS_OUT}"

echo "==> Predicting perf -> ${PERF_OUT}"
python3 -m solar.cli.predict_perf_model \
  --analysis-path "${ANALYSIS_OUT}/analysis.yaml" \
  --output-dir "${PERF_OUT}" \
  --arch-config "H100_PCIe" \
  --precision "fp32"

echo "==> Converting to Timeloop format -> ${TIMELOOP_OUT}"
python3 -m solar.cli.totimeloop \
  --einsum-graph-path "${EINSUM_OUT}/einsum_graph_renamed.yaml" \
  --output-dir "${TIMELOOP_OUT}"

echo ""
echo "Done."
echo ""
echo "=== Sliding Window Attention Example Outputs ==="
echo "PyTorch graph:   ${GRAPH_OUT}/pytorch_graph.yaml"
echo "Einsum graph:    ${EINSUM_OUT}/einsum_graph.yaml"
echo "Einsum renamed:  ${EINSUM_OUT}/einsum_graph_renamed.yaml"
echo "Graph PDF:       ${EINSUM_OUT}/einsum_graph.pdf"
echo "Analysis:        ${ANALYSIS_OUT}/analysis.yaml"
echo "Perf:            ${PERF_OUT}/perf_H100_PCIe.yaml"
echo "Timeloop graph:  ${TIMELOOP_OUT}/timeloop_graph.yaml"
echo ""
echo "Sliding window attention (window_size=4):"
echo "  - Each position attends only to nearby positions"
echo "  - Complexity: O(n * window_size) instead of O(n^2)"
echo "  - Used in Longformer, BigBird, etc."

