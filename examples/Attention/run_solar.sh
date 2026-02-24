#!/usr/bin/env bash
set -euo pipefail

# Run Solar processing + einsum pipeline for the standalone Attention example.
#
# This example demonstrates the multi-head attention mechanism:
#   Q, K, V projections -> QK attention scores -> softmax -> context -> output projection
#
# Outputs are written under:
#   solar/examples/Attention/output/{graph,einsum,analysis,perf,timeloop}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FILE="${SCRIPT_DIR}/Attention.py"
OUT_BASE="${SOLAR_ATTENTION_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GRAPH_OUT="${OUT_BASE}/graph"
EINSUM_OUT="${OUT_BASE}/einsum"
ANALYSIS_OUT="${OUT_BASE}/analysis"
PERF_OUT="${OUT_BASE}/perf"
TIMELOOP_OUT="${OUT_BASE}/timeloop"

if ! mkdir -p "${GRAPH_OUT}" "${EINSUM_OUT}" "${ANALYSIS_OUT}" "${PERF_OUT}" "${TIMELOOP_OUT}"; then
  echo "❌ Failed to create output directories under: ${OUT_BASE}" >&2
  echo "   Tip: set SOLAR_ATTENTION_OUTPUT_DIR to a writable path, e.g.:" >&2
  echo "     SOLAR_ATTENTION_OUTPUT_DIR=/tmp/solar_attention_output bash ${SCRIPT_DIR}/run_solar.sh" >&2
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
echo "=== Attention Example Outputs ==="
echo "PyTorch graph:   ${GRAPH_OUT}/pytorch_graph.yaml"
echo "Einsum graph:    ${EINSUM_OUT}/einsum_graph.yaml"
echo "Einsum renamed:  ${EINSUM_OUT}/einsum_graph_renamed.yaml"
echo "Graph PDF:       ${EINSUM_OUT}/einsum_graph.pdf"
echo "Analysis:        ${ANALYSIS_OUT}/analysis.yaml"
echo "Perf:            ${PERF_OUT}/perf_H100_PCIe.yaml"
echo "Timeloop graph:  ${TIMELOOP_OUT}/timeloop_graph.yaml"
echo ""
echo "The attention mechanism includes:"
echo "  - Q, K, V linear projections"
echo "  - QK = Q @ K^T / sqrt(d_k)  (scaled dot-product)"
echo "  - softmax(QK)"
echo "  - context = attn_weights @ V"
echo "  - output projection"

