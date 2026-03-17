#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pytest \
  tests/test_pytorch_to_einsum_regressions.py \
  tests/test_graph_analyzer_regression.py::TestLinearWithBiasEndToEnd::test_total_macs \
  tests/test_graph_analyzer_regression.py::TestConv2dEndToEnd::test_total_macs \
  -v
