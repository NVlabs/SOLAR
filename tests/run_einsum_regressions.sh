#!/usr/bin/env bash
set -euo pipefail

# Run all Solar regression and unit tests.
#
# Usage:
#   bash tests/run_einsum_regressions.sh          # Run all tests
#   bash tests/run_einsum_regressions.sh quick     # Skip slow end-to-end tests

cd "$(dirname "$0")/.."

MODE="${1:-all}"

# Core unit tests (fast, no model tracing)
UNIT_TESTS=(
  tests/test_einsum_analyzer.py
  tests/test_pytorch_to_einsum_regressions.py
)

# End-to-end regression tests (trace models, slower)
E2E_TESTS=(
  tests/test_graph_analyzer_regression.py
  tests/test_graph_analyzer_memory.py
  tests/test_zero_compute_ops.py
  tests/test_perf_quant.py
)

# Other tests
OTHER_TESTS=(
  tests/test_transpose_args.py
  tests/test_extended_einsum.py
  tests/test_hidden_tensor_removal.py
  tests/test_torchview_processor.py
  tests/test_graph_processing.py
)

if [[ "$MODE" == "quick" ]]; then
  echo "==> Running unit tests only (quick mode)"
  python3 -m pytest "${UNIT_TESTS[@]}" -v
elif [[ "$MODE" == "e2e" ]]; then
  echo "==> Running end-to-end regression tests"
  python3 -m pytest "${E2E_TESTS[@]}" -v
elif [[ "$MODE" == "all" ]]; then
  echo "==> Running all tests"
  python3 -m pytest \
    "${UNIT_TESTS[@]}" \
    "${E2E_TESTS[@]}" \
    "${OTHER_TESTS[@]}" \
    -v
else
  echo "Usage: $0 [quick|e2e|all]"
  exit 1
fi
