#!/usr/bin/env bash
#
# Convert SolBench models to KernelBench-compatible format.
#
# Usage:
#   ./convert_solbench.sh                    # Convert all models
#   ./convert_solbench.sh --max-files 10     # Convert first 10 models
#   ./convert_solbench.sh --debug            # Enable debug output
#
# Outputs: solbench-postprocess/<filename>.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

# Set PYTHONPATH for solar module imports
export PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH:-}"

SOLBENCH_INPUT="${REPO_ROOT}/sol-bench/data/sample"
SOLBENCH_OUTPUT="${REPO_ROOT}/solbench-postprocess"

# Default configuration
BATCH=32
SEQ_LEN=32
HIDDEN_SIZE=6144
MAX_FILES=""
DEBUG=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Convert SolBench models to KernelBench-compatible format.

Options:
  --batch N          Batch size (default: $BATCH)
  --seq-len N        Sequence length (default: $SEQ_LEN)
  --hidden-size N    Hidden size (default: $HIDDEN_SIZE)
  --max-files N      Maximum number of files to convert
  --debug            Enable debug output
  -h, --help         Show this help message

Configuration used:
  hidden_size: $HIDDEN_SIZE
  num_attention_heads: 64
  num_key_value_heads: 8
  head_dim: 96
  intermediate_size: 19648
  rms_norm_eps: 1e-05
  dtype: bfloat16
  batch: $BATCH
  seq_len: $SEQ_LEN

Output: $SOLBENCH_OUTPUT/
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check input directory
if [[ ! -d "$SOLBENCH_INPUT" ]]; then
    echo -e "${RED}Error: SolBench input directory not found: $SOLBENCH_INPUT${NC}"
    echo "Please ensure sol-bench is cloned to: ${REPO_ROOT}/sol-bench/"
    exit 1
fi

# Create output directory
mkdir -p "$SOLBENCH_OUTPUT"

echo -e "${BLUE}=========================================="
echo "SolBench to KernelBench Converter"
echo "==========================================${NC}"
echo ""
echo "Input:  $SOLBENCH_INPUT"
echo "Output: $SOLBENCH_OUTPUT"
echo ""
echo "Configuration:"
echo "  batch: $BATCH"
echo "  seq_len: $SEQ_LEN"
echo "  hidden_size: $HIDDEN_SIZE"
echo ""

# Build command
CMD="python3 -m solar.benchmark.solbench.converter"
CMD="$CMD --input-dir $SOLBENCH_INPUT"
CMD="$CMD --output-dir $SOLBENCH_OUTPUT"
CMD="$CMD --batch $BATCH"
CMD="$CMD --seq-len $SEQ_LEN"
CMD="$CMD --hidden-size $HIDDEN_SIZE"

if [[ -n "$MAX_FILES" ]]; then
    CMD="$CMD --max-files $MAX_FILES"
fi

if [[ -n "$DEBUG" ]]; then
    CMD="$CMD $DEBUG"
fi

# Run conversion
echo -e "${BLUE}Converting models...${NC}"
cd "${SOLAR_ROOT}"
$CMD

echo ""
echo -e "${GREEN}Conversion complete!${NC}"
echo ""
echo "Converted files are in: $SOLBENCH_OUTPUT/"
echo ""
echo "To run Solar analysis on converted models:"
echo "  ./run_solbench.sh"
