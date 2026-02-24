#!/usr/bin/env bash
#
# Convert and run Solar analysis on all SolBench V1 benchmark levels.
#
# This script:
#   1. Converts all benchmark files (L1, L2, L1-Quant, L2-Quant) to Solar format
#   2. Runs full Solar pipeline on all converted models
#
# Usage:
#   ./run_solbenchv1_all.sh                    # Convert and run all
#   ./run_solbenchv1_all.sh --convert-only     # Only convert, don't run analysis
#   ./run_solbenchv1_all.sh --run-only         # Only run analysis (assumes already converted)
#   ./run_solbenchv1_all.sh --max-models 10    # Limit to 10 models per level
#   ./run_solbenchv1_all.sh --arch H100_PCIe   # Use H100 instead of B200
#
# Output:
#   Converted files: solbenchv1-postprocess/<level>/
#   Analysis results: solar/output_solbenchv1/<level>/<model>/
#   Compliance CSV: solbenchv1-postprocess/solbenchv1_compliance_summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default settings
CONVERT_ONLY=false
RUN_ONLY=false
MAX_MODELS=""
ARCH="B200"
PRECISION="bf16"
DEBUG=""
SKIP_EXISTING=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Convert and run Solar analysis on all SolBench V1 benchmark levels.

Options:
  --convert-only       Only convert files, don't run Solar analysis
  --run-only           Only run analysis (assumes files already converted)
  --max-models N       Maximum number of models to process per level
  --arch ARCH          Architecture config (default: B200)
  --precision PRE      Precision (fp32, fp16, bf16) (default: bf16)
  --skip-existing      Skip models that already have output
  --debug              Enable debug output
  -h, --help           Show this help message

Examples:
  $(basename "$0")                           # Full pipeline with B200
  $(basename "$0") --convert-only            # Just convert files
  $(basename "$0") --arch H100_PCIe          # Use H100 architecture
  $(basename "$0") --max-models 5            # Process 5 models per level
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --convert-only)
            CONVERT_ONLY=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --max-models)
            MAX_MODELS="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            shift
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

echo -e "${BLUE}=========================================="
echo "SolBench V1 Full Pipeline"
echo "==========================================${NC}"
echo ""
echo "Architecture: ${ARCH}"
echo "Precision: ${PRECISION}"
echo ""

# Step 1: Convert all benchmark files
if [[ "$RUN_ONLY" != "true" ]]; then
    echo -e "${BLUE}Step 1: Converting benchmark files...${NC}"
    echo ""
    
    CONVERT_CMD="${SCRIPT_DIR}/convert_solbenchv1.sh"
    
    if [[ -n "$MAX_MODELS" ]]; then
        CONVERT_CMD="$CONVERT_CMD --max-files $MAX_MODELS"
    fi
    
    if [[ -n "$DEBUG" ]]; then
        CONVERT_CMD="$CONVERT_CMD $DEBUG"
    fi
    
    $CONVERT_CMD
    
    echo ""
    echo -e "${GREEN}Conversion complete!${NC}"
    echo ""
fi

# Step 2: Run Solar analysis
if [[ "$CONVERT_ONLY" != "true" ]]; then
    echo -e "${BLUE}Step 2: Running Solar analysis...${NC}"
    echo ""
    
    RUN_CMD="${SCRIPT_DIR}/run_solbenchv1.sh"
    RUN_CMD="$RUN_CMD --arch $ARCH"
    RUN_CMD="$RUN_CMD --precision $PRECISION"
    
    if [[ -n "$MAX_MODELS" ]]; then
        RUN_CMD="$RUN_CMD --max-models $MAX_MODELS"
    fi
    
    if [[ -n "$SKIP_EXISTING" ]]; then
        RUN_CMD="$RUN_CMD $SKIP_EXISTING"
    fi
    
    if [[ -n "$DEBUG" ]]; then
        RUN_CMD="$RUN_CMD $DEBUG"
    fi
    
    $RUN_CMD
    
    echo ""
    echo -e "${GREEN}Analysis complete!${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Pipeline Complete!"
echo "==========================================${NC}"
echo ""
echo "Outputs:"
echo "  - Converted files: solbenchv1-postprocess/<level>/"
echo "  - Compliance CSV:  solbenchv1-postprocess/solbenchv1_compliance_summary.csv"
if [[ "$CONVERT_ONLY" != "true" ]]; then
    echo "  - Analysis:        solar/output_solbenchv1/<level>/<model>/"
fi
echo ""
