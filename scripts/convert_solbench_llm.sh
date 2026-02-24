#!/usr/bin/env bash
#
# Convert SolBench-20260206v2 files to KernelBench format using LLM
#
# Usage:
#   ./convert_solbench_llm.sh                    # Convert all levels
#   ./convert_solbench_llm.sh --level L1         # Convert only L1
#   ./convert_solbench_llm.sh --max-files 5      # Test with 5 files
#   ./convert_solbench_llm.sh --dry-run          # Just list files
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/../.." && pwd)"

INPUT_BASE="${REPO_ROOT}/sol-bench-20260206v2"
OUTPUT_BASE="${REPO_ROOT}/solbench-20260206v2-postprocess-llm"

# Default settings
MAX_FILES=""
DRY_RUN=""
VERBOSE=""
LEVEL=""
MODEL="gpt-4o"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Convert SolBench-20260206v2 files to KernelBench format using LLM.

Options:
  --level LEVEL      Convert only specified level (L1, L2, L1-Quant, L2-Quant)
  --max-files N      Maximum number of files to convert per level
  --model MODEL      OpenAI model to use (default: gpt-4o)
  --dry-run          Don't actually convert, just list files
  --verbose, -v      Verbose output
  -h, --help         Show this help message

Requirements:
  - OpenAI API key in OPENAI_API_KEY environment variable
  - pip install openai

Examples:
  $(basename "$0")                              # Convert all levels
  $(basename "$0") --level L1                   # Convert only L1
  $(basename "$0") --max-files 5 --dry-run     # Test with 5 files
  $(basename "$0") --model gpt-4o-mini         # Use cheaper model
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="--max-files $2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
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

# Check for API key
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY=your-api-key"
    exit 1
fi

# Check input directory
if [[ ! -d "$INPUT_BASE" ]]; then
    echo "Error: Input directory not found: $INPUT_BASE"
    exit 1
fi

# Determine levels to process
levels_to_process=()
if [[ -n "$LEVEL" ]]; then
    if [[ ! -d "$INPUT_BASE/$LEVEL" ]]; then
        echo "Error: Level directory not found: $INPUT_BASE/$LEVEL"
        exit 1
    fi
    levels_to_process=("$LEVEL")
else
    for level_dir in "$INPUT_BASE"/*; do
        if [[ -d "$level_dir" ]]; then
            level_name=$(basename "$level_dir")
            levels_to_process+=("$level_name")
        fi
    done
fi

echo "=========================================="
echo "LLM-Based SolBench Converter"
echo "=========================================="
echo "Model: $MODEL"
echo "Input:  $INPUT_BASE"
echo "Output: $OUTPUT_BASE"
echo "Levels: ${levels_to_process[*]}"
echo "=========================================="
echo ""

# Check if openai package is installed
if ! python3 -c "import openai" 2>/dev/null; then
    echo "Error: openai package not installed"
    echo "Please install it with: pip install openai"
    exit 1
fi

total_succeeded=0
total_failed=0

# Process each level
for level in "${levels_to_process[@]}"; do
    echo "Processing level: $level"
    echo ""

    level_input="$INPUT_BASE/$level"
    level_output="$OUTPUT_BASE/$level"

    # Run converter
    cd "${SOLAR_ROOT}/solar/benchmark/solbench"
    python3 llm_converter.py \
        --input-dir "$level_input" \
        --output-dir "$level_output" \
        --model "$MODEL" \
        $MAX_FILES \
        $DRY_RUN \
        $VERBOSE

    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        if [[ -z "$DRY_RUN" ]]; then
            file_count=$(find "$level_output" -name "*.py" 2>/dev/null | wc -l)
            total_succeeded=$((total_succeeded + file_count))
            echo "✓ Level $level completed: $file_count files"
        fi
    else
        echo "✗ Level $level failed"
        total_failed=$((total_failed + 1))
    fi

    echo ""
done

if [[ -z "$DRY_RUN" ]]; then
    echo "=========================================="
    echo "FINAL SUMMARY"
    echo "=========================================="
    echo "Total files converted: $total_succeeded"
    echo "Levels failed: $total_failed"
    echo "Output directory: $OUTPUT_BASE"
    echo "=========================================="
fi

exit 0
