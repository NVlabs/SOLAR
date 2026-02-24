#!/bin/bash
#
# run_kernelbench_einsum_verification.sh
#
# Runs einsum verification on all kernelbench benchmarks and generates
# a summary CSV with pass/fail status and failure reasons.
#
# Usage:
#   ./run_kernelbench_einsum_verification.sh                    # Run all levels
#   ./run_kernelbench_einsum_verification.sh --level level1     # Run specific level
#   ./run_kernelbench_einsum_verification.sh --kernel-ids 19 20 # Run specific kernels
#   ./run_kernelbench_einsum_verification.sh --verbose          # Verbose output
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SOLAR_ROOT/.." && pwd)"

# Default values
OUTPUT_DIR="${SOLAR_ROOT}/output_kernelbench"
LEVEL="level1"
KERNEL_IDS=""
VERBOSE=""
SCALE="0.01"
CSV_OUTPUT="${SOLAR_ROOT}/einsum_verification_results.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --kernel-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                KERNEL_IDS="$KERNEL_IDS $1"
                shift
            done
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --output-csv)
            CSV_OUTPUT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run einsum verification on kernelbench benchmarks."
            echo ""
            echo "Options:"
            echo "  --level LEVEL          Kernel level (default: level1)"
            echo "  --kernel-ids IDS...    Specific kernel IDs to verify"
            echo "  --verbose, -v          Enable verbose output"
            echo "  --scale FACTOR         Scale factor for tensor dimensions (default: 0.01)"
            echo "  --output-csv FILE      Output CSV file (default: einsum_verification_results.csv)"
            echo "  --output-dir DIR       Output directory (default: output_kernelbench)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Verify all level1 kernels"
            echo "  $0 --level level2                    # Verify all level2 kernels"
            echo "  $0 --kernel-ids 19 20 21             # Verify specific kernels"
            echo "  $0 --verbose --kernel-ids 19         # Verbose verification of kernel 19"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo "========================================================================"
echo "KERNELBENCH EINSUM VERIFICATION"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Level: ${LEVEL}"
echo "  Scale factor: ${SCALE}"
echo "  CSV output: ${CSV_OUTPUT}"
if [[ -n "$KERNEL_IDS" ]]; then
    echo "  Kernel IDs:${KERNEL_IDS}"
fi
echo ""

# Check if output directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "❌ Output directory not found: ${OUTPUT_DIR}"
    exit 1
fi

# Check level directory
LEVEL_DIR="${OUTPUT_DIR}/${LEVEL}"
if [[ ! -d "$LEVEL_DIR" ]]; then
    echo "❌ Level directory not found: ${LEVEL_DIR}"
    exit 1
fi

# Build Python command
VERIFY_CMD="python3 -m solar.cli.verify_einsum"
VERIFY_ARGS="--output-dir ${OUTPUT_DIR} --level ${LEVEL} --scale ${SCALE}"

if [[ -n "$KERNEL_IDS" ]]; then
    VERIFY_ARGS="$VERIFY_ARGS --kernel-ids${KERNEL_IDS}"
fi

if [[ -n "$VERBOSE" ]]; then
    VERIFY_ARGS="$VERIFY_ARGS $VERBOSE"
fi

# Change to solar directory for module import
cd "$SOLAR_ROOT"

# Add solar to PYTHONPATH
export PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH}"

# Run verification
echo "Running verification..."
echo "Command: $VERIFY_CMD $VERIFY_ARGS"
echo ""

# Run verification (allow failure to continue with summary)
set +e
$VERIFY_CMD $VERIFY_ARGS
VERIFY_EXIT_CODE=$?
set -e

echo ""
echo "========================================================================"
echo "COLLECTING VERIFICATION RESULTS"
echo "========================================================================"
echo ""

# Initialize CSV
echo "benchmark_name,status,error_type,error_message" > "$CSV_OUTPUT"

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Collect results from einsum_verification.yaml files
for KERNEL_DIR in "$LEVEL_DIR"/*; do
    if [[ ! -d "$KERNEL_DIR" ]]; then
        continue
    fi
    
    KERNEL_NAME=$(basename "$KERNEL_DIR")
    VERIFICATION_FILE="${KERNEL_DIR}/einsum_verification/einsum_verification.yaml"
    
    if [[ ! -f "$VERIFICATION_FILE" ]]; then
        # No verification result - skip or mark as not run
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    
    # Parse YAML using Python for reliability
    RESULT=$(python3 -c "
import yaml
import sys

try:
    with open('$VERIFICATION_FILE') as f:
        data = yaml.safe_load(f)
    
    status = data.get('status', 'unknown')
    error_type = ''
    error_message = ''
    
    if status == 'failed':
        error = data.get('error', {})
        error_type = error.get('type', 'unknown')
        error_message = error.get('message', 'Unknown error')
        # Escape quotes and commas for CSV
        error_message = error_message.replace('\"', '\"\"').replace(',', ';')
    
    print(f'{status},{error_type},{error_message}')
except Exception as e:
    print(f'error,parse_error,Failed to parse: {e}')
")
    
    STATUS=$(echo "$RESULT" | cut -d',' -f1)
    ERROR_TYPE=$(echo "$RESULT" | cut -d',' -f2)
    ERROR_MSG=$(echo "$RESULT" | cut -d',' -f3-)
    
    # Add to CSV
    echo "\"${KERNEL_NAME}\",\"${STATUS}\",\"${ERROR_TYPE}\",\"${ERROR_MSG}\"" >> "$CSV_OUTPUT"
    
    if [[ "$STATUS" == "passed" ]]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# Print summary
echo ""
echo "========================================================================"
echo "VERIFICATION SUMMARY"
echo "========================================================================"
echo ""
echo "Total benchmarks: ${TOTAL}"
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo ""

if [[ $TOTAL -gt 0 ]]; then
    PASS_RATE=$(echo "scale=1; ${PASSED} * 100 / ${TOTAL}" | bc)
    echo "Pass rate: ${PASS_RATE}%"
    echo ""
fi

echo "Results saved to: ${CSV_OUTPUT}"
echo ""

# Print failed benchmarks
if [[ $FAILED -gt 0 ]]; then
    echo "Failed benchmarks:"
    echo "------------------------------------------------------------------------"
    # Skip header and print failed entries
    tail -n +2 "$CSV_OUTPUT" | grep -v ",\"passed\"," | while IFS= read -r line; do
        NAME=$(echo "$line" | cut -d'"' -f2)
        ERROR_TYPE=$(echo "$line" | cut -d'"' -f6)
        ERROR_MSG=$(echo "$line" | cut -d'"' -f8)
        echo "  ❌ ${NAME}: ${ERROR_TYPE}"
        if [[ -n "$VERBOSE" ]]; then
            echo "     ${ERROR_MSG}"
        fi
    done
    echo ""
fi

# Print passed benchmarks
if [[ $PASSED -gt 0 ]]; then
    echo "Passed benchmarks:"
    echo "------------------------------------------------------------------------"
    tail -n +2 "$CSV_OUTPUT" | grep ",\"passed\"," | while IFS= read -r line; do
        NAME=$(echo "$line" | cut -d'"' -f2)
        echo "  ✅ ${NAME}"
    done
    echo ""
fi

echo "========================================================================"
if [[ $FAILED -eq 0 ]]; then
    echo "✅ ALL VERIFICATIONS PASSED"
else
    echo "❌ ${FAILED} VERIFICATION(S) FAILED"
fi
echo "========================================================================"

exit $VERIFY_EXIT_CODE
