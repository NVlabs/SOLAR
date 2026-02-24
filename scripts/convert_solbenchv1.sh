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

#
# Convert SolBench V1 benchmark files to Solar format.
#
# This script:
# 1. Checks if files have required functions (get_inputs, ReferenceModel/reference_backward, launch_reference_implementation)
# 2. Copies compliant files as-is
# 3. Modifies non-compliant files (e.g., renames nn.Module class to ReferenceModel)
# 4. Outputs CSV summary with compliance status and changes made
#
# Usage:
#   ./convert_solbenchv1.sh                    # Convert all levels
#   ./convert_solbenchv1.sh --max-files 10     # Limit to 10 files per level
#   ./convert_solbenchv1.sh --debug            # Enable debug output
#   ./convert_solbenchv1.sh -h                 # Show help
#
# Output:
#   solbenchv1-postprocess/<level>/            # Converted files
#   solbenchv1-postprocess/solbenchv1_summary.csv  # Summary CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOLAR_ROOT="${REPO_ROOT}/solar"
SOLBENCH_OUTPUT="${REPO_ROOT}/solbenchv1-postprocess"

# Args
MAX_FILES=""
DEBUG=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Convert SolBench V1 benchmark files to Solar format.

Options:
  --max-files N   Maximum files to convert per level (for testing)
  --debug         Enable debug output
  -h, --help      Show this help

Output:
  ${SOLBENCH_OUTPUT}/<level>/                # Converted files
  ${SOLBENCH_OUTPUT}/solbenchv1_summary.csv  # Compliance and changes CSV
EOF
    exit 0
}

log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
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
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check benchmark directory exists
if [[ ! -d "${REPO_ROOT}/benchmark" ]]; then
    log_error "Benchmark directory not found: ${REPO_ROOT}/benchmark"
    exit 1
fi

# Levels to convert
LEVEL_DIRS=(
    "${REPO_ROOT}/benchmark/L1"
    "${REPO_ROOT}/benchmark/L1-Quant"
    "${REPO_ROOT}/benchmark/L2"
    "${REPO_ROOT}/benchmark/L2-Quant"
)

# Create output directory
mkdir -p "${SOLBENCH_OUTPUT}"

# Track all CSV files for merging
CSV_FILES=()

log_info "Converting SolBench V1 files to: ${SOLBENCH_OUTPUT}"
log_info ""

for level_dir in "${LEVEL_DIRS[@]}"; do
    if [[ ! -d "$level_dir" ]]; then
        log_info "Skipping $level_dir (not found)"
        continue
    fi
    
    level_name=$(basename "$level_dir")
    log_info "Converting ${level_name}..."
    
    # Build command
    CONVERT_CMD="python3 -m solar.benchmark.solbenchv1.converter"
    CONVERT_CMD="$CONVERT_CMD --input-dir ${level_dir}"
    CONVERT_CMD="$CONVERT_CMD --output-dir ${SOLBENCH_OUTPUT}"
    CONVERT_CMD="$CONVERT_CMD --csv ${SOLBENCH_OUTPUT}/${level_name}_summary.csv"
    
    if [[ -n "$MAX_FILES" ]]; then
        CONVERT_CMD="$CONVERT_CMD --max-files $MAX_FILES"
    fi
    if [[ -n "$DEBUG" ]]; then
        CONVERT_CMD="$CONVERT_CMD $DEBUG"
    fi
    
    # Run from solar directory (for module imports)
    cd "${SOLAR_ROOT}"
    if $CONVERT_CMD; then
        CSV_FILES+=("${SOLBENCH_OUTPUT}/${level_name}_summary.csv")
    else
        log_error "Conversion failed for ${level_name}"
        exit 1
    fi
    
    echo ""
done

# Merge all CSV files into one
if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
    log_info "Creating merged summary CSV..."
    MERGED_CSV="${SOLBENCH_OUTPUT}/solbenchv1_summary.csv"
    
    # Write header from first file
    head -1 "${CSV_FILES[0]}" > "${MERGED_CSV}"
    
    # Append data rows from all files
    for csv_file in "${CSV_FILES[@]}"; do
        tail -n +2 "$csv_file" >> "${MERGED_CSV}"
    done
    
    log_info "Merged CSV: ${MERGED_CSV}"
    
    # Show summary statistics
    echo ""
    log_info "=== Summary Statistics ==="
    TOTAL=$(tail -n +2 "${MERGED_CSV}" | wc -l | tr -d ' ')
    COPIED=$(grep -c ',COPIED,' "${MERGED_CSV}" || echo 0)
    MODIFIED=$(grep -c ',MODIFIED,' "${MERGED_CSV}" || echo 0)
    SKIPPED=$(grep -c ',SKIPPED,' "${MERGED_CSV}" || echo 0)
    ERRORS=$(grep -c ',ERROR,' "${MERGED_CSV}" || echo 0)
    
    log_info "Total files processed: ${TOTAL}"
    log_info "  Copied (compliant):  ${COPIED}"
    log_info "  Modified:            ${MODIFIED}"
    log_info "  Skipped:             ${SKIPPED}"
    log_info "  Errors:              ${ERRORS}"
    
    # List modified files
    if [[ $MODIFIED -gt 0 ]]; then
        echo ""
        log_info "Modified files:"
        grep ',MODIFIED,' "${MERGED_CSV}" | cut -d',' -f1,12 | while IFS=',' read -r fname changes; do
            echo "  - ${fname}: ${changes}"
        done
    fi
fi

log_info ""
log_info "Done! Output in: ${SOLBENCH_OUTPUT}"
