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
# Run Solar CLI pipeline on kernelbench models.
#
# Usage:
#   ./run_kernelbench.sh                              # Process all levels
#   ./run_kernelbench.sh level1                       # Process only level1
#   ./run_kernelbench.sh level3 1 10                  # Process level3 kernels 1-10
#   ./run_kernelbench.sh level3 5                     # Process level3 kernel 5 only
#   ./run_kernelbench.sh level3/1_MLP.py              # Process single file by name
#   ./run_kernelbench.sh --help                       # Show help
#
# Inputs:  ./kernelbench/level{1,2,3,4}/*.py
# Outputs: solar/output_kernelbench/level{1,2,3,4}/<model_name>/
#
# Each model output directory contains:
#   - graph/pytorch_graph.yaml, torchview_graph.pdf
#   - einsum/einsum_graph.yaml, einsum_graph_renamed.yaml, einsum_graph.pdf
#   - analysis/analysis.yaml
#   - perf/perf_H100_PCIe.yaml
#   - timeloop/timeloop_graph.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

KERNELBENCH_DIR="${REPO_ROOT}/kernelbench"
OUTPUT_BASE="${SOLAR_ROOT}/output_kernelbench"

# Default settings
TIMEOUT=300
SKIP_EXISTING=false
DEBUG=false
PRECISION="fp16"
ARCH_CONFIG="H100_PCIe"
LEVELS=("level1" "level2" "level3" "level4")

# Kernel ID range (optional)
START_ID=""
END_ID=""

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Track failed kernels
FAILED_KERNELS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [LEVEL] [START_ID] [END_ID]
       $(basename "$0") [OPTIONS] [LEVEL/FILE.py]

Run Solar CLI pipeline on kernelbench models.

Arguments:
  LEVEL            Level to process (level1, level2, level3, level4)
  START_ID         Starting kernel ID (e.g., 1)
  END_ID           Ending kernel ID (e.g., 100). If omitted, only START_ID is processed.
  LEVEL/FILE.py    Specific file path (e.g., level3/1_MLP.py)

Options:
  --skip-existing  Skip models that already have output
  --timeout SEC    Timeout per model in seconds (default: $TIMEOUT)
  --precision PRE  Precision for analysis (fp32, fp16, bf16, int8) (default: $PRECISION)
  --arch ARCH      Architecture config name (default: $ARCH_CONFIG)
  --debug          Enable debug output
  -h, --help       Show this help message

Examples:
  $(basename "$0")                           # Process all levels
  $(basename "$0") level3                    # Process all of level3
  $(basename "$0") level3 1 10               # Process level3 kernels 1-10
  $(basename "$0") level3 5                  # Process level3 kernel 5 only
  $(basename "$0") level1 1 100 --skip-existing  # Process level1 1-100, skip existing
  $(basename "$0") level3/1_MLP.py           # Process single file by name
  $(basename "$0") --skip-existing level1    # Process all level1, skip existing
  $(basename "$0") --precision fp16 level1   # Process level1 with fp16 precision
  $(basename "$0") --arch A100 level1        # Process level1 with A100 architecture
EOF
    exit 0
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[SKIP]${NC} $*"
}

# Find kernel file by ID in a level directory
# Kernels are named like: 1_MLP.py, 10_ResNet101.py, etc.
find_kernel_by_id() {
    local level_dir="$1"
    local kernel_id="$2"
    
    # Look for files starting with the kernel ID followed by underscore
    local pattern="${kernel_id}_*.py"
    local found_file
    found_file=$(find "$level_dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | head -1)
    
    if [[ -n "$found_file" ]]; then
        echo "$found_file"
        return 0
    fi
    
    return 1
}

# Get list of kernel files for a level within ID range
get_kernels_in_range() {
    local level_dir="$1"
    local start_id="$2"
    local end_id="$3"
    
    local kernel_files=()
    for id in $(seq "$start_id" "$end_id"); do
        local kernel_file
        kernel_file=$(find_kernel_by_id "$level_dir" "$id")
        if [[ -n "$kernel_file" ]]; then
            kernel_files+=("$kernel_file")
        fi
    done
    
    printf '%s\n' "${kernel_files[@]}"
}

process_model() {
    local model_file="$1"
    local model_name
    local level
    local output_dir
    
    # Extract level and model name from path
    # e.g., kernelbench/level3/1_MLP.py -> level3, 1_MLP
    level=$(echo "$model_file" | sed -n 's|.*/\(level[0-9]\)/.*|\1|p')
    model_name=$(basename "$model_file" .py)
    
    if [[ -z "$level" ]]; then
        level="unknown"
    fi
    
    output_dir="${OUTPUT_BASE}/${level}/${model_name}"
    
    log_info "Processing: ${level}/${model_name}"
    
    # Check if already processed
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
        log_warn "Already processed, skipping: ${model_name}"
        ((SKIPPED++)) || true
        return 0
    fi
    
    # Create output directories
    mkdir -p "${output_dir}"/{graph,einsum,analysis,perf,timeloop}
    
    cd "${SOLAR_ROOT}"
    
    local step_failed=false
    
    # Step 1: Process model to PyTorch graph
    log_info "  Step 1/5: Generating PyTorch graph..."
    if timeout "$TIMEOUT" python3 -m solar.cli.process_model \
        --model-file "$model_file" \
        --output-dir "${output_dir}/graph" \
        --save-graph \
        --force-rerun \
        ${DEBUG:+--debug} 2>&1 | tee -a "${output_dir}/process.log"; then
        :
    else
        log_error "  Failed to generate PyTorch graph"
        step_failed=true
    fi
    
    # Step 2: Convert to einsum graph
    if [[ "$step_failed" == "false" ]] && [[ -f "${output_dir}/graph/pytorch_graph.yaml" ]]; then
        log_info "  Step 2/5: Converting to einsum graph..."
        if timeout "$TIMEOUT" python3 -m solar.cli.toeinsum_model \
            --graph-path "${output_dir}/graph/pytorch_graph.yaml" \
            --output-dir "${output_dir}/einsum" \
            --save-graph \
            ${DEBUG:+--debug} 2>&1 | tee -a "${output_dir}/process.log"; then
            :
        else
            log_error "  Failed to convert to einsum graph"
            step_failed=true
        fi
    fi
    
    # Step 3: Analyze einsum graph
    if [[ "$step_failed" == "false" ]] && [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
        log_info "  Step 3/5: Analyzing einsum graph (precision: ${PRECISION})..."
        if timeout "$TIMEOUT" python3 -m solar.cli.analyze_model \
            --einsum-graph-path "${output_dir}/einsum/einsum_graph_renamed.yaml" \
            --output-dir "${output_dir}/analysis" \
            --precision "${PRECISION}" \
            ${DEBUG:+--debug} 2>&1 | tee -a "${output_dir}/process.log"; then
            :
        else
            log_error "  Failed to analyze einsum graph"
            step_failed=true
        fi
    fi
    
    # Step 4: Predict performance
    if [[ "$step_failed" == "false" ]] && [[ -f "${output_dir}/analysis/analysis.yaml" ]]; then
        log_info "  Step 4/5: Predicting performance (arch: ${ARCH_CONFIG}, precision: ${PRECISION})..."
        if timeout "$TIMEOUT" python3 -m solar.cli.predict_perf_model \
            --analysis-path "${output_dir}/analysis/analysis.yaml" \
            --output-dir "${output_dir}/perf" \
            --arch-config "${ARCH_CONFIG}" \
            --precision "${PRECISION}" \
            ${DEBUG:+--debug} 2>&1 | tee -a "${output_dir}/process.log"; then
            :
        else
            log_error "  Failed to predict performance"
            step_failed=true
        fi
    fi
    
    # Step 5: Convert to Timeloop format
    if [[ "$step_failed" == "false" ]] && [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
        log_info "  Step 5/5: Converting to Timeloop format..."
        if timeout "$TIMEOUT" python3 -m solar.cli.totimeloop \
            --einsum-graph "${output_dir}/einsum/einsum_graph_renamed.yaml" \
            --output-dir "${output_dir}/timeloop" \
            ${DEBUG:+--debug} 2>&1 | tee -a "${output_dir}/process.log"; then
            :
        else
            log_error "  Failed to convert to Timeloop format"
            step_failed=true
        fi
    fi
    
    # Update counters
    if [[ "$step_failed" == "true" ]]; then
        ((FAILED++)) || true
        FAILED_KERNELS+=("${level}/${model_name}")
        return 1
    else
        log_success "Completed: ${level}/${model_name}"
        ((PASSED++)) || true
        return 0
    fi
}

process_level() {
    local level="$1"
    local start_id="${2:-}"
    local end_id="${3:-}"
    local level_dir="${KERNELBENCH_DIR}/${level}"
    
    if [[ ! -d "$level_dir" ]]; then
        log_error "Level directory not found: $level_dir"
        return 1
    fi
    
    log_info "=========================================="
    log_info "Processing ${level}"
    if [[ -n "$start_id" ]] && [[ -n "$end_id" ]]; then
        log_info "Kernel ID range: ${start_id} to ${end_id}"
    elif [[ -n "$start_id" ]]; then
        log_info "Kernel ID: ${start_id}"
    fi
    log_info "=========================================="
    
    local model_files
    
    if [[ -n "$start_id" ]] && [[ -n "$end_id" ]]; then
        # Process specific ID range
        model_files=$(get_kernels_in_range "$level_dir" "$start_id" "$end_id")
    elif [[ -n "$start_id" ]]; then
        # Process single kernel by ID
        model_files=$(find_kernel_by_id "$level_dir" "$start_id")
        if [[ -z "$model_files" ]]; then
            log_error "Kernel ID ${start_id} not found in ${level}"
            return 1
        fi
    else
        # Process all kernels in level
        model_files=$(find "$level_dir" -maxdepth 1 -name "*.py" -type f | sort)
    fi
    
    if [[ -z "$model_files" ]]; then
        log_warn "No kernel files found"
        return 0
    fi
    
    # Count total files to process
    local file_count
    file_count=$(echo "$model_files" | wc -l)
    log_info "Found ${file_count} kernel(s) to process"
    
    local current=0
    while IFS= read -r model_file; do
        [[ -z "$model_file" ]] && continue
        ((TOTAL++)) || true
        ((current++)) || true
        log_info "Progress: ${current}/${file_count}"
        process_model "$model_file" || true
        echo ""
    done <<< "$model_files"
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --arch)
            ARCH_CONFIG="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]:-}"

# Check if kernelbench directory exists
if [[ ! -d "$KERNELBENCH_DIR" ]]; then
    log_error "Kernelbench directory not found: $KERNELBENCH_DIR"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Determine what to process
if [[ $# -eq 0 ]]; then
    # Process all levels
    log_info "Processing all kernelbench levels..."
    for level in "${LEVELS[@]}"; do
        process_level "$level"
    done
elif [[ $# -ge 2 ]] && [[ "$1" =~ ^level[0-9]$ ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
    # Process level with kernel ID(s): level3 1 10 or level3 5
    LEVEL="$1"
    START_ID="$2"
    END_ID="${3:-$START_ID}"  # If no end ID, use start ID (single kernel)
    
    # Validate END_ID is a number if provided
    if [[ -n "${3:-}" ]] && ! [[ "$3" =~ ^[0-9]+$ ]]; then
        log_error "Invalid end ID: $3 (must be a number)"
        exit 1
    fi
    
    log_info "Processing ${LEVEL} kernels ${START_ID} to ${END_ID}..."
    process_level "$LEVEL" "$START_ID" "$END_ID"
elif [[ -f "${KERNELBENCH_DIR}/$1" ]]; then
    # Process single file by path
    ((TOTAL++)) || true
    process_model "${KERNELBENCH_DIR}/$1"
elif [[ -d "${KERNELBENCH_DIR}/$1" ]]; then
    # Process single level (all kernels)
    process_level "$1"
elif [[ "$1" =~ ^level[0-9]$ ]]; then
    # Process level by name (all kernels)
    process_level "$1"
else
    log_error "Invalid argument: $1"
    log_error "Expected a level (level1-level4), kernel ID range, or file path"
    log_error ""
    log_error "Examples:"
    log_error "  $0 level3              # Process all of level3"
    log_error "  $0 level3 1 10         # Process level3 kernels 1-10"
    log_error "  $0 level3 5            # Process level3 kernel 5 only"
    log_error "  $0 level3/1_MLP.py     # Process by filename"
    exit 1
fi

# Print summary
echo ""
echo "=========================================="
echo "KERNELBENCH PROCESSING SUMMARY"
echo "=========================================="
echo -e "Total:   ${TOTAL}"
echo -e "${GREEN}Passed:  ${PASSED}${NC}"
echo -e "${RED}Failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
echo "=========================================="

# Print failed kernels if any
if [[ ${#FAILED_KERNELS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed kernels:${NC}"
    for kernel in "${FAILED_KERNELS[@]}"; do
        echo -e "  - ${kernel}"
    done
    echo ""
fi

echo "Output directory: ${OUTPUT_BASE}"
echo ""

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

