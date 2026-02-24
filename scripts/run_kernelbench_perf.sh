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
# Re-run specific pipeline phases on existing kernelbench results.
# This is useful when you want to re-run only certain steps without
# re-running the entire pipeline.
#
# Usage:
#   ./run_kernelbench_perf.sh                              # Process all levels (perf only)
#   ./run_kernelbench_perf.sh level1                       # Process only level1 (perf only)
#   ./run_kernelbench_perf.sh level3 1 10                  # Process level3 kernels 1-10
#   ./run_kernelbench_perf.sh --phase perf level1          # Run perf phase only
#   ./run_kernelbench_perf.sh --phase einsum,analysis level1  # Run einsum and analysis
#   ./run_kernelbench_perf.sh --phase all level1           # Run all phases
#   ./run_kernelbench_perf.sh --precision fp16 level1      # Use fp16 precision
#   ./run_kernelbench_perf.sh --arch A100 level1           # Use A100 architecture
#   ./run_kernelbench_perf.sh --help                       # Show help
#
# Phases:
#   graph    - Generate PyTorch graph from model file (requires kernelbench/*.py)
#   einsum   - Convert PyTorch graph to einsum graph (requires graph/pytorch_graph.yaml)
#   analysis - Analyze einsum graph (requires einsum/einsum_graph_renamed.yaml)
#   perf     - Predict performance (requires analysis/analysis.yaml)
#   timeloop - Convert to Timeloop format (requires einsum/einsum_graph_renamed.yaml)
#   all      - Run all phases
#
# Prerequisites: Required input files must exist from previous runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

OUTPUT_BASE="${SOLAR_ROOT}/output_kernelbench"
KERNELBENCH_DIR="${REPO_ROOT}/kernelbench"

# Default settings
TIMEOUT=60
DEBUG=false
PRECISION="fp32"
ARCH_CONFIG="H100_PCIe"
LEVELS=("level1" "level2" "level3" "level4")

# Phases to run (default: perf only for backward compatibility)
PHASES=("perf")

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

Re-run specific pipeline phases on existing kernelbench results.

Arguments:
  LEVEL            Level to process (level1, level2, level3, level4)
  START_ID         Starting kernel ID (e.g., 1)
  END_ID           Ending kernel ID (e.g., 100). If omitted, only START_ID is processed.

Options:
  --phase PHASES   Comma-separated list of phases to run (default: perf)
                   Available phases: graph, einsum, analysis, perf, timeloop, all
  --precision PRE  Precision for analysis (fp32, fp16, bf16, int8) (default: $PRECISION)
  --arch ARCH      Architecture config name (default: $ARCH_CONFIG)
  --timeout SEC    Timeout per step in seconds (default: $TIMEOUT)
  --debug          Enable debug output
  -h, --help       Show this help message

Phases:
  graph     Generate PyTorch graph from model file
  einsum    Convert PyTorch graph to einsum graph
  analysis  Analyze einsum graph
  perf      Predict performance
  timeloop  Convert to Timeloop format
  all       Run all phases (graph,einsum,analysis,perf,timeloop)

Examples:
  $(basename "$0")                              # Process all levels, perf only
  $(basename "$0") level1                       # Process all of level1, perf only
  $(basename "$0") level3 1 10                  # Process level3 kernels 1-10
  $(basename "$0") --phase perf level1          # Run only perf phase
  $(basename "$0") --phase einsum,analysis level1   # Run einsum and analysis phases
  $(basename "$0") --phase all level1           # Run all phases
  $(basename "$0") --precision fp16 level1      # Process level1 with fp16 precision
  $(basename "$0") --arch A100 level1           # Process level1 with A100 architecture
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

# Check if a phase should be run
should_run_phase() {
    local phase="$1"
    for p in "${PHASES[@]}"; do
        if [[ "$p" == "$phase" ]] || [[ "$p" == "all" ]]; then
            return 0
        fi
    done
    return 1
}

# Find the model source file for a kernel
find_model_file() {
    local output_dir="$1"
    local level model_name
    
    level=$(echo "$output_dir" | sed -n 's|.*/\(level[0-9]\)/.*|\1|p')
    model_name=$(basename "$output_dir")
    
    # Try to find the model file in kernelbench
    local model_file="${KERNELBENCH_DIR}/${level}/${model_name}.py"
    if [[ -f "$model_file" ]]; then
        echo "$model_file"
        return 0
    fi
    
    # Also check in output_dir/graph for copied source
    local source_file="${output_dir}/graph/source_${model_name}.py"
    if [[ -f "$source_file" ]]; then
        echo "$source_file"
        return 0
    fi
    
    return 1
}

process_model() {
    local output_dir="$1"
    local model_name
    local level
    
    # Extract level and model name from path
    level=$(echo "$output_dir" | sed -n 's|.*/\(level[0-9]\)/.*|\1|p')
    model_name=$(basename "$output_dir")
    
    if [[ -z "$level" ]]; then
        level="unknown"
    fi
    
    log_info "Processing: ${level}/${model_name}"
    
    cd "${SOLAR_ROOT}"
    
    local step_failed=false
    local any_step_run=false
    
    # Create output directories if needed
    mkdir -p "${output_dir}"/{graph,einsum,analysis,perf,timeloop} 2>/dev/null || true
    
    # Phase 1: Graph generation
    if should_run_phase "graph"; then
        local model_file
        model_file=$(find_model_file "$output_dir") || true
        
        if [[ -n "$model_file" ]]; then
            log_info "  [graph] Generating PyTorch graph..."
            any_step_run=true
            if python3 -m solar.cli.process_model \
                --model-file "$model_file" \
                --output-dir "${output_dir}/graph" \
                --save-graph \
                --force-rerun \
                2>&1 | tee -a "${output_dir}/phase_rerun.log"; then
                :
            else
                log_error "  [graph] Failed to generate PyTorch graph"
                step_failed=true
            fi
        else
            log_warn "  [graph] No model file found, skipping"
        fi
    fi
    
    # Phase 2: Einsum conversion
    if should_run_phase "einsum" && [[ "$step_failed" == "false" ]]; then
        if [[ -f "${output_dir}/graph/pytorch_graph.yaml" ]]; then
            log_info "  [einsum] Converting to einsum graph..."
            any_step_run=true
            if python3 -m solar.cli.toeinsum_model \
                --graph-path "${output_dir}/graph/pytorch_graph.yaml" \
                --output-dir "${output_dir}/einsum" \
                --save-graph \
                2>&1 | tee -a "${output_dir}/phase_rerun.log"; then
                :
            else
                log_error "  [einsum] Failed to convert to einsum graph"
                step_failed=true
            fi
        else
            log_warn "  [einsum] No pytorch_graph.yaml found, skipping"
        fi
    fi
    
    # Phase 3: Analysis
    if should_run_phase "analysis" && [[ "$step_failed" == "false" ]]; then
        if [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            log_info "  [analysis] Analyzing einsum graph (precision: ${PRECISION})..."
            any_step_run=true
            if [[ "$DEBUG" == "true" ]]; then
                echo "DEBUG: Running from $(pwd)"
                echo "DEBUG: output_dir=${output_dir}"
                echo "DEBUG: Command: python3 -m solar.cli.analyze_model --einsum-graph-path ${output_dir}/einsum/einsum_graph_renamed.yaml --output-dir ${output_dir}/analysis --precision ${PRECISION}"
            fi
            if python3 -m solar.cli.analyze_model \
                --einsum-graph-path "${output_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${output_dir}/analysis" \
                --precision "${PRECISION}" \
                2>&1 | tee -a "${output_dir}/phase_rerun.log"; then
                :
            else
                log_error "  [analysis] Failed to analyze einsum graph"
                step_failed=true
            fi
        else
            log_warn "  [analysis] No einsum_graph_renamed.yaml found, skipping"
        fi
    fi
    
    # Phase 4: Performance prediction
    if should_run_phase "perf" && [[ "$step_failed" == "false" ]]; then
        if [[ -f "${output_dir}/analysis/analysis.yaml" ]]; then
            log_info "  [perf] Predicting performance (arch: ${ARCH_CONFIG}, precision: ${PRECISION})..."
            any_step_run=true
            if python3 -m solar.cli.predict_perf_model \
                --analysis-path "${output_dir}/analysis/analysis.yaml" \
                --output-dir "${output_dir}/perf" \
                --arch-config "${ARCH_CONFIG}" \
                --precision "${PRECISION}" \
                2>&1 | tee -a "${output_dir}/phase_rerun.log"; then
                :
            else
                log_error "  [perf] Failed to predict performance"
                step_failed=true
            fi
        else
            log_warn "  [perf] No analysis.yaml found, skipping"
        fi
    fi
    
    # Phase 5: Timeloop conversion
    if should_run_phase "timeloop" && [[ "$step_failed" == "false" ]]; then
        if [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            log_info "  [timeloop] Converting to Timeloop format..."
            any_step_run=true
            if python3 -m solar.cli.totimeloop \
                --einsum-graph "${output_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${output_dir}/timeloop" \
                2>&1 | tee -a "${output_dir}/phase_rerun.log"; then
                :
            else
                log_error "  [timeloop] Failed to convert to Timeloop format"
                step_failed=true
            fi
        else
            log_warn "  [timeloop] No einsum_graph_renamed.yaml found, skipping"
        fi
    fi
    
    # Update counters
    if [[ "$any_step_run" == "false" ]]; then
        log_warn "No phases could run (missing prerequisites): ${model_name}"
        ((SKIPPED++)) || true
        return 0
    elif [[ "$step_failed" == "true" ]]; then
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
    local level_dir="${OUTPUT_BASE}/${level}"
    
    if [[ ! -d "$level_dir" ]]; then
        log_error "Level directory not found: $level_dir"
        return 1
    fi
    
    log_info "=========================================="
    log_info "Processing ${level}"
    log_info "Phases: ${PHASES[*]}"
    log_info "Precision: ${PRECISION}, Arch: ${ARCH_CONFIG}"
    if [[ -n "$start_id" ]] && [[ -n "$end_id" ]]; then
        log_info "Kernel ID range: ${start_id} to ${end_id}"
    elif [[ -n "$start_id" ]]; then
        log_info "Kernel ID: ${start_id}"
    fi
    log_info "=========================================="
    
    local kernel_dirs=()
    
    if [[ -n "$start_id" ]] && [[ -n "$end_id" ]]; then
        # Process specific ID range
        for id in $(seq "$start_id" "$end_id"); do
            local pattern="${id}_*"
            local found_dir
            found_dir=$(find "$level_dir" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | head -1)
            if [[ -n "$found_dir" ]]; then
                kernel_dirs+=("$found_dir")
            fi
        done
    elif [[ -n "$start_id" ]]; then
        # Process single kernel by ID
        local pattern="${start_id}_*"
        local found_dir
        found_dir=$(find "$level_dir" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | head -1)
        if [[ -z "$found_dir" ]]; then
            log_error "Kernel ID ${start_id} not found in ${level}"
            return 1
        fi
        kernel_dirs+=("$found_dir")
    else
        # Process all kernels in level
        while IFS= read -r dir; do
            kernel_dirs+=("$dir")
        done < <(find "$level_dir" -maxdepth 1 -type d ! -path "$level_dir" | sort)
    fi
    
    if [[ ${#kernel_dirs[@]} -eq 0 ]]; then
        log_warn "No kernel directories found"
        return 0
    fi
    
    log_info "Found ${#kernel_dirs[@]} kernel(s) to process"
    
    local current=0
    for kernel_dir in "${kernel_dirs[@]}"; do
        ((TOTAL++)) || true
        ((current++)) || true
        log_info "Progress: ${current}/${#kernel_dirs[@]}"
        process_model "$kernel_dir" || true
        echo ""
    done
}

# Parse phases from comma-separated string
parse_phases() {
    local phase_str="$1"
    PHASES=()
    
    IFS=',' read -ra phase_array <<< "$phase_str"
    for phase in "${phase_array[@]}"; do
        phase=$(echo "$phase" | tr -d ' ')  # Remove spaces
        case "$phase" in
            graph|einsum|analysis|perf|timeloop|all)
                PHASES+=("$phase")
                ;;
            *)
                echo "Unknown phase: $phase"
                echo "Available phases: graph, einsum, analysis, perf, timeloop, all"
                exit 1
                ;;
        esac
    done
    
    if [[ ${#PHASES[@]} -eq 0 ]]; then
        PHASES=("perf")  # Default
    fi
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            parse_phases "$2"
            shift 2
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

# Check if output directory exists
if [[ ! -d "$OUTPUT_BASE" ]]; then
    log_error "Output directory not found: $OUTPUT_BASE"
    log_error "Run run_kernelbench.sh first to generate initial results."
    exit 1
fi

# Determine what to process
if [[ $# -eq 0 ]]; then
    # Process all levels
    log_info "Processing all kernelbench levels..."
    for level in "${LEVELS[@]}"; do
        if [[ -d "${OUTPUT_BASE}/${level}" ]]; then
            process_level "$level"
        fi
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
elif [[ "$1" =~ ^level[0-9]$ ]]; then
    # Process level by name (all kernels)
    process_level "$1"
else
    log_error "Invalid argument: $1"
    log_error "Expected a level (level1-level4) or kernel ID range"
    exit 1
fi

# Print summary
echo ""
echo "=========================================="
echo "PHASE RE-RUN SUMMARY"
echo "=========================================="
echo -e "Phases: ${PHASES[*]}"
echo -e "Precision: ${PRECISION}"
echo -e "Architecture: ${ARCH_CONFIG}"
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
