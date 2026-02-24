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
# Run Solar CLI pipeline on SolBench V1 models.
#
# Usage:
#   ./run_solbenchv1.sh                              # Process all models
#   ./run_solbenchv1.sh --level L1 --index 0130      # Process specific model
#   ./run_solbenchv1.sh --name flux_context          # Process model by name pattern
#   ./run_solbenchv1.sh --max-models 10              # Process first 10 models
#   ./run_solbenchv1.sh --generate-only              # Only generate models, don't run solar
#   ./run_solbenchv1.sh --phase analysis,perf        # Only run analysis and perf phases
#   ./run_solbenchv1.sh --arch H100_PCIe --phase perf # Rerun perf with H100 arch
#   ./run_solbenchv1.sh --help                       # Show help
#
# Prerequisites:
#   - solbenchv1-postprocess directory must exist (run convert_solbenchv1.sh first)
#
# Outputs:
#   solar/output_solbenchv1/<level>/<model_name>/    # Model outputs
#   solar/output_solbenchv1/results_summary.csv      # Results CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

# Set PYTHONPATH for solar module imports
export PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH:-}"

SOLBENCHV1_POSTPROCESS="${REPO_ROOT}/solbenchv1-postprocess"
OUTPUT_BASE="${SOLAR_ROOT}/output_solbenchv1"

# Default settings - B200 is the default architecture
TIMEOUT=300
SKIP_EXISTING=false
DEBUG=false
PRECISION="bf16"
ARCH_CONFIG="B200"
GENERATE_ONLY=false
MAX_MODELS=""
PHASES=""  # Empty = all phases
MODEL_INDEX=""
MODEL_NAME=""
MODEL_LEVEL=""
FORCE=false

# Valid phases
VALID_PHASES=("graph" "einsum" "analysis" "perf" "timeloop")

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Track results for CSV
declare -a RESULTS_CSV

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run Solar CLI pipeline on SolBench V1 benchmark models.

Default architecture: B200 (NVIDIA Blackwell)

Options:
  --level LEVEL        Process only specific level (L1, L2, L1-Quant, L2-Quant)
  --index INDEX        Process model by index (e.g., 0130)
  --name NAME          Process model matching name pattern (e.g., flux_context)
  --max-models N       Maximum number of models to process
  --generate-only      Only prepare model files, don't run solar pipeline
  --phase PHASES       Comma-separated phases to run: graph,einsum,analysis,perf,timeloop
                       Default: all phases
  --skip-existing      Skip models that already have output
  --timeout SEC        Timeout per model in seconds (default: $TIMEOUT)
  --precision PRE      Precision for analysis (fp32, fp16, bf16) (default: $PRECISION)
  --arch ARCH          Architecture config name (default: $ARCH_CONFIG)
  --force              Force regeneration of model files
  --debug              Enable debug output
  -h, --help           Show this help message

Phases:
  graph     - Generate PyTorch graph from model
  einsum    - Convert to einsum graph
  analysis  - Analyze einsum graph (compute/memory stats)
  perf      - Predict performance using roofline model
  timeloop  - Convert to Timeloop format

Examples:
  $(basename "$0")                                    # Process all models with B200
  $(basename "$0") --level L1 --index 0130           # Process specific model
  $(basename "$0") --name flux_context               # Process models matching pattern
  $(basename "$0") --max-models 5                    # Process first 5 models
  $(basename "$0") --phase perf --arch H100_PCIe    # Rerun perf with H100
  $(basename "$0") --generate-only                   # Only prepare model files

Output:
  Results CSV: ${OUTPUT_BASE}/results_summary.csv
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

# Check if a phase should run
should_run_phase() {
    local phase="$1"
    if [[ -z "$PHASES" ]]; then
        return 0  # Run all phases
    fi
    [[ ",$PHASES," == *",$phase,"* ]]
}

# Add result to CSV array
add_result() {
    local model_name="$1"
    local level="$2"
    local status="$3"
    local failed_phase="$4"
    local error_msg="$5"
    local graph_ok="$6"
    local einsum_ok="$7"
    local analysis_ok="$8"
    local perf_ok="$9"
    local timeloop_ok="${10}"
    
    # Escape commas in error message
    error_msg="${error_msg//,/;}"
    error_msg="${error_msg//$'\n'/ }"
    
    RESULTS_CSV+=("${model_name},${level},${status},${failed_phase},${error_msg},${graph_ok},${einsum_ok},${analysis_ok},${perf_ok},${timeloop_ok}")
}

# Write results CSV
write_results_csv() {
    local csv_file="${OUTPUT_BASE}/results_summary.csv"
    
    # Write header
    echo "model_name,level,status,failed_phase,error_message,graph,einsum,analysis,perf,timeloop" > "$csv_file"
    
    # Write results
    for result in "${RESULTS_CSV[@]}"; do
        echo "$result" >> "$csv_file"
    done
    
    log_info "Results CSV written to: ${csv_file}"
}

generate_models() {
    log_info "Preparing SolBench V1 model files from solbenchv1-postprocess..."
    
    # Check if solbenchv1-postprocess exists
    if [[ ! -d "$SOLBENCHV1_POSTPROCESS" ]]; then
        log_error "solbenchv1-postprocess directory not found: $SOLBENCHV1_POSTPROCESS"
        log_error "Please run convert_solbenchv1.sh first to convert SolBench V1 models"
        exit 1
    fi
    
    # Find Python files to process
    local py_files=()
    local search_dir="$SOLBENCHV1_POSTPROCESS"
    
    # Filter by level if specified
    if [[ -n "$MODEL_LEVEL" ]]; then
        search_dir="${SOLBENCHV1_POSTPROCESS}/${MODEL_LEVEL}"
    fi
    
    if [[ -n "$MODEL_INDEX" ]]; then
        # Find by index pattern
        while IFS= read -r f; do
            py_files+=("$f")
        done < <(find "$search_dir" -name "${MODEL_INDEX}*.py" 2>/dev/null | sort)
    elif [[ -n "$MODEL_NAME" ]]; then
        # Find by name pattern
        while IFS= read -r f; do
            py_files+=("$f")
        done < <(find "$search_dir" -name "*${MODEL_NAME}*.py" 2>/dev/null | sort)
    else
        # Process all
        while IFS= read -r f; do
            py_files+=("$f")
        done < <(find "$search_dir" -name "*.py" 2>/dev/null | sort)
    fi
    
    # Apply max-models limit
    if [[ -n "$MAX_MODELS" ]] && [[ ${#py_files[@]} -gt $MAX_MODELS ]]; then
        py_files=("${py_files[@]:0:$MAX_MODELS}")
    fi
    
    log_info "Found ${#py_files[@]} model(s) to prepare"
    
    # Create output directories and copy source files
    for py_file in "${py_files[@]}"; do
        local basename
        basename=$(basename "$py_file" .py)
        
        # Determine level from path
        local rel_path="${py_file#$SOLBENCHV1_POSTPROCESS/}"
        local level_dir
        level_dir=$(dirname "$rel_path")
        
        local model_dir="${OUTPUT_BASE}/${level_dir}/${basename}"
        
        mkdir -p "$model_dir"
        
        # Always copy latest source file from postprocess
        cp "$py_file" "${model_dir}/source_${basename}.py"
        
        # Skip metadata/further prep if already exists and not forcing
        if [[ -d "$model_dir" ]] && [[ -f "${model_dir}/metadata.yaml" ]] && [[ "$FORCE" != "true" ]]; then
            if [[ "$DEBUG" == "true" ]]; then
                log_info "  Updated source, skipping prep: $basename"
            fi
            continue
        fi
        
        # Create metadata.yaml
        cat > "${model_dir}/metadata.yaml" << EOF
name: ${basename}
level: ${level_dir}
source: solbenchv1-postprocess
original_file: ${py_file}
generated: $(date -Iseconds)
EOF
        
        if [[ "$DEBUG" == "true" ]]; then
            log_info "  Prepared: $basename"
        fi
    done
}

process_model() {
    local model_dir="$1"
    local model_name
    model_name=$(basename "$model_dir")
    
    # Get level from path
    local rel_path="${model_dir#$OUTPUT_BASE/}"
    local level_name
    level_name=$(dirname "$rel_path")
    
    # Find the source file
    local source_file
    source_file=$(find "$model_dir" -name "source_*.py" | head -1)
    
    if [[ -z "$source_file" ]] || [[ ! -f "$source_file" ]]; then
        log_error "No source file found in $model_dir"
        add_result "$model_name" "$level_name" "FAIL" "setup" "No source file found" "N/A" "N/A" "N/A" "N/A" "N/A"
        return 1
    fi
    
    log_info "Processing: ${model_name}"
    if [[ -n "$PHASES" ]]; then
        log_info "  Phases: ${PHASES}"
    fi
    
    # Check if already processed
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "${model_dir}/perf/perf_${ARCH_CONFIG}.yaml" ]]; then
        if [[ -z "$PHASES" ]] || should_run_phase "perf"; then
            log_warn "Already processed, skipping: ${model_name}"
            ((SKIPPED++)) || true
            add_result "$model_name" "$level_name" "SKIP" "" "Already processed" "SKIP" "SKIP" "SKIP" "SKIP" "SKIP"
            return 0
        fi
    fi
    
    # Create output directories
    mkdir -p "${model_dir}"/{graph,einsum,analysis,perf,timeloop}
    
    cd "${SOLAR_ROOT}"
    
    local step_failed=false
    local failed_phase=""
    local error_msg=""
    local any_step_run=false
    
    # Track phase status
    local graph_status="SKIP"
    local einsum_status="SKIP"
    local analysis_status="SKIP"
    local perf_status="SKIP"
    local timeloop_status="SKIP"
    
    # Step 1: Process model to PyTorch graph
    if should_run_phase "graph"; then
        any_step_run=true
        log_info "  Step 1/5: Generating PyTorch graph..."
        local graph_output
        if graph_output=$(timeout "$TIMEOUT" python3 -m solar.cli.process_model \
            --model-file "$source_file" \
            --output-dir "${model_dir}/graph" \
            --save-graph \
            --force-rerun \
            ${DEBUG:+--debug} 2>&1); then
            echo "$graph_output" >> "${model_dir}/process.log"
            graph_status="PASS"
        else
            echo "$graph_output" >> "${model_dir}/process.log"
            log_error "  Failed to generate PyTorch graph"
            graph_status="FAIL"
            step_failed=true
            failed_phase="graph"
            error_msg="${graph_output: -200}"  # Last 200 chars
        fi
    fi
    
    # Step 2: Convert to einsum graph
    if should_run_phase "einsum"; then
        if [[ -f "${model_dir}/graph/pytorch_graph.yaml" ]]; then
            any_step_run=true
            log_info "  Step 2/5: Converting to einsum graph..."
            local einsum_output
            if einsum_output=$(timeout "$TIMEOUT" python3 -m solar.cli.toeinsum_model \
                --graph-path "${model_dir}/graph/pytorch_graph.yaml" \
                --output-dir "${model_dir}/einsum" \
                --save-graph \
                ${DEBUG:+--debug} 2>&1); then
                echo "$einsum_output" >> "${model_dir}/process.log"
                einsum_status="PASS"
            else
                echo "$einsum_output" >> "${model_dir}/process.log"
                log_error "  Failed to convert to einsum graph"
                einsum_status="FAIL"
                if [[ "$step_failed" != "true" ]]; then
                    step_failed=true
                    failed_phase="einsum"
                    error_msg="${einsum_output: -200}"
                fi
            fi
        else
            log_warn "  Skipping einsum: pytorch_graph.yaml not found"
            einsum_status="SKIP"
        fi
    fi
    
    # Step 3: Analyze einsum graph
    if should_run_phase "analysis"; then
        if [[ -f "${model_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
            log_info "  Step 3/5: Analyzing einsum graph (precision: ${PRECISION})..."
            local analysis_output
            if analysis_output=$(timeout "$TIMEOUT" python3 -m solar.cli.analyze_model \
                --einsum-graph-path "${model_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${model_dir}/analysis" \
                --precision "${PRECISION}" \
                ${DEBUG:+--debug} 2>&1); then
                echo "$analysis_output" >> "${model_dir}/process.log"
                analysis_status="PASS"
            else
                echo "$analysis_output" >> "${model_dir}/process.log"
                log_error "  Failed to analyze einsum graph"
                analysis_status="FAIL"
                if [[ "$step_failed" != "true" ]]; then
                    step_failed=true
                    failed_phase="analysis"
                    error_msg="${analysis_output: -200}"
                fi
            fi
        else
            log_warn "  Skipping analysis: einsum_graph_renamed.yaml not found"
            analysis_status="SKIP"
        fi
    fi
    
    # Step 4: Predict performance
    if should_run_phase "perf"; then
        if [[ -f "${model_dir}/analysis/analysis.yaml" ]]; then
            any_step_run=true
            log_info "  Step 4/5: Predicting performance (arch: ${ARCH_CONFIG}, precision: ${PRECISION})..."
            local perf_output
            if perf_output=$(timeout "$TIMEOUT" python3 -m solar.cli.predict_perf_model \
                --analysis-path "${model_dir}/analysis/analysis.yaml" \
                --output-dir "${model_dir}/perf" \
                --arch-config "${ARCH_CONFIG}" \
                --precision "${PRECISION}" \
                ${DEBUG:+--debug} 2>&1); then
                echo "$perf_output" >> "${model_dir}/process.log"
                perf_status="PASS"
            else
                echo "$perf_output" >> "${model_dir}/process.log"
                log_error "  Failed to predict performance"
                perf_status="FAIL"
                if [[ "$step_failed" != "true" ]]; then
                    step_failed=true
                    failed_phase="perf"
                    error_msg="${perf_output: -200}"
                fi
            fi
        else
            log_warn "  Skipping perf: analysis.yaml not found"
            perf_status="SKIP"
        fi
    fi
    
    # Step 5: Convert to Timeloop format
    if should_run_phase "timeloop"; then
        if [[ -f "${model_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
            log_info "  Step 5/5: Converting to Timeloop format..."
            local timeloop_output
            if timeloop_output=$(timeout "$TIMEOUT" python3 -m solar.cli.totimeloop \
                --einsum-graph "${model_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${model_dir}/timeloop" \
                ${DEBUG:+--debug} 2>&1); then
                echo "$timeloop_output" >> "${model_dir}/process.log"
                timeloop_status="PASS"
            else
                echo "$timeloop_output" >> "${model_dir}/process.log"
                log_error "  Failed to convert to Timeloop format"
                timeloop_status="FAIL"
                if [[ "$step_failed" != "true" ]]; then
                    step_failed=true
                    failed_phase="timeloop"
                    error_msg="${timeloop_output: -200}"
                fi
            fi
        else
            log_warn "  Skipping timeloop: einsum_graph_renamed.yaml not found"
            timeloop_status="SKIP"
        fi
    fi
    
    # Update counters and add result
    if [[ "$any_step_run" == "false" ]]; then
        log_warn "No phases executed for: ${model_name}"
        ((SKIPPED++)) || true
        add_result "$model_name" "$level_name" "SKIP" "" "No phases executed" "$graph_status" "$einsum_status" "$analysis_status" "$perf_status" "$timeloop_status"
        return 0
    elif [[ "$step_failed" == "true" ]]; then
        ((FAILED++)) || true
        add_result "$model_name" "$level_name" "FAIL" "$failed_phase" "$error_msg" "$graph_status" "$einsum_status" "$analysis_status" "$perf_status" "$timeloop_status"
        return 1
    else
        log_success "Completed: ${model_name}"
        ((PASSED++)) || true
        add_result "$model_name" "$level_name" "PASS" "" "" "$graph_status" "$einsum_status" "$analysis_status" "$perf_status" "$timeloop_status"
        return 0
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            MODEL_LEVEL="$2"
            shift 2
            ;;
        --index)
            MODEL_INDEX="$2"
            shift 2
            ;;
        --name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --phase)
            PHASES="$2"
            shift 2
            ;;
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
        --max-models)
            MAX_MODELS="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --debug)
            DEBUG=true
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

# Validate phases if specified
if [[ -n "$PHASES" ]]; then
    IFS=',' read -ra PHASE_ARRAY <<< "$PHASES"
    for phase in "${PHASE_ARRAY[@]}"; do
        valid=false
        for vp in "${VALID_PHASES[@]}"; do
            if [[ "$phase" == "$vp" ]]; then
                valid=true
                break
            fi
        done
        if [[ "$valid" == "false" ]]; then
            log_error "Invalid phase: $phase"
            log_error "Valid phases: ${VALID_PHASES[*]}"
            exit 1
        fi
    done
    log_info "Running phases: ${PHASES}"
fi

# Check if solbenchv1-postprocess directory exists
if [[ ! -d "$SOLBENCHV1_POSTPROCESS" ]]; then
    log_error "solbenchv1-postprocess directory not found: $SOLBENCHV1_POSTPROCESS"
    log_error "Please run convert_solbenchv1.sh first to convert SolBench V1 models"
    exit 1
fi

# Generate/prepare models
generate_models

if [[ "$GENERATE_ONLY" == "true" ]]; then
    log_info "Model preparation complete. Run without --generate-only to process."
    exit 0
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Find and process model directories
log_info "=========================================="
log_info "Processing SolBench V1 Models"
log_info "=========================================="

model_dirs=()

# Search for model directories
search_base="$OUTPUT_BASE"
if [[ -n "$MODEL_LEVEL" ]]; then
    search_base="${OUTPUT_BASE}/${MODEL_LEVEL}"
fi

if [[ -n "$MODEL_INDEX" ]]; then
    while IFS= read -r dir; do
        model_dirs+=("$dir")
    done < <(find "$search_base" -type d -name "${MODEL_INDEX}*" 2>/dev/null | sort)
elif [[ -n "$MODEL_NAME" ]]; then
    while IFS= read -r dir; do
        model_dirs+=("$dir")
    done < <(find "$search_base" -type d -name "*${MODEL_NAME}*" 2>/dev/null | sort)
else
    while IFS= read -r dir; do
        if [[ -f "${dir}/metadata.yaml" ]]; then
            model_dirs+=("$dir")
        fi
    done < <(find "$search_base" -type d 2>/dev/null | sort)
fi

# Apply max-models limit
if [[ -n "$MAX_MODELS" ]] && [[ ${#model_dirs[@]} -gt $MAX_MODELS ]]; then
    log_info "Limiting to first ${MAX_MODELS} model(s) (found ${#model_dirs[@]})"
    model_dirs=("${model_dirs[@]:0:$MAX_MODELS}")
fi

if [[ ${#model_dirs[@]} -eq 0 ]]; then
    log_warn "No models found to process"
    exit 0
fi

log_info "Found ${#model_dirs[@]} model(s) to process"

# Process each model
current=0
for model_dir in "${model_dirs[@]}"; do
    ((TOTAL++)) || true
    ((current++)) || true
    log_info "Progress: ${current}/${#model_dirs[@]}"
    process_model "$model_dir" || true
    echo ""
done

# Write results CSV
write_results_csv

# Print summary
echo ""
echo "=========================================="
echo "SOLBENCH V1 BENCHMARK SUMMARY"
echo "=========================================="
echo -e "Total:   ${TOTAL}"
echo -e "${GREEN}Passed:  ${PASSED}${NC}"
echo -e "${RED}Failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
if [[ -n "$PHASES" ]]; then
    echo -e "Phases:  ${PHASES}"
fi
echo -e "Arch:    ${ARCH_CONFIG}"
echo -e "Precision: ${PRECISION}"
echo "=========================================="

# Print failed models from CSV
if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed models:${NC}"
    # Extract failed models from results
    for result in "${RESULTS_CSV[@]}"; do
        if [[ "$result" == *",FAIL,"* ]]; then
            local_name=$(echo "$result" | cut -d',' -f1)
            local_phase=$(echo "$result" | cut -d',' -f4)
            echo -e "  - ${local_name} (failed at: ${local_phase})"
        fi
    done
    echo ""
fi

echo "Output directory: ${OUTPUT_BASE}"
echo "Results CSV: ${OUTPUT_BASE}/results_summary.csv"
echo ""

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
