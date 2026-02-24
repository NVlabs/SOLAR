#!/usr/bin/env bash
#
# Run Solar CLI pipeline on FlashInfer benchmark models.
#
# Usage:
#   ./run_flashinferbench.sh                              # Process all op types
#   ./run_flashinferbench.sh gemm                         # Process only gemm
#   ./run_flashinferbench.sh gemm gemm_n128_k2048         # Process specific definition
#   ./run_flashinferbench.sh gemm gemm_n128_k2048 0 5     # Process workloads 0-5
#   ./run_flashinferbench.sh --generate-only              # Only generate models, don't run solar
#   ./run_flashinferbench.sh --phase analysis,perf        # Only run analysis and perf phases
#   ./run_flashinferbench.sh --arch B200 --phase perf     # Rerun perf with B200 arch
#   ./run_flashinferbench.sh --help                       # Show help
#
# Prerequisites:
#   - flashinfer-trace directory must exist (download from HuggingFace)
#   - Run with --generate-only first to create model files
#
# Outputs: solar/output_flashinferbench/<op_type>/<name>/<row_id>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

# Set PYTHONPATH for solar module imports
export PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH:-}"

TRACE_DIR="${REPO_ROOT}/flashinfer-trace"
POSTPROCESS_DIR="${REPO_ROOT}/flashinfer-trace-postprocess"
OUTPUT_BASE="${SOLAR_ROOT}/output_flashinferbench"

# Default settings
TIMEOUT=300
SKIP_EXISTING=false
DEBUG=false
PRECISION="fp16"
ARCH_CONFIG="H100_PCIe"
GENERATE_ONLY=false
MAX_WORKLOADS=""
PHASES=""  # Empty = all phases

# Valid phases
VALID_PHASES=("graph" "einsum" "analysis" "perf" "timeloop")

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Track failed models
FAILED_MODELS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [OP_TYPE] [NAME] [START_ROW] [END_ROW]

Run Solar CLI pipeline on FlashInfer benchmark models.

Arguments:
  OP_TYPE          Operation type (e.g., gemm, rmsnorm)
  NAME             Definition name (e.g., gemm_n128_k2048)
  START_ROW        Starting workload row ID (e.g., 0)
  END_ROW          Ending workload row ID (e.g., 10)

Options:
  --generate-only  Only generate model files, don't run solar pipeline
  --phase PHASES   Comma-separated phases to run: graph,einsum,analysis,perf,timeloop
                   Default: all phases
  --skip-existing  Skip models that already have output
  --timeout SEC    Timeout per model in seconds (default: $TIMEOUT)
  --precision PRE  Precision for analysis (fp32, fp16, bf16) (default: $PRECISION)
  --arch ARCH      Architecture config name (default: $ARCH_CONFIG)
  --max-workloads N  Maximum workloads per definition (default: all)
  --debug          Enable debug output
  -h, --help       Show this help message

Phases:
  graph     - Generate PyTorch graph from model
  einsum    - Convert to einsum graph
  analysis  - Analyze einsum graph (compute/memory stats)
  perf      - Predict performance using roofline model
  timeloop  - Convert to Timeloop format

Examples:
  $(basename "$0") --generate-only                    # Generate all model files
  $(basename "$0") gemm                               # Process all gemm definitions
  $(basename "$0") gemm gemm_n128_k2048               # Process specific definition
  $(basename "$0") gemm gemm_n128_k2048 0 5           # Process workloads 0-5
  $(basename "$0") --precision fp16 gemm              # Use fp16 precision
  $(basename "$0") --phase analysis,perf --arch B200  # Rerun analysis+perf with B200
  $(basename "$0") --phase perf --arch B200 gemm     # Rerun perf for gemm with B200
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

generate_models() {
    local op_type="${1:-}"
    local name="${2:-}"
    
    log_info "Generating FlashInfer model files..."
    
    local gen_args=("--trace-dir" "$TRACE_DIR" "--output-dir" "$POSTPROCESS_DIR")
    
    if [[ -n "$op_type" ]]; then
        gen_args+=("--op-type" "$op_type")
    fi
    
    if [[ -n "$name" ]]; then
        gen_args+=("--name" "$name")
    fi
    
    if [[ -n "$MAX_WORKLOADS" ]]; then
        gen_args+=("--max-workloads" "$MAX_WORKLOADS")
    fi
    
    cd "${SOLAR_ROOT}"
    PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH:-}" python3 -m solar.benchmark.flashinfer.cli "${gen_args[@]}"
}

process_model() {
    local model_file="$1"
    local model_name
    local output_dir
    
    # Extract path components
    # e.g., flashinfer-trace-postprocess/gemm/gemm_n128_k2048/0/gemm_n128_k2048.py
    local rel_path="${model_file#$POSTPROCESS_DIR/}"
    local dir_path
    dir_path=$(dirname "$rel_path")
    model_name=$(basename "$model_file" .py)
    
    output_dir="${OUTPUT_BASE}/${dir_path}"
    
    log_info "Processing: ${dir_path}/${model_name}"
    if [[ -n "$PHASES" ]]; then
        log_info "  Phases: ${PHASES}"
    fi
    
    # Check if already processed (only for full pipeline or perf phase)
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "${output_dir}/perf/perf_${ARCH_CONFIG}.yaml" ]]; then
        if [[ -z "$PHASES" ]] || should_run_phase "perf"; then
            log_warn "Already processed, skipping: ${model_name}"
            ((SKIPPED++)) || true
            return 0
        fi
    fi
    
    # Create output directories
    mkdir -p "${output_dir}"/{graph,einsum,analysis,perf,timeloop}
    
    # Copy uuid.yaml from postprocess dir if it exists
    local model_dir
    model_dir=$(dirname "$model_file")
    if [[ -f "${model_dir}/uuid.yaml" ]]; then
        cp "${model_dir}/uuid.yaml" "${output_dir}/"
    fi
    
    cd "${SOLAR_ROOT}"
    
    local step_failed=false
    local any_step_run=false
    
    # Step 1: Process model to PyTorch graph
    if should_run_phase "graph"; then
        any_step_run=true
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
    fi
    
    # Step 2: Convert to einsum graph
    if should_run_phase "einsum"; then
        if [[ -f "${output_dir}/graph/pytorch_graph.yaml" ]]; then
            any_step_run=true
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
        else
            log_warn "  Skipping einsum: pytorch_graph.yaml not found"
        fi
    fi
    
    # Step 3: Analyze einsum graph
    if should_run_phase "analysis"; then
        if [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
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
        else
            log_warn "  Skipping analysis: einsum_graph_renamed.yaml not found"
        fi
    fi
    
    # Step 4: Predict performance
    if should_run_phase "perf"; then
        if [[ -f "${output_dir}/analysis/analysis.yaml" ]]; then
            any_step_run=true
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
        else
            log_warn "  Skipping perf: analysis.yaml not found"
        fi
    fi
    
    # Step 5: Convert to Timeloop format
    if should_run_phase "timeloop"; then
        if [[ -f "${output_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
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
        else
            log_warn "  Skipping timeloop: einsum_graph_renamed.yaml not found"
        fi
    fi
    
    # Update counters
    if [[ "$any_step_run" == "false" ]]; then
        log_warn "No phases executed for: ${dir_path}/${model_name}"
        ((SKIPPED++)) || true
        return 0
    elif [[ "$step_failed" == "true" ]]; then
        ((FAILED++)) || true
        FAILED_MODELS+=("${dir_path}")
        return 1
    else
        log_success "Completed: ${dir_path}/${model_name}"
        ((PASSED++)) || true
        return 0
    fi
}

process_definition() {
    local op_type="$1"
    local name="$2"
    local start_row="${3:-}"
    local end_row="${4:-}"
    
    local def_dir="${POSTPROCESS_DIR}/${op_type}/${name}"
    
    if [[ ! -d "$def_dir" ]]; then
        log_error "Definition directory not found: $def_dir"
        log_error "Run with --generate-only first to create model files"
        return 1
    fi
    
    log_info "=========================================="
    log_info "Processing ${op_type}/${name}"
    if [[ -n "$start_row" ]] && [[ -n "$end_row" ]]; then
        log_info "Workload range: ${start_row} to ${end_row}"
    fi
    log_info "=========================================="
    
    local model_files=()
    
    if [[ -n "$start_row" ]] && [[ -n "$end_row" ]]; then
        # Process specific row range
        for row in $(seq "$start_row" "$end_row"); do
            local model_file="${def_dir}/${row}/${name}.py"
            if [[ -f "$model_file" ]]; then
                model_files+=("$model_file")
            fi
        done
    else
        # Process all workload instances (numbered directories)
        while IFS= read -r dir; do
            local model_file="${dir}/${name}.py"
            if [[ -f "$model_file" ]]; then
                model_files+=("$model_file")
            fi
        done < <(find "$def_dir" -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V)
    fi
    
    if [[ ${#model_files[@]} -eq 0 ]]; then
        log_warn "No workload instances found"
        return 0
    fi
    
    # Apply max-workloads limit if set
    if [[ -n "$MAX_WORKLOADS" ]] && [[ ${#model_files[@]} -gt $MAX_WORKLOADS ]]; then
        log_info "Limiting to first ${MAX_WORKLOADS} workload(s) (found ${#model_files[@]})"
        model_files=("${model_files[@]:0:$MAX_WORKLOADS}")
    fi
    
    log_info "Found ${#model_files[@]} workload(s) to process"
    
    local current=0
    for model_file in "${model_files[@]}"; do
        ((TOTAL++)) || true
        ((current++)) || true
        log_info "Progress: ${current}/${#model_files[@]}"
        process_model "$model_file" || true
        echo ""
    done
}

process_op_type() {
    local op_type="$1"
    local op_dir="${POSTPROCESS_DIR}/${op_type}"
    
    if [[ ! -d "$op_dir" ]]; then
        log_error "Op type directory not found: $op_dir"
        return 1
    fi
    
    log_info "Processing all definitions in ${op_type}..."
    
    while IFS= read -r def_dir; do
        local name
        name=$(basename "$def_dir")
        process_definition "$op_type" "$name"
    done < <(find "$op_dir" -maxdepth 1 -type d ! -path "$op_dir" | sort)
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --max-workloads)
            MAX_WORKLOADS="$2"
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

# Check if trace directory exists
if [[ ! -d "$TRACE_DIR" ]]; then
    log_error "FlashInfer trace directory not found: $TRACE_DIR"
    log_error "Download from: https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace"
    exit 1
fi

# Determine what to process
OP_TYPE="${1:-}"
NAME="${2:-}"
START_ROW="${3:-}"
END_ROW="${4:-}"

# Generate models if needed or if --generate-only
if [[ "$GENERATE_ONLY" == "true" ]] || [[ ! -d "$POSTPROCESS_DIR" ]]; then
    generate_models "$OP_TYPE" "$NAME"
    
    if [[ "$GENERATE_ONLY" == "true" ]]; then
        log_info "Model generation complete. Run without --generate-only to process."
        exit 0
    fi
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Process based on arguments
if [[ -z "$OP_TYPE" ]]; then
    # Process all op types
    log_info "Processing all FlashInfer benchmarks..."
    for op_dir in "$POSTPROCESS_DIR"/*/; do
        if [[ -d "$op_dir" ]]; then
            op_type=$(basename "$op_dir")
            process_op_type "$op_type"
        fi
    done
elif [[ -z "$NAME" ]]; then
    # Process all definitions in op type
    process_op_type "$OP_TYPE"
elif [[ -z "$START_ROW" ]]; then
    # Process all workloads for definition
    process_definition "$OP_TYPE" "$NAME"
else
    # Process specific row range
    if [[ -z "$END_ROW" ]]; then
        END_ROW="$START_ROW"
    fi
    process_definition "$OP_TYPE" "$NAME" "$START_ROW" "$END_ROW"
fi

# Print summary
echo ""
echo "=========================================="
echo "FLASHINFER BENCHMARK SUMMARY"
echo "=========================================="
echo -e "Total:   ${TOTAL}"
echo -e "${GREEN}Passed:  ${PASSED}${NC}"
echo -e "${RED}Failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
if [[ -n "$PHASES" ]]; then
    echo -e "Phases:  ${PHASES}"
fi
echo -e "Arch:    ${ARCH_CONFIG}"
echo "=========================================="

# Print failed models if any
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed models:${NC}"
    for model in "${FAILED_MODELS[@]}"; do
        echo -e "  - ${model}"
    done
    echo ""
fi

echo "Output directory: ${OUTPUT_BASE}"
echo ""

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
