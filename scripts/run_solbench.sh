#!/usr/bin/env bash
#
# Run Solar CLI pipeline on SolBench models.
#
# Usage:
#   ./run_solbench.sh                              # Process all models
#   ./run_solbench.sh --index 0000                 # Process specific model by index
#   ./run_solbench.sh --name mtp_embedding         # Process model by name pattern
#   ./run_solbench.sh --max-models 10              # Process first 10 models
#   ./run_solbench.sh --generate-only              # Only generate models, don't run solar
#   ./run_solbench.sh --phase analysis,perf        # Only run analysis and perf phases
#   ./run_solbench.sh --arch B200 --phase perf     # Rerun perf with B200 arch
#   ./run_solbench.sh --help                       # Show help
#
# Prerequisites:
#   - solbench-postprocess directory must exist (run convert_solbench.sh first)
#
# Outputs: solar/output_solbench/<model_name>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

# Set PYTHONPATH for solar module imports
export PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH:-}"

SOLBENCH_POSTPROCESS="${REPO_ROOT}/solbench-postprocess"
OUTPUT_BASE="${SOLAR_ROOT}/output_solbench"

# Default settings
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
FORCE=false

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
Usage: $(basename "$0") [OPTIONS]

Run Solar CLI pipeline on SolBench benchmark models.

Options:
  --index INDEX      Process model by index (e.g., 0000, 0001)
  --name NAME        Process model matching name pattern
  --max-models N     Maximum number of models to process
  --generate-only    Only prepare model files, don't run solar pipeline
  --phase PHASES     Comma-separated phases to run: graph,einsum,analysis,perf,timeloop
                     Default: all phases
  --skip-existing    Skip models that already have output
  --timeout SEC      Timeout per model in seconds (default: $TIMEOUT)
  --precision PRE    Precision for analysis (fp32, fp16, bf16) (default: $PRECISION)
  --arch ARCH        Architecture config name (default: $ARCH_CONFIG)
  --force            Force regeneration of model files
  --debug            Enable debug output
  -h, --help         Show this help message

Phases:
  graph     - Generate PyTorch graph from model
  einsum    - Convert to einsum graph
  analysis  - Analyze einsum graph (compute/memory stats)
  perf      - Predict performance using roofline model
  timeloop  - Convert to Timeloop format

Examples:
  $(basename "$0")                                    # Process all models with B200
  $(basename "$0") --index 0000                       # Process first model
  $(basename "$0") --name mtp_embedding               # Process model by name
  $(basename "$0") --max-models 5                     # Process first 5 models
  $(basename "$0") --phase perf --arch H100_PCIe      # Rerun perf with H100
  $(basename "$0") --generate-only                    # Only prepare model files
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
    log_info "Preparing SolBench model files from solbench-postprocess..."
    
    # Check if solbench-postprocess exists
    if [[ ! -d "$SOLBENCH_POSTPROCESS" ]]; then
        log_error "solbench-postprocess directory not found: $SOLBENCH_POSTPROCESS"
        log_error "Please run convert_solbench.sh first to convert SolBench models"
        exit 1
    fi
    
    # Find Python files to process
    local py_files=()
    
    if [[ -n "$MODEL_INDEX" ]]; then
        # Find by index pattern
        while IFS= read -r f; do
            py_files+=("$f")
        done < <(find "$SOLBENCH_POSTPROCESS" -maxdepth 1 -name "${MODEL_INDEX}*.py" 2>/dev/null | sort)
    elif [[ -n "$MODEL_NAME" ]]; then
        # Find by name pattern
        while IFS= read -r f; do
            py_files+=("$f")
        done < <(find "$SOLBENCH_POSTPROCESS" -maxdepth 1 -name "*${MODEL_NAME}*.py" 2>/dev/null | sort)
    else
        # Process all
        while IFS= read -r f; do
            py_files+=("$f")
        done < <(find "$SOLBENCH_POSTPROCESS" -maxdepth 1 -name "*.py" 2>/dev/null | sort)
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
        local model_dir="${OUTPUT_BASE}/${basename}"
        
        mkdir -p "$model_dir"
        
        # Always copy latest source file from postprocess (even if skipping)
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
source: solbench-postprocess
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
    
    # Find the source file
    local source_file
    source_file=$(find "$model_dir" -name "source_*.py" | head -1)
    
    if [[ -z "$source_file" ]] || [[ ! -f "$source_file" ]]; then
        log_error "No source file found in $model_dir"
        return 1
    fi
    
    log_info "Processing: ${model_name}"
    if [[ -n "$PHASES" ]]; then
        log_info "  Phases: ${PHASES}"
    fi
    
    # Check if already processed (only for full pipeline or perf phase)
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "${model_dir}/perf/perf_${ARCH_CONFIG}.yaml" ]]; then
        if [[ -z "$PHASES" ]] || should_run_phase "perf"; then
            log_warn "Already processed, skipping: ${model_name}"
            ((SKIPPED++)) || true
            return 0
        fi
    fi
    
    # Create output directories
    mkdir -p "${model_dir}"/{graph,einsum,analysis,perf,timeloop}
    
    cd "${SOLAR_ROOT}"
    
    local step_failed=false
    local any_step_run=false
    
    # Step 1: Process model to PyTorch graph
    if should_run_phase "graph"; then
        any_step_run=true
        log_info "  Step 1/5: Generating PyTorch graph..."
        if timeout "$TIMEOUT" python3 -m solar.cli.process_model \
            --model-file "$source_file" \
            --output-dir "${model_dir}/graph" \
            --save-graph \
            --force-rerun \
            ${DEBUG:+--debug} 2>&1 | tee -a "${model_dir}/process.log"; then
            :
        else
            log_error "  Failed to generate PyTorch graph"
            step_failed=true
        fi
    fi
    
    # Step 2: Convert to einsum graph
    if should_run_phase "einsum"; then
        if [[ -f "${model_dir}/graph/pytorch_graph.yaml" ]]; then
            any_step_run=true
            log_info "  Step 2/5: Converting to einsum graph..."
            if timeout "$TIMEOUT" python3 -m solar.cli.toeinsum_model \
                --graph-path "${model_dir}/graph/pytorch_graph.yaml" \
                --output-dir "${model_dir}/einsum" \
                --save-graph \
                ${DEBUG:+--debug} 2>&1 | tee -a "${model_dir}/process.log"; then
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
        if [[ -f "${model_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
            log_info "  Step 3/5: Analyzing einsum graph (precision: ${PRECISION})..."
            if timeout "$TIMEOUT" python3 -m solar.cli.analyze_model \
                --einsum-graph-path "${model_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${model_dir}/analysis" \
                --precision "${PRECISION}" \
                ${DEBUG:+--debug} 2>&1 | tee -a "${model_dir}/process.log"; then
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
        if [[ -f "${model_dir}/analysis/analysis.yaml" ]]; then
            any_step_run=true
            log_info "  Step 4/5: Predicting performance (arch: ${ARCH_CONFIG}, precision: ${PRECISION})..."
            if timeout "$TIMEOUT" python3 -m solar.cli.predict_perf_model \
                --analysis-path "${model_dir}/analysis/analysis.yaml" \
                --output-dir "${model_dir}/perf" \
                --arch-config "${ARCH_CONFIG}" \
                --precision "${PRECISION}" \
                ${DEBUG:+--debug} 2>&1 | tee -a "${model_dir}/process.log"; then
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
        if [[ -f "${model_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
            log_info "  Step 5/5: Converting to Timeloop format..."
            if timeout "$TIMEOUT" python3 -m solar.cli.totimeloop \
                --einsum-graph "${model_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${model_dir}/timeloop" \
                ${DEBUG:+--debug} 2>&1 | tee -a "${model_dir}/process.log"; then
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
        log_warn "No phases executed for: ${model_name}"
        ((SKIPPED++)) || true
        return 0
    elif [[ "$step_failed" == "true" ]]; then
        ((FAILED++)) || true
        FAILED_MODELS+=("${model_name}")
        return 1
    else
        log_success "Completed: ${model_name}"
        ((PASSED++)) || true
        return 0
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Check if solbench-postprocess directory exists
if [[ ! -d "$SOLBENCH_POSTPROCESS" ]]; then
    log_error "solbench-postprocess directory not found: $SOLBENCH_POSTPROCESS"
    log_error "Please run convert_solbench.sh first to convert SolBench models"
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
log_info "Processing SolBench Models"
log_info "=========================================="

model_dirs=()

if [[ -n "$MODEL_INDEX" ]]; then
    # Find by index pattern
    while IFS= read -r dir; do
        model_dirs+=("$dir")
    done < <(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "${MODEL_INDEX}*" 2>/dev/null | sort)
elif [[ -n "$MODEL_NAME" ]]; then
    # Find by name pattern
    while IFS= read -r dir; do
        model_dirs+=("$dir")
    done < <(find "$OUTPUT_BASE" -maxdepth 1 -type d -name "*${MODEL_NAME}*" 2>/dev/null | sort)
else
    # Process all
    while IFS= read -r dir; do
        if [[ -f "${dir}/metadata.yaml" ]]; then
            model_dirs+=("$dir")
        fi
    done < <(find "$OUTPUT_BASE" -maxdepth 1 -type d ! -path "$OUTPUT_BASE" 2>/dev/null | sort)
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

# Print summary
echo ""
echo "=========================================="
echo "SOLBENCH BENCHMARK SUMMARY"
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
