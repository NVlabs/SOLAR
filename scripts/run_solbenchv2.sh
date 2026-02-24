#!/usr/bin/env bash
#
# Run Solar CLI pipeline on SolBench v2 models from sol-bench/data/benchmark.
#
# Usage:
#   ./run_solbenchv2.sh                     # Process all levels (L1, L2, Quant)
#   ./run_solbenchv2.sh --level L1          # Process only L1 kernels
#   ./run_solbenchv2.sh --level L2          # Process only L2 kernels
#   ./run_solbenchv2.sh --level Quant       # Process only Quant kernels
#   ./run_solbenchv2.sh --name mamba        # Process kernels matching name pattern
#   ./run_solbenchv2.sh --max-models 10     # Process first 10 models
#   ./run_solbenchv2.sh --help              # Show help
#
# Prerequisites:
#   - sol-bench/data/benchmark directory must exist
#
# Default architecture: B200
# Output: solar/output_solbenchv2/<level>/<model_name>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

# Set PYTHONPATH for solar module imports
export PYTHONPATH="${SOLAR_ROOT}:${PYTHONPATH:-}"

BENCHMARK_DIR="${REPO_ROOT}/sol-bench/data/benchmark"
OUTPUT_BASE="${SOLAR_ROOT}/output_solbenchv2"

# Default settings
TIMEOUT=300
SKIP_EXISTING=false
DEBUG=false
PRECISION="bf16"
ARCH_CONFIG="B200"  # Default to B200
MAX_MODELS=""
LEVEL=""  # Empty = all levels (L1, L2, Quant)
MODEL_NAME=""
FORCE=false
PHASES=""  # Empty = all phases
POSTPROCESS_ONLY=false
KERNEL_INDEX=""

# Valid phases
VALID_PHASES=("graph" "einsum" "analysis" "perf" "timeloop")
VALID_LEVELS=("L1" "L2" "Quant")

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Track failed models
FAILED_MODELS=()

# Track all results for CSV
declare -a RESULTS_LEVEL=()
declare -a RESULTS_NAME=()
declare -a RESULTS_STATUS=()
declare -a RESULTS_ERROR=()
declare -a RESULTS_GRAPH=()
declare -a RESULTS_EINSUM=()
declare -a RESULTS_ANALYSIS=()
declare -a RESULTS_PERF=()
declare -a RESULTS_TIMELOOP=()

# Log file
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run Solar CLI pipeline on SolBench v2 benchmark kernels from sol-bench/data/benchmark.

Options:
  --level LEVEL      Process specific level: L1, L2, Quant (default: all)
  --name NAME        Process kernels matching name pattern
  --index N          Process kernel by index (0-based, e.g., --index 8)
  --max-models N     Maximum number of models to process
  --phase PHASES     Comma-separated phases to run: graph,einsum,analysis,perf,timeloop
                     Default: all phases
  --postprocess-only Run postprocessing only (no Solar analysis)
  --skip-existing    Skip models that already have output
  --timeout SEC      Timeout per model in seconds (default: $TIMEOUT)
  --precision PRE    Precision for analysis (fp32, fp16, bf16) (default: $PRECISION)
  --arch ARCH        Architecture config name (default: $ARCH_CONFIG)
  --force            Force regeneration even if output exists
  --debug            Enable debug output
  -h, --help         Show this help message

Levels:
  L1    - Level 1 kernels (simpler operations)
  L2    - Level 2 kernels (more complex fused operations)
  Quant - Quantized kernels (FP8, NVFP4, etc.)

Phases:
  graph     - Generate PyTorch graph from model
  einsum    - Convert to einsum graph
  analysis  - Analyze einsum graph (compute/memory stats)
  perf      - Predict performance using roofline model
  timeloop  - Convert to Timeloop format

Examples:
  $(basename "$0")                                    # Process all with B200
  $(basename "$0") --level L1                         # Process only L1 kernels
  $(basename "$0") --level L2 --arch H100_PCIe        # Process L2 with H100
  $(basename "$0") --level Quant --max-models 5       # Process first 5 Quant kernels
  $(basename "$0") --name mamba                       # Process kernels with 'mamba' in name
  $(basename "$0") --index 8                          # Process kernel at index 8
  $(basename "$0") --postprocess-only                 # Run postprocessing only
  $(basename "$0") --level Quant --postprocess-only   # Postprocess all Quant kernels
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

# Log to both console and file
log_all() {
    local msg="$1"
    echo -e "$msg"
    if [[ -n "$MASTER_LOG" ]]; then
        # Strip color codes for log file
        echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$MASTER_LOG"
    fi
}

# Add result to tracking arrays
add_result() {
    local level="$1"
    local name="$2"
    local status="$3"
    local error="$4"
    local graph="$5"
    local einsum="$6"
    local analysis="$7"
    local perf="$8"
    local timeloop="$9"
    
    RESULTS_LEVEL+=("$level")
    RESULTS_NAME+=("$name")
    RESULTS_STATUS+=("$status")
    RESULTS_ERROR+=("$error")
    RESULTS_GRAPH+=("$graph")
    RESULTS_EINSUM+=("$einsum")
    RESULTS_ANALYSIS+=("$analysis")
    RESULTS_PERF+=("$perf")
    RESULTS_TIMELOOP+=("$timeloop")
}

# Generate CSV summary
generate_csv() {
    local csv_file="${OUTPUT_BASE}/solbenchv2_results_${RUN_TIMESTAMP}.csv"
    
    # Write header
    echo "level,kernel_name,status,error,graph,einsum,analysis,perf,timeloop" > "$csv_file"
    
    # Write results
    local count=${#RESULTS_NAME[@]}
    local i=0
    while [[ $i -lt $count ]]; do
        # Escape commas and quotes in error messages
        local error="${RESULTS_ERROR[$i]//\"/\"\"}"
        error="${error//,/;}"
        echo "${RESULTS_LEVEL[$i]},${RESULTS_NAME[$i]},${RESULTS_STATUS[$i]},\"${error}\",${RESULTS_GRAPH[$i]},${RESULTS_EINSUM[$i]},${RESULTS_ANALYSIS[$i]},${RESULTS_PERF[$i]},${RESULTS_TIMELOOP[$i]}" >> "$csv_file"
        i=$((i + 1))
    done
    
    echo "$csv_file"
}

# Check if a phase should run
should_run_phase() {
    local phase="$1"
    if [[ -z "$PHASES" ]]; then
        return 0  # Run all phases
    fi
    [[ ",$PHASES," == *",$phase,"* ]]
}

# Check if kernel is Solar-compatible
check_kernel_compatible() {
    local input_file="$1"
    
    python3.10 -c "
import sys
sys.path.insert(0, '${SOLAR_ROOT}')
from solar.benchmark.solbenchv2.parser import parse_solbenchv2_file

kernel = parse_solbenchv2_file('${input_file}')
if kernel.is_valid:
    print('OK')
else:
    missing = []
    if not kernel.has_get_inputs:
        missing.append('get_inputs()')
    if not kernel.has_reference_model and not kernel.has_reference_backward:
        missing.append('ReferenceModel or reference_backward')
    if not kernel.has_launch_reference_implementation:
        missing.append('launch_reference_implementation()')
    msg = ', '.join(missing)
    print('MISSING: ' + msg)
    exit(1)
"
}

# Postprocess kernel file to remove device specs and replace Triton _fused_fma
postprocess_kernel() {
    local input_file="$1"
    local output_dir="$2"
    
    python3.10 "${SOLAR_ROOT}/solar/benchmark/solbenchv2/postprocess.py" \
        --file "$input_file" \
        --output-dir "$output_dir" 2>&1
}

# Run postprocessing on all collected files
run_postprocessing_only() {
    log_all "${BLUE}[INFO]${NC} =========================================="
    log_all "${BLUE}[INFO]${NC} Postprocessing SolBench v2 Kernels"
    log_all "${BLUE}[INFO]${NC} =========================================="
    
    local count=0
    local results=()
    
    for entry in "${model_files[@]}"; do
        IFS='|' read -r model_file level <<< "$entry"
        ((count++)) || true
        
        local model_name
        model_name=$(basename "$model_file" .py)
        local model_dir="${OUTPUT_BASE}/${level}/${model_name}"
        
        # Create graph directory for postprocessed files
        mkdir -p "${model_dir}/graph"
        
        log_all "${BLUE}[INFO]${NC} [${count}/${#model_files[@]}] Postprocessing: ${level}/${model_name}"
        
        if postprocess_kernel "$model_file" "${model_dir}/graph" >> "$MASTER_LOG" 2>&1; then
            log_all "${GREEN}[PASS]${NC}   Postprocessed: ${model_name}"
        else
            log_all "${RED}[FAIL]${NC}   Failed: ${model_name}"
        fi
    done
    
    log_all "${GREEN}[PASS]${NC} Postprocessing complete. Files in: ${OUTPUT_BASE}/<level>/<kernel>/graph/"
    
    # Generate summary by collecting all postprocessed files
    local summary_file="${OUTPUT_BASE}/postprocess_summary_${RUN_TIMESTAMP}.csv"
    
    # Use a temporary directory structure for summary generation
    local temp_postprocess="${OUTPUT_BASE}/.temp_postprocess"
    rm -rf "$temp_postprocess"
    mkdir -p "$temp_postprocess"
    
    # Copy postprocessed files to temp structure for summary
    for entry in "${model_files[@]}"; do
        IFS='|' read -r model_file level <<< "$entry"
        local model_name
        model_name=$(basename "$model_file" .py)
        local model_dir="${OUTPUT_BASE}/${level}/${model_name}"
        local level_temp="${temp_postprocess}/${level}"
        mkdir -p "$level_temp"
        
        if [[ -f "${model_dir}/graph/src_${model_name}.py" ]]; then
            cp "${model_dir}/graph/src_${model_name}.py" "${level_temp}/src_${model_name}.py"
        fi
        if [[ -f "${model_dir}/graph/orig_${model_name}.py" ]]; then
            cp "${model_dir}/graph/orig_${model_name}.py" "${level_temp}/orig_${model_name}.py"
        fi
    done
    
    python3.10 "${SOLAR_ROOT}/solar/benchmark/solbenchv2/postprocess.py" \
        --input-dir "$BENCHMARK_DIR" \
        --output-dir "$temp_postprocess" \
        ${LEVEL:+--level "$LEVEL"} \
        --summary "$summary_file" 2>&1 | tee -a "$MASTER_LOG"
    
    rm -rf "$temp_postprocess"
    
    log_all "${BLUE}[INFO]${NC} Summary: ${summary_file}"
    
    exit 0
}

process_model() {
    local model_file="$1"
    local level="$2"
    local model_name
    model_name=$(basename "$model_file" .py)
    
    local model_dir="${OUTPUT_BASE}/${level}/${model_name}"
    local model_log="${model_dir}/process.log"
    
    # Track phase results
    local phase_graph="N/A"
    local phase_einsum="N/A"
    local phase_analysis="N/A"
    local phase_perf="N/A"
    local phase_timeloop="N/A"
    local error_msg=""
    
    log_all "${BLUE}[INFO]${NC} Processing: ${level}/${model_name}"
    
    # Check if already processed
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "${model_dir}/perf/perf_${ARCH_CONFIG}.yaml" ]]; then
        log_all "${YELLOW}[SKIP]${NC} Already processed, skipping: ${model_name}"
        add_result "$level" "$model_name" "SKIPPED" "Already processed" "-" "-" "-" "-" "-"
        ((SKIPPED++)) || true
        return 0
    fi
    
    # Create output directories
    mkdir -p "${model_dir}"/{graph,einsum,analysis,perf,timeloop}
    
    # Initialize model log
    echo "=== Processing ${level}/${model_name} ===" > "$model_log"
    echo "Timestamp: $(date -Iseconds)" >> "$model_log"
    echo "Source: ${model_file}" >> "$model_log"
    echo "" >> "$model_log"
    
    # Check if kernel is Solar-compatible
    log_all "${BLUE}[INFO]${NC}   Checking kernel compatibility..."
    local check_result
    check_result=$(check_kernel_compatible "$model_file" 2>&1)
    
    if [[ "$check_result" != "OK" ]]; then
        error_msg="Not Solar-compatible: $check_result"
        log_all "${RED}[FAIL]${NC}   Kernel not Solar-compatible: $check_result"
        echo "ERROR: $error_msg" >> "$model_log"
        add_result "$level" "$model_name" "FAILED" "$error_msg" "SKIP" "SKIP" "SKIP" "SKIP" "SKIP"
        ((FAILED++)) || true
        FAILED_MODELS+=("${level}/${model_name}")
        return 1
    fi
    
    # Postprocess the source file (remove device specs, replace Triton _fused_fma)
    # Place postprocessed files in graph/ directory
    log_all "${BLUE}[INFO]${NC}   Postprocessing source file..."
    postprocess_kernel "$model_file" "${model_dir}/graph" >> "$model_log" 2>&1
    
    # Find the processed source file in graph/ directory
    local source_file="${model_dir}/graph/src_${model_name}.py"
    if [[ ! -f "$source_file" ]]; then
        log_all "${RED}[FAIL]${NC}   Postprocessing failed: output file not found"
        error_msg="Postprocessing failed"
        add_result "$level" "$model_name" "FAILED" "$error_msg" "SKIP" "SKIP" "SKIP" "SKIP" "SKIP"
        ((FAILED++)) || true
        FAILED_MODELS+=("${level}/${model_name}")
        return 1
    fi
    
    # Check if original was saved (meaning modifications were made)
    if [[ -f "${model_dir}/graph/orig_${model_name}.py" ]]; then
        log_all "${BLUE}[INFO]${NC}   Source file postprocessed (original saved to graph/orig_${model_name}.py)"
    else
        log_all "${BLUE}[INFO]${NC}   Source file copied (no modifications needed)"
    fi
    
    # Check if conversion metadata was created
    if [[ -f "${model_dir}/graph/metadata.yaml" ]]; then
        log_all "${BLUE}[INFO]${NC}   Dtype conversion metadata saved to graph/metadata.yaml"
    fi
    
    # Create metadata.yaml
    cat > "${model_dir}/metadata.yaml" << EOF
name: ${model_name}
level: ${level}
source: sol-bench/data/benchmark/${level}
original_file: ${model_file}
generated: $(date -Iseconds)
arch: ${ARCH_CONFIG}
precision: ${PRECISION}
EOF
    
    cd "${SOLAR_ROOT}"
    
    local step_failed=false
    local any_step_run=false
    
    # Helper: log last N error lines from process.log to master log
    _log_error_tail() {
        local stage="$1"
        if [[ -f "$model_log" ]]; then
            local err_lines
            err_lines=$(tail -30 "$model_log" | grep -i -E "error|exception|traceback|failed|not implemented|RuntimeError|NotImplementedError|TypeError|AttributeError|NoneType|missing.*argument|SyntaxError" | tail -5)
            if [[ -n "$err_lines" ]]; then
                log_all "${RED}[ERR]${NC}   Error details for ${stage}:"
                while IFS= read -r line; do
                    log_all "${RED}[ERR]${NC}     $line"
                done <<< "$err_lines"
            fi
        fi
    }
    
    # Step 1: Process model to PyTorch graph
    # Use the postprocessed source file from graph/ directory
    if should_run_phase "graph"; then
        any_step_run=true
        log_all "${BLUE}[INFO]${NC}   Step 1/5: Generating PyTorch graph..."
        echo "=== Step 1: Graph ===" >> "$model_log"
        if timeout "$TIMEOUT" env PYTHONUNBUFFERED=1 python3.10 -m solar.cli.process_model \
            --model-file "$source_file" \
            --output-dir "${model_dir}/graph" \
            --save-graph \
            --force-rerun \
            ${DEBUG:+--debug} 2>&1 | tee -a "$model_log"; then
            phase_graph="OK"
        else
            log_all "${RED}[FAIL]${NC}   Failed to generate PyTorch graph"
            _log_error_tail "graph"
            phase_graph="FAILED"
            error_msg="Graph generation failed"
            step_failed=true
        fi
    fi
    
    # Step 2: Convert to einsum graph
    if should_run_phase "einsum" && [[ "$step_failed" != "true" ]]; then
        if [[ -f "${model_dir}/graph/pytorch_graph.yaml" ]]; then
            any_step_run=true
            log_all "${BLUE}[INFO]${NC}   Step 2/5: Converting to einsum graph..."
            echo "=== Step 2: Einsum ===" >> "$model_log"
            if timeout "$TIMEOUT" python3.10 -m solar.cli.toeinsum_model \
                --graph-path "${model_dir}/graph/pytorch_graph.yaml" \
                --output-dir "${model_dir}/einsum" \
                --save-graph \
                ${DEBUG:+--debug} 2>&1 | tee -a "$model_log"; then
                phase_einsum="OK"
            else
                log_all "${RED}[FAIL]${NC}   Failed to convert to einsum graph"
                _log_error_tail "einsum"
                phase_einsum="FAILED"
                error_msg="Einsum conversion failed"
                step_failed=true
            fi
        else
            log_all "${YELLOW}[SKIP]${NC}   Skipping einsum: pytorch_graph.yaml not found"
            phase_einsum="SKIP"
        fi
    fi
    
    # Step 3: Analyze einsum graph
    if should_run_phase "analysis" && [[ "$step_failed" != "true" ]]; then
        if [[ -f "${model_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
            log_all "${BLUE}[INFO]${NC}   Step 3/5: Analyzing einsum graph (precision: ${PRECISION})..."
            echo "=== Step 3: Analysis ===" >> "$model_log"
            if timeout "$TIMEOUT" python3.10 -m solar.cli.analyze_model \
                --einsum-graph-path "${model_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${model_dir}/analysis" \
                --precision "${PRECISION}" \
                ${DEBUG:+--debug} 2>&1 | tee -a "$model_log"; then
                phase_analysis="OK"
            else
                log_all "${RED}[FAIL]${NC}   Failed to analyze einsum graph"
                _log_error_tail "analysis"
                phase_analysis="FAILED"
                error_msg="Analysis failed"
                step_failed=true
            fi
        else
            log_all "${YELLOW}[SKIP]${NC}   Skipping analysis: einsum_graph_renamed.yaml not found"
            phase_analysis="SKIP"
        fi
    fi
    
    # Step 4: Predict performance
    if should_run_phase "perf" && [[ "$step_failed" != "true" ]]; then
        if [[ -f "${model_dir}/analysis/analysis.yaml" ]]; then
            any_step_run=true
            log_all "${BLUE}[INFO]${NC}   Step 4/5: Predicting performance (arch: ${ARCH_CONFIG}, precision: ${PRECISION})..."
            echo "=== Step 4: Perf ===" >> "$model_log"
            if timeout "$TIMEOUT" python3.10 -m solar.cli.predict_perf_model \
                --analysis-path "${model_dir}/analysis/analysis.yaml" \
                --output-dir "${model_dir}/perf" \
                --arch-config "${ARCH_CONFIG}" \
                --precision "${PRECISION}" \
                ${DEBUG:+--debug} 2>&1 | tee -a "$model_log"; then
                phase_perf="OK"
            else
                log_all "${RED}[FAIL]${NC}   Failed to predict performance"
                _log_error_tail "perf"
                phase_perf="FAILED"
                error_msg="Performance prediction failed"
                step_failed=true
            fi
        else
            log_all "${YELLOW}[SKIP]${NC}   Skipping perf: analysis.yaml not found"
            phase_perf="SKIP"
        fi
    fi
    
    # Step 5: Convert to Timeloop format
    if should_run_phase "timeloop" && [[ "$step_failed" != "true" ]]; then
        if [[ -f "${model_dir}/einsum/einsum_graph_renamed.yaml" ]]; then
            any_step_run=true
            log_all "${BLUE}[INFO]${NC}   Step 5/5: Converting to Timeloop format..."
            echo "=== Step 5: Timeloop ===" >> "$model_log"
            if timeout "$TIMEOUT" python3.10 -m solar.cli.totimeloop \
                --einsum-graph "${model_dir}/einsum/einsum_graph_renamed.yaml" \
                --output-dir "${model_dir}/timeloop" \
                ${DEBUG:+--debug} 2>&1 | tee -a "$model_log"; then
                phase_timeloop="OK"
            else
                log_all "${RED}[FAIL]${NC}   Failed to convert to Timeloop format"
                _log_error_tail "timeloop"
                phase_timeloop="FAILED"
                error_msg="Timeloop conversion failed"
                step_failed=true
            fi
        else
            log_all "${YELLOW}[SKIP]${NC}   Skipping timeloop: einsum_graph_renamed.yaml not found"
            phase_timeloop="SKIP"
        fi
    fi
    
    # Update counters and record result
    if [[ "$any_step_run" == "false" ]]; then
        log_all "${YELLOW}[SKIP]${NC} No phases executed for: ${model_name}"
        add_result "$level" "$model_name" "SKIPPED" "No phases executed" "$phase_graph" "$phase_einsum" "$phase_analysis" "$phase_perf" "$phase_timeloop"
        ((SKIPPED++)) || true
        return 0
    elif [[ "$step_failed" == "true" ]]; then
        add_result "$level" "$model_name" "FAILED" "$error_msg" "$phase_graph" "$phase_einsum" "$phase_analysis" "$phase_perf" "$phase_timeloop"
        ((FAILED++)) || true
        FAILED_MODELS+=("${level}/${model_name}")
        return 1
    else
        log_all "${GREEN}[PASS]${NC} Completed: ${level}/${model_name}"
        add_result "$level" "$model_name" "PASSED" "" "$phase_graph" "$phase_einsum" "$phase_analysis" "$phase_perf" "$phase_timeloop"
        ((PASSED++)) || true
        return 0
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --name)
            MODEL_NAME="$2"
            shift 2
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
        --postprocess-only)
            POSTPROCESS_ONLY=true
            shift
            ;;
        --index)
            KERNEL_INDEX="$2"
            shift 2
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

# Validate level if specified
if [[ -n "$LEVEL" ]]; then
    valid=false
    for vl in "${VALID_LEVELS[@]}"; do
        if [[ "$LEVEL" == "$vl" ]]; then
            valid=true
            break
        fi
    done
    if [[ "$valid" == "false" ]]; then
        log_error "Invalid level: $LEVEL"
        log_error "Valid levels: ${VALID_LEVELS[*]}"
        exit 1
    fi
fi

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
    echo -e "${BLUE}[INFO]${NC} Running phases: ${PHASES}"
fi

# Check if benchmark directory exists
if [[ ! -d "$BENCHMARK_DIR" ]]; then
    log_error "Benchmark directory not found: $BENCHMARK_DIR"
    log_error "Please ensure sol-bench/data/benchmark exists"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Initialize master log file
MASTER_LOG="${OUTPUT_BASE}/run_${RUN_TIMESTAMP}.log"
echo "=== SolBench v2 Run Log ===" > "$MASTER_LOG"
echo "Timestamp: $(date -Iseconds)" >> "$MASTER_LOG"
echo "Architecture: ${ARCH_CONFIG}" >> "$MASTER_LOG"
echo "Precision: ${PRECISION}" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Collect files to process
log_all "${BLUE}[INFO]${NC} =========================================="
log_all "${BLUE}[INFO]${NC} Processing SolBench v2 Models"
log_all "${BLUE}[INFO]${NC} =========================================="
log_all "${BLUE}[INFO]${NC} Architecture: ${ARCH_CONFIG}"
log_all "${BLUE}[INFO]${NC} Precision: ${PRECISION}"

model_files=()
levels_to_process=()

if [[ -n "$LEVEL" ]]; then
    levels_to_process=("$LEVEL")
else
    levels_to_process=("L1" "L2" "Quant")
fi

for lvl in "${levels_to_process[@]}"; do
    level_dir="${BENCHMARK_DIR}/${lvl}"
    if [[ ! -d "$level_dir" ]]; then
        log_all "${YELLOW}[SKIP]${NC} Level directory not found: $level_dir"
        continue
    fi
    
    # Find Python files
    while IFS= read -r f; do
        if [[ -n "$MODEL_NAME" ]]; then
            # Filter by name pattern
            if [[ "$(basename "$f")" == *"$MODEL_NAME"* ]]; then
                model_files+=("$f|$lvl")
            fi
        else
            model_files+=("$f|$lvl")
        fi
    done < <(find "$level_dir" -maxdepth 1 -name "*.py" 2>/dev/null | sort)
done

# Apply index selection if specified
if [[ -n "$KERNEL_INDEX" ]]; then
    if [[ ! "$KERNEL_INDEX" =~ ^[0-9]+$ ]]; then
        log_error "Invalid index: $KERNEL_INDEX (must be a number)"
        exit 1
    fi
    
    if [[ $KERNEL_INDEX -ge ${#model_files[@]} ]] || [[ $KERNEL_INDEX -lt 0 ]]; then
        log_error "Index $KERNEL_INDEX out of range (0-$((${#model_files[@]} - 1)))"
        exit 1
    fi
    
    log_all "${BLUE}[INFO]${NC} Selecting kernel at index ${KERNEL_INDEX}"
    model_files=("${model_files[$KERNEL_INDEX]}")
fi

# Apply max-models limit
if [[ -n "$MAX_MODELS" ]] && [[ ${#model_files[@]} -gt $MAX_MODELS ]]; then
    log_all "${BLUE}[INFO]${NC} Limiting to first ${MAX_MODELS} model(s) (found ${#model_files[@]})"
    model_files=("${model_files[@]:0:$MAX_MODELS}")
fi

if [[ ${#model_files[@]} -eq 0 ]]; then
    log_all "${YELLOW}[SKIP]${NC} No models found to process"
    exit 0
fi

log_all "${BLUE}[INFO]${NC} Found ${#model_files[@]} model(s) to process"
echo ""

# Run postprocessing only if requested
if [[ "$POSTPROCESS_ONLY" == "true" ]]; then
    run_postprocessing_only
fi

# Process each model
current=0
for entry in "${model_files[@]}"; do
    IFS='|' read -r model_file level <<< "$entry"
    ((TOTAL++)) || true
    ((current++)) || true
    log_all "${BLUE}[INFO]${NC} Progress: ${current}/${#model_files[@]}"
    process_model "$model_file" "$level" || true
    echo ""
done

# Generate CSV summary
CSV_FILE=$(generate_csv)

# Print summary
echo ""
echo "=========================================="
echo "SOLBENCH V2 BENCHMARK SUMMARY"
echo "=========================================="
echo -e "Total:   ${TOTAL}"
echo -e "${GREEN}Passed:  ${PASSED}${NC}"
echo -e "${RED}Failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
if [[ -n "$LEVEL" ]]; then
    echo -e "Level:   ${LEVEL}"
else
    echo -e "Level:   all (L1, L2, Quant)"
fi
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
echo "Master log:       ${MASTER_LOG}"
echo "Results CSV:      ${CSV_FILE}"
echo ""

# Also save summary to log
{
    echo ""
    echo "=========================================="
    echo "SUMMARY"
    echo "=========================================="
    echo "Total:   ${TOTAL}"
    echo "Passed:  ${PASSED}"
    echo "Failed:  ${FAILED}"
    echo "Skipped: ${SKIPPED}"
    echo "CSV:     ${CSV_FILE}"
} >> "$MASTER_LOG"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
