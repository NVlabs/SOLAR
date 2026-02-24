#!/usr/bin/env bash
#
# Run Solar pipeline on FlashInfer benchmark ops with limited workloads.
#
# Usage:
#   ./run_flashinferbench_all.sh                              # Generate and run all (1 workload each)
#   ./run_flashinferbench_all.sh --ops gemm,rmsnorm           # Only process gemm and rmsnorm
#   ./run_flashinferbench_all.sh --generate                   # Only generate models
#   ./run_flashinferbench_all.sh --run                        # Only run pipeline (assumes models exist)
#   ./run_flashinferbench_all.sh --phase analysis,perf --arch B200  # Rerun analysis+perf with B200
#   ./run_flashinferbench_all.sh --copy-uuid                  # Copy uuid.yaml to output dirs
#   ./run_flashinferbench_all.sh --max-workloads 5            # Use 5 workloads per definition
#
# This generates N concrete workloads per operation definition (default: 1), then runs
# the full Solar pipeline (graph → einsum → analysis → perf → timeloop).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SOLAR_ROOT}/.." && pwd)"

POSTPROCESS_DIR="${REPO_ROOT}/flashinfer-trace-postprocess"
OUTPUT_BASE="${SOLAR_ROOT}/output_flashinferbench"

# Defaults
DO_GENERATE=true
DO_RUN=true
DO_COPY_UUID=false
DO_SUMMARY=false
MAX_WORKLOADS=1
SKIP_EXISTING=true
OP_TYPES=""  # Empty = all ops
PHASES=""    # Empty = all phases
ARCH_CONFIG="H100_PCIe"
PRECISION="fp16"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generate)
            DO_RUN=false
            shift
            ;;
        --run)
            DO_GENERATE=false
            shift
            ;;
        --copy-uuid)
            DO_COPY_UUID=true
            DO_GENERATE=false
            DO_RUN=false
            shift
            ;;
        --summary)
            DO_SUMMARY=true
            DO_GENERATE=false
            DO_RUN=false
            shift
            ;;
        --max-workloads)
            MAX_WORKLOADS="$2"
            shift 2
            ;;
        --ops)
            OP_TYPES="$2"
            shift 2
            ;;
        --phase)
            PHASES="$2"
            shift 2
            ;;
        --arch)
            ARCH_CONFIG="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --no-skip)
            SKIP_EXISTING=false
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ops TYPES        Comma-separated op types (e.g., gemm,rmsnorm). Default: all"
            echo "  --phase PHASES     Comma-separated phases: graph,einsum,analysis,perf,timeloop"
            echo "  --arch ARCH        Architecture config (default: H100_PCIe)"
            echo "  --precision PRE    Precision: fp32, fp16, bf16 (default: fp16)"
            echo "  --generate         Only generate models, don't run pipeline"
            echo "  --run              Only run pipeline (assumes models exist)"
            echo "  --copy-uuid        Copy uuid.yaml from postprocess to output dirs"
            echo "  --summary          Generate comparison CSV with solutions from traces"
            echo "  --max-workloads N  Max workloads per definition (default: 1)"
            echo "  --no-skip          Force reprocess even if output exists (default: skip finished)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Available op types: gemm, rmsnorm, sampling, moe, mla_paged, gqa_paged, gqa_ragged"
            echo "Available phases: graph, einsum, analysis, perf, timeloop"
            echo ""
            echo "Examples:"
            echo "  $(basename "$0") --ops gemm --phase perf --arch B200    # Rerun perf for gemm with B200"
            echo "  $(basename "$0") --phase analysis,perf --arch B200     # Rerun analysis+perf with B200"
            echo "  $(basename "$0") --copy-uuid                           # Copy uuid.yaml to output dirs"
            echo "  $(basename "$0") --summary --arch B200                 # Generate comparison CSV"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Copy UUID files only
if [[ "$DO_COPY_UUID" == "true" ]]; then
    echo "=========================================="
    echo "Copying uuid.yaml files to output directories"
    echo "=========================================="
    
    copied=0
    created=0
    
    for src in "${POSTPROCESS_DIR}"/*/*/*/uuid.yaml; do
        if [[ -f "$src" ]]; then
            rel_path="${src#$POSTPROCESS_DIR/}"
            dest="${OUTPUT_BASE}/${rel_path}"
            dest_dir=$(dirname "$dest")
            
            # Create directory if it doesn't exist
            if [[ ! -d "$dest_dir" ]]; then
                mkdir -p "$dest_dir"
                ((created++)) || true
            fi
            
            cp "$src" "$dest"
            echo "  Copied: $rel_path"
            ((copied++)) || true
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "Copied: ${copied}, Created dirs: ${created}"
    echo "=========================================="
    exit 0
fi

# Generate summary CSV
if [[ "$DO_SUMMARY" == "true" ]]; then
    echo "=========================================="
    echo "Generating comparison summary (arch: ${ARCH_CONFIG})"
    echo "=========================================="
    
    python3 "${SCRIPT_DIR}/parse_flashinferbench_results.py" --arch "${ARCH_CONFIG}"
    exit 0
fi

# Build header
echo "=========================================="
if [[ -n "$OP_TYPES" ]]; then
    echo "FlashInfer Benchmark - Ops: ${OP_TYPES}"
else
    echo "FlashInfer Benchmark - All Ops"
fi
echo "  Max workloads: ${MAX_WORKLOADS}"
echo "  Arch: ${ARCH_CONFIG}"
if [[ -n "$PHASES" ]]; then
    echo "  Phases: ${PHASES}"
fi
echo "=========================================="

# Convert comma-separated ops to array
if [[ -n "$OP_TYPES" ]]; then
    IFS=',' read -ra OPS_ARRAY <<< "$OP_TYPES"
else
    OPS_ARRAY=()
fi

# Step 1: Generate models (only if not specifying phases, or if --generate explicitly)
if [[ "$DO_GENERATE" == "true" ]] && [[ -z "$PHASES" ]]; then
    echo ""
    echo "[Step 1] Generating models (--max-workloads ${MAX_WORKLOADS})..."
    
    if [[ ${#OPS_ARRAY[@]} -gt 0 ]]; then
        # Generate for specific ops
        for op in "${OPS_ARRAY[@]}"; do
            echo "  Generating for op: $op"
            "${SCRIPT_DIR}/run_flashinferbench.sh" --generate-only --max-workloads "${MAX_WORKLOADS}" "$op"
        done
    else
        # Generate all
        "${SCRIPT_DIR}/run_flashinferbench.sh" --generate-only --max-workloads "${MAX_WORKLOADS}"
    fi
    echo ""
    echo "[Step 1] Done generating models."
fi

# Step 2: Run the solar pipeline
if [[ "$DO_RUN" == "true" ]]; then
    echo ""
    if [[ -n "$PHASES" ]]; then
        echo "[Step 2] Running Solar pipeline (phases: ${PHASES}, arch: ${ARCH_CONFIG})..."
    else
        echo "[Step 2] Running Solar pipeline..."
    fi
    
    RUN_ARGS=("--max-workloads" "${MAX_WORKLOADS}" "--arch" "${ARCH_CONFIG}" "--precision" "${PRECISION}")
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -z "$PHASES" ]]; then
        # Only skip existing for full pipeline, not partial reruns
        RUN_ARGS+=("--skip-existing")
    fi
    if [[ -n "$PHASES" ]]; then
        RUN_ARGS+=("--phase" "${PHASES}")
    fi
    
    if [[ ${#OPS_ARRAY[@]} -gt 0 ]]; then
        # Run for specific ops
        for op in "${OPS_ARRAY[@]}"; do
            echo "  Processing op: $op"
            "${SCRIPT_DIR}/run_flashinferbench.sh" "${RUN_ARGS[@]}" "$op"
        done
    else
        # Run all
        "${SCRIPT_DIR}/run_flashinferbench.sh" "${RUN_ARGS[@]}"
    fi
    echo ""
    echo "[Step 2] Done running pipeline."
fi

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
