#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install Solar using uv (https://github.com/astral-sh/uv).
#
# This script:
#   1. Ensures uv is installed (installs to ~/.local/bin or uses existing)
#   2. Optionally clones and patches torchview (same as install.sh)
#   3. Runs uv sync --python 3.10 (creates .venv and installs dependencies)
#   4. Installs patched torchview from source if available
#   5. Runs uv pip install -e . (editable install of Solar)
#
# Prerequisites: Python 3.10 on PATH (or uv will download it).
#
# Usage:
#   bash install_uv.sh              # Full install with patched torchview
#   bash install_uv.sh --skip-torchview   # Use torchview from PyPI only
#   bash install_uv.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TORCHVIEW_REPO="https://github.com/mert-kurttutan/torchview.git"
TORCHVIEW_COMMIT="edbe1fa"
TORCHVIEW_DIR="${REPO_ROOT}/torchview"
PATCH_FILE="${SCRIPT_DIR}/patches/torchview-parameter-tensors.patch"

SKIP_TORCHVIEW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-torchview)
            SKIP_TORCHVIEW=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash install_uv.sh [--skip-torchview]"
            echo ""
            echo "Options:"
            echo "  --skip-torchview  Use torchview from PyPI (do not clone/patch from source)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Solar installation (uv) ==="
echo ""

# Step 0: Ensure uv is installed
echo "==> Step 0: Ensuring uv is installed..."
if command -v uv >/dev/null 2>&1; then
    echo "  uv found: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "  ERROR: uv not found after install. Add ~/.local/bin to PATH and retry."
        exit 1
    fi
fi
echo ""

# Step 1: Optional clone and patch torchview
if [[ "$SKIP_TORCHVIEW" != "true" ]]; then
    echo "==> Step 1: Setting up patched torchview..."
    if [[ -d "${TORCHVIEW_DIR}" ]]; then
        echo "  torchview directory exists: ${TORCHVIEW_DIR}"
        cd "${TORCHVIEW_DIR}"
        current_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        if [[ "$current_commit" != "${TORCHVIEW_COMMIT}"* ]]; then
            echo "  Checking out commit ${TORCHVIEW_COMMIT}..."
            git fetch origin
            git checkout "${TORCHVIEW_COMMIT}"
        fi
    else
        echo "  Cloning torchview..."
        git clone "${TORCHVIEW_REPO}" "${TORCHVIEW_DIR}"
        cd "${TORCHVIEW_DIR}"
        git checkout "${TORCHVIEW_COMMIT}"
    fi

    if [[ -f "${PATCH_FILE}" ]]; then
        echo "  Applying Solar patch..."
        if git apply --check "${PATCH_FILE}" 2>/dev/null; then
            git apply "${PATCH_FILE}"
            echo "  Patch applied successfully."
        else
            echo "  Patch already applied or conflicts detected, skipping."
        fi
    else
        echo "  Warning: Patch file not found: ${PATCH_FILE}"
    fi
    echo ""
else
    echo "==> Step 1: Skipping torchview clone/patch (--skip-torchview)."
    echo ""
fi

# Step 2: uv sync with Python 3.10 (creates .venv, installs from pyproject.toml)
echo "==> Step 2: uv sync --python 3.10..."
cd "${SCRIPT_DIR}"
uv sync --python 3.10
echo "  Sync complete."
echo ""

# Step 3: Install patched torchview from source if we set it up
if [[ "$SKIP_TORCHVIEW" != "true" ]] && [[ -d "${TORCHVIEW_DIR}" ]]; then
    echo "==> Step 3: Installing patched torchview from source into venv..."
    uv pip install -e "${TORCHVIEW_DIR}" --no-deps
    echo "  torchview installed from source."
else
    echo "==> Step 3: Using torchview from PyPI (already installed by uv sync)."
fi
echo ""

# Step 4: Editable install of Solar
echo "==> Step 4: uv pip install -e . (Solar editable install)..."
cd "${SCRIPT_DIR}"
uv pip install -e .
echo "  Solar installed in editable mode."
echo ""

echo "=== Installation complete ==="
echo ""
echo "Virtual env: ${SCRIPT_DIR}/.venv"
echo "Activate with: source ${SCRIPT_DIR}/.venv/bin/activate"
echo ""
echo "To verify:"
echo "  uv run python -c 'import torchview; print(torchview.__file__)'"
echo "  uv run python -c 'from solar.graph import PyTorchProcessor; print(\"OK\")'"
echo ""
echo "Optional: run 'uv lock' and commit uv.lock for reproducible installs."
