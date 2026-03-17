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

set -euo pipefail

# Install Solar and its patched torchview dependency.
#
# This script:
#   1. Clones torchview and checks out the tested commit
#   2. Applies the Solar patch (parameter tensor support)
#   3. Installs torchview from source
#   4. Installs Solar dependencies
#
# Usage:
#   bash install.sh              # Install everything
#   bash install.sh --skip-torch # Skip torch installation (if already installed)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TORCHVIEW_REPO="https://github.com/mert-kurttutan/torchview.git"
TORCHVIEW_COMMIT="edbe1fa"
TORCHVIEW_DIR="${REPO_ROOT}/torchview"
PATCH_FILE="${SCRIPT_DIR}/patches/torchview-parameter-tensors.patch"

SKIP_TORCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-torch)
            SKIP_TORCH=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash install.sh [--skip-torch]"
            echo ""
            echo "Options:"
            echo "  --skip-torch  Skip PyTorch installation (use if already installed)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Solar Installation ==="
echo ""

# Step 1: Clone and patch torchview
echo "==> Step 1: Setting up patched torchview..."
if [[ -d "${TORCHVIEW_DIR}" ]]; then
    echo "  torchview directory exists: ${TORCHVIEW_DIR}"
    cd "${TORCHVIEW_DIR}"

    # Check if already at the right commit
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

# Apply patch
if [[ -f "${PATCH_FILE}" ]]; then
    echo "  Applying Solar patch..."
    # Check if patch is already applied
    if git apply --check "${PATCH_FILE}" 2>/dev/null; then
        git apply "${PATCH_FILE}"
        echo "  Patch applied successfully."
    else
        echo "  Patch already applied or conflicts detected, skipping."
    fi
else
    echo "  Warning: Patch file not found: ${PATCH_FILE}"
    echo "  torchview will be installed without parameter tensor support."
fi

# Step 2: Install torchview from source
echo ""
echo "==> Step 2: Installing torchview from source..."
cd "${TORCHVIEW_DIR}"
pip install -e . --no-deps
echo "  torchview installed."

# Step 3: Install Solar dependencies
echo ""
echo "==> Step 3: Installing Solar dependencies..."
cd "${SCRIPT_DIR}"

if [[ "$SKIP_TORCH" == "true" ]]; then
    echo "  Skipping torch (--skip-torch specified)."
    # Install everything except torch
    pip install networkx>=3.0 pyyaml>=6.0 numpy>=1.24.0 matplotlib>=3.5.0 \
                scipy>=1.10.0 pandas>=1.5.0 einops>=0.6.0
else
    pip install -r requirements.txt
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Torchview: ${TORCHVIEW_DIR} (commit ${TORCHVIEW_COMMIT} + Solar patch)"
echo "Solar:     ${SCRIPT_DIR}"
echo ""
echo "To verify:"
echo "  python -c 'import torchview; print(torchview.__file__)'"
echo "  python -c 'from solar.graph import PyTorchProcessor; print(\"OK\")'"
