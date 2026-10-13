#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run SOLAR on one SOL-ExecBench problem at one workload shape.
#
# Usage:
#   scripts/run_execbench_problem.sh [problem_dir] [extra args for run_execbench_problem.py]
#
# Examples:
#   scripts/run_execbench_problem.sh                       # first L1 problem, workload 0, B200
#   scripts/run_execbench_problem.sh ../SOL-ExecBench/data/benchmark/L1/<problem> --workload-index 3
#   scripts/run_execbench_problem.sh ../SOL-ExecBench/data/benchmark/L2/<problem> --arch-config H200_SXM -v
#
# Sources ../env.sh (conda env + CUDA paths) if present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_SH="${SOLAR_ROOT}/../env.sh"

if [[ -f "${ENV_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SH}"
fi

cd "${SOLAR_ROOT}"
exec python3 "${SCRIPT_DIR}/run_execbench_problem.py" "$@"
