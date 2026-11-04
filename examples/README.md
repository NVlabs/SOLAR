<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Examples

Each `examples/<Name>/` directory holds a kernelbench-style PyTorch model
(`class Model(nn.Module)` plus `get_inputs()`, optionally
`get_init_inputs()`) and runner scripts. Run everything from the repository root with a
Python 3.10+ interpreter that has SOLAR installed; set `PYTHON=` to pick one.

| Runner | Route | API call | What it does |
|---|---|---|---|
| `<Name>/run_solar.sh` | mechanical | none | extract -> lower -> analyze -> predict (most also export Timeloop); writes to `<Name>/output/` or `SOLAR_<NAME>_OUTPUT_DIR` |

## Checked expected counts

Hand-derived counts that the mechanical route reproduced at the internal
source snapshot of 2026-10-22 from which this candidate was exported (re-run
at the exact public candidate pending). If a run differs, the lowering is not
trustworthy for that installation; see `docs/SOLAR_V1_FLOW.md` and
`docs/TROUBLESHOOTING.md`.

| Example | Shapes | `total.macs` | `total.unfused_elements` | Router decision | Notes |
|---|---|---|---|---|---|
| `TwoInputMatmul` | A[32,64] @ B[64,128] | 262,144 | 14,336 | `mechanical` | audited happy path; `run_solar.sh` checks the values |
| `Matmul` | W[4,32,64] (parameter) @ x[64,128] | 1,048,576 | 32,768 | `mechanical` | depends on learned-parameter capture (shim or patched torchview); earlier unpatched runs undercounted 4x |
| `Attention` | B=2, S=32, H=64, 4 heads | 1,310,720 | 114,688 | `agentic` | converter lowers it; the router declines the `scores * scale` node (one tensor operand). Softmax and scaling count zero MACs by convention |

The other examples (`BERT`, `Conv2d`) run through the mechanical route but
their counts have not been independently hand-checked for this
documentation. Treat their outputs as smoke results until you audit at least
one layer.

The agentic route's shipped, verified example lives in
`skills/verified-translation/example/` (RMSNorm adapter + DSL); see
`docs/SOLAR_V2_FLOW.md`. `examples_jax/` holds three small JAX/Flax sources
used as fixtures for that route; Triton and CUDA example sources are not
shipped in this candidate. Agentic sources need the per-source
reference contract from `docs/SOLAR_V2_CONTRACTS.md` and were not executed
for this documentation.
