<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable window padding / crop controls

The maintainers' window-attention padding/crop regression test and its fixture
bundle are withheld from the public tree: the bundle derives from benchmark
content outside the public export allowlist (see the `omitted` rows in
`release/packaging/RUNTIME_UPDATE_MANIFEST.json`). This directory replaces
that coverage with content that is public by construction.

## Provenance of every byte here

- `fixtures/window_padding_crop_einsum_lib.py` is a hand-authored DSL
  program built from two public patterns only: the SAM-style zero-padded
  window partition already used by `tests/test_campaign_repairs.py`
  (`TestWindowCounts`) and the scatter/crop pattern documented in
  `docs/einsum_lib_prompt.md`. Window size 3, 6 channels and 12 projection
  features are arbitrary hand-chosen constants.
- Every expected number in `test_window_padding_crop_counts.py` is a literal
  derived by hand from `NH = ceil(H / 3)`, `NW = ceil(W / 3)` (table below).
  No benchmark definition, case row, reference program, measured value or
  file from the withheld bundle was read into these files.
- Nothing is exported from the maintainers' repository; the files are
  recorded as `lane_authored` in the runtime-update manifest.

| B, H, W | NH, NW | padded windows (B·NH·NW) | padded H, W | partial? |
|---|---|---|---|---|
| 1, 1, 1 | 1, 1 | 1 | 3, 3 | yes |
| 1, 4, 7 | 2, 3 | 6 | 6, 9 | yes |
| 2, 6, 6 | 2, 2 | 8 | 6, 6 | no (divisible boundary) |
| 1, 3, 3 | 1, 1 | 1 | 3, 3 | no (one exact window) |
| 2, 5, 9 | 2, 3 | 12 | 6, 9 | H partial, W divisible |
| 2, 50, 70 | 17, 24 | 816 | 51, 72 | yes |

## What the static controls check (collected by default)

`test_window_padding_crop_counts.py` parses the fixture with the public
`ast_parser` and applies the converter's pure counting functions
(`_eval_size_spec`, `_kernel_loop_iterations`) to hand-built call-site
shapes. No DSL program, model, reference or GPU executes.

- the parsed partition nest is one weighted path `B, NH, NW, I, J, C` whose
  `NH`/`NW` specs are `(X.shape[axis] + 3 - 1) // 3`, and those specs
  evaluate to the hand-derived counts at every shape in the table;
- the statically declared partition output is `[B·NH·NW, 9, C]` and the
  declared crop output is the original `[B, H, W, C]`;
- the partition and the per-window projection count the **padded** domain
  (`B·NH·NW·9·C` and `B·NH·NW·9·F·C`), strictly more than a floored
  translation whenever a window is partial and equal on the divisible
  boundary;
- the crop/unpartition counts the **original** domain `B·H·W·C`, strictly
  less than the padded domain whenever a window is partial;
- the same partition kernel with the window size passed as an `int`
  parameter is **refused** (`ValueError`, "Expose its extent") instead of
  being counted with the outer window loops dropped;
- the reuse guard (`solar.translate.reuse.guard_profile`) flags the ceil-
  divided partition dims, the two `X.shape` padding guards and the `//`/`%`
  crop reads, so a verification at one shape is never transferred to another
  shape (the documented lesson of a 28x28 check that exercised no padding);
- fixture integrity: both fixture modules compile, the numerical control is
  not collected by default, and the fixture names none of the withheld
  bundle's artefacts.

Run from the checkout root in the approved sourced environment:

```bash
source "$SOLAR_ENV_SH"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest -q -p no:cacheprovider tests/release_portable
```

## What is delivered but NOT executed

`pending_numerical_window_padding_crop.py` executes the fixture DSL through
`solar.einsum_lib.verify.execute_translated` and compares the padded windows,
the projection and the crop against a reference built from public PyTorch
calls (`F.pad`, reshape/permute partition, `F.linear`, `[:, :h, :w]`), and
checks that padded positions are exactly zero. It was written under a task
that forbids DSL execution and **has not been run**; its file name keeps it
out of default collection on purpose (no `[tool.pytest]` section overrides
the default `test_*.py` pattern). The first execution and its receipt belong
to an independent owner; run it explicitly:

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest -q -p no:cacheprovider \
  tests/release_portable/pending_numerical_window_padding_crop.py
```

Until that run is recorded, this tree makes **no numerical padding/crop
claim**. The static controls above certify geometry and loop counts only,
not numerical equality, traffic or SOL.
