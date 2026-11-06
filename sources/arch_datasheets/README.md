<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture datasheet excerpts (issue #14)

Provenance for every `# source:` note in `configs/arch/*.yaml`. Nothing here
is used at run time.

`SOURCES.json` is the record. Per capture it keeps the URL, effective URL,
HTTP status, fetch time (UTC), size and sha256 of the raw bytes, the sha256 and
line count of the full text extract that was read (`full_extract`), and the
public excerpt shipped here (`extract_file`, `extract_sha256`,
`excerpt_of_full_extract_lines`). `figures_by_config` quotes the printed
figures, columns and footnotes each profile uses; `other_fetches_and_gaps`
records URLs that returned 404 / 403 / a JavaScript viewer; figures no fetched
document states are listed as `unverified_from_source` and flagged in the yaml
comment (Jetson Thor FP32; the Thor FP16-dense / BF16 / TF32 / INT8-dense rates
are `derived` from the published sparse figures, as the yaml says).

The `.txt` files are **excerpts**, not page captures: the specification-table
rows and footnotes that support the citations, reproduced verbatim from the
full extract (`pdftotext -layout` for PDFs, tag-stripped text for HTML). The
first line of each file names the full-extract lines it reproduces, the full
extract's sha256 and the raw document's sha256, so an excerpt can be checked
against the original. The full extracts and raw bytes are retained by the
release lane outside the repository. Raw-capture hashes, URLs and fetch dates
are those of the original fetch; `extract_sha256` is the excerpt's own hash.

Where the documents come from: ten captures are nvidia.com or
developer.nvidia.com pages and PDFs. The H100 datasheet PDF is an
NVIDIA-authored document whose copy was fetched from a partner mirror
(megware.com), because the nvidia.com candidates served the H100 NVL product
brief, an HTML page or HTTP 403; its H100 SXM column matches the nvidia.com
product page figure for figure, and it is the only fetched document that
prints the H100 PCIe column.

Sparse versus dense: vendor tables print tensor-core rates "with sparsity"
(2:4 structured sparsity) or as `dense | sparse` pairs. SOLAR records both;
the sparse figure is a lower bound that assumes 2:4 eligibility and the dense
figure is the realistic estimate for dense workloads (`docs/SOL_GUIDE.md`,
`configs/arch/README.md`).

Attribution: the excerpted text and figures are NVIDIA-published product
specifications, copyright NVIDIA Corporation, reproduced here for citation and
verification of the configs. They are third-party material, not licensed under
this project's Apache-2.0 licence, and no redistribution right beyond this
citation is asserted (`REUSE.toml`,
`LICENSES/LicenseRef-NVIDIA-Published-Specification-Excerpt.txt`). Refer to the
URLs in `SOURCES.json` for the documents themselves.
