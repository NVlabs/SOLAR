# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for the unified source-to-SOL API."""
import argparse
from dataclasses import asdict
import json
from solar.estimate import estimate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["auto", "v0", "v1", "v2"], default="auto")
    p.add_argument("--source-type", choices=["pytorch", "cuda", "jax", "triton"], default="pytorch")
    p.add_argument("--arch", default="B200")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--ir-path")
    p.add_argument("--api-key-file")
    p.add_argument("--llm-model")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--mem-budget-gb", type=float, default=16)
    p.add_argument("--sol-mode", choices=["sparse", "dense"], default="sparse",
                   help="tensor-core peak set for the headline SOL (issue #14): sparse (default; assumes structured-sparsity eligibility, a lower bound) or dense (the realistic bound); result.json carries both")
    a = vars(p.parse_args())
    from solar.common.cli_logging import configure as _configure_cli_logging
    _configure_cli_logging(bool(a.get("debug")))
    a["source_path"] = a.pop("source")
    result = estimate(**a)
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
