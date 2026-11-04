# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-check template (skill step 4): run the DSL in a verify-style namespace
against the adapter golden — never `import` the DSL module.

Usage: $PY selfcheck.py [manual_dir]   (defaults to this example directory)

Inputs are fed as float64 (matching official verification). Expected error is
RELATIVE to output magnitude: |err| ~ rtol_class * max|output| — e.g. a fp32
golden gives ~1e-7 relative, so outputs of magnitude ~64 legitimately show
~1e-5 absolute. Compare against the tolerance CLASS (reference.md), not raw
atol.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

spec = importlib.util.spec_from_file_location("adapter", d / "adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

golden_in = adapter.get_inputs()
golden_out = adapter.kernel_fn(golden_in)

# exec the DSL (never import): verify-style namespace
from solar.einsum_lib import verify as _v
ns = {**_v._MATH_BUILTINS, "math": math, "np": np}
exec(compile((d / "solar_model_einsum_lib.py").read_text(),
             str(d / "solar_model_einsum_lib.py"), "exec"), ns)

fp64_in = {k: np.asarray(v, dtype=np.float64) if np.asarray(v).dtype.kind == "f"
           else np.asarray(v) for k, v in golden_in.items()}
dsl_out = ns["run_model"](fp64_in)

worst = 0.0
for k, ref in golden_out.items():
    got = np.asarray(dsl_out[k], dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    err = float(np.max(np.abs(got - ref))) if ref.size else 0.0
    rel = err / max(float(np.max(np.abs(ref))), 1e-30)
    print(f"{k}: max_abs={err:.3e}  max_rel_to_peak={rel:.3e}")
    worst = max(worst, rel)
print(f"WORST relative-to-peak: {worst:.3e}  "
      f"({'OK (fp32-golden noise class)' if worst < 1e-5 else 'inspect vs tolerance class'})")
