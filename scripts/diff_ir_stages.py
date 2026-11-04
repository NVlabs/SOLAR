#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localise a verification mismatch to the stage where it starts.

``verify_saved_ir.py`` reports *that* a saved einsum_lib program disagrees with
its source model.  This reports *where*: it runs the same program twice -- once
interpreted, once through the numba converter -- recording the output of every
function each call makes, and lines the two runs up stage by stage.

Reading the result:

* A stage that differs between the two paths is a converter bug.  The IR is one
  file, so any disagreement is introduced by the lowering.
* Agreement on every stage, with the final output still off, puts the
  discrepancy in the translation itself or in how the reference is built.
  ``--stages`` then narrows that down: it takes a file defining
  ``stages(inputs) -> {name: array}``, computed however the source model
  computes them, and diffs those against the recorded stages of the same name.

No LLM is involved, and sizing follows ``verify_saved_ir.py``: ``--config``
rewrites the source module's ``CONFIG`` before ``get_inputs()`` runs, which is
how the JAXBench baselines are shrunk to a debuggable size.  Exits non-zero when
some stage diverges, so a caller can tell "found it" from "nothing to see".

Usage:
  python3 scripts/diff_ir_stages.py MODEL_PY GENERATED_IR_PY [options]

Examples:
  # Reproduce 10p_Sparse_MoE at the size the pipeline verified it at.
  python3 scripts/diff_ir_stages.py \\
      jaxbench/solar_output/10p_Sparse_MoE/solar_model.py \\
      jaxbench/solar_output/10p_Sparse_MoE/dsl/solar_model_einsum_lib.py \\
      --config batch=1 --config seq_len=128 --config emb_dim=128 \\
      --config mlp_dim=256 --config num_experts=4

  # Same, also comparing against reference intermediates.
  python3 scripts/diff_ir_stages.py MODEL.py IR.py --config batch=1 \\
      --stages jaxbench/solar_output/10p_Sparse_MoE/stages_probe.py
"""

import argparse
import os
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) in sys.path:  # the checkout this script lives in must take precedence
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
if os.environ.get("SOLAR_ENGINE_SOURCE_PROBE"):  # provenance self-binding probe (tests/diagnostics)
    from solar.engine_source import probe_hook  # noqa: E402
    probe_hook()
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# verify_saved_ir.py is a thin wrapper over solar.translate since an earlier revision; its
# former private helpers live in the facade's runner under public names.
from solar.translate._runner import coerce_config_value as _coerce  # noqa: E402
from solar.translate._runner import sized_model as _sized_model  # noqa: E402


# Names that live in the execution namespace without being a stage of the
# computation: types, decorators, the modules, and the entry point itself
# (whose return value is a dict rather than an array).
_NOT_A_STAGE = frozenset({
    "Tensor", "Dim", "clear_dims", "kernel", "run_model",
    "np", "math", "numba", "exp", "log", "sqrt", "abs", "max", "min",
    "softmax",
})


def _as_array(value):
    """The array behind a stage's return value, or None if it is not one."""
    data = getattr(value, "data", value)
    if isinstance(data, np.ndarray):
        # float64 so integer and float stages diff the same way; these are
        # debug-sized, so the copy is affordable.
        return np.asarray(data, dtype=np.float64)
    return None


def _install_recorders(namespace: dict, log: list) -> None:
    """Wrap every callable in *namespace* so its output lands in *log*.

    Called after the module is executed, so decorators have already run and the
    wrappers only ever see real calls.  Recording by call order rather than by
    name keeps repeated use of one kernel distinguishable.
    """
    for name, obj in list(namespace.items()):
        if name in _NOT_A_STAGE or name.startswith("_"):
            continue
        if not callable(obj) or isinstance(obj, type):
            continue

        def _recorder(fn=obj, stage=name):
            def wrapper(*args, **kwargs):
                out = fn(*args, **kwargs)
                arr = _as_array(out)
                if arr is not None:
                    log.append((stage, arr))
                return out
            return wrapper

        namespace[name] = _recorder()


def _keyed(log: list) -> dict:
    """Turn a call-ordered log into ``{stage: array}``, numbering repeats."""
    seen: dict = {}
    keyed: dict = {}
    for name, arr in log:
        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name}#{seen[name]}"
        keyed[key] = arr
    return keyed


def _run_interpreted(code: str, inputs: dict):
    """Execute the IR as written, returning (outputs, stages)."""
    # Same globals verify.execute_translated() interprets under: kernels call
    # exp/max/sqrt as bare names that the framework injects.
    from solar.einsum_lib.verify import _MATH_BUILTINS
    import math

    namespace: dict = {**_MATH_BUILTINS, "math": math, "np": np}
    exec(code, namespace)  # noqa: S102
    log: list = []
    _install_recorders(namespace, log)
    outputs = namespace["run_model"](dict(inputs))
    return outputs, _keyed(log)


def _run_lowered(code: str, inputs: dict, use_fp64: bool):
    """Execute the IR through the numba converter, returning (outputs, stages).

    Raises if the converter declines, since the whole point here is to compare
    the two paths -- silently reporting the interpreter twice would be worse
    than failing.
    """
    from solar.einsum_lib.numba_accel import (
        execution_namespace, numba_lower,
    )

    lowered = numba_lower(code, boundscheck=True, use_fp64=use_fp64)
    if lowered is None:
        raise RuntimeError("numba_lower() could not lower this IR")

    namespace = execution_namespace(use_fp64=use_fp64)
    exec(lowered, namespace)  # noqa: S102
    log: list = []
    _install_recorders(namespace, log)
    outputs = namespace["run_model"](dict(inputs))
    return outputs, _keyed(log)


def _load_stages_probe(path: Path):
    """Import *path* and return its ``stages`` callable."""
    spec = importlib.util.spec_from_file_location("_stages_probe", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "stages"):
        raise SystemExit(f"{path.name} defines no stages(inputs) function")
    return module.stages


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _report(title: str, left: dict, right: dict, left_name: str,
            right_name: str) -> bool:
    """Print a stage-by-stage diff.  Returns True if every stage matched."""
    shared = [k for k in left if k in right]
    print(f"\n{title}")
    if not shared:
        print(f"  no stages in common ({left_name}: {sorted(left)}; "
              f"{right_name}: {sorted(right)})")
        return True

    print(f"  {'stage':<24} {'shape':<22} {'max |diff|':>12}   {'|values|':>10}")
    clean = True
    for key in shared:
        a, b = left[key], right[key]
        if a.shape != b.shape:
            print(f"  {key:<24} {str(a.shape):<22} {'shape ' + str(b.shape):>12}")
            clean = False
            continue
        d = _diff(a, b)
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        flag = "" if d <= 1e-9 * max(scale, 1.0) else "  <-- diverges"
        clean = clean and not flag
        print(f"  {key:<24} {str(a.shape):<22} {d:>12.3e}   {scale:>10.3e}{flag}")

    only_left = [k for k in left if k not in right]
    only_right = [k for k in right if k not in left]
    if only_left:
        print(f"  only in {left_name}: {only_left}")
    if only_right:
        print(f"  only in {right_name}: {only_right}")
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the stage where a saved einsum_lib program starts to "
                    "disagree, with no LLM calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Source model .py (the pipeline's input).")
    parser.add_argument("ir", help="Generated *_einsum_lib.py. Pass the DSL "
                                  "file, not the _numba.py one.")
    parser.add_argument("--source-type", default="jax",
                        choices=["jax", "triton", "cuda"],
                        help="Frontend that produced the reference (default: jax).")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a CONFIG field before building inputs. "
                             "Repeatable.")
    parser.add_argument("--stages", metavar="FILE",
                        help="Python file defining stages(inputs) -> "
                             "{stage_name: array} for the reference's "
                             "intermediates, keyed by IR function name.")
    parser.add_argument("--skip-lowered", action="store_true",
                        help="Interpret only. Use when the converter declines "
                             "and you only want the reference comparison.")
    parser.add_argument("--fp32", action="store_true",
                        help="Work at fp32 instead of the default fp64.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress frontend debug output.")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    ir_path = Path(args.ir).resolve()
    for path, what in ((model_path, "model"), (ir_path, "IR")):
        if not path.is_file():
            print(f"ERROR: {what} file not found: {path}", file=sys.stderr)
            return 2

    overrides = {}
    for item in args.config:
        if "=" not in item:
            print(f"ERROR: --config expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, _, value = item.partition("=")
        overrides[key.strip()] = _coerce(value)

    from solar.einsum_lib.verify import compare_outputs

    if args.source_type == "jax":
        from solar.einsum_lib.jax_frontend import prepare_jax as prepare
    elif args.source_type == "triton":
        from solar.einsum_lib.triton_frontend import prepare_triton as prepare
    else:
        from solar.einsum_lib.cuda_frontend import prepare_cuda as prepare

    use_fp64 = not args.fp32
    code = ir_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="diff_ir_stages_") as tmp, \
            _sized_model(model_path, overrides, Path(tmp)) as sized:
        if overrides:
            print(f"CONFIG overrides: {overrides}")

        print(f"Building reference from {sized.name} ...")
        t0 = time.perf_counter()
        prepared = prepare(str(sized), True, cap_input_size=False,
                           use_fp64=use_fp64, debug=not args.quiet)
        (_source, _shapes, input_dtypes, reference_outputs, all_inputs,
         _sparsity, _mask_fill, _buffers, verify_shapes) = prepared[:9]
        print(f"Reference built in {time.perf_counter() - t0:.1f}s")
        for name, arr in (reference_outputs or {}).items():
            print(f"  reference output {name!r}: {arr.shape} {arr.dtype}")

        print("\nInterpreting the IR ...")
        t0 = time.perf_counter()
        interp_out, interp_stages = _run_interpreted(code, all_inputs)
        print(f"  {len(interp_stages)} stages in {time.perf_counter() - t0:.1f}s")

        lowered_out = None
        lowered_stages = None
        if not args.skip_lowered:
            print("Running the IR through the numba converter ...")
            t0 = time.perf_counter()
            lowered_out, lowered_stages = _run_lowered(code, all_inputs, use_fp64)
            print(f"  {len(lowered_stages)} stages in "
                  f"{time.perf_counter() - t0:.1f}s")

        for label, outputs in (("interpreted", interp_out),
                               ("lowered", lowered_out)):
            if outputs is None:
                continue
            result = compare_outputs(outputs, reference_outputs)
            verdict = ("matches the reference" if result.success
                       else f"differs: {result.error}")
            print(f"\n{label:<12} {verdict}")

        clean = True
        if lowered_stages is not None:
            clean = _report("Interpreted vs lowered (a difference here is a "
                            "converter bug):",
                            interp_stages, lowered_stages,
                            "interpreted", "lowered")

        if args.stages:
            stages_fn = _load_stages_probe(Path(args.stages).resolve())
            ref_stages = {k: np.asarray(v, dtype=np.float64)
                          for k, v in stages_fn(all_inputs).items()}
            clean = _report("Reference vs interpreted IR (a difference here is "
                            "a translation or harness bug):",
                            ref_stages, interp_stages, "reference", "IR") and clean
        elif lowered_stages is not None and clean:
            print("\nBoth paths agree on every stage, so the converter is not "
                  "the problem.  Write a stages(inputs) probe for this model "
                  "and re-run with --stages to place the difference.")

        return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
