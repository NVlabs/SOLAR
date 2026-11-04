#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Trace a saved einsum_lib program at ORIGINAL (full) shapes, with no LLM.

Verification may run at capped shapes (see the capping-weights skill), but
SOL/graph analysis must see the true workload. This tool rebuilds the trace
inputs straight from a SOL-ExecBench ``definition.json`` (const axes at their
original values, var axes from ``workload.jsonl``), materializes every tensor
as a ZERO-MEMORY broadcast placeholder (``np.broadcast_to`` of a scalar), runs
the shape-only trace, and emits the dsl dir (``*_einsum_lib.py`` +
``*_einsum_lib.yaml``) that ``solar.einsum_dsl.cli`` consumes.

Custom inputs the DSL needs beyond the definition (e.g. ``inv_freq``,
``seq_id``, ``cu_seqlens``):
  * ``--fallback-adapter adapter.py``  — reuse the adapter's array when its
    shape does not involve a capped/overridden axis (safe for head-dim tables);
    axis-dependent fallbacks are refused with an error naming the key.
  * ``--custom-py file.py``            — module defining
    ``full_inputs(axes: dict) -> dict[str, np.ndarray]`` for keys needing
    original-scale values (small int arrays like cu_seqlens are cheap).

Usage:
  python scripts/trace_saved_ir.py DSL.py --definition <problem_dir> \
      --out-dir OUT [--fallback-adapter adapter.py] [--custom-py extra.py] \
      [--axis NAME=VALUE ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # import the checkout this script lives in
if str(REPO_ROOT) in sys.path:  # the checkout this script lives in must take precedence
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
if os.environ.get("SOLAR_ENGINE_SOURCE_PROBE"):  # provenance self-binding probe (tests/diagnostics)
    from solar.engine_source import probe_hook  # noqa: E402
    probe_hook()

import numpy as np


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_axes(defn: dict, workload: dict | None, cli_axes: dict) -> dict:
    """Original axis values: const -> definition, var -> workload (UNCAPPED)."""
    axes: dict[str, int] = {}
    for name, spec in defn["axes"].items():
        if spec["type"] == "const":
            axes[name] = spec["value"]
        elif spec["type"] == "var":
            if name in cli_axes:
                axes[name] = cli_axes[name]
            elif workload and name in workload.get("axes", {}):
                axes[name] = workload["axes"][name]
            else:
                axes[name] = 4
    for name, spec in defn["axes"].items():
        if spec["type"] == "expr":
            axes[name] = int(eval(spec["expression"], {"__builtins__": {}}, dict(axes)))
    axes.update(cli_axes)
    return axes


_NP_DTYPES = {
    "float32": np.float32, "float64": np.float64, "bfloat16": np.float32,
    "float16": np.float32, "int64": np.int64, "int32": np.int32,
    "bool": np.float32,
}


def zeros_placeholder(shape: tuple, np_dtype) -> np.ndarray:
    """Zero-memory full-shape array (read-only broadcast view of a scalar)."""
    return np.broadcast_to(np.zeros((), dtype=np_dtype), tuple(int(s) for s in shape))


def build_definition_inputs(defn: dict, axes: dict) -> dict:
    out = {}
    for name, spec in defn["inputs"].items():
        shape = spec.get("shape")
        if shape is None:  # scalar input — adapters hardcode these; skip
            continue
        resolved = tuple(axes[d] if d in axes else int(d) for d in shape)
        out[name] = zeros_placeholder(resolved, _NP_DTYPES.get(spec.get("dtype", "float32"), np.float32))
    return out


def dsl_required_keys(code: str) -> set:
    # Both access forms count: inputs["key"] and inputs.get("key").
    return set(re.findall(r"inputs\[[\"']([\w]+)[\"']\]", code)) | set(
        re.findall(r"inputs\.get\(\s*[\"']([\w]+)[\"']", code))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dsl_py", help="Saved *_einsum_lib.py program")
    ap.add_argument("--definition", required=True,
                    help="SOL-ExecBench problem dir (definition.json [+ workload.jsonl])")
    ap.add_argument("--out-dir", required=True, help="Output dsl dir")
    ap.add_argument("--workload-idx", type=int, default=0)
    ap.add_argument("--axis", action="append", default=[], metavar="NAME=VALUE",
                    help="Override/add an axis value (repeatable)")
    ap.add_argument("--fallback-adapter", default=None,
                    help="adapter.py supplying custom inputs whose shapes don't involve capped axes")
    ap.add_argument("--custom-py", default=None,
                    help="module with full_inputs(axes)->dict for original-scale custom inputs")
    ap.add_argument("--measure-density", action="store_true",
                    help="Harvest per-tensor densities from a FULL-VALUE run at the "
                         "fallback-adapter's capped shapes and inject them into the "
                         "original-shape yaml through a TRUST LADDER: (1) analytic "
                         "densities from the original trace (top_k_mask k/E) always "
                         "win; (2) shape-invariant measurements inject directly; "
                         "(3) scale-coupled ones (tensor shape differs between "
                         "harvest and original) AUTO-REVERT to a non-cap full-value "
                         "run at original shapes when it fits --mem-budget; "
                         "(4) otherwise they are REFUSED (SOL stays dense) with a "
                         "--density-override remedy printed.")
    ap.add_argument("--density-override", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="Force a tensor's density in the output yaml (repeatable; "
                         "takes precedence over measured values)")
    ap.add_argument("--measure-attention-density", action="store_true",
                    help="With --measure-density, also measure attention probabilities. "
                         "By default softmax/probability names and tensors with "
                         "role='attention_probability' stay dense. This opt-in "
                         "changes the dense-attention accounting convention.")
    ap.add_argument("--mem-budget", type=float, default=30.0,
                    help="GB allowed for the auto-revert full-value run at ORIGINAL "
                         "shapes (ladder rung 3; default 30). Estimated as 3x the "
                         "fp64 footprint of all traced tensors; if it exceeds the "
                         "budget the scale-coupled densities are refused instead.")
    ap.add_argument("--measure-timeout", type=int, default=300,
                    help="Seconds allowed for the full-value density run (default 300; "
                         "on timeout the trace proceeds without measured densities)")
    args = ap.parse_args()
    if args.measure_attention_density and not args.measure_density:
        ap.error("--measure-attention-density requires --measure-density")

    pdir = Path(args.definition)
    defn = json.loads((pdir / "definition.json").read_text())
    workload = None
    wkl = pdir / "workload.jsonl"
    if wkl.exists():
        lines = wkl.read_text().strip().split("\n")
        if args.workload_idx < len(lines):
            workload = json.loads(lines[args.workload_idx])
    cli_axes = {}
    for item in args.axis:
        k, _, v = item.partition("=")
        cli_axes[k.strip()] = int(v)

    axes = resolve_axes(defn, workload, cli_axes)
    inputs = build_definition_inputs(defn, axes)

    code = open(args.dsl_py).read()
    needed = dsl_required_keys(code)
    missing = needed - set(inputs)

    if args.custom_py:
        cmod = _load_module(args.custom_py, "_custom_full_inputs")
        extra = cmod.full_inputs(dict(axes))
        # custom-py is explicit user intent: it may OVERRIDE definition-shaped
        # inputs (e.g. quant DSLs that take pre-decomposed weights/scales under
        # the same names the definition declares in packed form).
        for k, v in extra.items():
            if k not in needed:
                continue
            if k in inputs and tuple(inputs[k].shape) != tuple(v.shape):
                print(f"  [custom] {k}: overriding definition shape "
                      f"{inputs[k].shape} -> {v.shape}", file=sys.stderr)
            inputs[k] = v
        missing = needed - set(inputs)

    if missing and args.fallback_adapter:
        amod = _load_module(args.fallback_adapter, "_fallback_adapter")
        cap = amod.get_inputs()
        # Reuse the adapter's array for custom keys. SAFE when the key's shape
        # depends only on preserved architectural dims (rope tables like
        # inv_freq = head_dim/2). UNSAFE when it depends on a capped var axis
        # (seq-derived masks, cu_seqlens) — we can't detect that reliably here,
        # so every reuse is logged with its shape; the sweep report surfaces
        # these lines for review, and unsafe keys should move to --custom-py.
        for k in sorted(missing):
            if k in cap:
                print(f"  [fallback] {k}: reusing adapter array shape={cap[k].shape} "
                      f"(REVIEW: must not depend on a capped axis)", file=sys.stderr)
                inputs[k] = cap[k]
        missing = needed - set(inputs)

    if missing:
        print(f"ERROR: DSL needs inputs not derivable from the definition: {sorted(missing)}\n"
              f"Provide them via --custom-py (full_inputs(axes)) or --fallback-adapter.",
              file=sys.stderr)
        return 2

    from solar.einsum_lib.verify import execute_translated
    from solar.einsum_lib import build_metadata, get_trace
    from solar.einsum_lib.metadata import write_metadata_yaml

    # ── Optional: measured-density harvest (zeros-are-free SOL) ──
    # Full-value interpreted run at the adapter's capped shapes with the
    # density profiler on; densities are harvested by tensor NAME and injected
    # into the original-shape trace below (the dataflow — hence naming — is
    # identical at both scales).
    harvested: dict = {}
    if args.measure_density:
        if not args.fallback_adapter:
            print("ERROR: --measure-density needs --fallback-adapter "
                  "(capped inputs to execute with real values)", file=sys.stderr)
            return 2
        from solar.einsum_lib import set_density_profiling
        amod_d = _load_module(args.fallback_adapter, "_density_adapter")
        cap_in = {}
        for k, v in amod_d.get_inputs().items():
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            cap_in[k] = arr
        set_density_profiling(True, measure_attention=args.measure_attention_density)
        # NOTE: execute_translated's own timeout runs run_model in a worker
        # thread, where the thread-local tracer/profiler record nothing — so
        # the run must stay on THIS thread; time-box it with SIGALRM instead.
        import signal

        def _density_alarm(signum, frame):
            raise TimeoutError(
                f"density run exceeded {args.measure_timeout}s")

        _old_alarm = signal.signal(signal.SIGALRM, _density_alarm)
        signal.alarm(max(1, int(args.measure_timeout)))
        try:
            execute_translated(
                code, {k: v.shape for k, v in cap_in.items()}, all_inputs=cap_in,
                trace=True, shape_only=False, fast=False,
            )
            md_cap = build_metadata(get_trace())
            for name, meta in (md_cap.get("tensors") or {}).items():
                if meta.get("sparsity") is not None:
                    harvested[name] = (
                        float(meta["sparsity"]),
                        tuple((meta.get("dims") or {}).values()),
                    )
        except Exception as exc:  # timeout or heavy interpreted run
            print(f"[density] WARNING: full-value density run failed/timed out "
                  f"({exc}); continuing WITHOUT measured densities", file=sys.stderr)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old_alarm)
            set_density_profiling(False)
        if harvested:
            print("[density] measured at capped shapes: "
                  + ", ".join(f"{n}={d:.4g}" for n, (d, _) in sorted(harvested.items())),
                  file=sys.stderr)

    density_overrides = {}
    for item in args.density_override:
        k, _, v = item.partition("=")
        density_overrides[k.strip()] = float(v)

    execute_translated(
        code, {k: v.shape for k, v in inputs.items()}, all_inputs=inputs,
        trace=True, shape_only=True, fast=False,
    )
    md = build_metadata(get_trace())
    tensors_md = md.get("tensors") or {}

    # ── Density trust ladder ──
    # rung 1 analytic (original-scale, set by the trace itself, e.g. top_k_mask
    # k/E) > overrides > rung 2 shape-invariant measurement > rung 3 auto-
    # revert (re-measure at original shapes when it fits --mem-budget) >
    # rung 4 refuse (stay dense: zeros-are-free only on evidence).
    n_analytic = sum(1 for m in tensors_md.values() if m.get("sparsity") is not None)
    n_shape_inv = n_revalidated = n_refused = n_overridden = 0

    for name, dens in density_overrides.items():
        meta = tensors_md.get(name)
        if meta is None:
            avail = ", ".join(sorted(tensors_md.keys())[:10])
            print(f"[density] WARNING: no tensor named {name!r} in the "
                  f"original-shape trace — override skipped. Names: {avail}",
                  file=sys.stderr)
            continue
        meta["sparsity"] = float(dens)
        meta["sparsity_source"] = "override"
        n_overridden += 1
        print(f"[density] {name}: sparsity={dens:.4g} (override)", file=sys.stderr)

    scale_coupled: dict = {}
    for name, (dens, cap_shape) in harvested.items():
        if name in density_overrides:
            continue
        meta = tensors_md.get(name)
        if meta is None:
            avail = ", ".join(sorted(tensors_md.keys())[:10])
            print(f"[density] WARNING: no tensor named {name!r} in the "
                  f"original-shape trace — skipped. Trace tensor names "
                  f"(inputs may be renamed __const_N): {avail}", file=sys.stderr)
            continue
        if meta.get("sparsity") is not None:
            # rung 1: analytic original-scale density already present
            print(f"[density] {name}: keeping analytic sparsity="
                  f"{meta['sparsity']:.4g} (capped measurement {dens:.4g} "
                  f"discarded)", file=sys.stderr)
            continue
        orig_shape = tuple((meta.get("dims") or {}).values())
        if tuple(cap_shape) == orig_shape:
            meta["sparsity"] = float(dens)
            meta["sparsity_source"] = "measured"
            n_shape_inv += 1
            print(f"[density] {name}: sparsity={dens:.4g} "
                  f"(measured, shape-invariant {orig_shape})", file=sys.stderr)
        else:
            scale_coupled[name] = (dens, cap_shape, orig_shape)

    if scale_coupled:
        # rung 3: auto-revert — re-measure with REAL values at ORIGINAL shapes
        # when the full-value run fits the memory budget.
        est_bytes = 3 * 8 * sum(
            int(np.prod(list((m.get("dims") or {}).values()) or [0]))
            for m in tensors_md.values())
        budget_bytes = args.mem_budget * 1e9
        revalidated: dict = {}
        if est_bytes <= budget_bytes:
            print(f"[density] {len(scale_coupled)} scale-coupled tensor(s); "
                  f"auto-reverting to a NON-CAP full-value run at original "
                  f"shapes (est {est_bytes/1e9:.2f} GB <= budget "
                  f"{args.mem_budget:g} GB)", file=sys.stderr)
            from solar.einsum_lib import set_density_profiling
            rng = np.random.default_rng(0)
            real_inputs = {}
            for k, v in inputs.items():
                if np.issubdtype(v.dtype, np.floating):
                    real_inputs[k] = rng.standard_normal(v.shape).astype(v.dtype)
                else:
                    real_inputs[k] = np.array(v)  # materialize int indices
            import signal

            def _revert_alarm(signum, frame):
                raise TimeoutError(
                    f"original-shape density run exceeded {args.measure_timeout}s")

            _old = signal.signal(signal.SIGALRM, _revert_alarm)
            signal.alarm(max(1, int(args.measure_timeout)))
            set_density_profiling(True, measure_attention=args.measure_attention_density)
            try:
                execute_translated(
                    code, {k: v.shape for k, v in real_inputs.items()},
                    all_inputs=real_inputs, trace=True, shape_only=False,
                    fast=False,
                )
                md_rev = build_metadata(get_trace())
                for name, meta in (md_rev.get("tensors") or {}).items():
                    if meta.get("sparsity") is not None:
                        revalidated[name] = float(meta["sparsity"])
            except Exception as exc:
                print(f"[density] WARNING: original-shape density run failed "
                      f"({exc}); scale-coupled densities will be refused",
                      file=sys.stderr)
                revalidated = None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, _old)
                set_density_profiling(False)
        else:
            print(f"[density] original-shape full-value run needs est "
                  f"{est_bytes/1e9:.2f} GB > budget {args.mem_budget:g} GB — "
                  f"cannot auto-revert", file=sys.stderr)
            revalidated = None

        for name, (dens, cap_shape, orig_shape) in scale_coupled.items():
            meta = tensors_md[name]
            if revalidated is not None:
                rd = revalidated.get(name)
                if rd is not None:
                    meta["sparsity"] = rd
                    meta["sparsity_source"] = "revalidated_original"
                    print(f"[density] {name}: sparsity={rd:.4g} (revalidated "
                          f"at original shapes, non-cap; capped measurement "
                          f"was {dens:.4g})", file=sys.stderr)
                else:
                    print(f"[density] {name}: measured DENSE at original "
                          f"shapes (capped measurement {dens:.4g} discarded; "
                          f"no discount applied)", file=sys.stderr)
                n_revalidated += 1
            else:
                n_refused += 1
                print(
                    "[density] " + "=" * 60 + "\n"
                    f"[density] REFUSED (stays dense): {name}\n"
                    f"[density]   capped-measured density: {dens:.4g} at "
                    f"shape {tuple(cap_shape)}\n"
                    f"[density]   original shape:          {orig_shape}\n"
                    f"[density]   The density is scale-coupled and could not "
                    f"be revalidated at original shapes.\n"
                    f"[density]   Remedy: compute the original-scale density "
                    f"and pass --density-override {name}=VAL\n"
                    "[density] " + "=" * 60, file=sys.stderr)

    if args.measure_density or density_overrides:
        print(f"[density] summary: {n_analytic} analytic / {n_shape_inv} "
              f"shape-invariant / {n_revalidated} revalidated / "
              f"{n_refused} REFUSED (need --density-override) / "
              f"{n_overridden} overridden", file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.basename(args.dsl_py)
    shutil.copy(args.dsl_py, os.path.join(args.out_dir, base))
    yaml_path = os.path.join(args.out_dir, base.replace(".py", ".yaml"))
    from solar.engine_source import engine_source
    md["engine_source"] = engine_source()  # self-binding provenance (119-TRACE-PROVENANCE-1)
    write_metadata_yaml(md, yaml_path)
    print(f"traced at ORIGINAL shapes; axes={axes}")
    print(f"wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
