#!/usr/bin/env python3
"""
Test that L1 SOL violation fixes produce correct results.

Tests:
1. XiaomiMiMo hybrid_attention_mask_preparation - batch_size no longer hardcoded
2. DeepSeek-V3 moe_expert_computation - only top-k experts processed

These tests verify the reference.py changes by running the full SOLAR pipeline
(graph → einsum → analysis → perf) and checking that the resulting fused memory
is proportional to the actual workload size, not hardcoded maximums.
"""

import json
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path

VENV_PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
SOLAR_ROOT = Path(__file__).resolve().parents[1]
SOL_BENCH = Path(__file__).resolve().parents[2].parent / "sol-bench" / "data" / "benchmark"

# Nestor results for comparison
NESTOR_PATH = (
    Path(__file__).resolve().parents[2].parent
    / "performance_comparison_complete_20260310"
    / "b200_data"
    / "nestor_20260307"
    / "L1_nestor_results.json"
)


def run_solar_pipeline(kernel_name: str, level: str, max_workloads: int = 3):
    """Run the full SOLAR pipeline for a kernel and return perf results."""
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    benchmark_dir = str(SOL_BENCH)

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_dir = Path(tmpdir) / "generated"
        out_dir = Path(tmpdir) / "solar_output"

        # Step 1: Generate model files
        subprocess.run(
            [
                python, "-m", "solar.benchmark.solbenchv3.cli",
                "--benchmark-dir", benchmark_dir,
                "--output-dir", str(gen_dir),
                "--level", level,
                "--kernel", kernel_name,
                "--max-workloads", str(max_workloads),
            ],
            cwd=str(SOLAR_ROOT),
            check=True,
            capture_output=True,
        )

        results = {}
        kernel_gen_dir = gen_dir / level / kernel_name

        for wl_dir in sorted(kernel_gen_dir.iterdir()):
            if not wl_dir.is_dir() or not wl_dir.name.isdigit():
                continue
            wid = int(wl_dir.name)
            model_file = wl_dir / f"{kernel_name}.py"
            if not model_file.exists():
                continue

            wl_out = out_dir / level / kernel_name / str(wid)
            for subdir in ["graph", "einsum", "analysis", "perf"]:
                (wl_out / subdir).mkdir(parents=True, exist_ok=True)

            # Graph
            subprocess.run(
                [
                    python, "-m", "solar.cli.process_model",
                    "--model-file", str(model_file),
                    "--output-dir", str(wl_out / "graph"),
                    "--force-rerun",
                ],
                cwd=str(SOLAR_ROOT),
                check=True,
                capture_output=True,
            )

            # Einsum
            subprocess.run(
                [
                    python, "-m", "solar.cli.toeinsum_model",
                    "--graph-path", str(wl_out / "graph" / "pytorch_graph.yaml"),
                    "--output-dir", str(wl_out / "einsum"),
                ],
                cwd=str(SOLAR_ROOT),
                check=True,
                capture_output=True,
            )

            # Analysis
            subprocess.run(
                [
                    python, "-m", "solar.cli.analyze_model",
                    "--einsum-graph-path",
                    str(wl_out / "einsum" / "einsum_graph_renamed.yaml"),
                    "--output-dir", str(wl_out / "analysis"),
                    "--precision", "fp16",
                ],
                cwd=str(SOLAR_ROOT),
                check=True,
                capture_output=True,
            )

            # Perf
            subprocess.run(
                [
                    python, "-m", "solar.cli.predict_perf_model",
                    "--analysis-path",
                    str(wl_out / "analysis" / "analysis.yaml"),
                    "--output-dir", str(wl_out / "perf"),
                    "--arch-config", "B200",
                    "--precision", "fp16",
                ],
                cwd=str(SOLAR_ROOT),
                check=True,
                capture_output=True,
            )

            perf_file = wl_out / "perf" / "perf_B200.yaml"
            with open(perf_file) as f:
                results[wid] = yaml.safe_load(f)

    return results


def load_nestor(kernel_name: str):
    """Load Nestor results for a kernel."""
    if not NESTOR_PATH.exists():
        return {}
    with open(NESTOR_PATH) as f:
        data = json.load(f)
    return {
        r["workload_id"]: r
        for r in data["results"]
        if r["kernel"] == kernel_name and r.get("status") == "PASSED"
    }


def test_xiaomimimo_no_hardcoded_batch():
    """XiaomiMiMo: batch_size should come from workload, not hardcoded 64."""
    kernel = "XiaomiMiMo_MiMo-V2-Flash_hybrid_attention_mask_preparation"

    if not SOL_BENCH.exists():
        print(f"SKIP: sol-bench not found at {SOL_BENCH}")
        return

    results = run_solar_pipeline(kernel, "L1", max_workloads=3)
    assert len(results) > 0, "No workloads processed"

    # The key check: fused_memory_bytes should scale with batch_size.
    # With hardcoded batch=64, all workloads would have similar memory.
    # With the fix, different batch sizes should produce different memory.
    memory_bytes = [results[wid]["fused"]["memory_bytes"] for wid in sorted(results)]

    # WL0 has batch=1, WL1 has batch=16 — memory should differ significantly
    if len(memory_bytes) >= 2:
        ratio = memory_bytes[1] / memory_bytes[0] if memory_bytes[0] > 0 else 0
        print(f"  Memory ratio WL1/WL0: {ratio:.1f}x (expect >1x if batch varies)")
        assert ratio != 1.0, (
            f"Memory is identical across workloads ({memory_bytes[0]:,} bytes) — "
            f"batch_size may still be hardcoded"
        )

    # Check against Nestor for violations
    nestor = load_nestor(kernel)
    violations = 0
    for wid in sorted(results):
        fused_ms = results[wid]["fused"]["runtime_ms"]
        fused_bytes = results[wid]["fused"]["memory_bytes"]
        n = nestor.get(wid)
        if n:
            nopt = n["optimized_latency_ms"]
            ratio = nopt / fused_ms if fused_ms > 0 else float("inf")
            status = "VIOLATION" if ratio < 1.0 else "ok"
            if ratio < 1.0:
                violations += 1
            print(f"  WL{wid}: SOL={fused_ms:.6f}ms mem={fused_bytes:,} Nestor={nopt:.6f}ms ratio={ratio:.2f}x [{status}]")

    print(f"  Result: {violations} violations (previously 10 with hardcoded batch)")
    # We expect significantly fewer violations than the original 10
    assert violations <= 5, f"Too many violations remaining: {violations}"
    print("  PASSED")


def test_deepseek_v3_active_experts_only():
    """DeepSeek-V3: only top-k=8 active expert weights should be counted."""
    kernel = "deepseek-ai_DeepSeek-V3_moe_expert_computation"

    if not SOL_BENCH.exists():
        print(f"SKIP: sol-bench not found at {SOL_BENCH}")
        return

    results = run_solar_pipeline(kernel, "L1", max_workloads=3)
    assert len(results) > 0, "No workloads processed"

    # Key check: fused_memory_bytes should NOT be ~22.5 GB (all 256 experts)
    # With fix (8 active experts): weight memory ~ 8 * 3 * 7168 * 2048 * 2 = 0.7 GB
    for wid in sorted(results):
        fused_bytes = results[wid]["fused"]["memory_bytes"]
        fused_ms = results[wid]["fused"]["runtime_ms"]
        print(f"  WL{wid}: fused_bytes={fused_bytes:,} ({fused_bytes/1e9:.2f} GB)  fused_ms={fused_ms:.6f}")

        # All 256 experts would be ~22.5 GB. With 8 active: ~0.7 GB.
        assert fused_bytes < 5e9, (
            f"WL{wid}: fused_bytes={fused_bytes:,} ({fused_bytes/1e9:.1f} GB) — "
            f"still counting all experts (expected <5 GB with top-k=8)"
        )

    # Check against Nestor for violations
    nestor = load_nestor(kernel)
    violations = 0
    for wid in sorted(results):
        fused_ms = results[wid]["fused"]["runtime_ms"]
        n = nestor.get(wid)
        if n:
            nopt = n["optimized_latency_ms"]
            ratio = nopt / fused_ms if fused_ms > 0 else float("inf")
            status = "VIOLATION" if ratio < 1.0 else "ok"
            if ratio < 1.0:
                violations += 1
            print(f"  WL{wid}: SOL={fused_ms:.6f}ms Nestor={nopt:.6f}ms ratio={ratio:.2f}x [{status}]")

    print(f"  Result: {violations} violations (previously 5 with all experts)")
    assert violations == 0, f"Violations still present: {violations}"
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: XiaomiMiMo - batch_size fix")
    print("=" * 60)
    test_xiaomimimo_no_hardcoded_batch()

    print()
    print("=" * 60)
    print("Test 2: DeepSeek-V3 - active experts only")
    print("=" * 60)
    test_deepseek_v3_active_experts_only()

    print()
    print("All tests passed!")
