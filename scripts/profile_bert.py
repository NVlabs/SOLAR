#!/usr/bin/env python3
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

"""Profile the Solar pipeline steps to identify slowdowns.

Run on your remote VM:
    cd ~/llm4arch/solar
    python3 scripts/profile_bert.py

This will show you:
1. Import times for different modules
2. Time for actual performance prediction work
3. Overhead from Python startup + imports
"""

import sys
import time
from pathlib import Path

# Add solar to path
SCRIPT_DIR = Path(__file__).resolve().parent
SOLAR_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SOLAR_ROOT))


def time_imports():
    """Time the import phase - this is often the slowest part."""
    print("=" * 60)
    print("PROFILING IMPORT TIMES")
    print("=" * 60)
    
    timings = {}
    
    # Time lightweight imports first
    start = time.time()
    import yaml
    timings['yaml'] = time.time() - start
    
    # Time importing just the perf module (lightweight path)
    start = time.time()
    from solar.perf.perf_model import EinsumGraphPerfModel  # Direct import!
    timings['EinsumGraphPerfModel (direct)'] = time.time() - start
    
    # Now time the heavy imports
    print("\nNow importing heavy modules...")
    
    try:
        start = time.time()
        import networkx
        timings['networkx'] = time.time() - start
    except ImportError as e:
        timings['networkx'] = f"Not installed: {e}"
    
    try:
        start = time.time()
        import torch
        timings['torch'] = time.time() - start
    except ImportError as e:
        timings['torch'] = f"Not installed: {e}"
    
    try:
        start = time.time()
        from solar.einsum import PyTorchToEinsum  # This triggers heavy imports
        timings['PyTorchToEinsum'] = time.time() - start
    except Exception as e:
        timings['PyTorchToEinsum'] = f"ERROR: {e}"
    
    try:
        start = time.time()
        from solar.analysis import ModelAnalyzer
        timings['ModelAnalyzer'] = time.time() - start
    except Exception as e:
        timings['ModelAnalyzer'] = f"ERROR: {e}"
    
    # Time importing full solar package (worst case)
    try:
        # Force reimport by removing from cache
        for key in list(sys.modules.keys()):
            if key.startswith('solar') and key != 'solar.perf.perf_model':
                pass  # Can't easily reimport, so skip
        
        start = time.time()
        import solar  # Full package import
        timings['import solar (full)'] = time.time() - start
    except Exception as e:
        timings['import solar (full)'] = f"ERROR: {e}"
    
    print("\nImport times:")
    for name, t in sorted(timings.items(), key=lambda x: (not isinstance(x[1], float), x[1] if isinstance(x[1], float) else 0)):
        if isinstance(t, float):
            status = "⚡ FAST" if t < 0.5 else "🐢 SLOW" if t > 2.0 else ""
            print(f"  {name}: {t:.4f}s {status}")
        else:
            print(f"  {name}: {t}")
    
    return timings


def time_perf_prediction():
    """Time just the performance prediction step."""
    print("\n" + "=" * 60)
    print("PROFILING PERFORMANCE PREDICTION")
    print("=" * 60)
    
    from solar.perf import EinsumGraphPerfModel
    
    bert_analysis = SOLAR_ROOT / "examples/BERT/output/analysis/analysis.yaml"
    output_dir = Path("/tmp/bert_perf_profile")
    output_dir.mkdir(exist_ok=True)
    
    if not bert_analysis.exists():
        print(f"ERROR: BERT analysis not found at {bert_analysis}")
        print("Run the BERT example first: bash solar/examples/BERT/run_solar.sh")
        return None
    
    model = EinsumGraphPerfModel(debug=True)
    
    # Time single prediction
    start = time.time()
    perf = model.predict(
        bert_analysis,
        output_dir,
        arch_config="H100_PCIe",
        precision="fp32"
    )
    single_time = time.time() - start
    
    print(f"\nSingle prediction time: {single_time:.4f}s")
    
    # Time multiple predictions (to see amortized cost)
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        model.predict(
            bert_analysis,
            output_dir,
            arch_config="H100_PCIe",
            precision="fp32"
        )
    total_time = time.time() - start
    
    print(f"{num_runs} predictions time: {total_time:.4f}s ({total_time/num_runs:.4f}s avg)")
    
    return {
        'single': single_time,
        'average': total_time / num_runs
    }


def time_full_cli_invocation():
    """Time a full CLI invocation (includes Python startup)."""
    print("\n" + "=" * 60)
    print("PROFILING FULL CLI INVOCATION (with Python startup)")
    print("=" * 60)
    
    import subprocess
    
    bert_analysis = SOLAR_ROOT / "examples/BERT/output/analysis/analysis.yaml"
    output_dir = Path("/tmp/bert_perf_profile_cli")
    
    if not bert_analysis.exists():
        print(f"ERROR: BERT analysis not found")
        return None
    
    # Standard CLI invocation (loads full solar package)
    cmd_standard = [
        sys.executable, "-m", "solar.cli.predict_perf_model",
        "--analysis-path", str(bert_analysis),
        "--output-dir", str(output_dir),
        "--arch-config", "H100_PCIe",
        "--precision", "fp32"
    ]
    
    # Fast CLI invocation (direct module run)
    fast_script = '''
import sys
sys.path.insert(0, "{}")
from solar.perf.perf_model import EinsumGraphPerfModel
from pathlib import Path
model = EinsumGraphPerfModel()
model.predict(Path("{}"), Path("{}"), arch_config="H100_PCIe", precision="fp32")
'''.format(SOLAR_ROOT, bert_analysis, output_dir)
    
    cmd_fast = [sys.executable, "-c", fast_script]
    
    print("\n1. Standard CLI (python -m solar.cli.predict_perf_model):")
    
    # Time standard CLI
    start = time.time()
    result = subprocess.run(cmd_standard, capture_output=True, text=True, cwd=str(SOLAR_ROOT))
    standard_time = time.time() - start
    
    if result.returncode != 0:
        print(f"   CLI failed: {result.stderr[:200]}...")
        standard_time = None
    else:
        print(f"   Single run: {standard_time:.4f}s")
    
    print("\n2. Fast direct import (bypasses solar/__init__.py):")
    
    # Time fast approach
    start = time.time()
    result = subprocess.run(cmd_fast, capture_output=True, text=True)
    fast_time = time.time() - start
    
    if result.returncode != 0:
        print(f"   Failed: {result.stderr[:200]}...")
        fast_time = None
    else:
        print(f"   Single run: {fast_time:.4f}s")
    
    if standard_time and fast_time:
        speedup = standard_time / fast_time
        overhead = standard_time - fast_time
        print(f"\n   Speedup: {speedup:.1f}x")
        print(f"   Overhead from full imports: {overhead:.2f}s per invocation")
    
    return {
        'standard': standard_time,
        'fast': fast_time
    }


def main():
    print("=" * 60)
    print("SOLAR PERFORMANCE PROFILER")
    print("=" * 60)
    print(f"SOLAR_ROOT: {SOLAR_ROOT}")
    print(f"Python: {sys.executable}")
    print()
    
    # Profile imports
    import_times = time_imports()
    
    # Profile performance prediction
    perf_times = time_perf_prediction()
    
    # Profile full CLI
    cli_times = time_full_cli_invocation()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    if cli_times and cli_times.get('standard') and cli_times.get('fast'):
        overhead = cli_times['standard'] - cli_times['fast']
        print(f"\nImport overhead per CLI call: {overhead:.2f}s")
        
        # Extrapolate to 100 kernels with 4 steps each
        est_overhead = overhead * 4 * 100  # 4 steps per kernel, 100 kernels
        print(f"\nFor 100 kernels × 4 CLI steps = 400 subprocess calls:")
        print(f"  Total overhead: {est_overhead:.0f}s = {est_overhead/60:.1f} minutes!")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS TO SPEED UP:")
        print("=" * 60)
        
        if overhead > 1.0:
            print("""
1. QUICK FIX: Use lazy imports in solar/__init__.py
   The main package imports everything eagerly, even when CLIs
   only need a small subset. Making imports lazy would help.

2. BETTER: Batch processing mode
   Process multiple kernels in a single Python process to
   amortize import overhead. Example:
   
   python3 -c "
   from solar.perf.perf_model import EinsumGraphPerfModel
   model = EinsumGraphPerfModel()
   for kernel_dir in glob('output_kernelbench/level1/*'):
       analysis = f'{kernel_dir}/analysis/analysis.yaml'
       if os.path.exists(analysis):
           model.predict(analysis, f'{kernel_dir}/perf', ...)
   "

3. BEST: Single-process pipeline
   Modify run_kernelbench.sh to call a Python script that
   processes all kernels in one process instead of spawning
   a new Python process for each step.
""")
        else:
            print("  Import overhead is acceptable (<1s per call).")
            print("  Slowness may be due to actual computation.")
    
    if perf_times:
        print(f"\nActual prediction work: {perf_times.get('single', 0):.4f}s per kernel")
        print("  (This is the minimum time per kernel, excluding imports)")


if __name__ == "__main__":
    main()
