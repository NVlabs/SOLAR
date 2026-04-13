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

"""Performance model for an analyzed einsum graph.

This module implements the **third stage** of the Solar pipeline:

  `analysis.yaml` + `configs/arch/<ARCH>.yaml`  ->  `perf_<ARCH>.yaml`

Three SOL (Speed-of-Light) roofline models are computed:
1. Unfused: Each op runs in isolation, all tensors from DRAM
2. Fused: Per-op roofline, intermediate tensors excluded from memory cost
3. Fused+Prefetched: Single roofline for entire graph, perfect overlap assumed

See SOL_GUIDE.md for detailed explanation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from solar.common.constants import BYTES_PER_ELEMENT, DEFAULT_PRECISION
from solar.common.utils import ensure_directory, NoAliasDumper


PathLike = Union[str, Path]


class EinsumGraphPerfModel:
    """Compute SOL-style roofline predictions from `analysis.yaml`.
    
    Computes three performance models:
    - unfused: Each operation's roofline computed independently, summed
    - fused: Per-op roofline with intermediate tensors excluded
    - fused_prefetched: Single roofline for entire graph (best case)
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def predict(
        self,
        analysis_path: PathLike,
        output_dir: PathLike,
        *,
        arch_config: str = "H100_PCIe",
        precision: str = DEFAULT_PRECISION,
        copy_analysis: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Predict performance and write `perf_<arch>.yaml`.

        Args:
            analysis_path: Path to `analysis.yaml`.
            output_dir: Directory to write perf outputs into.
            arch_config: Architecture name (e.g., "H100_PCIe") or path to a YAML file.
            precision: Precision used for selecting MAC throughput keys.
            copy_analysis: If True, copy analysis into output dir as `analysis.yaml`.

        Returns:
            Perf dict or None on failure.
        """
        analysis_path = Path(analysis_path)
        out_dir = ensure_directory(output_dir)

        if not analysis_path.exists():
            if self.debug:
                print(f"Debug: analysis not found: {analysis_path}")
            return None

        try:
            with open(analysis_path) as f:
                analysis = yaml.safe_load(f) or {}
        except Exception as exc:
            if self.debug:
                print(f"Debug: failed reading analysis: {exc}")
            return None

        if copy_analysis:
            try:
                dst = out_dir / "analysis.yaml"
                if analysis_path.resolve() != dst.resolve():
                    dst.write_text(analysis_path.read_text())
            except Exception:
                if self.debug:
                    print("Debug: failed copying analysis.yaml")

        arch = self._load_arch_config(arch_config)
        if not arch:
            if self.debug:
                print(f"Debug: failed loading arch config: {arch_config}")
            return None

        arch_name = str(arch.get("name") or Path(str(arch_config)).stem or "arch")

        # Check for quantization metadata (e.g., nvfp4/fp8 conversions)
        quant_metadata = self._load_quant_metadata(analysis_path)
        quant_mac_key = None
        quant_bpe = None
        quant_label = None
        if quant_metadata:
            quant_mac_key, quant_bpe, quant_label = self._resolve_quant_overrides(
                quant_metadata, arch, precision
            )

        total = analysis.get("total") or {}
        metadata = analysis.get("metadata") or {}
        
        # Get bytes_per_element: quant override > precision flag > metadata > default
        # The precision flag takes priority over analysis metadata because the user
        # may run perf with a different precision than the analysis was generated with.
        if quant_bpe is not None:
            bytes_per_element = quant_bpe
        elif precision in BYTES_PER_ELEMENT:
            bytes_per_element = BYTES_PER_ELEMENT[precision]
        else:
            bytes_per_element = float(metadata.get("bytes_per_element", 4))
        
        total_macs = float(total.get("macs", 0))
        total_flops = float(total.get("flops", 0))
        
        # Parse elements from new format, with fallback to old bytes format
        # New format uses unfused_elements, old format uses orojenesis_elements or _bytes suffix
        unfused_val = total.get("unfused_elements") or total.get("orojenesis_elements")
        if unfused_val is not None:
            # New format: elements
            total_orojenesis_elems = float(unfused_val)
            total_fused_elems = float(total.get("fused_elements", 0))
            total_fused_prefetched_elems = float(
                total.get("fused_prefetched_elements", total_fused_elems)
            )
            total_weight_elems = float(total.get("weight_elements", 0))
            total_model_io_elems = float(total.get("model_io_elements", 0))
            total_intermediate_elems = float(total.get("intermediate_elements", 0))
            
            # Convert elements to bytes
            total_orojenesis_bytes = total_orojenesis_elems * bytes_per_element
            total_fused_bytes = total_fused_elems * bytes_per_element
            total_fused_prefetched_bytes = total_fused_prefetched_elems * bytes_per_element
            total_weight_bytes = total_weight_elems * bytes_per_element
            total_model_io_bytes = total_model_io_elems * bytes_per_element
            total_intermediate_bytes = total_intermediate_elems * bytes_per_element
        else:
            # Old format: bytes (backward compatibility)
            total_orojenesis_bytes = float(total.get("orojenesis_bytes", 0))
            total_fused_bytes = float(total.get("fused_bytes", 0))
            total_fused_prefetched_bytes = float(
                total.get("fused_prefetched_bytes", total_fused_bytes)
            )
            total_weight_bytes = float(total.get("weight_bytes", 0))
            total_model_io_bytes = float(total.get("model_io_bytes", 0))
            total_intermediate_bytes = float(total.get("intermediate_bytes", 0))
            
            # Convert bytes to elements
            total_orojenesis_elems = total_orojenesis_bytes / bytes_per_element
            total_fused_elems = total_fused_bytes / bytes_per_element
            total_fused_prefetched_elems = total_fused_prefetched_bytes / bytes_per_element
            total_weight_elems = total_weight_bytes / bytes_per_element
            total_model_io_elems = total_model_io_bytes / bytes_per_element
            total_intermediate_elems = total_intermediate_bytes / bytes_per_element

        freq_ghz = float(arch.get("freq_GHz", 1.0))
        dram_bw = float(arch.get("DRAM_byte_per_cycle", 1.0))

        # MAC throughput: quant override takes priority over precision flag
        if quant_mac_key:
            mac_key = quant_mac_key
        else:
            mac_key = f"MAC_per_cycle_{precision}_tc"
        mac_per_cycle = arch.get(mac_key)

        if mac_per_cycle is None:
            # Fallback chain based on precision similarity
            if precision in ["bf16", "bfloat16"]:
                # BF16 has same throughput as FP16 on modern tensor cores
                mac_per_cycle = arch.get("MAC_per_cycle_fp16_tc")
            elif precision in ["fp16", "float16", "half"]:
                # Try FP16, then BF16
                mac_per_cycle = arch.get("MAC_per_cycle_bf16_tc")

            # Final fallback to FP32
            if mac_per_cycle is None:
                mac_per_cycle = arch.get("MAC_per_cycle_fp32_tc", arch.get("MAC_per_cycle_fp32_sm", 1.0))

        mac_per_cycle = float(mac_per_cycle)

        total_other_ops = float(total.get("other_ops", 0))
        sm_per_cycle = float(arch.get("MAC_per_cycle_fp32_sm", 0))

        # Tensor-core cycles (matmul/conv MACs)
        compute_tc_cycles = total_macs / mac_per_cycle if mac_per_cycle > 0 else 0.0
        # SM cycles (elementwise / reduction ops that run on CUDA cores).
        # NOTE: Disabled for SOL computation.  Elementwise/reshape ops are
        # memory-bound in practice — their cost is already captured by
        # fused_memory_cycles.  Including SM cycles here would double-count
        # and inflate SOL by 10-34,000x for elementwise-heavy kernels.
        # SM cycle stats are still reported for informational purposes.
        compute_sm_cycles = total_other_ops / sm_per_cycle if sm_per_cycle > 0 else 0.0
        # SOL compute = tensor-core cycles only
        compute_cycles = compute_tc_cycles
        
        # Memory cycles for each model (using bytes)
        unfused_mem_cycles = total_orojenesis_bytes / dram_bw if dram_bw > 0 else 0.0
        fused_mem_cycles = total_fused_bytes / dram_bw if dram_bw > 0 else 0.0
        fused_prefetched_mem_cycles = total_fused_prefetched_bytes / dram_bw if dram_bw > 0 else 0.0

        # Total cycles (roofline: max of compute and memory)
        unfused_total_cycles = max(compute_cycles, unfused_mem_cycles)
        fused_total_cycles = max(compute_cycles, fused_mem_cycles)
        fused_prefetched_total_cycles = max(compute_cycles, fused_prefetched_mem_cycles)

        # Calculate arithmetic intensity for each model (MACs / bytes)
        unfused_ai = total_macs / total_orojenesis_bytes if total_orojenesis_bytes > 0 else float('inf')
        fused_ai = total_macs / total_fused_bytes if total_fused_bytes > 0 else float('inf')
        fused_prefetched_ai = total_macs / total_fused_prefetched_bytes if total_fused_prefetched_bytes > 0 else float('inf')

        # Ridge point: where compute-bound meets memory-bound
        ridge_point = mac_per_cycle / dram_bw if dram_bw > 0 else 0.0

        perf: Dict[str, Any] = {
            "arch": {
                "name": arch_name,
                "freq_GHz": freq_ghz,
                "DRAM_byte_per_cycle": dram_bw,
                "mac_per_cycle_key": mac_key,
                "MAC_per_cycle": mac_per_cycle,
                "MAC_per_cycle_fp32_sm": sm_per_cycle,
                "ridge_point": ridge_point,
            },
            "workload": {
                "total_macs": int(total_macs),
                "total_other_ops": int(total_other_ops),
                "total_flops": int(total_flops),
                "bytes_per_element": bytes_per_element,
                **({"quant_orig_dtype": quant_label} if quant_label else {}),
            },
            "unfused": {
                "description": "Each op in isolation, all tensors from DRAM",
                "memory_elements": int(total_orojenesis_elems),
                "memory_bytes": int(total_orojenesis_bytes),
                "compute_tc_cycles": int(compute_tc_cycles),
                "compute_sm_cycles": int(compute_sm_cycles),
                "compute_cycles": int(compute_cycles),
                "memory_cycles": int(unfused_mem_cycles),
                "total_cycles": int(unfused_total_cycles),
                "runtime_ms": unfused_total_cycles / (freq_ghz * 1e6) if freq_ghz > 0 else 0.0,
                "arithmetic_intensity": unfused_ai,
                "bottleneck": "compute" if compute_cycles >= unfused_mem_cycles else "memory",
            },
            "fused": {
                "description": "Per-op roofline, intermediate tensors excluded",
                "memory_elements": int(total_fused_elems),
                "memory_bytes": int(total_fused_bytes),
                "compute_tc_cycles": int(compute_tc_cycles),
                "compute_sm_cycles": int(compute_sm_cycles),
                "compute_cycles": int(compute_cycles),
                "memory_cycles": int(fused_mem_cycles),
                "total_cycles": int(fused_total_cycles),
                "runtime_ms": fused_total_cycles / (freq_ghz * 1e6) if freq_ghz > 0 else 0.0,
                "arithmetic_intensity": fused_ai,
                "bottleneck": "compute" if compute_cycles >= fused_mem_cycles else "memory",
            },
            "fused_prefetched": {
                "description": "Single roofline for entire graph, perfect overlap",
                "memory_elements": int(total_fused_prefetched_elems),
                "memory_bytes": int(total_fused_prefetched_bytes),
                "compute_tc_cycles": int(compute_tc_cycles),
                "compute_sm_cycles": int(compute_sm_cycles),
                "compute_cycles": int(compute_cycles),
                "memory_cycles": int(fused_prefetched_mem_cycles),
                "total_cycles": int(fused_prefetched_total_cycles),
                "runtime_ms": fused_prefetched_total_cycles / (freq_ghz * 1e6) if freq_ghz > 0 else 0.0,
                "arithmetic_intensity": fused_prefetched_ai,
                "bottleneck": "compute" if compute_cycles >= fused_prefetched_mem_cycles else "memory",
            },
            "memory_breakdown": {
                "weight_elements": int(total_weight_elems),
                "weight_bytes": int(total_weight_bytes),
                "model_io_elements": int(total_model_io_elems),
                "model_io_bytes": int(total_model_io_bytes),
                "intermediate_elements": int(total_intermediate_elems),
                "intermediate_bytes": int(total_intermediate_bytes),
            },
            "speedup": {
                "fused_vs_unfused": (unfused_total_cycles / fused_total_cycles) if fused_total_cycles > 0 else 1.0,
                "fused_prefetched_vs_unfused": (unfused_total_cycles / fused_prefetched_total_cycles) if fused_prefetched_total_cycles > 0 else 1.0,
                "fused_prefetched_vs_fused": (fused_total_cycles / fused_prefetched_total_cycles) if fused_prefetched_total_cycles > 0 else 1.0,
            },
            "memory_reduction": {
                "fused_vs_unfused": 1.0 - (total_fused_bytes / total_orojenesis_bytes) if total_orojenesis_bytes > 0 else 0.0,
                "fused_prefetched_vs_unfused": 1.0 - (total_fused_prefetched_bytes / total_orojenesis_bytes) if total_orojenesis_bytes > 0 else 0.0,
            },
        }

        out_path = out_dir / f"perf_{arch_name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(perf, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            print(f"✅ Wrote perf: {out_path}")

        return perf

    # Maps orig_dtypes keywords from metadata.yaml to (precision_key, bytes_per_element)
    _QUANT_DTYPE_MAP = {
        "nvfp4": ("nvfp4", 0.5),
        "float4_e2m1fn_x2": ("nvfp4", 0.5),
        "fp4": ("nvfp4", 0.5),
        "fp8": ("fp8", 1),
        "float8_e4m3fn": ("fp8", 1),
        "float8_e5m2": ("fp8", 1),
        "float8_e4m3fnuz": ("fp8", 1),
        "float8_e5m2fnuz": ("fp8", 1),
    }

    def _load_quant_metadata(
        self, analysis_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Search for metadata.yaml near the analysis path.

        Typical layout::

            <model_output>/metadata.yaml
            <model_output>/analysis/analysis.yaml   <-- analysis_path

        We walk up from ``analysis_path`` checking each parent for
        ``metadata.yaml`` (max 3 levels).
        """
        search_dir = analysis_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    return None
            search_dir = search_dir.parent
        return None

    def _resolve_quant_overrides(
        self,
        metadata: Dict[str, Any],
        arch: Dict[str, Any],
        precision: str,
    ) -> tuple:
        """Derive MAC key and bytes_per_element from quantization metadata.

        Scans ``dtype_conversions`` for the *highest-throughput* original
        quantized dtype (nvfp4 > fp8).  Returns the corresponding
        ``(mac_per_cycle_key, bytes_per_element, orig_dtype_label)`` or
        ``(None, None, None)`` if no quantized dtypes are found.
        """
        conversions = metadata.get("dtype_conversions") or []
        if not conversions:
            return None, None, None

        # Priority: nvfp4 > fp8  (pick highest throughput)
        best_precision = None
        best_bpe = None
        best_label = None

        for conv in conversions:
            orig = str(conv.get("orig_dtypes", "")).lower()
            for keyword, (prec, bpe) in self._QUANT_DTYPE_MAP.items():
                if keyword in orig:
                    if best_precision is None or bpe < best_bpe:
                        best_precision = prec
                        best_bpe = bpe
                        best_label = orig
                    break

        if best_precision is None:
            return None, None, None

        mac_key = f"MAC_per_cycle_{best_precision}_tc"
        if mac_key not in arch:
            if self.debug:
                print(f"Debug: arch config missing {mac_key}, falling back to precision={precision}")
            return None, None, None

        if self.debug:
            print(f"  Quant override: orig_dtype={best_label} -> "
                  f"mac_key={mac_key}, bytes_per_element={best_bpe}")

        return mac_key, best_bpe, best_label

    def _load_arch_config(self, arch_config: str) -> Dict[str, Any]:
        """Load an architecture YAML by name or path."""
        # Explicit path.
        cfg_path = Path(arch_config)
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}

        # Look under solar root: solar/configs/arch/<name>.yaml
        solar_root = Path(__file__).resolve().parents[2]
        candidate = solar_root / "configs" / "arch" / f"{arch_config}.yaml"
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f) or {}

        # Fallback: return empty.
        return {}


__all__ = ["EinsumGraphPerfModel"]
