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

from solar.common.constants import DEFAULT_PRECISION
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

        total = analysis.get("total") or {}
        metadata = analysis.get("metadata") or {}
        
        # Get bytes_per_element from metadata (default to 4 for fp32)
        bytes_per_element = int(metadata.get("bytes_per_element", 4))
        
        total_macs = float(total.get("macs", 0))
        total_flops = float(total.get("flops", 0))
        
        # Parse elements from new format, with fallback to old bytes format
        # New format uses _elements suffix, old format uses _bytes suffix
        if "orojenesis_elements" in total:
            # New format: elements
            total_orojenesis_elems = float(total.get("orojenesis_elements", 0))
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

        # Smart fallback for MAC throughput based on precision
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

        # Compute cycles (same for all models - total compute doesn't change)
        compute_cycles = total_macs / mac_per_cycle if mac_per_cycle > 0 else 0.0
        
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
                "ridge_point": ridge_point,
            },
            "workload": {
                "total_macs": int(total_macs),
                "total_flops": int(total_flops),
                "bytes_per_element": bytes_per_element,
            },
            "unfused": {
                "description": "Each op in isolation, all tensors from DRAM",
                "memory_elements": int(total_orojenesis_elems),
                "memory_bytes": int(total_orojenesis_bytes),
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
