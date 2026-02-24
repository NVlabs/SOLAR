"""Benchmark-oriented einsum conversion.

This module provides `BenchmarkEinsumConverter`, which understands benchmark
directory layouts (kernelbench) and orchestrates per-model conversion:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

using `solar.einsum.pytorch_to_einsum.PyTorchToEinsum`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from solar.common.utils import ensure_directory
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


class BenchmarkEinsumConverter:
    """Convert benchmark suites (kernelbench) to einsum graphs."""

    def __init__(
        self,
        debug: bool = False,
        enable_agent: bool = False,
        api_key: Optional[str] = None,
        cache_dir: str = "./solar_handlers_cache",
    ) -> None:
        self.debug = debug
        self.per_graph = PyTorchToEinsum(
            debug=debug,
            enable_agent=enable_agent,
            api_key=api_key,
            cache_dir=cache_dir,
        )

    def get_output_directories(
        self,
        base_dir: str = "kernelbench_outputs",
        level: Optional[str] = None,
        kernel_ids: Optional[List[int]] = None,
    ) -> List[Path]:
        """Return benchmark output directories containing `pytorch_graph.yaml`."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        target_file = "pytorch_graph.yaml"

        level_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("level")]
        if level:
            level_dirs = [d for d in level_dirs if d.name == level]

        kernel_id_set = {str(k) for k in kernel_ids} if kernel_ids else None
        valid_dirs: List[Path] = []

        for level_dir in level_dirs:
            kernel_dirs = [d for d in level_dir.iterdir() if d.is_dir() and d.name.isdigit()]

            if kernel_id_set is not None:
                kernel_dirs = [d for d in kernel_dirs if d.name in kernel_id_set]

            for kernel_dir in kernel_dirs:
                if (kernel_dir / target_file).exists():
                    valid_dirs.append(kernel_dir)

        return sorted(valid_dirs)

    def convert_directory(
        self,
        dir_path: Path,
        output_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Convert one benchmark output directory into an einsum graph."""
        graph_path = dir_path / "pytorch_graph.yaml"

        if not graph_path.exists():
            if self.debug:
                print(f"Debug: missing graph in {dir_path}")
            return None

        ensure_directory(output_dir)
        return self.per_graph.convert_graph(graph_path, output_dir, copy_graph=True)

    # Backward-compatible alias.
    analyze_directory = convert_directory

    def analyze_kernels(
        self,
        level: Optional[str] = None,
        kernel_ids: Optional[List[int]] = None,
        output_dir: str = "solar_outputs/kernelbench",
        base_dir: str = "kernelbench_outputs",
    ) -> Dict[str, Any]:
        """Convert multiple benchmark directories to einsum graphs."""
        output_base = ensure_directory(output_dir)
        kernel_dirs = self.get_output_directories(
            base_dir=base_dir,
            level=level,
            kernel_ids=kernel_ids,
        )
        if not kernel_dirs:
            return {}

        results: Dict[str, Any] = {}
        for kernel_dir in kernel_dirs:
            kernel_name = f"{kernel_dir.parent.name}_{kernel_dir.name}"
            kernel_out = output_base / kernel_name
            ensure_directory(kernel_out)
            converted = self.convert_directory(kernel_dir, kernel_out)
            if converted is not None:
                results[kernel_name] = converted

        combined = {
            "total_kernels": len(kernel_dirs),
            "successful_conversions": len(results),
            "failed_conversions": len(kernel_dirs) - len(results),
        }
        with open(output_base / "combined_analysis_summary.yaml", "w") as f:
            yaml.dump(combined, f, default_flow_style=False, sort_keys=False)
        return results

    def collect_operation_statistics(
        self,
        kernel_dirs: List[Path],
    ) -> Dict[str, Any]:
        """Collect operation statistics across benchmark directories."""
        stats: Dict[str, Any] = {
            "total_kernels": len(kernel_dirs),
            "total_operations": 0,
            "operation_counts": Counter(),
            "supported_einsum_ops": set(),
            "unsupported_ops": set(),
            "timeloop_runnable_ops": set(),
            "timeloop_failed_ops": set(),
            "kernel_operation_details": {},
            "summary": {},
        }

        target_file = "pytorch_graph.yaml"

        for kernel_dir in kernel_dirs:
            kernel_name = f"{kernel_dir.parent.name}_{kernel_dir.name}"
            graph_path = kernel_dir / target_file
            if not graph_path.exists():
                continue

            try:
                with open(graph_path) as f:
                    model_info = yaml.safe_load(f) or {}

                kernel_ops: List[str] = []
                kernel_supported: List[str] = []
                kernel_timeloop: List[str] = []

                for _, layer_info in (model_info.get("layers") or {}).items():
                    op_type = layer_info.get("type") or layer_info.get("node_type") or ""
                    if not op_type:
                        continue

                    kernel_ops.append(op_type)
                    stats["operation_counts"][op_type] += 1
                    stats["total_operations"] += 1

                    if self._is_op_supported(op_type, layer_info):
                        stats["supported_einsum_ops"].add(op_type)
                        kernel_supported.append(op_type)
                    else:
                        stats["unsupported_ops"].add(op_type)

                    if (kernel_dir / "orojenesis.csv").exists():
                        stats["timeloop_runnable_ops"].add(op_type)
                        kernel_timeloop.append(op_type)
                    else:
                        stats["timeloop_failed_ops"].add(op_type)

                stats["kernel_operation_details"][kernel_name] = {
                    "total_ops": len(kernel_ops),
                    "operations": kernel_ops,
                    "supported_ops": kernel_supported,
                    "timeloop_runnable": kernel_timeloop,
                }
            except Exception:
                if self.debug:
                    print(f"Debug: failed collecting stats for {kernel_name}")
                continue

        stats["summary"] = {
            "total_operations": stats["total_operations"],
            "unique_operation_types": len(stats["operation_counts"]),
            "supported_einsum_ops_count": len(stats["supported_einsum_ops"]),
            "unsupported_ops_count": len(stats["unsupported_ops"]),
            "timeloop_runnable_ops_count": len(stats["timeloop_runnable_ops"]),
            "timeloop_failed_ops_count": len(stats["timeloop_failed_ops"]),
            "einsum_support_rate": (
                len(stats["supported_einsum_ops"]) / len(stats["operation_counts"]) * 100
                if stats["operation_counts"]
                else 0
            ),
            "timeloop_support_rate": (
                len(stats["timeloop_runnable_ops"]) / len(stats["supported_einsum_ops"]) * 100
                if stats["supported_einsum_ops"]
                else 0
            ),
        }
        return stats

    def _extract_basic_shapes(self, layer_info: Dict[str, Any]) -> Dict[str, List[int]]:
        """Extract a minimal ShapeDict from a pytorch_graph.yaml layer entry."""
        shapes: Dict[str, List[int]] = {}
        input_shapes = layer_info.get("input_shapes") or []
        output_shapes = layer_info.get("output_shapes") or []
        weight_shapes = layer_info.get("weight_shapes") or []

        if input_shapes:
            shapes["Input"] = list(input_shapes[0])
            for i, s in enumerate(input_shapes[1:], start=1):
                shapes[f"Input_{i}"] = list(s)

        if output_shapes:
            shapes["Output"] = list(output_shapes[0])

        if weight_shapes:
            shapes["Weight"] = list(weight_shapes[0])

        # Common fallback for matmul-like ops where the second tensor is an input, not a weight.
        if "Weight" not in shapes and "Input_1" in shapes:
            shapes["Weight"] = shapes["Input_1"]

        return shapes

    def _is_op_supported(self, op_type: str, layer_info: Dict[str, Any]) -> bool:
        """Return True if we can generate an einsum op for this layer."""
        shapes = self._extract_basic_shapes(layer_info)
        try:
            op_norm = self.per_graph.einsum_analyzer._get_operation_from_name(str(op_type))
            self.per_graph.einsum_analyzer.get_einsum_op(op_norm, shapes)
            return True
        except Exception:
            return False

    def print_kernel_status(
        self,
        base_dir: str = "kernelbench_outputs",
        level: Optional[str] = None,
        kernel_ids: Optional[List[int]] = None,
    ) -> None:
        """Print a lightweight status report for available graphs."""
        kernel_dirs = self.get_output_directories(
            base_dir=base_dir,
            level=level,
            kernel_ids=kernel_ids,
        )
        if not kernel_dirs:
            print("No model directories found!")
            return

        print("=" * 100)
        print(f"{'Kernel':<20} {'Graph':<10} {'Analysis':<10}")
        print("-" * 100)
        for kernel_dir in kernel_dirs:
            kernel_name = f"{kernel_dir.parent.name}_{kernel_dir.name}"
            has_graph = (kernel_dir / "pytorch_graph.yaml").exists()
            has_analysis = (kernel_dir / "analysis.yaml").exists()
            print(
                f"{kernel_name:<20} "
                f"{('✓' if has_graph else '✗'):<10} "
                f"{('✓' if has_analysis else '✗'):<10}"
            )
        print("=" * 100)


