"""Benchmark-oriented model processing utilities.

This module provides `BenchmarkProcessor`, a thin wrapper around
`solar.graph.pytorch_processor.PyTorchProcessor` that understands benchmark-style
directory hierarchies.

Responsibilities:
- Map a benchmark model file path to an output directory (e.g. `level1/55/`)
- Optionally run each model in a subprocess (safe mode)
- Batch processing of directories/levels/kernel IDs

`PyTorchProcessor` itself is intentionally single-model and takes an explicit
`output_dir` for the model being processed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from solar.common.types import ProcessingConfig
from solar.common.utils import ensure_directory, get_file_prefix, setup_safe_environment
from solar.graph.pytorch_processor import PyTorchProcessor


class BenchmarkProcessor:
    """Process benchmark model suites."""

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        """Initialize the benchmark processor.

        Args:
            config: Processing configuration. If None, uses defaults.
        """
        self.config = config or ProcessingConfig()
        self._setup_environment()
        self.model_processor = PyTorchProcessor(self.config)

    def _setup_environment(self) -> None:
        """Set up safe execution environment (thread limits, CPU-only, etc.)."""
        if self.config.safe_mode:
            setup_safe_environment()

    def process_file(self, file_path: str) -> bool:
        """Process a single benchmark model file.

        Args:
            file_path: Path to the Python model file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.config.debug:
                print(f"Processing {file_path}...")

            output_dir = self._prepare_output_directory(file_path)
            return self.model_processor.process_model_file(file_path, str(output_dir))
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            if self.config.debug:
                import traceback

                traceback.print_exc()
            return False

    def process_directory(
        self,
        directory: str,
        level: str = "level1",
        kernel_ids: Optional[List[int]] = None,
    ) -> Dict[str, bool]:
        """Process all benchmark model files in a given level directory.

        Args:
            directory: Repo root directory (contains `kernelbench/`).
            level: Benchmark level to process (e.g. "level1").
            kernel_ids: Optional list of specific kernel IDs to process.

        Returns:
            Dictionary mapping file paths to success status.
        """
        target_dir = Path(directory) / "kernelbench" / level

        if not target_dir.exists():
            print(f"Directory {target_dir} does not exist!")
            return {}

        python_files = list(target_dir.glob("*.py"))
        if kernel_ids:
            python_files = self._filter_by_kernel_ids(python_files, kernel_ids)

        results: Dict[str, bool] = {}
        total = len(python_files)

        for i, file_path in enumerate(python_files, 1):
            print(f"\n[{i}/{total}] Processing {file_path.name}...")
            if self.config.safe_mode:
                success = self._process_file_subprocess(str(file_path))
            else:
                success = self.process_file(str(file_path))
            results[str(file_path)] = success

        successful = sum(results.values())
        failed = len(results) - successful
        print("\n🎉 Processing complete!")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")
        return results

    def _prepare_output_directory(self, file_path: str) -> Path:
        """Prepare output directory for a benchmark model file.

        Output layout:
          <config.output_dir>/<level>/<id>/

        Args:
            file_path: Path to the model file.

        Returns:
            Path to the output directory.
        """
        file_path_obj = Path(file_path)
        prefix = get_file_prefix(file_path_obj.name)
        level = file_path_obj.parent.name

        output_dir = Path(self.config.output_dir) / level / prefix
        ensure_directory(output_dir)

        # Copy source file into the output directory for reproducibility.
        source_copy = output_dir / f"source_{file_path_obj.name}"
        if not source_copy.exists():
            import shutil

            shutil.copy2(file_path, source_copy)

        return output_dir

    def _filter_by_kernel_ids(
        self, files: List[Path], kernel_ids: List[int]
    ) -> List[Path]:
        """Filter benchmark files by kernel IDs."""
        kernel_ids_str = {str(kid) for kid in kernel_ids}
        filtered: List[Path] = []
        for file_path in files:
            prefix = get_file_prefix(file_path.name)
            if prefix in kernel_ids_str:
                filtered.append(file_path)
        return filtered

    def _process_file_subprocess(self, file_path: str) -> bool:
        """Process a benchmark file in a subprocess for safety."""
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
from pathlib import Path

sys.path.insert(0, '{Path(__file__).parent.parent.parent}')

from solar.graph import BenchmarkProcessor
from solar.common.types import ProcessingConfig

config = ProcessingConfig(
    output_dir='{self.config.output_dir}',
    save_graph={self.config.save_graph},
    force_rerun={self.config.force_rerun},
    debug={self.config.debug},
    timeout={self.config.timeout},
    safe_mode=False,
)
processor = BenchmarkProcessor(config)
success = processor.process_file('{file_path}')
sys.exit(0 if success else 1)
""",
        ]

        env = os.environ.copy()
        env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTORCH_DISABLE_CUDA": "1",
            }
        )

        try:
            result = subprocess.run(
                cmd,
                env=env,
                timeout=self.config.timeout,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True

            if self.config.debug:
                print(f"Subprocess failed: {result.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after {self.config.timeout} seconds")
            return False
        except Exception as e:
            print(f"Subprocess error: {e}")
            return False


