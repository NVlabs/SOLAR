"""Generator for SolBench model files for Solar analysis."""

import os
import shutil
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parser import SolBenchParser


class SolBenchGenerator:
    """Generate Solar-compatible model files from SolBench models.
    
    This prepares SolBench Python model files for the Solar pipeline by:
    1. Copying model files to the output directory
    2. Creating metadata files for tracking
    3. Organizing by model index/name
    """

    def __init__(self, solbench_dir: Path, output_dir: Path, debug: bool = False):
        """Initialize generator.
        
        Args:
            solbench_dir: Path to sol-bench directory
            output_dir: Path for generated output files
            debug: Enable debug output
        """
        self.solbench_dir = Path(solbench_dir)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.parser = SolBenchParser(solbench_dir)

    def generate_model(
        self,
        model_info: Dict[str, Any],
        force: bool = False
    ) -> Optional[Path]:
        """Generate Solar-compatible model from SolBench model.
        
        Args:
            model_info: Parsed model info from parser
            force: Overwrite existing files
            
        Returns:
            Path to generated model directory, or None if failed
        """
        model_name = model_info["name"]
        model_index = model_info["index"]
        
        # Create output directory: output_dir/<index>_<model_name>/
        # Include index prefix for easier filtering and organization
        dir_name = f"{model_index}_{model_name}"
        model_output_dir = self.output_dir / dir_name
        
        if model_output_dir.exists() and not force:
            if self.debug:
                print(f"Skipping {model_name}: already exists")
            return model_output_dir
        
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for Solar pipeline
        (model_output_dir / "graph").mkdir(exist_ok=True)
        (model_output_dir / "einsum").mkdir(exist_ok=True)
        (model_output_dir / "analysis").mkdir(exist_ok=True)
        (model_output_dir / "perf").mkdir(exist_ok=True)
        
        # Copy model file with Python 3.8 compatibility fix
        source_file = Path(model_info["path"])
        dest_file = model_output_dir / f"source_{dir_name}.py"
        
        # Read source and add __future__ annotations for Python 3.8 compatibility
        with open(source_file, 'r') as f:
            source_content = f.read()
        
        # Add future annotations import at the very beginning if not already present
        # This allows Python 3.8 to handle Python 3.9+ type hint syntax like tuple[...], list[...]
        if 'from __future__ import annotations' not in source_content:
            # Insert after the docstring if present, otherwise at the beginning
            if source_content.startswith('"""'):
                # Find the end of the docstring
                end_docstring = source_content.find('"""', 3) + 3
                source_content = (
                    source_content[:end_docstring] + 
                    '\n\nfrom __future__ import annotations\n' + 
                    source_content[end_docstring:]
                )
            else:
                source_content = 'from __future__ import annotations\n\n' + source_content
        
        with open(dest_file, 'w') as f:
            f.write(source_content)
        
        # Create metadata file
        metadata = {
            "name": model_name,
            "index": model_index,
            "filename": model_info["filename"],
            "source_file": str(source_file),
            "op_type": model_info["op_type"],
            "priority": model_info["priority"],
            "description": model_info["description"],
            "config": model_info["config"],
            "module_class": model_info["module_class"],
            "optimization_notes": model_info["optimization_notes"],
            "fusion_opportunities": model_info["fusion_opportunities"],
        }
        
        metadata_file = model_output_dir / "metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        
        if self.debug:
            print(f"Generated model: {model_name}")
            print(f"  Output: {model_output_dir}")
            print(f"  Module class: {model_info['module_class']}")
            print(f"  Op type: {model_info['op_type']}")
        
        return model_output_dir

    def generate_all(
        self,
        indices: Optional[List[str]] = None,
        max_models: Optional[int] = None,
        force: bool = False
    ) -> List[Path]:
        """Generate all SolBench models.
        
        Args:
            indices: Optional list of indices to process (e.g., ["0000", "0001"])
            max_models: Maximum number of models to generate
            force: Overwrite existing files
            
        Returns:
            List of generated model directory paths
        """
        models = self.parser.get_all_models()
        
        if indices:
            # Filter by indices
            indices_set = {idx.zfill(4) for idx in indices}
            models = [m for m in models if m["index"] in indices_set]
        
        if max_models:
            models = models[:max_models]
        
        if self.debug:
            print(f"Generating {len(models)} SolBench model(s)...")
        
        generated = []
        for model_info in models:
            try:
                output_dir = self.generate_model(model_info, force=force)
                if output_dir:
                    generated.append(output_dir)
            except Exception as e:
                print(f"Error generating {model_info['name']}: {e}")
        
        if self.debug:
            print(f"Generated {len(generated)} model(s)")
        
        return generated

    def generate_by_index(self, index: str, force: bool = False) -> Optional[Path]:
        """Generate a single model by index.
        
        Args:
            index: Model index (e.g., "0000")
            force: Overwrite existing files
            
        Returns:
            Path to generated model directory
        """
        model_info = self.parser.get_model_by_index(index)
        if model_info:
            return self.generate_model(model_info, force=force)
        else:
            print(f"Model not found: {index}")
            return None

    def generate_by_name(self, name: str, force: bool = False) -> Optional[Path]:
        """Generate a single model by name pattern.
        
        Args:
            name: Part of model name to match
            force: Overwrite existing files
            
        Returns:
            Path to generated model directory
        """
        model_info = self.parser.get_model_by_name(name)
        if model_info:
            return self.generate_model(model_info, force=force)
        else:
            print(f"Model not found: {name}")
            return None

    def list_generated(self) -> List[str]:
        """List all generated model names."""
        if not self.output_dir.exists():
            return []
        
        return [d.name for d in self.output_dir.iterdir() 
                if d.is_dir() and (d / "metadata.yaml").exists()]
