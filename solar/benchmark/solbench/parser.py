"""Parser for SolBench model files."""

import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SolBenchParser:
    """Parse SolBench model files for Solar analysis.
    
    SolBench files are Python modules containing PyTorch nn.Module definitions
    with metadata in docstrings.
    """

    def __init__(self, solbench_dir: Path):
        """Initialize parser with sol-bench directory.
        
        Args:
            solbench_dir: Path to sol-bench directory containing data/sample/
        """
        self.solbench_dir = Path(solbench_dir)
        self.sample_dir = self.solbench_dir / "data" / "sample"

    def parse_model_file(self, model_path: Path) -> Dict[str, Any]:
        """Parse a single SolBench model file.
        
        Args:
            model_path: Path to the Python model file.
            
        Returns:
            Parsed model info with keys: name, filename, description, 
            op_type, config, module_class, source_code.
        """
        with open(model_path) as f:
            source_code = f.read()
        
        # Parse the docstring at the beginning
        docstring_match = re.search(r'^"""(.*?)"""', source_code, re.DOTALL)
        docstring = docstring_match.group(1) if docstring_match else ""
        
        # Extract metadata from docstring
        metadata = self._parse_docstring(docstring)
        
        # Extract filename info
        filename = model_path.name
        # Format: NNNN_model-name_NN_operation_name.py
        # e.g., 0000_inclusionAI-Ling-flash-2_0_19_mtp_embedding_hidden_concat_project.py
        name_match = re.match(r'^(\d+)_(.+)\.py$', filename)
        if name_match:
            index = name_match.group(1)
            full_name = name_match.group(2)
        else:
            index = "0000"
            full_name = filename.replace('.py', '')
        
        # Find the module class name
        class_match = re.search(r'class\s+(\w+)\s*\(nn\.Module\)', source_code)
        module_class = class_match.group(1) if class_match else "UnknownModule"
        
        return {
            "name": full_name,
            "index": index,
            "filename": filename,
            "path": str(model_path),
            "description": metadata.get("description", ""),
            "op_type": metadata.get("op_type", "fused_op"),
            "priority": metadata.get("priority", "medium"),
            "config": metadata.get("config", {}),
            "optimization_notes": metadata.get("optimization_notes", ""),
            "fusion_opportunities": metadata.get("fusion_opportunities", []),
            "module_class": module_class,
            "source_code": source_code,
        }

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse metadata from model docstring.
        
        Args:
            docstring: The module docstring.
            
        Returns:
            Dict with parsed metadata.
        """
        result = {}
        
        # Extract Subgraph name
        match = re.search(r'Subgraph:\s*(.+)', docstring)
        if match:
            result["subgraph"] = match.group(1).strip()
        
        # Extract Operation Type
        match = re.search(r'Operation Type:\s*(.+)', docstring)
        if match:
            result["op_type"] = match.group(1).strip()
        
        # Extract Priority
        match = re.search(r'Priority:\s*(.+)', docstring)
        if match:
            result["priority"] = match.group(1).strip()
        
        # Extract Description
        match = re.search(r'Description:\s*(.+?)(?=\n\n|Configuration|Optimization|Kernel|$)', 
                         docstring, re.DOTALL)
        if match:
            result["description"] = match.group(1).strip()
        
        # Extract Configuration Constants
        match = re.search(r'Configuration Constants:\s*(\{[^}]+\})', docstring, re.DOTALL)
        if match:
            try:
                config_str = match.group(1).strip()
                result["config"] = ast.literal_eval(config_str)
            except:
                result["config"] = {}
        
        # Extract Optimization Notes
        match = re.search(r'Optimization Notes:\s*(.+?)(?=\n\n|Kernel|$)', docstring, re.DOTALL)
        if match:
            result["optimization_notes"] = match.group(1).strip()
        
        # Extract Kernel Fusion Opportunities
        match = re.search(r'Kernel Fusion Opportunities:\s*(.+?)(?="""|\n\n[A-Z]|$)', 
                         docstring, re.DOTALL)
        if match:
            opportunities_text = match.group(1).strip()
            opportunities = [line.strip().lstrip('- ') 
                           for line in opportunities_text.split('\n') 
                           if line.strip().startswith('-')]
            result["fusion_opportunities"] = opportunities
        
        return result

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all SolBench models.
        
        Returns:
            List of parsed model info dicts.
        """
        if not self.sample_dir.exists():
            return []
        
        models = []
        for model_file in sorted(self.sample_dir.glob("*.py")):
            try:
                model_info = self.parse_model_file(model_file)
                models.append(model_info)
            except Exception as e:
                print(f"Warning: Failed to parse {model_file}: {e}")
        
        return models

    def get_model_by_index(self, index: str) -> Optional[Dict[str, Any]]:
        """Get a model by its index number.
        
        Args:
            index: The index number (e.g., "0000", "0001").
            
        Returns:
            Model info dict or None if not found.
        """
        # Pad index if needed
        index = index.zfill(4)
        
        for model_file in self.sample_dir.glob(f"{index}_*.py"):
            return self.parse_model_file(model_file)
        
        return None

    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a model by its name (partial match).
        
        Args:
            name: Part of the model name to match.
            
        Returns:
            Model info dict or None if not found.
        """
        for model_file in self.sample_dir.glob("*.py"):
            if name in model_file.name:
                return self.parse_model_file(model_file)
        
        return None

    def list_models(self) -> List[Tuple[str, str]]:
        """List all available models.
        
        Returns:
            List of (index, name) tuples.
        """
        models = []
        for model_file in sorted(self.sample_dir.glob("*.py")):
            filename = model_file.name
            match = re.match(r'^(\d+)_(.+)\.py$', filename)
            if match:
                models.append((match.group(1), match.group(2)))
        return models

    def get_model_count(self) -> int:
        """Get total number of models."""
        return len(list(self.sample_dir.glob("*.py")))
