"""Parser for FlashInfer trace definitions and workloads."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class FlashInferParser:
    """Parse FlashInfer trace definitions and workloads."""

    def __init__(self, trace_dir: Path):
        """Initialize parser with flashinfer-trace directory.
        
        Args:
            trace_dir: Path to flashinfer-trace directory containing
                       definitions/, workloads/, and traces/ subdirs.
        """
        self.trace_dir = Path(trace_dir)
        self.definitions_dir = self.trace_dir / "definitions"
        self.workloads_dir = self.trace_dir / "workloads"
        self.traces_dir = self.trace_dir / "traces"

    def parse_definition(self, definition_path: Path) -> Dict[str, Any]:
        """Parse a single definition JSON file.
        
        Args:
            definition_path: Path to definition JSON file.
            
        Returns:
            Parsed definition dict with keys: name, description, op_type,
            axes, inputs, outputs, reference.
        """
        with open(definition_path) as f:
            return json.load(f)

    def parse_workloads(self, workload_path: Path) -> List[Dict[str, Any]]:
        """Parse a workloads JSONL file.
        
        Args:
            workload_path: Path to workloads JSONL file.
            
        Returns:
            List of workload dicts, each with keys: definition, workload
            (containing uuid, axes, inputs).
        """
        workloads = []
        with open(workload_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    workloads.append(json.loads(line))
        return workloads

    def get_all_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get all definitions organized by op_type.
        
        Returns:
            Dict mapping op_type -> {name -> definition_dict}
        """
        result = {}
        for op_dir in self.definitions_dir.iterdir():
            if op_dir.is_dir():
                op_type = op_dir.name
                result[op_type] = {}
                for json_file in op_dir.glob("*.json"):
                    defn = self.parse_definition(json_file)
                    result[op_type][defn["name"]] = defn
        return result

    def get_definition_and_workloads(
        self, op_type: str, name: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Get a definition and its associated workloads.
        
        Args:
            op_type: Operation type (e.g., "gemm", "rmsnorm")
            name: Definition name (e.g., "gemm_n128_k2048")
            
        Returns:
            Tuple of (definition_dict, list_of_workloads)
        """
        defn_path = self.definitions_dir / op_type / f"{name}.json"
        workload_path = self.workloads_dir / op_type / f"{name}.jsonl"
        
        definition = self.parse_definition(defn_path)
        workloads = []
        if workload_path.exists():
            workloads = self.parse_workloads(workload_path)
        
        return definition, workloads

    def resolve_shape(
        self, 
        shape_spec: List[str], 
        axes: Dict[str, Any],
        workload_axes: Dict[str, int]
    ) -> List[int]:
        """Resolve a shape specification to concrete dimensions.
        
        Args:
            shape_spec: Shape like ["M", "K"] or ["batch_size", "hidden_size"]
            axes: Axes definition from the definition file
            workload_axes: Concrete axis values from workload
            
        Returns:
            List of concrete integer dimensions
        """
        result = []
        for dim in shape_spec:
            if dim in workload_axes:
                # Variable axis with concrete value from workload
                result.append(workload_axes[dim])
            elif dim in axes:
                axis_info = axes[dim]
                if axis_info.get("type") == "const":
                    result.append(axis_info["value"])
                elif dim in workload_axes:
                    result.append(workload_axes[dim])
                else:
                    raise ValueError(f"Variable axis '{dim}' not in workload_axes")
            else:
                raise ValueError(f"Unknown dimension: {dim}")
        return result

    def list_op_types(self) -> List[str]:
        """List all available operation types."""
        return [d.name for d in self.definitions_dir.iterdir() if d.is_dir()]

    def list_definitions(self, op_type: str) -> List[str]:
        """List all definition names for an operation type."""
        op_dir = self.definitions_dir / op_type
        if not op_dir.exists():
            return []
        return [f.stem for f in op_dir.glob("*.json")]
