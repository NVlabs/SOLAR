"""Rename dimension ranks in einsum graphs using BFS traversal.

This module provides functionality to rename all dimension labels consistently
using breadth-first traversal, ensuring that connected tensors share dimension
labels where appropriate by propagating dimension names from producers to
consumers.

Example:
    >>> from solar.einsum.einsum_rank_renamer import EinsumRankRenamer
    >>> renamer = EinsumRankRenamer()
    >>> renamed = renamer.rename(graph_dict, "output/einsum_graph_renamed.yaml")

The renaming algorithm:
1. Find all start nodes (nodes with no predecessors)
2. Perform BFS from start nodes
3. For each node:
   - Parse the existing einsum equation
   - Map input tokens to predecessor output labels
   - Assign fresh labels to unmapped tokens
   - Propagate output labels to successors
"""

from __future__ import annotations

import string
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from solar.common.utils import (
    ensure_directory,
    load_einsum_graph_to_networkx,
    NoAliasDumper,
    parse_einsum_equation,
)


PathLike = Union[str, Path]


def build_equation(
    input_operands: List[List[str]],
    output_tokens: List[str],
) -> str:
    """Build an einsum equation string from token lists.
    
    Args:
        input_operands: List of input operand token lists.
        output_tokens: Output dimension tokens.
        
    Returns:
        Einsum equation string (e.g., "ABC,DE->ADE").
        
    Example:
        >>> build_equation([["A", "B"], ["C", "D"]], ["A", "C"])
        'AB,CD->AC'
    """
    if not output_tokens:
        return ""
    
    lhs_parts = ["".join(operand) for operand in input_operands]
    lhs = ",".join(lhs_parts)
    rhs = "".join(output_tokens)
    
    return f"{lhs}->{rhs}"


class LabelGenerator:
    """Generate fresh dimension labels sequentially.
    
    Generates labels in order:
    1. Single letters: A, B, C, ..., Z (26 labels)
    2. Letter + 0: A0, B0, C0, ..., Z0 (26 labels)
    3. Letter + 1: A1, B1, C1, ..., Z1 (26 labels)
    4. And so on with incrementing integers: A10, B10, ..., Z10, A11, ...
    
    This format ensures labels are always a single capital letter optionally
    followed by an integer, providing unlimited unique labels.
    
    Note: Multi-letter prefixes (e.g., AA0, AB1) are NOT allowed.
    """
    
    def __init__(self) -> None:
        """Initialize the label generator."""
        self._counter = 0
    
    def _index_to_label(self, idx: int) -> str:
        """Convert a numeric index to a label string.
        
        Args:
            idx: Zero-based index.
            
        Returns:
            Label string:
            - 0-25: A, B, C, ..., Z
            - 26-51: A0, B0, C0, ..., Z0
            - 52-77: A1, B1, C1, ..., Z1
            - 78-103: A2, B2, C2, ..., Z2
            - ... and so on with incrementing integers
        """
        # First 26 labels are single letters A-Z
        if idx < 26:
            return string.ascii_uppercase[idx]
        
        # After that, use letter + integer format
        # idx 26 -> A0, idx 27 -> B0, ..., idx 51 -> Z0
        # idx 52 -> A1, idx 53 -> B1, ..., idx 77 -> Z1
        # idx 78 -> A2, ...
        idx_after_letters = idx - 26
        letter_idx = idx_after_letters % 26
        number = idx_after_letters // 26  # 0, 1, 2, 3, ...
        
        return f"{string.ascii_uppercase[letter_idx]}{number}"
    
    def next(self, count: int = 1) -> List[str]:
        """Get the next `count` fresh dimension labels.
        
        Args:
            count: Number of labels to generate.
            
        Returns:
            List of fresh dimension labels.
        """
        labels = []
        for _ in range(count):
            labels.append(self._index_to_label(self._counter))
            self._counter += 1
        return labels
    
    def reset(self) -> None:
        """Reset the label counter."""
        self._counter = 0


class EinsumRankRenamer:
    """Rename dimension ranks in einsum graphs using BFS traversal.
    
    This class provides methods to rename all dimension labels in an einsum
    graph consistently, propagating labels from producers to consumers.
    
    Attributes:
        debug: Whether to print debug information.
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the renamer.
        
        Args:
            debug: Enable debug output.
        """
        self._debug = debug

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    def rename(
        self,
        graph_dict: Dict[str, Any],
        output_path: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """Rename all dimension labels in the einsum graph using BFS.
        
        This method:
        1. Finds all start nodes (nodes with no predecessors)
        2. Performs BFS from start nodes
        3. Renames dimension labels by:
           - Parsing the existing einsum equation
           - Building a mapping from old tokens to new tokens
           - Propagating output tokens from producers to consumers

        Args:
            graph_dict: The einsum graph dictionary (with 'layers' key).
            output_path: Optional path to write the renamed graph.

        Returns:
            The renamed graph dictionary.
            
        Raises:
            ValueError: If graph_dict is invalid.
        """
        if not isinstance(graph_dict, dict) or "layers" not in graph_dict:
            raise ValueError("Invalid einsum graph format: missing 'layers' key")

        layers = graph_dict.get("layers", {})
        graph = load_einsum_graph_to_networkx(layers)
        
        label_gen = LabelGenerator()
        node_output_labels: Dict[str, List[str]] = {}
        processed: Set[str] = set()
        
        # Find start nodes
        start_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        
        if self._debug:
            print(f"Start nodes: {start_nodes}")

        # BFS traversal
        queue = deque(start_nodes)
        
        while queue:
            node_id = queue.popleft()
            
            if node_id in processed:
                continue
            
            # Ensure all predecessors are processed
            predecessors = list(graph.predecessors(node_id))
            if not all(p in processed for p in predecessors):
                queue.append(node_id)
                continue
            
            processed.add(node_id)
            
            # Process this node
            new_output_labels = self._process_node(
                node_id, layers, node_output_labels, label_gen
            )
            node_output_labels[node_id] = new_output_labels
            
            # Add successors to queue
            for succ in graph.successors(node_id):
                if succ not in processed:
                    queue.append(succ)

        # Write output if path provided
        if output_path:
            out_path = Path(output_path)
            ensure_directory(out_path.parent)
            with open(out_path, "w") as f:
                yaml.dump(
                    graph_dict, f,
                    Dumper=NoAliasDumper,
                    sort_keys=False,
                    default_flow_style=False
                )
            if self._debug:
                print(f"✅ Wrote renamed einsum graph: {out_path}")

        return graph_dict

    # Backward compatibility alias
    rename_ranks_bfs = rename

    def _process_node(
        self,
        node_id: str,
        layers: Dict[str, Any],
        node_output_labels: Dict[str, List[str]],
        label_gen: LabelGenerator,
    ) -> List[str]:
        """Process a single node and return its new output labels."""
        node_data = layers.get(node_id, {})
        old_equation = node_data.get("einsum_equation", "")
        shapes = node_data.get("shapes", {})
        
        if self._debug:
            print(f"Processing {node_id}: eq={old_equation}")
        
        # Parse existing equation
        input_operands, output_tokens = parse_einsum_equation(old_equation)
        
        # Determine output dimensions
        output_shape = shapes.get("Output", [])
        num_output_dims = len(output_shape) if output_shape else len(output_tokens)
        
        # Build token rename mapping
        token_map: Dict[str, str] = {}
        connections = node_data.get("connections", {})
        input_node_ids = connections.get("inputs", [])
        
        # First, identify which tokens are shared across multiple operands
        # vs unique to a single operand. Shared tokens should be mapped from
        # predecessors; unique tokens should get fresh labels to avoid collisions.
        token_operand_count: Dict[str, int] = {}
        for operand in input_operands:
            for token in operand:
                token_operand_count[token] = token_operand_count.get(token, 0) + 1
        
        # Also count output tokens
        for token in output_tokens:
            token_operand_count[token] = token_operand_count.get(token, 0) + 1
        
        # Map input operands to predecessor output labels
        # Only map tokens that appear in multiple operands (shared dimensions)
        # or in output (preserved dimensions). Unique tokens get fresh labels.
        used_labels: Set[str] = set()
        
        for i, pred_id in enumerate(input_node_ids):
            if pred_id in node_output_labels and i < len(input_operands):
                pred_labels = node_output_labels[pred_id]
                operand_tokens = input_operands[i]
                
                for j, old_token in enumerate(operand_tokens):
                    if j < len(pred_labels) and old_token not in token_map:
                        new_label = pred_labels[j]
                        # Check if this label would cause a collision
                        # (different token already mapped to same label)
                        label_already_used = new_label in used_labels
                        if not label_already_used:
                            token_map[old_token] = new_label
                            used_labels.add(new_label)
        
        # Assign fresh labels to unmapped tokens
        all_tokens: Set[str] = set()
        for operand in input_operands:
            all_tokens.update(operand)
        all_tokens.update(output_tokens)
        
        for token in all_tokens:
            if token not in token_map:
                # Get a fresh label that's not already used
                new_label = label_gen.next(1)[0]
                while new_label in used_labels:
                    new_label = label_gen.next(1)[0]
                token_map[token] = new_label
                used_labels.add(new_label)
        
        # Apply mapping
        new_input_operands = [
            [token_map.get(t, t) for t in operand]
            for operand in input_operands
        ]
        new_output_tokens = [token_map.get(t, t) for t in output_tokens]
        
        # Generate fresh labels if output is empty but shape exists
        if not new_output_tokens and num_output_dims > 0:
            new_output_tokens = label_gen.next(num_output_dims)
        
        # Build and update equation
        new_equation = build_equation(new_input_operands, new_output_tokens)
        
        if self._debug:
            print(f"  Old: {old_equation} -> New: {new_equation}")
        
        layers[node_id]["einsum_equation"] = new_equation
        
        return new_output_tokens

    def rename_from_file(
        self,
        einsum_graph_path: PathLike,
        output_path: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """Load an einsum graph from file and rename ranks.

        Args:
            einsum_graph_path: Path to einsum_graph.yaml.
            output_path: Optional path to write the renamed graph.

        Returns:
            The renamed graph dictionary.
            
        Raises:
            FileNotFoundError: If the input file doesn't exist.
        """
        path = Path(einsum_graph_path)
        if not path.exists():
            raise FileNotFoundError(f"Einsum graph not found: {path}")

        with open(path) as f:
            graph_dict = yaml.safe_load(f)

        return self.rename(graph_dict, output_path)

    # Backward compatibility alias
    rename_ranks_from_file = rename_from_file


def rename_einsum_ranks(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Convenience function to rename ranks in an einsum graph from file.
    
    Args:
        input_path: Path to input einsum_graph.yaml.
        output_path: Optional path to write renamed graph.
        debug: Enable debug output.
        
    Returns:
        The renamed graph dictionary.
    """
    renamer = EinsumRankRenamer(debug=debug)
    return renamer.rename_from_file(input_path, output_path)


def rename_einsum_ranks_dict(
    graph_dict: Dict[str, Any],
    output_path: Optional[PathLike] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Convenience function to rename ranks in an einsum graph dictionary.
    
    Args:
        graph_dict: The einsum graph dictionary.
        output_path: Optional path to write renamed graph.
        debug: Enable debug output.
        
    Returns:
        The renamed graph dictionary.
    """
    renamer = EinsumRankRenamer(debug=debug)
    return renamer.rename(graph_dict, output_path)


# Backward compatibility alias
EinsumGraphRenamer = EinsumRankRenamer


__all__ = [
    "EinsumRankRenamer",
    "EinsumGraphRenamer",  # Backward compatibility
    "LabelGenerator",
    "build_equation",
    "rename_einsum_ranks",
    "rename_einsum_ranks_dict",
]

