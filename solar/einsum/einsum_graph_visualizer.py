"""Visualize einsum graphs as PDF files.

This module provides functionality to render einsum graphs as directed
graph visualizations using NetworkX and Graphviz, with nodes displaying
operation details (name, einsum equation, weight shapes).

Example:
    >>> from solar.einsum.einsum_graph_visualizer import EinsumGraphVisualizer
    >>> visualizer = EinsumGraphVisualizer()
    >>> visualizer.save_graph_pdf("input/einsum_graph.yaml", "output/graph.pdf")

The visualization shows:
- Nodes with operation name, einsum equation, and weight info
- Directed edges representing data flow (inputs -> outputs)
- Left-to-right layout for clear flow visualization
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from solar.common.utils import load_einsum_graph_to_networkx


PathLike = Union[str, Path]


def _format_shapes(shapes: Dict[str, Any]) -> str:
    """Format shape dictionary for display.
    
    Args:
        shapes: Dictionary with 'inputs' and 'outputs' arrays.
        
    Returns:
        Formatted string showing input/output shapes.
    """
    parts = []
    inputs = shapes.get("inputs", [])
    outputs = shapes.get("outputs", [])
    
    # Show inputs (skip first which is usually activation, show weights)
    if len(inputs) > 1:
        for i, shape in enumerate(inputs[1:], start=1):
            shape_str = "×".join(str(d) for d in shape)
            parts.append(f"W{i}: {shape_str}")
    
    return "\n".join(parts) if parts else ""


def _create_node_label(node_id: str, node_data: Dict[str, Any]) -> str:
    """Create a multi-line label for a graph node.
    
    Args:
        node_id: The node identifier.
        node_data: Node attributes from the einsum graph.
        
    Returns:
        Formatted label string for the node.
    """
    lines = []
    
    # Node name (bold in HTML-like labels)
    node_type = node_data.get("type", "unknown")
    lines.append(f"<b>{node_id}</b>")
    lines.append(f"({node_type})")
    
    # Einsum equation
    equation = node_data.get("einsum_equation", "")
    if equation:
        lines.append(f"<i>{equation}</i>")
    
    # Weight shapes
    shapes = node_data.get("shapes", {})
    weight_info = _format_shapes(shapes)
    if weight_info:
        lines.append(weight_info)
    
    return "<br/>".join(lines)


class EinsumGraphVisualizer:
    """Visualize einsum graphs as PDF files.
    
    This class renders einsum graphs using NetworkX and Graphviz/matplotlib,
    producing publication-quality PDF visualizations.
    
    Attributes:
        debug: Whether to print debug information.
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the visualizer.
        
        Args:
            debug: Enable debug output.
        """
        self._debug = debug

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    def save_graph_pdf(
        self,
        einsum_graph_path: PathLike,
        output_path: PathLike,
        *,
        title: Optional[str] = None,
        use_graphviz: bool = True,
    ) -> Path:
        """Load an einsum graph and save it as a PDF visualization.
        
        Args:
            einsum_graph_path: Path to einsum_graph.yaml or einsum_graph_renamed.yaml.
            output_path: Path for the output PDF file.
            title: Optional title for the graph.
            use_graphviz: If True, use Graphviz for layout (requires pygraphviz).
                         If False, use matplotlib with spring layout.
        
        Returns:
            Path to the saved PDF file.
            
        Raises:
            FileNotFoundError: If the input file doesn't exist.
        """
        graph_path = Path(einsum_graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Einsum graph not found: {graph_path}")

        with open(graph_path) as f:
            graph_dict = yaml.safe_load(f)

        return self.save_graph_pdf_from_dict(
            graph_dict,
            output_path,
            title=title or graph_dict.get("model_name", "Einsum Graph"),
            use_graphviz=use_graphviz,
        )

    def save_graph_pdf_from_dict(
        self,
        graph_dict: Dict[str, Any],
        output_path: PathLike,
        *,
        title: Optional[str] = None,
        use_graphviz: bool = True,
    ) -> Path:
        """Save an einsum graph dictionary as a PDF visualization.
        
        Args:
            graph_dict: The einsum graph dictionary with 'layers' key.
            output_path: Path for the output PDF file.
            title: Optional title for the graph.
            use_graphviz: If True, use Graphviz for layout.
        
        Returns:
            Path to the saved PDF file.
        """
        import networkx as nx
        
        layers = graph_dict.get("layers", {})
        G = load_einsum_graph_to_networkx(layers)
        
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try Graphviz first for better layout
        if use_graphviz:
            try:
                self._save_with_graphviz(G, layers, out_path, title)
                if self._debug:
                    print(f"✅ Saved graph PDF (Graphviz): {out_path}")
                return out_path
            except ImportError:
                if self._debug:
                    print("Graphviz not available, falling back to matplotlib")
        
        # Fallback to matplotlib
        self._save_with_matplotlib(G, layers, out_path, title)
        if self._debug:
            print(f"✅ Saved graph PDF (matplotlib): {out_path}")
        return out_path

    def _save_with_graphviz(
        self,
        G: Any,
        layers: Dict[str, Any],
        output_path: Path,
        title: Optional[str],
    ) -> None:
        """Save graph using Graphviz for high-quality layout."""
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError(
                "graphviz package required. Install with: pip install graphviz"
            )
        
        # Create Graphviz digraph
        dot = Digraph(
            name="einsum_graph",
            format="pdf",
            engine="dot",
        )
        
        # Graph attributes for left-to-right layout
        dot.attr(rankdir="LR")  # Left to right
        dot.attr(splines="ortho")  # Orthogonal edges
        dot.attr(nodesep="0.5")
        dot.attr(ranksep="1.0")
        
        if title:
            dot.attr(label=title, labelloc="t", fontsize="16")
        
        # Node attributes
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="lightblue",
            fontname="Helvetica",
            fontsize="10",
        )
        
        # Edge attributes
        dot.attr(
            "edge",
            fontname="Helvetica",
            fontsize="9",
        )
        
        # Add nodes
        for node_id in G.nodes():
            node_data = layers.get(node_id, {})
            label = self._create_graphviz_label(node_id, node_data)
            
            # Color based on node type
            node_type = node_data.get("type", "").lower()
            if node_type == "start":
                fillcolor = "lightgreen"
            elif node_data.get("is_real_einsum", False):
                fillcolor = "lightyellow"
            else:
                fillcolor = "lightblue"
            
            dot.node(node_id, label=label, fillcolor=fillcolor)
        
        # Add edges
        for src, dst in G.edges():
            dot.edge(src, dst)
        
        # Save to PDF (graphviz appends .pdf automatically)
        output_stem = output_path.with_suffix("")
        dot.render(str(output_stem), cleanup=True)

    def _create_graphviz_label(
        self,
        node_id: str,
        node_data: Dict[str, Any],
    ) -> str:
        """Create a plain text label for a Graphviz node."""
        lines = []
        
        # Node name
        node_type = node_data.get("type", "unknown")
        lines.append(f"{node_id}")
        lines.append(f"({node_type})")
        
        # Einsum equation
        equation = node_data.get("einsum_equation", "")
        if equation:
            lines.append(equation)
        
        # Weight shapes
        shapes = node_data.get("shapes", {})
        weight_info = _format_shapes(shapes)
        if weight_info:
            lines.append(weight_info)
        
        return "\n".join(lines)

    def _save_with_matplotlib(
        self,
        G: Any,
        layers: Dict[str, Any],
        output_path: Path,
        title: Optional[str],
    ) -> None:
        """Save graph using matplotlib (fallback when Graphviz unavailable)."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create figure
        num_nodes = len(G.nodes())
        fig_width = max(12, num_nodes * 1.5)
        fig_height = max(8, num_nodes * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Compute hierarchical layout (left-to-right)
        try:
            # Try to use graphviz_layout if available
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
        except Exception:
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            # Rotate for left-to-right flow
            pos = {node: (y, -x) for node, (x, y) in pos.items()}
        
        # Color nodes based on type
        node_colors = []
        for node_id in G.nodes():
            node_data = layers.get(node_id, {})
            node_type = node_data.get("type", "").lower()
            if node_type == "start":
                node_colors.append("lightgreen")
            elif node_data.get("is_real_einsum", False):
                node_colors.append("lightyellow")
            else:
                node_colors.append("lightblue")
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=3000,
            node_shape="s",
            alpha=0.9,
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )
        
        # Create labels
        labels = {}
        for node_id in G.nodes():
            node_data = layers.get(node_id, {})
            node_type = node_data.get("type", "unknown")
            equation = node_data.get("einsum_equation", "")
            
            label_lines = [node_id, f"({node_type})"]
            if equation:
                label_lines.append(equation)
            
            # Add weight info (abbreviated) - weights are inputs[1:]
            shapes = node_data.get("shapes", {})
            inputs = shapes.get("inputs", [])
            if len(inputs) > 1:
                shape = inputs[1]  # First weight
                shape_str = "×".join(str(d) for d in shape)
                label_lines.append(f"W:{shape_str}")
            
            labels[node_id] = "\n".join(label_lines)
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_family="sans-serif",
        )
        
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)


def save_einsum_graph_pdf(
    einsum_graph_path: PathLike,
    output_path: PathLike,
    *,
    title: Optional[str] = None,
    debug: bool = False,
) -> Path:
    """Convenience function to save an einsum graph as PDF.
    
    Args:
        einsum_graph_path: Path to einsum_graph.yaml.
        output_path: Path for output PDF file.
        title: Optional title for the graph.
        debug: Enable debug output.
        
    Returns:
        Path to the saved PDF file.
    """
    visualizer = EinsumGraphVisualizer(debug=debug)
    return visualizer.save_graph_pdf(einsum_graph_path, output_path, title=title)


__all__ = [
    "EinsumGraphVisualizer",
    "save_einsum_graph_pdf",
]

