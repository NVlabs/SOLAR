"""Node type registry for managing operation handlers.

This module provides a registry system for node type handlers,
supporting both built-in and dynamically generated handlers.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class NodeTypeHandler:
    """Handler for a specific node type.
    
    Attributes:
        node_type: The node type this handler manages.
        create_subgraph_func: Function to create subgraph.
        generate_einsum_func: Function to generate einsum notation.
        is_generated: Whether this handler was dynamically generated.
        metadata: Additional metadata about the handler.
    """
    node_type: str
    create_subgraph_func: Optional[Callable] = None
    generate_einsum_func: Optional[Callable] = None
    is_generated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_expand(self) -> bool:
        """Check if this handler can expand nodes."""
        return self.create_subgraph_func is not None
    
    def can_generate_einsum(self) -> bool:
        """Check if this handler can generate einsum."""
        return self.generate_einsum_func is not None


class NodeTypeRegistry:
    """Registry for managing node type handlers.
    
    This registry manages handlers for different node types,
    supporting both built-in and dynamically generated handlers.
    """
    
    def __init__(self, cache_dir: str = "./node_handlers_cache"):
        """Initialize the registry.
        
        Args:
            cache_dir: Directory for caching generated handlers.
        """
        self._handlers: Dict[str, NodeTypeHandler] = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_cached_handlers()
    
    def register(self,
                node_type: str,
                handler: NodeTypeHandler) -> None:
        """Register a handler for a node type.
        
        Args:
            node_type: The node type to register.
            handler: The handler for this node type.
        """
        self._handlers[node_type.lower()] = handler
    
    def get_handler(self, node_type: str) -> Optional[NodeTypeHandler]:
        """Get handler for a node type.
        
        Args:
            node_type: The node type to look up.
            
        Returns:
            Handler if found, None otherwise.
        """
        return self._handlers.get(node_type.lower())
    
    def has_handler(self, node_type: str) -> bool:
        """Check if a handler exists for a node type.
        
        Args:
            node_type: The node type to check.
            
        Returns:
            True if handler exists, False otherwise.
        """
        return node_type.lower() in self._handlers
    
    def list_handlers(self) -> List[str]:
        """List all registered node types.
        
        Returns:
            List of registered node type names.
        """
        return sorted(self._handlers.keys())
    
    def list_expandable(self) -> List[str]:
        """List node types that can be expanded.
        
        Returns:
            List of expandable node types.
        """
        return sorted([
            node_type for node_type, handler in self._handlers.items()
            if handler.can_expand()
        ])
    
    def list_einsum_capable(self) -> List[str]:
        """List node types that can generate einsum.
        
        Returns:
            List of einsum-capable node types.
        """
        return sorted([
            node_type for node_type, handler in self._handlers.items()
            if handler.can_generate_einsum()
        ])
    
    def save_generated_handler(self,
                              node_type: str,
                              handler: NodeTypeHandler) -> None:
        """Save a generated handler to cache.
        
        Args:
            node_type: The node type.
            handler: The handler to save.
        """
        cache_file = self.cache_dir / f"{node_type}_handler.json"
        
        # Save metadata and source code (if available)
        cache_data = {
            "node_type": node_type,
            "is_generated": handler.is_generated,
            "metadata": handler.metadata
        }
        
        # If the handler has source code in metadata, save it
        if "source_code" in handler.metadata:
            code_file = self.cache_dir / f"{node_type}.py"
            code_file.write_text(handler.metadata["source_code"])
            cache_data["code_file"] = str(code_file)
        
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    
    def _load_cached_handlers(self) -> None:
        """Load previously generated handlers from cache."""
        for cache_file in self.cache_dir.glob("*_handler.json"):
            try:
                with open(cache_file) as f:
                    cache_data = json.load(f)
                
                if cache_data.get("code_file"):
                    code_file = Path(cache_data["code_file"])
                    if code_file.exists():
                        # Load and execute the code to recreate the handler
                        code = code_file.read_text()
                        handler = NodeTypeHandlerFactory.create_handler_from_code(
                            node_type=cache_data["node_type"],
                            source_code=code,
                            metadata=cache_data["metadata"]
                        )
                        self.register(cache_data["node_type"], handler)
                        
            except Exception as e:
                print(f"Warning: Failed to load cached handler from {cache_file}: {e}")


class NodeTypeHandlerFactory:
    """Factory for creating node type handlers."""
    
    @staticmethod
    def create_handler_from_code(node_type: str,
                                source_code: str,
                                metadata: Optional[Dict[str, Any]] = None) -> NodeTypeHandler:
        """Create a handler from source code.
        
        Args:
            node_type: The node type.
            source_code: Python source code defining handler functions.
            metadata: Optional metadata.
            
        Returns:
            NodeTypeHandler instance.
        """
        # Execute the code to get the function
        local_vars = {}
        exec(source_code, {"Dict": Dict, "Any": Any}, local_vars)
        
        # Look for the create function
        create_func = None
        einsum_func = None
        
        for name, obj in local_vars.items():
            if callable(obj):
                if "create" in name and "subgraph" in name:
                    create_func = obj
                elif "einsum" in name:
                    einsum_func = obj
        
        metadata = metadata or {}
        metadata["source_code"] = source_code
        
        return NodeTypeHandler(
            node_type=node_type,
            create_subgraph_func=create_func,
            generate_einsum_func=einsum_func,
            is_generated=True,
            metadata=metadata
        )
    
    @staticmethod
    def create_handler_from_methods(
        node_type: str,
        create_subgraph_method: Optional[Callable] = None,
        generate_einsum_method: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeTypeHandler:
        """Create a handler from existing methods.
        
        Args:
            node_type: The node type.
            create_subgraph_method: Method to create subgraph.
            generate_einsum_method: Method to generate einsum.
            metadata: Optional metadata.
            
        Returns:
            NodeTypeHandler instance.
        """
        return NodeTypeHandler(
            node_type=node_type,
            create_subgraph_func=create_subgraph_method,
            generate_einsum_func=generate_einsum_method,
            is_generated=False,
            metadata=metadata or {}
        )


class NodeExpansionStrategy:
    """Strategy for determining which nodes should be expanded."""
    
    def should_expand(self,
                     node_id: str,
                     node_data: Dict[str, Any]) -> bool:
        """Determine if a node should be expanded.
        
        Args:
            node_id: Node identifier.
            node_data: Node data dictionary.
            
        Returns:
            True if node should be expanded, False otherwise.
        """
        raise NotImplementedError


class DefaultNodeExpansionStrategy(NodeExpansionStrategy):
    """Default expansion strategy based on node types."""
    
    def __init__(self,
                registry: NodeTypeRegistry,
                always_expand: Optional[Set[str]] = None,
                never_expand: Optional[Set[str]] = None,
                debug: bool = False):
        """Initialize the strategy.
        
        Args:
            registry: Node type registry.
            always_expand: Set of node types to always expand.
            never_expand: Set of node types to never expand.
            debug: Enable debug output.
        """
        self.registry = registry
        self.debug = debug
        
        # Default expansion rules
        self.always_expand = always_expand or {
            "gru", "lstm", "multihead_attention",
            "multi_head_attention",
            "scaled_dot_product_attention", "flex_attention",
            "attention",
            "cosine_similarity", "layer_norm", "batch_norm",
            "group_norm", "instance_norm", "softmax",
            "logsumexp", "mish", "hardswish", "gelu"
        }
        
        self.never_expand = never_expand or {
            "input-tensor", "output-tensor", "parameter",
            "add", "mul", "sub", "div", "matmul", "linear",
            "conv1d", "conv2d", "conv3d", "reshape", "view",
            "transpose", "permute", "flatten"
        }
    
    def should_expand(self,
                     node_id: str,
                     node_data: Dict[str, Any]) -> bool:
        """Determine if a node should be expanded.
        
        Args:
            node_id: Node identifier.
            node_data: Node data dictionary.
            
        Returns:
            True if node should be expanded, False otherwise.
        """
        node_type = (node_data.get("type") or node_data.get("node_type") or "").lower()
        
        # Check explicit rules
        if node_type in self.never_expand:
            return False
        
        # Always-expand types are considered complex by default. Expansion may still
        # fail later if no handler/agent is available, but it's useful to mark them
        # as candidates here.
        if node_type in self.always_expand:
            return True
        
        # Check if it starts with "reduction_" (composite operations)
        if node_type.startswith("reduction_"):
            return True
        
        # Check if handler exists and can expand
        handler = self.registry.get_handler(node_type)
        if handler and handler.can_expand():
            # For unknown types, expand if they're complex enough
            input_shapes = node_data.get("input_shapes", [])
            if len(input_shapes) > 2:  # Multiple inputs suggest complexity
                return True
        
        return False
