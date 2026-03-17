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

"""Registry for einsum operation handlers.

This module provides a centralized registry for managing einsum operation
handlers. Handlers can be registered using a decorator or explicit registration.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from solar.einsum.ops.base import EinsumOpHandler, EinsumOp
    from solar.common.types import TensorShapes

# Flag to track if handlers are being loaded (to prevent circular imports)
_loading_handlers = False


class EinsumOpRegistry:
    """Registry for einsum operation handlers.
    
    This class manages a collection of EinsumOpHandler instances and provides
    methods for registration and lookup.
    
    Usage:
        registry = EinsumOpRegistry()
        
        # Register a handler class
        registry.register_handler(MatmulHandler)
        
        # Or use the decorator
        @registry.register
        class MyHandler(EinsumOpHandler):
            supported_ops = ["my_op"]
            ...
        
        # Get einsum for an operation
        einsum_op = registry.get_einsum_op("matmul", shapes)
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the registry.
        
        Args:
            debug: Enable debug output for handlers.
        """
        self.debug = debug
        self._handlers: Dict[str, "EinsumOpHandler"] = {}
        self._handler_classes: List[Type["EinsumOpHandler"]] = []
        self._op_to_handler: Dict[str, "EinsumOpHandler"] = {}
    
    def register_handler(self, handler_class: Type["EinsumOpHandler"]) -> None:
        """Register a handler class.
        
        Args:
            handler_class: Handler class to register.
        """
        # Instantiate the handler
        handler = handler_class(debug=self.debug)
        self._handler_classes.append(handler_class)
        
        # Map each supported operation to this handler
        for op_name in handler.supported_ops:
            op_key = op_name.lower()
            self._op_to_handler[op_key] = handler
            if self.debug:
                print(f"Registered handler for '{op_key}': {handler_class.__name__}")
    
    def register(self, handler_class: Type["EinsumOpHandler"]) -> Type["EinsumOpHandler"]:
        """Decorator to register a handler class.
        
        Usage:
            @registry.register
            class MyHandler(EinsumOpHandler):
                ...
        """
        self.register_handler(handler_class)
        return handler_class
    
    def get_handler(self, op_name: str) -> Optional["EinsumOpHandler"]:
        """Get the handler for an operation.
        
        Args:
            op_name: Operation name.
            
        Returns:
            Handler if registered, None otherwise.
        """
        return self._op_to_handler.get(op_name.lower())
    
    def has_handler(self, op_name: str) -> bool:
        """Check if a handler is registered for an operation.
        
        Args:
            op_name: Operation name.
            
        Returns:
            True if a handler exists.
        """
        return op_name.lower() in self._op_to_handler
    
    def get_einsum_op(
        self,
        op_name: str,
        shapes: "TensorShapes",
        **kwargs: Any
    ) -> "EinsumOp":
        """Get an einsum operation for the given operation name.
        
        Args:
            op_name: Operation name.
            shapes: Positional input/output tensor shapes.
            **kwargs: Additional operation-specific parameters.
            
        Returns:
            EinsumOp for the operation.
            
        Raises:
            ValueError: If no handler is registered for the operation.
        """
        handler = self.get_handler(op_name)
        if handler is None:
            raise ValueError(f"No handler registered for operation: {op_name}")
        
        return handler.generate_einsum(op_name, shapes, **kwargs)
    
    def list_supported_ops(self) -> List[str]:
        """List all supported operation names.
        
        Returns:
            List of operation names.
        """
        return sorted(self._op_to_handler.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics.
        """
        return {
            "total_handlers": len(self._handler_classes),
            "total_ops": len(self._op_to_handler),
            "ops": self.list_supported_ops(),
        }


# Global registry instance
_global_registry: Optional[EinsumOpRegistry] = None
_handlers_loaded = False


def get_global_registry(load_handlers: bool = True) -> EinsumOpRegistry:
    """Get the global einsum operation registry.
    
    Args:
        load_handlers: If True, load all handlers on first access.
    
    Returns:
        The global registry instance.
    """
    global _global_registry, _handlers_loaded, _loading_handlers
    
    if _global_registry is None:
        _global_registry = EinsumOpRegistry()
    
    # Load handlers if requested and not already loaded/loading
    if load_handlers and not _handlers_loaded and not _loading_handlers:
        _loading_handlers = True
        try:
            _load_all_handlers()
            _handlers_loaded = True
        finally:
            _loading_handlers = False
    
    return _global_registry


def _load_all_handlers() -> None:
    """Load all built-in handlers."""
    # Import handler modules to trigger registration
    # These imports will register handlers via the decorator
    # We import them explicitly to avoid circular import issues
    import solar.einsum.ops.matmul_ops
    import solar.einsum.ops.conv_ops
    import solar.einsum.ops.elementwise_ops
    import solar.einsum.ops.reduction_ops
    import solar.einsum.ops.cumulative_ops  # cumsum, cumprod, cummax, cummin
    import solar.einsum.ops.loss_ops        # kl_div, cross_entropy, mse_loss, etc.
    import solar.einsum.ops.attention_ops
    import solar.einsum.ops.norm_ops
    import solar.einsum.ops.pooling_ops
    import solar.einsum.ops.shape_ops
    import solar.einsum.ops.misc_ops


def register_einsum_op(handler_class: Type["EinsumOpHandler"]) -> Type["EinsumOpHandler"]:
    """Decorator to register a handler with the global registry.
    
    Usage:
        @register_einsum_op
        class MyHandler(EinsumOpHandler):
            supported_ops = ["my_op"]
            ...
    """
    # Get registry without loading handlers to avoid circular import
    get_global_registry(load_handlers=False).register_handler(handler_class)
    return handler_class


__all__ = ["EinsumOpRegistry", "get_global_registry", "register_einsum_op"]

