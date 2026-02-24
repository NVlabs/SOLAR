"""Convert einsum equations to TACO (Tensor Algebra Compiler) expressions.

This module provides functionality to convert Solar's einsum notation to
TACO expression format for compatibility with sparse tensor compilers.

TACO expression format:
    Output[indices] = Input0[indices] <elementwise_op> Input1[indices]
    
The reduction operation (e.g., sum over contracted indices) is specified
separately in the reduction_op field, not embedded in the expression.

For convolution operations, spatial indices use + notation to show the
sliding window relationship (e.g., p+r instead of p,r).

Examples:
    >>> from solar.einsum.einsum_to_taco import EinsumToTaco, generate_taco_expression
    >>> # Matrix multiplication: MK,KN->MN with reduction over K
    >>> expr = generate_taco_expression("MK,KN->MN", "mul", "add")
    >>> print(expr)
    O0[m,n] = In0[m,k] * In1[k,n]
    >>> # Addmm (bias + matmul): MN,MK,KN->MN
    >>> expr = generate_taco_expression("MN,MK,KN->MN", "mul", "add")
    >>> print(expr)
    O0[m,n] = In0[m,n] + In1[m,k] * In2[k,n]
    >>> # ReLU: MN->MN
    >>> expr = generate_taco_expression("MN->MN", "relu", None)
    >>> print(expr)
    O0[m,n] = relu(In0[m,n])
    >>> # Conv2d: BC(P+R)(Q+S),OCRS->BOPQ
    >>> expr = generate_taco_expression("BC(P+R)(Q+S),OCRS->BOPQ", "mul", "add")
    >>> print(expr)
    O0[b,o,p,q] = In0[b,c,p+r,q+s] * In1[o,c,r,s]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from solar.common.utils import parse_dim_tokens, parse_einsum_equation


class EinsumToTaco:
    """Convert einsum equations to TACO expression format.
    
    TACO (Tensor Algebra Compiler) is a library for compiling tensor algebra
    expressions to efficient code. This class converts Solar's einsum notation
    to TACO's expression syntax.
    
    Attributes:
        debug: Whether to print debug information.
    """

    # Mapping from Solar operator names to TACO symbols
    ELEMENTWISE_OP_MAP = {
        "mul": "*",
        "add": "+",
        "sub": "-",
        "div": "/",
        "max": "max",
        "min": "min",
        None: None,
        "null": None,
    }
    
    REDUCTION_OP_MAP = {
        "add": "+",
        "mul": "*",
        "max": "max",
        "min": "min",
        "mean": "+",  # mean is sum / count, represented as sum in TACO
        None: None,
        "null": None,
    }
    
    # Special elementwise operations (unary)
    UNARY_OPS = frozenset({
        "relu", "leaky_relu", "prelu", "rrelu",
        "sigmoid", "tanh", "gelu", "selu", "elu", "celu", "mish", "silu",
        "softmax", "log_softmax", "softmin",
        "exp", "log", "log2", "log10", "sqrt", "rsqrt",
        "abs", "neg", "softplus", "softsign",
        "hardswish", "hardsigmoid", "hardtanh",
        "sin", "cos", "tan",
        "dropout",
        # Normalization ops (treated as unary for TACO)
        # Both underscore and no-underscore variants
        "batch_norm", "batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d",
        "layer_norm", "layernorm",
        "group_norm", "groupnorm",
        "instance_norm", "instancenorm",
        "normalize",
    })
    
    # Operations with special constraints (annotated in TACO expression)
    CONSTRAINT_OPS = {
        "diag": "diagonal",      # Only diagonal elements (i==j)
        "triu": "upper_tri",     # Upper triangular (i<=j)
        "tril": "lower_tri",     # Lower triangular (i>=j)
    }

    def __init__(self, debug: bool = False) -> None:
        """Initialize the converter.
        
        Args:
            debug: Enable debug output.
        """
        self._debug = debug

    def convert(
        self,
        einsum_equation: str,
        elementwise_op: Optional[str] = "mul",
        reduction_op: Optional[str] = "add",
        input_names: Optional[List[str]] = None,
        output_name: str = "O0",
    ) -> str:
        """Convert an einsum equation to TACO expression.
        
        Args:
            einsum_equation: Einsum equation string (e.g., "MK,KN->MN").
            elementwise_op: Elementwise operation (e.g., "mul", "add").
            reduction_op: Reduction operation (e.g., "add", "max").
            input_names: Names for input tensors (default: In0, In1, ...).
            output_name: Name for output tensor (default: O0).
            
        Returns:
            TACO expression string.
        """
        if not einsum_equation or "->" not in einsum_equation:
            return ""
        
        # Parse the einsum equation
        input_dims_list, output_dims = parse_einsum_equation(einsum_equation)
        
        if not input_dims_list:
            return ""
        
        # Generate input tensor names
        if input_names is None:
            input_names = [f"In{i}" for i in range(len(input_dims_list))]
        
        # Convert dimensions to lowercase for TACO indices
        output_indices = self._dims_to_indices(output_dims)
        input_indices_list = [self._dims_to_indices(dims) for dims in input_dims_list]
        
        # Find contracted dimensions (in inputs but not in output)
        all_input_dims = set()
        for dims in input_dims_list:
            all_input_dims.update(dims)
        output_dims_set = set(output_dims)
        contracted_dims = all_input_dims - output_dims_set
        
        # Build TACO expression
        return self._build_taco_expression(
            output_name=output_name,
            output_indices=output_indices,
            input_names=input_names,
            input_indices_list=input_indices_list,
            elementwise_op=elementwise_op,
            reduction_op=reduction_op,
            has_contraction=bool(contracted_dims),
        )

    def _dims_to_indices(self, dims: List[str]) -> str:
        """Convert dimension tokens to TACO index string.
        
        Handles grouped dimensions with + notation (e.g., "(P+R)" -> "p+r").
        
        Args:
            dims: List of dimension tokens (e.g., ["B0", "B1", "K"] or ["B", "C", "(P+R)", "(Q+S)"]).
            
        Returns:
            Comma-separated lowercase indices (e.g., "b0,b1,k" or "b,c,p+r,q+s").
        """
        if not dims:
            return ""
        
        result = []
        for d in dims:
            # Handle grouped dimensions like "(P+R)"
            if d.startswith("(") and d.endswith(")") and "+" in d:
                # Extract inner content and convert to lowercase
                inner = d[1:-1]  # Remove parentheses
                result.append(inner.lower())
            else:
                result.append(d.lower())
        
        return ",".join(result)

    def _build_taco_expression(
        self,
        output_name: str,
        output_indices: str,
        input_names: List[str],
        input_indices_list: List[str],
        elementwise_op: Optional[str],
        reduction_op: Optional[str],
        has_contraction: bool,
    ) -> str:
        """Build the TACO expression string.
        
        Args:
            output_name: Output tensor name.
            output_indices: Output indices string.
            input_names: Input tensor names.
            input_indices_list: Input indices strings.
            elementwise_op: Elementwise operation.
            reduction_op: Reduction operation.
            has_contraction: Whether there are contracted dimensions.
            
        Returns:
            TACO expression string.
        """
        # Output tensor with indices (using square brackets)
        if output_indices:
            output_tensor = f"{output_name}[{output_indices}]"
        else:
            output_tensor = output_name
        
        # Build input expressions (using square brackets)
        input_exprs = []
        for name, indices in zip(input_names, input_indices_list):
            if indices:
                input_exprs.append(f"{name}[{indices}]")
            else:
                input_exprs.append(name)
        
        # Handle different cases
        num_inputs = len(input_exprs)
        
        # Check for unary operation
        if num_inputs == 1:
            return self._build_unary_expression(
                output_tensor, input_exprs[0], elementwise_op, reduction_op, has_contraction
            )
        
        # Binary or multi-input operation
        return self._build_binary_expression(
            output_tensor, input_exprs, elementwise_op, reduction_op, has_contraction
        )

    def _build_unary_expression(
        self,
        output_tensor: str,
        input_expr: str,
        elementwise_op: Optional[str],
        reduction_op: Optional[str],
        has_contraction: bool,
    ) -> str:
        """Build TACO expression for unary operations.
        
        The reduction operation is implicit (specified separately), so we don't
        embed it in the expression as "O = O + ...".
        
        Args:
            output_tensor: Output tensor expression.
            input_expr: Input tensor expression.
            elementwise_op: Elementwise operation.
            reduction_op: Reduction operation (for reference, not embedded).
            has_contraction: Whether there are contracted dimensions.
            
        Returns:
            TACO expression string.
        """
        # Check for special unary ops (relu, exp, etc.)
        if elementwise_op and elementwise_op.lower() in self.UNARY_OPS:
            op_name = elementwise_op.lower()
            return f"{output_tensor} = {op_name}({input_expr})"
        
        # Simple copy (reduction is implicit over contracted dims)
        return f"{output_tensor} = {input_expr}"

    def _build_binary_expression(
        self,
        output_tensor: str,
        input_exprs: List[str],
        elementwise_op: Optional[str],
        reduction_op: Optional[str],
        has_contraction: bool,
    ) -> str:
        """Build TACO expression for binary/multi-input operations.
        
        The reduction operation is implicit (specified separately), so we don't
        embed it in the expression as "O = O + ...".
        
        For addmm-like operations (3 inputs: bias + matmul), we explicitly show
        the bias addition: O = A + B * C
        
        Args:
            output_tensor: Output tensor expression.
            input_exprs: Input tensor expressions.
            elementwise_op: Elementwise operation.
            reduction_op: Reduction operation (for reference, not embedded).
            has_contraction: Whether there are contracted dimensions.
            
        Returns:
            TACO expression string.
        """
        elem_sym = self.ELEMENTWISE_OP_MAP.get(elementwise_op, "*")
        
        # Handle addmm-like operations: 3 inputs where first is bias
        # Pattern: bias + matmul(A, B) -> O = bias + A * B
        if len(input_exprs) == 3 and elem_sym == "*":
            # First input is bias (added), second and third are multiplied
            bias_expr = input_exprs[0]
            matmul_rhs = f"{input_exprs[1]} {elem_sym} {input_exprs[2]}"
            return f"{output_tensor} = {bias_expr} + {matmul_rhs}"
        
        # Standard binary operation (matmul, element-wise, etc.)
        if elem_sym and elem_sym not in ("max", "min"):
            rhs = f" {elem_sym} ".join(input_exprs)
        elif elem_sym in ("max", "min"):
            rhs = f"{elem_sym}({', '.join(input_exprs)})"
        else:
            # No elementwise op, just use first input
            rhs = input_exprs[0]
        
        return f"{output_tensor} = {rhs}"

    def convert_layer(self, layer_data: Dict[str, Any]) -> str:
        """Convert a layer dictionary to TACO expression.
        
        Args:
            layer_data: Layer data dictionary with einsum_equation, etc.
            
        Returns:
            TACO expression string.
        """
        einsum_equation = layer_data.get("einsum_equation", "")
        elementwise_op = layer_data.get("elementwise_op", "mul")
        reduction_op = layer_data.get("reduction_op", "add")
        layer_type = layer_data.get("type", "").lower()
        
        # Handle null/None values
        if elementwise_op in (None, "null", "None"):
            elementwise_op = None
        if reduction_op in (None, "null", "None"):
            reduction_op = None
        
        taco_expr = self.convert(einsum_equation, elementwise_op, reduction_op)
        
        # Add constraint annotations for special operations
        if layer_type in self.CONSTRAINT_OPS and taco_expr:
            constraint = self.CONSTRAINT_OPS[layer_type]
            taco_expr = f"{taco_expr}  # constraint: {constraint}"
        
        return taco_expr

    def add_taco_to_graph(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add TACO expressions to all layers in an einsum graph.
        
        Args:
            graph_dict: Einsum graph dictionary with 'layers' key.
            
        Returns:
            Modified graph dictionary with taco_expression added to each layer.
        """
        if "layers" not in graph_dict:
            return graph_dict
        
        for layer_id, layer_data in graph_dict["layers"].items():
            if "einsum_equation" in layer_data:
                taco_expr = self.convert_layer(layer_data)
                if taco_expr:
                    layer_data["taco_expression"] = taco_expr
        
        return graph_dict


def generate_taco_expression(
    einsum_equation: str,
    elementwise_op: Optional[str] = "mul",
    reduction_op: Optional[str] = "add",
) -> str:
    """Convenience function to generate a TACO expression.
    
    Args:
        einsum_equation: Einsum equation string.
        elementwise_op: Elementwise operation.
        reduction_op: Reduction operation.
        
    Returns:
        TACO expression string.
    """
    converter = EinsumToTaco()
    return converter.convert(einsum_equation, elementwise_op, reduction_op)


def add_taco_expressions(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to add TACO expressions to an einsum graph.
    
    Args:
        graph_dict: Einsum graph dictionary.
        
    Returns:
        Modified graph dictionary with taco_expression fields.
    """
    converter = EinsumToTaco()
    return converter.add_taco_to_graph(graph_dict)


__all__ = [
    "EinsumToTaco",
    "generate_taco_expression",
    "add_taco_expressions",
]
