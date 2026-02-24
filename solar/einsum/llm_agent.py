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

"""LLM Agent for dynamic node type handling in Solar.

This module provides an LLM-based agent that can generate handlers for
unknown node types dynamically, following Google's Python style guide.
"""

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None

# Optional OpenAI client class (OpenAI SDK v1+). Tests patch this symbol.
OpenAI = getattr(openai, "OpenAI", None) if openai is not None else None

@dataclass
class AgentConfig:
    """Configuration for the LLM agent.
    
    Attributes:
        api_key: OpenAI API key.
        model: Model to use (default: gpt-4).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        cache_dir: Directory for caching generated handlers.
    """
    api_key: str
    model: str = "gpt-4"
    temperature: float = 0.2
    max_tokens: int = 2000
    cache_dir: str = "./llm_handlers_cache"


class NodeTypeConversionAgent:
    """LLM agent for converting unknown node types to known operations.
    
    This agent uses an LLM to generate code that converts unknown node types
    into subgraphs of known operations.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the conversion agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config
        self._init_openai()
        self._ensure_cache_dir()
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if openai is None:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

            openai.api_key = self.config.api_key
        self.client = openai
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_conversion_code(self,
                                node_type: str,
                                sample_node_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate conversion code for an unknown node type.
        
        Args:
            node_type: The unknown node type.
            sample_node_data: Sample data for the node type.
            
        Returns:
            Tuple of (generated_code, metadata).
        """
        # Check cache first
        cached_code = self._check_cache(node_type)
        if cached_code:
            return cached_code, {"source": "cache", "is_fallback": False}
        
        # Generate prompt
        prompt = self._create_conversion_prompt(node_type, sample_node_data)
        
        try:
            # Call LLM
            response = self._call_llm(prompt)
            code = self._extract_code(response)
            
            # Validate and clean code
            code = self._validate_code(code, node_type)
            
            # Cache the result
            self._cache_code(node_type, code)
            
            return code, {"source": "generated", "is_fallback": False}
                
        except Exception as e:
            print(f"LLM generation failed: {e}")
            # Return fallback implementation
            return self._generate_fallback(node_type), {
                "source": "fallback",
                "is_fallback": True,
                "error": str(e)
            }
    
    def _create_conversion_prompt(self,
                      node_type: str,
                      sample_node_data: Dict[str, Any]) -> str:
        """Create prompt for the LLM.
        
        Args:
            node_type: The unknown node type.
            sample_node_data: Sample data for the node.
            
        Returns:
            Formatted prompt string.
        """
        return f"""Generate a Python function that converts a '{node_type}' node into a subgraph of basic einsum operations.

The function should have this signature:
def create_{node_type}_subgraph(node_id: str, node_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:

Sample node data:
{json.dumps(sample_node_data, indent=2)}

## EINSUM EQUATION FORMAT

Solar uses uppercase einsum notation. Each dimension is a single uppercase letter, optionally followed by a number (e.g., B, H, B0, B1).

### Common Einsum Patterns:
- Matrix multiplication: `MK,KN->MN` (K is contracted/summed)
- Batched matmul: `BMK,BKN->BMN`
- Attention Q@K^T: `BHQD,BHKD->BHQK` (D is contracted)
- Attention weights@V: `BHQK,BHKV->BHQV` (K is contracted)
- Elementwise: `ABC->ABC` (no contraction)
- Reduction: `ABC->AB` (C is reduced)
- Conv2d sliding window: `BC(P+R)(Q+S),OCRS->BOPQ`

### Dimension Naming Convention:
- B = batch dimension
- H = heads (for attention)
- Q, K, V = query/key/value sequence lengths
- D = embedding dimension
- M, N = matrix dimensions
- C, O = input/output channels
- P, Q, R, S = spatial dimensions

## SUBGRAPH NODE FORMAT

Each node in the subgraph must have:
```python
{{
    "type": "matmul",  # or "add", "mul", "softmax", "relu", etc.
    "einsum_equation": "BHQD,BHKD->BHQK",  # UPPERCASE einsum
    "elementwise_op": "mul",  # mul, add, sub, div, softmax, relu, etc.
    "reduction_op": "add",    # add, mul, max, min, none
    "is_real_einsum": True,   # True for matmul/conv, False for elementwise
    "is_einsum_supportable": True,
    "shapes": {{
        "Input": [...],
        "Input_1": [...],  # for binary ops
        "Output": [...],
    }},
    "tensor_names": {{
        "inputs": ["prev_node.Output", ...],
        "outputs": ["this_node.Output"],
    }},
    "tensor_shapes": {{
        "inputs": [[...], [...]],
        "outputs": [[...]],
    }},
    "connections": {{
        "inputs": ["prev_node_id"],
        "outputs": ["next_node_id"],
    }},
}}
```

## EXAMPLE: Scaled Dot-Product Attention

For `scaled_dot_product_attention(Q, K, V)`:
1. `qk_matmul`: Q @ K^T -> scores (einsum: `BHQD,BHKD->BHQK`)
2. `scale`: scores * (1/sqrt(d_k)) (einsum: `BHQK->BHQK`, elementwise_op: "mul")
3. `softmax`: softmax(scores, dim=-1) (einsum: `BHQK->BHQK`, elementwise_op: "softmax")
4. `av_matmul`: weights @ V -> output (einsum: `BHQK,BHKV->BHQV`)

## YOUR TASK

Generate a function that:
1. Extracts input/output shapes from node_data
2. Creates a subgraph of basic operations with proper einsum equations
3. Connects nodes correctly (first node gets external inputs, last node provides outputs)
4. Returns the subgraph dictionary

```python
def create_{node_type}_subgraph(node_id: str, node_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    subgraph = {{}}
    
    # Extract shapes
    input_shapes = node_data.get("input_shapes", [])
    output_shapes = node_data.get("output_shapes", [])
    module_args = node_data.get("module_args", {{}})
    
    # Create subgraph nodes with proper einsum equations
    # ...
    
    return subgraph
```

Generate only the function code, no explanations."""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API.
        
        Args:
            prompt: The prompt to send.
            
        Returns:
            LLM response text.
        """
        try:
            # New-style client: client.chat.completions.create(...)
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content

            # Legacy: openai.ChatCompletion.create(...)
            if hasattr(self.client, "ChatCompletion"):
                response = self.client.ChatCompletion.create(
                model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
            
            # Fallback: create a new client if available.
            if OpenAI is not None:
                client = OpenAI(api_key=self.config.api_key)
                response = client.chat.completions.create(
                model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
            )
                return response.choices[0].message.content
            
            raise RuntimeError("OpenAI client is not available")
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response.
        
        Args:
            response: Raw LLM response.
            
        Returns:
            Extracted code.
        """
        # Look for code blocks
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        
        return code.strip()
    
    def _validate_code(self, code: str, node_type: str) -> str:
        """Validate and clean generated code.
        
        Args:
            code: Generated code.
            node_type: Node type name.
            
        Returns:
            Validated code.
        """
        # Basic validation - check if it's a valid function
        if not code.startswith("def "):
            raise ValueError("Generated code is not a valid function")
            
        # Check function name
        expected_name = f"create_{node_type}_subgraph"
        if expected_name not in code:
            # Try to fix the function name
            code = code.replace(
                code.split("(")[0].replace("def ", ""),
                expected_name
            )
        
        return code
    
    def _generate_fallback(self, node_type: str) -> str:
        """Generate fallback implementation.
        
        Args:
            node_type: Node type name.
            
        Returns:
            Fallback code.
        """
        return f"""def create_{node_type}_subgraph(node_id: str, node_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Fallback: treat as identity/passthrough operation
    subgraph = {{}}
    
    input_shapes = node_data.get("input_shapes", [])
    output_shapes = node_data.get("output_shapes", [])
    
    # Create identity node
    subgraph[f"{{node_id}}_identity"] = {{
        "node_type": "identity",
        "input_shapes": input_shapes,
        "output_shapes": output_shapes or input_shapes,
        "module_args": node_data.get("module_args", {{}})
    }}
    
    return subgraph"""
    
    def _check_cache(self, node_type: str) -> Optional[str]:
        """Check if code is cached.
    
    Args:
            node_type: Node type to check.
        
    Returns:
            Cached code if found, None otherwise.
    """
        cache_file = Path(self.config.cache_dir) / f"{node_type}.py"
        if cache_file.exists():
            return cache_file.read_text()
        return None
    
    def _cache_code(self, node_type: str, code: str) -> None:
        """Cache generated code.
        
        Args:
            node_type: Node type name.
            code: Code to cache.
        """
        cache_file = Path(self.config.cache_dir) / f"{node_type}.py"
        cache_file.write_text(code)


def get_api_key_interactive() -> str:
    """Get API key interactively from user.
            
        Returns:
        API key string.
        """
    print("\nTo use the LLM agent for unknown node types, an OpenAI API key is required.")
    print("You can get one at: https://platform.openai.com/api-keys")
    print("\nOptions:")
    print("1. Enter your API key now (recommended)")
    print("2. Set the OPENAI_API_KEY environment variable")
    print("3. Disable the agent and use only built-in handlers")
    
    api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
            
    if not api_key:
        print("\nNo API key provided. Agent will be disabled.")
        return ""
    
    # Validate format (basic check)
    if not api_key.startswith("sk-"):
        print("\nWarning: API key doesn't start with 'sk-'. It might be invalid.")
    
    # Offer to save to environment
    save = input("\nSave API key to .env file for future use? (y/n): ").strip().lower()
    if save == 'y':
        env_file = Path(".env")
        with open(env_file, "a") as f:
            f.write(f"\nOPENAI_API_KEY={api_key}\n")
        print(f"API key saved to {env_file}")
    
    return api_key