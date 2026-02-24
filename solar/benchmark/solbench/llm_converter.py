#!/usr/bin/env python3
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

"""
LLM-based converter for SolBench files to KernelBench format.

Uses OpenAI API to intelligently convert SolBench files into the format required
by torchview for compute graph extraction.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import time

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert Python programmer specializing in PyTorch model conversion. Your task is to convert SolBench benchmark files into KernelBench format for torchview compatibility.

## Required Output Format

The converted file must have these components:

1. **Model class**: A PyTorch nn.Module with __init__ and forward methods
2. **get_init_inputs()**: Returns arguments for Model.__init__()
3. **get_inputs()**: Returns arguments for model.forward()

## How torchview Uses These Components

```python
# torchview calls it like this:
init_inputs = get_init_inputs()  # Get initialization arguments
model = Model(*init_inputs)      # Create model instance

inputs = get_inputs()            # Get forward pass inputs
graph = torchview.draw_graph(    # Generate compute graph
    model,
    input_data=inputs,           # Unpack inputs to forward()
    ...
)
```

## Critical Requirements

1. **get_inputs() must match forward() signature**:
   - If forward(self, x, y, z), then get_inputs() must return (x, y, z)
   - If forward(self, *args), then get_inputs() can return any tuple

2. **Preserve original computation**:
   - Model class must wrap the original reference implementation
   - No behavioral changes to the computation logic

3. **Handle parameter sources correctly**:
   - Analyze main() to see how launch_reference_implementation is called
   - Distinguish between:
     * Variables from get_inputs() (e.g., x, y from `x, y = get_inputs()`)
     * Global constants (e.g., BATCH_SIZE, EPS)
     * Derived values (e.g., `dtype = x.dtype`)
   - Generate get_inputs() that provides ALL parameters needed by forward()

4. **CRITICAL - Preserve dtype configuration**:
   - Check if main() sets torch.set_default_dtype() (e.g., torch.bfloat16, torch.float16)
   - If found, add explicit dtype to ALL tensor creation in get_inputs()
   - Examples:
     * torch.randn(...) → torch.randn(..., dtype=torch.bfloat16)
     * torch.zeros(...) → torch.zeros(..., dtype=torch.bfloat16)
     * torch.ones(...) → torch.ones(..., dtype=torch.bfloat16)
   - Exception: Integer tensors (torch.long, torch.int) should keep their dtype
   - This ensures inputs match model weights dtype when SOLAR loads the module

## Conversion Strategy

### Step 1: Analyze the source file
- Locate main() function
- Find get_inputs() and what it returns
- Find launch_reference_implementation() and its signature
- Understand how launch_reference_implementation is called in main()

### Step 2: Design Model class
- Extract model/function instantiation from main() → Model.__init__()
- Extract launch_reference_implementation() parameters → Model.forward()
- Remove model parameter if present (first param that's a model instance)
- Replace model references with self.<attr>

### Step 3: Design get_inputs() function
- Analyze ALL parameters needed by forward()
- For each parameter, determine its source:
  * From original get_inputs(): keep as-is
  * Global constant: include in new get_inputs()
  * Derived from get_inputs() output: compute in new get_inputs()
- Generate get_inputs() that returns tuple matching forward() signature

### Step 4: Design get_init_inputs() function
- Extract initialization parameters from main()
- Return empty list if Model.__init__ takes no arguments
- Return list/tuple of arguments if Model needs initialization params

## Output Format

Return ONLY the converted Python code with NO markdown formatting, NO explanations, NO ```python blocks. Just the raw Python code that can be directly appended to the original file.

The output should start with:

# ===== KernelBench-compatible Model =====

And contain the Model class, get_init_inputs(), and get_inputs() functions."""


def get_conversion_examples() -> str:
    """Get example conversions to include in the prompt."""
    return """## Example Conversions

### Example 1: Forward Pass with Class

**Source (relevant parts):**
```python
class GroupedQueryCrossAttention(nn.Module):
    def __init__(self):
        # ... model definition

def launch_reference_implementation(model: GroupedQueryCrossAttention, *inputs):
    return model(*inputs)

def get_inputs():
    hidden_states = torch.randn(BATCH_SIZE, SEQ_Q, HIDDEN_SIZE)
    encoder_hidden_states = torch.randn(BATCH_SIZE, SEQ_K, CROSS_HIDDEN_SIZE)
    attention_mask = torch.randn(BATCH_SIZE, 1, SEQ_Q, SEQ_K)
    return hidden_states, encoder_hidden_states, attention_mask

def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16)

    ref_model = GroupedQueryCrossAttention()
    hidden_states, encoder_hidden_states, attention_mask = get_inputs()

    ref_attn_output, ref_attn_weights = launch_reference_implementation(
        ref_model, hidden_states, encoder_hidden_states, attention_mask
    )
```

**Converted Output:**
```python
# ===== KernelBench-compatible Model =====
class Model(nn.Module):
    \"\"\"Model wrapper for torchview compatibility.\"\"\"

    def __init__(self):
        super().__init__()
        self.ref_model = GroupedQueryCrossAttention()

    def forward(self, hidden_states, encoder_hidden_states, attention_mask):
        \"\"\"Forward pass.\"\"\"
        return self.ref_model(hidden_states, encoder_hidden_states, attention_mask)


def get_init_inputs():
    \"\"\"Return initialization inputs for Model.\"\"\"
    return []


def get_inputs():
    \"\"\"Generate input tensors for forward pass.\"\"\"
    # Note: Explicit dtype=torch.float16 to match torch.set_default_dtype(torch.float16) from main()
    hidden_states = torch.randn(BATCH_SIZE, SEQ_Q, HIDDEN_SIZE, dtype=torch.float16)
    encoder_hidden_states = torch.randn(BATCH_SIZE, SEQ_K, CROSS_HIDDEN_SIZE, dtype=torch.float16)
    attention_mask = torch.randn(BATCH_SIZE, 1, SEQ_Q, SEQ_K, dtype=torch.float16)
    return hidden_states, encoder_hidden_states, attention_mask
```

### Example 2: Backward Pass with Additional Parameters

**Source (relevant parts):**
```python
def reference_backward(grad_output, x_fp32, rstd, x_norm, weight, eps, og_dtype):
    # ... computation
    return grad_x, grad_weight, grad_bias

def launch_reference_implementation(grad_output, x_fp32, rstd, x_norm, weight, eps, og_dtype):
    return reference_backward(grad_output, x_fp32, rstd, x_norm, weight, eps, og_dtype)

BATCH_SIZE = 32
SEQ_LEN = 2048
D_MODEL = 4096
EPS = 1e-05

def get_inputs():
    og_dtype = torch.float16
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, dtype=og_dtype)
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + EPS)
    x_norm_fp32 = x_fp32 * rstd
    x_norm = x_norm_fp32.to(og_dtype)
    weight = torch.randn(D_MODEL, dtype=og_dtype)
    grad_output = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, dtype=og_dtype)
    return grad_output, x_fp32, rstd, x_norm, weight

def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)

    grad_output, x_fp32, rstd, x_norm, weight = get_inputs()
    og_dtype = torch.float16

    ref_grad_x, ref_grad_weight, ref_grad_bias = launch_reference_implementation(
        grad_output, x_fp32, rstd, x_norm, weight, EPS, og_dtype
    )
```

**Converted Output:**
```python
# ===== KernelBench-compatible Model =====
class Model(nn.Module):
    \"\"\"Model wrapper for torchview compatibility.\"\"\"

    def __init__(self):
        super().__init__()
        pass

    def forward(self, grad_output, x_fp32, rstd, x_norm, weight, eps, og_dtype):
        \"\"\"Forward pass.\"\"\"
        return reference_backward(grad_output, x_fp32, rstd, x_norm, weight, eps, og_dtype)


def get_init_inputs():
    \"\"\"Return initialization inputs for Model.\"\"\"
    return []


def get_inputs():
    \"\"\"Generate input tensors for forward pass.

    Note: This now includes EPS and og_dtype to match forward() signature.
    \"\"\"
    og_dtype = torch.float16
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, dtype=og_dtype)
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + EPS)
    x_norm_fp32 = x_fp32 * rstd
    x_norm = x_norm_fp32.to(og_dtype)
    weight = torch.randn(D_MODEL, dtype=og_dtype)
    grad_output = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, dtype=og_dtype)

    # Include additional parameters needed by forward
    eps = EPS  # Global constant
    # og_dtype already defined above

    return grad_output, x_fp32, rstd, x_norm, weight, eps, og_dtype
```

### Example 3: Backward Pass with Class

**Source (relevant parts):**
```python
class ReferenceBackward:
    def __init__(self, mm_tokens_per_image: int, sliding_window: int):
        self.mm_tokens_per_image = mm_tokens_per_image
        self.sliding_window = sliding_window

    def backward(self, grad_mask, token_type_ids, image_group_ids):
        # ... computation
        return grad_token_type_ids, grad_cache_position

def launch_reference_implementation(
    reference_backward: ReferenceBackward,
    grad_mask: Optional[torch.Tensor],
    token_type_ids: torch.Tensor,
    image_group_ids: torch.Tensor,
):
    return reference_backward.backward(grad_mask, token_type_ids, image_group_ids)

MM_TOKENS_PER_IMAGE = 256
SLIDING_WINDOW = 4096

def get_inputs():
    token_type_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    token_type_ids[:, :MM_TOKENS_PER_IMAGE] = 1
    # ... compute image_group_ids
    grad_mask = torch.randn(BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    return grad_mask, token_type_ids, image_group_ids

def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)

    reference_backward = ReferenceBackward(
        mm_tokens_per_image=MM_TOKENS_PER_IMAGE,
        sliding_window=SLIDING_WINDOW
    )

    grad_mask, token_type_ids, image_group_ids = get_inputs()

    ref_grad_token_type_ids, ref_grad_cache_position = launch_reference_implementation(
        reference_backward,
        grad_mask,
        token_type_ids,
        image_group_ids
    )
```

**Converted Output:**
```python
# ===== KernelBench-compatible Model =====
class Model(nn.Module):
    \"\"\"Model wrapper for torchview compatibility.\"\"\"

    def __init__(self):
        super().__init__()
        self.reference_backward = ReferenceBackward(
            mm_tokens_per_image=MM_TOKENS_PER_IMAGE,
            sliding_window=SLIDING_WINDOW
        )

    def forward(self, grad_mask, token_type_ids, image_group_ids):
        \"\"\"Forward pass.\"\"\"
        return self.reference_backward.backward(grad_mask, token_type_ids, image_group_ids)


def get_init_inputs():
    \"\"\"Return initialization inputs for Model.\"\"\"
    return []


def get_inputs():
    \"\"\"Generate input tensors for forward pass.\"\"\"
    token_type_ids = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    token_type_ids[:, :MM_TOKENS_PER_IMAGE] = 1

    # Compute image_group_ids (as would be done in forward pass)
    device = token_type_ids.device
    is_image = (token_type_ids == 1).to(device)
    padded_is_image = nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
    new_image_start = is_image & ~padded_is_image
    image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
    image_group_ids = torch.where(
        is_image,
        image_group_ids,
        torch.full_like(token_type_ids, -1, device=device)
    )

    grad_mask = torch.randn(BATCH_SIZE, SEQ_LEN, SEQ_LEN)

    return grad_mask, token_type_ids, image_group_ids
```

## Key Points from Examples

1. **Parameter matching**: forward() signature MUST match get_inputs() return values
2. **Global constants**: Include them in get_inputs() if needed by forward()
3. **Derived values**: Compute them in get_inputs() if needed by forward()
4. **Model instantiation**: Move from main() to Model.__init__()
5. **Preserve computation**: Keep the original logic intact"""


class LLMConverter:
    """LLM-based converter using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", verbose: bool = False):
        """Initialize the converter.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-4o)
            verbose: Enable verbose output
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.verbose = verbose

        # Stats
        self.total_tokens = 0
        self.total_cost = 0.0

    def convert_file(self, source_path: Path, output_path: Path) -> Tuple[bool, Optional[str]]:
        """Convert a single file.

        Args:
            source_path: Path to source file
            output_path: Path to output file

        Returns:
            (success: bool, error_message: Optional[str])
        """
        try:
            if self.verbose:
                print(f"Converting: {source_path.name}")

            # Read source file
            with open(source_path, 'r') as f:
                source_code = f.read()

            # Call LLM to generate conversion
            converted_code = self._call_llm(source_code, source_path.name)

            if not converted_code:
                return False, "LLM returned empty response"

            # Validate conversion
            validation_error = self._validate_conversion(converted_code)
            if validation_error:
                return False, f"Validation failed: {validation_error}"

            # Write output: original + converted
            output_code = source_code + "\n\n" + converted_code
            with open(output_path, 'w') as f:
                f.write(output_code)

            if self.verbose:
                print(f"  ✓ Success")

            return True, None

        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                print(f"  ✗ Error: {error_msg}")
            return False, error_msg

    def _call_llm(self, source_code: str, filename: str) -> str:
        """Call OpenAI API to convert the code.

        Args:
            source_code: Source code to convert
            filename: Name of the file (for context)

        Returns:
            Converted code
        """
        examples = get_conversion_examples()

        user_prompt = f"""Convert the following SolBench file to KernelBench format for torchview compatibility.

File: {filename}

Source code:
```python
{source_code}
```

Remember:
1. Output ONLY the Python code (no markdown, no explanations)
2. Start with: # ===== KernelBench-compatible Model =====
3. Ensure get_inputs() returns values matching forward() signature EXACTLY
4. Include all necessary parameters (from get_inputs, globals, derived values)
5. Preserve the original computation logic
6. **CRITICAL**: Check main() for torch.set_default_dtype() and add explicit dtype to ALL tensor creation in get_inputs():
   - If main() has torch.set_default_dtype(torch.bfloat16), add dtype=torch.bfloat16 to torch.randn(), torch.zeros(), torch.ones(), etc.
   - If main() has torch.set_default_dtype(torch.float16), add dtype=torch.float16
   - Keep integer dtypes (torch.long, torch.int) unchanged

Return the converted code now:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + examples},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=4096,
            )

            # Update stats
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens
                # Rough cost estimation (GPT-4o rates as of 2024)
                prompt_cost = response.usage.prompt_tokens * 0.0025 / 1000
                completion_cost = response.usage.completion_tokens * 0.01 / 1000
                self.total_cost += prompt_cost + completion_cost

            converted_code = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if converted_code.startswith("```python"):
                converted_code = converted_code[len("```python"):].strip()
            if converted_code.startswith("```"):
                converted_code = converted_code[3:].strip()
            if converted_code.endswith("```"):
                converted_code = converted_code[:-3].strip()

            return converted_code

        except Exception as e:
            if self.verbose:
                print(f"  LLM API error: {e}")
            raise

    def _validate_conversion(self, code: str) -> Optional[str]:
        """Validate the converted code.

        Args:
            code: Converted code

        Returns:
            Error message if validation fails, None if valid
        """
        # Check for required components
        if "class Model(nn.Module)" not in code:
            return "Missing Model class definition"

        if "def get_init_inputs()" not in code:
            return "Missing get_init_inputs() function"

        if "def get_inputs()" not in code:
            return "Missing get_inputs() function"

        if "def forward(self" not in code:
            return "Missing forward() method in Model class"

        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return f"Syntax error: {e}"

        return None

    def get_stats(self) -> dict:
        """Get conversion statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert SolBench files to KernelBench format using LLM"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--max-files", type=int, help="Maximum files to convert")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually convert, just list files")

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Create converter
    try:
        converter = LLMConverter(api_key=args.api_key, model=args.model, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Find all Python files
    input_files = sorted(args.input_dir.glob("*.py"))

    if args.max_files:
        input_files = input_files[:args.max_files]

    if not input_files:
        print(f"No Python files found in {args.input_dir}")
        return 1

    if args.dry_run:
        print(f"Would convert {len(input_files)} files:")
        for f in input_files:
            print(f"  - {f.name}")
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert files
    succeeded = 0
    failed = 0
    failed_files = []

    print(f"Converting {len(input_files)} files with {args.model}...")
    print()

    for i, input_file in enumerate(input_files, 1):
        output_file = args.output_dir / input_file.name

        print(f"[{i}/{len(input_files)}] {input_file.name}")

        success, error = converter.convert_file(input_file, output_file)

        if success:
            succeeded += 1
        else:
            failed += 1
            failed_files.append((input_file.name, error))
            if not args.verbose:
                print(f"  ✗ Failed: {error}")

        # Rate limiting - be nice to the API
        if i < len(input_files):
            time.sleep(0.5)

    # Print summary
    print()
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total:     {len(input_files)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")

    stats = converter.get_stats()
    print(f"\nAPI Usage:")
    print(f"  Tokens: {stats['total_tokens']:,}")
    print(f"  Cost:   ${stats['total_cost']:.2f}")

    if failed_files:
        print(f"\nFailed files:")
        for fname, error in failed_files:
            print(f"  - {fname}: {error}")

    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
