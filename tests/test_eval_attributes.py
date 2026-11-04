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

"""Tests for torchview raw_attributes parsing.

The _eval_attributes_string method parses torchview's stringify_attributes()
output, which has the form [[args...], {kwargs...}].  These tests cover
patterns that appear in HuggingFace models (BigBird) which previously
caused 'closing parenthesis ) does not match opening parenthesis [' errors.
"""

import pytest

from solar.graph.torchview_processor import TorchviewProcessor


@pytest.fixture
def processor():
    return TorchviewProcessor(debug=True)


class TestEvalAttributesString:
    """Tests for _eval_attributes_string parsing."""

    def test_simple_tensor_single_arg(self, processor):
        """Baseline: single Tensor arg with nested shape parens."""
        attrs = "[[Tensor(shape=(768, 768), dtype=torch.float32)], {}]"
        args, kwargs = processor._eval_attributes_string(attrs)
        assert args is not None
        assert isinstance(args, list)
        assert len(args) == 1

    def test_tensor_with_scalar_arg(self, processor):
        """Baseline: Tensor followed by a scalar arg."""
        attrs = "[[Tensor(shape=(768, 768), dtype=torch.float32), True], {}]"
        args, kwargs = processor._eval_attributes_string(attrs)
        assert args is not None
        assert True in args

    def test_multi_arg_tensor_with_dtype(self, processor):
        """Tensor with shape AND dtype — regex must match full Tensor(...)."""
        attrs = "[[Tensor(shape=(1024, 32), dtype=torch.int64), [slice(None, None, None), None, None, slice(None, None, None)]], {}]"
        args, kwargs = processor._eval_attributes_string(attrs)
        assert args is not None, (
            "Failed to parse multi-arg Tensor — regex likely truncated "
            "at first ')' inside shape, leaving dangling ', dtype=...)'"
        )

    def test_getitem_with_slice_and_ellipsis(self, processor):
        """BigBird __getitem__: Tensor + [slice(...), [..., ...]]."""
        attrs = "[[Tensor(shape=(1024, 32), dtype=torch.int64), [slice(None, None, None), [Ellipsis, Ellipsis]]], {}]"
        args, kwargs = processor._eval_attributes_string(attrs)
        assert args is not None

    def test_bare_keyword_kwargs(self, processor):
        """kaiming_uniform_ uses bare keywords: {tensor: ..., a: ..., mode: ...}."""
        attrs = "[[], {tensor: Tensor(shape=(768, 768), dtype=torch.float32), a: 2.23606797749979, mode: 'fan_in', nonlinearity: 'leaky_relu', generator: None}]"
        args, kwargs = processor._eval_attributes_string(attrs)
        assert kwargs is not None, (
            "Failed to parse bare-keyword kwargs — unquoted keys like "
            "'tensor:', 'a:' are not valid Python dict syntax for eval()"
        )

    def test_uniform_bare_kwargs(self, processor):
        """uniform_ uses bare keywords: {tensor: ..., a: ..., b: ...}."""
        attrs = "[[], {tensor: Tensor(shape=(768,), dtype=torch.float32), a: -0.036, b: 0.036, generator: None}]"
        args, kwargs = processor._eval_attributes_string(attrs)
        assert kwargs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
