# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate consumed-generator control without the existing unsupported list() callee."""
import pytest
import test_round4_scope as original

@pytest.mark.parametrize('fp64',[False,True])
def test_consumed_generator_scope(fp64):
    name='generator_local_consumed_by_for'
    original.CASES[name]=(original.program('    g=(Tensor for Tensor in range(3))\n    for item in g:\n        pass\n'),'prescan')
    original.test_round4_scope(name,fp64)
