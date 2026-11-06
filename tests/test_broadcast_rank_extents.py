# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A size-1 (broadcast) rank extent must not mask the real extent (GitHub issue #13).

``compute_cost_from_equation`` bills the product of every unique rank.  It used
to take each rank's extent from its FIRST occurrence, so a rank that is size 1
in an earlier operand and N in a later one was billed as 1:
``[2, 1, 4, 5] @ [3, 5, 6]`` gave 240 MACs instead of 720.  Under broadcast
semantics the larger extent wins; a size-1 extent never replaces a larger one.
"""
import pytest

from solar.common.types import TensorShapes
from solar.einsum.ops.base import compute_cost_from_equation


def _cost(eq, inputs, outputs):
    return compute_cost_from_equation(eq, TensorShapes(inputs=inputs, outputs=outputs))


def test_issue_example_size1_batch_rank_in_first_operand():
    # [2,1,4,5] @ [3,5,6] -> [2,3,4,6]: 2*3*4*5*6
    eq = "ABMK,BKN->ABMN"
    assert _cost(eq, [[2, 1, 4, 5], [3, 5, 6]], [[2, 3, 4, 6]]) == 720


def test_size1_in_second_operand_unchanged():
    eq = "BMK,ABKN->ABMN"
    assert _cost(eq, [[3, 4, 5], [2, 1, 5, 6]], [[2, 3, 4, 6]]) == 720


def test_both_operand_orders_agree():
    a = _cost("ABMK,BKN->ABMN", [[2, 1, 4, 5], [3, 5, 6]], [[2, 3, 4, 6]])
    b = _cost("BKN,ABMK->ABMN", [[3, 5, 6], [2, 1, 4, 5]], [[2, 3, 4, 6]])
    assert a == b == 720


def test_elementwise_outer_broadcast_counts_every_output_element():
    assert _cost("AB,AB->AB", [[4, 1], [1, 5]], [[4, 5]]) == 20
    assert _cost("AB,AB->AB", [[1, 5], [4, 1]], [[4, 5]]) == 20


def test_size1_only_in_output_still_bills_one():
    # keepdim reduction: the size-1 output rank comes after the inputs
    assert _cost("AB->AB", [[4, 5]], [[4, 1]]) == 20


def test_size1_output_never_shrinks_a_resolved_rank():
    assert _cost("AB->A", [[4, 5]], [[4]]) == 20


def test_non_broadcast_cases_unchanged():
    assert _cost("MK,KN->MN", [[3, 5], [5, 7]], [[3, 7]]) == 105
    assert _cost("BMK,BKN->BMN", [[2, 3, 5], [2, 5, 7]], [[2, 3, 7]]) == 210


def test_genuine_size_one_everywhere_stays_one():
    assert _cost("MK,KN->MN", [[1, 5], [5, 1]], [[1, 1]]) == 5
