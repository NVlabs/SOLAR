# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rank-token normalizer in equation_utils."""

import pytest

from equation_utils import normalize_equation


@pytest.mark.parametrize("a,b", [
    ("MK,KN->MN", "XY,YZ->XZ"),
    ("AB(C+D),EBD->AEC", "PQ(R+S),TQS->PTR"),
    ("A(X+2)->A(X+2)", "B(Y+2)->B(Y+2)"),
])
def test_renaming_equivalent_equations_match(a, b):
    assert normalize_equation(a) == normalize_equation(b)


@pytest.mark.parametrize("a,b", [
    ("MK,KN->MN", "MK,KN->NM"),
    ("A(X+2)->A(X+2)", "A(X+3)->A(X+3)"),
    ("A(X-2)->A(X-2)", "A(X+2)->A(X+2)"),
    ("A(X+2)->A", "A(X+2)->A(X+2)"),
])
def test_structurally_different_equations_differ(a, b):
    assert normalize_equation(a) != normalize_equation(b)


def test_docstring_convolution_example():
    assert normalize_equation("BC(P+R)(Q+S),OCRS->BOPQ") == "AB(C+D)(E+F),GBDF->AGCE"


def test_integer_constant_is_preserved_verbatim():
    assert normalize_equation("A(X+2)->A(X+2)") == "A(B+2)->A(B+2)"


@pytest.mark.parametrize("eq", ["1A->A", "A->1A", "A,1->A", "A1->A"])
def test_standalone_one_is_rejected_not_silently_dropped(eq):
    with pytest.raises(ValueError):
        normalize_equation(eq)
