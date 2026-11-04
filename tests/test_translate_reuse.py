# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe cross-size reuse of translations + invalidation of incompatible cases."""

from solar.translate.reuse import (cache_key, definition_hash, guard_profile,
                                   reuse_decision, semantic_source_hash)

POLY_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def mm(A: Tensor("M,K", dtype="fp32"), B: Tensor("K,N", dtype="fp32")) -> Tensor("M,N"):
    Y = Tensor(M=A.shape[0], N=B.shape[1], dtype="fp32")
    for m in Dim("M", size=A.shape[0]):
        for n in Dim("N", size=B.shape[1]):
            acc = 0.0
            for k in Dim("K", size=A.shape[1]):
                acc += A[m, k] * B[k, n]
            Y[m, n] = acc
    return Y
'''

GQA_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def gqa(GO: Tensor("B,N,D", dtype="fp32"), V: Tensor("B,K,D", dtype="fp32"),
        groups: int) -> Tensor("B,N"):
    Y = Tensor(B=GO.shape[0], N=GO.shape[1], dtype="fp32")
    for b in Dim("B"):
        for n in Dim("N"):
            acc = 0.0
            for d in Dim("D"):
                acc += GO[b, n, d] * V[b, n // groups, d]
            Y[b, n] = acc
    return Y
'''

BRANCHY_DSL = '''
import numpy as np
def run_model(inputs):
    x = inputs["x"]
    if x.shape[0] > 128:
        x = x[:128]
    return {"y": x * 2}
'''

LITERAL_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel

@kernel
def fixed(A: Tensor("M,H", dtype="fp32")) -> Tensor("M,H"):
    Y = Tensor(M=A.shape[0], H=4096, dtype="fp32")
    for m in Dim("M"):
        for h in Dim("H", size=4096):
            Y[m, h] = A[m, h] * 2.0
    return Y
'''

DEFN = {
    "axes": {"batch": {"type": "var"}, "hidden": {"type": "const", "value": 64}},
    "inputs": {"x": {"shape": ["batch", "hidden"], "dtype": "float32"}},
    "outputs": {"y": {"shape": ["batch", "hidden"], "dtype": "float32"}},
    "reference": "def run(x): return x",
}


# --- semantic hashing: comments/docstrings never invalidate the cache ------

def test_semantic_hash_ignores_comments_and_docstrings():
    a = semantic_source_hash(POLY_DSL)
    b = semantic_source_hash('"""A new docstring."""\n# comment\n' + POLY_DSL)
    assert a == b


def test_semantic_hash_changes_on_behavioral_edit():
    assert semantic_source_hash(POLY_DSL) != semantic_source_hash(
        POLY_DSL.replace("acc += A[m, k] * B[k, n]",
                         "acc += A[m, k] + B[k, n]"))


# --- guard profiling --------------------------------------------------------

def test_clean_polymorphic_dsl_has_no_hazards():
    prof = guard_profile(POLY_DSL)
    assert prof.hazards == []
    assert prof.symbolic_dims >= 3 and prof.literal_dims == 0


def test_divisibility_index_arithmetic_detected():
    prof = guard_profile(GQA_DSL)
    assert "divisibility_index_arithmetic" in prof.hazards


def test_shape_dependent_branch_detected():
    prof = guard_profile(BRANCHY_DSL)
    assert "shape_dependent_branch" in prof.hazards


def test_literal_dim_constants_detected():
    prof = guard_profile(LITERAL_DSL)
    assert prof.literal_dims == 1 and prof.baked_dim_constants


# --- reuse decisions ---------------------------------------------------------

def test_symbolic_reuse_across_sizes_for_clean_dsl():
    prof = guard_profile(POLY_DSL)
    mode, reasons = reuse_decision(prof, DEFN, {"batch": 4, "hidden": 64},
                                   {"batch": 512, "hidden": 64})
    assert mode == "symbolic_reuse"


def test_identical_axes_always_reusable():
    prof = guard_profile(GQA_DSL)
    mode, _ = reuse_decision(prof, DEFN, {"batch": 4, "hidden": 64},
                             {"batch": 4, "hidden": 64})
    assert mode == "symbolic_reuse"


def test_hazardous_dsl_requires_per_shape_revalidation():
    prof = guard_profile(GQA_DSL)
    mode, reasons = reuse_decision(prof, DEFN, {"batch": 4, "hidden": 64},
                                   {"batch": 8, "hidden": 64})
    assert mode == "revalidate_per_shape"
    assert "divisibility_index_arithmetic" in reasons


def test_baked_literal_dim_with_varying_axes_requires_revalidation():
    prof = guard_profile(LITERAL_DSL)
    mode, reasons = reuse_decision(prof, DEFN, {"batch": 4, "hidden": 64},
                                   {"batch": 8, "hidden": 64})
    assert mode == "revalidate_per_shape"
    assert "baked_dim_constant_with_varying_axes" in reasons


def test_const_axis_mismatch_rejected():
    prof = guard_profile(POLY_DSL)
    mode, reasons = reuse_decision(prof, DEFN, {"batch": 4, "hidden": 64},
                                   {"batch": 4, "hidden": 128})
    assert mode == "reject"
    assert any("const axis hidden" in r for r in reasons)


# --- cache keys --------------------------------------------------------------

def test_cache_key_sensitivity():
    base = dict(dsl_sha="d", defn_sha="f", axes={"b": 4},
                dtypes={"x": "fp32"}, options={"arch": "B200"},
                skill_rev="abc")
    k0 = cache_key(**base)
    assert k0 == cache_key(**base)  # deterministic
    assert k0 != cache_key(**{**base, "axes": {"b": 8}})
    assert k0 != cache_key(**{**base, "dtypes": {"x": "bf16"}})
    assert k0 != cache_key(**{**base, "skill_rev": "def"})
    assert k0 != cache_key(**{**base, "options": {"arch": "H200_SXM"}})


def test_definition_hash_stable_and_sensitive():
    h = definition_hash(DEFN)
    assert h == definition_hash(dict(DEFN))
    changed = dict(DEFN)
    changed["reference"] = "def run(x): return x + 1"
    assert h != definition_hash(changed)
