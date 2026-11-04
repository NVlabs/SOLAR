# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work-minimal DSL translation: counting contracts and the prompts that ask for it.

The loop nest of a translated kernel is the ground truth Solar bills, so the
translation must (a) be counted at the extent it actually loops and (b) bill
static masks at the work they leave live.  Three groups of tests:

* L1/029 regression: a half-range projection that indexes its weight with the
  bare loop variable (``W[i, h]``, ``Dim("I", size=W.shape[0] // 2)``) was
  billed at the FULL weight axis by checkouts that predate BinOp size-spec
  resolution (an earlier revision; this MR), i.e. 2x the identical gate half (7 SOL "violations" in the
  v2 all-shapes crosscheck).  Pinned so the repair cannot silently regress.
* Static-mask billing: an explicit analytic ``sparsity=`` on a 0/1 mask tensor
  is what makes zeros free; the same mask hidden in an arithmetic guard inside
  the kernel body is billed dense.  This is why the translation prompts ask for
  the mask-tensor form.
* Prompt/skill text: the API pipeline preambles and the agent-mode skill both
  carry the work-minimal and static-mask rules (and keep "Do NOT optimize").
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SOLAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR))

from test_sparse_cost import _analyze, _traced_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# L1/029: half-range projection with a bare loop-variable weight index
# ---------------------------------------------------------------------------

B, L, H, P = 2, 8, 16, 32          # P = 2 * I rows of the fused in_proj
I = P // 2

HALF_RANGE_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def proj_conv_masked(X: Tensor("B,L,H", dtype="fp32"),
                     W: Tensor("P,H", role="weight", dtype="fp32"),
                     bias: Tensor("P", role="weight", dtype="fp32"),
                     M: Tensor("B,L", dtype="fp32")) -> Tensor("B,I,L"):
    half = W.shape[0] // 2
    Y = Tensor(B=X.shape[0], I=half, L=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for i in Dim("I", size=half):
            for l in Dim("L"):
                acc = bias[i]
                for h in Dim("H"):
                    acc += X[b, l, h] * W[i, h]
                Y[b, i, l] = acc * M[b, l]
    return Y


@kernel
def proj_gate(X: Tensor("B,L,H", dtype="fp32"),
              W: Tensor("P,H", role="weight", dtype="fp32"),
              bias: Tensor("P", role="weight", dtype="fp32")) -> Tensor("B,I,L"):
    half = W.shape[0] // 2
    Y = Tensor(B=X.shape[0], I=half, L=X.shape[1], dtype="fp32")
    for b in Dim("B", parallel=True):
        for i in Dim("I", size=half):
            for l in Dim("L"):
                acc = bias[half + i]
                for h in Dim("H"):
                    acc += X[b, l, h] * W[half + i, h]
                Y[b, i, l] = acc
    return Y


def run_model(inputs):
    clear_dims()
    x = Tensor.from_numpy(inputs["x"], "B,L,H", dtype="fp32")
    w = Tensor.from_numpy(inputs["w"], "P,H", role="weight", dtype="fp32")
    bias = Tensor.from_numpy(inputs["bias"], "P", role="weight", dtype="fp32")
    m = Tensor.from_numpy(inputs["m"], "B,L", dtype="fp32")
    return {"conv": proj_conv_masked(x, w, bias, m).data,
            "gate": proj_gate(x, w, bias).data}
'''


def _half_range_inputs():
    rng = np.random.default_rng(0)
    return {
        "x": rng.standard_normal((B, L, H)),
        "w": rng.standard_normal((P, H)),
        "bias": rng.standard_normal((P,)),
        "m": np.ones((B, L)),
    }


def test_half_range_projection_is_counted_at_the_loop_extent(tmp_path):
    md = _traced_metadata(HALF_RANGE_DSL, _half_range_inputs(), profile=False)
    layers = _analyze(HALF_RANGE_DSL, md, tmp_path)["layers"]
    conv = layers["proj_conv_masked#0"]
    gate = layers["proj_gate#0"]
    # Identical work: B*L*H MACs for each of the I rows of its half.
    assert gate["macs"] == B * L * H * I
    assert conv["macs"] == B * L * H * I, (
        "bare-index half-range loop billed at the full weight axis "
        f"({conv['macs']} vs {B * L * H * I}); BinOp size-spec resolution "
        "is missing or regressed")


# ---------------------------------------------------------------------------
# Static masks: what the perf count bills
# ---------------------------------------------------------------------------

S, T, D = 32, 32, 16
ANALYTIC_DENSITY = 0.21875         # original-scale causal window density (S=4096, W=1024)

MASK_TENSOR_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def banded_scores(Q: Tensor("S,D", dtype="fp32"),
                  K: Tensor("T,D", dtype="fp32"),
                  B: Tensor("S,T", dtype="fp32")) -> Tensor("S,T"):
    Y = Tensor(S=Q.shape[0], T=K.shape[0], dtype="fp32")
    for s in Dim("S", parallel=True):
        for t in Dim("T"):
            acc = 0.0
            for d in Dim("D"):
                acc += Q[s, d] * K[t, d]
            Y[s, t] = acc * B[s, t]
    return Y


def run_model(inputs):
    clear_dims()
    q = Tensor.from_numpy(inputs["q"], "S,D", dtype="fp32")
    k = Tensor.from_numpy(inputs["k"], "T,D", dtype="fp32")
    # static mask built from shapes in run_model, analytic ORIGINAL-scale density
    band = Tensor.from_numpy(inputs["band"], "S,T", dtype="fp32", sparsity=%r)
    return {"output": banded_scores(q, k, band).data}
''' % ANALYTIC_DENSITY

ARITHMETIC_GUARD_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims


@kernel
def banded_scores(Q: Tensor("S,D", dtype="fp32"),
                  K: Tensor("T,D", dtype="fp32")) -> Tensor("S,T"):
    Y = Tensor(S=Q.shape[0], T=K.shape[0], dtype="fp32")
    for s in Dim("S", parallel=True):
        for t in Dim("T"):
            acc = 0.0
            for d in Dim("D"):
                acc += Q[s, d] * K[t, d]
            Y[s, t] = acc * max(0.0, min(1.0, s - t + 1.0))
    return Y


def run_model(inputs):
    clear_dims()
    q = Tensor.from_numpy(inputs["q"], "S,D", dtype="fp32")
    k = Tensor.from_numpy(inputs["k"], "T,D", dtype="fp32")
    return {"output": banded_scores(q, k).data}
'''


def _band_inputs():
    rng = np.random.default_rng(1)
    band = np.tril(np.ones((S, T)))
    return {"q": rng.standard_normal((S, D)), "k": rng.standard_normal((T, D)),
            "band": band}


def test_explicit_analytic_density_on_a_static_mask_tensor_is_billed(tmp_path):
    md = _traced_metadata(MASK_TENSOR_DSL, _band_inputs(), profile=False)
    layer = _analyze(MASK_TENSOR_DSL, md, tmp_path)["layers"]["banded_scores#0"]
    # The capped mask is a full lower triangle (~0.52 dense); the explicit
    # analytic original-scale density wins and sets the billed work.
    assert layer["compute_density"] == pytest.approx(ANALYTIC_DENSITY)
    assert layer["macs"] == pytest.approx(S * T * D * ANALYTIC_DENSITY, rel=0.01)


def test_static_mask_hidden_in_an_arithmetic_guard_is_billed_dense(tmp_path):
    inputs = _band_inputs()
    inputs.pop("band")
    md = _traced_metadata(ARITHMETIC_GUARD_DSL, inputs, profile=False)
    layer = _analyze(ARITHMETIC_GUARD_DSL, md, tmp_path)["layers"]["banded_scores#0"]
    assert "compute_density" not in layer
    assert layer["macs"] == S * T * D


# ---------------------------------------------------------------------------
# Prompts: API pipeline preambles + agent-mode skill
# ---------------------------------------------------------------------------

SKILLS = SOLAR / "skills" / "verified-translation"


def _preambles():
    from solar.einsum_lib import agent
    return {"pytorch": agent._PREAMBLE, "jax": agent._PREAMBLE_JAX,
            "triton": agent._PREAMBLE_TRITON}


@pytest.mark.parametrize("which", ["pytorch", "jax", "triton"])
def test_api_preamble_asks_for_work_minimal_translation(which):
    text = _preambles()[which]
    # the existing contract stays
    assert "Do NOT optimize" in text
    # and the work-minimal rule qualifies it
    assert "WORK-MINIMAL TRANSLATION" in text
    assert "most efficient" in text and "functionally correct" in text
    assert "not a licence to change numerics" in text or "correctness wins" in text


@pytest.mark.parametrize("which", ["pytorch", "jax", "triton"])
def test_api_preamble_asks_for_the_most_efficient_static_mask_form(which):
    text = _preambles()[which]
    assert "STATIC MASKS" in text
    assert "sparsity=" in text                       # explicit analytic density
    assert "ORIGINAL" in text                        # not the capped-shape density
    assert "arithmetic guard" in text                # do not hide the mask
    assert "data-dependent" in text                  # never claim sparsity for those


def test_agent_mode_skill_carries_the_same_rules():
    skill = (SKILLS / "SKILL.md").read_text()
    ref = (SKILLS / "reference.md").read_text()
    assert "Work-minimal translation" in skill
    assert "Static masks" in skill
    for needle in ("Work-minimal translation", "Static masks", "sparsity=",
                   "arithmetic guard", "data-dependent"):
        assert needle in ref, needle


# ---------------------------------------------------------------------------
# Size-spec bindings that must NOT be accepted (reviewer findings on the port).
# Contract since opaque-size-20: an explicit ``size=`` the parser cannot
# reduce is preserved as an ("opaque", text, line, col) spec and the converter
# refuses it at the call; ``None`` is reserved for "no explicit size".
# ---------------------------------------------------------------------------

from solar.einsum_dsl.ast_parser import extract_kernel_info  # noqa: E402

_BINDING_HEADER = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
DIVISOR = 2
'''


def _i_size_spec(tmp_path, body: str, params: str = 'X: Tensor("N", dtype="fp32")'):
    src = _BINDING_HEADER + f'''

@kernel
def k({params}) -> Tensor("I"):
{body}
    return Y
'''
    path = tmp_path / "dsl.py"
    path.write_text(src)
    info = extract_kernel_info(path)["k"]
    specs = {name: spec for p, _w in (info["loop_paths"] or []) for name, spec in p}
    return specs.get("I", "no-path")


from solar.einsum_dsl.einsum_converter import (  # noqa: E402
    LOOP_DIM_SCALAR_UNRESOLVED, LoopDimExtentMismatch)

OPAQUE_REFUSAL = "loop_dim_extent_unresolved"


def _assert_opaque(tmp_path, spec, text):
    """The parser's contract for an explicit ``size=`` it cannot reduce (opaque-size-20): the expression is
    PRESERVED as ``("opaque", <source text>, line, col)`` pointing at the ``Dim("I", size=...)`` header written to
    ``tmp_path/dsl.py`` -- never ``None``, which means "no explicit size" and would let the registry binding of the
    name (or the positional fallback) size the loop."""
    assert isinstance(spec, tuple) and len(spec) == 4 and spec[0] == "opaque", spec
    assert spec[1] == text, spec
    line = (tmp_path / "dsl.py").read_text().splitlines()[spec[2] - 1]
    assert 'Dim("I", size=' in line and line[spec[3]:].startswith(text), (spec, line)
    return spec


def _refused(kernel, code=OPAQUE_REFUSAL):
    """The converter refuses the kernel's loop size with a typed ``LoopDimExtentMismatch`` carrying *code*."""
    with pytest.raises(LoopDimExtentMismatch) as excinfo:
        _kernel_loop_iterations(kernel, {})
    assert excinfo.value.code == code, str(excinfo.value)
    return excinfo.value


def _assert_refused_at_the_call(spec, code=OPAQUE_REFUSAL, param_names=("X",), tensor_param_count=1,
                                input_shapes=(8,)):
    """Downstream of the parser the converter REFUSES the spec at the call (header-fallback-19) instead of
    sizing the loop from the registry binding of the name, the positional fallback or any assumed value."""
    k = _kernel_with(list(param_names), tensor_param_count, list(input_shapes), spec)
    return _refused(k, code)


def _assert_scalar_parameter_sizes_the_loop(spec):
    """A ``("scalar", name)`` leaf is the KERNEL PARAMETER, resolved from the call's recorded ``scalar_args`` (the
    tracer records int arguments and defaults per call; TARGET-KB-MAXPOOL round 4 / SCALAR-EXTENT-18) -- never the
    module constant ``DIVISOR = 2`` it shadows -- and a trace that did not record the scalar is refused
    (``loop_dim_scalar_unresolved``) whether or not the registry binds the name."""
    assert spec == ("binop", "//", ("shape", "X", 0), ("scalar", "DIVISOR")), spec
    _assert_refused_at_the_call(spec, code=LOOP_DIM_SCALAR_UNRESOLVED, param_names=("X", "DIVISOR"))
    k = _kernel_with(["X", "DIVISOR"], 1, [8], spec)
    k.scalar_args = {"DIVISOR": 4}
    assert _kernel_loop_iterations(k, {})[0] == 8 // 4       # the parameter (4), not the module constant (2 -> 4)


def test_half_range_binding_resolves_to_a_binop_spec(tmp_path):
    body = '''    n = X.shape[0] // 2
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    assert _i_size_spec(tmp_path, body) == ("binop", "//", ("shape", "X", 0), 2)


@pytest.mark.parametrize("rebinding", [
    "    n *= 2\n",                                   # augmented assignment
    "    n = max(n, X.shape[0])\n",                   # reassignment to an unresolvable value
    "    n = n + X.shape[0] - n // 1 + 0 if False else n\n",  # conditional expression (unresolvable)
])
def test_rebinding_after_a_resolved_size_invalidates_it(tmp_path, rebinding):
    body = '''    n = X.shape[0] // 2
''' + rebinding + '''    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


def test_assignment_inside_a_branch_is_not_accepted(tmp_path):
    body = '''    n = X.shape[0]
    if False:
        n = X.shape[0] // 2
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


def test_kernel_parameter_shadows_a_module_constant(tmp_path):
    body = '''    Y = Tensor(I=X.shape[0] // DIVISOR, dtype="fp32")
    for i in Dim("I", size=X.shape[0] // DIVISOR):
        Y[i] = X[i * 4]
'''
    params = 'X: Tensor("N", dtype="fp32"), DIVISOR: int = 4'
    _assert_scalar_parameter_sizes_the_loop(_i_size_spec(tmp_path, body, params))


def test_loop_invariant_binding_inside_an_outer_dim_loop_is_still_accepted(tmp_path):
    body = '''    Y = Tensor(B=X.shape[0], I=X.shape[0] // 2, dtype="fp32")
    for b in Dim("B"):
        n = X.shape[0] // 2
        for i in Dim("I", size=n):
            Y[b, i] = X[i] * X[b]
'''
    assert _i_size_spec(tmp_path, body) == ("binop", "//", ("shape", "X", 0), 2)


# ---------------------------------------------------------------------------
# apply_mask on the OUTPUT does not discount the producing kernel
# ---------------------------------------------------------------------------

APPLY_MASK_OUTPUT_DSL = '''
from solar.einsum_lib import Tensor, Dim, kernel, clear_dims, apply_mask


@kernel
def banded_scores(Q: Tensor("S,D", dtype="fp32"),
                  K: Tensor("T,D", dtype="fp32")) -> Tensor("S,T"):
    Y = Tensor(S=Q.shape[0], T=K.shape[0], dtype="fp32")
    for s in Dim("S", parallel=True):
        for t in Dim("T"):
            acc = 0.0
            for d in Dim("D"):
                acc += Q[s, d] * K[t, d]
            Y[s, t] = acc
    return Y


def run_model(inputs):
    clear_dims()
    q = Tensor.from_numpy(inputs["q"], "S,D", dtype="fp32")
    k = Tensor.from_numpy(inputs["k"], "T,D", dtype="fp32")
    band = Tensor.from_numpy(inputs["band"], "S,T", dtype="fp32", sparsity=%r)
    return {"output": apply_mask(banded_scores(q, k), band, masked_value=0.0).data}
''' % ANALYTIC_DENSITY


def test_apply_mask_on_the_output_leaves_the_producer_billed_dense(tmp_path):
    md = _traced_metadata(APPLY_MASK_OUTPUT_DSL, _band_inputs(), profile=False)
    layer = _analyze(APPLY_MASK_OUTPUT_DSL, md, tmp_path)["layers"]["banded_scores#0"]
    # Same scores as MASK_TENSOR_DSL, but the mask never enters the kernel:
    # the producer is billed at full S*T*D; only the mask result carries density.
    assert layer["macs"] == S * T * D
    assert layer.get("compute_density") in (None, 1, 1.0)


# ---------------------------------------------------------------------------
# Prompt text: the band-density closed form must be the boundary-correct one
# ---------------------------------------------------------------------------

def _band_density(s: int, w: int) -> float:
    i, j = np.indices((s, s))
    return float((np.abs(i - j) < w).mean())


@pytest.mark.parametrize("s,w", [(4, 2), (8, 3), (16, 16), (5, 1)])
def test_symmetric_band_closed_form_matches_brute_force(s, w):
    closed = ((2 * w - 1) * s - w * (w - 1)) / s ** 2
    assert closed == pytest.approx(_band_density(s, w))


def test_prompts_and_skill_state_the_truncated_band_formula():
    for text in list(_preambles().values()) + [(SKILLS / "reference.md").read_text()]:
        assert "W*(W-1)" in text or "W(W-1)" in text
        assert "does not discount the producing kernel" in text


# ---------------------------------------------------------------------------
# Round-2 reviewer cases: use-site validity, store targets, all binding kinds
# ---------------------------------------------------------------------------

def test_a_resolvable_reassignment_after_the_loop_does_not_stand_in(tmp_path):
    body = '''    n = X.shape[0]
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
    n = X.shape[0] // 2
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


def test_an_indexed_store_reading_the_size_name_keeps_its_spec(tmp_path):
    body = '''    n = X.shape[0]
    Y = Tensor(I=2 * n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i + n] = X[i % n] * X[i % n]
'''
    assert _i_size_spec(tmp_path, body) == ("shape", "X", 0)


def test_keyword_only_parameter_shadows_a_module_constant(tmp_path):
    body = '''    Y = Tensor(I=X.shape[0] // DIVISOR, dtype="fp32")
    for i in Dim("I", size=X.shape[0] // DIVISOR):
        Y[i] = X[i * DIVISOR]
'''
    params = 'X: Tensor("N", dtype="fp32"), *, DIVISOR: int = 4'
    _assert_scalar_parameter_sizes_the_loop(_i_size_spec(tmp_path, body, params))


def test_a_with_as_target_rebinds_the_size_name(tmp_path):
    body = '''    n = X.shape[0] // 2
    with nullcontext(X.shape[0]) as n:
        pass
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


def test_tuple_destructuring_counts_as_a_binding(tmp_path):
    body = '''    n = X.shape[0] // 2
    n, m = X.shape[0], 1
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


# ---------------------------------------------------------------------------
# Round-3 reviewer cases: annotations, lexical scopes, shape-leaf provenance,
# positional-only parameter order
# ---------------------------------------------------------------------------

def test_a_bare_annotation_is_not_a_second_value_write(tmp_path):
    body = '''    n: int
    n = X.shape[0]
    Y = Tensor(I=2 * n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i + n] = X[i % n] * X[i % n]
'''
    assert _i_size_spec(tmp_path, body) == ("shape", "X", 0)


def test_locals_of_a_nested_helper_do_not_rebind_the_kernel_name(tmp_path):
    body = '''    n = X.shape[0]
    def allocation_size():
        n = 1 + 1
        return n * X.shape[0]
    Y = Tensor(I=allocation_size(), dtype="fp32")
    for i in Dim("I", size=n):
        Y[i + n] = X[i % n] * X[i % n]
'''
    assert _i_size_spec(tmp_path, body) == ("shape", "X", 0)


@pytest.mark.parametrize("size_expr", ["n", "X.shape[0] // 2"])
def test_a_rebound_tensor_parameter_cannot_supply_a_shape_leaf(tmp_path, size_expr):
    body = f'''    X = Z
    n = X.shape[0] // 2
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size={size_expr}):
        Y[i] = X[2 * i] * X[2 * i]
'''
    params = 'X: Tensor("N", dtype="fp32"), Z: Tensor("M", dtype="fp32")'
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body, params), size_expr)
    _assert_refused_at_the_call(spec, param_names=("X", "Z"), tensor_param_count=2, input_shapes=(8, 16))


def test_keyword_only_tensor_parameter_cannot_supply_a_shape_leaf(tmp_path):
    body = '''    Y = Tensor(I=Z.shape[0] // 2, dtype="fp32")
    for i in Dim("I", size=Z.shape[0] // 2):
        Y[i] = Z[2 * i] * Z[2 * i]
'''
    params = 'X: Tensor("N", dtype="fp32"), *, Z: Tensor("M", dtype="fp32")'
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body, params), "Z.shape[0] // 2")
    _assert_refused_at_the_call(spec, param_names=("X", "Z"), tensor_param_count=None, input_shapes=(8, 16))


def test_positional_only_parameters_keep_call_order_in_param_names(tmp_path):
    src = _BINDING_HEADER + '''

@kernel
def k(X: Tensor("N", dtype="fp32"), /, Z: Tensor("M", dtype="fp32")) -> Tensor("I"):
    Y = Tensor(I=Z.shape[0] // 2, dtype="fp32")
    for i in Dim("I", size=Z.shape[0] // 2):
        Y[i] = Z[2 * i] * Z[2 * i]
    return Y
'''
    path = tmp_path / "dsl.py"
    path.write_text(src)
    info = extract_kernel_info(path)["k"]
    assert info["param_names"] == ["X", "Z"]
    specs = {name: spec for p, _w in info["loop_paths"] for name, spec in p}
    assert specs["I"] == ("binop", "//", ("shape", "Z", 0), 2)


# ---------------------------------------------------------------------------
# Round-4 reviewer cases: input/parameter correspondence, enclosing-scope
# writes from nested code, stale module constants
# ---------------------------------------------------------------------------

from solar.einsum_dsl.model import DSLKernel, DSLTensor, DSLInput  # noqa: E402
from solar.einsum_dsl.einsum_converter import _kernel_loop_iterations  # noqa: E402


def _kernel_with(param_names, tensor_param_count, input_shapes, spec):
    inputs = [DSLInput(tensor=DSLTensor(name=f"t{i}", dims={f"D{i}": s}, dtype="fp32", role="input"),
                       from_kernel="input") for i, s in enumerate(input_shapes)]
    out = DSLTensor(name="Y", dims={"I": 999}, dtype="fp32", role="")
    return DSLKernel(id="k#0", function_name="k", inputs=inputs, output=out,
                     loop_expr="Y[i] = X[i] * X[i]", param_names=param_names,
                     loop_paths=[([("I", spec)], 1)], tensor_param_count=tensor_param_count,
                     inputs_in_signature_order=True)


HALF_OF_X = ("binop", "//", ("shape", "X", 0), 2)


def test_binop_spec_resolves_when_inputs_correspond_to_the_tensor_prefix():
    k = _kernel_with(["X", "Z"], 2, [8, 16], HALF_OF_X)
    assert _kernel_loop_iterations(k, {})[0] == 4


def test_binop_spec_stays_unresolved_when_a_scalar_precedes_the_tensors():
    # def f(offset: int, X, Z): recorded inputs [x, z] would zip onto ["offset", "X"]
    k = _kernel_with(["offset", "X", "Z"], None, [8, 16], HALF_OF_X)
    err = _refused(k)                 # typed refusal, never the registry / positional fallback (HEADER-FALLBACK-19)
    assert "cannot be evaluated at this call" in err.detail and "registry binding" in err.detail


def test_binop_spec_stays_unresolved_when_recorded_inputs_do_not_match_the_prefix():
    k = _kernel_with(["X", "Z"], 2, [8], HALF_OF_X)          # a tensor passed keyword-only / missing
    err = _refused(k)
    assert "cannot be evaluated at this call" in err.detail and "registry binding" in err.detail


def test_plain_shape_spec_keeps_base_behaviour_without_correspondence():
    k = _kernel_with(["offset", "X"], None, [8], ("shape", "offset", 0))
    assert _kernel_loop_iterations(k, {})[0] == 8          # unchanged pre-existing mapping


def test_tracer_records_tensor_inputs_in_signature_order_for_keyword_calls():
    from solar.einsum_lib import core

    @core.kernel
    def f(X: core.Tensor("N", dtype="fp32"), Z: core.Tensor("M", dtype="fp32")) -> core.Tensor("N"):
        return X

    core.clear_dims()
    core._tracer.start()
    try:
        x = core.Tensor.from_numpy(np.ones(8), "N", dtype="fp32")
        z = core.Tensor.from_numpy(np.ones(16), "M", dtype="fp32")
        f(Z=z, X=x)                                   # keyword call, reversed order
        calls = [e for e in core._tracer.entries if e.get("type") == "call" and e.get("name") == "f"]
    finally:
        core._tracer.stop()
    assert calls and calls[-1]["input_ids"] == [x._trace_id, z._trace_id]


def test_tensor_param_count_is_the_clean_tensor_prefix(tmp_path):
    from solar.einsum_dsl.ast_parser import _tensor_param_count
    import ast as _ast
    def count(sig):
        return _tensor_param_count(_ast.parse(f"def k({sig}): pass").body[0])
    assert count('X: Tensor("N"), Z: Tensor("M")') == 2
    assert count('X: Tensor("N"), n: int') == 1              # trailing scalar is fine
    assert count('offset: int, X: Tensor("N")') is None      # scalar before a tensor
    assert count('X, Z: Tensor("M")') is None                 # unannotated positional
    assert count('X: Tensor("N"), *, Z: Tensor("M")') is None  # keyword-only tensor
    assert count('X: Tensor("N"), *, n: int = 2') == 1        # keyword-only scalar is fine
    assert count('X: Tensor("N"), *rest') is None


def test_nonlocal_write_from_a_nested_helper_invalidates_the_size(tmp_path):
    body = '''    n = X.shape[0] // 2
    def grow():
        nonlocal n
        n *= 2
    grow()
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


@pytest.mark.parametrize("definition", [
    "    def helper(unused=(n := X.shape[0])):\n        return unused\n",
    "    helper = lambda unused=(n := X.shape[0]): unused\n",
])
def test_a_definition_time_walrus_counts_as_a_write(tmp_path, definition):
    body = '''    n = X.shape[0] // 2
''' + definition + '''    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


@pytest.mark.parametrize("module_write", ["DIVISOR *= 2", "DIVISOR = 1 + 3"])
def test_a_module_constant_rewritten_later_is_not_a_constant(tmp_path, module_write):
    src = _BINDING_HEADER + module_write + '''

@kernel
def k(X: Tensor("N", dtype="fp32")) -> Tensor("I"):
    Y = Tensor(I=X.shape[0] // DIVISOR, dtype="fp32")
    for i in Dim("I", size=X.shape[0] // DIVISOR):
        Y[i] = X[i * DIVISOR] * X[i * DIVISOR]
    return Y
'''
    path = tmp_path / "dsl.py"
    path.write_text(src)
    info = extract_kernel_info(path)["k"]
    specs = {name: spec for p, _w in info["loop_paths"] for name, spec in p}
    spec = _assert_opaque(tmp_path, specs["I"], "X.shape[0] // DIVISOR")   # DIVISOR is rewritten: not a constant
    _assert_refused_at_the_call(spec)


# ---------------------------------------------------------------------------
# Round-5 reviewer cases: legacy trace metadata, variadic flattening,
# lexical target of nested nonlocal
# ---------------------------------------------------------------------------

def test_binop_spec_needs_the_signature_order_marker_from_the_trace():
    k = _kernel_with(["X", "Z"], 2, [8, 16], HALF_OF_X)
    k.inputs_in_signature_order = False                  # legacy artifact: call order unknown
    assert "signature input-order marker" in _refused(k).detail
    k.inputs_in_signature_order = True                   # fresh trace with marker
    assert _kernel_loop_iterations(k, {})[0] == 4


def test_trace_marker_round_trips_into_graph_metadata():
    from solar.einsum_lib import core, metadata

    @core.kernel
    def f(X: core.Tensor("N", dtype="fp32")) -> core.Tensor("N"):
        return X

    core.clear_dims()
    core._tracer.start()
    try:
        x = core.Tensor.from_numpy(np.ones(8), "N", dtype="fp32")
        f(x)
        md = metadata.build_metadata(list(core._tracer.entries))
    finally:
        core._tracer.stop()
    assert md["graph"]["f#0"]["input_order"] == "signature"


def test_tracer_does_not_flatten_an_ordinary_tuple_parameter():
    from solar.einsum_lib import core

    @core.kernel
    def f(X: core.Tensor("P", dtype="fp32"), extras: tuple, Z: core.Tensor("Q", dtype="fp32"), *, flag: int = 0) -> core.Tensor("Q"):
        return Z

    core.clear_dims()
    core._tracer.start()
    try:
        x = core.Tensor.from_numpy(np.ones(8), "P", dtype="fp32")
        z = core.Tensor.from_numpy(np.ones(8), "Q", dtype="fp32")
        a = core.Tensor.from_numpy(np.ones(16), "A", dtype="fp32")
        b = core.Tensor.from_numpy(np.ones(32), "B", dtype="fp32")
        f(x, (a, b), z, flag=0)
        calls = [e for e in core._tracer.entries if e.get("type") == "call" and e.get("name") == "f"]
    finally:
        core._tracer.stop()
    assert calls[-1]["input_ids"] == [x._trace_id, z._trace_id]


def test_deep_nonlocal_targeting_an_intermediate_helper_keeps_the_kernel_size(tmp_path):
    body = '''    n = X.shape[0]
    def allocation_size():
        n = 1 + 0
        def double():
            nonlocal n
            n *= 2
        double()
        return n * X.shape[0]
    Y = Tensor(I=allocation_size(), dtype="fp32")
    for i in Dim("I", size=n):
        Y[i + n] = X[i % n] * X[i % n]
'''
    assert _i_size_spec(tmp_path, body) == ("shape", "X", 0)


def test_deep_nonlocal_reaching_the_kernel_still_invalidates(tmp_path):
    body = '''    n = X.shape[0] // 2
    def outer():
        def inner():
            nonlocal n
            n *= 2
        inner()
    outer()
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)


def test_global_declaration_inside_a_kernel_helper_does_not_touch_kernel_locals_but_kills_module_const(tmp_path):
    src = _BINDING_HEADER + '''

def bump():
    global DIVISOR
    DIVISOR = 4

@kernel
def k(X: Tensor("N", dtype="fp32")) -> Tensor("I"):
    n = X.shape[0]
    Y = Tensor(I=X.shape[0] // DIVISOR, dtype="fp32")
    for i in Dim("I", size=X.shape[0] // DIVISOR):
        Y[i] = X[i * DIVISOR] * X[i % n]
    return Y
'''
    path = tmp_path / "dsl.py"
    path.write_text(src)
    info = extract_kernel_info(path)["k"]
    specs = {name: spec for p, _w in info["loop_paths"] for name, spec in p}
    spec = _assert_opaque(tmp_path, specs["I"], "X.shape[0] // DIVISOR")   # written via global elsewhere: not a constant
    _assert_refused_at_the_call(spec)


# ---------------------------------------------------------------------------
# Round-6 reviewer cases: intermediate bindings via match capture and
# definition-time walrus must shield the kernel from a deeper nonlocal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intermediate_binding", [
    "        match 1:\n            case n:\n                pass\n",
    "        def seed(value=(n := 1)):\n            return value\n",
    "        seed = lambda value=(n := 1): value\n",
])
def test_intermediate_binding_shields_the_kernel_from_a_deeper_nonlocal(tmp_path, intermediate_binding):
    body = '''    n = X.shape[0]
    def allocation_size():
''' + intermediate_binding + '''        def double():
            nonlocal n
            n *= 2
        double()
        return n * X.shape[0]
    Y = Tensor(I=allocation_size(), dtype="fp32")
    for i in Dim("I", size=n):
        Y[i + n] = X[i % n] * X[i % n]
'''
    assert _i_size_spec(tmp_path, body) == ("shape", "X", 0)


def test_match_capture_in_the_kernel_itself_counts_as_a_write(tmp_path):
    body = '''    n = X.shape[0] // 2
    match 1:
        case n:
            pass
    Y = Tensor(I=n, dtype="fp32")
    for i in Dim("I", size=n):
        Y[i] = X[i] * X[i]
'''
    spec = _assert_opaque(tmp_path, _i_size_spec(tmp_path, body), "n")
    _assert_refused_at_the_call(spec)
