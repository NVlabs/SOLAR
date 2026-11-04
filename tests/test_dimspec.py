# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``solar.translate._dimspec.resolve_dimension``: the bounded resolver for declared shape dimensions.

The function moved verbatim from the retired benchmark package; these cases pin the grammar
``solar.translate.sol`` relies on for declared output shapes (ints, axis names, bounded
arithmetic, ``{"expr": ...}``, operator dicts, infix lists) and the loud ``ValueError`` on
anything else.
"""
import pytest

from solar.translate._dimspec import resolve_dimension

AXES = {"batch_size": 2, "seq_len": 64, "num_heads": 12, "head_dim": 64, "hidden": 768, "n_groups": 8}


@pytest.mark.parametrize("spec,expected", [
    (7, 7), (7.9, 7), ("seq_len", 64), ("num_heads * head_dim", 768), ("hidden // n_groups", 96),
    ("hidden / n_groups", 96), ("(num_heads + 1) * head_dim", 832), ("-seq_len", -64),
    ({"expr": "num_heads*head_dim"}, 768), ({"mul": ["num_heads", "head_dim"]}, 768),
    ({"div": ["hidden", 2]}, 384), ({"add": ["seq_len", "seq_len"]}, 128), ({"sub": ["seq_len", 1]}, 63),
    ({"mul": ["batch_size", {"add": ["seq_len", 1]}]}, 130),
    (["num_heads", "*", "head_dim"], 768), (["hidden", "//", "n_groups"], 96),
    ([{"mul": ["batch_size", 2]}, "*", "seq_len"], 256),
])
def test_resolves_the_definition_grammar(spec, expected):
    assert resolve_dimension(spec, dict(AXES)) == expected


@pytest.mark.parametrize("spec", [
    "unknown_axis", "num_heads * unknown", "max(seq_len, 1)", "seq_len.real", "1 if seq_len else 2",
    "2 ** 3", {"expr": "import os"}, {"pow": ["a", "b"]}, {"mul": ["seq_len"]}, ["seq_len", "**", 2], None,
])
def test_refuses_unknown_axes_and_disallowed_constructs(spec):
    with pytest.raises(ValueError):
        resolve_dimension(spec, dict(AXES))


def test_translate_facade_uses_the_core_resolver():
    import inspect
    import solar.translate as facade
    src = inspect.getsource(facade)
    assert "solar.translate._dimspec import resolve_dimension" in src
    assert "solar.benchmark" not in src
