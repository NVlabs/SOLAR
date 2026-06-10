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

"""Removed: BFS-based einsum rank renamer.

This module previously implemented :class:`EinsumRankRenamer` — a
per-node BFS pass that propagated rank labels from predecessor outputs
to consumer inputs. It was the first of five spaghetti passes that
together emitted the AccelForge graph, and its lack of a global rank
registry caused the "Rk reuse for different sizes" failure class on
DenseNet, Mamba2, RNN/LSTM/GRU, ShuffleNet, NetVlad, and ~15 other
L3 problems.

It was replaced in 2026-05 by :mod:`solar.einsum.af_graph_builder`,
which performs the rename via a single principled union-find pass
over ``(layer, role, position)`` axis keys. See that module's
docstring for the algorithm.

This file is kept as a placeholder so external imports of
``EinsumRankRenamer`` fail loudly with a clear migration message
rather than silently re-importing dead code.
"""


class EinsumRankRenamer:
    """Removed — see :mod:`solar.einsum.af_graph_builder`."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "EinsumRankRenamer was removed. The AccelForge graph is now "
            "emitted in a single principled pass by "
            "`solar.einsum.af_graph_builder.build_af_graph_from_dict`."
        )


# Backward-compatibility aliases that raise on use.
EinsumGraphRenamer = EinsumRankRenamer


def rename_einsum_ranks(*args, **kwargs):
    raise RuntimeError(
        "rename_einsum_ranks was removed. The AccelForge graph is now "
        "emitted by `solar.einsum.af_graph_builder.build_af_graph_from_dict`."
    )


def rename_einsum_ranks_dict(*args, **kwargs):
    raise RuntimeError(
        "rename_einsum_ranks_dict was removed. The AccelForge graph is now "
        "emitted by `solar.einsum.af_graph_builder.build_af_graph_from_dict`."
    )


__all__ = [
    "EinsumRankRenamer",
    "EinsumGraphRenamer",
    "rename_einsum_ranks",
    "rename_einsum_ranks_dict",
]
