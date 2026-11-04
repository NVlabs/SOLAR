# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The einsum_lib tensor builtins a DSL program may call without a kernel body.

A DSL kernel node is either a PARSED kernel (its ``@kernel`` source was parsed:
loop paths, loop_expr, parameter names) or a bodiless call to one of these
builtins (``gather``, ``slice_at``, ...).  The DSL->einsum converter stamps every
kernel node with explicit provenance -- ``dsl_route: true`` and ``builtin: <name>``
(the resolved builtin, or null) -- and the graph analyzer admits the generic
shape-based cost fallback for a bodiless node ONLY when ``builtin`` is in this
table (kernel-origin-10).  A bodiless node whose function is not a known
builtin is exported as ``unrecognised_op`` and reported incomplete; a torch-route
node that merely happens to be NAMED ``kernel`` carries no DSL provenance and
takes the unknown-op fail-closed path.

Keep in sync with ``solar.einsum_lib.__all__`` and
``solar.einsum_lib.numba_accel.execution_namespace()`` (tests pin both).
"""
DSL_TENSOR_BUILTINS = frozenset({"apply_mask", "top_k_mask", "slice_at", "stack", "concat", "gather"})

__all__ = ["DSL_TENSOR_BUILTINS"]
