# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace-local return ownership. Never serialize Python object IDs.

Weak references plus ``is`` checks prevent recycled IDs from acquiring an old
owner without retaining every full-size intermediate array. No shape/value or
shared-memory heuristic can establish ownership of an unrecorded view.
"""
import weakref

import numpy as np


class OutputBindings:
    def __init__(self):
        self.arrays = {}
        self.tensors = {}
        self.descriptors = {}
        self.inputs = {}

    def register_inputs(self, inputs):
        self.inputs = {}
        for name, array in inputs.items():
            if not isinstance(array, np.ndarray):
                continue
            key = id(array)
            row = self.inputs.setdefault(key, (weakref.ref(array), []))
            row[1].append(name)

    def bind(self, tensor_id, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("output provenance: traced data must be an ndarray")
        previous = self.tensors.get(tensor_id)
        if previous is not None and previous() is not array:
            old = previous()
            if old is not None:
                row = self.arrays.get(id(old))
                if row and row[0]() is old:
                    row[1].discard(tensor_id)
        key = id(array)
        row = self.arrays.get(key)
        if row is None or row[0]() is not array:
            row = (weakref.ref(array), set())
            self.arrays[key] = row
        row[1].add(tensor_id)
        self.tensors[tensor_id] = weakref.ref(array)
        self.descriptors[tensor_id] = (tuple(array.shape), str(array.dtype), tuple(array.strides))

    def returns(self, outputs):
        leaves = []
        ancestors = set()

        def visit(value, path):
            if isinstance(value, (dict, tuple, list)):
                if id(value) in ancestors:
                    raise ValueError("output provenance: cyclic return structure")
                ancestors.add(id(value))
                if isinstance(value, dict):
                    if any(not isinstance(k, str) for k in value):
                        raise ValueError("output provenance: return dict keys must be strings")
                    items = value.items()
                else:
                    items = enumerate(value)
                for key, child in items:
                    visit(child, path + [key])
                ancestors.remove(id(value))
                return
            if not isinstance(value, np.ndarray):
                raise ValueError("output provenance: unsupported non-array return leaf")
            row = self.arrays.get(id(value))
            owners = row[1] if row and row[0]() is value else set()
            inp = self.inputs.get(id(value))
            names = inp[1] if inp and inp[0]() is value else []
            if len(owners) > 1 or len(names) > 1:
                raise ValueError("output provenance: ambiguous array ownership")
            # A distinct ndarray may still alias a live tensor/input. Without
            # an explicit view/version IR its writes cannot be deduplicated
            # safely. A conservative overlap check is a refusal, never a
            # mechanism for assigning an owner or merging equal-shaped data.
            for ref, identities in [*self.arrays.values(), *self.inputs.values()]:
                other = ref()
                if (identities and other is not None and other is not value
                        and np.may_share_memory(value, other)):
                    raise ValueError("output provenance: aliased array requires explicit view/version mapping")
            if len(owners) == 1:
                tid = next(iter(owners))
                if self.descriptors[tid] != (tuple(value.shape), str(value.dtype), tuple(value.strides)):
                    raise ValueError("output provenance: array descriptor changed after binding")
                leaves.append({"path": path, "tensor_id": tid})
            elif len(names) == 1:
                leaves.append({"path": path, "input": {"name": names[0],
                    "shape": list(value.shape), "dtype": str(value.dtype)}})
            else:
                raise ValueError("output provenance: unbound array/view or retired data version")

        visit(outputs, [])
        if not leaves:
            raise ValueError("output provenance: no returned array leaves")
        return leaves
