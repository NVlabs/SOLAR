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

"""Unit tests for solar.analysis.access_regions.

Raw-attribute strings in these tests reproduce the exact format torchview
records (tuple indices stringified as lists, kwargs with bare keys).
"""

import pytest

from solar.analysis.access_regions import (
    box_size,
    parse_call_attributes,
    partition_output_box,
    slice_op_boxes,
    union_size,
)


TENSOR = "Tensor(shape=(8, 12), dtype=torch.float32)"


# ---------------------------------------------------------------------------
# parse_call_attributes
# ---------------------------------------------------------------------------
class TestParseCallAttributes:
    def test_int_index(self):
        args, kwargs = parse_call_attributes(f"[[{TENSOR}, 0], {{}}]")
        assert args[1] == 0 and kwargs == {}

    def test_slice_index(self):
        args, _ = parse_call_attributes(f"[[{TENSOR}, slice(2, 6, None)], {{}}]")
        assert args[1] == slice(2, 6, None)

    def test_tuple_index_stringified_as_list(self):
        args, _ = parse_call_attributes(f"[[{TENSOR}, [1, slice(2, 6, None)]], {{}}]")
        assert args[1] == [1, slice(2, 6, None)]

    def test_negative_int(self):
        args, _ = parse_call_attributes(f"[[{TENSOR}, -1], {{}}]")
        assert args[1] == -1

    def test_bare_key_kwargs(self):
        _, kwargs = parse_call_attributes(f"[[{TENSOR}, 3], {{dim: 1}}]")
        assert kwargs == {"dim": 1}

    def test_nested_tensor_repr_with_device(self):
        raw = "[[Tensor(shape=(2, (3)), dtype=torch.float32, device='cuda:0'), 1], {}]"
        args, _ = parse_call_attributes(raw)
        assert args[1] == 1

    def test_ellipsis_name(self):
        args, _ = parse_call_attributes(f"[[{TENSOR}, [Ellipsis, slice(0, 4, None)]], {{}}]")
        assert args[1][0] is Ellipsis

    def test_unbalanced_returns_none(self):
        assert parse_call_attributes("[[Tensor(shape=(2, 8], 1], {}]") is None

    def test_garbage_returns_none(self):
        assert parse_call_attributes("not a list") is None
        assert parse_call_attributes(None) is None
        assert parse_call_attributes("") is None

    def test_call_injection_is_unsupported(self):
        # Arbitrary calls must not evaluate; they must degrade, not raise.
        args, _ = parse_call_attributes(f"[[{TENSOR}, __import__('os')], {{}}]")
        assert slice_op_boxes("__getitem__", f"[[{TENSOR}, __import__('os')], {{}}]", [8, 12], 12) is None
        assert args is not None  # structure parsed; payload is Unsupported


# ---------------------------------------------------------------------------
# slice_op_boxes: __getitem__
# ---------------------------------------------------------------------------
class TestGetitemBoxes:
    def test_int_index(self):
        boxes = slice_op_boxes("__getitem__", f"[[{TENSOR}, 1], {{}}]", [8, 12], 12)
        assert boxes == [((1, 2), (0, 12))]

    def test_negative_int_index(self):
        boxes = slice_op_boxes("__getitem__", f"[[{TENSOR}, -1], {{}}]", [8, 12], 12)
        assert boxes == [((7, 8), (0, 12))]

    def test_basic_slice(self):
        boxes = slice_op_boxes(
            "__getitem__", f"[[{TENSOR}, slice(2, 6, None)], {{}}]", [8, 12], 4 * 12
        )
        assert boxes == [((2, 6), (0, 12))]

    def test_tuple_index(self):
        boxes = slice_op_boxes(
            "__getitem__", f"[[{TENSOR}, [1, slice(2, 6, None)]], {{}}]", [8, 12], 4
        )
        assert boxes == [((1, 2), (2, 6))]

    def test_ellipsis(self):
        boxes = slice_op_boxes(
            "__getitem__", f"[[{TENSOR}, [Ellipsis, slice(0, 4, None)]], {{}}]", [8, 12], 8 * 4
        )
        assert boxes == [((0, 8), (0, 4))]

    def test_newaxis_consumes_no_dim(self):
        boxes = slice_op_boxes(
            "__getitem__", f"[[{TENSOR}, [None, slice(0, 4, None)]], {{}}]", [8, 12], 4 * 12
        )
        assert boxes == [((0, 4), (0, 12))]

    def test_negative_step_one(self):
        boxes = slice_op_boxes(
            "__getitem__", f"[[{TENSOR}, slice(None, None, -1)], {{}}]", [8, 12], 8 * 12
        )
        assert boxes == [((0, 8), (0, 12))]

    def test_strided_slice_rejected(self):
        assert (
            slice_op_boxes(
                "__getitem__", f"[[{TENSOR}, slice(None, None, 2)], {{}}]", [8, 12], 4 * 12
            )
            is None
        )

    def test_tensor_index_rejected(self):
        raw = f"[[{TENSOR}, Tensor(shape=(3,), dtype=torch.int64)], {{}}]"
        assert slice_op_boxes("__getitem__", raw, [8, 12], 3 * 12) is None

    def test_size_mismatch_rejected(self):
        # Recorded output size disagrees with the parsed region.
        assert (
            slice_op_boxes("__getitem__", f"[[{TENSOR}, slice(2, 6, None)], {{}}]", [8, 12], 99)
            is None
        )

    def test_out_of_bounds_rejected(self):
        assert slice_op_boxes("__getitem__", f"[[{TENSOR}, 8], {{}}]", [8, 12], 12) is None

    def test_too_many_indices_rejected(self):
        assert (
            slice_op_boxes("__getitem__", f"[[{TENSOR}, [0, 1, 2]], {{}}]", [8, 12], 1)
            is None
        )

    def test_fancy_int_list(self):
        # x[[0, 2, 2]] on (8, 12): output counts duplicates (3*12), the
        # union must not (2 distinct rows).
        boxes = slice_op_boxes("__getitem__", f"[[{TENSOR}, [0, 2, 2]], {{}}]", [8, 12], 3 * 12)
        assert boxes is not None
        assert union_size(boxes) == 2 * 12

    def test_all_int_list_prefers_valid_interpretation(self):
        # [0, 2] on (8, 12): dim-tuple reading -> 1 element; fancy gather
        # -> 24 elements.  The recorded output size selects the reading.
        as_dim_tuple = slice_op_boxes("__getitem__", f"[[{TENSOR}, [0, 2]], {{}}]", [8, 12], 1)
        assert as_dim_tuple == [((0, 1), (2, 3))]
        as_fancy = slice_op_boxes("__getitem__", f"[[{TENSOR}, [0, 2]], {{}}]", [8, 12], 24)
        assert as_fancy is not None
        assert union_size(as_fancy) == 24

    def test_missing_raw_attributes_rejected(self):
        assert slice_op_boxes("__getitem__", None, [8, 12], 12) is None

    def test_dynamic_shape_rejected(self):
        assert slice_op_boxes("__getitem__", f"[[{TENSOR}, 0], {{}}]", [8, 0], 0) is None


# ---------------------------------------------------------------------------
# slice_op_boxes: narrow / select / slice
# ---------------------------------------------------------------------------
class TestNarrowSelectSlice:
    def test_narrow(self):
        boxes = slice_op_boxes("narrow", f"[[{TENSOR}, 1, 2, 4], {{}}]", [8, 12], 8 * 4)
        assert boxes == [((0, 8), (2, 6))]

    def test_select(self):
        boxes = slice_op_boxes("select", f"[[{TENSOR}, 0, 3], {{}}]", [8, 12], 12)
        assert boxes == [((3, 4), (0, 12))]

    def test_select_kwargs(self):
        boxes = slice_op_boxes("select", f"[[{TENSOR}], {{dim: 0, index: -1}}]", [8, 12], 12)
        assert boxes == [((7, 8), (0, 12))]

    def test_narrow_out_of_bounds_rejected(self):
        assert slice_op_boxes("narrow", f"[[{TENSOR}, 1, 10, 4], {{}}]", [8, 12], 8 * 4) is None


# ---------------------------------------------------------------------------
# partition_output_box
# ---------------------------------------------------------------------------
class TestPartitionOutputBox:
    CHUNK_RAW = f"[[{TENSOR}, 3], {{dim: 1}}]"
    OUT_SHAPES = [[8, 4], [8, 4], [8, 4]]

    def test_chunk_outputs(self):
        for k, expected in enumerate([((0, 8), (0, 4)), ((0, 8), (4, 8)), ((0, 8), (8, 12))]):
            box = partition_output_box("chunk", self.CHUNK_RAW, [8, 12], self.OUT_SHAPES, k)
            assert box == expected

    def test_dim_inferred_without_attributes(self):
        box = partition_output_box("chunk", None, [8, 12], self.OUT_SHAPES, 1)
        assert box == ((0, 8), (4, 8))

    def test_uneven_split(self):
        box = partition_output_box("split", None, [8, 12], [[8, 5], [8, 5], [8, 2]], 2)
        assert box == ((0, 8), (10, 12))

    def test_incomplete_partition_rejected(self):
        # Output sizes do not sum to the base along any axis.
        assert partition_output_box("chunk", None, [8, 12], [[8, 4], [8, 4]], 0) is None

    def test_bad_output_index_rejected(self):
        assert partition_output_box("chunk", self.CHUNK_RAW, [8, 12], self.OUT_SHAPES, 3) is None


# ---------------------------------------------------------------------------
# union_size
# ---------------------------------------------------------------------------
class TestUnionSize:
    def test_empty(self):
        assert union_size([]) == 0

    def test_single(self):
        assert union_size([((2, 6), (0, 12))]) == 48

    def test_repeated(self):
        assert union_size([((0, 10),), ((0, 10),)]) == 10

    def test_overlapping(self):
        assert union_size([((0, 10),), ((5, 15),)]) == 15

    def test_disjoint(self):
        assert union_size([((0, 4),), ((8, 12),)]) == 8

    def test_nested(self):
        assert union_size([((0, 10), (0, 12)), ((2, 4), (3, 5))]) == 120

    def test_2d_cross(self):
        # Rows [0,2) of (8,64) plus columns [0,16) of all rows.
        assert union_size([((0, 2), (0, 64)), ((0, 8), (0, 16))]) == 128 + 128 - 32

    def test_mixed_rank_returns_none(self):
        assert union_size([((0, 4),), ((0, 4), (0, 4))]) is None

    def test_budget_overflow_returns_none(self):
        boxes = [((i, i + 2), (0, 4)) for i in range(0, 200, 1)]
        assert union_size(boxes, cell_budget=10) is None

    def test_stacked_kv_pattern(self):
        # Per-layer slices kv[i] of a (4, 8, 64) stacked tensor.
        boxes = [((i, i + 1), (0, 8), (0, 64)) for i in range(4)]
        assert union_size(boxes) == 4 * 8 * 64
        assert all(box_size(b) == 8 * 64 for b in boxes)
