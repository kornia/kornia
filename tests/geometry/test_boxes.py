# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import partial

import pytest
import torch

from kornia.geometry import boxes as boxes_module
from kornia.geometry.bbox import bbox_to_mask, bbox_to_mask3d, infer_bbox_shape, infer_bbox_shape3d
from kornia.geometry.boxes import Boxes, Boxes3D, VideoBoxes

from testing.base import BaseTester


def _unbatched_geometry_data(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 2.0], [4.0, 2.0], [4.0, 5.0], [1.0, 5.0]],
            [[8.0, 0.0], [10.0, 0.0], [10.0, 1.0], [8.0, 1.0]],
        ],
        device=device,
        dtype=dtype,
    )


class TestBoxes2D(BaseTester):
    def test_convention_from_tensor_xyxy_stores_inclusive_vertices_in_tl_tr_br_bl_order(self, device, dtype):
        # Convention pin: Boxes stores four inclusive (x, y) vertices in clockwise
        # top-left, top-right, bottom-right, bottom-left order. The asymmetric
        # exclusive xyxy input makes both the coordinate order and the +1 conversion
        # observable. The expected data carries the -1 tracked in kornia#3934.
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 2.0, 5.0, 4.0]]], device=device, dtype=dtype), mode="xyxy")
        expected = torch.tensor([[[[1.0, 2.0], [4.0, 2.0], [4.0, 3.0], [1.0, 3.0]]]], device=device, dtype=dtype)
        self.assert_close(boxes.data, expected, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("mode", ["xyxy", "xyxy_plus", "xywh", "vertices", "vertices_plus"])
    def test_convention_axis_aligned_box_round_trips_in_each_mode(self, mode, device, dtype):
        # Convention pin: an axis-aligned rectangle whose extent is at least one unit
        # per axis round-trips in each mode. The source values describe the same
        # asymmetric box with exclusive extent (4, 2). Sub-unit extents are the
        # documented boundary of this guarantee and are pinned separately below.
        source_by_mode = {
            "xyxy": [1.0, 2.0, 5.0, 4.0],
            "xyxy_plus": [1.0, 2.0, 4.0, 3.0],
            "xywh": [1.0, 2.0, 4.0, 2.0],
            "vertices": [[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]],
            "vertices_plus": [[1.0, 2.0], [4.0, 2.0], [4.0, 3.0], [1.0, 3.0]],
        }
        source = torch.tensor([source_by_mode[mode]], device=device, dtype=dtype)
        output = Boxes.from_tensor(source, mode=mode).to_tensor(mode=mode)
        self.assert_close(output, source, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        ("box_dtype", "source_values", "expected_values"),
        [
            (torch.bfloat16, [256.0, 256.0, 258.0, 258.0], [256.0, 256.0, 256.0, 256.0]),
            (torch.float16, [-385.25, 0.0, 400.0, 2.0], [-385.25, 0.0, 399.75, 2.0]),
        ],
    )
    def test_convention_round_trip_requires_exact_intermediate_arithmetic(
        self, device, box_dtype, source_values, expected_values
    ):
        # bfloat16 cannot represent the +/-1 intermediate at 256. The float16
        # case can represent its offsets, but rounds the cross-zero width first.
        # Both discrepancies exist only because of the +/-1 offsets tracked in kornia#3934.
        source = torch.tensor([source_values], device=device, dtype=box_dtype)
        output = Boxes.from_tensor(source, mode="xyxy").to_tensor("xyxy")
        expected = torch.tensor([expected_values], device=device, dtype=box_dtype)
        self.assert_close(output, expected, atol=0.0, rtol=0.0)
        assert not torch.equal(output, source)

    @pytest.mark.parametrize("mode", ["xyxy", "xyxy_plus", "xywh", "vertices", "vertices_plus"])
    def test_wart_sub_unit_extent_round_trip_boundary_4061(self, mode, device, dtype):
        # Wart pin for kornia#4061: the three converting modes place the top-right
        # vertex at ``xmin + width - 1``, which lands left of the top-left vertex when
        # the extent is below one unit. The stored quadrilateral is inverted on both
        # axes and to_tensor recovers a larger box. 'xyxy_plus' cancels the -1,
        # while 'vertices_plus' bypasses offset conversion.
        source_by_mode = {
            "xyxy": [0.1, 0.1, 0.6, 0.9],
            "xyxy_plus": [0.1, 0.1, 0.6, 0.9],
            "xywh": [0.1, 0.1, 0.5, 0.8],
            "vertices": [[0.1, 0.1], [0.6, 0.1], [0.6, 0.9], [0.1, 0.9]],
            "vertices_plus": [[0.1, 0.1], [0.6, 0.1], [0.6, 0.9], [0.1, 0.9]],
        }
        expected_by_mode = {
            "xyxy": [-0.4, -0.1, 1.1, 1.1],
            "xyxy_plus": [0.1, 0.1, 0.6, 0.9],
            "xywh": [-0.4, -0.1, 1.5, 1.2],
            "vertices": [[-0.4, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.4, 1.1]],
            "vertices_plus": [[0.1, 0.1], [0.6, 0.1], [0.6, 0.9], [0.1, 0.9]],
        }
        source = torch.tensor([source_by_mode[mode]], device=device, dtype=dtype)
        expected = torch.tensor([expected_by_mode[mode]], device=device, dtype=dtype)
        # validate_boxes=True does not reject the input: the extents are positive.
        # Half-precision converting modes use dtype-aware tolerance because their
        # expected decimal results are not all exactly representable.
        output = Boxes.from_tensor(source, mode=mode, validate_boxes=True).to_tensor(mode=mode)
        self.assert_close(output, expected)

    def test_wart_get_boxes_shape_uses_inclusive_extent_3934(self, device, dtype):
        # Wart pin for kornia#3934: raw inclusive vertices spanning x=1..4 and
        # y=2..3 report (height, width) = (2, 4) because both axes add one.
        vertices = torch.tensor([[[1.0, 2.0], [4.0, 2.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        heights, widths = Boxes(vertices).get_boxes_shape()
        self.assert_close(heights, torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(widths, torch.tensor([4.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_get_boxes_shape_includes_list_padding(self, device, dtype):
        # get_boxes_shape uses the padded xywh export, so padding entries appear as
        # 1-by-1 boxes even though an ordinary to_tensor export trims them. The 1-by-1
        # value depends on the inclusive +1 tracked in kornia#3934.
        first = torch.tensor([[[1.0, 2.0], [4.0, 2.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        second = torch.cat([first, first])
        boxes = Boxes([first, second])
        exported = boxes.to_tensor("xywh")
        assert isinstance(exported, list)
        assert [item.shape for item in exported] == [(1, 4), (2, 4)]

        heights, widths = boxes.get_boxes_shape()
        expected_heights = torch.tensor([[2.0, 1.0], [2.0, 2.0]], device=device, dtype=dtype)
        expected_widths = torch.tensor([[4.0, 1.0], [4.0, 4.0]], device=device, dtype=dtype)
        self.assert_close(heights, expected_heights, atol=0.0, rtol=0.0)
        self.assert_close(widths, expected_widths, atol=0.0, rtol=0.0)

    def test_wart_vertices_export_is_exclusive_for_inclusive_bbox_consumers_4009(self, device, dtype):
        # Wart pin for kornia#4009: vertices is an exclusive export, while
        # infer_bbox_shape reads vertices as inclusive and therefore adds one per axis.
        boxes = Boxes.from_tensor(torch.tensor([[1.0, 2.0, 5.0, 4.0]], device=device, dtype=dtype), mode="xyxy")
        heights, widths = infer_bbox_shape(boxes.to_tensor("vertices"))
        self.assert_close(heights, torch.tensor([3.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(widths, torch.tensor([5.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("mode", ["vertices", "vertices_plus"])
    def test_wart_vertices_import_is_not_validated_4177(self, mode, device, dtype):
        # Wart pin for kornia#4177: neither vertex mode is validated. The exclusive
        # 'vertices' import also subtracts one from fixed positions, so a non-rectangular
        # quadrilateral is silently reshaped instead of rejected with validate_boxes=True.
        # The -1 deformation is the inclusive offset tracked in kornia#3934.
        quadrilateral = torch.tensor([[[0.0, 0.0], [9.0, 0.0], [3.0, 7.0], [0.0, 1.0]]], device=device, dtype=dtype)
        boxes = Boxes.from_tensor(quadrilateral, mode=mode, validate_boxes=True)
        expected = quadrilateral.clone()
        if mode == "vertices":
            expected = torch.tensor([[[0.0, 0.0], [8.0, 0.0], [2.0, 6.0], [0.0, 0.0]]], device=device, dtype=dtype)
        self.assert_close(boxes.data, expected, atol=0.0, rtol=0.0)

    def test_convention_constructor_mode_is_only_an_export_label(self, device, dtype):
        vertices = torch.tensor([[[1.0, 2.0], [5.0, 2.0], [5.0, 4.0], [1.0, 4.0]]], device=device, dtype=dtype)
        constructed = Boxes(vertices, mode="vertices").to_tensor()
        imported = Boxes.from_tensor(vertices, mode="vertices").to_tensor()
        expected_constructed = torch.tensor(
            [[[1.0, 2.0], [6.0, 2.0], [6.0, 5.0], [1.0, 5.0]]], device=device, dtype=dtype
        )
        self.assert_close(constructed, expected_constructed, atol=0.0, rtol=0.0)
        self.assert_close(imported, vertices, atol=0.0, rtol=0.0)

    def test_convention_to_tensor_exports_axis_aligned_bounds_of_a_rotated_box(self, device, dtype):
        # Convention pin: to_tensor reduces the stored vertices with amin/amax, so it
        # exports axis-aligned bounds. After a shear the export is lossy and
        # to_tensor('vertices_plus') is not the identity on data.
        boxes = Boxes(torch.tensor([[[1.0, 2.0], [4.0, 2.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype))
        shear = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        sheared = boxes.transform_boxes(shear)
        expected_data = torch.tensor([[[3.0, 2.0], [6.0, 2.0], [7.0, 3.0], [4.0, 3.0]]], device=device, dtype=dtype)
        self.assert_close(sheared.data, expected_data, atol=0.0, rtol=0.0)
        expected_export = torch.tensor([[[3.0, 2.0], [7.0, 2.0], [7.0, 3.0], [3.0, 3.0]]], device=device, dtype=dtype)
        self.assert_close(sheared.to_tensor("vertices_plus"), expected_export, atol=0.0, rtol=0.0)

    def test_wart_constructor_and_from_tensor_have_different_integer_policies_4012(self, device):
        # Wart pin for kornia#4012: the constructor rejects integer coordinates,
        # while from_tensor silently casts them to float32.
        vertices = torch.tensor([[[1, 2], [4, 2], [4, 3], [1, 3]]], device=device)
        with pytest.raises(ValueError, match="floating point"):
            Boxes(vertices)
        coordinates = torch.tensor([[1, 2, 5, 4]], device=device)
        assert Boxes.from_tensor(coordinates, mode="xyxy").dtype == torch.float32

        # A list is padded into a tensor of its first element's dtype before the
        # check, so a mixed-dtype list is judged by its first box alone.
        floating = vertices.to(torch.float32)
        assert Boxes([floating, vertices]).dtype == torch.float32
        with pytest.raises(ValueError, match="floating point"):
            Boxes([vertices, floating])

        # from_tensor converts each list element independently before the same
        # first-element dtype merge, so reversing mixed float16/integer inputs
        # changes the output dtype.
        half = vertices.to(torch.float16)
        assert Boxes.from_tensor([half, vertices], mode="vertices_plus").dtype == torch.float16
        assert Boxes.from_tensor([vertices, half], mode="vertices_plus").dtype == torch.float32

    def test_convention_merge_concatenates_batched_boxes_without_mutating_by_default(self, device, dtype):
        first = Boxes.from_tensor(torch.tensor([[[1.0, 2.0, 5.0, 4.0]]], device=device, dtype=dtype))
        second = Boxes.from_tensor(torch.tensor([[[6.0, 3.0, 9.0, 8.0]]], device=device, dtype=dtype))
        first_before = first.data.clone()
        second_before = second.data.clone()
        merged = first.merge(second)
        assert merged is not first
        assert merged.data.shape == (1, 2, 4, 2)
        self.assert_close(merged.data[:, 0], first_before[:, 0], atol=0.0, rtol=0.0)
        self.assert_close(merged.data[:, 1], second_before[:, 0], atol=0.0, rtol=0.0)
        self.assert_close(first.data, first_before, atol=0.0, rtol=0.0)
        self.assert_close(second.data, second_before, atol=0.0, rtol=0.0)

    def test_convention_index_put_replaces_coordinates_without_mutating_by_default(self, device, dtype):
        boxes = Boxes.from_tensor(
            torch.tensor([[[1.0, 2.0, 5.0, 4.0], [6.0, 3.0, 9.0, 8.0]]], device=device, dtype=dtype)
        )
        replacement = Boxes.from_tensor(torch.tensor([[10.0, 20.0, 14.0, 23.0]], device=device, dtype=dtype))
        boxes_before = boxes.data.clone()
        replacement_before = replacement.data.clone()
        updated = boxes.index_put((torch.tensor([0], device=device), torch.tensor([1], device=device)), replacement)
        assert updated is not boxes
        self.assert_close(updated.data[:, 0], boxes_before[:, 0], atol=0.0, rtol=0.0)
        self.assert_close(updated.data[:, 1], replacement_before, atol=0.0, rtol=0.0)
        self.assert_close(boxes.data, boxes_before, atol=0.0, rtol=0.0)
        self.assert_close(replacement.data, replacement_before, atol=0.0, rtol=0.0)

    def test_smoke(self, device, dtype):
        def _create_tensor_box():
            # Sample two points of the rectangle
            points = torch.rand(1, 4, device=device, dtype=dtype)

            # Fill according missing points
            tensor_boxes = torch.zeros(1, 4, 2, device=device, dtype=dtype)
            tensor_boxes[0, 0] = points[0][:2]
            tensor_boxes[0, 1, 0] = points[0][2]
            tensor_boxes[0, 1, 1] = points[0][1]
            tensor_boxes[0, 2] = points[0][2:]
            tensor_boxes[0, 3, 0] = points[0][0]
            tensor_boxes[0, 3, 1] = points[0][3]
            return tensor_boxes

        # Validate
        assert Boxes(_create_tensor_box())  # Validate 1 box

        # 2 boxes without batching (N, 4, 2) where N=2
        two_boxes = torch.cat([_create_tensor_box(), _create_tensor_box()])
        assert Boxes(two_boxes)

        # 2 boxes in batch (B, 1, 4, 2) where B=2
        batched_bbox = torch.stack([_create_tensor_box(), _create_tensor_box()])
        assert Boxes(batched_bbox)

    def test_get_boxes_shape(self, device, dtype):
        box = Boxes(torch.tensor([[[1.0, 1.0], [3.0, 2.0], [1.0, 2.0], [3.0, 1.0]]], device=device, dtype=dtype))
        t_boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [1.0, 2.0], [3.0, 2.0]], [[5.0, 4.0], [2.0, 2.0], [5.0, 2.0], [2.0, 4.0]]],
            device=device,
            dtype=dtype,
        )  # (2, 4, 2)
        boxes = Boxes(t_boxes)
        boxes_batch = Boxes(t_boxes[None])  # (1, 2, 4, 2)

        # Single box
        h, w = box.get_boxes_shape()
        assert (h.item(), w.item()) == (2, 3)

        # Boxes
        h, w = boxes.get_boxes_shape()
        assert h.ndim == 1
        assert w.ndim == 1
        assert len(h) == 2
        assert len(w) == 2
        self.assert_close(h, torch.as_tensor([2.0, 3.0], device=device, dtype=dtype))
        self.assert_close(w, torch.as_tensor([3.0, 4.0], device=device, dtype=dtype))

        # Box batch
        h, w = boxes_batch.get_boxes_shape()
        assert h.ndim == 2
        assert w.ndim == 2
        assert h.shape == (1, 2)
        assert w.shape == (1, 2)
        self.assert_close(h, torch.as_tensor([[2.0, 3.0]], device=device, dtype=dtype))
        self.assert_close(w, torch.as_tensor([[3.0, 4.0]], device=device, dtype=dtype))

    def test_get_boxes_shape_batch(self, device, dtype):
        t_box1 = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [3.0, 1.0], [1.0, 2.0]]], device=device, dtype=dtype)
        t_box2 = torch.tensor([[[5.0, 2.0], [2.0, 2.0], [5.0, 4.0], [2.0, 4.0]]], device=device, dtype=dtype)
        batched_boxes = Boxes(torch.stack([t_box1, t_box2]))

        h, w = batched_boxes.get_boxes_shape()
        assert h.ndim == 2
        assert w.ndim == 2
        assert h.shape == (2, 1)
        assert w.shape == (2, 1)
        self.assert_close(h, torch.as_tensor([[2], [3]], device=device, dtype=dtype))
        self.assert_close(w, torch.as_tensor([[3], [4]], device=device, dtype=dtype))

    @pytest.mark.parametrize("shape", [(1, 4), (1, 1, 4)])
    def test_from_tensor(self, shape, device, dtype):
        box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        box_xyxy_plus = torch.as_tensor([[1, 2, 2, 3]], device=device, dtype=dtype).view(*shape)
        box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        box_vertices = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(*shape, 2)
        box_vertices_plus = torch.as_tensor([[[1, 2], [2, 2], [2, 3], [1, 3]]], device=device, dtype=dtype).view(
            *shape, 2
        )

        expected_box = torch.as_tensor([[[1, 2], [2, 2], [2, 3], [1, 3]]], device=device, dtype=dtype).view(*shape, 2)

        boxes_xyxy = Boxes.from_tensor(box_xyxy, mode="xyxy").data
        boxes_xyxy_plus = Boxes.from_tensor(box_xyxy_plus, mode="xyxy_plus").data
        boxes_xywh = Boxes.from_tensor(box_xywh, mode="xywh").data
        box_vertices = Boxes.from_tensor(box_vertices, mode="vertices").data
        boxes_vertices_plus = Boxes.from_tensor(box_vertices_plus, mode="vertices_plus").data

        assert boxes_xyxy.shape == expected_box.shape
        self.assert_close(boxes_xyxy, expected_box)

        assert boxes_xyxy_plus.shape == expected_box.shape
        self.assert_close(boxes_xyxy_plus, expected_box)

        assert boxes_xywh.shape == expected_box.shape
        self.assert_close(boxes_xywh, expected_box)

        assert box_vertices.shape == expected_box.shape
        self.assert_close(box_vertices, expected_box)

        assert boxes_vertices_plus.shape == expected_box.shape
        self.assert_close(boxes_vertices_plus, expected_box)

    @pytest.mark.parametrize("shape", [(1, 4), (1, 1, 4)])
    def test_from_invalid_tensor(self, shape, device, dtype):
        box_xyxy = torch.as_tensor([[1, 2, -3, 4]], device=device, dtype=dtype).view(*shape)  # Invalid width
        box_xyxy_plus = torch.as_tensor([[1, 2, 0, 3]], device=device, dtype=dtype).view(*shape)  # Invalid height

        try:
            Boxes.from_tensor(box_xyxy, mode="xyxy")
            raise AssertionError("Boxes.from_tensor should have raised any exception")
        except ValueError:
            pass

        try:
            Boxes.from_tensor(box_xyxy_plus, mode="xyxy_plus")
            raise AssertionError("Boxes.from_tensor should have raised any exception")
        except ValueError:
            pass

    @pytest.mark.parametrize("shape", [(1, 4), (1, 1, 4)])
    def test_boxes_to_tensor(self, shape, device, dtype):
        # quadrilateral with randomized vertices to reflect possible transforms.
        box = Boxes(torch.as_tensor([[[2, 2], [2, 3], [1, 3], [1, 2]]], device=device, dtype=dtype).view(*shape, 2))

        expected_box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        expected_box_xyxy_plus = torch.as_tensor([[1, 2, 2, 3]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_vertices = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(
            *shape, 2
        )
        expected_vertices_plus = torch.as_tensor([[[1, 2], [2, 2], [2, 3], [1, 3]]], device=device, dtype=dtype).view(
            *shape, 2
        )

        boxes_xyxy = box.to_tensor(mode="xyxy")
        boxes_xyxy_plus = box.to_tensor(mode="xyxy_plus")
        boxes_xywh = box.to_tensor(mode="xywh")
        boxes_vertices = box.to_tensor(mode="vertices")
        boxes_vertices_plus = box.to_tensor(mode="vertices_plus")

        assert boxes_xyxy.shape == expected_box_xyxy.shape
        self.assert_close(boxes_xyxy, expected_box_xyxy)

        assert boxes_xyxy_plus.shape == expected_box_xyxy_plus.shape
        self.assert_close(boxes_xyxy_plus, expected_box_xyxy_plus)

        assert boxes_xywh.shape == expected_box_xywh.shape
        self.assert_close(boxes_xywh, expected_box_xywh)

        assert boxes_vertices.shape == expected_vertices.shape
        self.assert_close(boxes_vertices, expected_vertices)

        assert boxes_vertices_plus.shape == expected_vertices_plus.shape
        self.assert_close(boxes_vertices_plus, expected_vertices_plus)

    @pytest.mark.parametrize("mode", ["xyxy", "xyxy_plus", "xywh", "vertices", "vertices_plus"])
    def test_boxes_list_to_tensor_list(self, mode, device, dtype):
        if mode == "vertices":
            item = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype)
        elif mode == "vertices_plus":
            item = torch.as_tensor([[[1, 2], [2, 2], [2, 3], [1, 3]]], device=device, dtype=dtype)
        else:
            item = torch.as_tensor([[1, 1, 5, 5]], device=device, dtype=dtype)
        src = [item, torch.cat([item, item])]
        box = Boxes.from_tensor(src, mode=mode)
        out = box.to_tensor(mode)
        assert isinstance(out, list)
        self.assert_close(out[0], src[0], atol=0.0, rtol=0.0)
        self.assert_close(out[1], src[1], atol=0.0, rtol=0.0)

    def test_boxes_to_mask(self, device, dtype):
        t_box1 = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype
        )  # (1, 4, 2)
        t_box2 = torch.tensor(
            [[[2.0, 2.0], [4.0, 2.0], [4.0, 5.0], [2.0, 4.0]]], device=device, dtype=dtype
        )  # (1, 4, 2)
        box1, box2 = Boxes(t_box1), Boxes(t_box2)
        two_boxes = Boxes(torch.cat([t_box1, t_box2]))  # (2, 4, 2)
        batched_boxes = Boxes(torch.stack([t_box1, t_box2]))  # (2, 1, 4, 2)

        height, width = 7, 5

        expected_mask1 = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_mask2 = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected_two_masks = torch.cat([expected_mask1, expected_mask2])
        expected_batched_masks = torch.stack([expected_mask1, expected_mask2])

        mask1 = box1.to_mask(height, width)
        mask2 = box2.to_mask(height, width)
        two_masks = two_boxes.to_mask(height, width)
        batched_masks = batched_boxes.to_mask(height, width)

        assert mask1.shape == expected_mask1.shape
        self.assert_close(mask1, expected_mask1)

        assert mask2.shape == expected_mask2.shape
        self.assert_close(mask2, expected_mask2)

        assert two_masks.shape == expected_two_masks.shape
        self.assert_close(two_masks, expected_two_masks)

        assert batched_masks.shape == expected_batched_masks.shape
        self.assert_close(batched_masks, expected_batched_masks)

    def test_to(self, device, dtype):
        boxes = Boxes.from_tensor(torch.as_tensor([[1, 2, 3, 4]], device="cpu", dtype=torch.float32))
        assert boxes.to(device=device).data.device == device
        assert boxes.to(dtype=dtype).data.dtype == dtype

        boxes_moved = boxes.to(device, dtype)
        assert boxes_moved is boxes  # to is an inplace op.
        assert boxes_moved.data.device == device, boxes_moved.data.dtype == dtype

    def test_gradcheck(self, device):
        def apply_boxes_method(tensor: torch.Tensor, method: str, **kwargs):
            boxes = Boxes(tensor)
            result = getattr(boxes, method)(**kwargs)
            return result.data if isinstance(result, Boxes) else result

        t_boxes1 = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=torch.float64)

        t_boxes2 = t_boxes1.detach().clone()
        t_boxes3 = t_boxes1.detach().clone()
        t_boxes4 = t_boxes1.detach().clone()
        t_boxes_xyxy = torch.tensor([[1.0, 3.0, 5.0, 6.0]])
        t_boxes_xyxy1 = t_boxes_xyxy.detach().clone()

        self.gradcheck(partial(apply_boxes_method, method="to_tensor"), (t_boxes2,))
        self.gradcheck(partial(apply_boxes_method, method="to_tensor", mode="xyxy_plus"), (t_boxes3,))
        self.gradcheck(partial(apply_boxes_method, method="to_tensor", mode="vertices_plus"), (t_boxes4,))
        self.gradcheck(partial(apply_boxes_method, method="get_boxes_shape"), (t_boxes1,))
        self.gradcheck(lambda x: Boxes.from_tensor(x, mode="xyxy_plus").data, (t_boxes_xyxy,))
        self.gradcheck(lambda x: Boxes.from_tensor(x, mode="xywh").data, (t_boxes_xyxy1,))

    def test_merge(self, device, dtype):
        # --- Unbatched: (N, 4, 2) + (M, 4, 2) -> (N+M, 4, 2) ---
        # On base (dim=1) this produced shape (1, 8, 2) and to_tensor("xyxy")
        # returned the single union row [[1, 2, 9, 8]] instead of two rows.
        a = Boxes.from_tensor(torch.tensor([[1.0, 2.0, 5.0, 4.0]], device=device, dtype=dtype), mode="xyxy")
        b = Boxes.from_tensor(torch.tensor([[6.0, 3.0, 9.0, 8.0]], device=device, dtype=dtype), mode="xyxy")
        m = a.merge(b)

        assert m.data.shape == (2, 4, 2), f"Unbatched merge shape: expected (2, 4, 2), got {m.data.shape}"
        expected_unbatched = torch.tensor([[1.0, 2.0, 5.0, 4.0], [6.0, 3.0, 9.0, 8.0]], device=device, dtype=dtype)
        self.assert_close(m.to_tensor("xyxy"), expected_unbatched)

        # --- Batched: (B, N, 4, 2) + (B, M, 4, 2) -> (B, N+M, 4, 2) ---
        # Verify that dim=-3 leaves the batched axis unchanged.
        a_b = Boxes.from_tensor(
            torch.tensor([[[1.0, 2.0, 5.0, 4.0]], [[0.0, 0.0, 3.0, 3.0]]], device=device, dtype=dtype), mode="xyxy"
        )
        b_b = Boxes.from_tensor(
            torch.tensor([[[6.0, 3.0, 9.0, 8.0]], [[4.0, 4.0, 7.0, 7.0]]], device=device, dtype=dtype), mode="xyxy"
        )
        m_b = a_b.merge(b_b)

        assert m_b.data.shape == (2, 2, 4, 2), f"Batched merge shape: expected (2, 2, 4, 2), got {m_b.data.shape}"
        expected_batched = torch.tensor(
            [[[1.0, 2.0, 5.0, 4.0], [6.0, 3.0, 9.0, 8.0]], [[0.0, 0.0, 3.0, 3.0], [4.0, 4.0, 7.0, 7.0]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(m_b.to_tensor("xyxy"), expected_batched)

        # --- Inplace path returns self and updates data ---
        a2 = Boxes.from_tensor(torch.tensor([[1.0, 2.0, 5.0, 4.0]], device=device, dtype=dtype), mode="xyxy")
        b2 = Boxes.from_tensor(torch.tensor([[6.0, 3.0, 9.0, 8.0]], device=device, dtype=dtype), mode="xyxy")
        result = a2.merge(b2, inplace=True)
        assert result is a2, "inplace=True must return self"
        assert a2.data.shape == (2, 4, 2)
        self.assert_close(a2.to_tensor("xyxy"), expected_unbatched)

        # --- List padding is moved behind merged boxes and remains metadata ---
        # Build from a list so that _N is set (variable-length list padding).
        # from_tensor with a list of (N, 4) tensors of different lengths produces a
        # batched (B, max_N, 4, 2) tensor with _N recording per-batch padding.
        src1 = torch.tensor([[1.0, 2.0, 5.0, 4.0]], device=device, dtype=dtype)  # (1, 4) — 1 box
        src2 = torch.tensor(
            [[1.0, 2.0, 5.0, 4.0], [6.0, 3.0, 9.0, 8.0]], device=device, dtype=dtype
        )  # (2, 4) — 2 boxes
        list_boxes = Boxes.from_tensor([src1, src2], mode="xyxy")
        assert list_boxes._N is not None, "Prerequisite: _N must be set for list-constructed Boxes"
        # list_boxes is batched (B=2, max_N=2, 4, 2); extra must match batch dim B=2.
        extra_batched = Boxes.from_tensor(
            torch.tensor([[[6.0, 3.0, 9.0, 8.0]], [[0.0, 0.0, 3.0, 3.0]]], device=device, dtype=dtype),
            mode="xyxy",
        )
        merged_list = list_boxes.merge(extra_batched)
        assert merged_list._N == [1, 0]
        merged_tensors = merged_list.to_tensor("xyxy")
        assert isinstance(merged_tensors, list)
        extra_tensors = extra_batched.to_tensor("xyxy")
        assert isinstance(extra_tensors, torch.Tensor)
        self.assert_close(merged_tensors[0], torch.cat([src1, extra_tensors[0]]))
        self.assert_close(merged_tensors[1], torch.cat([src2, extra_tensors[1]]))

        # Both operands may be list-backed, so their per-row padding counts must be combined.
        other_list = Boxes.from_tensor(
            [
                torch.tensor([[6.0, 3.0, 9.0, 8.0], [0.0, 0.0, 3.0, 3.0]], device=device, dtype=dtype),
                torch.tensor([[4.0, 4.0, 7.0, 7.0]], device=device, dtype=dtype),
            ],
            mode="xyxy",
        )
        other_tensors = other_list.to_tensor("xyxy")
        assert isinstance(other_tensors, list)
        expected_list = [torch.cat([src1, other_tensors[0]]), torch.cat([src2, other_tensors[1]])]

        merged_lists = list_boxes.merge(other_list)
        assert merged_lists._N == [1, 1]
        actual_list = merged_lists.to_tensor("xyxy")
        assert isinstance(actual_list, list)
        for actual, expected in zip(actual_list, expected_list):
            self.assert_close(actual, expected)

        # The dense operand may also be on the left of a list-backed operand.
        dense_first = extra_batched.merge(other_list)
        assert dense_first._N == [0, 1]
        actual_dense_first = dense_first.to_tensor("xyxy")
        assert isinstance(actual_dense_first, list)
        for dense_row, other_row, actual in zip(extra_tensors, other_tensors, actual_dense_first):
            self.assert_close(actual, torch.cat([dense_row, other_row]))

        result = list_boxes.merge(other_list, inplace=True)
        assert result is list_boxes
        assert list_boxes._N == [1, 1]
        actual_inplace = list_boxes.to_tensor("xyxy")
        assert isinstance(actual_inplace, list)
        for actual, expected in zip(actual_inplace, expected_list):
            self.assert_close(actual, expected)

    def test_compute_area(self):
        # Rectangle
        box_1 = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
        # Trapezoid
        box_2 = [[0.0, 0.0], [60.0, 0.0], [40.0, 50.0], [20.0, 50.0]]
        # Parallelogram
        box_3 = [[0.0, 0.0], [100.0, 0.0], [120.0, 50.0], [20.0, 50.0]]
        # Random quadrilateral
        box_4 = [
            [50.0, 50.0],
            [150.0, 250.0],
            [0.0, 500.0],
            [27.0, 80],
        ]
        # Random quadrilateral
        box_5 = [
            [0.0, 0.0],
            [150.0, 0.0],
            [150.0, 150.0],
            [0.0, 0.5],
        ]
        # Rectangle with minus coordinates
        box_6 = [[-500.0, -500.0], [-300.0, -500.0], [-300.0, -300.0], [-500.0, -300.0]]

        expected_values = [5000.0, 2000.0, 5000.0, 31925.0, 11287.5, 40000.0]
        box_coordinates = torch.tensor([box_1, box_2, box_3, box_4, box_5, box_6])
        computed_areas = Boxes(box_coordinates).compute_area().tolist()
        computed_areas_w_batch = Boxes(box_coordinates.reshape(2, 3, 4, 2)).compute_area().tolist()
        flattened_computed_areas_w_batch = [area for batch in computed_areas_w_batch for area in batch]
        assert all(
            computed_area == expected_area for computed_area, expected_area in zip(computed_areas, expected_values)
        )
        assert all(
            computed_area == expected_area
            for computed_area, expected_area in zip(flattened_computed_areas_w_batch, expected_values)
        )

    def test_wart_compute_area_is_shoelace_of_inclusive_vertices_4010(self, device, dtype):
        # Wart pin for kornia#4010: compute_area applies shoelace to the stored
        # inclusive vertices. A valid exclusive 2-by-1 box collapses to a line,
        # and a raw four-by-three rectangle has area six rather than the twelve
        # reported by get_boxes_shape. These are current values, not a contract.
        two_by_one = Boxes.from_tensor(torch.tensor([[[1.0, 1.0, 3.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        four_by_three = Boxes(
            torch.tensor([[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        )
        self.assert_close(
            two_by_one.compute_area(), torch.tensor([[0.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        self.assert_close(
            four_by_three.compute_area(), torch.tensor([6.0], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )

    @pytest.mark.xfail(strict=True, reason="kornia#4010: compute_area disagrees with get_boxes_shape")
    def test_convention_compute_area_matches_get_boxes_shape_product_4010(self, device, dtype):
        # The intended contract is that area agrees with the container's own
        # height and width terms. A repair must XPASS this strict xfail.
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 1.0, 3.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        heights, widths = boxes.get_boxes_shape()
        self.assert_close(boxes.compute_area(), heights * widths)

    def test_convention_pad_and_unpad_use_left_right_top_bottom_in_place(self, device, dtype):
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 2.0, 5.0, 4.0]]], device=device, dtype=dtype), mode="xyxy")
        original = boxes.data.clone()
        padding = torch.tensor([[10.0, 99.0, 20.0, 88.0]], device=device, dtype=dtype)
        result = boxes.pad(padding)
        assert result is boxes
        expected = original + torch.tensor([10.0, 20.0], device=device, dtype=dtype)
        self.assert_close(boxes.data, expected, atol=0.0, rtol=0.0)
        assert boxes.unpad(padding) is boxes
        self.assert_close(boxes.data, original, atol=0.0, rtol=0.0)

    def test_wart_clamp_tuple_bounds_and_outside_box_behavior_4017(self, device, dtype):
        # Wart pin for kornia#4017: tuple bounds are advertised but unsupported;
        # tensor bounds clamp every vertex, collapsing a wholly outside box.
        boxes = Boxes.from_tensor(
            torch.tensor([[[8.0, 9.0, 10.0, 11.0]]], device=device, dtype=dtype),
            mode="xyxy",
        )
        with pytest.raises(NotImplementedError):
            boxes.clamp((0, 0), (5, 5))
        clamped = boxes.clamp(
            torch.tensor([[0.0, 0.0]], device=device, dtype=dtype),
            torch.tensor([[5.0, 5.0]], device=device, dtype=dtype),
        )
        expected = torch.full((1, 1, 4, 2), 5.0, device=device, dtype=dtype)
        self.assert_close(clamped.data, expected, atol=0.0, rtol=0.0)
        self.assert_close(
            boxes.data,
            Boxes.from_tensor(torch.tensor([[[8.0, 9.0, 10.0, 11.0]]], device=device, dtype=dtype), mode="xyxy").data,
            atol=0.0,
            rtol=0.0,
        )

    def test_wart_trim_and_fast_translate_are_unimplemented_4017(self, device, dtype):
        # Wart pin for kornia#4017: both documented entry points raise instead
        # of implementing their advertised operations.
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 2.0, 5.0, 4.0]]], device=device, dtype=dtype), mode="xyxy")
        with pytest.raises(NotImplementedError):
            boxes.trim()
        with pytest.raises(NotImplementedError):
            boxes.translate(torch.tensor([[1.0, 2.0]], device=device, dtype=dtype), method="fast")

    def test_convention_transform_boxes_in_place_rebinds_data(self, device, dtype):
        # transform_boxes leaves its input unchanged by default. Its in-place
        # twin returns self but rebinds storage, leaving prior data references stale.
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 2.0, 5.0, 4.0]]], device=device, dtype=dtype), mode="xyxy")
        original = boxes.data
        matrix = torch.tensor([[[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        transformed = boxes.transform_boxes(matrix)
        assert transformed is not boxes
        self.assert_close(transformed.data, original + torch.tensor([10.0, 20.0], device=device, dtype=dtype))
        self.assert_close(boxes.data, original, atol=0.0, rtol=0.0)
        assert boxes.transform_boxes_(matrix) is boxes
        assert boxes.data.data_ptr() != original.data_ptr()
        self.assert_close(
            original,
            Boxes.from_tensor(torch.tensor([[[1.0, 2.0, 5.0, 4.0]]], device=device, dtype=dtype), mode="xyxy").data,
            atol=0.0,
            rtol=0.0,
        )

    def test_convention_translate_warp_uses_batched_xy_displacements(self, device, dtype):
        # Convention pin: each row of size supplies an (x, y) displacement for
        # its batch item. Asymmetric values catch an accidental axis swap.
        boxes = Boxes.from_tensor(
            torch.tensor([[[1.0, 2.0, 5.0, 4.0]], [[10.0, 20.0, 14.0, 23.0]]], device=device, dtype=dtype),
            mode="xyxy",
        )
        translated = boxes.translate(torch.tensor([[3.0, -7.0], [-5.0, 11.0]], device=device, dtype=dtype))
        expected = torch.tensor(
            [
                [[[4.0, -5.0], [7.0, -5.0], [7.0, -4.0], [4.0, -4.0]]],
                [[[5.0, 31.0], [8.0, 31.0], [8.0, 33.0], [5.0, 33.0]]],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(translated.data, expected, atol=0.0, rtol=0.0)
        assert translated is not boxes

    def test_wart_filter_boxes_by_area_zeroes_small_boxes_4010(self, device, dtype):
        # Wart pin for kornia#4010: filtering acts on compute_area, so the
        # valid two-by-one box with shoelace area zero is zeroed, not removed.
        boxes = Boxes.from_tensor(torch.tensor([[[1.0, 1.0, 3.0, 2.0]]], device=device, dtype=dtype), mode="xyxy")
        filtered = boxes.filter_boxes_by_area(min_area=1.0)
        assert filtered is not boxes
        self.assert_close(filtered.data, torch.zeros_like(boxes.data), atol=0.0, rtol=0.0)
        assert filtered.data.shape == boxes.data.shape
        assert not torch.equal(boxes.data, torch.zeros_like(boxes.data))

    def test_convention_filter_boxes_by_area_maximum_zeroes_in_place(self, device, dtype):
        # The shoelace areas are 2 and 8. Equal lower/upper bounds retain the
        # first box, pinning both inclusive endpoints; the larger box is zeroed.
        # In-place filtering returns the original wrapper and keeps its shape.
        boxes = Boxes(
            torch.tensor(
                [
                    [
                        [[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]],
                        [[1.0, 1.0], [5.0, 1.0], [5.0, 3.0], [1.0, 3.0]],
                    ]
                ],
                device=device,
                dtype=dtype,
            )
        )
        first = boxes.data[:, :1].clone()
        assert boxes.filter_boxes_by_area(min_area=2.0, max_area=2.0, inplace=True) is boxes
        self.assert_close(boxes.data[:, :1], first, atol=0.0, rtol=0.0)
        self.assert_close(boxes.data[:, 1:], torch.zeros_like(boxes.data[:, 1:]), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("operation, inplace", [("pad", None), ("unpad", None), ("clamp", False), ("clamp", True)])
    @pytest.mark.parametrize("num_boxes", [1, 2])
    def test_wart_unbatched_geometry_operations_raise_4244(self, operation, inplace, num_boxes, device, dtype):
        # Wart pin for kornia#4244: the documented unbatched (N, 4, 2) form
        # fails although its singleton-batched counterpart works. Clamp reaches
        # indexing failure for one box and broadcasting failure for two boxes.
        boxes = Boxes(_unbatched_geometry_data(device, dtype)[:num_boxes])
        if operation == "clamp":
            with pytest.raises((RuntimeError, IndexError)):
                boxes.clamp(
                    torch.tensor([[2.0, 3.0]], device=device, dtype=dtype),
                    torch.tensor([[6.0, 7.0]], device=device, dtype=dtype),
                    inplace=inplace,
                )
        else:
            with pytest.raises((RuntimeError, IndexError)):
                getattr(boxes, operation)(torch.tensor([[10.0, 99.0, 20.0, 88.0]], device=device, dtype=dtype))

    @pytest.mark.parametrize("operation, inplace", [("pad", None), ("unpad", None), ("clamp", False), ("clamp", True)])
    @pytest.mark.parametrize("num_boxes", [1, 2])
    @pytest.mark.xfail(
        strict=True,
        raises=(RuntimeError, IndexError),
        reason="kornia#4244: unbatched geometry operations do not match singleton batches",
    )
    def test_convention_unbatched_geometry_operations_match_singleton_batch_4244(
        self, operation, inplace, num_boxes, device, dtype
    ):
        data = _unbatched_geometry_data(device, dtype)[:num_boxes]
        batched = Boxes(data[None].clone())
        unbatched = Boxes(data.clone())
        if operation == "clamp":
            limits = (
                torch.tensor([[2.0, 3.0]], device=device, dtype=dtype),
                torch.tensor([[6.0, 7.0]], device=device, dtype=dtype),
            )
            try:
                expected = batched.clamp(*limits, inplace=inplace)
            except (RuntimeError, IndexError) as error:
                raise AssertionError("The supported singleton-batch reference must succeed") from error
            assert (expected is batched) is inplace
            actual = unbatched.clamp(*limits, inplace=inplace)
            assert (actual is unbatched) is inplace
            if not inplace:
                self.assert_close(unbatched.data, data, atol=0.0, rtol=0.0)
        else:
            padding = torch.tensor([[10.0, 99.0, 20.0, 88.0]], device=device, dtype=dtype)
            try:
                expected = getattr(batched, operation)(padding)
            except (RuntimeError, IndexError) as error:
                raise AssertionError("The supported singleton-batch reference must succeed") from error
            assert expected is batched
            actual = getattr(unbatched, operation)(padding)
            assert actual is unbatched
        self.assert_close(actual.data, expected.data.squeeze(0), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_wart_transform_boxes_empty_copy_aliases_input_4020(self, batched, inplace, device, dtype):
        # Wart pin for tracking issue #4020: transforming an empty container
        # preserves its tensor storage. The non-inplace wrapper is new but
        # aliases the input data; the in-place wrapper remains self.
        data = torch.empty((1, 0, 4, 2) if batched else (0, 4, 2), device=device, dtype=dtype)
        boxes = Boxes(data)
        original = boxes.data
        matrix = torch.eye(3, device=device, dtype=dtype)
        transformed = boxes.transform_boxes_(matrix) if inplace else boxes.transform_boxes(matrix)
        assert (transformed is boxes) is inplace
        assert transformed.data is original
        assert transformed.data.shape == original.shape

    def test_wart_to_mask_and_bbox_to_mask_take_opposite_size_orders_4014(self, device, dtype):
        # Wart pin for kornia#4014: Boxes.to_mask(height, width) and bbox_to_mask(boxes, width, height)
        # are the same call, and both return (N, height, width) in the box dtype.
        vertices = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        method = Boxes(vertices).to_mask(3, 5)
        assert method.shape == (1, 3, 5)
        assert method.dtype == dtype
        self.assert_close(method, bbox_to_mask(vertices, 5, 3), atol=0.0, rtol=0.0)
        assert Boxes(vertices).to_mask(5, 3).shape == (1, 5, 3)
        assert Boxes(vertices[None]).to_mask(3, 5).shape == (1, 1, 3, 5)

    def test_wart_to_mask_rounds_the_exclusive_export_half_open_4015(self, device, dtype):
        # Wart pin for kornia#4015: the fractional box [1.4, 3.6] x [1.4, 2.6] exports as xyxy
        # [1.4, 1.4, 4.6, 3.6], rounds to [1, 1, 5, 4], and fills the half-open ranges: twelve pixels,
        # where bbox_to_mask fills two (test_bbox.py). A box entirely outside the image is clamped
        # onto the border and fills nothing, and a box that requires grad is rejected.
        boxes = torch.tensor([[[1.4, 1.4], [3.6, 1.4], [3.6, 2.6], [1.4, 2.6]]], device=device, dtype=dtype)
        expected = torch.zeros(1, 5, 6, device=device, dtype=dtype)
        expected[0, 1:4, 1:5] = 1.0
        self.assert_close(Boxes(boxes).to_mask(5, 6), expected, atol=0.0, rtol=0.0)
        assert bbox_to_mask(boxes, 6, 5).sum().item() == 2.0

        outside = torch.tensor([[[6.0, 6.0], [9.0, 6.0], [9.0, 9.0], [6.0, 9.0]]], device=device, dtype=dtype)
        assert Boxes(outside).to_mask(5, 5).sum().item() == 0.0
        with pytest.raises(RuntimeError, match="differentiable"):
            Boxes(boxes.clone().requires_grad_()).to_mask(5, 6)

    def test_wart_to_mask_fills_the_origin_pixel_for_list_padding_rows(self, device, dtype):
        # Wart pin: a list-backed object exports its zero padding rows as the exclusive xyxy box
        # [0, 0, 1, 1] (see the padding note on the class), so to_mask marks pixel (0, 0) in the
        # mask channel of every padding entry instead of leaving it empty.
        box = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        boxes = Boxes([box, torch.cat([box, box + 2.0])])
        mask = boxes.to_mask(5, 5)
        assert mask.shape == (2, 2, 5, 5)
        expected_padding = torch.zeros(5, 5, device=device, dtype=dtype)
        expected_padding[0, 0] = 1.0
        self.assert_close(mask[0, 1], expected_padding, atol=0.0, rtol=0.0)
        self.assert_close(mask[0, 0], mask[1, 0], atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("case", ["fractional", "outside", "negative", "rotated", "batched"])
    def test_convention_to_mask_export_path_matches_loop_path(self, case, device, dtype, monkeypatch):
        # Convention pin: the CPU/MPS loop path and the vectorized path taken on CUDA and under
        # export produce the same mask. The vectorized branch is selected by patching the module's
        # is_exporting predicate; on CUDA both calls already take it.
        fractional = torch.tensor([[[1.4, 1.4], [3.6, 1.4], [3.6, 2.6], [1.4, 2.6]]], device=device, dtype=dtype)
        data = {
            "fractional": fractional,
            "outside": torch.tensor([[[3.0, 3.0], [9.0, 3.0], [9.0, 9.0], [3.0, 9.0]]], device=device, dtype=dtype),
            "negative": fractional - 3.0,
            "rotated": torch.tensor([[[2.0, 0.0], [4.0, 2.0], [2.0, 4.0], [0.0, 2.0]]], device=device, dtype=dtype),
            "batched": fractional.expand(2, 3, 4, 2).contiguous(),
        }[case]
        loop = Boxes(data).to_mask(5, 6)
        monkeypatch.setattr(boxes_module, "is_exporting", lambda: True)
        vectorized = Boxes(data).to_mask(5, 6)
        assert loop.shape == vectorized.shape
        self.assert_close(loop, vectorized, atol=0.0, rtol=0.0)


class TestTransformBoxes2D(BaseTester):
    def test_transform_boxes(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor([[139.2640, 103.0150, 398.3120, 411.5225]], device=device, dtype=dtype)
        expected_boxes_xyxy = torch.tensor([[372.7360, 103.0150, 115.6880, 411.5225]], device=device, dtype=dtype)

        boxes = Boxes.from_tensor(boxes_xyxy)
        expected_boxes = Boxes.from_tensor(expected_boxes_xyxy, validate_boxes=False)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        transformed_boxes = boxes.transform_boxes(trans_mat)
        self.assert_close(transformed_boxes.data, expected_boxes.data, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is not boxes

    def test_transform_boxes_(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor([[139.2640, 103.0150, 398.3120, 411.5225]], device=device, dtype=dtype)
        expected_boxes_xyxy = torch.tensor([[372.7360, 103.0150, 115.6880, 411.5225]], device=device, dtype=dtype)

        boxes = Boxes.from_tensor(boxes_xyxy)
        expected_boxes = Boxes.from_tensor(expected_boxes_xyxy, validate_boxes=False)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        transformed_boxes = boxes.transform_boxes_(trans_mat)
        self.assert_close(transformed_boxes.data, expected_boxes.data, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is boxes

    def test_transform_multiple_boxes(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor(
            [
                [139.2640, 103.0150, 398.3120, 411.5225],
                [1.0240, 80.5547, 513.0000, 513.0000],
                [165.2053, 262.1440, 511.6347, 509.9280],
                [119.8080, 144.2067, 258.0240, 411.1292],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1)  # 2 x 4 x 4 two images 4 boxes each

        expected_boxes_xyxy = torch.tensor(
            [
                [
                    [372.7360, 103.0150, 115.6880, 411.5225],
                    [510.9760, 80.5547, 1.0000, 513.0000],
                    [346.7947, 262.1440, 2.3653, 509.9280],
                    [392.1920, 144.2067, 255.9760, 411.1292],
                ],
                [
                    [139.2640, 103.0150, 398.3120, 411.5225],
                    [1.0240, 80.5547, 513.0000, 513.0000],
                    [165.2053, 262.1440, 511.6347, 509.9280],
                    [119.8080, 144.2067, 258.0240, 411.1292],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor(
            [
                [[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )

        boxes = Boxes.from_tensor(boxes_xyxy)
        expected_boxes = Boxes.from_tensor(expected_boxes_xyxy, validate_boxes=False)

        out = boxes.transform_boxes(trans_mat)
        self.assert_close(out.data, expected_boxes.data, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor(
            [
                [139.2640, 103.0150, 258.0480, 307.5075],
                [1.0240, 80.5547, 510.9760, 431.4453],
                [165.2053, 262.1440, 345.4293, 546.7840],
                [119.8080, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=torch.float64,
        )
        boxes = Boxes.from_tensor(boxes_xyxy)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float64
        )

        def _wrapper_transform_boxes(quadrilaterals, M):
            boxes = Boxes(quadrilaterals)
            boxes = boxes.transform_boxes(M)
            return boxes.data

        self.gradcheck(_wrapper_transform_boxes, (boxes.data, trans_mat))


class TestBbox3D(BaseTester):
    def test_smoke(self, device, dtype):
        def _create_tensor_box():
            # Sample two points of the 3d rect
            points = torch.rand(1, 6, device=device, dtype=dtype)

            # Fill according missing points
            tensor_boxes = torch.zeros(1, 8, 3, device=device, dtype=dtype)
            tensor_boxes[0, 0] = points[0][:3]
            tensor_boxes[0, 1, 0] = points[0][3]
            tensor_boxes[0, 1, 1] = points[0][1]
            tensor_boxes[0, 1, 2] = points[0][2]
            tensor_boxes[0, 2, 0] = points[0][3]
            tensor_boxes[0, 2, 1] = points[0][4]
            tensor_boxes[0, 2, 2] = points[0][2]
            tensor_boxes[0, 3, 0] = points[0][0]
            tensor_boxes[0, 3, 1] = points[0][4]
            tensor_boxes[0, 3, 2] = points[0][2]
            tensor_boxes[0, 4, 0] = points[0][0]
            tensor_boxes[0, 4, 1] = points[0][1]
            tensor_boxes[0, 4, 2] = points[0][5]
            tensor_boxes[0, 5, 0] = points[0][3]
            tensor_boxes[0, 5, 1] = points[0][1]
            tensor_boxes[0, 5, 2] = points[0][5]
            tensor_boxes[0, 6] = points[0][3:]
            tensor_boxes[0, 7, 0] = points[0][0]
            tensor_boxes[0, 7, 1] = points[0][4]
            tensor_boxes[0, 7, 2] = points[0][5]
            return tensor_boxes

        # Validate
        assert Boxes3D(_create_tensor_box())  # Validate 1 box

        # 2 boxes without batching (N, 8, 3) where N=2
        two_boxes = torch.cat([_create_tensor_box(), _create_tensor_box()])
        assert Boxes3D(two_boxes)

        # 2 boxes in batch (B, 1, 8, 3) where B=2
        batched_bbox = torch.stack([_create_tensor_box(), _create_tensor_box()])
        assert Boxes3D(batched_bbox)

    def test_get_boxes_shape(self, device, dtype):
        box = Boxes3D(
            torch.tensor(
                [[[0, 1, 2], [0, 1, 32], [10, 21, 2], [0, 21, 2], [10, 1, 32], [10, 21, 32], [10, 1, 2], [0, 21, 32]]],
                device=device,
                dtype=dtype,
            )
        )  # 1x8x3
        t_boxes = torch.tensor(
            [
                [[0, 21, 32], [0, 1, 2], [10, 1, 2], [0, 21, 2], [0, 1, 32], [10, 21, 2], [10, 1, 32], [10, 21, 32]],
                [[3, 4, 5], [3, 4, 65], [43, 54, 5], [3, 54, 5], [43, 4, 5], [43, 4, 65], [43, 54, 65], [3, 54, 65]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x8x3
        boxes = Boxes3D(t_boxes)
        boxes_batch = Boxes3D(t_boxes[None])  # (1, 2, 8, 3)

        # Single box
        d, h, w = box.get_boxes_shape()
        assert (d.item(), h.item(), w.item()) == (31.0, 21.0, 11.0)

        # Boxes
        d, h, w = boxes.get_boxes_shape()
        assert h.ndim == 1
        assert w.ndim == 1
        assert len(d) == 2
        assert len(h) == 2
        assert len(w) == 2
        self.assert_close(d, torch.as_tensor([31.0, 61.0], device=device, dtype=dtype))
        self.assert_close(h, torch.as_tensor([21.0, 51.0], device=device, dtype=dtype))
        self.assert_close(w, torch.as_tensor([11.0, 41.0], device=device, dtype=dtype))

        # Box batch
        d, h, w = boxes_batch.get_boxes_shape()
        assert h.ndim == 2
        assert w.ndim == 2
        assert h.shape == (1, 2)
        assert w.shape == (1, 2)
        self.assert_close(d, torch.as_tensor([[31.0, 61.0]], device=device, dtype=dtype))
        self.assert_close(h, torch.as_tensor([[21.0, 51.0]], device=device, dtype=dtype))
        self.assert_close(w, torch.as_tensor([[11.0, 41.0]], device=device, dtype=dtype))

    def test_get_boxes_shape_batch(self, device, dtype):
        t_box1 = torch.tensor(
            [[[0, 1, 2], [0, 1, 32], [10, 21, 2], [0, 21, 2], [10, 1, 32], [10, 21, 32], [10, 1, 2], [0, 21, 32]]],
            device=device,
            dtype=dtype,
        )
        t_box2 = torch.tensor(
            [[[3, 4, 5], [3, 4, 65], [43, 54, 5], [3, 54, 5], [43, 4, 5], [43, 4, 65], [43, 54, 65], [3, 54, 65]]],
            device=device,
            dtype=dtype,
        )
        batched_boxes = Boxes3D(torch.stack([t_box1, t_box2]))

        d, h, w = batched_boxes.get_boxes_shape()
        assert d.ndim == 2
        assert h.ndim == 2
        assert w.ndim == 2
        assert d.shape == (2, 1)
        assert h.shape == (2, 1)
        assert w.shape == (2, 1)
        self.assert_close(d, torch.as_tensor([[31.0], [61.0]], device=device, dtype=dtype))
        self.assert_close(h, torch.as_tensor([[21.0], [51.0]], device=device, dtype=dtype))
        self.assert_close(w, torch.as_tensor([[11.0], [41.0]], device=device, dtype=dtype))

    @pytest.mark.parametrize("shape", [(1, 6), (1, 1, 6)])
    def test_from_tensor(self, shape, device, dtype):
        box_xyzxyz = torch.as_tensor([[1, 2, 3, 3, 5, 7]], device=device, dtype=dtype).view(*shape)
        box_xyzxyz_plus = torch.as_tensor([[1, 2, 3, 2, 4, 6]], device=device, dtype=dtype).view(*shape)
        box_xyzwhd = torch.as_tensor([[1, 2, 3, 2, 3, 4]], device=device, dtype=dtype).view(*shape)

        expected_box = torch.as_tensor(
            [[[1, 2, 3], [2, 2, 3], [2, 4, 3], [1, 4, 3], [1, 2, 6], [2, 2, 6], [2, 4, 6], [1, 4, 6]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        kornia_xyzxyz = Boxes3D.from_tensor(box_xyzxyz, mode="xyzxyz").data
        kornia_xyzxyz_plus = Boxes3D.from_tensor(box_xyzxyz_plus, mode="xyzxyz_plus").data
        kornia_xyzwhd = Boxes3D.from_tensor(box_xyzwhd, mode="xyzwhd").data

        assert kornia_xyzxyz.shape == expected_box.shape
        self.assert_close(kornia_xyzxyz, expected_box)

        assert kornia_xyzxyz_plus.shape == expected_box.shape
        self.assert_close(kornia_xyzxyz_plus, expected_box)

        assert kornia_xyzwhd.shape == expected_box.shape
        self.assert_close(kornia_xyzwhd, expected_box)

    @pytest.mark.parametrize("shape", [(1, 6), (1, 1, 6)])
    def test_from_invalid_tensor(self, shape, device, dtype):
        box_xyzxyz = torch.as_tensor([[1, 2, 3, 4, -5, 6]], device=device, dtype=dtype).view(*shape)
        box_xyzxyz_plus = torch.as_tensor([[1, 2, 3, 0, 6, 4]], device=device, dtype=dtype).view(*shape)

        try:
            Boxes3D.from_tensor(box_xyzxyz, mode="xyzxyz")
            raise AssertionError("Boxes3D.from_tensor should have raised any exception")
        except ValueError:
            pass

        try:
            Boxes3D.from_tensor(box_xyzxyz_plus, mode="xyzxyz_plus")
            raise AssertionError("Boxes3D.from_tensor should have raised any exception")
        except ValueError:
            pass

    @pytest.mark.parametrize("shape", [(1, 6), (1, 1, 6)])
    def test_boxes_to_tensor(self, shape, device, dtype):
        # Hexahedron with randomized vertices to reflect possible transforms.
        box = Boxes3D(
            torch.as_tensor(
                [[[2, 2, 1], [1, 2, 1], [2, 3, 2], [1, 3, 2], [2, 2, 2], [1, 3, 1], [2, 3, 1], [1, 2, 2]]],
                device=device,
                dtype=dtype,
            ).view(*shape[:-1], 8, 3)
        )

        expected_box_xyzxyz = torch.as_tensor([[1, 2, 1, 3, 4, 3]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzxyz_plus = torch.as_tensor([[1, 2, 1, 2, 3, 2]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzwhd = torch.as_tensor([[1, 2, 1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_vertices = torch.as_tensor(
            [[[1, 2, 1], [3, 2, 1], [3, 4, 1], [1, 4, 1], [1, 2, 3], [3, 2, 3], [3, 4, 3], [1, 4, 3]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)
        expected_vertices_plus = torch.as_tensor(
            [[[1, 2, 1], [2, 2, 1], [2, 3, 1], [1, 3, 1], [1, 2, 2], [2, 2, 2], [2, 3, 2], [1, 3, 2]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        kornia_xyzxyz = box.to_tensor(mode="xyzxyz")
        kornia_xyzxyz_plus = box.to_tensor(mode="xyzxyz_plus")
        kornia_xyzwhd = box.to_tensor(mode="xyzwhd")
        kornia_vertices = box.to_tensor(mode="vertices")
        kornia_vertices_plus = box.to_tensor(mode="vertices_plus")

        assert kornia_xyzxyz.shape == expected_box_xyzxyz.shape
        self.assert_close(kornia_xyzxyz, expected_box_xyzxyz)

        assert kornia_xyzxyz_plus.shape == expected_box_xyzxyz_plus.shape
        self.assert_close(kornia_xyzxyz_plus, expected_box_xyzxyz_plus)

        assert kornia_xyzwhd.shape == expected_box_xyzwhd.shape
        self.assert_close(kornia_xyzwhd, expected_box_xyzwhd)

        assert kornia_vertices.shape == expected_vertices.shape
        self.assert_close(kornia_vertices, expected_vertices)

        assert kornia_vertices_plus.shape == expected_vertices_plus.shape
        self.assert_close(kornia_vertices_plus, expected_vertices_plus)

    def test_bbox_to_mask(self, device, dtype):
        t_box1 = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0],  # Front
                    [1.0, 1.0, 2.0],
                    [3.0, 1.0, 2.0],
                    [3.0, 2.0, 2.0],
                    [1.0, 2.0, 2.0],  # Back
                ]
            ],
            device=device,
            dtype=dtype,
        )  # (1, 8, 3)
        t_box2 = torch.tensor(
            [
                [
                    [2.0, 2.0, 1.0],
                    [4.0, 2.0, 1.0],
                    [4.0, 5.0, 1.0],
                    [4.0, 2.0, 1.0],  # Front
                    [2.0, 2.0, 1.0],
                    [4.0, 2.0, 1.0],
                    [4.0, 5.0, 1.0],
                    [4.0, 5.0, 1.0],  # Back
                ]
            ],
            device=device,
            dtype=dtype,
        )  # (1, 8, 3)

        box1, box2 = Boxes3D(t_box1), Boxes3D(t_box2)
        two_boxes = Boxes3D(torch.cat([t_box1, t_box2]))  # (2, 8, 3)
        batched_boxes = Boxes3D(torch.stack([t_box1, t_box2]))  # (2, 1, 8, 3)

        depth, height, width = 3, 7, 5

        expected_mask1 = torch.tensor(
            [
                [
                    [  # Depth 0
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [  # Depth 1
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [  # Depth 2
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_mask2 = torch.tensor(
            [
                [
                    [  # Depth 0
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [  # Depth 1
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                    ],
                    [  # Depth 2
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected_two_masks = torch.cat([expected_mask1, expected_mask2])
        expected_batched_masks = torch.stack([expected_mask1, expected_mask2])

        mask1 = box1.to_mask(depth, height, width)
        mask2 = box2.to_mask(depth, height, width)
        two_masks = two_boxes.to_mask(depth, height, width)
        batched_masks = batched_boxes.to_mask(depth, height, width)

        assert mask1.shape == expected_mask1.shape
        self.assert_close(mask1, expected_mask1)

        assert mask2.shape == expected_mask2.shape
        self.assert_close(mask2, expected_mask2)

        assert two_masks.shape == expected_two_masks.shape
        self.assert_close(two_masks, expected_two_masks)

        assert batched_masks.shape == expected_batched_masks.shape
        self.assert_close(batched_masks, expected_batched_masks)

    def test_to(self, device, dtype):
        boxes = Boxes3D.from_tensor(torch.as_tensor([[1, 2, 3, 4, 5, 6]], device="cpu", dtype=torch.float32))
        assert boxes.to(device=device).data.device == device
        assert boxes.to(dtype=dtype).data.dtype == dtype

        boxes_moved = boxes.to(device, dtype)
        assert boxes_moved is boxes  # to is an inplace op.
        assert boxes_moved.data.device == device, boxes_moved.data.dtype == dtype

    def test_gradcheck(self, device):
        def apply_boxes_method(tensor: torch.Tensor, method: str, **kwargs):
            boxes = Boxes3D(tensor)
            result = getattr(boxes, method)(**kwargs)
            return result.data if isinstance(result, Boxes3D) else result

        # to_tensor (and get_boxes_shape, which calls it) reduce the 8 vertices with amin/amax, whose
        # backward is exact everywhere except where multiple vertices exactly tie for an axis extremum
        # -- see the Note on Boxes3D.to_tensor. An axis-aligned box has a 4-way tie on every face by
        # construction, so it is the wrong input for gradcheck's central-difference comparison: it
        # probes the reduction exactly at its one genuine kink, not a representative point. Jittering
        # every vertex breaks the ties without changing which corner is the true min/max, so gradcheck
        # verifies the reduction everywhere else, which is everywhere a real (non-degenerate) box lives.
        # The jitter is a fixed pattern, not RNG-derived: it only needs to be small and distinct per
        # component to break the exact ties, and a fixed pattern avoids mutating global RNG state that
        # could leak into other tests.
        t_boxes1 = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [10, 1, 2],
                    [10, 21, 2],
                    [0, 21, 2],
                    [0, 1, 32],
                    [10, 1, 32],
                    [10, 21, 32],
                    [0, 21, 32],
                ]
            ],
            device=device,
            dtype=torch.float64,
        )
        jitter = torch.arange(1, t_boxes1.numel() + 1, dtype=torch.float64, device=device).view_as(t_boxes1) * 1e-4
        t_boxes1 = t_boxes1 + jitter

        t_boxes2 = t_boxes1.detach().clone()
        t_boxes3 = t_boxes1.detach().clone()
        t_boxes4 = t_boxes1.detach().clone()

        self.gradcheck(partial(apply_boxes_method, method="to_tensor"), (t_boxes2,))
        self.gradcheck(partial(apply_boxes_method, method="to_tensor", mode="xyzxyz_plus"), (t_boxes3,))
        self.gradcheck(partial(apply_boxes_method, method="to_tensor", mode="vertices_plus"), (t_boxes4,))
        self.gradcheck(partial(apply_boxes_method, method="get_boxes_shape"), (t_boxes1.detach().clone(),))

        t_boxes_xyzxyz = torch.tensor([[1.0, 3.0, 8.0, 5.0, 6.0, 12.0]], device=device, dtype=torch.float64)
        t_boxes_xyzxyz1 = t_boxes_xyzxyz.detach().clone()
        self.gradcheck(lambda x: Boxes3D.from_tensor(x, mode="xyzxyz_plus").data, (t_boxes_xyzxyz,))
        self.gradcheck(lambda x: Boxes3D.from_tensor(x, mode="xyzwhd").data, (t_boxes_xyzxyz1,))

    def test_convention_to_tensor_tie_gradient_is_an_even_subgradient_1396(self, device):
        # #1396: to_tensor used to raise RuntimeError whenever its input required grad, because
        # gradcheck disagreed with the analytical gradient on an axis-aligned box -- every face of
        # such a box has a 4-way vertex tie, and PyTorch's amin/amax backward splits the gradient
        # evenly across tied vertices (1/4 each here) rather than picking one, which is a valid
        # subgradient but not what central-difference gradcheck expects at a kink (it does not probe
        # a genuine derivative there, since none exists in the classical sense). This pins that even
        # split as the actual, correct, and now-unguarded behavior, so a future change that alters it
        # (e.g. reverting to computing to_tensor without amin/amax) has to touch this test.
        vertices = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [10.0, 1.0, 2.0],
                    [10.0, 21.0, 2.0],
                    [0.0, 21.0, 2.0],
                    [0.0, 1.0, 32.0],
                    [10.0, 1.0, 32.0],
                    [10.0, 21.0, 32.0],
                    [0.0, 21.0, 32.0],
                ]
            ],
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        boxes = Boxes3D(vertices)
        out = boxes.to_tensor(mode="xyzxyz")  # (N=1, 6): not batched, so to_tensor squeezes the batch dim
        out[0, 0].backward()  # d(xmin)/d(vertices): xmin ties across vertices 0, 3, 4, 7

        expected_grad = torch.zeros_like(vertices)
        for tied_vertex in (0, 3, 4, 7):
            expected_grad[0, tied_vertex, 0] = 0.25
        self.assert_close(vertices.grad, expected_grad)

    @staticmethod
    def _asymmetric_xyzxyz(device, dtype) -> torch.Tensor:
        # Exclusive corners: x in 1..5, y in 2..5, z in 3..8, so width 4, height 3, depth 5.
        return torch.tensor([[1.0, 2.0, 3.0, 5.0, 5.0, 8.0]], device=device, dtype=dtype)

    def test_convention_from_tensor_xyzxyz_stores_inclusive_front_then_back_vertices(self, device, dtype):
        # Convention pin: the stored form is inclusive (each max corner is one less than the
        # exclusive input) in front top-left, top-right, bottom-right, bottom-left order followed by
        # the same four back vertices; get_boxes_shape returns (depths, heights, widths); every
        # export mode derives from that stored form, and the mode string is lowercased.
        xyzxyz = self._asymmetric_xyzxyz(device, dtype)
        boxes = Boxes3D.from_tensor(xyzxyz, mode="XYZXYZ")
        assert boxes.mode == "xyzxyz"
        expected_data = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 2.0, 3.0],
                    [4.0, 4.0, 3.0],
                    [1.0, 4.0, 3.0],
                    [1.0, 2.0, 7.0],
                    [4.0, 2.0, 7.0],
                    [4.0, 4.0, 7.0],
                    [1.0, 4.0, 7.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(boxes.data, expected_data, atol=0.0, rtol=0.0)
        for extent, expected in zip(boxes.get_boxes_shape(), (5.0, 3.0, 4.0)):
            self.assert_close(extent, torch.tensor([expected], device=device, dtype=dtype), atol=0.0, rtol=0.0)

        self.assert_close(boxes.to_tensor("xyzxyz"), xyzxyz, atol=0.0, rtol=0.0)
        expected_plus = torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 7.0]], device=device, dtype=dtype)
        self.assert_close(boxes.to_tensor("xyzxyz_plus"), expected_plus, atol=0.0, rtol=0.0)
        expected_whd = torch.tensor([[1.0, 2.0, 3.0, 4.0, 3.0, 5.0]], device=device, dtype=dtype)
        self.assert_close(boxes.to_tensor("xyzwhd"), expected_whd, atol=0.0, rtol=0.0)
        self.assert_close(boxes.to_tensor("vertices_plus"), expected_data, atol=0.0, rtol=0.0)
        offsets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device)
        offsets = torch.cat([offsets, offsets + torch.tensor([0.0, 0.0, 1.0], device=device)]).to(dtype)
        self.assert_close(boxes.to_tensor("vertices"), expected_data + offsets, atol=0.0, rtol=0.0)
        with pytest.raises(ValueError, match="shape"):
            Boxes3D.from_tensor(expected_data, mode="vertices")

    def test_wart_vertices_export_is_exclusive_for_inclusive_bbox3d_consumers_4009(self, device, dtype):
        # Wart pin for kornia#4009 (its 3D form): 'vertices' is an exclusive export, while
        # infer_bbox_shape3d reads vertices as inclusive and therefore adds one per axis.
        boxes = Boxes3D.from_tensor(self._asymmetric_xyzxyz(device, dtype), mode="xyzxyz")
        for extent, expected in zip(infer_bbox_shape3d(boxes.to_tensor("vertices")), (6.0, 4.0, 5.0)):
            self.assert_close(extent, torch.tensor([expected], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        for extent, expected in zip(infer_bbox_shape3d(boxes.to_tensor("vertices_plus")), (5.0, 3.0, 4.0)):
            self.assert_close(extent, torch.tensor([expected], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_from_tensor_rejects_non_positive_extents_in_mode_convention(self, device, dtype):
        # Convention pin: validate_boxes=True rejects extents <= 0 measured in the given mode, so a
        # zero-span 'xyzxyz_plus' box (inclusive extent one) passes where the same corners in
        # 'xyzxyz' fail; each axis is checked; validate_boxes=False accepts everything.
        def build(values, mode, validate_boxes=True):
            return Boxes3D.from_tensor(torch.tensor([values], device=device, dtype=dtype), mode, validate_boxes)

        with pytest.raises(ValueError, match="widths"):
            build([1.0, 2.0, 3.0, 1.0, 6.0, 8.0], "xyzxyz")
        assert build([1.0, 2.0, 3.0, 1.0, 6.0, 8.0], "xyzxyz_plus").data.shape == (1, 8, 3)
        with pytest.raises(ValueError, match="widths"):
            build([1.0, 2.0, 3.0, 0.0, 6.0, 8.0], "xyzxyz_plus")
        with pytest.raises(ValueError, match="heights"):
            build([1.0, 2.0, 3.0, 5.0, 2.0, 8.0], "xyzxyz")
        with pytest.raises(ValueError, match="depths"):
            build([1.0, 2.0, 3.0, 5.0, 6.0, 3.0], "xyzxyz")
        with pytest.raises(ValueError, match="widths"):
            build([1.0, 2.0, 3.0, 0.0, 4.0, 5.0], "xyzwhd")
        assert build([1.0, 2.0, 3.0, 0.5, 4.0, 5.0], "xyzwhd").data.shape == (1, 8, 3)
        assert build([1.0, 2.0, 3.0, 1.0, 6.0, 8.0], "xyzxyz", validate_boxes=False).data.shape == (1, 8, 3)

    def test_wart_constructor_and_from_tensor_have_different_integer_policies_4012(self, device):
        # Wart pin for kornia#4012 (its 3D form): the constructor rejects integer coordinates
        # unless told to cast, while from_tensor silently casts them to float32.
        vertices = torch.tensor([[[1, 2, 3]] * 8], device=device)
        with pytest.raises(ValueError, match="floating point"):
            Boxes3D(vertices)
        assert Boxes3D(vertices, raise_if_not_floating_point=False).dtype == torch.float32
        integer = torch.tensor([[1, 2, 3, 5, 5, 8]], device=device)
        assert Boxes3D.from_tensor(integer, mode="xyzxyz").dtype == torch.float32
        assert Boxes3D.from_tensor(integer.to(torch.float16), mode="xyzxyz").dtype == torch.float16

    def test_wart_to_tensor_default_mode_ignores_the_stored_label(self, device, dtype):
        # Wart pin: Boxes3D.to_tensor() defaults to 'xyzxyz' whatever mode the object was built in,
        # while Boxes.to_tensor() defaults to the stored mode. The copy path of transform_boxes
        # also resets the label to 'xyzxyz_plus' where the in-place path keeps it.
        xyzwhd = torch.tensor([[1.0, 2.0, 3.0, 4.0, 3.0, 5.0]], device=device, dtype=dtype)
        boxes = Boxes3D.from_tensor(xyzwhd, mode="xyzwhd")
        assert boxes.mode == "xyzwhd"
        self.assert_close(boxes.to_tensor(), self._asymmetric_xyzxyz(device, dtype), atol=0.0, rtol=0.0)
        xywh = torch.tensor([[1.0, 2.0, 4.0, 3.0]], device=device, dtype=dtype)
        self.assert_close(Boxes.from_tensor(xywh, mode="xywh").to_tensor(), xywh, atol=0.0, rtol=0.0)

        identity = torch.eye(4, device=device, dtype=dtype)
        assert boxes.transform_boxes(identity).mode == "xyzxyz_plus"
        assert boxes.transform_boxes_(identity).mode == "xyzwhd"

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="Boxes3D.to_tensor() ignores the stored mode label")
    def test_convention_to_tensor_default_mode_is_the_stored_label(self, device, dtype):
        xyzwhd = torch.tensor([[1.0, 2.0, 3.0, 4.0, 3.0, 5.0]], device=device, dtype=dtype)
        self.assert_close(Boxes3D.from_tensor(xyzwhd, mode="xyzwhd").to_tensor(), xyzwhd, atol=0.0, rtol=0.0)

    @staticmethod
    def _fractional_cuboid(device, dtype) -> torch.Tensor:
        # x in [1.6, 3.4], y and z in [1.6, 2.4].
        return torch.tensor(
            [
                [
                    [1.6, 1.6, 1.6],
                    [3.4, 1.6, 1.6],
                    [3.4, 2.4, 1.6],
                    [1.6, 2.4, 1.6],
                    [1.6, 1.6, 2.4],
                    [3.4, 1.6, 2.4],
                    [3.4, 2.4, 2.4],
                    [1.6, 2.4, 2.4],
                ]
            ],
            device=device,
            dtype=dtype,
        )

    def test_convention_to_mask_takes_depth_height_width_and_matches_the_free_function(self, device, dtype):
        # Convention pin: to_mask(depth, height, width) returns (N, depth, height, width) in the box
        # dtype, equal to bbox_to_mask3d's channel on integer vertices; a box entirely outside the
        # volume fills nothing, and a box that requires grad is rejected.
        cuboid = Boxes3D.from_tensor(
            torch.tensor([[1.0, 1.0, 1.0, 3.0, 3.0, 2.0]], device=device, dtype=dtype), mode="xyzxyz_plus"
        )
        mask = cuboid.to_mask(4, 5, 6)
        assert mask.shape == (1, 4, 5, 6)
        assert mask.dtype == dtype
        self.assert_close(mask, bbox_to_mask3d(cuboid.data, (4, 5, 6))[:, 0].to(dtype), atol=0.0, rtol=0.0)
        assert mask.sum().item() == 18.0
        assert Boxes3D(cuboid.data[None]).to_mask(4, 5, 6).shape == (1, 1, 4, 5, 6)
        assert Boxes3D(cuboid.data + 10.0).to_mask(4, 5, 6).sum().item() == 0.0
        with pytest.raises(RuntimeError, match="differentiable"):
            Boxes3D(cuboid.data.clone().requires_grad_()).to_mask(4, 5, 6)

    def test_wart_to_mask_rounds_the_exclusive_export_half_open_4015(self, device, dtype):
        # Wart pin for kornia#4015 (3D): the fractional cuboid exports as xyzxyz
        # [1.6, 1.6, 1.6, 4.4, 3.4, 3.4], rounds to [2, 2, 2, 4, 3, 3], and fills the half-open
        # ranges: two voxels, where bbox_to_mask3d truncates and fills twelve (test_bbox.py).
        boxes = self._fractional_cuboid(device, dtype)
        expected = torch.zeros(1, 5, 5, 6, device=device, dtype=dtype)
        expected[0, 2, 2, 2:4] = 1.0
        self.assert_close(Boxes3D(boxes).to_mask(5, 5, 6), expected, atol=0.0, rtol=0.0)
        assert bbox_to_mask3d(boxes, (5, 5, 6)).sum().item() == 12.0

    @pytest.mark.parametrize("case", ["fractional", "outside", "negative", "batched"])
    def test_convention_to_mask_export_path_matches_loop_path(self, case, device, dtype, monkeypatch):
        # Convention pin: the loop path and the grid-comparison path taken under export produce
        # the same mask. The export branch is selected by patching the module's is_exporting predicate.
        fractional = self._fractional_cuboid(device, dtype)
        data = {
            "fractional": fractional,
            "outside": fractional + 10.0,
            "negative": fractional - 2.0,
            "batched": fractional.expand(2, 2, 8, 3).contiguous(),
        }[case]
        loop = Boxes3D(data).to_mask(5, 5, 6)
        monkeypatch.setattr(boxes_module, "is_exporting", lambda: True)
        vectorized = Boxes3D(data).to_mask(5, 5, 6)
        assert loop.shape == vectorized.shape
        self.assert_close(loop, vectorized, atol=0.0, rtol=0.0)

    def test_convention_transform_boxes_in_place_rebinds_data(self, device, dtype):
        # Convention pin: the copy path leaves the source untouched and returns a new object; the
        # in-place path returns self but rebinds the internal tensor, so an earlier data reference
        # is stale. A (3, 3) matrix is rejected and a (1, 4, 4) matrix applies to unbatched boxes.
        source = Boxes3D.from_tensor(self._asymmetric_xyzxyz(device, dtype), mode="xyzxyz")
        original = source.data
        scale = torch.diag(torch.tensor([2.0, 3.0, 4.0, 1.0], device=device, dtype=dtype))
        copied = source.transform_boxes(scale)
        assert copied is not source
        assert source.data is original
        expected = torch.tensor([[2.0, 6.0, 12.0, 8.0, 12.0, 28.0]], device=device, dtype=dtype)
        self.assert_close(copied.to_tensor("xyzxyz_plus"), expected, atol=0.0, rtol=0.0)
        self.assert_close(source.transform_boxes(scale[None]).data, copied.data, atol=0.0, rtol=0.0)

        result = source.transform_boxes_(scale)
        assert result is source
        assert source.data is not original
        self.assert_close(source.data, copied.data, atol=0.0, rtol=0.0)
        untouched = Boxes3D.from_tensor(self._asymmetric_xyzxyz(device, dtype)).data
        self.assert_close(original, untouched, atol=0.0, rtol=0.0)
        with pytest.raises(ValueError, match="4, 4"):
            source.transform_boxes(torch.eye(3, device=device, dtype=dtype))


class TestTransformBoxes3D(BaseTester):
    def test_transform_boxes(self, device, dtype):
        # Define boxes in XYZXYZ format with integer coordinates (TF32-safe on CUDA).
        boxes_xyzxyz = torch.tensor([[140, 104, 284, 398, 412, 454]], device=device, dtype=dtype)
        expected_boxes_xyzxyz = torch.tensor([[372, 104, 569, 116, 412, 908]], device=device, dtype=dtype)

        boxes = Boxes3D.from_tensor(boxes_xyzxyz)
        expected_boxes = Boxes3D.from_tensor(expected_boxes_xyzxyz, validate_boxes=False)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        transformed_boxes = boxes.transform_boxes(trans_mat)
        self.assert_close(transformed_boxes.data, expected_boxes.data, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is not boxes

    def test_transform_boxes_(self, device, dtype):
        # Define boxes in XYZXYZ format with integer coordinates (TF32-safe on CUDA).
        boxes_xyzxyz = torch.tensor([[140, 104, 284, 398, 412, 454]], device=device, dtype=dtype)
        expected_boxes_xyzxyz = torch.tensor([[372, 104, 569, 116, 412, 908]], device=device, dtype=dtype)

        boxes = Boxes3D.from_tensor(boxes_xyzxyz)
        expected_boxes = Boxes3D.from_tensor(expected_boxes_xyzxyz, validate_boxes=False)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        transformed_boxes = boxes.transform_boxes_(trans_mat)
        self.assert_close(transformed_boxes.data, expected_boxes.data, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is boxes

    def test_transform_multiple_boxes(self, device, dtype):
        # Define boxes in XYZXYZ format with integer coordinates (TF32-safe on CUDA).
        boxes_xyzxyz = torch.tensor(
            [
                [140, 104, 284, 398, 412, 454],
                [2, 81, 470, 512, 513, 513],
                [166, 263, 43, 512, 510, 786],
                [120, 145, 235, 258, 412, 387],
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1)  # 2 x 4 x 4 two images 4 boxes each

        expected_boxes_xyzxyz = torch.tensor(
            [
                [
                    [372, 104, 569, 116, 412, 908],
                    [510, 81, 941, 2, 513, 1026],
                    [346, 263, 87, 2, 510, 1572],
                    [392, 145, 471, 256, 412, 774],
                ],
                [
                    [140, 104, 284, 398, 412, 454],
                    [2, 81, 470, 512, 513, 513],
                    [166, 263, 43, 512, 510, 786],
                    [120, 145, 235, 258, 412, 387],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor(
            [
                [[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )

        boxes = Boxes3D.from_tensor(boxes_xyzxyz)
        expected_boxes = Boxes3D.from_tensor(expected_boxes_xyzxyz, validate_boxes=False)

        out = boxes.transform_boxes(trans_mat)
        self.assert_close(out.data, expected_boxes.data, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # Define boxes in XYZXYZ format for simplicity.
        boxes_xyzxyz = torch.tensor(
            [
                [139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185],
                [1.0240, 80.5547, 469.50, 512.0000, 512.0000, 512.0],
                [165.2053, 262.1440, 42.98, 510.6347, 508.9280, 784.443],
                [119.8080, 144.2067, 234.21, 257.0240, 410.1292, 386.14],
            ],
            device=device,
            dtype=torch.float64,
        )
        boxes = Boxes3D.from_tensor(boxes_xyzxyz)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=torch.float64,
        )

        def _wrapper_transform_boxes(hexahedrons, M):
            boxes = Boxes3D(hexahedrons)
            boxes = boxes.transform_boxes(M)
            return boxes.data

        self.gradcheck(_wrapper_transform_boxes, (boxes.data, trans_mat))


class TestVideoBoxes(BaseTester):
    """Public API and round-trip coverage for :class:`VideoBoxes` (#4016)."""

    @staticmethod
    def _sample_video_boxes(device, dtype, batch: int = 2, time: int = 3, n_boxes: int = 1) -> torch.Tensor:
        # Clockwise vertices_plus corners for a 3x3 box starting at (1, 1).
        frame = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]],
            device=device,
            dtype=dtype,
        )  # (1, 4, 2)
        return frame.view(1, 1, 1, 4, 2).expand(batch, time, n_boxes, 4, 2).contiguous().clone()

    def test_smoke(self, device, dtype):
        boxes = self._sample_video_boxes(device, dtype)
        video_boxes = VideoBoxes.from_tensor(boxes)
        assert isinstance(video_boxes, VideoBoxes)
        assert video_boxes.temporal_channel_size == boxes.size(1)

    def test_exception(self, device, dtype):
        frame = self._sample_video_boxes(device, dtype, batch=1, time=1)[0]  # (T, N, 4, 2)
        with pytest.raises(ValueError):
            VideoBoxes.from_tensor(frame)
        with pytest.raises(ValueError):
            VideoBoxes.from_tensor([self._sample_video_boxes(device, dtype)])

    def test_cardinality(self, device, dtype):
        boxes = self._sample_video_boxes(device, dtype, batch=2, time=3, n_boxes=2)
        video_boxes = VideoBoxes.from_tensor(boxes)
        assert video_boxes.data.shape == (boxes.size(0) * boxes.size(1), boxes.size(2), 4, 2)
        out = video_boxes.to_tensor()
        assert isinstance(out, torch.Tensor)
        assert out.shape == boxes.shape

    def test_roundtrip_and_clone(self, device, dtype):
        boxes = self._sample_video_boxes(device, dtype)
        video_boxes = VideoBoxes.from_tensor(boxes)
        restored = video_boxes.to_tensor()
        assert isinstance(restored, torch.Tensor)
        self.assert_close(restored, boxes)

        cloned = video_boxes.clone()
        assert isinstance(cloned, VideoBoxes)
        assert cloned is not video_boxes
        assert cloned.temporal_channel_size == video_boxes.temporal_channel_size
        self.assert_close(cloned.data, video_boxes.data)
        cloned.data[0, 0, 0, 0] = -1
        assert not torch.equal(cloned.data, video_boxes.data)

    def test_public_api_surface(self):
        # Pin #4016: VideoBoxes is exported and the overrides are documented.
        assert "VideoBoxes" in boxes_module.__all__
        assert VideoBoxes.__doc__ is not None
        for name in ("from_tensor", "to_tensor", "clone"):
            assert getattr(VideoBoxes, name).__doc__ is not None

    def test_gradcheck(self, device):
        boxes = self._sample_video_boxes(device, torch.float64)

        def _wrap(x: torch.Tensor) -> torch.Tensor:
            return VideoBoxes.from_tensor(x).to_tensor()  # type: ignore[return-value]

        self.gradcheck(_wrap, (boxes,))

    def test_convention_from_tensor_stores_vertices_plus_and_restores_the_temporal_axis(self, device, dtype):
        # Convention pin: the (B, T, N, 4, 2) input is stored unchanged as (B * T, N, 4, 2) batched
        # 'vertices_plus' data; every Boxes export mode is available and comes back with the
        # temporal axis restored; integer input is cast to float32; a transformation matrix must
        # carry the flattened batch of B * T matrices.
        boxes = self._sample_video_boxes(device, dtype, batch=2, time=3, n_boxes=1)
        video_boxes = VideoBoxes.from_tensor(boxes)
        assert video_boxes.mode == "vertices_plus"
        assert video_boxes.temporal_channel_size == 3
        self.assert_close(video_boxes.data, boxes.reshape(6, 1, 4, 2), atol=0.0, rtol=0.0)

        xyxy = video_boxes.to_tensor("xyxy")
        assert isinstance(xyxy, torch.Tensor)
        assert xyxy.shape == (2, 3, 1, 4)
        expected_xyxy = torch.tensor([1.0, 1.0, 4.0, 4.0], device=device, dtype=dtype).expand(2, 3, 1, 4)
        self.assert_close(xyxy, expected_xyxy, atol=0.0, rtol=0.0)
        assert VideoBoxes.from_tensor(boxes.to(torch.int64)).dtype == torch.float32
        with pytest.raises(ValueError, match="BxTxNx4x2"):
            VideoBoxes.from_tensor(torch.zeros(2, 3, 1, 4, 3, device=device, dtype=dtype))

        with pytest.raises(ValueError, match="Batch size mismatch"):
            video_boxes.transform_boxes(torch.eye(3, device=device, dtype=dtype))
        transformed = video_boxes.transform_boxes(torch.eye(3, device=device, dtype=dtype).expand(6, 3, 3))
        assert isinstance(transformed, VideoBoxes)
        assert transformed.temporal_channel_size == 3
        self.assert_close(transformed.to_tensor(), boxes, atol=0.0, rtol=0.0)

    def test_wart_inherited_methods_break_on_the_temporal_wrapper(self, device, dtype):
        # Wart pin: the to_tensor override drops the as_padded_sequence keyword that the inherited
        # get_boxes_shape and to_mask pass, and indexing builds a wrapper without the temporal size,
        # so its to_tensor fails. Methods that copy through clone keep the temporal size.
        video_boxes = VideoBoxes.from_tensor(self._sample_video_boxes(device, dtype, batch=2, time=3, n_boxes=1))
        with pytest.raises(TypeError, match="as_padded_sequence"):
            video_boxes.get_boxes_shape()
        with pytest.raises(TypeError, match="as_padded_sequence"):
            video_boxes.to_mask(4, 5)
        frame = video_boxes[0]
        assert isinstance(frame, VideoBoxes)
        with pytest.raises(AttributeError, match="temporal_channel_size"):
            frame.to_tensor()
        bounds = torch.zeros(6, 2, device=device, dtype=dtype)
        assert video_boxes.clamp(bounds, bounds + 2.0).temporal_channel_size == 3
        assert video_boxes.filter_boxes_by_area(1.0).temporal_channel_size == 3
        assert video_boxes.translate(bounds + 1.0).temporal_channel_size == 3
        assert video_boxes.pad(torch.ones(6, 4, device=device, dtype=dtype)).temporal_channel_size == 3
        assert video_boxes.merge(video_boxes).temporal_channel_size == 3
        assert video_boxes.to(dtype=torch.float32).temporal_channel_size == 3

    @pytest.mark.xfail(
        strict=True, raises=AssertionError, reason="VideoBoxes.get_boxes_shape and to_mask raise TypeError"
    )
    def test_convention_inherited_shape_and_mask_work_on_the_temporal_wrapper(self, device, dtype):
        video_boxes = VideoBoxes.from_tensor(self._sample_video_boxes(device, dtype, batch=2, time=3, n_boxes=1))
        try:
            heights, widths = video_boxes.get_boxes_shape()
            mask = video_boxes.to_mask(4, 5)
        except TypeError as error:
            raise AssertionError(str(error)) from error
        # One extent per box and one (4, 5) mask per box, whichever way the temporal axis is laid out.
        assert heights.numel() == widths.numel() == 6
        assert mask.shape[-2:] == (4, 5) and mask.numel() == 6 * 4 * 5
