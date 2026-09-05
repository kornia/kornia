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
from kornia.geometry.bbox import infer_bbox_shape
from kornia.geometry.boxes import Boxes, Boxes3D, VideoBoxes

from testing.base import BaseTester


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
