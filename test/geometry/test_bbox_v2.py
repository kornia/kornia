from typing import Callable, Tuple

import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.testing as utils
from kornia.geometry.bbox_v2 import (
    bbox3d_to_kornia_bbox3d,
    bbox3d_to_mask3d,
    bbox_to_kornia_bbox,
    bbox_to_mask,
    infer_bbox3d_shape,
    infer_bbox_shape,
    kornia_bbox3d_to_bbox3d,
    kornia_bbox_to_bbox,
    transform_bbox,
    transform_bbox3d,
    validate_bbox,
    validate_bbox3d,
)


class TestBbox2D:
    def test_smoke(self, device, dtype):
        def _create_bbox():
            # Sample two points of the rectangle
            points = torch.rand(1, 4, device=device, dtype=dtype)

            # Fill according missing points
            bbox = torch.zeros(1, 4, 2, device=device, dtype=dtype)
            bbox[0, 0] = points[0][:2]
            bbox[0, 1, 0] = points[0][2]
            bbox[0, 1, 1] = points[0][1]
            bbox[0, 2] = points[0][2:]
            bbox[0, 3, 0] = points[0][0]
            bbox[0, 3, 1] = points[0][3]
            return bbox

        # Validate
        assert validate_bbox(_create_bbox())  # Validate 1 box
        two_boxes = torch.cat([_create_bbox(), _create_bbox()])  # 2 boxes without batching (N, 4, 2) where N=2
        assert validate_bbox(two_boxes)
        batched_bbox = torch.stack([_create_bbox(), _create_bbox()])  # 2 boxes in batch (B, 1, 4, 2) where B=2
        assert validate_bbox(batched_bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        box = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [1.0, 2.0], [3.0, 1.0]]], device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [1.0, 2.0], [3.0, 2.0]], [[5.0, 4.0], [2.0, 2.0], [5.0, 2.0], [2.0, 4.0]]],
            device=device,
            dtype=dtype,
        )  # (2, 4, 2)
        boxes_batch = boxes[None]  # (1, 2, 4, 2)

        # Single box
        h, w = infer_bbox_shape(box)
        assert (h.item(), w.item()) == (1, 2)

        # Boxes
        h, w = infer_bbox_shape(boxes)
        assert h.ndim == 1 and w.ndim == 1
        assert len(h) == 2 and len(w) == 2
        assert (h == torch.as_tensor([1.0, 2.0])).all() and (w == torch.as_tensor([2.0, 3.0])).all()

        # Box batch
        h, w = infer_bbox_shape(boxes_batch)
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (1, 2) and w.shape == (1, 2)
        assert (h == torch.as_tensor([[1.0, 2.0]])).all() and (w == torch.as_tensor([[2.0, 3.0]])).all()

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        box1 = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [3.0, 1.0], [1.0, 2.0]]], device=device, dtype=dtype)
        box2 = torch.tensor([[[5.0, 2.0], [2.0, 2.0], [5.0, 4.0], [2.0, 4.0]]], device=device, dtype=dtype)
        batched_boxes = torch.stack([box1, box2])

        h, w = infer_bbox_shape(batched_boxes)
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (2, 1) and w.shape == (2, 1)
        assert (h == torch.as_tensor([[1], [2]])).all() and (w == torch.as_tensor([[2], [3]])).all()

    @pytest.mark.parametrize('shape', [(1, 4), (1, 1, 4)])
    def test_bounding_boxes_convert_to_kornia(self, shape: Tuple[int], device, dtype):
        box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        box_xyxy_plus_1 = torch.as_tensor([[1, 2, 4, 5]], device=device, dtype=dtype).view(*shape)
        box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        box_xywh_plus_1 = torch.as_tensor([[1, 2, 3, 3]], device=device, dtype=dtype).view(*shape)

        expected_box = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(*shape, 2)

        kornia_xyxy = bbox_to_kornia_bbox(box_xyxy, mode='xyxy')
        kornia_xyxy_plus_1 = bbox_to_kornia_bbox(box_xyxy_plus_1, mode='xyxy_plus_1')
        kornia_xywh = bbox_to_kornia_bbox(box_xywh, mode='xywh')
        kornia_xywh_plus_1 = bbox_to_kornia_bbox(box_xywh_plus_1, mode='xywh_plus_1')

        assert kornia_xyxy.shape == expected_box.shape
        assert_allclose(kornia_xyxy, expected_box)

        assert kornia_xyxy_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xyxy_plus_1, expected_box)

        assert kornia_xywh.shape == expected_box.shape
        assert_allclose(kornia_xywh, expected_box)

        assert kornia_xywh_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xywh_plus_1, expected_box)

    @pytest.mark.parametrize('shape', [(1, 4), (1, 1, 4)])
    def test_bounding_boxes_convert_from_kornia(self, shape: Tuple[int], device, dtype):
        # Quadrilater with randommized vertices to reflect possible transforms.
        box = torch.as_tensor([[[3, 2], [3, 4], [1, 4], [1, 2]]], device=device, dtype=dtype).view(*shape, 2)

        expected_box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        expected_box_xyxy_plus_1 = torch.as_tensor([[1, 2, 4, 5]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh_plus_1 = torch.as_tensor([[1, 2, 3, 3]], device=device, dtype=dtype).view(*shape)
        expected_vertices = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(
            *shape, 2
        )

        kornia_xyxy = kornia_bbox_to_bbox(box, mode='xyxy')
        kornia_xyxy_plus_1 = kornia_bbox_to_bbox(box, mode='xyxy_plus_1')
        kornia_xywh = kornia_bbox_to_bbox(box, mode='xywh')
        kornia_xywh_plus_1 = kornia_bbox_to_bbox(box, mode='xywh_plus_1')
        kornia_vertices = kornia_bbox_to_bbox(box, mode='vertices')

        assert kornia_xyxy.shape == expected_box_xyxy.shape
        assert_allclose(kornia_xyxy, expected_box_xyxy)

        assert kornia_xyxy_plus_1.shape == expected_box_xyxy_plus_1.shape
        assert_allclose(kornia_xyxy_plus_1, expected_box_xyxy_plus_1)

        assert kornia_xywh.shape == expected_box_xywh.shape
        assert_allclose(kornia_xywh, expected_box_xywh)

        assert kornia_xywh_plus_1.shape == expected_box_xywh_plus_1.shape
        assert_allclose(kornia_xywh_plus_1, expected_box_xywh_plus_1)

        assert kornia_vertices.shape == expected_vertices.shape
        assert_allclose(kornia_vertices, expected_vertices)

    def test_bbox_to_mask(self, device, dtype):
        box1 = torch.tensor([[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)  # (1, 4, 2)
        box2 = torch.tensor([[[2.0, 2.0], [5.0, 2.0], [5.0, 6.0], [5.0, 2.0]]], device=device, dtype=dtype)  # (1, 4, 2)
        two_boxes = torch.cat([box1, box2])  # (2, 4, 2)
        batched_boxes = torch.stack([box1, box2])  # (2, 1, 4, 2)

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

        mask1 = bbox_to_mask(box1, height, width)
        mask2 = bbox_to_mask(box2, height, width)
        two_masks = bbox_to_mask(two_boxes, height, width)
        batched_masks = bbox_to_mask(batched_boxes, height, width)

        assert mask1.shape == expected_mask1.shape
        assert_allclose(mask1, expected_mask1)

        assert mask2.shape == expected_mask2.shape
        assert_allclose(mask2, expected_mask2)

        assert two_masks.shape == expected_two_masks.shape
        assert_allclose(two_masks, expected_two_masks)

        assert batched_masks.shape == expected_batched_masks.shape
        assert_allclose(batched_masks, expected_batched_masks)

    def test_gradcheck(self, device, dtype):
        boxes1 = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)

        boxes1 = utils.tensor_to_gradcheck_var(boxes1)
        boxes2 = utils.tensor_to_gradcheck_var(boxes1.detach().clone())
        boxes_xyxy = torch.tensor([[1.0, 3.0, 5.0, 6.0]])

        assert gradcheck(infer_bbox_shape, (boxes1,), raise_exception=True)
        assert gradcheck(kornia_bbox_to_bbox, (boxes2,), raise_exception=True)
        assert gradcheck(bbox_to_kornia_bbox, (boxes_xyxy,), raise_exception=True)

    @pytest.mark.parametrize('op', [validate_bbox, infer_bbox_shape, kornia_bbox_to_bbox])
    def test_jit(self, op: Callable, device, dtype):
        # Define script
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)

    def test_jit_bbox_to_mask(self, device, dtype):
        # Define script
        op = bbox_to_mask
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        height, width = 5, 10
        # Run
        expected = op(boxes, height, width)
        actual = op_script(boxes, height, width)
        # Compare
        assert_allclose(actual, expected)

    def test_jit_convert_to_kornia_format(self, device, dtype):
        # Define script
        op = bbox_to_kornia_bbox
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[1, 2, 3, 4]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)


class TestTransformBoxes2D:
    def test_transform_boxes(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)
        expected_boxes_xyxy = torch.tensor([[372.7360, 103.0150, 114.6880, 410.5225]], device=device, dtype=dtype)

        boxes = bbox_to_kornia_bbox(boxes_xyxy)
        expected_boxes = bbox_to_kornia_bbox(expected_boxes_xyxy)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        transformed_boxes = transform_bbox(boxes, trans_mat)
        assert_allclose(transformed_boxes, expected_boxes, atol=1e-4, rtol=1e-4)

    def test_transform_multiple_boxes(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor(
            [
                [139.2640, 103.0150, 397.3120, 410.5225],
                [1.0240, 80.5547, 512.0000, 512.0000],
                [165.2053, 262.1440, 510.6347, 508.9280],
                [119.8080, 144.2067, 257.0240, 410.1292],
            ],
            device=device,
            dtype=dtype,
        ).repeat(
            2, 1, 1
        )  # 2 x 4 x 4 two images 4 boxes each

        expected_boxes_xyxy = torch.tensor(
            [
                [
                    [372.7360, 103.0150, 114.6880, 410.5225],
                    [510.9760, 80.5547, 0.0000, 512.0000],
                    [346.7947, 262.1440, 1.3653, 508.9280],
                    [392.1920, 144.2067, 254.9760, 410.1292],
                ],
                [
                    [139.2640, 103.0150, 397.3120, 410.5225],
                    [1.0240, 80.5547, 512.0000, 512.0000],
                    [165.2053, 262.1440, 510.6347, 508.9280],
                    [119.8080, 144.2067, 257.0240, 410.1292],
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

        boxes = bbox_to_kornia_bbox(boxes_xyxy)
        expected_boxes = bbox_to_kornia_bbox(expected_boxes_xyxy)

        out = transform_bbox(boxes, trans_mat)
        assert_allclose(out, expected_boxes, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor(
            [
                [139.2640, 103.0150, 258.0480, 307.5075],
                [1.0240, 80.5547, 510.9760, 431.4453],
                [165.2053, 262.1440, 345.4293, 246.7840],
                [119.8080, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=dtype,
        )
        boxes = bbox_to_kornia_bbox(boxes_xyxy)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        boxes = utils.tensor_to_gradcheck_var(boxes)

        assert gradcheck(transform_bbox, (boxes, trans_mat), raise_exception=True)

    def test_jit(self, device, dtype):
        boxes_xyxy = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075]], device=device, dtype=dtype)
        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        args = (bbox_to_kornia_bbox(boxes_xyxy), trans_mat)
        op = transform_bbox
        op_jit = torch.jit.script(op)
        assert_allclose(op(*args), op_jit(*args))


class TestBbox3D:
    def test_smoke(self, device, dtype):
        def _create_bbox():
            # Sample two points of the 3d rect
            points = torch.rand(1, 6, device=device, dtype=dtype)

            # Fill acording missing points
            bbox = torch.zeros(1, 8, 3, device=device, dtype=dtype)
            bbox[0, 0] = points[0][:3]
            bbox[0, 1, 0] = points[0][3]
            bbox[0, 1, 1] = points[0][1]
            bbox[0, 1, 2] = points[0][2]
            bbox[0, 2, 0] = points[0][3]
            bbox[0, 2, 1] = points[0][4]
            bbox[0, 2, 2] = points[0][2]
            bbox[0, 3, 0] = points[0][0]
            bbox[0, 3, 1] = points[0][4]
            bbox[0, 3, 2] = points[0][2]
            bbox[0, 4, 0] = points[0][0]
            bbox[0, 4, 1] = points[0][1]
            bbox[0, 4, 2] = points[0][5]
            bbox[0, 5, 0] = points[0][3]
            bbox[0, 5, 1] = points[0][1]
            bbox[0, 5, 2] = points[0][5]
            bbox[0, 6] = points[0][3:]
            bbox[0, 7, 0] = points[0][0]
            bbox[0, 7, 1] = points[0][4]
            bbox[0, 7, 2] = points[0][5]
            return bbox

        # Validate
        assert validate_bbox3d(_create_bbox())  # Validate 1 box
        two_boxes = torch.cat([_create_bbox(), _create_bbox()])  # 2 boxes without batching (N, 8, 3) where N=2
        assert validate_bbox3d(two_boxes)
        batched_bbox = torch.stack([_create_bbox(), _create_bbox()])  # 2 boxes in batch (B, 1, 8, 3) where B=2
        assert validate_bbox3d(batched_bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        box = torch.tensor(
            [[[0, 1, 2], [0, 1, 32], [10, 21, 2], [0, 21, 2], [10, 1, 32], [10, 21, 32], [10, 1, 2], [0, 21, 32]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        boxes = torch.tensor(
            [
                [[0, 21, 32], [0, 1, 2], [10, 1, 2], [0, 21, 2], [0, 1, 32], [10, 21, 2], [10, 1, 32], [10, 21, 32]],
                [[3, 4, 5], [3, 4, 65], [43, 54, 5], [3, 54, 5], [43, 4, 5], [43, 4, 65], [43, 54, 65], [3, 54, 65]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x8x3
        boxes_batch = boxes[None]  # (1, 2, 8, 3)

        # Single box
        d, h, w = infer_bbox3d_shape(box)
        assert (d.item(), h.item(), w.item()) == (30.0, 20.0, 10.0)

        # Boxes
        d, h, w = infer_bbox3d_shape(boxes)
        assert h.ndim == 1 and w.ndim == 1
        assert len(d) == 2 and len(h) == 2 and len(w) == 2
        assert (
            (d == torch.as_tensor([30.0, 60.0])).all()
            and (h == torch.as_tensor([20.0, 50.0])).all()
            and (w == torch.as_tensor([10.0, 40.0])).all()
        )

        # Box batch
        d, h, w = infer_bbox3d_shape(boxes_batch)
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (1, 2) and w.shape == (1, 2)
        assert (
            (d == torch.as_tensor([[30.0, 60.0]])).all()
            and (h == torch.as_tensor([[20.0, 50.0]])).all()
            and (w == torch.as_tensor([[10.0, 40.0]])).all()
        )

    @pytest.mark.parametrize('shape', [(1, 6), (1, 1, 6)])
    def test_bounding_boxes_convert_to_kornia(self, shape: Tuple[int], device, dtype):
        box_xyzxyz = torch.as_tensor([[1, 2, 3, 4, 5, 6]], device=device, dtype=dtype).view(*shape)
        box_xyzxyz_plus_1 = torch.as_tensor([[1, 2, 3, 5, 6, 7]], device=device, dtype=dtype).view(*shape)
        box_xyzwhd = torch.as_tensor([[1, 2, 3, 3, 3, 3]], device=device, dtype=dtype).view(*shape)
        box_xyzwhd_plus_1 = torch.as_tensor([[1, 2, 3, 4, 4, 4]], device=device, dtype=dtype).view(*shape)

        expected_box = torch.as_tensor(
            [[[1, 2, 3], [4, 2, 3], [4, 5, 3], [1, 5, 3], [1, 2, 6], [4, 2, 6], [4, 5, 6], [1, 5, 6]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        kornia_xyzxyz = bbox3d_to_kornia_bbox3d(box_xyzxyz, mode='xyzxyz')
        kornia_xyzxyz_plus_1 = bbox3d_to_kornia_bbox3d(box_xyzxyz_plus_1, mode='xyzxyz_plus_1')
        kornia_xyzwhd = bbox3d_to_kornia_bbox3d(box_xyzwhd, mode='xyzwhd')
        kornia_xyzwhd_plus_1 = bbox3d_to_kornia_bbox3d(box_xyzwhd_plus_1, mode='xyzwhd_plus_1')

        assert kornia_xyzxyz.shape == expected_box.shape
        assert_allclose(kornia_xyzxyz, expected_box)

        assert kornia_xyzxyz_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xyzxyz_plus_1, expected_box)

        assert kornia_xyzwhd.shape == expected_box.shape
        assert_allclose(kornia_xyzwhd, expected_box)

        assert kornia_xyzwhd_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xyzwhd_plus_1, expected_box)

    @pytest.mark.parametrize('shape', [(1, 6), (1, 1, 6)])
    def test_bounding_boxes_convert_from_kornia(self, shape: Tuple[int], device, dtype):
        # Hexahedron with randommized vertices to reflect possible transforms.
        box = torch.as_tensor(
            [[[3, 2, 1], [1, 2, 1], [3, 4, 3], [1, 4, 3], [3, 2, 3], [1, 4, 1], [3, 4, 1], [1, 2, 3]]],
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        expected_box_xyzxyz = torch.as_tensor([[1, 2, 1, 3, 4, 3]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzxyz_plus_1 = torch.as_tensor([[1, 2, 1, 4, 5, 4]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzwhd = torch.as_tensor([[1, 2, 1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzwhd_plus_1 = torch.as_tensor([[1, 2, 1, 3, 3, 3]], device=device, dtype=dtype).view(*shape)
        expected_vertices = torch.as_tensor(
            [[[1, 2, 1], [3, 2, 1], [3, 4, 1], [1, 4, 1], [1, 2, 3], [3, 2, 3], [3, 4, 3], [1, 4, 3]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        kornia_xyzxyz = kornia_bbox3d_to_bbox3d(box, mode='xyzxyz')
        kornia_xyzxyz_plus_1 = kornia_bbox3d_to_bbox3d(box, mode='xyzxyz_plus_1')
        kornia_xyzwhd = kornia_bbox3d_to_bbox3d(box, mode='xyzwhd')
        kornia_xyzwhd_plus_1 = kornia_bbox3d_to_bbox3d(box, mode='xyzwhd_plus_1')
        kornia_vertices = kornia_bbox3d_to_bbox3d(box, mode='vertices')

        assert kornia_xyzxyz.shape == expected_box_xyzxyz.shape
        assert_allclose(kornia_xyzxyz, expected_box_xyzxyz)

        assert kornia_xyzxyz_plus_1.shape == expected_box_xyzxyz_plus_1.shape
        assert_allclose(kornia_xyzxyz_plus_1, expected_box_xyzxyz_plus_1)

        assert kornia_xyzwhd.shape == expected_box_xyzwhd.shape
        assert_allclose(kornia_xyzwhd, expected_box_xyzwhd)

        assert kornia_xyzwhd_plus_1.shape == expected_box_xyzwhd_plus_1.shape
        assert_allclose(kornia_xyzwhd_plus_1, expected_box_xyzwhd_plus_1)

        assert kornia_vertices.shape == expected_vertices.shape
        assert_allclose(kornia_vertices, expected_vertices)

    def test_bbox_to_mask(self, device, dtype):
        box1 = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [4.0, 1.0, 1.0],
                    [4.0, 3.0, 1.0],
                    [1.0, 3.0, 1.0],  # Front
                    [1.0, 1.0, 3.0],
                    [4.0, 1.0, 3.0],
                    [4.0, 3.0, 3.0],
                    [1.0, 3.0, 3.0],  # Back
                ]
            ],
            device=device,
            dtype=dtype,
        )  # (1, 4, 2)
        box2 = torch.tensor(
            [
                [
                    [2.0, 2.0, 1.0],
                    [5.0, 2.0, 1.0],
                    [5.0, 6.0, 1.0],
                    [5.0, 2.0, 1.0],  # Front
                    [2.0, 2.0, 2.0],
                    [5.0, 2.0, 2.0],
                    [5.0, 6.0, 2.0],
                    [5.0, 2.0, 2.0],  # Back
                ]
            ],
            device=device,
            dtype=dtype,
        )  # (1, 4, 2)
        two_boxes = torch.cat([box1, box2])  # (2, 4, 2)
        batched_boxes = torch.stack([box1, box2])  # (2, 1, 4, 2)

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

        mask1 = bbox3d_to_mask3d(box1, depth, height, width)
        mask2 = bbox3d_to_mask3d(box2, depth, height, width)
        two_masks = bbox3d_to_mask3d(two_boxes, depth, height, width)
        batched_masks = bbox3d_to_mask3d(batched_boxes, depth, height, width)

        assert mask1.shape == expected_mask1.shape
        assert_allclose(mask1, expected_mask1)

        assert mask2.shape == expected_mask2.shape
        assert_allclose(mask2, expected_mask2)

        assert two_masks.shape == expected_two_masks.shape
        assert_allclose(two_masks, expected_two_masks)

        assert batched_masks.shape == expected_batched_masks.shape
        assert_allclose(batched_masks, expected_batched_masks)

    def test_gradcheck(self, device, dtype):
        boxes1 = torch.tensor(
            [[[0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2], [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]]],
            device=device,
            dtype=dtype,
        )

        boxes1 = utils.tensor_to_gradcheck_var(boxes1)
        boxes2 = utils.tensor_to_gradcheck_var(boxes1.detach().clone())
        boxes_xyzxyz = torch.tensor([[1.0, 3.0, 8.0, 5.0, 6.0, 12.0]])

        assert gradcheck(infer_bbox3d_shape, (boxes1,), raise_exception=True)
        assert gradcheck(kornia_bbox3d_to_bbox3d, (boxes2,), raise_exception=True)
        assert gradcheck(bbox3d_to_kornia_bbox3d, (boxes_xyzxyz,), raise_exception=True)

    @pytest.mark.parametrize('op', [validate_bbox3d, infer_bbox3d_shape, kornia_bbox3d_to_bbox3d])
    def test_jit(self, op: Callable, device, dtype):
        # Define script
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)

    def test_jit_bbox3d_to_mask3d(self, device, dtype):
        # Define script
        op = bbox3d_to_mask3d
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3
        depth, height, width = 7, 5, 10
        # Run
        expected = op(boxes, depth, height, width)
        actual = op_script(boxes, depth, height, width)
        # Compare
        assert_allclose(actual, expected)

    def test_jit_convert_to_kornia_format(self, device, dtype):
        # Define script
        op = bbox3d_to_kornia_bbox3d
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_allclose(actual, expected)


class TestTransformBoxes3D:
    def test_transform_boxes(self, device, dtype):
        # Define boxes in XYZXYZ format for simplicity.
        boxes_xyzxyz = torch.tensor(
            [[139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185]], device=device, dtype=dtype
        )
        expected_boxes_xyzxyz = torch.tensor(
            [[372.7360, 103.0150, 567.324, 114.6880, 410.5225, 907.37]], device=device, dtype=dtype
        )

        boxes = bbox3d_to_kornia_bbox3d(boxes_xyzxyz)
        expected_boxes = bbox3d_to_kornia_bbox3d(expected_boxes_xyzxyz)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        transformed_boxes = transform_bbox3d(boxes, trans_mat)
        assert_allclose(transformed_boxes, expected_boxes, atol=1e-4, rtol=1e-4)

    def test_transform_multiple_boxes(self, device, dtype):
        # Define boxes in XYZXYZ format for simplicity.
        boxes_xyzxyz = torch.tensor(
            [
                [139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185],
                [1.0240, 80.5547, 469.50, 512.0000, 512.0000, 512.0],
                [165.2053, 262.1440, 42.98, 510.6347, 508.9280, 784.443],
                [119.8080, 144.2067, 234.21, 257.0240, 410.1292, 86.14],
            ],
            device=device,
            dtype=dtype,
        ).repeat(
            2, 1, 1
        )  # 2 x 4 x 4 two images 4 boxes each

        expected_boxes_xyzxyz = torch.tensor(
            [
                [
                    [372.7360, 103.0150, 567.324, 114.6880, 410.5225, 907.37],
                    [510.9760, 80.5547, 940.0, 0.0000, 512.0000, 1025.0],
                    [346.7947, 262.1440, 86.96, 1.3653, 508.9280, 1569.886],
                    [392.1920, 144.2067, 469.42, 254.9760, 410.1292, 173.28],
                ],
                [
                    [139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185],
                    [1.0240, 80.5547, 469.50, 512.0000, 512.0000, 512.0],
                    [165.2053, 262.1440, 42.98, 510.6347, 508.9280, 784.443],
                    [119.8080, 144.2067, 234.21, 257.0240, 410.1292, 86.14],
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

        boxes = bbox3d_to_kornia_bbox3d(boxes_xyzxyz)
        expected_boxes = bbox3d_to_kornia_bbox3d(expected_boxes_xyzxyz)

        out = transform_bbox3d(boxes, trans_mat)
        assert_allclose(out, expected_boxes, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # Define boxes in XYZXYZ format for simplicity.
        boxes_xyzxyz = torch.tensor(
            [
                [139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185],
                [1.0240, 80.5547, 469.50, 512.0000, 512.0000, 512.0],
                [165.2053, 262.1440, 42.98, 510.6347, 508.9280, 784.443],
                [119.8080, 144.2067, 234.21, 257.0240, 410.1292, 86.14],
            ],
            device=device,
            dtype=dtype,
        )
        boxes = bbox3d_to_kornia_bbox3d(boxes_xyzxyz)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        boxes = utils.tensor_to_gradcheck_var(boxes)

        assert gradcheck(transform_bbox3d, (boxes, trans_mat), raise_exception=True)

    def test_jit(self, device, dtype):
        boxes_xyzxyz = torch.tensor(
            [[[139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185]]], device=device, dtype=dtype
        )

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        args = (bbox3d_to_kornia_bbox3d(boxes_xyzxyz), trans_mat)
        op = transform_bbox3d
        op_jit = torch.jit.script(op)
        assert_allclose(op(*args), op_jit(*args))
