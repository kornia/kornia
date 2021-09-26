from functools import partial
from typing import Tuple

import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.testing as utils
from kornia.geometry.bbox_v2 import Boxes, Boxes3D


class TestBoxes2D:
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
        assert (h.item(), w.item()) == (1, 2)

        # Boxes
        h, w = boxes.get_boxes_shape()
        assert h.ndim == 1 and w.ndim == 1
        assert len(h) == 2 and len(w) == 2
        assert (h == torch.as_tensor([1.0, 2.0], device=device)).all() and (
            w == torch.as_tensor([2.0, 3.0], device=device)
        ).all()

        # Box batch
        h, w = boxes_batch.get_boxes_shape()
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (1, 2) and w.shape == (1, 2)
        assert (h == torch.as_tensor([[1.0, 2.0]], device=device)).all() and (
            w == torch.as_tensor([[2.0, 3.0]], device=device)
        ).all()

    def test_get_boxes_shape_batch(self, device, dtype):
        t_box1 = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [3.0, 1.0], [1.0, 2.0]]], device=device, dtype=dtype)
        t_box2 = torch.tensor([[[5.0, 2.0], [2.0, 2.0], [5.0, 4.0], [2.0, 4.0]]], device=device, dtype=dtype)
        batched_boxes = Boxes(torch.stack([t_box1, t_box2]))

        h, w = batched_boxes.get_boxes_shape()
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (2, 1) and w.shape == (2, 1)
        assert (h == torch.as_tensor([[1], [2]], device=device)).all() and (
            w == torch.as_tensor([[2], [3]], device=device)
        ).all()

    @pytest.mark.parametrize('shape', [(1, 4), (1, 1, 4)])
    def test_from_tensor(self, shape: Tuple[int], device, dtype):
        box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        box_xyxy_plus_1 = torch.as_tensor([[1, 2, 4, 5]], device=device, dtype=dtype).view(*shape)
        box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        box_xywh_plus_1 = torch.as_tensor([[1, 2, 3, 3]], device=device, dtype=dtype).view(*shape)

        expected_box = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(*shape, 2)

        boxes_xyxy = Boxes.from_tensor(box_xyxy, mode='xyxy')._boxes
        boxes_xyxy_plus_1 = Boxes.from_tensor(box_xyxy_plus_1, mode='xyxy_plus_1')._boxes
        boxes_xywh = Boxes.from_tensor(box_xywh, mode='xywh')._boxes
        boxes_xywh_plus_1 = Boxes.from_tensor(box_xywh_plus_1, mode='xywh_plus_1')._boxes

        assert boxes_xyxy.shape == expected_box.shape
        assert_allclose(boxes_xyxy, expected_box)

        assert boxes_xyxy_plus_1.shape == expected_box.shape
        assert_allclose(boxes_xyxy_plus_1, expected_box)

        assert boxes_xywh.shape == expected_box.shape
        assert_allclose(boxes_xywh, expected_box)

        assert boxes_xywh_plus_1.shape == expected_box.shape
        assert_allclose(boxes_xywh_plus_1, expected_box)

    @pytest.mark.parametrize('shape', [(1, 4), (1, 1, 4)])
    def test_boxes_to_tensor(self, shape: Tuple[int], device, dtype):
        # quadrilateral with randomized vertices to reflect possible transforms.
        box = Boxes(torch.as_tensor([[[3, 2], [3, 4], [1, 4], [1, 2]]], device=device, dtype=dtype).view(*shape, 2))

        expected_box_xyxy = torch.as_tensor([[1, 2, 3, 4]], device=device, dtype=dtype).view(*shape)
        expected_box_xyxy_plus_1 = torch.as_tensor([[1, 2, 4, 5]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh = torch.as_tensor([[1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_box_xywh_plus_1 = torch.as_tensor([[1, 2, 3, 3]], device=device, dtype=dtype).view(*shape)
        expected_vertices = torch.as_tensor([[[1, 2], [3, 2], [3, 4], [1, 4]]], device=device, dtype=dtype).view(
            *shape, 2
        )

        boxes_xyxy = box.to_tensor(mode='xyxy')
        boxes_xyxy_plus_1 = box.to_tensor(mode='xyxy_plus_1')
        boxes_xywh = box.to_tensor(mode='xywh')
        boxes_xywh_plus_1 = box.to_tensor(mode='xywh_plus_1')
        boxes_vertices = box.to_tensor(mode='vertices')

        assert boxes_xyxy.shape == expected_box_xyxy.shape
        assert_allclose(boxes_xyxy, expected_box_xyxy)

        assert boxes_xyxy_plus_1.shape == expected_box_xyxy_plus_1.shape
        assert_allclose(boxes_xyxy_plus_1, expected_box_xyxy_plus_1)

        assert boxes_xywh.shape == expected_box_xywh.shape
        assert_allclose(boxes_xywh, expected_box_xywh)

        assert boxes_xywh_plus_1.shape == expected_box_xywh_plus_1.shape
        assert_allclose(boxes_xywh_plus_1, expected_box_xywh_plus_1)

        assert boxes_vertices.shape == expected_vertices.shape
        assert_allclose(boxes_vertices, expected_vertices)

    def test_boxes_to_mask(self, device, dtype):
        t_box1 = torch.tensor(
            [[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype
        )  # (1, 4, 2)
        t_box2 = torch.tensor(
            [[[2.0, 2.0], [5.0, 2.0], [5.0, 6.0], [5.0, 2.0]]], device=device, dtype=dtype
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
        assert_allclose(mask1, expected_mask1)

        assert mask2.shape == expected_mask2.shape
        assert_allclose(mask2, expected_mask2)

        assert two_masks.shape == expected_two_masks.shape
        assert_allclose(two_masks, expected_two_masks)

        assert batched_masks.shape == expected_batched_masks.shape
        assert_allclose(batched_masks, expected_batched_masks)

    def test_to(self, device, dtype):
        if device == torch.device('cpu'):
            pass

        boxes = Boxes.from_tensor(torch.as_tensor([[1, 2, 3, 4]], device='cpu', dtype=torch.float32))
        assert boxes.to(device=device)._boxes.device == device
        assert boxes.to(dtype=dtype)._boxes.dtype == dtype

        boxes_moved = boxes.to(device, dtype)
        assert boxes_moved is boxes  # to is an inplace op.
        assert boxes_moved._boxes.device == device, boxes_moved._boxes.dtype == dtype

    def test_gradcheck(self, device, dtype):
        def apply_boxes_method(tensor: torch.Tensor, method: str, **kwargs):
            boxes = Boxes(tensor)
            result = getattr(boxes, method)(**kwargs)
            return result._boxes if isinstance(result, Boxes) else result

        t_boxes1 = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)

        t_boxes1 = utils.tensor_to_gradcheck_var(t_boxes1)
        t_boxes2 = utils.tensor_to_gradcheck_var(t_boxes1.detach().clone())
        t_boxes3 = utils.tensor_to_gradcheck_var(t_boxes1.detach().clone())
        t_boxes4 = utils.tensor_to_gradcheck_var(t_boxes1.detach().clone())
        t_boxes_xyxy = utils.tensor_to_gradcheck_var(torch.tensor([[1.0, 3.0, 5.0, 6.0]]))
        t_boxes_xyxy1 = utils.tensor_to_gradcheck_var(t_boxes_xyxy.detach().clone())

        assert gradcheck(partial(apply_boxes_method, method='to_tensor'), (t_boxes2,), raise_exception=True)
        assert gradcheck(
            partial(apply_boxes_method, method='to_tensor', mode='xywh_plus_1'), (t_boxes3,), raise_exception=True
        )
        assert gradcheck(
            partial(apply_boxes_method, method='to_tensor', mode='vertices_plus_1'), (t_boxes4,), raise_exception=True
        )
        assert gradcheck(partial(apply_boxes_method, method='get_boxes_shape'), (t_boxes1,), raise_exception=True)
        assert gradcheck(lambda x: Boxes.from_tensor(x)._boxes, (t_boxes_xyxy,), raise_exception=True)
        assert gradcheck(
            lambda x: Boxes.from_tensor(x, mode='xywh_plus_1')._boxes, (t_boxes_xyxy1,), raise_exception=True
        )


class TestTransformBoxes2D:
    def test_transform_boxes(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)
        expected_boxes_xyxy = torch.tensor([[372.7360, 103.0150, 114.6880, 410.5225]], device=device, dtype=dtype)

        boxes = Boxes.from_tensor(boxes_xyxy)
        expected_boxes = Boxes.from_tensor(expected_boxes_xyxy)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        transformed_boxes = boxes.transform_boxes(trans_mat)
        assert_allclose(transformed_boxes._boxes, expected_boxes._boxes, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is not boxes

    def test_transform_boxes_(self, device, dtype):
        # Define boxes in XYXY format for simplicity.
        boxes_xyxy = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)
        expected_boxes_xyxy = torch.tensor([[372.7360, 103.0150, 114.6880, 410.5225]], device=device, dtype=dtype)

        boxes = Boxes.from_tensor(boxes_xyxy)
        expected_boxes = Boxes.from_tensor(expected_boxes_xyxy)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        transformed_boxes = boxes.transform_boxes_(trans_mat)
        assert_allclose(transformed_boxes._boxes, expected_boxes._boxes, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is boxes

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

        boxes = Boxes.from_tensor(boxes_xyxy)
        expected_boxes = Boxes.from_tensor(expected_boxes_xyxy)

        out = boxes.transform_boxes(trans_mat)
        assert_allclose(out._boxes, expected_boxes._boxes, atol=1e-4, rtol=1e-4)

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
        boxes = Boxes.from_tensor(boxes_xyxy)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        t_boxes = utils.tensor_to_gradcheck_var(boxes._boxes)

        def _wrapper_transform_boxes(quadrilaterals, M):
            boxes = Boxes(quadrilaterals)
            boxes = boxes.transform_boxes(M)
            return boxes._boxes

        assert gradcheck(_wrapper_transform_boxes, (t_boxes, trans_mat), raise_exception=True)


class TestBbox3D:
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
        assert (d.item(), h.item(), w.item()) == (30.0, 20.0, 10.0)

        # Boxes
        d, h, w = boxes.get_boxes_shape()
        assert h.ndim == 1 and w.ndim == 1
        assert len(d) == 2 and len(h) == 2 and len(w) == 2
        assert (
            (d == torch.as_tensor([30.0, 60.0], device=device)).all()
            and (h == torch.as_tensor([20.0, 50.0], device=device)).all()
            and (w == torch.as_tensor([10.0, 40.0], device=device)).all()
        )

        # Box batch
        d, h, w = boxes_batch.get_boxes_shape()
        assert h.ndim == 2 and w.ndim == 2
        assert h.shape == (1, 2) and w.shape == (1, 2)
        assert (
            (d == torch.as_tensor([[30.0, 60.0]], device=device)).all()
            and (h == torch.as_tensor([[20.0, 50.0]], device=device)).all()
            and (w == torch.as_tensor([[10.0, 40.0]], device=device)).all()
        )

    @pytest.mark.parametrize('shape', [(1, 6), (1, 1, 6)])
    def test_from_tensor(self, shape: Tuple[int], device, dtype):
        box_xyzxyz = torch.as_tensor([[1, 2, 3, 4, 5, 6]], device=device, dtype=dtype).view(*shape)
        box_xyzxyz_plus_1 = torch.as_tensor([[1, 2, 3, 5, 6, 7]], device=device, dtype=dtype).view(*shape)
        box_xyzwhd = torch.as_tensor([[1, 2, 3, 3, 3, 3]], device=device, dtype=dtype).view(*shape)
        box_xyzwhd_plus_1 = torch.as_tensor([[1, 2, 3, 4, 4, 4]], device=device, dtype=dtype).view(*shape)

        expected_box = torch.as_tensor(
            [[[1, 2, 3], [4, 2, 3], [4, 5, 3], [1, 5, 3], [1, 2, 6], [4, 2, 6], [4, 5, 6], [1, 5, 6]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        kornia_xyzxyz = Boxes3D.from_tensor(box_xyzxyz, mode='xyzxyz')._boxes
        kornia_xyzxyz_plus_1 = Boxes3D.from_tensor(box_xyzxyz_plus_1, mode='xyzxyz_plus_1')._boxes
        kornia_xyzwhd = Boxes3D.from_tensor(box_xyzwhd, mode='xyzwhd')._boxes
        kornia_xyzwhd_plus_1 = Boxes3D.from_tensor(box_xyzwhd_plus_1, mode='xyzwhd_plus_1')._boxes

        assert kornia_xyzxyz.shape == expected_box.shape
        assert_allclose(kornia_xyzxyz, expected_box)

        assert kornia_xyzxyz_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xyzxyz_plus_1, expected_box)

        assert kornia_xyzwhd.shape == expected_box.shape
        assert_allclose(kornia_xyzwhd, expected_box)

        assert kornia_xyzwhd_plus_1.shape == expected_box.shape
        assert_allclose(kornia_xyzwhd_plus_1, expected_box)

    @pytest.mark.parametrize('shape', [(1, 6), (1, 1, 6)])
    def test_boxes_to_tensor(self, shape: Tuple[int], device, dtype):
        # Hexahedron with randomized vertices to reflect possible transforms.
        box = Boxes3D(
            torch.as_tensor(
                [[[3, 2, 1], [1, 2, 1], [3, 4, 3], [1, 4, 3], [3, 2, 3], [1, 4, 1], [3, 4, 1], [1, 2, 3]]],
                device=device,
                dtype=dtype,
            ).view(*shape[:-1], 8, 3)
        )

        expected_box_xyzxyz = torch.as_tensor([[1, 2, 1, 3, 4, 3]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzxyz_plus_1 = torch.as_tensor([[1, 2, 1, 4, 5, 4]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzwhd = torch.as_tensor([[1, 2, 1, 2, 2, 2]], device=device, dtype=dtype).view(*shape)
        expected_box_xyzwhd_plus_1 = torch.as_tensor([[1, 2, 1, 3, 3, 3]], device=device, dtype=dtype).view(*shape)
        expected_vertices = torch.as_tensor(
            [[[1, 2, 1], [3, 2, 1], [3, 4, 1], [1, 4, 1], [1, 2, 3], [3, 2, 3], [3, 4, 3], [1, 4, 3]]],  # Front  # Back
            device=device,
            dtype=dtype,
        ).view(*shape[:-1], 8, 3)

        kornia_xyzxyz = box.to_tensor(mode='xyzxyz')
        kornia_xyzxyz_plus_1 = box.to_tensor(mode='xyzxyz_plus_1')
        kornia_xyzwhd = box.to_tensor(mode='xyzwhd')
        kornia_xyzwhd_plus_1 = box.to_tensor(mode='xyzwhd_plus_1')
        kornia_vertices = box.to_tensor(mode='vertices')

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
        t_box1 = torch.tensor(
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
        )  # (1, 8, 3)
        t_box2 = torch.tensor(
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
        assert_allclose(mask1, expected_mask1)

        assert mask2.shape == expected_mask2.shape
        assert_allclose(mask2, expected_mask2)

        assert two_masks.shape == expected_two_masks.shape
        assert_allclose(two_masks, expected_two_masks)

        assert batched_masks.shape == expected_batched_masks.shape
        assert_allclose(batched_masks, expected_batched_masks)

    def test_to(self, device, dtype):
        if device == torch.device('cpu'):
            pass

        boxes = Boxes3D.from_tensor(torch.as_tensor([[1, 2, 3, 4, 5, 6]], device='cpu', dtype=torch.float32))
        assert boxes.to(device=device)._boxes.device == device
        assert boxes.to(dtype=dtype)._boxes.dtype == dtype

        boxes_moved = boxes.to(device, dtype)
        assert boxes_moved is boxes  # to is an inplace op.
        assert boxes_moved._boxes.device == device, boxes_moved._boxes.dtype == dtype

    def test_gradcheck(self, device, dtype):
        def apply_boxes_method(tensor: torch.Tensor, method: str, **kwargs):
            boxes = Boxes3D(tensor)
            result = getattr(boxes, method)(**kwargs)
            return result._boxes if isinstance(result, Boxes3D) else result

        t_boxes1 = torch.tensor(
            [[[0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2], [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]]],
            device=device,
            dtype=dtype,
        )

        t_boxes1 = utils.tensor_to_gradcheck_var(t_boxes1)
        t_boxes2 = utils.tensor_to_gradcheck_var(t_boxes1.detach().clone())
        t_boxes3 = utils.tensor_to_gradcheck_var(t_boxes1.detach().clone())
        t_boxes4 = utils.tensor_to_gradcheck_var(t_boxes1.detach().clone())
        t_boxes_xyzxyz = utils.tensor_to_gradcheck_var(torch.tensor([[1.0, 3.0, 8.0, 5.0, 6.0, 12.0]]))
        t_boxes_xyzxyz1 = utils.tensor_to_gradcheck_var(t_boxes_xyzxyz.detach().clone())

        assert gradcheck(partial(apply_boxes_method, method='to_tensor'), (t_boxes2,), raise_exception=True)
        assert gradcheck(partial(apply_boxes_method, method='to_tensor'), (t_boxes3,), raise_exception=True)
        assert gradcheck(partial(apply_boxes_method, method='to_tensor'), (t_boxes4,), raise_exception=True)
        assert gradcheck(partial(apply_boxes_method, method='get_boxes_shape'), (t_boxes1,), raise_exception=True)
        assert gradcheck(lambda x: Boxes3D.from_tensor(x)._boxes, (t_boxes_xyzxyz,), raise_exception=True)
        assert gradcheck(
            lambda x: Boxes3D.from_tensor(x, mode='xyzwhd_plus_1')._boxes, (t_boxes_xyzxyz1,), raise_exception=True
        )


class TestTransformBoxes3D:
    def test_transform_boxes(self, device, dtype):
        # Define boxes in XYZXYZ format for simplicity.
        boxes_xyzxyz = torch.tensor(
            [[139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185]], device=device, dtype=dtype
        )
        expected_boxes_xyzxyz = torch.tensor(
            [[372.7360, 103.0150, 567.324, 114.6880, 410.5225, 907.37]], device=device, dtype=dtype
        )

        boxes = Boxes3D.from_tensor(boxes_xyzxyz)
        expected_boxes = Boxes3D.from_tensor(expected_boxes_xyzxyz)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        transformed_boxes = boxes.transform_boxes(trans_mat)
        assert_allclose(transformed_boxes._boxes, expected_boxes._boxes, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is not boxes

    def test_transform_boxes_(self, device, dtype):
        # Define boxes in XYZXYZ format for simplicity.
        boxes_xyzxyz = torch.tensor(
            [[139.2640, 103.0150, 283.162, 397.3120, 410.5225, 453.185]], device=device, dtype=dtype
        )
        expected_boxes_xyzxyz = torch.tensor(
            [[372.7360, 103.0150, 567.324, 114.6880, 410.5225, 907.37]], device=device, dtype=dtype
        )

        boxes = Boxes3D.from_tensor(boxes_xyzxyz)
        expected_boxes = Boxes3D.from_tensor(expected_boxes_xyzxyz)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        transformed_boxes = boxes.transform_boxes_(trans_mat)
        assert_allclose(transformed_boxes._boxes, expected_boxes._boxes, atol=1e-4, rtol=1e-4)
        # inplace check
        assert transformed_boxes is boxes

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

        boxes = Boxes3D.from_tensor(boxes_xyzxyz)
        expected_boxes = Boxes3D.from_tensor(expected_boxes_xyzxyz)

        out = boxes.transform_boxes(trans_mat)
        assert_allclose(out._boxes, expected_boxes._boxes, atol=1e-4, rtol=1e-4)

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
        boxes = Boxes3D.from_tensor(boxes_xyzxyz)

        trans_mat = torch.tensor(
            [[[-1.0, 0.0, 0.0, 512.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        t_boxes = utils.tensor_to_gradcheck_var(boxes._boxes)

        def _wrapper_transform_boxes(hexahedrons, M):
            boxes = Boxes3D(hexahedrons)
            boxes = boxes.transform_boxes(M)
            return boxes._boxes

        assert gradcheck(_wrapper_transform_boxes, (t_boxes, trans_mat), raise_exception=True)
