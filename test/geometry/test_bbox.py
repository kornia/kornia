import torch

import kornia
from kornia.geometry.bbox import (
    infer_bbox_shape,
    infer_bbox_shape3d,
    nms,
    transform_bbox,
    validate_bbox,
    validate_bbox3d,
)
from kornia.testing import assert_close, gradcheck, tensor_to_gradcheck_var


class TestBbox2D:
    def test_smoke(self, device, dtype):
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

        # Validate
        assert validate_bbox(bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        h, w = infer_bbox_shape(boxes)
        assert (h, w) == (2, 3)

    def test_bounding_boxes_dim_inferring_batch(self, device, dtype):
        boxes = torch.tensor(
            [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]], [[2.0, 2.0], [4.0, 2.0], [4.0, 3.0], [2.0, 3.0]]],
            device=device,
            dtype=dtype,
        )
        h, w = infer_bbox_shape(boxes)
        assert (h.unique().item(), w.unique().item()) == (2, 3)

    def test_gradcheck(self, device):
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device)
        boxes = tensor_to_gradcheck_var(boxes)
        gradcheck(infer_bbox_shape, (boxes,))

    def test_jit(self, device, dtype):
        # Define script
        op = infer_bbox_shape
        op_script = torch.jit.script(op)
        # Define input
        boxes = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        # Run
        expected = op(boxes)
        actual = op_script(boxes)
        # Compare
        assert_close(actual, expected)


class TestTransformBoxes2D:
    def test_transform_boxes(self, device, dtype):
        boxes = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)

        expected = torch.tensor([[114.6880, 103.0150, 372.7360, 410.5225]], device=device, dtype=dtype)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = transform_bbox(trans_mat, boxes, restore_coordinates=True)
        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_transform_multiple_boxes(self, device, dtype):
        boxes = torch.tensor(
            [
                [139.2640, 103.0150, 397.3120, 410.5225],
                [1.0240, 80.5547, 512.0000, 512.0000],
                [165.2053, 262.1440, 510.6347, 508.9280],
                [119.8080, 144.2067, 257.0240, 410.1292],
            ],
            device=device,
            dtype=dtype,
        )

        boxes = boxes.repeat(2, 1, 1)  # 2 x 4 x 4 two images 4 boxes each

        expected = torch.tensor(
            [
                [
                    [114.6880, 103.0150, 372.7360, 410.5225],
                    [0.0000, 80.5547, 510.9760, 512.0000],
                    [1.3652, 262.1440, 346.7947, 508.9280],
                    [254.9760, 144.2067, 392.1920, 410.1292],
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

        out = transform_bbox(trans_mat, boxes, restore_coordinates=True)
        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_transform_boxes_wh(self, device, dtype):
        boxes = torch.tensor(
            [
                [139.2640, 103.0150, 258.0480, 307.5075],
                [1.0240, 80.5547, 510.9760, 431.4453],
                [165.2053, 262.1440, 345.4293, 246.7840],
                [119.8080, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [114.6880, 103.0150, 258.0480, 307.5075],
                [0.0000, 80.5547, 510.9760, 431.4453],
                [1.3654, 262.1440, 345.4293, 246.7840],
                [254.9760, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = transform_bbox(trans_mat, boxes, mode='xywh', restore_coordinates=True)
        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        boxes = torch.tensor(
            [
                [139.2640, 103.0150, 258.0480, 307.5075],
                [1.0240, 80.5547, 510.9760, 431.4453],
                [165.2053, 262.1440, 345.4293, 246.7840],
                [119.8080, 144.2067, 137.2160, 265.9225],
            ],
            device=device,
        )

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)

        trans_mat = tensor_to_gradcheck_var(trans_mat)
        boxes = tensor_to_gradcheck_var(boxes)

        gradcheck(transform_bbox, (trans_mat, boxes, "xyxy", True))

    def test_jit(self, device, dtype):
        boxes = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075]], device=device, dtype=dtype)
        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        args = (boxes, trans_mat)
        op = kornia.geometry.transform_points
        op_jit = torch.jit.script(op)
        assert_close(op(*args), op_jit(*args))


class TestBbox3D:
    def test_smoke(self, device, dtype):
        # Sample two points of the 3d rect
        points = torch.rand(1, 6, device=device, dtype=dtype)

        # Fill according missing points
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

        # Validate
        assert validate_bbox3d(bbox)

    def test_bounding_boxes_dim_inferring(self, device, dtype):
        boxes = torch.tensor(
            [
                [[0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2], [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]],
                [[3, 4, 5], [43, 4, 5], [43, 54, 5], [3, 54, 5], [3, 4, 65], [43, 4, 65], [43, 54, 65], [3, 54, 65]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x8x3
        d, h, w = infer_bbox_shape3d(boxes)

        assert_close(d, torch.tensor([31.0, 61.0], device=device, dtype=dtype))
        assert_close(h, torch.tensor([21.0, 51.0], device=device, dtype=dtype))
        assert_close(w, torch.tensor([11.0, 41.0], device=device, dtype=dtype))

    def test_gradcheck(self, device):
        boxes = torch.tensor(
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
        )
        boxes = tensor_to_gradcheck_var(boxes)
        gradcheck(infer_bbox_shape3d, (boxes,))

    def test_jit(self, device, dtype):
        # Define script
        op = infer_bbox_shape3d
        op_script = torch.jit.script(op)

        boxes = torch.tensor(
            [[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]],
            device=device,
            dtype=dtype,
        )  # 1x8x3

        actual = op_script(boxes)
        expected = op(boxes)
        assert_close(actual, expected)


class TestNMS:
    def test_smoke(self, device, dtype):
        boxes = torch.tensor(
            [
                [10.0, 10.0, 20.0, 20.0],
                [15.0, 5.0, 15.0, 25.0],
                [100.0, 100.0, 200.0, 200.0],
                [100.0, 100.0, 200.0, 200.0],
            ],
            device=device,
            dtype=dtype,
        )
        scores = torch.tensor([0.9, 0.8, 0.7, 0.9], device=device, dtype=dtype)
        expected = torch.tensor([0, 3, 1], device=device, dtype=torch.long)
        actual = nms(boxes, scores, iou_threshold=0.8)
        assert_close(actual, expected)
