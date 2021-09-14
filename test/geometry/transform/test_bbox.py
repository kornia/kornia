import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestTransformBoxes:
    def test_transform_bbox(self, device, dtype):

        boxes = torch.tensor([[139.2640, 103.0150, 397.3120, 410.5225]], device=device, dtype=dtype)

        expected = torch.tensor([[372.7360, 103.0150, 114.6880, 410.5225]], device=device, dtype=dtype)

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = kornia.transform_bbox(trans_mat, boxes)
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

        out = kornia.transform_bbox(trans_mat, boxes)
        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_transform_bbox_wh(self, device, dtype):

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
                [372.7360, 103.0150, -258.0480, 307.5075],
                [510.9760, 80.5547, -510.9760, 431.4453],
                [346.7947, 262.1440, -345.4293, 246.7840],
                [392.1920, 144.2067, -137.2160, 265.9225],
            ],
            device=device,
            dtype=dtype,
        )

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        out = kornia.transform_bbox(trans_mat, boxes, mode='xywh')
        assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):

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

        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        trans_mat = utils.tensor_to_gradcheck_var(trans_mat)
        boxes = utils.tensor_to_gradcheck_var(boxes)

        assert gradcheck(kornia.transform_bbox, (trans_mat, boxes), raise_exception=True)

    def test_jit(self, device, dtype):
        boxes = torch.tensor([[139.2640, 103.0150, 258.0480, 307.5075]], device=device, dtype=dtype)
        trans_mat = torch.tensor([[[-1.0, 0.0, 512.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        args = (boxes, trans_mat)
        op = kornia.geometry.transform_points
        op_jit = torch.jit.script(op)
        assert_close(op(*args), op_jit(*args))
