import pytest
import torch
from torch.autograd import gradcheck
from torch.nn.functional import mse_loss

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry.subpix.spatial_soft_argmax import _get_center_kernel2d, _get_center_kernel3d
from kornia.testing import assert_close


class TestCenterKernel2d:
    def test_smoke(self, device, dtype):
        kernel = _get_center_kernel2d(3, 4, device=device).to(dtype=dtype)
        assert kernel.shape == (2, 2, 3, 4)

    def test_odd(self, device, dtype):
        kernel = _get_center_kernel2d(3, 3, device=device).to(dtype=dtype)
        expected = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(kernel, expected, atol=1e-4, rtol=1e-4)

    def test_even(self, device, dtype):
        kernel = _get_center_kernel2d(2, 2, device=device).to(dtype=dtype)
        expected = torch.ones(2, 2, 2, 2, device=device, dtype=dtype) * 0.25
        expected[0, 1] = 0
        expected[1, 0] = 0
        assert_close(kernel, expected, atol=1e-4, rtol=1e-4)


class TestCenterKernel3d:
    def test_smoke(self, device, dtype):
        kernel = _get_center_kernel3d(6, 3, 4, device=device).to(dtype=dtype)
        assert kernel.shape == (3, 3, 6, 3, 4)

    def test_odd(self, device, dtype):
        kernel = _get_center_kernel3d(3, 5, 7, device=device).to(dtype=dtype)
        expected = torch.zeros(3, 3, 3, 5, 7, device=device, dtype=dtype)
        expected[0, 0, 1, 2, 3] = 1.0
        expected[1, 1, 1, 2, 3] = 1.0
        expected[2, 2, 1, 2, 3] = 1.0
        assert_close(kernel, expected, atol=1e-4, rtol=1e-4)

    def test_even(self, device, dtype):
        kernel = _get_center_kernel3d(2, 4, 3, device=device).to(dtype=dtype)
        expected = torch.zeros(3, 3, 2, 4, 3, device=device, dtype=dtype)
        expected[0, 0, :, 1:3, 1] = 0.25
        expected[1, 1, :, 1:3, 1] = 0.25
        expected[2, 2, :, 1:3, 1] = 0.25
        assert_close(kernel, expected, atol=1e-4, rtol=1e-4)


class TestSpatialSoftArgmax2d:
    def test_smoke(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.SpatialSoftArgmax2d()
        assert m(sample).shape == (1, 1, 2)

    def test_smoke_batch(self, device, dtype):
        sample = torch.zeros(2, 1, 2, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.SpatialSoftArgmax2d()
        assert m(sample).shape == (2, 1, 2)

    def test_top_left_normalized(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., 0, 0] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=True)
        assert_close(coord[..., 0].item(), -1.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[..., 1].item(), -1.0, atol=1e-4, rtol=1e-4)

    def test_top_left(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., 0, 0] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=False)
        assert_close(coord[..., 0].item(), 0.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[..., 1].item(), 0.0, atol=1e-4, rtol=1e-4)

    def test_bottom_right_normalized(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., -1, -1] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=True)
        assert_close(coord[..., 0].item(), 1.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[..., 1].item(), 1.0, atol=1e-4, rtol=1e-4)

    def test_bottom_right(self, device, dtype):
        sample = torch.zeros(1, 1, 2, 3, device=device, dtype=dtype)
        sample[..., -1, -1] = 1e16

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample, normalized_coordinates=False)
        assert_close(coord[..., 0].item(), 2.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[..., 1].item(), 1.0, atol=1e-4, rtol=1e-4)

    def test_batch2_n2(self, device, dtype):
        sample = torch.zeros(2, 2, 2, 3, device=device, dtype=dtype)
        sample[0, 0, 0, 0] = 1e16  # top-left
        sample[0, 1, 0, -1] = 1e16  # top-right
        sample[1, 0, -1, 0] = 1e16  # bottom-left
        sample[1, 1, -1, -1] = 1e16  # bottom-right

        coord = kornia.geometry.subpix.spatial_soft_argmax2d(sample)
        assert_close(coord[0, 0, 0].item(), -1.0, atol=1e-4, rtol=1e-4)  # top-left
        assert_close(coord[0, 0, 1].item(), -1.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[0, 1, 0].item(), 1.0, atol=1e-4, rtol=1e-4)  # top-right
        assert_close(coord[0, 1, 1].item(), -1.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[1, 0, 0].item(), -1.0, atol=1e-4, rtol=1e-4)  # bottom-left
        assert_close(coord[1, 0, 1].item(), 1.0, atol=1e-4, rtol=1e-4)
        assert_close(coord[1, 1, 0].item(), 1.0, atol=1e-4, rtol=1e-4)  # bottom-right
        assert_close(coord[1, 1, 1].item(), 1.0, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        sample = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        sample = utils.tensor_to_gradcheck_var(sample)  # to var
        assert gradcheck(kornia.geometry.subpix.spatial_soft_argmax2d, (sample), raise_exception=True, fast_mode=True)

    def test_end_to_end(self, device, dtype):
        sample = torch.full((1, 2, 7, 7), 1.0, requires_grad=True, device=device, dtype=dtype)
        target = torch.as_tensor([[[0.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype)
        std = torch.tensor([1.0, 1.0], device=device, dtype=dtype)

        hm = kornia.geometry.subpix.spatial_softmax2d(sample)
        assert_close(hm.sum(-1).sum(-1), torch.tensor([[1.0, 1.0]], device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

        pred = kornia.geometry.subpix.spatial_expectation2d(hm)
        assert_close(
            pred, torch.as_tensor([[[0.0, 0.0], [0.0, 0.0]]], device=device, dtype=dtype), atol=1e-4, rtol=1e-4
        )

        loss1 = mse_loss(pred, target, size_average=None, reduce=None, reduction='none').mean(-1, keepdim=False)
        expected_loss1 = torch.as_tensor([[0.0, 1.0]], device=device, dtype=dtype)
        assert_close(loss1, expected_loss1, atol=1e-4, rtol=1e-4)

        target_hm = kornia.geometry.subpix.render_gaussian2d(target, std, sample.shape[-2:]).contiguous()

        loss2 = kornia.losses.js_div_loss_2d(hm, target_hm, reduction='none')
        expected_loss2 = torch.as_tensor([[0.0087, 0.0818]], device=device, dtype=dtype)
        assert_close(loss2, expected_loss2, rtol=0, atol=1e-3)

        loss = (loss1 + loss2).mean()
        loss.backward()

    @pytest.mark.slow
    def test_dynamo(self, device, dtype, torch_optimizer):
        inpt = torch.rand((2, 3, 7, 7), dtype=dtype, device=device)
        op = kornia.geometry.subpix.spatial_soft_argmax2d
        op_optimized = torch_optimizer(op)

        assert_close(op(inpt), op_optimized(inpt))


class TestConvSoftArgmax2d:
    def test_smoke(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d((3, 3))
        assert m(sample).shape == (1, 1, 2, 3, 3)

    def test_smoke_batch(self, device, dtype):
        sample = torch.zeros(2, 5, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d()
        assert m(sample).shape == (2, 5, 2, 3, 3)

    def test_smoke_with_val(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(sample)
        assert coords.shape == (1, 1, 2, 3, 3)
        assert val.shape == (1, 1, 3, 3)

    def test_smoke_batch_with_val(self, device, dtype):
        sample = torch.zeros(2, 5, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(sample)
        assert coords.shape == (2, 5, 2, 3, 3)
        assert val.shape == (2, 5, 3, 3)

    def test_gradcheck(self, device, dtype):
        sample = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        sample = utils.tensor_to_gradcheck_var(sample)  # to var
        assert gradcheck(
            kornia.geometry.subpix.conv_soft_argmax2d, (sample), nondet_tol=1e-8, raise_exception=True, fast_mode=True
        )

    def test_cold_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=0.05, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[1.0, 3.0], [1.0, 3.0]], [[1.0, 1.0], [3.0, 3.0]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=10.0, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[0.1214, 0.0], [0.0, 0.1214]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[1.0, 3.0], [1.0, 3.0]], [[1.0, 1.0], [3.0, 3.0]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_cold_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=0.05, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, -0.5], [0.5, 0.5]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax2d(
            (3, 3), (2, 2), (0, 0), temperature=10.0, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[0.1214, 0.0], [0.0, 0.1214]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, -0.5], [0.5, 0.5]]]]], device=device, dtype=dtype
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)


class TestConvSoftArgmax3d:
    def test_smoke(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax3d((3, 3, 3), output_value=False)
        assert m(sample).shape == (1, 1, 3, 3, 3, 3)

    def test_smoke_with_val(self, device, dtype):
        sample = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
        m = kornia.geometry.subpix.ConvSoftArgmax3d((3, 3, 3), output_value=True)
        coords, val = m(sample)
        assert coords.shape == (1, 1, 3, 3, 3, 3)
        assert val.shape == (1, 1, 3, 3, 3)

    def test_gradcheck(self, device, dtype):
        sample = torch.rand(1, 2, 3, 5, 5, device=device, dtype=dtype)
        sample = utils.tensor_to_gradcheck_var(sample)  # to var
        assert gradcheck(
            kornia.geometry.subpix.conv_soft_argmax3d, (sample), nondet_tol=1e-8, raise_exception=True, fast_mode=True
        )

    def test_cold_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=0.05, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 3.0], [1.0, 3.0]]], [[[1.0, 1.0], [3.0, 3.0]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=10.0, normalized_coordinates=False, output_value=True
        )
        expected_val = torch.tensor([[[[[0.1214, 0.0], [0.0, 0.1214]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 3.0], [1.0, 3.0]]], [[[1.0, 1.0], [3.0, 3.0]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_cold_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=0.05, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[-1.0, -1.0], [-1.0, -1.0]]], [[[-0.5, 0.5], [-0.5, 0.5]]], [[[-0.5, -0.5], [0.5, 0.5]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)

    def test_hot_diag_norm(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        softargmax = kornia.geometry.subpix.ConvSoftArgmax3d(
            (1, 3, 3), (1, 2, 2), (0, 0, 0), temperature=10.0, normalized_coordinates=True, output_value=True
        )
        expected_val = torch.tensor([[[[[0.1214, 0.0], [0.0, 0.1214]]]]], device=device, dtype=dtype)
        expected_coord = torch.tensor(
            [[[[[[-1.0, -1.0], [-1.0, -1.0]]], [[[-0.5, 0.5], [-0.5, 0.5]]], [[[-0.5, -0.5], [0.5, 0.5]]]]]],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)


class TestConvQuadInterp3d:
    def test_smoke(self, device, dtype):
        sample = torch.randn(2, 3, 3, 4, 4, device=device, dtype=dtype)
        nms = kornia.geometry.ConvQuadInterp3d(1)
        coord, val = nms(sample)
        assert coord.shape == (2, 3, 3, 3, 4, 4)
        assert val.shape == (2, 3, 3, 4, 4)

    def test_gradcheck(self, device, dtype):
        sample = torch.rand(1, 1, 3, 5, 5, device=device, dtype=dtype)
        sample[0, 0, 1, 2, 2] += 20.0
        sample = utils.tensor_to_gradcheck_var(sample)  # to var
        assert gradcheck(
            kornia.geometry.ConvQuadInterp3d(strict_maxima_bonus=0),
            (sample),
            raise_exception=True,
            atol=1e-3,
            rtol=1e-3,
            fast_mode=True,
        )

    def test_diag(self, device, dtype):
        sample = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                        [0.0, 0, 0.0, 0, 0.0],
                        [0.0, 0.0, 0, 0, 0.0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0, 0],
                        [0.0, 0.0, 1, 0, 0.0],
                        [0.0, 1, 1.2, 1.1, 0.0],
                        [0.0, 0.0, 1.0, 0, 0.0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                        [0.0, 0, 0.0, 0, 0.0],
                        [0.0, 0.0, 0, 0, 0.0],
                        [0.0, 0.0, 0.0, 0, 0.0],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        sample = kornia.filters.gaussian_blur2d(sample, (5, 5), (0.5, 0.5)).unsqueeze(0)
        softargmax = kornia.geometry.ConvQuadInterp3d(10)
        expected_val = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0, 0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                            [0.0, 0, 0.0, 0, 0.0],
                            [0.0, 0.0, 0, 0, 0.0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                        ],
                        [
                            [2.2504e-04, 2.3146e-02, 1.6808e-01, 2.3188e-02, 2.3628e-04],
                            [2.3146e-02, 1.8118e-01, 7.4338e-01, 1.8955e-01, 2.5413e-02],
                            [1.6807e-01, 7.4227e-01, 1.1086e01, 8.0414e-01, 1.8482e-01],
                            [2.3146e-02, 1.8118e-01, 7.4338e-01, 1.8955e-01, 2.5413e-02],
                            [2.2504e-04, 2.3146e-02, 1.6808e-01, 2.3188e-02, 2.3628e-04],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0, 0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                            [0.0, 0, 0.0, 0, 0.0],
                            [0.0, 0.0, 0, 0, 0.0],
                            [0.0, 0.0, 0.0, 0, 0.0],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        expected_coord = torch.tensor(
            [
                [
                    [
                        [
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                            ],
                            [
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                            ],
                        ],
                        [
                            [
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                            ],
                            [
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0495, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                            ],
                            [
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                                [0.0, 1.0, 2.0, 3.0, 4.0],
                            ],
                        ],
                        [
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0, 4.0, 4.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0, 4.0, 4.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0, 3.0, 3.0],
                                [4.0, 4.0, 4.0, 4.0, 4.0],
                            ],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        coords, val = softargmax(sample)
        assert_close(val, expected_val, atol=1e-4, rtol=1e-4)
        assert_close(coords, expected_coord, atol=1e-4, rtol=1e-4)
