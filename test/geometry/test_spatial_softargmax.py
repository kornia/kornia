import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.nn.functional import mse_loss
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from kornia.geometry.spatial_soft_argmax import _get_center_kernel2d, _get_center_kernel3d


class TestCenterKernel2d:
    def test_smoke(self, device):
        kernel = _get_center_kernel2d(3, 4).to(device)
        assert kernel.shape == (2, 2, 3, 4)

    def test_odd(self, device):
        kernel = _get_center_kernel2d(3, 3).to(device)
        expected = torch.tensor([
            [[[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]],
             [[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]]],
            [[[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
             [[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]]]]).to(device)
        assert_allclose(kernel, expected)

    def test_even(self, device):
        kernel = _get_center_kernel2d(2, 2).to(device)
        expected = torch.ones(2, 2, 2, 2).to(device) * 0.25
        expected[0, 1] = 0
        expected[1, 0] = 0
        assert_allclose(kernel, expected)


class TestCenterKernel3d:
    def test_smoke(self, device):
        kernel = _get_center_kernel3d(6, 3, 4).to(device)
        assert kernel.shape == (3, 3, 6, 3, 4)

    def test_odd(self, device):
        kernel = _get_center_kernel3d(3, 5, 7).to(device)
        expected = torch.zeros(3, 3, 3, 5, 7).to(device)
        expected[0, 0, 1, 2, 3] = 1.
        expected[1, 1, 1, 2, 3] = 1.
        expected[2, 2, 1, 2, 3] = 1.
        assert_allclose(kernel, expected)

    def test_even(self, device):
        kernel = _get_center_kernel3d(2, 4, 3).to(device)
        expected = torch.zeros(3, 3, 2, 4, 3).to(device)
        expected[0, 0, :, 1:3, 1] = 0.25
        expected[1, 1, :, 1:3, 1] = 0.25
        expected[2, 2, :, 1:3, 1] = 0.25
        assert_allclose(kernel, expected)


class TestSpatialSoftArgmax2d:
    def test_smoke(self, device):
        input = torch.zeros(1, 1, 2, 3).to(device)
        m = kornia.SpatialSoftArgmax2d()
        assert m(input).shape == (1, 1, 2)

    def test_smoke_batch(self, device):
        input = torch.zeros(2, 1, 2, 3).to(device)
        m = kornia.SpatialSoftArgmax2d()
        assert m(input).shape == (2, 1, 2)

    def test_top_left_normalized(self, device):
        input = torch.zeros(1, 1, 2, 3).to(device)
        input[..., 0, 0] = 1e16

        coord = kornia.spatial_soft_argmax2d(input, normalized_coordinates=True)
        assert_allclose(coord[..., 0].item(), -1.0)
        assert_allclose(coord[..., 1].item(), -1.0)

    def test_top_left(self, device):
        input = torch.zeros(1, 1, 2, 3).to(device)
        input[..., 0, 0] = 1e16

        coord = kornia.spatial_soft_argmax2d(input, normalized_coordinates=False)
        assert_allclose(coord[..., 0].item(), 0.0)
        assert_allclose(coord[..., 1].item(), 0.0)

    def test_bottom_right_normalized(self, device):
        input = torch.zeros(1, 1, 2, 3).to(device)
        input[..., -1, -1] = 1e16

        coord = kornia.spatial_soft_argmax2d(input, normalized_coordinates=True)
        assert_allclose(coord[..., 0].item(), 1.0)
        assert_allclose(coord[..., 1].item(), 1.0)

    def test_bottom_right(self, device):
        input = torch.zeros(1, 1, 2, 3).to(device)
        input[..., -1, -1] = 1e16

        coord = kornia.spatial_soft_argmax2d(input, normalized_coordinates=False)
        assert_allclose(coord[..., 0].item(), 2.0)
        assert_allclose(coord[..., 1].item(), 1.0)

    def test_batch2_n2(self, device):
        input = torch.zeros(2, 2, 2, 3).to(device)
        input[0, 0, 0, 0] = 1e16  # top-left
        input[0, 1, 0, -1] = 1e16  # top-right
        input[1, 0, -1, 0] = 1e16  # bottom-left
        input[1, 1, -1, -1] = 1e16  # bottom-right

        coord = kornia.spatial_soft_argmax2d(input)
        assert_allclose(coord[0, 0, 0].item(), -1.0)  # top-left
        assert_allclose(coord[0, 0, 1].item(), -1.0)
        assert_allclose(coord[0, 1, 0].item(), 1.0)  # top-right
        assert_allclose(coord[0, 1, 1].item(), -1.0)
        assert_allclose(coord[1, 0, 0].item(), -1.0)  # bottom-left
        assert_allclose(coord[1, 0, 1].item(), 1.0)
        assert_allclose(coord[1, 1, 0].item(), 1.0)  # bottom-right
        assert_allclose(coord[1, 1, 1].item(), 1.0)

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 3, 2).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.spatial_soft_argmax2d,
                         (input), raise_exception=True)

    def test_end_to_end(self, device):
        input = torch.full((1, 2, 7, 7), 1.0, requires_grad=True).to(device)
        target = torch.as_tensor([[[0.0, 0.0], [1.0, 1.0]]]).to(device)
        std = torch.tensor([1.0, 1.0]).to(device)

        hm = kornia.geometry.dsnt.spatial_softmax2d(input)
        assert_allclose(hm.sum(-1).sum(-1), torch.tensor(1.0).to(device))

        pred = kornia.geometry.dsnt.spatial_expectation2d(hm)
        assert_allclose(pred, torch.as_tensor([[[0.0, 0.0], [0.0, 0.0]]]).to(device))

        loss1 = mse_loss(pred, target, size_average=None, reduce=None,
                         reduction='none').mean(-1, keepdim=False)
        expected_loss1 = torch.as_tensor([[0.0, 1.0]]).to(device)
        assert_allclose(loss1, expected_loss1)

        target_hm = kornia.geometry.dsnt.render_gaussian2d(
            target, std, input.shape[-2:]).contiguous()

        loss2 = kornia.losses.js_div_loss_2d(hm, target_hm, reduction='none')
        expected_loss2 = torch.as_tensor([[0.0087, 0.0818]]).to(device)
        assert_allclose(loss2, expected_loss2, rtol=0, atol=1e-3)

        loss = (loss1 + loss2).mean()
        loss.backward()

    def test_jit(self, device, dtype):
        input = torch.rand((2, 3, 7, 7), dtype=dtype, device=device)
        op = kornia.spatial_soft_argmax2d
        op_jit = kornia.jit.spatial_soft_argmax2d
        assert_allclose(op(input), op_jit(input), rtol=0, atol=1e-5)

    def test_jit_trace(self, device, dtype):
        input = torch.rand((2, 3, 7, 7), dtype=dtype, device=device)
        op = kornia.spatial_soft_argmax2d
        op_jit = torch.jit.trace(op, (input,))
        assert_allclose(op(input), op_jit(input), rtol=0, atol=1e-5)


class TestConvSoftArgmax2d:
    def test_smoke(self, device):
        input = torch.zeros(1, 1, 3, 3).to(device)
        m = kornia.ConvSoftArgmax2d((3, 3))
        assert m(input).shape == (1, 1, 2, 3, 3)

    def test_smoke_batch(self, device):
        input = torch.zeros(2, 5, 3, 3).to(device)
        m = kornia.ConvSoftArgmax2d()
        assert m(input).shape == (2, 5, 2, 3, 3)

    def test_smoke_with_val(self, device):
        input = torch.zeros(1, 1, 3, 3).to(device)
        m = kornia.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(input)
        assert coords.shape == (1, 1, 2, 3, 3)
        assert val.shape == (1, 1, 3, 3)

    def test_smoke_batch_with_val(self, device):
        input = torch.zeros(2, 5, 3, 3).to(device)
        m = kornia.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(input)
        assert coords.shape == (2, 5, 2, 3, 3)
        assert val.shape == (2, 5, 3, 3)

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 5, 5).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.conv_soft_argmax2d,
                         (input), raise_exception=True)

    def test_cold_diag(self, device):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[1., 0.],
                                       [0., 1.]]]]).to(device)
        expected_coord = torch.tensor([[[[[1., 3.],
                                          [1., 3.]],
                                         [[1., 1.],
                                          [3., 3.]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag(self, device):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=10.,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[0.1214, 0.],
                                       [0., 0.1214]]]]).to(device)
        expected_coord = torch.tensor([[[[[1., 3.],
                                          [1., 3.]],
                                         [[1., 1.],
                                          [3., 3.]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_cold_diag_norm(self, device):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[1., 0.],
                                       [0., 1.]]]]).to(device)
        expected_coord = torch.tensor([[[[[-0.5, 0.5],
                                          [-0.5, 0.5]],
                                         [[-0.5, -0.5],
                                          [0.5, 0.5]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag_norm(self, device):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]).to(device)
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=10.,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[0.1214, 0.],
                                       [0., 0.1214]]]]).to(device)
        expected_coord = torch.tensor([[[[[-0.5, 0.5],
                                          [-0.5, 0.5]],
                                         [[-0.5, -0.5],
                                          [0.5, 0.5]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)


class TestConvSoftArgmax3d:
    def test_smoke(self, device):
        input = torch.zeros(1, 1, 3, 3, 3).to(device)
        m = kornia.ConvSoftArgmax3d((3, 3, 3), output_value=False)
        assert m(input).shape == (1, 1, 3, 3, 3, 3)

    def test_smoke_with_val(self, device):
        input = torch.zeros(1, 1, 3, 3, 3).to(device)
        m = kornia.ConvSoftArgmax3d((3, 3, 3), output_value=True)
        coords, val = m(input)
        assert coords.shape == (1, 1, 3, 3, 3, 3)
        assert val.shape == (1, 1, 3, 3, 3)

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 3, 5, 5).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.conv_soft_argmax3d,
                         (input), raise_exception=True)

    def test_cold_diag(self, device):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]]).to(device)
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[[1., 0.],
                                        [0., 1.]]]]]).to(device)
        expected_coord = torch.tensor([[[
                                        [[[0., 0.],
                                          [0., 0.]]],
                                        [[[1., 3.],
                                          [1., 3.]]],
                                        [[[1., 1.],
                                          [3., 3.]]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag(self, device):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]]).to(device)
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=10.,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[[0.1214, 0.],
                                        [0., 0.1214]]]]]).to(device)
        expected_coord = torch.tensor([[[
                                        [[[0., 0.],
                                          [0., 0.]]],
                                        [[[1., 3.],
                                          [1., 3.]]],
                                        [[[1., 1.],
                                          [3., 3.]]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_cold_diag_norm(self, device):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]]).to(device)
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[[1., 0.],
                                        [0., 1.]]]]]).to(device)
        expected_coord = torch.tensor([[[
            [[[-1., -1.],
              [-1., -1.]]],
            [[[-0.5, 0.5],
              [-0.5, 0.5]]],
            [[[-0.5, -0.5],
              [0.5, 0.5]]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag_norm(self, device):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]]).to(device)
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=10.,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[[0.1214, 0.],
                                        [0., 0.1214]]]]]).to(device)
        expected_coord = torch.tensor([[[
            [[[-1., -1.],
              [-1., -1.]]],
            [[[-0.5, 0.5],
              [-0.5, 0.5]]],
            [[[-0.5, -0.5],
              [0.5, 0.5]]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)


class TestConvQuadInterp3d:
    def test_smoke(self, device):
        input = torch.randn(2, 3, 3, 4, 4).to(device)
        nms = kornia.geometry.ConvQuadInterp3d(1)
        coord, val = nms(input)
        assert coord.shape == (2, 3, 3, 3, 4, 4)
        assert val.shape == (2, 3, 3, 4, 4)

    def test_gradcheck(self, device):
        input = torch.rand(1, 1, 3, 5, 5).to(device)
        input[0, 0, 1, 2, 2] += 20.
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.geometry.ConvQuadInterp3d(strict_maxima_bonus=0),
                         (input), raise_exception=True, atol=1e-3, rtol=1e-3)

    def test_diag(self, device):
        input = torch.tensor([[
            [[0., 0., 0., 0, 0],
             [0., 0., 0.0, 0, 0.],
             [0., 0, 0., 0, 0.],
             [0., 0., 0, 0, 0.],
             [0., 0., 0., 0, 0.]],

            [[0., 0., 0., 0, 0],
             [0., 0., 1, 0, 0.],
             [0., 1, 1.2, 1.1, 0.],
             [0., 0., 1., 0, 0.],
             [0., 0., 0., 0, 0.]],

            [[0., 0., 0., 0, 0],
             [0., 0., 0.0, 0, 0.],
             [0., 0, 0., 0, 0.],
             [0., 0., 0, 0, 0.],
             [0., 0., 0., 0, 0.],
             ]]]).to(device)
        input = kornia.gaussian_blur2d(input, (5, 5), (0.5, 0.5)).unsqueeze(0)
        softargmax = kornia.geometry.ConvQuadInterp3d(10)
        expected_val = torch.tensor([[[
            [[0., 0., 0., 0, 0],
             [0., 0., 0.0, 0, 0.],
             [0., 0, 0., 0, 0.],
             [0., 0., 0, 0, 0.],
             [0., 0., 0., 0, 0.]],
            [[2.2504e-04, 2.3146e-02, 1.6808e-01, 2.3188e-02, 2.3628e-04],
             [2.3146e-02, 1.8118e-01, 7.4338e-01, 1.8955e-01, 2.5413e-02],
             [1.6807e-01, 7.4227e-01, 1.1086e+01, 8.0414e-01, 1.8482e-01],
             [2.3146e-02, 1.8118e-01, 7.4338e-01, 1.8955e-01, 2.5413e-02],
             [2.2504e-04, 2.3146e-02, 1.6808e-01, 2.3188e-02, 2.3628e-04]],
            [[0., 0., 0., 0, 0],
             [0., 0., 0.0, 0, 0.],
             [0., 0, 0., 0, 0.],
             [0., 0., 0, 0, 0.],
             [0., 0., 0., 0, 0.]]]]]).to(device)
        expected_coord = torch.tensor([[[[[[0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0]],

                                          [[1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0]],

                                          [[2.0, 2.0, 2.0, 2.0, 2.0],
                                           [2.0, 2.0, 2.0, 2.0, 2.0],
                                           [2.0, 2.0, 2.0, 2.0, 2.0],
                                           [2.0, 2.0, 2.0, 2.0, 2.0],
                                           [2.0, 2.0, 2.0, 2.0, 2.0]]],


                                         [[[0.0, 1.0, 2.0, 3.0, 4.0],
                                           [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0]],

                                          [[0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0]],

                                          [[0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0],
                                             [0.0, 1.0, 2.0, 3.0, 4.0]]],


                                         [[[0.0, 0.0, 0.0, 0.0, 0.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0],
                                             [2.0, 2.0, 2.0, 2.0, 2.0],
                                             [3.0, 3.0, 3.0, 3.0, 3.0],
                                             [4.0, 4.0, 4.0, 4.0, 4.0]],

                                          [[0.0, 0.0, 0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0, 1.0, 1.0],
                                             [2.0, 2.0, 2.0495, 2.0, 2.0],
                                             [3.0, 3.0, 3.0, 3.0, 3.0],
                                             [4.0, 4.0, 4.0, 4.0, 4.0]],

                                          [[0.0, 0.0, 0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0, 1.0, 1.0],
                                             [2.0, 2.0, 2.0, 2.0, 2.0],
                                             [3.0, 3.0, 3.0, 3.0, 3.0],
                                             [4.0, 4.0, 4.0, 4.0, 4.0]]]]]]).to(device)
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)
