import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.nn.functional import mse_loss
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from kornia.geometry.spatial_soft_argmax import _get_center_kernel2d, _get_center_kernel3d


class TestCenterKernel2d:
    def test_smoke(self):
        kernel = _get_center_kernel2d(3, 4)
        assert kernel.shape == (2, 2, 3, 4)

    def test_odd(self):
        kernel = _get_center_kernel2d(3, 3)
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
              [0., 0., 0.]]]])
        assert_allclose(kernel, expected)

    def test_even(self):
        kernel = _get_center_kernel2d(2, 2)
        expected = torch.ones(2, 2, 2, 2) * 0.25
        expected[0, 1] = 0
        expected[1, 0] = 0
        assert_allclose(kernel, expected)


class TestCenterKernel3d:
    def test_smoke(self):
        kernel = _get_center_kernel3d(6, 3, 4)
        assert kernel.shape == (3, 3, 6, 3, 4)

    def test_odd(self):
        kernel = _get_center_kernel3d(3, 5, 7)
        expected = torch.zeros(3, 3, 3, 5, 7)
        expected[0, 0, 1, 2, 3] = 1.
        expected[1, 1, 1, 2, 3] = 1.
        expected[2, 2, 1, 2, 3] = 1.
        assert_allclose(kernel, expected)

    def test_even(self):
        kernel = _get_center_kernel3d(2, 4, 3)
        expected = torch.zeros(3, 3, 2, 4, 3)
        expected[0, 0, :, 1:3, 1] = 0.25
        expected[1, 1, :, 1:3, 1] = 0.25
        expected[2, 2, :, 1:3, 1] = 0.25
        assert_allclose(kernel, expected)


class TestSpatialSoftArgmax2d:
    def test_smoke(self):
        input = torch.zeros(1, 1, 2, 3)
        m = kornia.SpatialSoftArgmax2d()
        assert m(input).shape == (1, 1, 2)

    def test_smoke_batch(self):
        input = torch.zeros(2, 1, 2, 3)
        m = kornia.SpatialSoftArgmax2d()
        assert m(input).shape == (2, 1, 2)

    def test_top_left(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = kornia.spatial_soft_argmax2d(input, True)
        assert pytest.approx(coord[..., 0].item(), -1.0)
        assert pytest.approx(coord[..., 1].item(), -1.0)

    def test_top_left_normalized(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = kornia.spatial_soft_argmax2d(input, False)
        assert pytest.approx(coord[..., 0].item(), 0.0)
        assert pytest.approx(coord[..., 1].item(), 0.0)

    def test_bottom_right(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = kornia.spatial_soft_argmax2d(input, True)
        assert pytest.approx(coord[..., 0].item(), 1.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def test_bottom_right_normalized(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = kornia.spatial_soft_argmax2d(input, False)
        assert pytest.approx(coord[..., 0].item(), 2.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def test_batch2_n2(self):
        input = torch.zeros(2, 2, 2, 3)
        input[0, 0, 0, 0] = 10.  # top-left
        input[0, 1, 0, -1] = 10.  # top-right
        input[1, 0, -1, 0] = 10.  # bottom-left
        input[1, 1, -1, -1] = 10.  # bottom-right

        coord = kornia.spatial_soft_argmax2d(input)
        assert pytest.approx(coord[0, 0, 0].item(), -1.0)  # top-left
        assert pytest.approx(coord[0, 0, 1].item(), -1.0)
        assert pytest.approx(coord[0, 1, 0].item(), 1.0)  # top-right
        assert pytest.approx(coord[0, 1, 1].item(), -1.0)
        assert pytest.approx(coord[1, 0, 0].item(), -1.0)  # bottom-left
        assert pytest.approx(coord[1, 0, 1].item(), 1.0)
        assert pytest.approx(coord[1, 1, 0].item(), 1.0)  # bottom-right
        assert pytest.approx(coord[1, 1, 1].item(), 1.0)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor,
                      temperature: torch.Tensor,
                      normalize_coords: bool,
                      eps: float) -> torch.Tensor:
            return kornia.spatial_soft_argmax2d(
                input, temperature, normalize_coords, eps)

        input = torch.rand(1, 2, 3, 4)
        actual = op_script(input, torch.tensor(1.0), True, 1e-8)
        expected = kornia.spatial_soft_argmax2d(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self):
        input = torch.rand(2, 3, 3, 2)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.spatial_soft_argmax2d,
                         (input), raise_exception=True)


class TestDSNT:
    GAUSSIAN = torch.tensor([
        [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
        [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
        [0.021938, 0.098320, 0.162103, 0.098320, 0.021938],
        [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
        [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
    ])

    def test_render_gaussian_2d(self):
        expected = self.GAUSSIAN
        actual = kornia.geometry.dsnt.render_gaussian_2d(torch.tensor([2.0, 2.0]),
                                                         torch.tensor([1.0, 1.0]),
                                                         (5, 5),
                                                         normalized_coordinates=False)
        assert_allclose(actual, expected, rtol=0, atol=1e-5)

    def test_render_gaussian_2d_normalized_coordinates(self):
        expected = self.GAUSSIAN
        actual = kornia.geometry.dsnt.render_gaussian_2d(torch.tensor([0.0, 0.0]),
                                                         torch.tensor([0.25, 0.25]),
                                                         (5, 5),
                                                         normalized_coordinates=True)
        assert_allclose(actual, expected, rtol=0, atol=1e-5)

    @pytest.mark.parametrize('input', [
        torch.ones(1, 1, 5, 7),
        torch.randn(2, 3, 16, 16),
    ])
    def test_spatial_softmax_2d(self, input):
        actual = kornia.geometry.dsnt.spatial_softmax_2d(input)
        assert actual.lt(0).sum().item() == 0, 'expected no negative values'
        sums = actual.sum(-1).sum(-1)
        assert_allclose(sums, torch.ones_like(sums))

    @pytest.mark.parametrize('input,expected_norm,expected_px', [
        (
            torch.tensor([[[
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]]]),
            torch.tensor([[[1.0, -1.0]]]),
            torch.tensor([[[2.0, 0.0]]]),
        ),
    ])
    def test_spatial_softargmax_2d(self, input, expected_norm, expected_px):
        actual_norm = kornia.geometry.dsnt.spatial_softargmax_2d(input, True)
        assert_allclose(actual_norm, expected_norm)
        actual_px = kornia.geometry.dsnt.spatial_softargmax_2d(input, False)
        assert_allclose(actual_px, expected_px)

    def test_end_to_end(self):
        input = torch.full((1, 2, 7, 7), 1.0, requires_grad=True)
        target = torch.as_tensor([[[0.0, 0.0], [1.0, 1.0]]])
        std = torch.tensor([1.0, 1.0])

        hm = kornia.geometry.dsnt.spatial_softmax_2d(input)
        assert_allclose(hm.sum(-1).sum(-1), 1.0)

        pred = kornia.geometry.dsnt.spatial_softargmax_2d(hm)
        assert_allclose(pred, torch.as_tensor([[[0.0, 0.0], [0.0, 0.0]]]))

        loss1 = mse_loss(pred, target, size_average=None, reduce=None,
                         reduction='none').mean(-1, keepdim=False)
        expected_loss1 = torch.as_tensor([[0.0, 1.0]])
        assert_allclose(loss1, expected_loss1)

        target_hm = kornia.geometry.dsnt.render_gaussian_2d(target, std, input.shape[-2:])
        loss2 = kornia.losses.js_div_loss_2d(hm, target_hm, reduction='none')
        expected_loss2 = torch.as_tensor([[0.0087, 0.0818]])
        assert_allclose(loss2, expected_loss2, rtol=0, atol=1e-3)

        loss = (loss1 + loss2).mean()
        loss.backward()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        def op(input: torch.Tensor, target: torch.Tensor,
               temperature: torch.Tensor, normalized_coordinates: bool,
               std: torch.Tensor) -> torch.Tensor:
            hm = kornia.geometry.dsnt.spatial_softmax_2d(input, temperature)
            pred = kornia.geometry.dsnt.spatial_softargmax_2d(hm, normalized_coordinates)
            size = (input.shape[-2], input.shape[-1])
            target_hm = kornia.geometry.dsnt.render_gaussian_2d(target, std, size,
                                                                normalized_coordinates)
            loss1 = mse_loss(pred, target, size_average=None, reduce=None,
                             reduction='mean')
            loss2 = kornia.losses.js_div_loss_2d(hm, target_hm, 'mean')
            return loss1 + loss2

        op_script = torch.jit.script(op)

        args = [torch.rand(2, 3, 7, 7), torch.rand(2, 3, 2) * 7,
                torch.tensor(1.0), False, torch.tensor([1.0, 1.0])]
        actual = op_script(*args)
        expected = op(*args)
        assert_allclose(actual, expected)


class TestConvSoftArgmax2d:
    def test_smoke(self):
        input = torch.zeros(1, 1, 3, 3)
        m = kornia.ConvSoftArgmax2d((3, 3))
        assert m(input).shape == (1, 1, 2, 3, 3)

    def test_smoke_batch(self):
        input = torch.zeros(2, 5, 3, 3)
        m = kornia.ConvSoftArgmax2d()
        assert m(input).shape == (2, 5, 2, 3, 3)

    def test_smoke_with_val(self):
        input = torch.zeros(1, 1, 3, 3)
        m = kornia.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(input)
        assert coords.shape == (1, 1, 2, 3, 3)
        assert val.shape == (1, 1, 3, 3)

    def test_smoke_batch_with_val(self):
        input = torch.zeros(2, 5, 3, 3)
        m = kornia.ConvSoftArgmax2d((3, 3), output_value=True)
        coords, val = m(input)
        assert coords.shape == (2, 5, 2, 3, 3)
        assert val.shape == (2, 5, 3, 3)

    def test_gradcheck(self):
        input = torch.rand(2, 3, 5, 5)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.conv_soft_argmax2d,
                         (input), raise_exception=True)

    def test_cold_diag(self):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]])
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[1., 0.],
                                       [0., 1.]]]])
        expected_coord = torch.tensor([[[[[1., 3.],
                                          [1., 3.]],
                                         [[1., 1.],
                                          [3., 3.]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag(self):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]])
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=10.,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[0.1214, 0.],
                                       [0., 0.1214]]]])
        expected_coord = torch.tensor([[[[[1., 3.],
                                          [1., 3.]],
                                         [[1., 1.],
                                          [3., 3.]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_cold_diag_norm(self):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]])
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[1., 0.],
                                       [0., 1.]]]])
        expected_coord = torch.tensor([[[[[-0.5, 0.5],
                                          [-0.5, 0.5]],
                                         [[-0.5, -0.5],
                                          [0.5, 0.5]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag_norm(self):
        input = torch.tensor([[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]])
        softargmax = kornia.ConvSoftArgmax2d((3, 3), (2, 2), (0, 0),
                                             temperature=10.,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[0.1214, 0.],
                                       [0., 0.1214]]]])
        expected_coord = torch.tensor([[[[[-0.5, 0.5],
                                          [-0.5, 0.5]],
                                         [[-0.5, -0.5],
                                          [0.5, 0.5]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)


class TestConvSoftArgmax3d:
    def test_smoke(self):
        input = torch.zeros(1, 1, 3, 3, 3)
        m = kornia.ConvSoftArgmax3d((3, 3, 3), output_value=False)
        assert m(input).shape == (1, 1, 3, 3, 3, 3)

    def test_smoke_with_val(self):
        input = torch.zeros(1, 1, 3, 3, 3)
        m = kornia.ConvSoftArgmax3d((3, 3, 3), output_value=True)
        coords, val = m(input)
        assert coords.shape == (1, 1, 3, 3, 3, 3)
        assert val.shape == (1, 1, 3, 3, 3)

    def test_gradcheck(self):
        input = torch.rand(1, 2, 3, 5, 5)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.conv_soft_argmax3d,
                         (input), raise_exception=True)

    def test_cold_diag(self):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]])
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[[1., 0.],
                                        [0., 1.]]]]])
        expected_coord = torch.tensor([[[
                                        [[[0., 0.],
                                          [0., 0.]]],
                                        [[[1., 3.],
                                          [1., 3.]]],
                                        [[[1., 1.],
                                          [3., 3.]]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag(self):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]])
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=10.,
                                             normalized_coordinates=False,
                                             output_value=True)
        expected_val = torch.tensor([[[[[0.1214, 0.],
                                        [0., 0.1214]]]]])
        expected_coord = torch.tensor([[[
                                        [[[0., 0.],
                                          [0., 0.]]],
                                        [[[1., 3.],
                                          [1., 3.]]],
                                        [[[1., 1.],
                                          [3., 3.]]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_cold_diag_norm(self):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]])
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=0.05,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[[1., 0.],
                                        [0., 1.]]]]])
        expected_coord = torch.tensor([[[
            [[[-1., -1.],
              [-1., -1.]]],
            [[[-0.5, 0.5],
              [-0.5, 0.5]]],
            [[[-0.5, -0.5],
              [0.5, 0.5]]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)

    def test_hot_diag_norm(self):
        input = torch.tensor([[[[
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
        ]]]])
        softargmax = kornia.ConvSoftArgmax3d((1, 3, 3), (1, 2, 2), (0, 0, 0),
                                             temperature=10.,
                                             normalized_coordinates=True,
                                             output_value=True)
        expected_val = torch.tensor([[[[[0.1214, 0.],
                                        [0., 0.1214]]]]])
        expected_coord = torch.tensor([[[
            [[[-1., -1.],
              [-1., -1.]]],
            [[[-0.5, 0.5],
              [-0.5, 0.5]]],
            [[[-0.5, -0.5],
              [0.5, 0.5]]]]]])
        coords, val = softargmax(input)
        assert_allclose(val, expected_val)
        assert_allclose(coords, expected_coord)
