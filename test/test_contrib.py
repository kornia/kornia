import pytest
import torch
from torch.autograd import gradcheck
from torch.nn.functional import mse_loss
from torch.testing import assert_allclose

import kornia
import kornia.contrib.dsnt as dsnt
import kornia.testing as utils  # test utils


class TestMaxBlurPool2d:
    def test_shape(self):
        input = torch.rand(1, 2, 4, 6)
        pool = kornia.contrib.MaxBlurPool2d(kernel_size=3)
        assert pool(input).shape == (1, 2, 2, 3)

    def test_shape_batch(self):
        input = torch.rand(3, 2, 6, 10)
        pool = kornia.contrib.MaxBlurPool2d(kernel_size=5)
        assert pool(input).shape == (3, 2, 3, 5)

    def test_gradcheck(self):
        input = torch.rand(2, 3, 4, 4)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.contrib.max_blur_pool2d,
                         (input, 3,), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
            return kornia.contrib.max_blur_pool2d(input, kernel_size)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img, kernel_size=3)
        expected = kornia.contrib.max_blur_pool2d(img, kernel_size=3)
        assert_allclose(actual, expected)


class TestExtractTensorPatches:
    def test_smoke(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(3)
        assert m(input).shape == (1, 4, 1, 3, 3)

    def test_b1_ch1_h4w4_ws3(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(input)
        assert patches.shape == (1, 4, 1, 3, 3)
        assert_allclose(input[0, :, :3, :3], patches[0, 0])
        assert_allclose(input[0, :, :3, 1:], patches[0, 1])
        assert_allclose(input[0, :, 1:, :3], patches[0, 2])
        assert_allclose(input[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch2_h4w4_ws3(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        input = input.expand(-1, 2, -1, -1)  # copy all channels
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(input)
        assert patches.shape == (1, 4, 2, 3, 3)
        assert_allclose(input[0, :, :3, :3], patches[0, 0])
        assert_allclose(input[0, :, :3, 1:], patches[0, 1])
        assert_allclose(input[0, :, 1:, :3], patches[0, 2])
        assert_allclose(input[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch1_h4w4_ws2(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2)
        patches = m(input)
        assert patches.shape == (1, 9, 1, 2, 2)
        assert_allclose(input[0, :, 0:2, 1:3], patches[0, 1])
        assert_allclose(input[0, :, 0:2, 2:4], patches[0, 2])
        assert_allclose(input[0, :, 1:3, 1:3], patches[0, 4])
        assert_allclose(input[0, :, 2:4, 1:3], patches[0, 7])

    def test_b1_ch1_h4w4_ws2_stride2(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2, stride=2)
        patches = m(input)
        assert patches.shape == (1, 4, 1, 2, 2)
        assert_allclose(input[0, :, 0:2, 0:2], patches[0, 0])
        assert_allclose(input[0, :, 0:2, 2:4], patches[0, 1])
        assert_allclose(input[0, :, 2:4, 0:2], patches[0, 2])
        assert_allclose(input[0, :, 2:4, 2:4], patches[0, 3])

    def test_b1_ch1_h4w4_ws2_stride21(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2, stride=(2, 1))
        patches = m(input)
        assert patches.shape == (1, 6, 1, 2, 2)
        assert_allclose(input[0, :, 0:2, 1:3], patches[0, 1])
        assert_allclose(input[0, :, 0:2, 2:4], patches[0, 2])
        assert_allclose(input[0, :, 2:4, 0:2], patches[0, 3])
        assert_allclose(input[0, :, 2:4, 2:4], patches[0, 5])

    def test_b1_ch1_h3w3_ws2_stride1_padding1(self):
        input = torch.arange(9.).view(1, 1, 3, 3)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(input)
        assert patches.shape == (1, 16, 1, 2, 2)
        assert_allclose(input[0, :, 0:2, 0:2], patches[0, 5])
        assert_allclose(input[0, :, 0:2, 1:3], patches[0, 6])
        assert_allclose(input[0, :, 1:3, 0:2], patches[0, 9])
        assert_allclose(input[0, :, 1:3, 1:3], patches[0, 10])

    def test_b2_ch1_h3w3_ws2_stride1_padding1(self):
        batch_size = 2
        input = torch.arange(9.).view(1, 1, 3, 3)
        input = input.expand(batch_size, -1, -1, -1)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(input)
        assert patches.shape == (batch_size, 16, 1, 2, 2)
        for i in range(batch_size):
            assert_allclose(
                input[i, :, 0:2, 0:2], patches[i, 5])
            assert_allclose(
                input[i, :, 0:2, 1:3], patches[i, 6])
            assert_allclose(
                input[i, :, 1:3, 0:2], patches[i, 9])
            assert_allclose(
                input[i, :, 1:3, 1:3], patches[i, 10])

    def test_b1_ch1_h3w3_ws23(self):
        input = torch.arange(9.).view(1, 1, 3, 3)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(input)
        assert patches.shape == (1, 2, 1, 2, 3)
        assert_allclose(input[0, :, 0:2, 0:3], patches[0, 0])
        assert_allclose(input[0, :, 1:3, 0:3], patches[0, 1])

    def test_b1_ch1_h3w4_ws23(self):
        input = torch.arange(12.).view(1, 1, 3, 4)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(input)
        assert patches.shape == (1, 4, 1, 2, 3)
        assert_allclose(input[0, :, 0:2, 0:3], patches[0, 0])
        assert_allclose(input[0, :, 0:2, 1:4], patches[0, 1])
        assert_allclose(input[0, :, 1:3, 0:3], patches[0, 2])
        assert_allclose(input[0, :, 1:3, 1:4], patches[0, 3])

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input: torch.Tensor, height: int,
                      width: int) -> torch.Tensor:
            return kornia.denormalize_pixel_coordinates(input, height, width)
        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(
            height, width, normalized_coordinates=True)

        actual = op_script(grid, height, width)
        expected = kornia.denormalize_pixel_coordinates(
            grid, height, width)

        assert_allclose(actual, expected)

    def test_gradcheck(self):
        input = torch.rand(2, 3, 4, 4)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.contrib.extract_tensor_patches,
                         (input, 3,), raise_exception=True)


class TestSpatialSoftArgmax2d:
    def test_smoke(self):
        input = torch.zeros(1, 1, 2, 3)
        m = kornia.contrib.SpatialSoftArgmax2d()
        assert m(input).shape == (1, 1, 2)

    def test_smoke_batch(self):
        input = torch.zeros(2, 1, 2, 3)
        m = kornia.contrib.SpatialSoftArgmax2d()
        assert m(input).shape == (2, 1, 2)

    def test_top_left(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = kornia.contrib.spatial_soft_argmax2d(input, True)
        assert pytest.approx(coord[..., 0].item(), -1.0)
        assert pytest.approx(coord[..., 1].item(), -1.0)

    def test_top_left_normalized(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = kornia.contrib.spatial_soft_argmax2d(input, False)
        assert pytest.approx(coord[..., 0].item(), 0.0)
        assert pytest.approx(coord[..., 1].item(), 0.0)

    def test_bottom_right(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = kornia.contrib.spatial_soft_argmax2d(input, True)
        assert pytest.approx(coord[..., 0].item(), 1.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def test_bottom_right_normalized(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = kornia.contrib.spatial_soft_argmax2d(input, False)
        assert pytest.approx(coord[..., 0].item(), 2.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def test_batch2_n2(self):
        input = torch.zeros(2, 2, 2, 3)
        input[0, 0, 0, 0] = 10.  # top-left
        input[0, 1, 0, -1] = 10.  # top-right
        input[1, 0, -1, 0] = 10.  # bottom-left
        input[1, 1, -1, -1] = 10.  # bottom-right

        coord = kornia.contrib.spatial_soft_argmax2d(input)
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
        assert gradcheck(kornia.contrib.spatial_soft_argmax2d,
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
        actual = dsnt.render_gaussian_2d(torch.tensor([2.0, 2.0]),
                                         torch.tensor([1.0, 1.0]),
                                         (5, 5),
                                         normalized_coordinates=False)
        assert_allclose(actual, expected, rtol=0, atol=1e-5)

    def test_render_gaussian_2d_normalized_coordinates(self):
        expected = self.GAUSSIAN
        actual = dsnt.render_gaussian_2d(torch.tensor([0.0, 0.0]),
                                         torch.tensor([0.25, 0.25]),
                                         (5, 5),
                                         normalized_coordinates=True)
        assert_allclose(actual, expected, rtol=0, atol=1e-5)

    @pytest.mark.parametrize('input', [
        torch.ones(1, 1, 5, 7),
        torch.randn(2, 3, 16, 16),
    ])
    def test_spatial_softmax_2d(self, input):
        actual = dsnt.spatial_softmax_2d(input)
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
        actual_norm = dsnt.spatial_softargmax_2d(input, True)
        assert_allclose(actual_norm, expected_norm)
        actual_px = dsnt.spatial_softargmax_2d(input, False)
        assert_allclose(actual_px, expected_px)

    def test_end_to_end(self):
        input = torch.full((1, 2, 7, 7), 1.0, requires_grad=True)
        target = torch.as_tensor([[[0.0, 0.0], [1.0, 1.0]]])
        std = torch.tensor([1.0, 1.0])

        hm = dsnt.spatial_softmax_2d(input)
        assert_allclose(hm.sum(-1).sum(-1), 1.0)

        pred = dsnt.spatial_softargmax_2d(hm)
        assert_allclose(pred, torch.as_tensor([[[0.0, 0.0], [0.0, 0.0]]]))

        loss1 = mse_loss(pred, target, size_average=None, reduce=None,
                         reduction='none').mean(-1, keepdim=False)
        expected_loss1 = torch.as_tensor([[0.0, 1.0]])
        assert_allclose(loss1, expected_loss1)

        target_hm = dsnt.render_gaussian_2d(target, std, input.shape[-2:])
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
            hm = dsnt.spatial_softmax_2d(input, temperature)
            pred = dsnt.spatial_softargmax_2d(hm, normalized_coordinates)
            size = (input.shape[-2], input.shape[-1])
            target_hm = dsnt.render_gaussian_2d(target, std, size,
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
