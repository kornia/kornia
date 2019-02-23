import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils


class TestExtractTensorPatches:
    def _test_smoke(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = tgm.contrib.ExtractTensorPatches(3)
        assert m(input).shape == (1, 4, 1, 3, 3)

    def _test_b1_ch1_h4w4_ws3(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = tgm.contrib.ExtractTensorPatches(3)
        patches = m(input)
        assert patches.shape == (1, 4, 1, 3, 3)
        assert utils.check_equal_torch(input[0, :, :3, :3], patches[0, 0])
        assert utils.check_equal_torch(input[0, :, :3, 1:], patches[0, 1])
        assert utils.check_equal_torch(input[0, :, 1:, :3], patches[0, 2])
        assert utils.check_equal_torch(input[0, :, 1:, 1:], patches[0, 3])

    def _test_b1_ch2_h4w4_ws3(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        input = input.expand(-1, 2, -1, -1)  # copy all channels
        m = tgm.contrib.ExtractTensorPatches(3)
        patches = m(input)
        assert patches.shape == (1, 4, 2, 3, 3)
        assert utils.check_equal_torch(input[0, :, :3, :3], patches[0, 0])
        assert utils.check_equal_torch(input[0, :, :3, 1:], patches[0, 1])
        assert utils.check_equal_torch(input[0, :, 1:, :3], patches[0, 2])
        assert utils.check_equal_torch(input[0, :, 1:, 1:], patches[0, 3])

    def _test_b1_ch1_h4w4_ws2(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = tgm.contrib.ExtractTensorPatches(2)
        patches = m(input)
        assert patches.shape == (1, 9, 1, 2, 2)
        assert utils.check_equal_torch(input[0, :, 0:2, 1:3], patches[0, 1])
        assert utils.check_equal_torch(input[0, :, 0:2, 2:4], patches[0, 2])
        assert utils.check_equal_torch(input[0, :, 1:3, 1:3], patches[0, 4])
        assert utils.check_equal_torch(input[0, :, 2:4, 1:3], patches[0, 7])

    def _test_b1_ch1_h4w4_ws2_stride2(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = tgm.contrib.ExtractTensorPatches(2, stride=2)
        patches = m(input)
        assert patches.shape == (1, 4, 1, 2, 2)
        assert utils.check_equal_torch(input[0, :, 0:2, 0:2], patches[0, 0])
        assert utils.check_equal_torch(input[0, :, 0:2, 2:4], patches[0, 1])
        assert utils.check_equal_torch(input[0, :, 2:4, 0:2], patches[0, 2])
        assert utils.check_equal_torch(input[0, :, 2:4, 2:4], patches[0, 3])

    def _test_b1_ch1_h4w4_ws2_stride21(self):
        input = torch.arange(16.).view(1, 1, 4, 4)
        m = tgm.contrib.ExtractTensorPatches(2, stride=(2, 1))
        patches = m(input)
        assert patches.shape == (1, 6, 1, 2, 2)
        assert utils.check_equal_torch(input[0, :, 0:2, 1:3], patches[0, 1])
        assert utils.check_equal_torch(input[0, :, 0:2, 2:4], patches[0, 2])
        assert utils.check_equal_torch(input[0, :, 2:4, 0:2], patches[0, 3])
        assert utils.check_equal_torch(input[0, :, 2:4, 2:4], patches[0, 5])

    def _test_b1_ch1_h3w3_ws2_stride1_padding1(self):
        input = torch.arange(9.).view(1, 1, 3, 3)
        m = tgm.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(input)
        assert patches.shape == (1, 16, 1, 2, 2)
        assert utils.check_equal_torch(input[0, :, 0:2, 0:2], patches[0, 5])
        assert utils.check_equal_torch(input[0, :, 0:2, 1:3], patches[0, 6])
        assert utils.check_equal_torch(input[0, :, 1:3, 0:2], patches[0, 9])
        assert utils.check_equal_torch(input[0, :, 1:3, 1:3], patches[0, 10])

    def _test_b2_ch1_h3w3_ws2_stride1_padding1(self):
        batch_size = 2
        input = torch.arange(9.).view(1, 1, 3, 3)
        input = input.expand(batch_size, -1, -1, -1)
        m = tgm.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(input)
        assert patches.shape == (batch_size, 16, 1, 2, 2)
        for i in range(batch_size):
            assert utils.check_equal_torch(
                input[i, :, 0:2, 0:2], patches[i, 5])
            assert utils.check_equal_torch(
                input[i, :, 0:2, 1:3], patches[i, 6])
            assert utils.check_equal_torch(
                input[i, :, 1:3, 0:2], patches[i, 9])
            assert utils.check_equal_torch(
                input[i, :, 1:3, 1:3], patches[i, 10])

    def _test_b1_ch1_h3w3_ws23(self):
        input = torch.arange(9.).view(1, 1, 3, 3)
        m = tgm.contrib.ExtractTensorPatches((2, 3))
        patches = m(input)
        assert patches.shape == (1, 2, 1, 2, 3)
        assert utils.check_equal_torch(input[0, :, 0:2, 0:3], patches[0, 0])
        assert utils.check_equal_torch(input[0, :, 1:3, 0:3], patches[0, 1])

    def _test_b1_ch1_h3w4_ws23(self):
        input = torch.arange(12.).view(1, 1, 3, 4)
        m = tgm.contrib.ExtractTensorPatches((2, 3))
        patches = m(input)
        assert patches.shape == (1, 4, 1, 2, 3)
        assert utils.check_equal_torch(input[0, :, 0:2, 0:3], patches[0, 0])
        assert utils.check_equal_torch(input[0, :, 0:2, 1:4], patches[0, 1])
        assert utils.check_equal_torch(input[0, :, 1:3, 0:3], patches[0, 2])
        assert utils.check_equal_torch(input[0, :, 1:3, 1:4], patches[0, 3])

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        input = torch.rand(2, 3, 4, 4)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(tgm.contrib.extract_tensor_patches,
                         (input, 3,), raise_exception=True)

    def test_run_all(self):
        self._test_smoke()
        self._test_b1_ch1_h4w4_ws3()
        self._test_b1_ch2_h4w4_ws3()
        self._test_b1_ch1_h4w4_ws2()
        self._test_b1_ch1_h3w3_ws23()
        self._test_b1_ch1_h3w4_ws23()
        self._test_b1_ch1_h4w4_ws2_stride2()
        self._test_b1_ch1_h4w4_ws2_stride21()
        self._test_b1_ch1_h3w3_ws2_stride1_padding1()
        self._test_b2_ch1_h3w3_ws2_stride1_padding1()
        self._test_gradcheck()


class TestSoftArgmax2d:
    def _test_smoke(self):
        input = torch.zeros(1, 1, 2, 3)
        m = tgm.contrib.SpatialSoftArgmax2d()
        assert m(input).shape == (1, 1, 2)

    def _test_top_left(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = tgm.contrib.spatial_soft_argmax2d(input, True)
        assert pytest.approx(coord[..., 0].item(), -1.0)
        assert pytest.approx(coord[..., 1].item(), -1.0)

    def _test_top_left_normalized(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = tgm.contrib.spatial_soft_argmax2d(input, False)
        assert pytest.approx(coord[..., 0].item(), 0.0)
        assert pytest.approx(coord[..., 1].item(), 0.0)

    def _test_bottom_right(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = tgm.contrib.spatial_soft_argmax2d(input, True)
        assert pytest.approx(coord[..., 0].item(), 1.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def _test_bottom_right_normalized(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = tgm.contrib.spatial_soft_argmax2d(input, False)
        assert pytest.approx(coord[..., 0].item(), 2.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def _test_batch2_n2(self):
        input = torch.zeros(2, 2, 2, 3)
        input[0, 0, 0, 0] = 10.  # top-left
        input[0, 1, 0, -1] = 10.  # top-right
        input[1, 0, -1, 0] = 10.  # bottom-left
        input[1, 1, -1, -1] = 10.  # bottom-right

        coord = tgm.contrib.spatial_soft_argmax2d(input)
        assert pytest.approx(coord[0, 0, 0].item(), -1.0)  # top-left
        assert pytest.approx(coord[0, 0, 1].item(), -1.0)
        assert pytest.approx(coord[0, 1, 0].item(), 1.0)  # top-right
        assert pytest.approx(coord[0, 1, 1].item(), -1.0)
        assert pytest.approx(coord[1, 0, 0].item(), -1.0)  # bottom-left
        assert pytest.approx(coord[1, 0, 1].item(), 1.0)
        assert pytest.approx(coord[1, 1, 0].item(), 1.0)  # bottom-right
        assert pytest.approx(coord[1, 1, 1].item(), 1.0)

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        input = torch.rand(2, 3, 3, 2)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(tgm.contrib.spatial_soft_argmax2d,
                         (input), raise_exception=True)

    def test_run_all(self):
        self._test_smoke()
        self._test_top_left()
        self._test_top_left_normalized()
        self._test_bottom_right()
        self._test_bottom_right_normalized()
        self._test_batch2_n2()
        self._test_gradcheck()
