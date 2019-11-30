import pytest

import kornia as kornia
import kornia.testing as utils  # test utils
from test.common import device

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestPyrUp:
    def test_shape(self, device):
        inp = torch.zeros(1, 2, 4, 4).to(device)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (1, 2, 8, 8)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 2, 4, 4).to(device)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (2, 2, 8, 8)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        # TODO: cuda test is not working
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.geometry.pyrup, (img,), raise_exception=True)

    @pytest.mark.skip("")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(input):
            return kornia.geometry.pyrup(input)
        img = torch.rand(2, 3, 4, 5).to(device)
        actual = op_script(img)
        expected = kornia.geometry.pyrup(img)
        assert_allclose(actual, expected)


class TestPyrDown:
    def test_shape(self, device):
        inp = torch.zeros(1, 2, 4, 4).to(device)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (1, 2, 2, 2)

    def test_shape_batch(self, device):
        inp = torch.zeros(2, 2, 4, 4).to(device)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (2, 2, 2, 2)

    def test_symmetry_preserving(self, device):
        inp = torch.zeros(1, 1, 6, 6).to(device)
        inp[:, :, 2:4, 2:4] = 1.0
        pyr_out = kornia.geometry.PyrDown()(inp).squeeze()
        assert torch.allclose(pyr_out, pyr_out.flip(0))
        assert torch.allclose(pyr_out, pyr_out.flip(1))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.geometry.pyrdown, (img,), raise_exception=True)

    @pytest.mark.skip("")
    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(input):
            return kornia.geometry.pyrdown(input)
        img = torch.rand(2, 3, 4, 5).to(device)
        actual = op_script(img)
        expected = kornia.geometry.pyrdown(img)
        assert_allclose(actual, expected)


class TestScalePyramid:
    def test_shape_tuple(self, device):
        inp = torch.zeros(3, 2, 6, 6).to(device)
        SP = kornia.geometry.ScalePyramid(n_levels=1)
        out = SP(inp)
        assert len(out) == 3
        assert len(out[0]) == 1
        assert len(out[1]) == 1
        assert len(out[2]) == 1

    def test_shape_batch(self, device):
        inp = torch.zeros(3, 2, 6, 6).to(device)
        SP = kornia.geometry.ScalePyramid(n_levels=1)
        sp, sigmas, pd = SP(inp)
        assert sp[0].shape == (3, 1, 2, 6, 6)

    def test_n_levels_shape(self, device):
        inp = torch.zeros(1, 1, 6, 6).to(device)
        SP = kornia.geometry.ScalePyramid(n_levels=5)
        sp, sigmas, pd = SP(inp)
        assert sp[0].shape == (1, 5, 1, 6, 6)

    def test_blur_order(self, device):
        inp = torch.rand(1, 1, 12, 12).to(device)
        SP = kornia.geometry.ScalePyramid(n_levels=5)
        sp, sigmas, pd = SP(inp)
        for i, pyr_level in enumerate(sp):
            for ii, img in enumerate(pyr_level):
                img = img.squeeze().view(5, -1)
                max_per_blur_level_val, _ = img.max(dim=1)
                assert torch.argmax(max_per_blur_level_val).item() == 0
        return

    def test_symmetry_preserving(self, device):
        inp = torch.zeros(1, 1, 12, 12).to(device)
        inp[0, 0, 4:8, 4:8] = 1.0
        SP = kornia.geometry.ScalePyramid(n_levels=5)
        sp, sigmas, pd = SP(inp)
        for i, pyr_level in enumerate(sp):
            for ii, img in enumerate(pyr_level):
                img = img.squeeze()
                assert torch.allclose(img, img.flip(1))
                assert torch.allclose(img, img.flip(2))
        return

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 7, 9
        # TODO: cuda test is not working
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        from kornia.geometry import ScalePyramid as SP

        def sp_tuple(img):
            sp, sigmas, pd = SP()(img)
            return tuple(sp)
        assert gradcheck(sp_tuple, (img,), raise_exception=True)


class TestBuildPyramid:
    def test_smoke(self, device):
        input = torch.ones(1, 2, 4, 5).to(device)
        pyramid = kornia.build_pyramid(input, max_level=1)
        assert len(pyramid) == 1
        assert pyramid[0].shape == (1, 2, 4, 5)

    @pytest.mark.parametrize("batch_size", (1, 2, 3))
    @pytest.mark.parametrize("channels", (1, 3))
    @pytest.mark.parametrize("max_level", (2, 3, 4))
    def test_num_levels(self, device, batch_size, channels, max_level):
        height, width = 16, 20
        input = torch.rand(batch_size, channels, height, width).to(device)
        pyramid = kornia.build_pyramid(input, max_level)
        assert len(pyramid) == max_level
        for i in range(1, max_level):
            img = pyramid[i]
            denom = 2 ** i
            expected_shape = (batch_size, channels, height // denom, width // denom)
            assert img.shape == expected_shape

    def test_gradcheck(self, device):
        max_level = 1
        batch_size, channels, height, width = 1, 2, 7, 9
        img = torch.rand(batch_size, channels, height, width).to(device)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.build_pyramid, (img, max_level,), raise_exception=True)
