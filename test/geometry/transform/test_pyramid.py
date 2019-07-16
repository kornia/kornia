import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestPyrUp:
    def test_shape(self):
        inp = torch.zeros(1, 2, 4, 4)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (1, 2, 8, 8)

    def test_shape_batch(self):
        inp = torch.zeros(2, 2, 4, 4)
        pyr = kornia.geometry.PyrUp()
        assert pyr(inp).shape == (2, 2, 8, 8)

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.geometry.pyrup, (img,), raise_exception=True)

    @pytest.mark.skip("")
    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.geometry.pyrup(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.geometry.pyrup(img)
        assert_allclose(actual, expected)


class TestPyrDown:
    def test_shape(self):
        inp = torch.zeros(1, 2, 4, 4)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (1, 2, 2, 2)

    def test_shape_batch(self):
        inp = torch.zeros(2, 2, 4, 4)
        pyr = kornia.geometry.PyrDown()
        assert pyr(inp).shape == (2, 2, 2, 2)

    def test_symmetry_preserving(self):
        inp = torch.zeros(1, 1, 6, 6)
        inp[:, :, 2:4, 2:4] = 1.0
        pyr_out = kornia.geometry.PyrDown()(inp).squeeze()
        assert torch.allclose(pyr_out, pyr_out.flip(0))
        assert torch.allclose(pyr_out, pyr_out.flip(1))

    def test_gradcheck(self):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.geometry.pyrdown, (img,), raise_exception=True)

    @pytest.mark.skip("")
    def test_jit(self):
        @torch.jit.script
        def op_script(input):
            return kornia.geometry.pyrdown(input)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img)
        expected = kornia.geometry.pyrdown(img)
        assert_allclose(actual, expected)


class TestScalePyramid:
    def test_shape_tuple(self):
        inp = torch.zeros(3, 2, 6, 6)
        SP = kornia.geometry.ScalePyramid(n_levels=1)
        out = SP(inp)
        assert len(out) == 3
        assert len(out[0]) == 1
        assert len(out[1]) == 1
        assert len(out[2]) == 1

    def test_shape_batch(self):
        inp = torch.zeros(3, 2, 6, 6)
        SP = kornia.geometry.ScalePyramid(n_levels=1)
        sp, sigmas, pd = SP(inp)
        assert sp[0].shape == (3, 1, 2, 6, 6)

    def test_n_levels_shape(self):
        inp = torch.zeros(1, 1, 6, 6)
        SP = kornia.geometry.ScalePyramid(n_levels=5)
        sp, sigmas, pd = SP(inp)
        assert sp[0].shape == (1, 5, 1, 6, 6)

    def test_blur_order(self):
        inp = torch.rand(1, 1, 12, 12)
        SP = kornia.geometry.ScalePyramid(n_levels=5)
        sp, sigmas, pd = SP(inp)
        for i, pyr_level in enumerate(sp):
            for ii, img in enumerate(pyr_level):
                img = img.squeeze().view(5, -1)
                max_per_blur_level_val, _ = img.max(dim=1)
                assert torch.argmax(max_per_blur_level_val).item() == 0
        return

    def test_symmetry_preserving(self):
        inp = torch.zeros(1, 1, 12, 12)
        inp[0, 0, 4:8, 4:8] = 1.0
        SP = kornia.geometry.ScalePyramid(n_levels=5)
        sp, sigmas, pd = SP(inp)
        for i, pyr_level in enumerate(sp):
            for ii, img in enumerate(pyr_level):
                img = img.squeeze()
                assert torch.allclose(img, img.flip(1))
                assert torch.allclose(img, img.flip(2))
        return
