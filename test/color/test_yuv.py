import pytest

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import BaseTester

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestRgbToYuv(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_yuv(img), torch.Tensor)

    @pytest.mark.parametrize(
        "shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_yuv(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_yuv([0.])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_yuv(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_yuv(img)

    # TODO: investigate and implement me
    # def test_unit(self, device, dtype):
    #    pass

    # TODO: improve accuracy
    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        yuv = kornia.color.rgb_to_yuv
        rgb = kornia.color.yuv_to_rgb

        data_out = rgb(yuv(data))
        assert_allclose(data_out, data, rtol=1e-2, atol=1e-2)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_yuv, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_yuv
        op_jit = torch.jit.script(op)
        assert_allclose(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToYuv().to(device, dtype)
        fcn = kornia.color.rgb_to_yuv
        assert_allclose(ops(img), fcn(img))


class TestYuvToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.yuv_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize(
        "shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.yuv_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.yuv_to_rgb([0.])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.yuv_to_rgb(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.yuv_to_rgb(img)

    # TODO: investigate and implement me
    # def test_unit(self, device, dtype):
    #    pass

    # TODO: improve accuracy
    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        rgb = kornia.color.yuv_to_rgb
        yuv = kornia.color.rgb_to_yuv

        data_out = rgb(yuv(data))
        assert_allclose(data_out, data, rtol=1e-2, atol=1e-2)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.yuv_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.yuv_to_rgb
        op_jit = torch.jit.script(op)
        assert_allclose(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.YuvToRgb().to(device, dtype)
        fcn = kornia.color.yuv_to_rgb
        assert_allclose(ops(img), fcn(img))
