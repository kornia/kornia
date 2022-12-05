import kornia.color as color
import kornia.color.via_conv as color_via_conv
import pytest
import torch

from kornia.testing import BaseTester
from torch.autograd import gradcheck


class TestRgbToYuv(BaseTester):
    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(color_via_conv.rgb_to_yuv(img)[0], torch.Tensor)
        assert isinstance(color_via_conv.rgb_to_yuv(img)[1], torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert color_via_conv.rgb_to_yuv(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.rgb_to_yuv([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv(img)

    def test_unit(self, device, dtype):
        data = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        self.assert_close(color.rgb_to_yuv(data), color_via_conv.rgb_to_yuv(data))

    # TODO: improve accuracy
    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        yuv = color_via_conv.rgb_to_yuv
        rgb = color_via_conv.yuv_to_rgb

        data_out = rgb(yuv(data))
        self.assert_close(data_out, data, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.rgb_to_yuv, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestRgbToYuv420(BaseTester):
    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 6
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(color_via_conv.rgb_to_yuv420(img)[0], torch.Tensor)
        assert isinstance(color_via_conv.rgb_to_yuv420(img)[1], torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-2] /= 2
        shapeuv[-1] /= 2
        assert color_via_conv.rgb_to_yuv420(img)[0].shape == tuple(shapey)
        assert color_via_conv.rgb_to_yuv420(img)[1].shape == tuple(shapeuv)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.rgb_to_yuv420([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv420(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv420(img)

        # dimensionality test
        with pytest.raises(ValueError):
            img = torch.ones(3, 2, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv420(img)

        # dimensionality test
        with pytest.raises(ValueError):
            img = torch.ones(3, 1, 2, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv420(img)

    # Test max/min values. This is essentially testing the transform rather than the subsampling
    # ref values manually checked vs rec 601
    def test_unit_white(self, device, dtype):  # skipcq: PYL-R0201
        rgb = (
            torch.tensor(
                [[[255, 255], [255, 255]], [[255, 255], [255, 255]], [[255, 255], [255, 255]]],
                device=device,
                dtype=torch.uint8,
            ).type(dtype)
            / 255.0
        )
        refy = torch.tensor([[[255, 255], [255, 255]]], device=device, dtype=torch.uint8)
        refuv = torch.tensor([[[0]], [[0]]], device=device, dtype=torch.int8)

        resy = (color_via_conv.rgb_to_yuv420(rgb)[0] * 255.0).round().type(torch.uint8)
        resuv = (color_via_conv.rgb_to_yuv420(rgb)[1] * 255.0).round().clamp(-128, 127).type(torch.int8)
        self.assert_close(refy, resy)
        self.assert_close(refuv, resuv)

    def test_unit_black(self, device, dtype):  # skipcq: PYL-R0201
        rgb = (
            torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], device=device, dtype=torch.uint8).type(
                dtype
            )
            / 255.0
        )
        refy = torch.tensor([[[0, 0], [0, 0]]], device=device, dtype=torch.uint8)
        refuv = torch.tensor([[[0]], [[0]]], device=device, dtype=torch.int8)

        resy = (color_via_conv.rgb_to_yuv420(rgb)[0] * 255.0).round().type(torch.uint8)
        resuv = (color_via_conv.rgb_to_yuv420(rgb)[1] * 255.0).round().clamp(-128, 127).type(torch.int8)
        self.assert_close(refy, resy)
        self.assert_close(refuv, resuv)

    def test_unit_gray(self, device, dtype):  # skipcq: PYL-R0201
        rgb = (
            torch.tensor(
                [[[127, 127], [127, 127]], [[127, 127], [127, 127]], [[127, 127], [127, 127]]],
                device=device,
                dtype=torch.uint8,
            ).type(dtype)
            / 255.0
        )
        refy = torch.tensor([[[127, 127], [127, 127]]], device=device, dtype=torch.uint8)
        refuv = torch.tensor([[[0]], [[0]]], device=device, dtype=torch.int8)

        resy = (color_via_conv.rgb_to_yuv420(rgb)[0] * 255.0).round().type(torch.uint8)
        resuv = (color_via_conv.rgb_to_yuv420(rgb)[1] * 255.0).round().clamp(-128, 127).type(torch.int8)
        self.assert_close(refy, resy)
        self.assert_close(refuv, resuv)

    def test_unit_red(self, device, dtype):  # skipcq: PYL-R0201
        rgb = (
            torch.tensor(
                [[[255, 255], [255, 255]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], device=device, dtype=torch.uint8
            ).type(dtype)
            / 255.0
        )
        refy = torch.tensor([[[76, 76], [76, 76]]], device=device, dtype=torch.uint8)
        refuv = torch.tensor([[[-37]], [[127]]], device=device, dtype=torch.int8)

        resy = (color_via_conv.rgb_to_yuv420(rgb)[0] * 255.0).round().type(torch.uint8)
        resuv = (color_via_conv.rgb_to_yuv420(rgb)[1] * 255.0).round().clamp(-128, 127).type(torch.int8)
        self.assert_close(refy, resy)
        self.assert_close(refuv, resuv)

    def test_unit_blue(self, device, dtype):  # skipcq: PYL-R0201
        rgb = (
            torch.tensor(
                [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[255, 255], [255, 255]]], device=device, dtype=torch.uint8
            ).type(dtype)
            / 255.0
        )
        refy = torch.tensor([[[29, 29], [29, 29]]], device=device, dtype=torch.uint8)
        refuv = torch.tensor([[[111]], [[-25]]], device=device, dtype=torch.int8)

        resy = (color_via_conv.rgb_to_yuv420(rgb)[0] * 255.0).type(torch.uint8)
        resuv = (color_via_conv.rgb_to_yuv420(rgb)[1] * 255.0).clamp(-128, 127).type(torch.int8)
        self.assert_close(refy, resy)
        self.assert_close(refuv, resuv)

    # This measures accuracy, given the impact of the subsampling we will avoid the issue by
    # repeating a 2x2 pattern for which mean will match what we get from upscaling again
    def test_forth_and_back(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.rand(3, 4, 5, device=device, dtype=dtype).repeat_interleave(2, dim=2).repeat_interleave(2, dim=1)

        yuv = color_via_conv.rgb_to_yuv420
        rgb = color_via_conv.yuv420_to_rgb
        (a, b) = yuv(data)
        data_out = rgb(a, b)
        self.assert_close(data_out, data, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.rgb_to_yuv420, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestRgbToYuv422(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 6
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(color_via_conv.rgb_to_yuv422(img)[0], torch.Tensor)
        assert isinstance(color_via_conv.rgb_to_yuv422(img)[1], torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-1] /= 2
        assert color_via_conv.rgb_to_yuv422(img)[0].shape == tuple(shapey)
        assert color_via_conv.rgb_to_yuv422(img)[1].shape == tuple(shapeuv)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.rgb_to_yuv422([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv422(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv422(img)

        # dimensionality test
        with pytest.raises(ValueError):
            img = torch.ones(3, 2, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_yuv422(img)

    # This measures accuracy, given the impact of the subsampling we will avoid the issue by
    # repeating a 2x2 pattern for which mena will match what we get from upscaling again
    def test_forth_and_back(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.rand(3, 4, 5, device=device, dtype=dtype).repeat_interleave(2, dim=2).repeat_interleave(2, dim=1)

        yuv = color_via_conv.rgb_to_yuv422
        rgb = color_via_conv.yuv422_to_rgb
        (a, b) = yuv(data)
        data_out = rgb(a, b)
        self.assert_close(data_out, data, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.rgb_to_yuv422, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestYuvToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(color_via_conv.yuv_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert color_via_conv.yuv_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.yuv_to_rgb([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv_to_rgb(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv_to_rgb(img)

    # TODO: investigate and implement me
    # def test_unit(self, device, dtype):
    #    pass

    # TODO: improve accuracy
    def test_forth_and_back(self, device, dtype):  # skipcq: PYL-R0201
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        rgb = color_via_conv.yuv_to_rgb
        yuv = color_via_conv.rgb_to_yuv

        data_out = rgb(yuv(data))
        self.assert_close(data_out, data, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.yuv_to_rgb, (img,), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestYuv420ToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        H, W = 4, 6
        imgy = torch.rand(1, H, W, device=device, dtype=dtype)
        imguv = torch.rand(2, int(H / 2), int(W / 2), device=device, dtype=dtype)
        assert isinstance(color_via_conv.yuv420_to_rgb(imgy, imguv), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2)])
    def test_cardinality(self, device, dtype, shape):
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-2] = int(shapeuv[-2] / 2)
        shapeuv[-1] = int(shapeuv[-1] / 2)

        imgy = torch.ones(shapey, device=device, dtype=dtype)
        imguv = torch.ones(shapeuv, device=device, dtype=dtype)
        assert color_via_conv.yuv420_to_rgb(imgy, imguv).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.yuv420_to_rgb([0.0], [0.0])

        with pytest.raises(TypeError):
            imguv = torch.ones(1, 1, device=device, dtype=dtype)
            imgy = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv420_to_rgb(imgy, imguv)

        with pytest.raises(TypeError):
            imgy = torch.ones(2, 2, 2, device=device, dtype=dtype)
            imguv = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv420_to_rgb(imgy, imguv)

        # dimensionality test
        with pytest.raises(TypeError):
            imgy = torch.ones(3, 2, 1, device=device, dtype=dtype)
            imguv = torch.ones(3, 1, 0, device=device, dtype=dtype)
            assert color_via_conv.yuv420_to_rgb(imgy, imguv)

        # dimensionality test
        with pytest.raises(TypeError):
            imgy = torch.ones(3, 1, 2, device=device, dtype=dtype)
            imguv = torch.ones(3, 0, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv420_to_rgb(imgy, imguv)

    # Test max/min values. This is essentially testing the transform rather than the subsampling
    # ref values manually checked vs rec 601
    def test_unit_white(self, device, dtype):  # skipcq: PYL-R0201
        refrgb = torch.tensor(
            [[[255, 255], [255, 255]], [[255, 255], [255, 255]], [[255, 255], [255, 255]]],
            device=device,
            dtype=torch.uint8,
        )
        y = torch.tensor([[[255, 255], [255, 255]]], device=device, dtype=torch.uint8).type(dtype) / 255.0
        uv = torch.tensor([[[0]], [[0]]], device=device, dtype=torch.int8).type(torch.float) / 255.0

        resrgb = (color_via_conv.yuv420_to_rgb(y, uv) * 255.0).round().type(torch.uint8)
        self.assert_close(refrgb, resrgb)

    def test_unit_red(self, device, dtype):  # skipcq: PYL-R0201
        refrgb = torch.tensor(
            [[[221, 221], [221, 221]], [[17, 17], [17, 17]], [[1, 1], [1, 1]]], device=device, dtype=torch.uint8
        )
        y = torch.tensor([[[76, 76], [76, 76]]], device=device, dtype=torch.uint8).type(dtype) / 255.0
        uv = torch.tensor([[[-37]], [[127]]], device=device, dtype=torch.int8).type(torch.float) / 255.0

        resrgb = (color_via_conv.yuv420_to_rgb(y, uv) * 255.0).round().type(torch.uint8)
        self.assert_close(refrgb, resrgb)

    # TODO: improve accuracy
    def test_forth_and_back(self, device, dtype):  # skipcq: PYL-R0201
        datay = torch.rand(1, 4, 6, device=device, dtype=dtype)
        datauv = torch.rand(2, 2, 3, device=device, dtype=dtype)
        rgb = color_via_conv.yuv420_to_rgb
        yuv = color_via_conv.rgb_to_yuv420

        (data_outy, data_outuv) = yuv(rgb(datay, datauv))
        self.assert_close(data_outy, datay, low_tolerance=True)
        self.assert_close(data_outuv, datauv, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, H, W = 2, 4, 4
        imgy = torch.rand(B, 1, H, W, device=device, dtype=torch.float64, requires_grad=True)
        imguv = torch.rand(B, 2, int(H / 2), int(W / 2), device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.yuv420_to_rgb, (imgy, imguv), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestYuv422ToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        H, W = 4, 6
        imgy = torch.rand(1, H, W, device=device, dtype=dtype)
        imguv = torch.rand(2, H, int(W / 2), device=device, dtype=dtype)
        assert isinstance(color_via_conv.yuv422_to_rgb(imgy, imguv), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 2), (3, 2, 2)])
    def test_cardinality(self, device, dtype, shape):
        shapey = list(shape)
        shapey[-3] = 1
        shapeuv = list(shape)
        shapeuv[-3] = 2
        shapeuv[-1] = int(shapeuv[-1] / 2)

        imgy = torch.ones(shapey, device=device, dtype=dtype)
        imguv = torch.ones(shapeuv, device=device, dtype=dtype)
        assert color_via_conv.yuv422_to_rgb(imgy, imguv).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.yuv422_to_rgb([0.0], [0.0])

        with pytest.raises(TypeError):
            imguv = torch.ones(1, 1, device=device, dtype=dtype)
            imgy = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv422_to_rgb(imgy, imguv)

        with pytest.raises(TypeError):
            imgy = torch.ones(2, 2, 2, device=device, dtype=dtype)
            imguv = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.yuv422_to_rgb(imgy, imguv)

        # dimensionality test
        with pytest.raises(TypeError):
            imgy = torch.ones(3, 2, 1, device=device, dtype=dtype)
            imguv = torch.ones(3, 1, 0, device=device, dtype=dtype)
            assert color_via_conv.yuv422_to_rgb(imgy, imguv)

    # TODO: investigate and implement me
    # def test_unit(self, device, dtype):
    #    pass

    # TODO: improve accuracy
    def test_forth_and_back(self, device, dtype):  # skipcq: PYL-R0201
        datay = torch.rand(1, 4, 6, device=device, dtype=dtype)
        datauv = torch.rand(2, 4, 3, device=device, dtype=dtype)
        rgb = color_via_conv.yuv422_to_rgb
        yuv = color_via_conv.rgb_to_yuv422

        (data_outy, data_outuv) = yuv(rgb(datay, datauv))
        self.assert_close(data_outy, datay, low_tolerance=True)
        self.assert_close(data_outuv, datauv, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, H, W = 2, 4, 4
        imgy = torch.rand(B, 1, H, W, device=device, dtype=torch.float64, requires_grad=True)
        imguv = torch.rand(B, 2, H, int(W / 2), device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.yuv422_to_rgb, (imgy, imguv), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass
