import pytest
import torch
from torch.autograd import gradcheck

import kornia.color as color
import kornia.color.covolutional as covolutional_color
from kornia.testing import BaseTester


class TestRgbToYcbcr(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(covolutional_color.rgb_to_ycbcr(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert covolutional_color.rgb_to_ycbcr(img).shape == shape

    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4)])
    def test_rgb_to_y(self, device, dtype, shape):
        img = torch.rand(*shape, device=device, dtype=dtype)
        output_y = covolutional_color.rgb_to_y(img)
        output_ycbcr = covolutional_color.rgb_to_ycbcr(img)
        self.assert_close(output_y, output_ycbcr[..., 0:1, :, :])

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert covolutional_color.rgb_to_ycbcr([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert covolutional_color.rgb_to_ycbcr(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert covolutional_color.rgb_to_ycbcr(img)

    def test_unit(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.rgb_to_ycbcr(data), covolutional_color.rgb_to_ycbcr(data), low_tolerance=True)

    # TODO: investigate and implement me
    # def test_forth_and_back(self, device, dtype):
    #    pass

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(covolutional_color.rgb_to_ycbcr, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestYcbcrToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(covolutional_color.ycbcr_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert covolutional_color.ycbcr_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert covolutional_color.ycbcr_to_rgb([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert covolutional_color.ycbcr_to_rgb(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert covolutional_color.ycbcr_to_rgb(img)

    def test_unit(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.ycbcr_to_rgb(data), covolutional_color.ycbcr_to_rgb(data), low_tolerance=True)

    # TODO: investigate and implement me
    # def test_forth_and_back(self, device, dtype):
    #    pass

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(covolutional_color.ycbcr_to_rgb, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass
