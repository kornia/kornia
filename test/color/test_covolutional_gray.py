import kornia.color as color
import kornia.color.covolutional as covolutional_color
import pytest
import torch

from kornia.testing import BaseTester
from torch.autograd import gradcheck


class TestGrayscaleToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 1, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(covolutional_color.grayscale_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        assert covolutional_color.grayscale_to_rgb(img).shape == (batch_size, 3, height, width)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert covolutional_color.grayscale_to_rgb([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert covolutional_color.grayscale_to_rgb(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert covolutional_color.grayscale_to_rgb(img)

        with pytest.raises(TypeError):
            img = torch.ones(1, 3, 1, 1, device=device, dtype=dtype)
            assert covolutional_color.grayscale_to_rgb(img)

    def test_color(self, device, dtype):
        data = torch.rand(2, 1, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.grayscale_to_rgb(data), covolutional_color.grayscale_to_rgb(data))

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(covolutional_color.grayscale_to_rgb, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestRgbToGrayscale(BaseTester):
    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        out = covolutional_color.rgb_to_grayscale(img)
        assert out.device == img.device
        assert out.dtype == img.dtype

    def test_smoke_byte(self, device):
        B, C, H, W = 2, 3, 4, 5
        img = torch.randint(0, 255, (B, C, H, W), device=device, dtype=torch.uint8)
        out = covolutional_color.rgb_to_grayscale(img)
        assert out.device == img.device
        assert out.dtype == img.dtype

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 3, height, width, device=device, dtype=dtype)
        assert covolutional_color.rgb_to_grayscale(img).shape == (batch_size, 1, height, width)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert covolutional_color.rgb_to_grayscale([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert covolutional_color.rgb_to_grayscale(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert covolutional_color.rgb_to_grayscale(img)

        with pytest.raises(RuntimeError):
            img = torch.ones(3, 1, 1, device=device, dtype=dtype)
            rgb_weights = torch.tensor([0.2, 0.8])
            assert covolutional_color.rgb_to_grayscale(img, rgb_weights=rgb_weights)

    def test_color(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.rgb_to_grayscale(data), covolutional_color.rgb_to_grayscale(data))

    def test_custom_rgb_weights(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)

        rgb_weights = torch.tensor([0.5, 0.25, 0.25])
        img_gray = covolutional_color.rgb_to_grayscale(img, rgb_weights=rgb_weights)
        assert img_gray.device == device
        assert img_gray.dtype == dtype

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(covolutional_color.rgb_to_grayscale, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestBgrToGrayscale(BaseTester):
    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert covolutional_color.bgr_to_grayscale(img) is not None

    @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
    def test_cardinality(self, device, dtype, batch_size, height, width):
        img = torch.ones(batch_size, 3, height, width, device=device, dtype=dtype)
        assert covolutional_color.bgr_to_grayscale(img).shape == (batch_size, 1, height, width)

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert covolutional_color.bgr_to_grayscale([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert covolutional_color.bgr_to_grayscale(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert covolutional_color.bgr_to_grayscale(img)

    def test_color(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.bgr_to_grayscale(data), covolutional_color.bgr_to_grayscale(data))

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(covolutional_color.bgr_to_grayscale, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass
