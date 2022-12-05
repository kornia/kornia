import kornia.color as color
import kornia.color.via_conv as color_via_conv
import pytest
import torch

from kornia.testing import BaseTester
from torch.autograd import gradcheck


class TestRgbToXyz(BaseTester):

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(color_via_conv.rgb_to_xyz(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert color_via_conv.rgb_to_xyz(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.rgb_to_xyz([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_xyz(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.rgb_to_xyz(img)

    def test_unit(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.rgb_to_xyz(data), color_via_conv.rgb_to_xyz(data))

    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        xyz = color_via_conv.rgb_to_xyz
        rgb = color_via_conv.xyz_to_rgb

        data_out = xyz(rgb(data))
        self.assert_close(data_out, data)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.rgb_to_xyz, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass


class TestXyzToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(color_via_conv.xyz_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert color_via_conv.xyz_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert color_via_conv.xyz_to_rgb([0.0])

        with pytest.raises(TypeError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert color_via_conv.xyz_to_rgb(img)

        with pytest.raises(TypeError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert color_via_conv.xyz_to_rgb(img)

    def test_unit(self, device, dtype):
        data = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(color.xyz_to_rgb(data), color_via_conv.xyz_to_rgb(data))

    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        xyz = color_via_conv.rgb_to_xyz
        rgb = color_via_conv.xyz_to_rgb

        data_out = rgb(xyz(data))
        self.assert_close(data_out, data, low_tolerance=True)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(color_via_conv.xyz_to_rgb, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        pass

    def test_module(self, device, dtype):
        pass
