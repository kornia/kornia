import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close
from packaging import version


class TestConnectedComponents:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == (1, 1, 3, 4)

    def test_exception(self, device, dtype):
        img = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert kornia.contrib.connected_components(img, 1.0)

        with pytest.raises(TypeError):
            assert kornia.contrib.connected_components(img, 0)

        with pytest.raises(ValueError):
            img = torch.rand(1, 3, 4, device=device, dtype=dtype)
            assert kornia.contrib.connected_components(img, 1)

        with pytest.raises(ValueError):
            img = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
            assert kornia.contrib.connected_components(img, 2)

    def test_value(self, device, dtype):
        img = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 14.0, 14.0, 0.0, 0.0, 11.0],
                        [0.0, 14.0, 14.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 34.0, 34.0, 0.0],
                        [0.0, 0.0, 0.0, 34.0, 34.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert_close(out, expected)

    @pytest.mark.parametrize("shape", [(1, 1, 2, 3), (2, 1, 4, 7)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.contrib.connected_components(img, num_iterations=10)
        assert out.shape == shape

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.9"), reason="Tuple cannot be used with PyTorch < v1.9"
    )
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.contrib.connected_components, (img,), raise_exception=True)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.contrib.connected_components
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))


class TestMaxBlurPool2d:
    def test_shape(self, device):
        input = torch.rand(1, 2, 4, 6).to(device)
        pool = kornia.contrib.MaxBlurPool2d(kernel_size=3)
        assert pool(input).shape == (1, 2, 2, 3)

    def test_shape_batch(self, device):
        input = torch.rand(3, 2, 6, 10).to(device)
        pool = kornia.contrib.MaxBlurPool2d(kernel_size=5)
        assert pool(input).shape == (3, 2, 3, 5)

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 4, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.contrib.max_blur_pool2d, (input, 3), raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
            return kornia.contrib.max_blur_pool2d(input, kernel_size)

        img = torch.rand(2, 3, 4, 5).to(device)
        actual = op_script(img, kernel_size=3)
        expected = kornia.contrib.max_blur_pool2d(img, kernel_size=3)
        assert_close(actual, expected)


class TestExtractTensorPatches:
    def test_smoke(self, device):
        input = torch.arange(16.0).view(1, 1, 4, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches(3)
        assert m(input).shape == (1, 4, 1, 3, 3)

    def test_b1_ch1_h4w4_ws3(self, device):
        input = torch.arange(16.0).view(1, 1, 4, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(input)
        assert patches.shape == (1, 4, 1, 3, 3)
        assert_close(input[0, :, :3, :3], patches[0, 0])
        assert_close(input[0, :, :3, 1:], patches[0, 1])
        assert_close(input[0, :, 1:, :3], patches[0, 2])
        assert_close(input[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch2_h4w4_ws3(self, device):
        input = torch.arange(16.0).view(1, 1, 4, 4).to(device)
        input = input.expand(-1, 2, -1, -1)  # copy all channels
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(input)
        assert patches.shape == (1, 4, 2, 3, 3)
        assert_close(input[0, :, :3, :3], patches[0, 0])
        assert_close(input[0, :, :3, 1:], patches[0, 1])
        assert_close(input[0, :, 1:, :3], patches[0, 2])
        assert_close(input[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch1_h4w4_ws2(self, device):
        input = torch.arange(16.0).view(1, 1, 4, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches(2)
        patches = m(input)
        assert patches.shape == (1, 9, 1, 2, 2)
        assert_close(input[0, :, 0:2, 1:3], patches[0, 1])
        assert_close(input[0, :, 0:2, 2:4], patches[0, 2])
        assert_close(input[0, :, 1:3, 1:3], patches[0, 4])
        assert_close(input[0, :, 2:4, 1:3], patches[0, 7])

    def test_b1_ch1_h4w4_ws2_stride2(self, device):
        input = torch.arange(16.0).view(1, 1, 4, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches(2, stride=2)
        patches = m(input)
        assert patches.shape == (1, 4, 1, 2, 2)
        assert_close(input[0, :, 0:2, 0:2], patches[0, 0])
        assert_close(input[0, :, 0:2, 2:4], patches[0, 1])
        assert_close(input[0, :, 2:4, 0:2], patches[0, 2])
        assert_close(input[0, :, 2:4, 2:4], patches[0, 3])

    def test_b1_ch1_h4w4_ws2_stride21(self, device):
        input = torch.arange(16.0).view(1, 1, 4, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches(2, stride=(2, 1))
        patches = m(input)
        assert patches.shape == (1, 6, 1, 2, 2)
        assert_close(input[0, :, 0:2, 1:3], patches[0, 1])
        assert_close(input[0, :, 0:2, 2:4], patches[0, 2])
        assert_close(input[0, :, 2:4, 0:2], patches[0, 3])
        assert_close(input[0, :, 2:4, 2:4], patches[0, 5])

    def test_b1_ch1_h3w3_ws2_stride1_padding1(self, device):
        input = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(input)
        assert patches.shape == (1, 16, 1, 2, 2)
        assert_close(input[0, :, 0:2, 0:2], patches[0, 5])
        assert_close(input[0, :, 0:2, 1:3], patches[0, 6])
        assert_close(input[0, :, 1:3, 0:2], patches[0, 9])
        assert_close(input[0, :, 1:3, 1:3], patches[0, 10])

    def test_b2_ch1_h3w3_ws2_stride1_padding1(self, device):
        batch_size = 2
        input = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        input = input.expand(batch_size, -1, -1, -1)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(input)
        assert patches.shape == (batch_size, 16, 1, 2, 2)
        for i in range(batch_size):
            assert_close(input[i, :, 0:2, 0:2], patches[i, 5])
            assert_close(input[i, :, 0:2, 1:3], patches[i, 6])
            assert_close(input[i, :, 1:3, 0:2], patches[i, 9])
            assert_close(input[i, :, 1:3, 1:3], patches[i, 10])

    def test_b1_ch1_h3w3_ws23(self, device):
        input = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(input)
        assert patches.shape == (1, 2, 1, 2, 3)
        assert_close(input[0, :, 0:2, 0:3], patches[0, 0])
        assert_close(input[0, :, 1:3, 0:3], patches[0, 1])

    def test_b1_ch1_h3w4_ws23(self, device):
        input = torch.arange(12.0).view(1, 1, 3, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(input)
        assert patches.shape == (1, 4, 1, 2, 3)
        assert_close(input[0, :, 0:2, 0:3], patches[0, 0])
        assert_close(input[0, :, 0:2, 1:4], patches[0, 1])
        assert_close(input[0, :, 1:3, 0:3], patches[0, 2])
        assert_close(input[0, :, 1:3, 1:4], patches[0, 3])

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(input: torch.Tensor, height: int, width: int) -> torch.Tensor:
            return kornia.denormalize_pixel_coordinates(input, height, width)

        height, width = 3, 4
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True).to(device)

        actual = op_script(grid, height, width)
        expected = kornia.denormalize_pixel_coordinates(grid, height, width)

        assert_close(actual, expected)

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 4, 4).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.contrib.extract_tensor_patches, (input, 3), raise_exception=True)
