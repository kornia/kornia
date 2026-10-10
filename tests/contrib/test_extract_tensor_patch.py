# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestExtractTensorPatches(BaseTester):
    def test_smoke(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(3)
        assert m(img).shape == (1, 4, 1, 3, 3)

    def test_b1_ch1_h4w4_ws3(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(img)
        assert patches.shape == (1, 4, 1, 3, 3)
        self.assert_close(img[0, :, :3, :3], patches[0, 0])
        self.assert_close(img[0, :, :3, 1:], patches[0, 1])
        self.assert_close(img[0, :, 1:, :3], patches[0, 2])
        self.assert_close(img[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch2_h4w4_ws3(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        img = img.expand(-1, 2, -1, -1)  # copy all channels
        m = kornia.contrib.ExtractTensorPatches(3)
        patches = m(img)
        assert patches.shape == (1, 4, 2, 3, 3)
        self.assert_close(img[0, :, :3, :3], patches[0, 0])
        self.assert_close(img[0, :, :3, 1:], patches[0, 1])
        self.assert_close(img[0, :, 1:, :3], patches[0, 2])
        self.assert_close(img[0, :, 1:, 1:], patches[0, 3])

    def test_b1_ch1_h4w4_ws2(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2)
        patches = m(img)
        assert patches.shape == (1, 9, 1, 2, 2)
        self.assert_close(img[0, :, 0:2, 1:3], patches[0, 1])
        self.assert_close(img[0, :, 0:2, 2:4], patches[0, 2])
        self.assert_close(img[0, :, 1:3, 1:3], patches[0, 4])
        self.assert_close(img[0, :, 2:4, 1:3], patches[0, 7])

    def test_b1_ch1_h4w4_ws2_stride2(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2, stride=2)
        patches = m(img)
        assert patches.shape == (1, 4, 1, 2, 2)
        self.assert_close(img[0, :, 0:2, 0:2], patches[0, 0])
        self.assert_close(img[0, :, 0:2, 2:4], patches[0, 1])
        self.assert_close(img[0, :, 2:4, 0:2], patches[0, 2])
        self.assert_close(img[0, :, 2:4, 2:4], patches[0, 3])

    def test_b1_ch1_h4w4_ws2_stride21(self, device):
        img = torch.arange(16.0, device=device).view(1, 1, 4, 4)
        m = kornia.contrib.ExtractTensorPatches(2, stride=(2, 1))
        patches = m(img)
        assert patches.shape == (1, 6, 1, 2, 2)
        self.assert_close(img[0, :, 0:2, 1:3], patches[0, 1])
        self.assert_close(img[0, :, 0:2, 2:4], patches[0, 2])
        self.assert_close(img[0, :, 2:4, 0:2], patches[0, 3])
        self.assert_close(img[0, :, 2:4, 2:4], patches[0, 5])

    def test_b1_ch1_h3w3_ws2_stride1_padding1(self, device):
        img = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(img)
        assert patches.shape == (1, 16, 1, 2, 2)
        self.assert_close(img[0, :, 0:2, 0:2], patches[0, 5])
        self.assert_close(img[0, :, 0:2, 1:3], patches[0, 6])
        self.assert_close(img[0, :, 1:3, 0:2], patches[0, 9])
        self.assert_close(img[0, :, 1:3, 1:3], patches[0, 10])

    def test_b2_ch1_h3w3_ws2_stride1_padding1(self, device):
        batch_size = 2
        img = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        img = img.expand(batch_size, -1, -1, -1)
        m = kornia.contrib.ExtractTensorPatches(2, stride=1, padding=1)
        patches = m(img)
        assert patches.shape == (batch_size, 16, 1, 2, 2)
        for i in range(batch_size):
            self.assert_close(img[i, :, 0:2, 0:2], patches[i, 5])
            self.assert_close(img[i, :, 0:2, 1:3], patches[i, 6])
            self.assert_close(img[i, :, 1:3, 0:2], patches[i, 9])
            self.assert_close(img[i, :, 1:3, 1:3], patches[i, 10])

    def test_b1_ch1_h3w3_ws23(self, device):
        img = torch.arange(9.0).view(1, 1, 3, 3).to(device)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(img)
        assert patches.shape == (1, 2, 1, 2, 3)
        self.assert_close(img[0, :, 0:2, 0:3], patches[0, 0])
        self.assert_close(img[0, :, 1:3, 0:3], patches[0, 1])

    def test_b1_ch1_h3w4_ws23(self, device):
        img = torch.arange(12.0).view(1, 1, 3, 4).to(device)
        m = kornia.contrib.ExtractTensorPatches((2, 3))
        patches = m(img)
        assert patches.shape == (1, 4, 1, 2, 3)
        self.assert_close(img[0, :, 0:2, 0:3], patches[0, 0])
        self.assert_close(img[0, :, 0:2, 1:4], patches[0, 1])
        self.assert_close(img[0, :, 1:3, 0:3], patches[0, 2])
        self.assert_close(img[0, :, 1:3, 1:4], patches[0, 3])

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
            return kornia.geometry.denormalize_pixel_coordinates(img, height, width)

        height, width = 3, 4
        grid = kornia.geometry.create_meshgrid(height, width, normalized_coordinates=True).to(device)

        actual = op_script(grid, height, width)
        expected = kornia.denormalize_pixel_coordinates(grid, height, width)

        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.contrib.extract_tensor_patches, (img, 3))

    def test_auto_padding_stride(self, device, dtype):
        img_shape = (11, 14)
        window_size = (3, 3)
        stride = 2
        rnge = img_shape[0] * img_shape[1]
        img = torch.arange(rnge, device=device, dtype=dtype).view(1, 1, *img_shape)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=stride, allow_auto_padding=True
        )
        # 5 patches vertical, 6 2/3 = 7 horizontal = 35 patches
        assert patches.shape == (1, 35, 1, *window_size)

    @pytest.mark.parametrize("img_shape, fits", [((4, 4), True), ((5, 4), False), ((4, 5), False)])
    def test_warns_when_the_window_does_not_fit(self, device, dtype, img_shape, fits):
        # With window 2 and stride 2, a remainder along either axis alone leaves pixels uncovered.
        img = torch.zeros(1, 1, *img_shape, device=device, dtype=dtype)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kornia.contrib.extract_tensor_patches(img, window_size=2, stride=2)
        assert any("will not fit" in str(w.message) for w in caught) is not fits

    @pytest.mark.parametrize("batch_size", [0, 1, 2])
    @pytest.mark.parametrize("noncontiguous", [False, True])
    @pytest.mark.parametrize(
        "image_size,window,stride,padding,auto_padding,pad,grid",
        [
            ((8, 12), (4, 4), (4, 4), 0, False, (0, 0, 0, 0), (2, 3)),
            ((7, 11), (3, 5), (2, 3), 0, False, (0, 0, 0, 0), (3, 3)),
            ((7, 11), (3, 5), (2, 3), (1, 1, 2, 1), False, (2, 1, 1, 1), (4, 4)),
            ((9, 13), (4, 5), (4, 5), 0, True, (1, 1, 1, 2), (3, 3)),
        ],
    )
    def test_patch_layout_and_empty_batch_4429(
        self, batch_size, noncontiguous, image_size, window, stride, padding, auto_padding, pad, grid, device, dtype
    ):
        height, width = image_size
        storage_size = (width, height) if noncontiguous else image_size
        image = (torch.arange(batch_size * 3 * height * width, device=device) % 251).to(dtype)
        image = image.reshape(batch_size, 3, *storage_size)
        if noncontiguous:
            image = image.transpose(-2, -1)
        image = image.detach().requires_grad_()
        original = image.detach().clone()
        module = kornia.contrib.ExtractTensorPatches(window, stride, padding, auto_padding)
        patches = module(image)
        functional = kornia.contrib.extract_tensor_patches(image, window, stride, padding, auto_padding)
        # Independent spatial slices pin both the window count and row-major ordering.
        padded = torch.nn.functional.pad(image, pad)
        expected = torch.stack(
            [
                padded[
                    ..., row * stride[0] : row * stride[0] + window[0], col * stride[1] : col * stride[1] + window[1]
                ]
                for row in range(grid[0])
                for col in range(grid[1])
            ],
            dim=1,
        )
        assert patches.shape == (batch_size, grid[0] * grid[1], 3, *window)
        assert patches.dtype == dtype
        assert patches.device == device
        self.assert_close(patches, expected, rtol=0, atol=0)
        self.assert_close(functional, expected, rtol=0, atol=0)
        self.assert_close(image, original, rtol=0, atol=0)
        weights = (torch.arange(patches.numel(), device=device) % 7 + 1).to(dtype).reshape(patches.shape)
        actual_grad = torch.autograd.grad((patches * weights).sum(), image)[0]
        expected_grad = torch.autograd.grad((expected * weights).sum(), image)[0]
        assert actual_grad.shape == image.shape
        self.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    @pytest.mark.parametrize("batch_size", [0, 1])
    def test_invalid_empty_planes_and_window_4429(self, batch_size, device, dtype):
        with pytest.raises(ValueError, match="non-empty channel"):
            kornia.contrib.extract_tensor_patches(torch.empty(batch_size, 0, 8, 12, device=device, dtype=dtype), 4, 4)
        with pytest.raises(ValueError, match=r"window_size.*positive"):
            kornia.contrib.extract_tensor_patches(
                torch.empty(batch_size, 3, 8, 12, device=device, dtype=dtype), (0, 4), (4, 4)
            )
        with pytest.raises(RuntimeError, match=r"step.*> 0"):
            kornia.contrib.extract_tensor_patches(
                torch.empty(batch_size, 3, 8, 12, device=device, dtype=dtype), 4, 0, padding=(0, 0, 0, 0)
            )

    @pytest.mark.parametrize("batch_size", [0, 2])
    def test_dynamo_patch_extraction_4429(self, batch_size, device, dtype, torch_optimizer):
        image = torch.arange(batch_size * 3 * 8 * 12, device=device).to(dtype).reshape(batch_size, 3, 8, 12)
        module = kornia.contrib.ExtractTensorPatches((4, 4), (4, 4))
        actual = torch_optimizer(module)(image)
        assert actual.shape == (batch_size, 6, 3, 4, 4)
        self.assert_close(actual, module(image), rtol=0, atol=0)
