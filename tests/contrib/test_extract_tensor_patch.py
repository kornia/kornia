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
