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


class TestCombineTensorPatches(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        m = kornia.contrib.CombineTensorPatches((4, 4), (2, 2), (2, 2))
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2))
        assert m(patches).shape == (1, 1, 4, 4)
        self.assert_close(img, m(patches))

    def test_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(3, 2), stride=(2, 2), padding=1
        )
        with pytest.raises(RuntimeError):
            kornia.contrib.combine_tensor_patches(patches, original_size=(4, 4), window_size=(2, 2), stride=(2, 2))

    def test_rect_odd_dim(self, device, dtype):
        img = torch.arange(12, device=device, dtype=dtype).view(1, 1, 4, 3)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=(0, 2))
        m = kornia.contrib.combine_tensor_patches(
            patches, original_size=(4, 3), window_size=(2, 2), stride=(2, 2), unpadding=(0, 2)
        )
        assert m.shape == (1, 1, 4, 3)
        self.assert_close(img, m)

    def test_pad_triple_error(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(36, device=device, dtype=dtype).view(1, 1, 6, 6), window_size=(4, 4), stride=(4, 4), padding=1
        )
        with pytest.raises(AssertionError):
            kornia.contrib.combine_tensor_patches(
                patches, original_size=(6, 6), window_size=(4, 4), stride=(4, 4), unpadding=(1, 1, 1)
            )

    def test_rectangle_array(self, device, dtype):
        img = torch.arange(24, device=device, dtype=dtype).view(1, 1, 4, 6)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((4, 6), (2, 2), (2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 4, 6)
        self.assert_close(img, m(patches))

    def test_padding1(self, device, dtype):
        img = torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4)
        padding = kornia.contrib.compute_padding((4, 4), (2, 2))
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(1, 1), padding=padding)
        m = kornia.contrib.CombineTensorPatches((4, 4), (2, 2), stride=(1, 1), unpadding=padding)
        assert m(patches).shape == (1, 1, 4, 4)
        self.assert_close(img, m(patches))

    def test_padding2(self, device, dtype):
        img = torch.arange(64, device=device, dtype=dtype).view(1, 1, 8, 8)
        patches = kornia.contrib.extract_tensor_patches(img, window_size=(2, 2), stride=(2, 2), padding=1)
        m = kornia.contrib.CombineTensorPatches((8, 8), (2, 2), stride=(2, 2), unpadding=1)
        assert m(patches).shape == (1, 1, 8, 8)
        self.assert_close(img, m(patches))

    def test_compute_padding(self, device, dtype):
        img_shape = (8, 13)
        rnge = img_shape[0] * img_shape[1]
        img = torch.arange(rnge, device=device, dtype=dtype).view(1, 1, *img_shape)
        window_size = (3, 3)
        padding = kornia.contrib.compute_padding(img_shape, window_size)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, padding=padding
        )
        m = kornia.contrib.CombineTensorPatches(img_shape, window_size, stride=window_size, unpadding=padding)
        assert m(patches).shape == (1, 1, *img_shape)
        self.assert_close(img, m(patches))

    def test_stride_greater_than_window_size(self, device, dtype):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16, device=device, dtype=dtype).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2), padding=1
        )
        with pytest.raises(AssertionError):
            kornia.contrib.combine_tensor_patches(patches, original_size=(4, 4), window_size=(2, 2), stride=(3, 2))

    def test_expl_autopadding(self, device, dtype):
        img_shape = (8, 13)
        img = torch.arange(img_shape[0] * img_shape[1], device=device, dtype=dtype).view(
            1, 1, img_shape[0], img_shape[1]
        )
        window_size = (3, 3)
        padding = kornia.contrib.compute_padding(img_shape, window_size)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, padding=padding
        )
        m = kornia.contrib.CombineTensorPatches(img_shape, window_size, stride=window_size, unpadding=padding)
        assert m(patches).shape == (1, 1, *img_shape)

        self.assert_close(img, m(patches))

    def test_impl_autopadding(self, device, dtype):
        img_shape = (11, 16)
        img = torch.arange(img_shape[0] * img_shape[1], device=device, dtype=dtype).view(1, 1, *img_shape)
        window_size = (3, 3)
        patches = kornia.contrib.extract_tensor_patches(
            img, window_size=window_size, stride=window_size, allow_auto_padding=True
        )
        recomb = kornia.contrib.combine_tensor_patches(
            patches, img_shape, window_size=window_size, stride=window_size, allow_auto_unpadding=True
        )
        assert recomb.shape == img.shape
        self.assert_close(img, recomb)

    def test_gradcheck(self, device):
        patches = kornia.contrib.extract_tensor_patches(
            torch.arange(16.0, device=device, dtype=torch.float64).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2)
        )
        self.gradcheck(kornia.contrib.combine_tensor_patches, (patches, (4, 4), (2, 2), (2, 2)))
