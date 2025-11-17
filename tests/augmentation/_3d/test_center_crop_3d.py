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

import torch

from kornia.augmentation import CenterCrop3D

from testing.base import BaseTester


class TestCenterCrop3D(BaseTester):
    def test_no_transform(self, device, dtype):
        inp = torch.rand(1, 2, 4, 4, 4, device=device, dtype=dtype)
        out = CenterCrop3D(2)(inp)
        assert out.shape == (1, 2, 2, 2, 2)

    def test_transform(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, 8, device=device, dtype=dtype)
        aug = CenterCrop3D(2)
        out = aug(inp)
        assert out.shape == (1, 2, 2, 2, 2)
        assert aug.transform_matrix.shape == (1, 4, 4)

    def test_no_transform_tuple(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, 8, device=device, dtype=dtype)
        out = CenterCrop3D((3, 4, 5))(inp)
        assert out.shape == (1, 2, 3, 4, 5)

    def test_gradcheck(self, device):
        input_tensor = torch.rand(1, 2, 3, 4, 5, device=device, dtype=torch.float64)
        self.gradcheck(CenterCrop3D(3), (input_tensor,))
