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
from kornia.augmentation import RandomEqualize3D

from testing.base import BaseTester


class TestRandomEqualize3D(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = RandomEqualize3D(p=0.5)
        repr = "RandomEqualize3D(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=None)"
        assert str(f) == repr

    def test_random_equalize(self, device, dtype):
        f = RandomEqualize3D(p=1.0)
        f1 = RandomEqualize3D(p=0.0)

        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor(
            [0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000],
            device=device,
            dtype=dtype,
        )
        expected = self.build_input(channels, depth, height, width, bs=1, row=row_expected, device=device, dtype=dtype)

        identity = kornia.eye_like(4, expected)

        self.assert_close(f(inputs3d), expected, rtol=1e-4, atol=1e-4)
        self.assert_close(f.transform_matrix, identity, rtol=1e-4, atol=1e-4)
        self.assert_close(f1(inputs3d), inputs3d, rtol=1e-4, atol=1e-4)
        self.assert_close(f1.transform_matrix, identity, rtol=1e-4, atol=1e-4)

    def test_batch_random_equalize(self, device, dtype):
        f = RandomEqualize3D(p=1.0)
        f1 = RandomEqualize3D(p=0.0)

        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = self.build_input(channels, depth, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor([0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000])
        expected = self.build_input(channels, depth, height, width, bs, row=row_expected, device=device, dtype=dtype)

        identity = kornia.eye_like(4, expected)  # 2 x 4 x 4

        self.assert_close(f(inputs3d), expected, rtol=1e-4, atol=1e-4)
        self.assert_close(f.transform_matrix, identity, rtol=1e-4, atol=1e-4)
        self.assert_close(f1(inputs3d), inputs3d, rtol=1e-4, atol=1e-4)
        self.assert_close(f1.transform_matrix, identity, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomEqualize3D(p=0.5, same_on_batch=True)
        input_tensor = torch.eye(4, device=device, dtype=dtype)
        input_tensor = input_tensor.unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 2, 1, 1)
        res = f(input_tensor)
        self.assert_close(res[0], res[1])

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility

        inputs3d = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        self.gradcheck(RandomEqualize3D(p=0.5), (inputs3d,))

    @staticmethod
    def build_input(channels, depth, height, width, bs=1, row=None, device="cpu", dtype=torch.float32):
        if row is None:
            row = torch.arange(width, device=device, dtype=dtype) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        image3d = torch.stack([image] * depth).transpose(0, 1)
        batch = torch.stack([image3d] * bs)

        return batch.to(device, dtype)
