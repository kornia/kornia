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

from kornia.color.utils import _apply_linear_transformation

from testing.base import BaseTester


class TestApplyLinearTransformation(BaseTester):
    def test_compute_operands(self, device):
        image = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64).view(1, 3, 1, 2)
        kernel = torch.tensor([[1.0, 0.5, -0.25], [-0.5, 1.0, 0.0], [0.0, 0.25, 1.0]], dtype=torch.float64)
        bias = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)

        image_compute = image.float()
        kernel_compute = kernel.to(device=device, dtype=torch.float32)
        bias_compute = bias.to(device=device, dtype=torch.float32)
        expected = torch.einsum("oi, ...ihw -> ...ohw", kernel_compute, image_compute)
        expected = expected + bias_compute.view(-1, 1, 1)

        actual = _apply_linear_transformation(image, kernel, bias)

        assert actual.shape == image.shape
        assert actual.device == image.device
        assert actual.dtype == torch.float32
        assert actual.is_contiguous()
        self.assert_close(actual, expected, atol=0.0, rtol=0.0)
