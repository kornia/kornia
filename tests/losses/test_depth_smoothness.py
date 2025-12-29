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


class TestDepthSmoothnessLoss(BaseTester):
    @pytest.mark.parametrize("data_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_smoke(self, device, dtype, data_shape):
        image = torch.rand(data_shape, device=device, dtype=dtype)
        depth = torch.rand(data_shape, device=device, dtype=dtype)

        criterion = kornia.losses.InverseDepthSmoothnessLoss()
        assert criterion(depth, image) is not None

    def test_exception(self):
        with pytest.raises(TypeError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(1, 1)
        assert "Input idepth type is not a torch.Tensor. Got" in str(errinf)

        with pytest.raises(TypeError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1), 1)
        assert "Input image type is not a torch.Tensor. Got" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1), torch.rand(1, 1, 1, 1))
        assert "Invalid idepth shape, we expect BxCxHxW. Got" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid image shape, we expect BxCxHxW. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "idepth and image shapes must be the same. Got" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "idepth and image must be in the same device. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.InverseDepthSmoothnessLoss()(
                torch.rand(1, 1, 1, 1, dtype=torch.float32), torch.rand(1, 1, 1, 1, dtype=torch.float64)
            )
        assert "idepth and image must be in the same dtype. Got:" in str(errinf)

    def test_dynamo(self, device, dtype, torch_optimizer):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(image, depth), op_optimized(image, depth))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_module = kornia.losses.InverseDepthSmoothnessLoss()

        self.assert_close(op(image, depth), op_module(image, depth))

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=torch.float64)
        self.gradcheck(kornia.losses.inverse_depth_smoothness_loss, (depth, image))
