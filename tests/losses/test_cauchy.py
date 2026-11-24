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


class TestCauchyLoss(BaseTester):
    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        from kornia.core.exceptions import BaseError

        with pytest.raises(BaseError) as execinfo:
            kornia.losses.cauchy_loss(img, img, reduction="test")
        assert "Given type of reduction is not supported. Got: test" in str(execinfo.value)

        # Check if both are tensors
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.cauchy_loss(1.0, img)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.cauchy_loss(img, 1.0)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        # Check if same shape
        from kornia.core.exceptions import ShapeError

        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            kornia.losses.cauchy_loss(img, img_b)
        assert "Shape mismatch" in str(errinfo.value)

    def test_gradcheck(self, device):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=torch.float64)

        self.gradcheck(kornia.losses.cauchy_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.cauchy_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))
