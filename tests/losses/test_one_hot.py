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
import torch.nn.functional as F

from kornia.core.exceptions import BaseError
from kornia.losses.one_hot import one_hot

from testing.base import assert_close


class TestOneHot:
    def test_smoke(self, device, dtype):
        num_classes = 4
        labels = torch.zeros(2, 2, 1, dtype=torch.int64, device=device)
        labels[0, 0, 0] = 0
        labels[0, 1, 0] = 1
        labels[1, 0, 0] = 2
        labels[1, 1, 0] = 3

        # convert labels to one hot tensor
        one_hot_tensor = one_hot(labels, num_classes, device, dtype)

        assert_close(one_hot_tensor[0, labels[0, 0, 0], 0, 0].item(), 1.0)
        assert_close(one_hot_tensor[0, labels[0, 1, 0], 1, 0].item(), 1.0)
        assert_close(one_hot_tensor[1, labels[1, 0, 0], 0, 0].item(), 1.0)
        assert_close(one_hot_tensor[1, labels[1, 1, 0], 1, 0].item(), 1.0)

    def test_exact_values(self, device, dtype):
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)
        expected = F.one_hot(labels, 3).movedim(-1, 1).to(dtype)

        one_hot_tensor = one_hot(labels, 3, device, dtype)

        assert one_hot_tensor.dtype == dtype
        assert_close(one_hot_tensor, expected, rtol=0, atol=0)
        assert_close(one_hot_tensor.sum(1), torch.ones_like(expected[:, 0]), rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.int64, torch.uint8, torch.bool])
    def test_integer_dtypes(self, device, dtype):
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)

        one_hot_tensor = one_hot(labels, 3, device, dtype)

        assert one_hot_tensor.dtype == dtype
        assert torch.equal(one_hot_tensor, F.one_hot(labels, 3).movedim(-1, 1).to(dtype))

    def test_eps_smoothing(self, device, dtype):
        labels = torch.tensor([[[0, 2, 1], [1, 0, 0]]], device=device)
        exact = F.one_hot(labels, 3).movedim(-1, 1).to(dtype)

        one_hot_tensor = one_hot(labels, 3, device, dtype, eps=0.125)

        assert_close(one_hot_tensor, torch.where(exact == 1, 1.0, 0.125).to(dtype))

    def test_exception_eps_with_integer_dtype(self, device):
        labels = torch.tensor([[0, 1]], device=device)

        with pytest.raises(BaseError, match="floating point dtype"):
            one_hot(labels, 2, device, torch.int64, eps=1e-6)
