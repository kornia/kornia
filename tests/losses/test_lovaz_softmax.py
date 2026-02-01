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


class TestLovaszSoftmaxLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 1, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 1) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszSoftmaxLoss()
        assert criterion(logits, labels) is not None

    def test_exception(self):
        from kornia.core.exceptions import ShapeError

        criterion = kornia.losses.LovaszSoftmaxLoss()

        with pytest.raises(ShapeError) as errinfo:
            criterion(torch.rand(1), torch.rand(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(ShapeError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid pred shape, we expect BxNxHxW, with N > 1." in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 2, 1, 1), torch.rand(1, 1, 2))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 2, 1, 1), torch.rand(1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinfo)

    def test_binary(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszSoftmaxLoss()
        with pytest.raises(Exception):
            criterion(logits, labels)

    def test_all_ones(self, device, dtype):
        num_classes = 2
        # make perfect prediction
        # note that softmax(prediction[:, 1]) == 1. softmax(prediction[:, 0]) == 0.
        prediction = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        prediction[:, 1] = 100.0
        labels = torch.ones(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.LovaszSoftmaxLoss()
        loss = criterion(prediction, labels)

        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_weight(self, device, dtype):
        num_classes = 2
        # make perfect prediction
        # note that softmax(prediction[:, 1]) == 1. softmax(prediction[:, 0]) == 0.
        prediction = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        prediction[:, 0] = 100.0
        labels = torch.ones(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.LovaszSoftmaxLoss(weight=torch.tensor([1.0, 0.0], device=device, dtype=dtype))
        loss = criterion(prediction, labels)

        self.assert_close(loss, 0.5 * torch.ones_like(loss), rtol=1e-3, atol=1e-3)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        num_classes = 4
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)
        self.gradcheck(kornia.losses.lovasz_softmax_loss, (logits, labels), dtypes=[torch.float64, torch.int64])

    @pytest.mark.skip(reason="Not matching results")
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: investigate if we can fix it or report the issue
        num_classes = 6
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.randint(0, num_classes, (2, 1, 2), device=device)

        op = kornia.losses.lovasz_softmax_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 5
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_softmax_loss
        op_module = kornia.losses.LovaszSoftmaxLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))
