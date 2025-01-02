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


class TestLovaszHingeLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 1, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 1) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszHingeLoss()
        assert criterion(logits, labels) is not None

    def test_exception(self):
        criterion = kornia.losses.LovaszHingeLoss()

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 2), torch.rand(1, 1, 1))
        assert "pred and target shapes must be the same. Got:" in str(errinfo)

        with pytest.raises(ValueError) as errinfo:
            criterion(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinfo)

    def test_multi_class(self, device, dtype):
        num_classes = 5
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszHingeLoss()
        with pytest.raises(Exception):
            criterion(logits, labels)

    def test_perfect_prediction(self, device, dtype):
        num_classes = 1
        prediction = torch.ones(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.ones(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.LovaszHingeLoss()
        loss = criterion(prediction, labels)
        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        dtype = torch.float64
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device, dtype=dtype)

        self.gradcheck(kornia.losses.lovasz_hinge_loss, (logits, labels))

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_hinge_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_hinge_loss
        op_module = kornia.losses.LovaszHingeLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))
