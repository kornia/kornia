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


class TestDiceLoss(BaseTester):
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.DiceLoss()
        assert criterion(logits, labels) is not None

    @pytest.mark.parametrize("ignore_index", [-100, None])
    def test_all_zeros(self, device, dtype, ignore_index):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.DiceLoss(ignore_index=ignore_index)
        loss = criterion(logits, labels)
        self.assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_exception(self):
        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1), torch.rand(1, 1, 1))
        assert "Invalid pred shape, we expect BxNxHxW. Got:" in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 2))
        assert "pred and target shapes must be the same. Got: " in str(errinf)

        with pytest.raises(ValueError) as errinf:
            kornia.losses.DiceLoss()(torch.rand(1, 1, 1, 1), torch.rand(1, 1, 1, 1, device="meta"))
        assert "pred and target must be in the same device. Got:" in str(errinf)

    def test_averaging_micro(self, device, dtype):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(1, num_classes, 4, 1, device=device, dtype=dtype)
        logits[:, 0, 0:3] = 10.0
        logits[:, 0, 3:4] = 1.0
        logits[:, 1, 0:3] = 1.0
        logits[:, 1, 3:4] = 10.0

        labels = torch.zeros(2, 4, 1, device=device, dtype=torch.int64)

        exp_1_0 = torch.exp(torch.tensor([1.0], device=device, dtype=dtype))
        exp_10_0 = torch.exp(torch.tensor([10.0], device=device, dtype=dtype))

        expected_intersection = (3.0 * exp_10_0 + 1.0 * exp_1_0) / (exp_1_0 + exp_10_0)
        expected_cardinality = 8.0  # for micro averaging cardinality is equal 2 * H * W
        expected_loss = 1.0 - 2.0 * expected_intersection / (expected_cardinality + eps)
        expected_loss = expected_loss.squeeze()

        criterion = kornia.losses.DiceLoss(average="micro", eps=eps)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("avg", ["micro", "macro"])
    def test_weight(self, device, dtype, avg):
        num_classes = 3
        eps = 1e-8
        logits = torch.zeros(4, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0] = 100.0
        logits[:, 2, :, 1:] = 100.0
        labels = torch.tensor([0, 1, 2, 2], device=device, dtype=torch.int64).expand((4, 1, -1))

        # class 0 is all correct
        expected_loss = torch.tensor([0.0], device=device, dtype=dtype).squeeze()
        weight = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # class 1 is all incorrect
        expected_loss = torch.tensor([1.0], device=device, dtype=dtype).squeeze()
        weight = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # class 2 is partially correct
        expected_loss = torch.tensor([1.0 / 5.0], device=device, dtype=dtype).squeeze()
        weight = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # ignore class 3
        expected_loss = kornia.losses.dice_loss(logits, labels, average=avg, eps=eps)
        weight = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(torch.cat([logits, logits.new_zeros((4, 1, 1, 4))], dim=1), labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

        # test non binary weights
        w_cl_0, w_cl_1 = 0.3, 0.7
        if avg == "macro":
            expected_loss = torch.tensor([0.7], device=device, dtype=dtype).squeeze()
        else:
            dims = (1, 2)
            preds = logits.argmax(1)
            tp_cl_0 = ((preds == 0) & (labels == 0)).sum(dims)
            tp_cl_1 = ((preds == 1) & (labels == 1)).sum(dims)

            fnfp_cl_0 = ((preds == 0) ^ (labels == 0)).sum(dims)
            fnfp_cl_1 = ((preds == 1) ^ (labels == 1)).sum(dims)

            expected_loss = (
                (
                    1
                    - 2
                    * (w_cl_0 * tp_cl_0 + w_cl_1 * tp_cl_1) ** 2
                    / (w_cl_0 * (2 * tp_cl_0 + fnfp_cl_0) + w_cl_1 * (2 * tp_cl_1 + fnfp_cl_1) + eps)
                )
                .mean()
                .to(dtype)
            )

        weight = torch.tensor([w_cl_0, w_cl_1, 0.0], device=device, dtype=dtype)
        criterion = kornia.losses.DiceLoss(average=avg, eps=eps, weight=weight)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_averaging_macro(self, device, dtype):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(1, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0:3] = 10.0
        logits[:, 0, :, 3:4] = 1.0
        logits[:, 1, :, 0:3] = 1.0
        logits[:, 1, :, 3:4] = 10.0

        labels = torch.zeros(2, 1, 4, device=device, dtype=torch.int64)

        exp_1_0 = torch.exp(torch.tensor([1.0], device=device, dtype=dtype))
        exp_10_0 = torch.exp(torch.tensor([10.0], device=device, dtype=dtype))

        expected_intersection_1 = (3.0 * exp_10_0 + exp_1_0) / (exp_1_0 + exp_10_0)
        expected_intersection_2 = 0.0  # all labels are 0 so the intersection for the second class is empty
        expected_cardinality_1 = 4.0 + (3.0 * exp_10_0 + 1.0 * exp_1_0) / (exp_1_0 + exp_10_0)
        expected_cardinality_2 = 0.0 + (1.0 * exp_10_0 + 3.0 * exp_1_0) / (exp_1_0 + exp_10_0)

        expected_loss_1 = 1.0 - 2.0 * expected_intersection_1 / (expected_cardinality_1 + eps)
        expected_loss_2 = 1.0 - 2.0 * expected_intersection_2 / (expected_cardinality_2 + eps)
        expected_loss = (expected_loss_1 + expected_loss_2) / 2.0
        expected_loss = expected_loss.squeeze()

        criterion = kornia.losses.DiceLoss(average="macro", eps=eps)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_ignore_index(self, device, dtype, ignore_index):
        num_classes = 2
        eps = 1e-8

        logits = torch.zeros(2, num_classes, 1, 4, device=device, dtype=dtype)
        logits[:, 0, :, 0] = 100.0
        logits[:, 1, :, 1:] = 100.0
        labels = torch.zeros(2, 1, 4, device=device, dtype=torch.int64)

        labels[..., 2:] = ignore_index
        expected_loss = torch.tensor([1.0 / 2.0], device=device, dtype=dtype).squeeze()
        criterion = kornia.losses.DiceLoss(average="micro", eps=eps, ignore_index=ignore_index)
        loss = criterion(logits, labels)
        self.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (2, 3, 2), device=device)
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100
        self.gradcheck(kornia.losses.dice_loss, (logits, labels), dtypes=[torch.float64, torch.int64])

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_module = kornia.losses.DiceLoss()

        self.assert_close(op(logits, labels), op_module(logits, labels))
