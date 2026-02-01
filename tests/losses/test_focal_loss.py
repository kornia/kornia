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

import kornia

from testing.base import BaseTester


class TestBinaryFocalLossWithLogits(BaseTester):
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("ignore_index", [-100, None])
    def test_value_same_as_torch_bce_loss(self, device, dtype, reduction, ignore_index):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        focal_equivalent_bce_loss = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=None, gamma=0, reduction=reduction, ignore_index=ignore_index
        )
        torch_bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
        self.assert_close(focal_equivalent_bce_loss, torch_bce_loss)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_value_same_as_torch_bce_loss_pos_weight_weight(self, device, dtype, reduction):
        num_classes = 3
        logits = torch.rand(2, num_classes, 2, dtype=dtype, device=device)
        labels = torch.rand(2, num_classes, 2, dtype=dtype, device=device)

        pos_weight = torch.rand(num_classes, 1, dtype=dtype, device=device)
        weight = torch.rand(num_classes, 1, dtype=dtype, device=device)

        focal_equivalent_bce_loss = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=None, gamma=0, reduction=reduction, pos_weight=pos_weight, weight=weight
        )
        torch_bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction=reduction, pos_weight=pos_weight, weight=weight
        )
        self.assert_close(focal_equivalent_bce_loss, torch_bce_loss)

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("alpha", [None, 0.2, 0.5])
    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
    def test_shape_alpha_gamma(self, device, dtype, reduction, expected_shape, alpha, gamma):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=alpha, gamma=gamma, reduction=reduction
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("pos_weight", [None, (1, 2, 5)])
    @pytest.mark.parametrize("weight", [None, (0.2, 0.5, 0.8)])
    def test_shape_pos_weight_weight(self, device, dtype, reduction, expected_shape, pos_weight, weight):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        pos_weight = None if pos_weight is None else torch.tensor(pos_weight, dtype=dtype, device=device)
        weight = None if weight is None else torch.tensor(weight, dtype=dtype, device=device)

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, pos_weight=pos_weight, weight=weight
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_shape_ignore_index(self, device, dtype, reduction, expected_shape, ignore_index):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        ignore = torch.rand(2, 3, 2, device=device) > 0.6
        labels[ignore] = ignore_index

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, ignore_index=ignore_index
        ).shape
        assert actual_shape == expected_shape

    def test_dynamo(self, device, dtype, torch_optimizer):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        op = kornia.losses.binary_focal_loss_with_logits
        op_optimized = torch_optimizer(op)

        args = (0.25, 2.0)
        actual = op_optimized(logits, labels, *args)
        expected = op(logits, labels, *args)
        self.assert_close(actual, expected)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        logits = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
        labels = torch.rand(2, 3, 2, device=device, dtype=torch.float64)

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        self.gradcheck(op, (logits, labels, *args))

    @pytest.mark.grad()
    def test_gradcheck_ignore_index(self, device, dtype):
        logits = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
        labels = torch.rand(2, 3, 2, device=device, dtype=torch.float64)
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        self.gradcheck(op, (logits, labels, *args), requires_grad=[True, False, False, False])

    def test_module(self, device, dtype):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        op_module = kornia.losses.BinaryFocalLossWithLogits(*args)
        self.assert_close(op_module(logits, labels), op(logits, labels, *args))

    def test_numeric_stability(self, device, dtype):
        logits = torch.tensor([[100.0, -100]], dtype=dtype, device=device)
        labels = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)

        args = (0.25, 2.0)
        actual = kornia.losses.binary_focal_loss_with_logits(logits, labels, *args)
        expected = torch.tensor([[0.0, 0.0]], dtype=dtype, device=device)
        self.assert_close(actual, expected)


class TestFocalLoss(BaseTester):
    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("alpha", [None, 0.2, 0.5])
    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
    def test_shape_alpha_gamma(self, device, dtype, reduction, expected_shape, alpha, gamma):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        actual_shape = kornia.losses.focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction=reduction).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3)), ("mean", ()), ("sum", ())])
    def test_shape_target_with_only_one_dim(self, device, dtype, reduction, expected_shape):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        actual_shape = kornia.losses.focal_loss(logits, labels, alpha=0.1, gamma=1.5, reduction=reduction).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("weight", [None, (0.2, 0.5, 0.8)])
    def test_shape_weight(self, device, dtype, reduction, expected_shape, weight):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        weight = None if weight is None else torch.tensor(weight, dtype=dtype, device=device)

        actual_shape = kornia.losses.focal_loss(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, weight=weight
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("reduction,expected_shape", [("none", (2, 3, 3, 2)), ("mean", ()), ("sum", ())])
    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_shape_ignore_index(self, device, dtype, reduction, expected_shape, ignore_index):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        ignore = torch.rand(2, 3, 2, device=device) > 0.6
        labels[ignore] = ignore_index

        actual_shape = kornia.losses.focal_loss(
            logits, labels, alpha=0.8, gamma=0.5, reduction=reduction, ignore_index=ignore_index
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize("ignore_index", [-100, 255])
    def test_value_ignore_index(self, device, dtype, ignore_index):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        ignore = torch.rand(2, 3, 2, device=device) > 0.6
        labels[ignore] = ignore_index

        labels_extra_class = labels.clone()
        labels_extra_class[ignore] = num_classes
        logits_extra_class = torch.cat([logits, logits.new_full((2, 1, 3, 2), float("-inf"))], dim=1)

        expected_values = kornia.losses.focal_loss(
            logits_extra_class, labels_extra_class, alpha=0.8, gamma=0.5, reduction="none"
        )[:, :-1, ...]

        actual_values = kornia.losses.focal_loss(
            logits, labels, alpha=0.8, gamma=0.5, reduction="none", ignore_index=ignore_index
        )

        self.assert_close(actual_values, expected_values)

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        op = kornia.losses.focal_loss
        op_optimized = torch_optimizer(op)

        args = (0.25, 2.0)
        actual = op_optimized(logits, labels, *args)
        expected = op(logits, labels, *args)
        self.assert_close(actual, expected)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=torch.float64)
        labels = torch.randint(num_classes, (2, 3, 2), device=device).long()
        ignore = torch.rand(2, 3, 2, device=device) > 0.8
        labels[ignore] = -100

        self.gradcheck(
            kornia.losses.focal_loss, (logits, labels, 0.25, 2.0), dtypes=[torch.float64, torch.int64, None, None]
        )

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        args = (0.25, 2.0)
        op = kornia.losses.focal_loss
        op_module = kornia.losses.FocalLoss(*args)
        self.assert_close(op_module(logits, labels), op(logits, labels, *args))
