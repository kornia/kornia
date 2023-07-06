import math

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester, assert_close, tensor_to_gradcheck_var
from kornia.utils import torch_meshgrid


class TestBinaryFocalLossWithLogits:
    @pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
    def test_value_same_as_torch_bce_loss(self, device, dtype, reduction):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        focal_equivalent_bce_loss = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=None, gamma=0, reduction=reduction
        )
        torch_bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
        assert_close(focal_equivalent_bce_loss, torch_bce_loss)

    @pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
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
        assert_close(focal_equivalent_bce_loss, torch_bce_loss)

    @pytest.mark.parametrize('reduction,expected_shape', [('none', (2, 3, 2)), ('mean', ()), ('sum', ())])
    @pytest.mark.parametrize('alpha', [None, 0.2, 0.5])
    @pytest.mark.parametrize('gamma', [0.0, 1.0, 2.0])
    def test_shape_alpha_gamma(self, device, dtype, reduction, expected_shape, alpha, gamma):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        actual_shape = kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=alpha, gamma=gamma, reduction=reduction
        ).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize('reduction,expected_shape', [('none', (2, 3, 2)), ('mean', ()), ('sum', ())])
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

    def test_dynamo(self, device, dtype, torch_optimizer):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        op = kornia.losses.binary_focal_loss_with_logits
        op_optimized = torch_optimizer(op)

        args = (0.25, 2.0)
        actual = op_optimized(logits, labels, *args)
        expected = op(logits, labels, *args)
        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):
        logits = torch.rand(2, 3, 2).to(device, dtype)
        logits = tensor_to_gradcheck_var(logits)  # to var
        labels = torch.rand(2, 3, 2).to(device, dtype)

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        assert gradcheck(op, (logits, labels, *args), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        logits = torch.rand(2, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        args = (0.25, 2.0)
        op = kornia.losses.binary_focal_loss_with_logits
        op_module = kornia.losses.BinaryFocalLossWithLogits(*args)
        assert_close(op_module(logits, labels), op(logits, labels, *args))

    def test_numeric_stability(self, device, dtype):
        logits = torch.tensor([[100.0, -100]], dtype=dtype, device=device)
        labels = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)

        args = (0.25, 2.0)
        actual = kornia.losses.binary_focal_loss_with_logits(logits, labels, *args)
        expected = torch.tensor([[0.0, 0.0]], dtype=dtype, device=device)
        assert_close(actual, expected)


class TestFocalLoss:
    @pytest.mark.parametrize('reduction,expected_shape', [('none', (2, 3, 3, 2)), ('mean', ()), ('sum', ())])
    @pytest.mark.parametrize('alpha', [None, 0.2, 0.5])
    @pytest.mark.parametrize('gamma', [0.0, 1.0, 2.0])
    def test_shape_alpha_gamma(self, device, dtype, reduction, expected_shape, alpha, gamma):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        actual_shape = kornia.losses.focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction=reduction).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize('reduction,expected_shape', [('none', (2, 3)), ('mean', ()), ('sum', ())])
    def test_shape_target_with_only_one_dim(self, device, dtype, reduction, expected_shape):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        actual_shape = kornia.losses.focal_loss(logits, labels, alpha=0.1, gamma=1.5, reduction=reduction).shape
        assert actual_shape == expected_shape

    @pytest.mark.parametrize('reduction,expected_shape', [('none', (2, 3, 3, 2)), ('mean', ()), ('sum', ())])
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

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        op = kornia.losses.focal_loss
        op_optimized = torch_optimizer(op)

        args = (0.25, 2.0)
        actual = op_optimized(logits, labels, *args)
        expected = op(logits, labels, *args)
        assert_close(actual, expected)

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        logits = tensor_to_gradcheck_var(logits)  # to var
        labels = torch.randint(num_classes, (2, 3, 2), device=device)

        args = (0.25, 2.0)
        assert gradcheck(kornia.losses.focal_loss, (logits, labels, *args), raise_exception=True, fast_mode=True)

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.randint(num_classes, (2,), device=device)

        args = (0.25, 2.0)
        op = kornia.losses.focal_loss
        op_module = kornia.losses.FocalLoss(*args)
        assert_close(op_module(logits, labels), op(logits, labels, *args))


class TestTverskyLoss:
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        assert criterion(logits, labels) is not None

    def test_all_zeros(self, device, dtype):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)
        assert_close(loss, torch.zeros_like(loss), atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        alpha, beta = 0.5, 0.5  # for tversky loss
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.tversky_loss, (logits, labels, alpha, beta), raise_exception=True, fast_mode=True
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        params = (0.5, 0.05)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_optimized = torch_optimizer(op)

        actual = op_optimized(logits, labels, *params)
        expected = op(logits, labels, *params)
        assert_close(actual, expected)

    def test_module(self, device, dtype):
        num_classes = 3
        params = (0.5, 0.5)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_module = kornia.losses.TverskyLoss(*params)

        actual = op_module(logits, labels)
        expected = op(logits, labels, *params)
        assert_close(actual, expected)


class TestDiceLoss:
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.DiceLoss()
        assert criterion(logits, labels) is not None

    def test_all_zeros(self, device, dtype):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2, device=device, dtype=dtype)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, device=device, dtype=torch.int64)

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)
        assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

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
        assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_averaging_macro(self, device, dtype):
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
        assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(kornia.losses.dice_loss, (logits, labels), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_optimized = torch_optimizer(op)

        assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_module = kornia.losses.DiceLoss()

        assert_close(op(logits, labels), op_module(logits, labels))


class TestDepthSmoothnessLoss:
    @pytest.mark.parametrize("data_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_smoke(self, device, dtype, data_shape):
        image = torch.rand(data_shape, device=device, dtype=dtype)
        depth = torch.rand(data_shape, device=device, dtype=dtype)

        criterion = kornia.losses.InverseDepthSmoothnessLoss()
        assert criterion(depth, image) is not None

    def test_dynamo(self, device, dtype, torch_optimizer):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_optimized = torch_optimizer(op)

        assert_close(op(image, depth), op_optimized(image, depth))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_module = kornia.losses.InverseDepthSmoothnessLoss()

        assert_close(op(image, depth), op_module(image, depth))

    def test_gradcheck(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = tensor_to_gradcheck_var(depth)  # to var
        image = tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(
            kornia.losses.inverse_depth_smoothness_loss, (depth, image), raise_exception=True, fast_mode=True
        )


class TestDivergenceLoss:
    @pytest.mark.parametrize(
        "input,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
        ],
    )
    def test_js_div_loss_2d(self, device, dtype, input, target, expected):
        actual = kornia.losses.js_div_loss_2d(input.to(device, dtype), target.to(device, dtype))
        expected = torch.tensor(expected).to(device, dtype)
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        "input,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
        ],
    )
    def test_kl_div_loss_2d(self, device, dtype, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input.to(device, dtype), target.to(device, dtype))
        expected = torch.tensor(expected).to(device, dtype)
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        "input,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1), 0.0)),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7), 0.0)),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), torch.full((1, 7), 0.0)),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7), math.inf)),
        ],
    )
    def test_kl_div_loss_2d_without_reduction(self, device, dtype, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input.to(device, dtype), target.to(device, dtype), reduction="none")
        assert_close(actual, expected.to(device, dtype))

    @pytest.mark.parametrize(
        "input,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
        ],
    )
    def test_noncontiguous_kl(self, device, dtype, input, target, expected):
        input = input.to(device, dtype).view(input.shape[::-1]).transpose(-2, -1)
        target = target.to(device, dtype).view(target.shape[::-1]).transpose(-2, -1)
        actual = kornia.losses.kl_div_loss_2d(input, target)
        expected = torch.tensor(expected).to(device, dtype)
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        "input,target,expected",
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.303251),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.303251),
        ],
    )
    def test_noncontiguous_js(self, device, dtype, input, target, expected):
        input = input.to(device, dtype).view(input.shape[::-1]).transpose(-2, -1)
        target = target.to(device, dtype).view(target.shape[::-1]).transpose(-2, -1)
        actual = kornia.losses.js_div_loss_2d(input, target)
        expected = torch.tensor(expected).to(device, dtype)
        assert_close(actual, expected)

    def test_gradcheck_kl(self, device, dtype):
        input = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        target = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        # evaluate function gradient
        input = tensor_to_gradcheck_var(input)  # to var
        target = tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.kl_div_loss_2d, (input, target), raise_exception=True, fast_mode=True)

    def test_gradcheck_js(self, device, dtype):
        input = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        target = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        # evaluate function gradient
        input = tensor_to_gradcheck_var(input)  # to var
        target = tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.js_div_loss_2d, (input, target), raise_exception=True, fast_mode=True)

    def test_dynamo_kl(self, device, dtype, torch_optimizer):
        input = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        target = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        args = (input, target)
        op = kornia.losses.kl_div_loss_2d
        op_optimized = torch_optimizer(op)
        assert_close(op(*args), op_optimized(*args), rtol=0, atol=1e-5)

    def test_dynamo_js(self, device, dtype, torch_optimizer):
        input = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        target = torch.full((1, 1, 2, 4), 0.125, dtype=dtype, device=device)
        args = (input, target)
        op = kornia.losses.js_div_loss_2d
        op_optimized = torch_optimizer(op)
        assert_close(op(*args), op_optimized(*args), rtol=0, atol=1e-5)


class TestTotalVariation:
    # Total variation of constant vectors is 0
    @pytest.mark.parametrize(
        "input, expected",
        [
            (torch.ones(3, 4, 5), torch.tensor([0.0, 0.0, 0.0])),
            (2 * torch.ones(2, 3, 4, 5), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        ],
    )
    def test_tv_on_constant(self, device, dtype, input, expected):
        actual = kornia.losses.total_variation(input.to(device, dtype))
        assert_close(actual, expected.to(device, dtype))

    # Total variation for 3D tensors
    @pytest.mark.parametrize(
        "input, expected",
        [
            (
                torch.tensor(
                    [
                        [
                            [0.11747694, 0.5717714, 0.89223915, 0.2929412, 0.63556224],
                            [0.5371079, 0.13416398, 0.7782737, 0.21392655, 0.1757018],
                            [0.62360305, 0.8563448, 0.25304103, 0.68539226, 0.6956515],
                            [0.9350611, 0.01694632, 0.78724295, 0.4760313, 0.73099905],
                        ],
                        [
                            [0.4788819, 0.45253807, 0.932798, 0.5721999, 0.7612051],
                            [0.5455887, 0.8836531, 0.79551977, 0.6677338, 0.74293613],
                            [0.4830376, 0.16420758, 0.15784949, 0.21445751, 0.34168917],
                            [0.8675162, 0.5468113, 0.6117004, 0.01305223, 0.17554593],
                        ],
                        [
                            [0.6423703, 0.5561105, 0.54304767, 0.20339686, 0.8553698],
                            [0.98024786, 0.31562763, 0.10122144, 0.17686582, 0.26260805],
                            [0.20522952, 0.14523649, 0.8601968, 0.02593213, 0.7382898],
                            [0.71935296, 0.9625162, 0.42287344, 0.07979459, 0.9149871],
                        ],
                    ]
                ),
                torch.tensor([12.6647, 7.9527, 12.3838]),
            ),
            (
                torch.tensor([[[0.09094203, 0.32630223, 0.8066123], [0.10921168, 0.09534764, 0.48588026]]]),
                torch.tensor([1.6900]),
            ),
        ],
    )
    def test_tv_on_3d(self, device, dtype, input, expected):
        actual = kornia.losses.total_variation(input.to(device, dtype))
        assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    # Total variation for 4D tensors
    @pytest.mark.parametrize(
        "input, expected",
        [
            (
                torch.tensor(
                    [
                        [
                            [[0.8756, 0.0920], [0.8034, 0.3107]],
                            [[0.3069, 0.2981], [0.9399, 0.7944]],
                            [[0.6269, 0.1494], [0.2493, 0.8490]],
                        ],
                        [
                            [[0.3256, 0.9923], [0.2856, 0.9104]],
                            [[0.4107, 0.4387], [0.2742, 0.0095]],
                            [[0.7064, 0.3674], [0.6139, 0.2487]],
                        ],
                    ]
                ),
                torch.tensor([[1.5672, 1.2836, 2.1544], [1.4134, 0.8584, 0.9154]]),
            ),
            (
                torch.tensor(
                    [
                        [[[0.1104, 0.2284, 0.4371], [0.4569, 0.1906, 0.8035]]],
                        [[[0.0552, 0.6831, 0.8310], [0.3589, 0.5044, 0.0802]]],
                        [[[0.5078, 0.5703, 0.9110], [0.4765, 0.8401, 0.2754]]],
                    ]
                ),
                torch.tensor([[1.9566], [2.5787], [2.2682]]),
            ),
        ],
    )
    def test_tv_on_4d(self, device, dtype, input, expected):
        actual = kornia.losses.total_variation(input.to(device, dtype))
        assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("input", [torch.rand(3, 5, 5), torch.rand(4, 3, 5, 5), torch.rand(4, 2, 3, 5, 5)])
    def test_tv_shapes(self, device, dtype, input):
        input = input.to(device, dtype)
        actual_lesser_dims = []
        for slice in torch.unbind(input, dim=0):
            slice_tv = kornia.losses.total_variation(slice)
            actual_lesser_dims.append(slice_tv)
        actual_lesser_dims = torch.stack(actual_lesser_dims, dim=0)
        actual_higher_dims = kornia.losses.total_variation(input)
        assert_close(actual_lesser_dims, actual_higher_dims.to(device, dtype), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("reduction, expected", [("sum", torch.tensor(20)), ("mean", torch.tensor(1))])
    def test_tv_reduction(self, device, dtype, reduction, expected):
        input, _ = torch_meshgrid([torch.arange(5), torch.arange(5)], "ij")
        input = input.to(device, dtype)
        actual = kornia.losses.total_variation(input, reduction=reduction)
        assert_close(actual, expected.to(device, dtype), rtol=1e-3, atol=1e-3)

    # Expect TypeError to be raised when non-torch tensors are passed
    @pytest.mark.parametrize("input", [1, [1, 2]])
    def test_tv_on_invalid_types(self, device, dtype, input):
        with pytest.raises(TypeError):
            kornia.losses.total_variation(input)

    def test_dynamo(self, device, dtype, torch_optimizer):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_optimized = torch_optimizer(op)

        assert_close(op(image), op_optimized(image))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_module = kornia.losses.TotalVariation()

        assert_close(op(image), op_module(image))

    def test_gradcheck(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        image = tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(kornia.losses.total_variation, (image,), raise_exception=True, fast_mode=True)


class TestPSNRLoss:
    def test_smoke(self, device, dtype):
        input = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        criterion = kornia.losses.PSNRLoss(1.0)
        loss = criterion(input, target)

        assert loss is not None

    def test_type(self, device, dtype):
        # Expecting an exception
        # since we pass integers instead of torch tensors
        criterion = kornia.losses.PSNRLoss(1.0).to(device, dtype)
        with pytest.raises(Exception):
            criterion(1, 2)

    def test_shape(self, device, dtype):
        # Expecting an exception
        # since we pass tensors of different shapes
        criterion = kornia.losses.PSNRLoss(1.0).to(device, dtype)
        with pytest.raises(Exception):
            criterion(torch.rand(2, 3, 3, 2), torch.rand(2, 3, 3))

    def test_loss(self, device, dtype):
        input = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(-20.0, device=device, dtype=dtype)
        actual = kornia.losses.psnr_loss(input, 1.2 * input, 2.0)
        assert_close(actual, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        input = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (input, target, 1.0)

        op = kornia.losses.psnr_loss
        op_optimized = torch_optimizer(op)

        assert_close(op(*args), op_optimized(*args))

    def test_module(self, device, dtype):
        input = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (input, target, 1.0)

        op = kornia.losses.psnr_loss
        op_module = kornia.losses.PSNRLoss(1.0)

        assert_close(op(*args), op_module(input, target))

    def test_gradcheck(self, device, dtype):
        input = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        input = tensor_to_gradcheck_var(input)  # to var
        target = tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.psnr_loss, (input, target, 1.0), raise_exception=True, fast_mode=True)


class TestLovaszHingeLoss:
    def test_smoke(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszHingeLoss()
        assert criterion(logits, labels) is not None

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
        assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(kornia.losses.lovasz_hinge_loss, (logits, labels), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_hinge_loss
        op_optimized = torch_optimizer(op)

        assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_hinge_loss
        op_module = kornia.losses.LovaszHingeLoss()

        assert_close(op(logits, labels), op_module(logits, labels))


class TestLovaszSoftmaxLoss:
    def test_smoke(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.LovaszSoftmaxLoss()
        assert criterion(logits, labels) is not None

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
        print(loss)
        assert_close(loss, torch.zeros_like(loss), rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device, dtype):
        num_classes = 4
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(kornia.losses.lovasz_softmax_loss, (logits, labels), raise_exception=True, fast_mode=True)

    @pytest.mark.skip(reason='Not matching results')
    def test_dynamo(self, device, dtype, torch_optimizer):
        # TODO: investigate if we can fix it or report the issue
        num_classes = 6
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.randint(0, num_classes, (2, 1, 2), device=device)

        op = kornia.losses.lovasz_softmax_loss
        op_optimized = torch_optimizer(op)

        assert_close(op(logits, labels), op_optimized(logits, labels))

    def test_module(self, device, dtype):
        num_classes = 5
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.lovasz_softmax_loss
        op_module = kornia.losses.LovaszSoftmaxLoss()

        assert_close(op(logits, labels), op_module(logits, labels))


class TestWelschLoss(BaseTester):
    def test_smoke(self, device, dtype):
        img1 = torch.rand(2, 3, 2, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 2, device=device, dtype=dtype)

        criterion = kornia.losses.WelschLoss()

        assert criterion(img1, img2) is not None

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.WelschLoss(reduction='none')(img, img)
        assert actual.shape == shape

        actual = kornia.losses.WelschLoss(reduction='mean')(img, img)
        assert actual.shape == ()

    def test_gradcheck(self, device):
        img1 = torch.rand(2, 3, 3, 3, device=device)
        img2 = torch.rand(2, 3, 3, 3, device=device)

        img1 = tensor_to_gradcheck_var(img1)  # to var
        assert self.gradcheck(kornia.losses.welsch_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.welsch_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.welsch_loss
        op_module = kornia.losses.WelschLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.welsch_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.welsch_loss(img1, img2, reduction=reduction)

        if reduction == 'mean':
            expected = torch.tensor(0.39346934028, device=device, dtype=dtype)
        elif reduction == 'sum':
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.39346934028).sum()

        self.assert_close(actual, expected)

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        with pytest.raises(Exception) as execinfo:
            kornia.losses.welsch_loss(img, img, reduction='test')
        assert 'Given type of reduction is not supported. Got: test' in str(execinfo)

        # Check if both are tensors
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.welsch_loss(1.0, img)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.losses.welsch_loss(img, 1.0)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.welsch_loss(img, img_b, 3)
        assert 'Not same shape for tensors. Got:' in str(errinfo)


class TestCauchyLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_smoke(self, device, dtype, reduction, shape):
        img1 = torch.rand(shape, device=device, dtype=dtype)
        img2 = torch.rand(shape, device=device, dtype=dtype)

        assert kornia.losses.cauchy_loss(img1, img2, reduction) is not None

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        with pytest.raises(Exception) as execinfo:
            kornia.losses.cauchy_loss(img, img, reduction='test')
        assert 'Given type of reduction is not supported. Got: test' in str(execinfo)

        # Check if both are tensors
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.cauchy_loss(1.0, img)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.losses.cauchy_loss(img, 1.0)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.cauchy_loss(img, img_b, 3)
        assert 'Not same shape for tensors. Got:' in str(errinfo)

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.cauchy_loss(img, img, reduction='none')
        assert actual.shape == shape

        actual = kornia.losses.cauchy_loss(img, img, reduction='sum')
        assert actual.shape == ()

        actual = kornia.losses.cauchy_loss(img, img, reduction='mean')
        assert actual.shape == ()

    def test_gradcheck(self, device):
        img1 = torch.rand(2, 3, 3, 3, device=device)
        img2 = torch.rand(2, 3, 3, 3, device=device)

        img1 = tensor_to_gradcheck_var(img1)  # to var
        img2 = tensor_to_gradcheck_var(img2)  # to var
        self.gradcheck(kornia.losses.cauchy_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.cauchy_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.cauchy_loss
        op_module = kornia.losses.CauchyLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.cauchy_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.cauchy_loss(img1, img2, reduction=reduction)

        if reduction == 'mean':
            expected = torch.tensor(0.40546512603759766, device=device, dtype=dtype)
        elif reduction == 'sum':
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.40546512603759766).sum()

        self.assert_close(actual, expected)


class TestGemanMcclureLossLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_smoke(self, device, dtype, reduction, shape):
        img1 = torch.rand(shape, device=device, dtype=dtype)
        img2 = torch.rand(shape, device=device, dtype=dtype)

        assert kornia.losses.geman_mcclure_loss(img1, img2, reduction) is not None

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        with pytest.raises(Exception) as execinfo:
            kornia.losses.geman_mcclure_loss(img, img, reduction='test')
        assert 'Given type of reduction is not supported. Got: test' in str(execinfo)

        # Check if both are tensors
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.geman_mcclure_loss(1.0, img)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.losses.geman_mcclure_loss(img, 1.0)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.geman_mcclure_loss(img, img_b, 3)
        assert 'Not same shape for tensors. Got:' in str(errinfo)

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.geman_mcclure_loss(img, img, reduction='none')
        assert actual.shape == shape

        actual = kornia.losses.geman_mcclure_loss(img, img, reduction='sum')
        assert actual.shape == ()

        actual = kornia.losses.geman_mcclure_loss(img, img, reduction='mean')
        assert actual.shape == ()

    def test_gradcheck(self, device):
        img1 = torch.rand(2, 3, 3, 3, device=device)
        img2 = torch.rand(2, 3, 3, 3, device=device)

        img1 = tensor_to_gradcheck_var(img1)  # to var
        img2 = tensor_to_gradcheck_var(img2)  # to var
        self.gradcheck(kornia.losses.geman_mcclure_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.geman_mcclure_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.geman_mcclure_loss
        op_module = kornia.losses.GemanMcclureLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.geman_mcclure_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.geman_mcclure_loss(img1, img2, reduction=reduction)

        if reduction == 'mean':
            expected = torch.tensor(0.4, device=device, dtype=dtype)
        elif reduction == 'sum':
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.4).sum()

        self.assert_close(actual, expected)


class TestCharbonnierLoss(BaseTester):
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_smoke(self, device, dtype, reduction, shape):
        img1 = torch.rand(shape, device=device, dtype=dtype)
        img2 = torch.rand(shape, device=device, dtype=dtype)

        assert kornia.losses.charbonnier_loss(img1, img2, reduction) is not None

    def test_exception(self, device, dtype):
        img = torch.rand(3, 3, 3, device=device, dtype=dtype)

        # wrong reduction
        with pytest.raises(Exception) as execinfo:
            kornia.losses.charbonnier_loss(img, img, reduction='test')
        assert 'Given type of reduction is not supported. Got: test' in str(execinfo)

        # Check if both are tensors
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.charbonnier_loss(1.0, img)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            kornia.losses.charbonnier_loss(img, 1.0)
        assert 'Not a Tensor type. Got:' in str(errinfo)

        # Check if same shape
        img_b = torch.rand(1, 1, 3, 3, 4, device=device, dtype=dtype)
        with pytest.raises(TypeError) as errinfo:
            kornia.losses.charbonnier_loss(img, img_b, 3)
        assert 'Not same shape for tensors. Got:' in str(errinfo)

    @pytest.mark.parametrize("shape", [(1, 3, 5, 5), (2, 5, 5)])
    def test_cardinality(self, shape, device, dtype):
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.charbonnier_loss(img, img, reduction='none')
        assert actual.shape == shape

        actual = kornia.losses.charbonnier_loss(img, img, reduction='sum')
        assert actual.shape == ()

        actual = kornia.losses.charbonnier_loss(img, img, reduction='mean')
        assert actual.shape == ()

    def test_gradcheck(self, device):
        img1 = torch.rand(2, 3, 3, 3, device=device)
        img2 = torch.rand(2, 3, 3, 3, device=device)

        img1 = tensor_to_gradcheck_var(img1)  # to var
        img2 = tensor_to_gradcheck_var(img2)  # to var
        self.gradcheck(kornia.losses.charbonnier_loss, (img1, img2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.charbonnier_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(img1, img2), op_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 3, 3, device=device, dtype=dtype)

        op = kornia.losses.charbonnier_loss
        op_module = kornia.losses.CharbonnierLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 9), (2, 4, 3, 6)])
    def test_perfect_prediction(self, device, dtype, reduction, shape):
        # Sanity test
        img = torch.rand(shape, device=device, dtype=dtype)
        actual = kornia.losses.charbonnier_loss(img, img, reduction=reduction)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(actual, expected)

        # Check loss computation
        img1 = torch.ones(shape, device=device, dtype=dtype)
        img2 = torch.zeros(shape, device=device, dtype=dtype)

        actual = kornia.losses.charbonnier_loss(img1, img2, reduction=reduction)

        if reduction == 'mean':
            expected = torch.tensor(0.41421356237, device=device, dtype=dtype)
        elif reduction == 'sum':
            expected = (torch.ones_like(img1, device=device, dtype=dtype) * 0.41421356237).sum()

        self.assert_close(actual, expected)
