import math

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestBinaryFocalLossWithLogits:
    def test_smoke_none(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        assert kornia.losses.binary_focal_loss_with_logits(
            logits, labels, alpha=0.5, gamma=2.0, reduction="none"
        ).shape == (2, 3, 2)

    def test_smoke_sum(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        assert (
            kornia.losses.binary_focal_loss_with_logits(logits, labels, alpha=0.5, gamma=2.0, reduction="sum").shape
            == ()
        )

    def test_smoke_mean(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        assert (
            kornia.losses.binary_focal_loss_with_logits(logits, labels, alpha=0.5, gamma=2.0, reduction="mean").shape
            == ()
        )

    def test_smoke_mean_flat(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        assert (
            kornia.losses.binary_focal_loss_with_logits(logits, labels, alpha=0.5, gamma=2.0, reduction="mean").shape
            == ()
        )

    def test_jit(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        op = kornia.losses.binary_focal_loss_with_logits
        op_script = torch.jit.script(op)
        actual = op_script(logits, labels, alpha=0.5, gamma=2.0, reduction="none")
        expected = op(logits, labels, alpha=0.5, gamma=2.0, reduction="none")
        assert_close(actual, expected)

    def test_gradcheck(self, device):
        num_classes = 1
        alpha, gamma = 0.5, 2.0  # for focal loss with logits
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 1, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.binary_focal_loss_with_logits, (logits, labels, alpha, gamma), raise_exception=True
        )

    def test_same_output(self, device, dtype):
        num_classes = 1
        logits = torch.rand(2, num_classes, 3, 2, dtype=dtype, device=device)
        labels = torch.rand(2, 3, 2, dtype=dtype, device=device)

        kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}

        assert kornia.losses.binary_focal_loss_with_logits(
            logits, labels, **kwargs
        ) == kornia.losses.BinaryFocalLossWithLogits(**kwargs)(logits, labels)


class TestFocalLoss:
    def test_smoke_none(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        assert kornia.losses.focal_loss(logits, labels, alpha=0.5, gamma=2.0, reduction="none").shape == (2, 3, 2)

    def test_smoke_sum(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        assert kornia.losses.focal_loss(logits, labels, alpha=0.5, gamma=2.0, reduction="sum").shape == ()

    def test_smoke_mean(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        assert kornia.losses.focal_loss(logits, labels, alpha=0.5, gamma=2.0, reduction="mean").shape == ()

    def test_smoke_mean_flat(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.rand(2) * num_classes
        labels = labels.to(device).long()
        assert kornia.losses.focal_loss(logits, labels, alpha=0.5, gamma=2.0, reduction="mean").shape == ()

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        alpha, gamma = 0.5, 2.0  # for focal loss
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(kornia.losses.focal_loss, (logits, labels, alpha, gamma), raise_exception=True)

    def test_jit(self, device, dtype):
        num_classes = 3
        params = (0.5, 2.0)
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.rand(2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.focal_loss
        op_script = torch.jit.script(op)

        actual = op_script(logits, labels, *params)
        expected = op(logits, labels, *params)
        assert_close(actual, expected)

    def test_module(self, device, dtype):
        num_classes = 3
        params = (0.5, 2.0)
        logits = torch.rand(2, num_classes, device=device, dtype=dtype)
        labels = torch.rand(2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.focal_loss
        op_module = kornia.losses.FocalLoss(*params)

        actual = op_module(logits, labels)
        expected = op(logits, labels, *params)
        assert_close(actual, expected)


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

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(kornia.losses.tversky_loss, (logits, labels, alpha, beta), raise_exception=True)

    def test_jit(self, device, dtype):
        num_classes = 3
        params = (0.5, 0.05)
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.tversky_loss
        op_script = torch.jit.script(op)

        actual = op_script(logits, labels, *params)
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

    def test_gradcheck(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(kornia.losses.dice_loss, (logits, labels), raise_exception=True)

    def test_jit(self, device, dtype):
        num_classes = 3
        logits = torch.rand(2, num_classes, 1, 2, device=device, dtype=dtype)
        labels = torch.rand(2, 1, 2) * num_classes
        labels = labels.to(device).long()

        op = kornia.losses.dice_loss
        op_script = torch.jit.script(op)

        assert_close(op(logits, labels), op_script(logits, labels))

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
        loss = criterion(depth, image)

    def test_jit(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_script = torch.jit.script(op)

        assert_close(op(image, depth), op_script(image, depth))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.inverse_depth_smoothness_loss
        op_module = kornia.losses.InverseDepthSmoothnessLoss()

        assert_close(op(image, depth), op_module(image, depth))

    def test_gradcheck(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        depth = utils.tensor_to_gradcheck_var(depth)  # to var
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(kornia.losses.inverse_depth_smoothness_loss, (depth, image), raise_exception=True)


class TestSSIMLoss:
    def test_ssim_equal_none(self, device, dtype):
        # input data
        img1 = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        ssim1 = kornia.losses.ssim_loss(img1, img1, window_size=5, reduction="none")
        ssim2 = kornia.losses.ssim_loss(img2, img2, window_size=5, reduction="none")

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(ssim1, torch.zeros_like(img1), rtol=tol_val, atol=tol_val)
        assert_close(ssim2, torch.zeros_like(img2), rtol=tol_val, atol=tol_val)

    @pytest.mark.parametrize("window_size", [5, 11])
    @pytest.mark.parametrize("reduction_type", ["mean", "sum"])
    @pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_ssim(self, device, dtype, batch_shape, window_size, reduction_type):
        if device.type == 'xla':
            pytest.skip("test highly unstable with tpu")

        # input data
        img = torch.rand(batch_shape, device=device, dtype=dtype)

        loss = kornia.losses.ssim_loss(img, img, window_size, reduction=reduction_type)

        tol_val: float = utils._get_precision_by_name(device, 'xla', 1e-1, 1e-4)
        assert_close(loss.item(), 0.0, rtol=tol_val, atol=tol_val)

    def test_jit(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-6, 'mean')

        op = kornia.losses.ssim_loss
        op_script = torch.jit.script(op)

        assert_close(op(*args), op_script(*args))

    def test_module(self, device, dtype):
        img1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        args = (img1, img2, 5, 1.0, 1e-12, 'mean')

        op = kornia.losses.ssim_loss
        op_module = kornia.losses.SSIMLoss(*args[2:])

        assert_close(op(*args), op_module(*args[:2]))

    def test_gradcheck(self, device, dtype):
        # input data
        window_size = 3
        img1 = torch.rand(1, 1, 5, 4, device=device, dtype=dtype)
        img2 = torch.rand(1, 1, 5, 4, device=device, dtype=dtype)

        # evaluate function gradient
        img1 = utils.tensor_to_gradcheck_var(img1)  # to var
        img2 = utils.tensor_to_gradcheck_var(img2)  # to var

        # TODO: review method since it needs `nondet_tol` in cuda sometimes.
        assert gradcheck(kornia.losses.ssim_loss, (img1, img2, window_size), raise_exception=True, nondet_tol=1e-8)


class TestDivergenceLoss:
    @pytest.mark.parametrize(
        'input,target,expected',
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
        ],
    )
    def test_js_div_loss_2d(self, device, dtype, input, target, expected):
        actual = kornia.losses.js_div_loss_2d(input.to(device, dtype), target.to(device, dtype))
        assert_close(actual.item(), expected)

    @pytest.mark.parametrize(
        'input,target,expected',
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
        ],
    )
    def test_kl_div_loss_2d(self, device, dtype, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input.to(device, dtype), target.to(device, dtype))
        assert_close(actual.item(), expected)

    @pytest.mark.parametrize(
        'input,target,expected',
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1), 0.0)),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7), 0.0)),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), torch.full((1, 7), 0.0)),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7), math.inf)),
        ],
    )
    def test_kl_div_loss_2d_without_reduction(self, device, dtype, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input.to(device, dtype), target.to(device, dtype), reduction='none')
        assert_close(actual, expected.to(device, dtype))

    @pytest.mark.parametrize(
        'input,target,expected',
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
        ],
    )
    def test_noncontiguous_kl(self, device, dtype, input, target, expected):
        input = input.to(device, dtype).view(input.shape[::-1]).T
        target = target.to(device, dtype).view(target.shape[::-1]).T
        actual = kornia.losses.kl_div_loss_2d(input, target).item()
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        'input,target,expected',
        [
            (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
            (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
            (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
        ],
    )
    def test_noncontiguous_js(self, device, dtype, input, target, expected):
        input = input.to(device, dtype).view(input.shape[::-1]).T
        target = target.to(device, dtype).view(target.shape[::-1]).T
        actual = kornia.losses.js_div_loss_2d(input, target).item()
        assert_close(actual, expected)

    def test_gradcheck_kl(self, device, dtype):
        input = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        target = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        target = utils.tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.kl_div_loss_2d, (input, target), raise_exception=True)

    def test_gradcheck_js(self, device, dtype):
        input = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)
        target = torch.rand(1, 1, 10, 16, device=device, dtype=dtype)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        target = utils.tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.js_div_loss_2d, (input, target), raise_exception=True)

    def test_jit_kl(self, device, dtype):
        input = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        target = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        args = (input, target)
        op = kornia.losses.kl_div_loss_2d
        op_jit = torch.jit.script(op, args)
        assert_close(op(*args), op_jit(*args), rtol=0, atol=1e-5)

    def test_jit_js(self, device, dtype):
        input = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        target = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        args = (input, target)
        op = kornia.losses.js_div_loss_2d
        op_jit = torch.jit.script(op, args)
        assert_close(op(*args), op_jit(*args), rtol=0, atol=1e-5)


class TestTotalVariation:
    # Total variation of constant vectors is 0
    @pytest.mark.parametrize(
        'input, expected', [(torch.ones(3, 4, 5), torch.zeros(())), (2 * torch.ones(2, 3, 4, 5), torch.zeros(2))]
    )
    def test_tv_on_constant(self, device, dtype, input, expected):
        actual = kornia.losses.total_variation(input.to(device, dtype))
        assert_close(actual, expected.to(device, dtype))

    # Total variation for 3D tensors
    @pytest.mark.parametrize(
        'input, expected',
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
                torch.tensor(33.001236),
            ),
            (
                torch.tensor([[[0.09094203, 0.32630223, 0.8066123], [0.10921168, 0.09534764, 0.48588026]]]),
                torch.tensor(1.6900232),
            ),
        ],
    )
    def test_tv_on_3d(self, device, dtype, input, expected):
        actual = kornia.losses.total_variation(input.to(device, dtype))
        assert_close(actual, expected.to(device, dtype))

    # Total variation for 4D tensors
    @pytest.mark.parametrize(
        'input, expected',
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
                torch.tensor([5.0054283, 3.1870906]),
            ),
            (
                torch.tensor(
                    [
                        [[[0.1104, 0.2284, 0.4371], [0.4569, 0.1906, 0.8035]]],
                        [[[0.0552, 0.6831, 0.8310], [0.3589, 0.5044, 0.0802]]],
                        [[[0.5078, 0.5703, 0.9110], [0.4765, 0.8401, 0.2754]]],
                    ]
                ),
                torch.tensor([1.9565653, 2.5786452, 2.2681699]),
            ),
        ],
    )
    def test_tv_on_4d(self, device, dtype, input, expected):
        actual = kornia.losses.total_variation(input.to(device, dtype))
        assert_close(actual, expected.to(device, dtype), rtol=1e-4, atol=1e-4)

    # Expect ValueError to be raised when tensors of ndim != 3 or 4 are passed
    @pytest.mark.parametrize('input', [torch.rand(2, 3, 4, 5, 3), torch.rand(3, 1)])
    def test_tv_on_invalid_dims(self, device, dtype, input):
        with pytest.raises(ValueError) as ex_info:
            kornia.losses.total_variation(input.to(device, dtype))

    # Expect TypeError to be raised when non-torch tensors are passed
    @pytest.mark.parametrize('input', [1, [1, 2]])
    def test_tv_on_invalid_types(self, device, dtype, input):
        with pytest.raises(TypeError) as ex_info:
            kornia.losses.total_variation(input)

    def test_jit(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_script = torch.jit.script(op)

        assert_close(op(image), op_script(image))

    def test_module(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)

        op = kornia.losses.total_variation
        op_module = kornia.losses.TotalVariation()

        assert_close(op(image), op_module(image))

    def test_gradcheck(self, device, dtype):
        image = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(kornia.losses.total_variation, (image,), raise_exception=True)


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
        with pytest.raises(Exception) as e:
            criterion(1, 2)

    def test_shape(self, device, dtype):
        # Expecting an exception
        # since we pass tensors of different shapes
        criterion = kornia.losses.PSNRLoss(1.0).to(device, dtype)
        with pytest.raises(Exception) as e:
            criterion(torch.rand(2, 3, 3, 2), torch.rand(2, 3, 3))

    def test_metric(self, device, dtype):
        input = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(20.0, device=device, dtype=dtype)
        actual = kornia.losses.psnr(input, 1.2 * input, 2.0)
        assert_close(actual, expected)

    def test_loss(self, device, dtype):
        input = torch.ones(1, device=device, dtype=dtype)
        expected = torch.tensor(-20.0, device=device, dtype=dtype)
        actual = kornia.losses.psnr_loss(input, 1.2 * input, 2.0)
        assert_close(actual, expected)

    def test_jit(self, device, dtype):
        input = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (input, target, 1.0)

        op = kornia.losses.psnr_loss
        op_script = torch.jit.script(op)

        assert_close(op(*args), op_script(*args))

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
        input = utils.tensor_to_gradcheck_var(input)  # to var
        target = utils.tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.psnr_loss, (input, target, 1.0), raise_exception=True)
