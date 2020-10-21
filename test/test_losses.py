import pytest

import kornia
import kornia.testing as utils  # test utils

import math
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestFocalLoss:
    def test_smoke_none(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        assert kornia.losses.focal_loss(
            logits, labels, alpha=0.5, gamma=2.0, reduction="none"
        ).shape == (2, 3, 2)

    def test_smoke_sum(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="sum"
            ).shape == ()
        )

    def test_smoke_mean(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="mean"
            ).shape == ()
        )

    def test_smoke_mean_flat(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes).to(device)
        labels = torch.rand(2) * num_classes
        labels = labels.to(device).long()
        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="mean"
            ).shape == ()
        )

    # TODO: implement me
    def test_jit(self, device):
        pass

    def test_gradcheck(self, device):
        num_classes = 3
        alpha, gamma = 0.5, 2.0  # for focal loss
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.focal_loss,
            (logits, labels, alpha, gamma),
            raise_exception=True,
        )


class TestTverskyLoss:
    def test_smoke(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)

    def test_all_zeros(self, device):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2).to(device)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64).to(device)

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)
        assert pytest.approx(loss.item(), 0.0)

    # TODO: implement me
    def test_jit(self, device):
        pass

    def test_gradcheck(self, device):
        num_classes = 3
        alpha, beta = 0.5, 0.5  # for tversky loss
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.tversky_loss,
            (logits, labels, alpha, beta),
            raise_exception=True,
        )


class TestDiceLoss:
    def test_smoke(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)

    def test_all_zeros(self, device):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2).to(device)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64).to(device)

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)
        assert pytest.approx(loss.item(), 0.0)

    # TODO: implement me
    def test_jit(self, device):
        pass

    def test_gradcheck(self, device):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2).to(device)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.to(device).long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.dice_loss, (logits, labels), raise_exception=True
        )


class TestDepthSmoothnessLoss:
    @pytest.mark.parametrize("data_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_smoke(self, device, data_shape):
        image = torch.rand(data_shape).to(device)
        depth = torch.rand(data_shape).to(device)

        criterion = kornia.losses.InverseDepthSmoothnessLoss()
        loss = criterion(depth, image)

    # TODO: implement me
    def test_1(self, device):
        pass

    # TODO: implement me
    def test_jit(self, device):
        pass

    def test_gradcheck(self, device):
        image = torch.rand(1, 1, 10, 16).to(device)
        depth = torch.rand(1, 1, 10, 16).to(device)
        depth = utils.tensor_to_gradcheck_var(depth)  # to var
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(
            kornia.losses.inverse_depth_smoothness_loss,
            (depth, image),
            raise_exception=True,
        )


class TestSSIMLoss:

    def test_ssim_equal_none(self, device):
        # input data
        img1 = torch.rand(1, 1, 10, 16).to(device)
        img2 = torch.rand(1, 1, 10, 16).to(device)

        ssim1 = kornia.ssim(img1, img1, window_size=5, reduction="none")
        ssim2 = kornia.ssim(img2, img2, window_size=5, reduction="none")

        assert_allclose(ssim1, torch.zeros_like(img1))
        assert_allclose(ssim2, torch.zeros_like(img2))

    @pytest.mark.parametrize("window_size", [5, 11])
    @pytest.mark.parametrize("reduction_type", ["mean", "sum"])
    @pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_ssim(self, device, batch_shape, window_size, reduction_type):
        # input data
        img = torch.rand(batch_shape).to(device)

        ssim = kornia.losses.SSIM(window_size, reduction_type)
        assert_allclose(ssim(img, img).item(), 0.0)

    def test_gradcheck(self, device):
        # input data
        window_size = 3
        img1 = torch.rand(1, 1, 10, 16).to(device)
        img2 = torch.rand(1, 1, 10, 16).to(device)

        # evaluate function gradient
        img1 = utils.tensor_to_gradcheck_var(img1)  # to var
        img2 = utils.tensor_to_gradcheck_var(img2, requires_grad=False)  # to var
        assert gradcheck(kornia.ssim, (img1, img2, window_size), raise_exception=True)


class TestDivergenceLoss:
    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
    ])
    def test_js_div_loss_2d(self, device, input, target, expected):
        actual = kornia.losses.js_div_loss_2d(input.to(device), target.to(device)).item()
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
    ])
    def test_kl_div_loss_2d(self, device, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input.to(device), target.to(device)).item()
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125),
         torch.full((1, 1, 2, 4), 0.125),
         torch.full((1, 1), 0.0)),
        (torch.full((1, 7, 2, 4), 0.125),
         torch.full((1, 7, 2, 4), 0.125),
         torch.full((1, 7), 0.0)),
        (torch.full((1, 7, 2, 4), 0.125),
         torch.zeros((1, 7, 2, 4)),
         torch.full((1, 7), 0.0)),
        (torch.zeros((1, 7, 2, 4)),
         torch.full((1, 7, 2, 4), 0.125),
         torch.full((1, 7), math.inf)),
    ])
    def test_kl_div_loss_2d_without_reduction(self, device, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input.to(device), target.to(device), reduction='none')
        assert_allclose(actual, expected.to(device))

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
    ])
    def test_noncontiguous_kl(self, device, input, target, expected):
        input = input.to(device).view(input.shape[::-1]).T
        target = target.to(device).view(target.shape[::-1]).T
        actual = kornia.losses.kl_div_loss_2d(input, target).item()
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
    ])
    def test_noncontiguous_js(self, device, input, target, expected):
        input = input.to(device).view(input.shape[::-1]).T
        target = target.to(device).view(target.shape[::-1]).T
        actual = kornia.losses.js_div_loss_2d(input, target).item()
        assert_allclose(actual, expected)

    def test_gradcheck_kl(self, device):
        input = torch.rand(1, 1, 10, 16).to(device)
        target = torch.rand(1, 1, 10, 16).to(device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        target = utils.tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.kl_div_loss_2d, (input, target),
                         raise_exception=True)

    def test_gradcheck_js(self, device):
        input = torch.rand(1, 1, 10, 16).to(device)
        target = torch.rand(1, 1, 10, 16).to(device)

        # evaluate function gradient
        input = utils.tensor_to_gradcheck_var(input)  # to var
        target = utils.tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(kornia.losses.js_div_loss_2d, (input, target),
                         raise_exception=True)

    def test_jit_trace_kl(self, device, dtype):
        input = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        target = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        args = (input, target)
        op = kornia.losses.kl_div_loss_2d
        op_jit = torch.jit.trace(op, args)
        assert_allclose(op(*args), op_jit(*args), rtol=0, atol=1e-5)

    def test_jit_trace_js(self, device, dtype):
        input = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        target = torch.randn((2, 4, 10, 16), dtype=dtype, device=device)
        args = (input, target)
        op = kornia.losses.js_div_loss_2d
        op_jit = torch.jit.trace(op, args)
        assert_allclose(op(*args), op_jit(*args), rtol=0, atol=1e-5)


class TestTotalVariation:
    # Total variation of constant vectors is 0
    @pytest.mark.parametrize('input,expected', [
        (torch.ones(3, 4, 5), torch.zeros(())),
        (2 * torch.ones(2, 3, 4, 5), torch.zeros(2)),
    ])
    def test_tv_on_constant(self, device, input, expected):
        actual = kornia.losses.total_variation(input.to(device))
        assert_allclose(actual, expected.to(device))

    # Total variation for 3D tensors
    @pytest.mark.parametrize('input,expected', [
        (torch.tensor([[[0.11747694, 0.5717714, 0.89223915, 0.2929412, 0.63556224],
                        [0.5371079, 0.13416398, 0.7782737, 0.21392655, 0.1757018],
                        [0.62360305, 0.8563448, 0.25304103, 0.68539226, 0.6956515],
                        [0.9350611, 0.01694632, 0.78724295, 0.4760313, 0.73099905]],

                       [[0.4788819, 0.45253807, 0.932798, 0.5721999, 0.7612051],
                        [0.5455887, 0.8836531, 0.79551977, 0.6677338, 0.74293613],
                        [0.4830376, 0.16420758, 0.15784949, 0.21445751, 0.34168917],
                        [0.8675162, 0.5468113, 0.6117004, 0.01305223, 0.17554593]],

                       [[0.6423703, 0.5561105, 0.54304767, 0.20339686, 0.8553698],
                        [0.98024786, 0.31562763, 0.10122144, 0.17686582, 0.26260805],
                        [0.20522952, 0.14523649, 0.8601968, 0.02593213, 0.7382898],
                        [0.71935296, 0.9625162, 0.42287344, 0.07979459, 0.9149871]]]), torch.tensor(33.001236)),
        (torch.tensor([[[0.09094203, 0.32630223, 0.8066123],
                        [0.10921168, 0.09534764, 0.48588026]]]), torch.tensor(1.6900232)),
    ])
    def test_tv_on_3d(self, device, input, expected):
        assert_allclose(kornia.losses.total_variation(input.to(device)), expected.to(device))

    # Total variation for 4D tensors
    @pytest.mark.parametrize('input,expected', [
        (torch.tensor([[[[0.8756, 0.0920],
                         [0.8034, 0.3107]],
                        [[0.3069, 0.2981],
                         [0.9399, 0.7944]],
                        [[0.6269, 0.1494],
                         [0.2493, 0.8490]]],
                       [[[0.3256, 0.9923],
                         [0.2856, 0.9104]],
                        [[0.4107, 0.4387],
                           [0.2742, 0.0095]],
                        [[0.7064, 0.3674],
                           [0.6139, 0.2487]]]]), torch.tensor([5.0054283, 3.1870906])),
        (torch.tensor([[[[0.1104, 0.2284, 0.4371],
                         [0.4569, 0.1906, 0.8035]]],
                       [[[0.0552, 0.6831, 0.8310],
                         [0.3589, 0.5044, 0.0802]]],
                       [[[0.5078, 0.5703, 0.9110],
                         [0.4765, 0.8401, 0.2754]]]]), torch.tensor([1.9565653, 2.5786452, 2.2681699])),
    ])
    def test_tv_on_4d(self, device, input, expected):
        assert_allclose(kornia.losses.total_variation(input.to(device)), expected.to(device))

    # Expect ValueError to be raised when tensors of ndim != 3 or 4 are passed
    @pytest.mark.parametrize('input', [
        torch.rand(2, 3, 4, 5, 3),
        torch.rand(3, 1),
    ])
    def test_tv_on_invalid_dims(self, device, input):
        with pytest.raises(ValueError) as ex_info:
            kornia.losses.total_variation(input.to(device))

    # Expect TypeError to be raised when non-torch tensors are passed
    @pytest.mark.parametrize('input', [
        1,
        [1, 2],
    ])
    def test_tv_on_invalid_types(self, input):
        with pytest.raises(TypeError) as ex_info:
            kornia.losses.total_variation(input)


class TestPSNRLoss:
    def test_smoke(self, device):
        input = torch.rand(2, 3, 3, 2).to(device)
        target = torch.rand(2, 3, 3, 2).to(device)

        criterion = kornia.losses.PSNRLoss(1.0)
        loss = criterion(input, target)

        assert loss.shape == tuple()

    def test_same_input(self, device):
        input = torch.rand(2, 3, 3, 2).to(device)
        target = input.clone()

        criterion = kornia.losses.PSNRLoss(1.0)
        loss = criterion(input, target)

        assert_allclose(loss, torch.tensor(float('inf')).to(device))

    def test_type(self):
        # Expecting an exception
        # since we pass integers instead of torch tensors
        criterion = kornia.losses.PSNRLoss(1.0)
        with pytest.raises(Exception) as e:
            criterion(1, 2)

    def test_shape(self):
        # Expecting an exception
        # since we pass tensors of different shapes
        criterion = kornia.losses.PSNRLoss(1.0)
        with pytest.raises(Exception) as e:
            criterion(torch.rand(2, 3, 3, 2), torch.rand(2, 3, 3))

    def test_simple(self, device):
        assert_allclose(
            kornia.losses.psnr_loss(
                torch.ones(1).to(device),
                1.2 * torch.ones(1).to(device),
                2),
            torch.tensor(20.0).to(device))

    @pytest.mark.skip(reason="TODO: implement me")
    def test_jit(self):
        pass

    def test_gradcheck(self, device):
        input = torch.rand(2, 3, 3, 2).to(device)
        target = torch.rand(2, 3, 3, 2).to(device)

        input = utils.tensor_to_gradcheck_var(input)  # to var
        target = utils.tensor_to_gradcheck_var(target)  # to var
        assert gradcheck(
            kornia.losses.psnr_loss, (input, target, 1.0), raise_exception=True
        )
