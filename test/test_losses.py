import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import device_type

import math
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestFocalLoss:
    def _test_smoke_none(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        assert kornia.losses.focal_loss(
            logits, labels, alpha=0.5, gamma=2.0, reduction="none"
        ).shape == (2, 3, 2)

    def _test_smoke_sum(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="sum"
            ).shape == ()
        )

    def _test_smoke_mean(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        assert (
            kornia.losses.focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="mean"
            ).shape == ()
        )

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        num_classes = 3
        alpha, gamma = 0.5, 2.0  # for focal loss
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.focal_loss,
            (logits, labels, alpha, gamma),
            raise_exception=True,
        )

    def test_run_all(self):
        self._test_smoke_none()
        self._test_smoke_sum()
        self._test_smoke_mean()
        self._test_gradcheck()


class TestTverskyLoss:
    def _test_smoke(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)

    def _test_all_zeros(self):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64)

        criterion = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
        loss = criterion(logits, labels)
        assert pytest.approx(loss.item(), 0.0)

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        num_classes = 3
        alpha, beta = 0.5, 0.5  # for tversky loss
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.tversky_loss,
            (logits, labels, alpha, beta),
            raise_exception=True,
        )

    def test_run_all(self):
        self._test_smoke()
        self._test_all_zeros()
        self._test_gradcheck()


class TestDiceLoss:
    def _test_smoke(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)

    def _test_all_zeros(self):
        num_classes = 3
        logits = torch.zeros(2, num_classes, 1, 2)
        logits[:, 0] = 10.0
        logits[:, 1] = 1.0
        logits[:, 2] = 1.0
        labels = torch.zeros(2, 1, 2, dtype=torch.int64)

        criterion = kornia.losses.DiceLoss()
        loss = criterion(logits, labels)
        assert pytest.approx(loss.item(), 0.0)

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        num_classes = 3
        logits = torch.rand(2, num_classes, 3, 2)
        labels = torch.rand(2, 3, 2) * num_classes
        labels = labels.long()

        logits = utils.tensor_to_gradcheck_var(logits)  # to var
        assert gradcheck(
            kornia.losses.dice_loss, (logits, labels), raise_exception=True
        )

    def test_run_all(self):
        self._test_smoke()
        self._test_all_zeros()
        self._test_gradcheck()


class TestDepthSmoothnessLoss:
    def _test_smoke(self):
        image = self.image.clone()
        depth = self.depth.clone()

        criterion = kornia.losses.InverseDepthSmoothnessLoss()
        loss = criterion(depth, image)

    # TODO: implement me
    def _test_1(self):
        pass

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        image = self.image.clone()
        depth = self.depth.clone()
        depth = utils.tensor_to_gradcheck_var(depth)  # to var
        image = utils.tensor_to_gradcheck_var(image)  # to var
        assert gradcheck(
            kornia.losses.inverse_depth_smoothness_loss,
            (depth, image),
            raise_exception=True,
        )

    @pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
    def test_run_all(self, batch_shape, device_type):
        self.image = torch.rand(batch_shape).to(torch.device(device_type))
        self.depth = torch.rand(batch_shape).to(torch.device(device_type))

        self._test_smoke()
        self._test_gradcheck()


@pytest.mark.parametrize("window_size", [5, 11])
@pytest.mark.parametrize("reduction_type", ["none", "mean", "sum"])
@pytest.mark.parametrize("batch_shape", [(1, 1, 10, 16), (2, 4, 8, 15)])
def test_ssim(batch_shape, device_type, window_size, reduction_type):
    # input data
    device = torch.device(device_type)
    img1 = torch.rand(batch_shape).to(device)
    img2 = torch.rand(batch_shape).to(device)

    ssim = kornia.losses.SSIM(window_size, reduction_type)
    ssim_loss_val = ssim(img1, img2)

    if reduction_type == "none":
        assert ssim_loss_val.shape == batch_shape
    else:
        assert ssim_loss_val.dim() == 0

    assert pytest.approx(ssim(img1, img1).sum().item(), 0.0)
    assert pytest.approx(ssim(img2, img2).sum().item(), 0.0)

    # functional
    assert_allclose(
        ssim_loss_val,
        kornia.losses.ssim(img1, img2, window_size, reduction_type),
    )

    # evaluate function gradient
    img1 = utils.tensor_to_gradcheck_var(img1)  # to var
    img2 = utils.tensor_to_gradcheck_var(img2, requires_grad=False)  # to var
    assert gradcheck(ssim, (img1, img2), raise_exception=True)


class TestDivergenceLoss:
    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.346574),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), 0.346574),
    ])
    def test_js_div_loss_2d(self, input, target, expected):
        actual = kornia.losses.js_div_loss_2d(input, target).item()
        assert actual == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize('input,target,expected', [
        (torch.full((1, 1, 2, 4), 0.125), torch.full((1, 1, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.full((1, 7, 2, 4), 0.125), 0.0),
        (torch.full((1, 7, 2, 4), 0.125), torch.zeros((1, 7, 2, 4)), 0.0),
        (torch.zeros((1, 7, 2, 4)), torch.full((1, 7, 2, 4), 0.125), math.inf),
    ])
    def test_kl_div_loss_2d(self, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input, target).item()
        assert actual == pytest.approx(expected, abs=1e-6)

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
    def test_kl_div_loss_2d_without_reduction(self, input, target, expected):
        actual = kornia.losses.kl_div_loss_2d(input, target, reduction='none')
        assert_allclose(actual, expected)


class TestTotalVariation:
    # Total variation of constant vectors is 0
    @pytest.mark.parametrize('input,expected', [
        (torch.ones(3, 4, 5), torch.zeros(())),
        (2 * torch.ones(2, 3, 4, 5), torch.zeros(2)),
    ])
    def test_tv_on_constant(self, input, expected):
        actual = kornia.losses.total_variation(input)
        assert_allclose(actual, expected)

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
    def test_tv_on_3d(self, input, expected):
        assert_allclose(kornia.losses.total_variation(input), expected)

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
    def test_tv_on_4d(self, input, expected):
        assert_allclose(kornia.losses.total_variation(input), expected)

    # Expect ValueError to be raised when tensors of ndim != 3 or 4 are passed
    @pytest.mark.parametrize('input', [
        torch.rand(2, 3, 4, 5, 3),
        torch.rand(3, 1),
    ])
    def test_tv_on_invalid_dims(self, input):
        with pytest.raises(ValueError) as ex_info:
            kornia.losses.total_variation(input)

    # Expect TypeError to be raised when non-torch tensors are passed
    @pytest.mark.parametrize('input', [
        1,
        [1, 2],
    ])
    def test_tv_on_invalid_types(self, input):
        with pytest.raises(TypeError) as ex_info:
            kornia.losses.total_variation(input)


class TestPSNRLoss:
    def test_smoke(self):
        input = torch.rand(2, 3, 3, 2)
        target = torch.rand(2, 3, 3, 2)

        criterion = kornia.losses.PSNRLoss(1.0)
        loss = criterion(input, target)

        assert loss.shape == tuple()

    def test_same_input(self):
        input = torch.rand(2, 3, 3, 2)
        target = input

        criterion = kornia.losses.PSNRLoss(1.0)
        loss = criterion(input, target)

        assert_allclose(loss, torch.tensor(float('inf')))

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

    def test_simple(self):
        assert_allclose(kornia.losses.psnr_loss(torch.ones(1), 1.2 * torch.ones(1), 2), torch.tensor(20.0))

    @pytest.mark.skip(reason="TODO: implement me")
    def test_jit(self):
        pass

    def test_gradcheck(self):
        input = torch.rand(2, 3, 3, 2).double()
        target = torch.rand(2, 3, 3, 2).double()

        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            kornia.losses.psnr_loss, (input, target, 1.0), raise_exception=True
        )
