import pytest
import torch
from torch import Tensor
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils
from kornia.constants import pi
from kornia.testing import BaseTester, assert_close, tensor_to_gradcheck_var


class TestInvert:
    def test_smoke(self, device, dtype):
        img = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        assert kornia.enhance.invert(img) is not None

    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 4, 3, 3), (1, 3, 2, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.invert(img, torch.tensor(1.0))
        assert out.shape == shape

    def test_max_val_1(self, device, dtype):
        img = torch.ones(1, 3, 4, 4, device=device, dtype=dtype)
        out = kornia.enhance.invert(img, torch.tensor(1.0))
        assert_close(out, torch.zeros_like(out))

    def test_max_val_255(self, device, dtype):
        img = 255.0 * torch.ones(1, 3, 4, 4, device=device, dtype=dtype)
        out = kornia.enhance.invert(img, torch.tensor(255.0))
        assert_close(out, torch.zeros_like(out))

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 1, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        max_val = torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.enhance.invert, (img, max_val), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.invert
        op_jit = torch.jit.script(op)
        assert_close(op(img), op_jit(img))

    @pytest.mark.nn
    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.invert
        op_mod = kornia.enhance.Invert()
        assert_close(op(img), op_mod(img))


class TestAdjustSaturation:
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_saturation(img, 1.0)
        assert out.shape == shape

    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality_with_gray_subtraction(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_saturation_with_gray_subtraction(img, 1.0)
        assert out.shape == shape

    def test_saturation_one(self, device, dtype):
        data = torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustSaturation(1.0)
        assert_close(f(data), expected)

    def test_saturation_with_gray_subtraction_one(self, device, dtype):
        data = torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustSaturationWithGraySubtraction(1.0)
        assert_close(f(data), expected)

    def test_saturation_one_batch(self, device, dtype):
        data = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = data
        f = kornia.enhance.AdjustSaturation(torch.ones(2))
        assert_close(f(data), expected)

    def test_saturation_with_gray_subtraction_one_batch(self, device, dtype):
        data = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = data
        f = kornia.enhance.AdjustSaturationWithGraySubtraction(torch.ones(2))
        assert_close(f(data), expected)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_saturation, (img, 2.0), raise_exception=True)

    def test_gradcheck_with_gray_subtraction(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_saturation_with_gray_subtraction,
                         (img, 2.0),
                         raise_exception=True)


class TestAdjustHue:
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_hue(img, 3.141516)
        assert out.shape == shape

    def test_hue_one(self, device, dtype):
        data = torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustHue(0.0)
        assert_close(f(data), expected)

    def test_hue_one_batch(self, device, dtype):
        data = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = data
        f = kornia.enhance.AdjustHue(torch.tensor([0, 0]))
        assert_close(f(data), expected)

    def test_hue_flip_batch(self, device, dtype):
        data = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        pi_t = torch.tensor([-pi, pi], device=device, dtype=dtype)
        f = kornia.enhance.AdjustHue(pi_t)

        result = f(data)
        assert_close(result, result.flip(0))

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_hue, (img, 2.0), raise_exception=True)


class TestAdjustGamma:
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_gamma(img, 1.0)
        assert out.shape == shape

    def test_gamma_zero(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.ones_like(data)

        f = kornia.enhance.AdjustGamma(0.0)
        assert_close(f(data), expected)

    def test_gamma_one(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustGamma(1.0)
        assert_close(f(data), expected)

    def test_gamma_one_gain_two(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]], device=device, dtype=dtype
        )  # 3x2x2

        f = kornia.enhance.AdjustGamma(1.0, 2.0)
        assert_close(f(data), expected)

    def test_gamma_two(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.25, 0.25], [0.25, 0.25]], [[0.0625, 0.0625], [0.0625, 0.0625]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        f = kornia.enhance.AdjustGamma(2.0)
        assert_close(f(data), expected)

    def test_gamma_two_batch(self, device, dtype):
        data = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.25, 0.25], [0.25, 0.25]], [[0.0625, 0.0625], [0.0625, 0.0625]]],
                [[[1.0, 1.0], [1.0, 1.0]], [[0.25, 0.25], [0.25, 0.25]], [[0.0625, 0.0625], [0.0625, 0.0625]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        p1 = torch.tensor([2.0, 2.0], device=device, dtype=dtype)
        p2 = torch.ones(2, device=device, dtype=dtype)

        f = kornia.enhance.AdjustGamma(p1, gain=p2)
        assert_close(f(data), expected)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_gamma, (img, 1.0, 2.0), raise_exception=True)


class TestAdjustContrast:
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_contrast(img, 0.5)
        assert out.shape == shape

    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality_with_mean_subtraction(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_contrast_with_mean_subtraction(img, 0.5)
        assert out.shape == shape

    def test_factor_zero(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.zeros_like(data)

        f = kornia.enhance.AdjustContrast(0.0)
        assert_close(f(data), expected)

    def test_factor_zero_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [[[0.6210, 0.6210], [0.6210, 0.6210]], [[0.6210, 0.6210], [0.6210, 0.6210]],
             [[0.6210, 0.6210], [0.6210, 0.6210]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(0.0)
        assert_close(f(data), expected)

    def test_factor_one_acumulative(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustContrast(1.0)
        assert_close(f(data), expected)

    def test_factor_one_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(1.0)
        assert_close(f(data), expected)

    def test_factor_two(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]], device=device, dtype=dtype
        )  # 3x2x2

        f = kornia.enhance.AdjustContrast(2.0)
        assert_close(f(data), expected)

    def test_factor_two_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [[[1.0000, 1.0000], [1.0000, 1.0000]], [[0.3790, 0.3790], [0.3790, 0.3790]],
             [[0.0000, 0.0000], [0.0000, 0.0000]]],
            device=device,
            dtype=dtype
        )  # 3x2x2

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(2.0)
        assert_close(f(data), expected)

    def test_factor_tensor(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.25, 0.25], [0.25, 0.25]],
                [[0.5, 0.5], [0.5, 0.5]],
            ],
            device=device,
            dtype=dtype,
        )  # 4x2x2

        expected = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.375, 0.375], [0.375, 0.375]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )  # 4x2x2

        factor = torch.tensor([0, 1, 1.5, 2], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrast(factor)
        assert_close(f(data), expected)

    def test_factor_tensor_color(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.3, 0.3], [0.3, 0.3]], [[0.6, 0.6], [0.6, 0.6]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.6, 0.6], [0.6, 0.6]], [[1.0, 1.0], [1.0, 1.0]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        factor = torch.tensor([1, 2], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrast(factor)
        assert_close(f(data), expected)

    def test_factor_tensor_color_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.3, 0.3], [0.3, 0.3]], [[0.6, 0.6], [0.6, 0.6]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.3555, 0.3555], [0.3555, 0.3555]], [[0.9555, 0.9555], [0.9555, 0.9555]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        factor = torch.tensor([1, 2], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(factor)
        assert_close(f(data), expected)

    def test_factor_tensor_shape(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [
                [
                    [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
                    [[0.5, 0.5, 0.25], [0.5, 0.5, 0.25]],
                    [[0.25, 0.25, 0.25], [0.6, 0.6, 0.3]],
                ],
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.25]],
                    [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]],
                    [[0.6, 0.6, 0.0], [0.3, 0.2, 0.1]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x3

        expected = torch.tensor(
            [
                [
                    [[1.0, 1.0, 0.75], [1.0, 1.0, 0.75]],
                    [[0.75, 0.75, 0.375], [0.75, 0.75, 0.375]],
                    [[0.375, 0.375, 0.375], [0.9, 0.9, 0.45]],
                ],
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
                    [[0.6, 0.6, 0.8], [0.6, 0.6, 0.8]],
                    [[1.0, 1.0, 0.0], [0.6, 0.4, 0.2]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x3

        factor = torch.tensor([1.5, 2.0], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrast(factor)
        assert_close(f(data), expected)

    def test_factor_tensor_shape_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [
                [
                    [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
                    [[0.5, 0.5, 0.25], [0.5, 0.5, 0.25]],
                    [[0.25, 0.25, 0.25], [0.6, 0.6, 0.3]],
                ],
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.25]],
                    [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]],
                    [[0.6, 0.6, 0.0], [0.3, 0.2, 0.1]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x3

        expected = torch.tensor(
            [
                [
                    [[1.0000, 1.0000, 0.4818], [1.0000, 1.0000, 0.4818]],
                    [[0.4818, 0.4818, 0.1068], [0.4818, 0.4818, 0.1068]],
                    [[0.1068, 0.1068, 0.1068], [0.6318, 0.6318, 0.1818]]
                ],
                [
                    [[0.0000, 0.0000, 1.0000], [0.0000, 0.0000, 0.2079]],
                    [[0.3079, 0.3079, 0.5079], [0.3079, 0.3079, 0.5079]],
                    [[0.9079, 0.9079, 0.0000], [0.3079, 0.1079, 0.0000]]
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x3

        factor = torch.tensor([1.5, 2.0], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(factor)

        assert_close(f(data), expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_contrast, (img, 2.0), raise_exception=True)

    def test_gradcheck_with_mean_subtraction(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_contrast_with_mean_subtraction, (img, 2.0), raise_exception=True)


class TestAdjustBrightness:
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_brightness(img, 1.0)
        assert out.shape == shape

    def test_factor_zero(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        f = kornia.enhance.AdjustBrightness(0.0)
        assert_close(f(data), data)

    def test_factor_saturat(self, device, dtype):
        # prepare input data
        data = 0.5 * torch.ones(1, 4, 3, 2, device=device, dtype=dtype)
        ones = torch.ones_like(data)

        f = kornia.enhance.AdjustBrightness(0.6)
        assert_close(f(data), ones)

    @pytest.mark.parametrize("channels", [1, 4, 5])
    def test_factor_tensor(self, device, dtype, channels):
        # prepare input data
        data = torch.ones(channels, 2, 3, device=device, dtype=dtype)  # Cx2x3
        factor = torch.arange(0, 1, channels, device=device, dtype=dtype)
        out = kornia.enhance.adjust_brightness(data, factor)
        assert out.shape == (channels, 2, 3)

    def test_factor_tensor_color_accumulative(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.3, 0.3], [0.3, 0.3]], [[0.6, 0.6], [0.6, 0.6]]],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        expected = torch.tensor(
            [
                [[[0.2500, 0.2500], [0.2500, 0.2500]], [[0.1250, 0.1250], [0.1250, 0.1250]],
                 [[0.0625, 0.0625], [0.0625, 0.0625]]],
                [[[0.0000, 0.0000], [0.0000, 0.0000]], [[0.0300, 0.0300], [0.0300, 0.0300]],
                 [[0.0600, 0.0600], [0.0600, 0.0600]]]
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        factor = torch.tensor([0.25, 0.1], device=device, dtype=dtype)

        f = kornia.enhance.AdjustBrightnessAccumulative(factor)
        assert_close(f(data), expected)

    def test_gradcheck_additive(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_brightness, (img, 1.0), raise_exception=True)

    def test_gradcheck_accumulative(self, device, dtype):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        img = tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.enhance.adjust_brightness_accumulative,
                         (img, 2.0),
                         raise_exception=True)

class TestSigmoid:
    f = kornia.enhance.adjust_sigmoid

    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4)])
    def test_shape_sigmoid(self, shape, device):
        inputs = torch.ones(*shape, device=device)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size(shape)


class TestEqualize:
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4), (3, 2, 3, 3, 4, 4)])
    def test_shape_equalize(self, shape, device, dtype):
        inputs = torch.ones(*shape, device=device, dtype=dtype)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size(shape)

    def test_shape_equalize_batch(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5

        inputs = torch.ones(bs, channels, height, width, device=device, dtype=dtype)
        f = kornia.enhance.equalize

        assert f(inputs).shape == torch.Size([bs, channels, height, width])

    def test_equalize(self, device, dtype):
        bs, channels, height, width = 1, 3, 20, 20

        inputs = self.build_input(bs, channels, height, width, device=device, dtype=dtype)

        row_expected = torch.tensor(
            [
                0.0000,
                0.07843,
                0.15686,
                0.2353,
                0.3137,
                0.3922,
                0.4706,
                0.5490,
                0.6275,
                0.7059,
                0.7843,
                0.8627,
                0.9412,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            device=device,
            dtype=dtype,
        )

        expected = self.build_input(bs, channels, height, width, device=device, dtype=dtype, row=row_expected)

        f = kornia.enhance.equalize

        assert_close(f(inputs), expected, rtol=1e-4, atol=1e-4)

    def test_equalize_batch(self, device, dtype):
        bs, channels, height, width = 2, 3, 20, 20

        inputs = self.build_input(bs, channels, height, width, device=device, dtype=dtype)

        row_expected = torch.tensor(
            [
                0.0000,
                0.07843,
                0.15686,
                0.2353,
                0.3137,
                0.3922,
                0.4706,
                0.5490,
                0.6275,
                0.7059,
                0.7843,
                0.8627,
                0.9412,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            device=device,
            dtype=dtype,
        )

        expected = self.build_input(bs, channels, height, width, device=device, dtype=dtype, row=row_expected)

        f = kornia.enhance.equalize

        assert_close(f(inputs), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 1, 2, 3, 3
        inputs = torch.ones(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(kornia.enhance.equalize, (inputs,), raise_exception=True)

    @pytest.mark.skip(reason="args and kwargs in decorator")
    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 3
        inp = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)

        op = kornia.enhance.equalize
        op_script = torch.jit.script(op)

        assert_close(op(inp), op_script(inp))

    @staticmethod
    def build_input(batch_size, channels, height, width, device, dtype, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height).to(device, dtype)
        image = torch.stack([channel] * channels).to(device, dtype)
        batch = torch.stack([image] * batch_size).to(device, dtype)

        return batch


class TestEqualize3D:
    @pytest.mark.parametrize("shape", [(3, 6, 10, 10), (2, 3, 6, 10, 10), (3, 2, 3, 6, 10, 10)])
    def test_shape_equalize3d(self, shape, device, dtype):
        inputs3d = torch.ones(*shape, device=device, dtype=dtype)
        f = kornia.enhance.equalize3d

        assert f(inputs3d).shape == torch.Size(shape)

    def test_shape_equalize3d_batch(self, device, dtype):
        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = torch.ones(bs, channels, depth, height, width, device=device, dtype=dtype)
        f = kornia.enhance.equalize3d

        assert f(inputs3d).shape == torch.Size([bs, channels, depth, height, width])

    def test_equalize3d(self, device, dtype):
        bs, channels, depth, height, width = 1, 3, 6, 10, 10

        inputs3d = self.build_input(bs, channels, depth, height, width, device, dtype)

        row_expected = torch.tensor(
            [0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000],
            device=device,
            dtype=dtype,
        )

        expected = self.build_input(bs, channels, depth, height, width, device, dtype, row=row_expected)

        f = kornia.enhance.equalize3d

        assert_close(f(inputs3d), expected, atol=1e-4, rtol=1e-4)

    def test_equalize3d_batch(self, device, dtype):
        bs, channels, depth, height, width = 2, 3, 6, 10, 10

        inputs3d = self.build_input(bs, channels, depth, height, width, device, dtype)

        row_expected = torch.tensor(
            [0.0000, 0.11764, 0.2353, 0.3529, 0.4706, 0.5882, 0.7059, 0.8235, 0.9412, 1.0000],
            device=device,
            dtype=dtype,
        )

        expected = self.build_input(bs, channels, depth, height, width, device, dtype, row=row_expected)

        f = kornia.enhance.equalize3d

        assert_close(f(inputs3d), expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        bs, channels, depth, height, width = 1, 2, 3, 4, 5
        inputs3d = torch.ones(bs, channels, depth, height, width, device=device, dtype=dtype)
        inputs3d = tensor_to_gradcheck_var(inputs3d)
        assert gradcheck(kornia.enhance.equalize3d, (inputs3d,), raise_exception=True)

    @pytest.mark.skip(reason="args and kwargs in decorator")
    def test_jit(self, device, dtype):
        batch_size, channels, depth, height, width = 1, 2, 1, 3, 3
        inp = torch.ones(batch_size, channels, depth, height, width, device=device, dtype=dtype)

        op = kornia.enhance.equalize3d
        op_script = torch.jit.script(op)

        assert_close(op(inp), op_script(inp))

    @staticmethod
    def build_input(batch_size, channels, depth, height, width, device, dtype, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height).to(device, dtype)
        image = torch.stack([channel] * channels).to(device, dtype)
        image3d = torch.stack([image] * depth).transpose(0, 1).to(device, dtype)
        batch = torch.stack([image3d] * batch_size).to(device, dtype)

        return batch


class TestSharpness(BaseTester):
    f = kornia.enhance.sharpness

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(TestSharpness.f(img, 0.8), Tensor)

    @pytest.mark.parametrize("shape", [(1, 1, 4, 5), (2, 3, 4, 5), (2, 5, 4, 5), (4, 5), (5, 4, 5), (2, 3, 2, 3, 4, 5)])
    @pytest.mark.parametrize("factor", [0.7, 0.8])
    def test_cardinality(self, shape, factor, device, dtype):
        inputs = torch.ones(*shape, device=device, dtype=dtype)
        assert TestSharpness.f(inputs, factor).shape == torch.Size(shape)

    def test_exception(self, device, dtype):
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)
        with pytest.raises(AssertionError):
            assert TestSharpness.f(img, [0.8, 0.9, 0.6])
        with pytest.raises(AssertionError):
            assert TestSharpness.f(img, torch.tensor([0.8, 0.9, 0.6]))
        with pytest.raises(AssertionError):
            assert TestSharpness.f(img, torch.tensor([0.8]))

    def test_value(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(1, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray(arr)
        # en = ImageEnhance.Sharpness(img).enhance(0.8)
        # np.array(en) / 255.
        expected = torch.tensor(
            [[[[0.4963, 0.7682, 0.0885], [0.1320, 0.3305, 0.6341], [0.4901, 0.8964, 0.4556]]]],
            device=device,
            dtype=dtype,
        )

        # If factor == 1, shall return original
        # TODO(jian): add test for this case
        # assert_close(TestSharpness.f(inputs, 0.), inputs, rtol=1e-4, atol=1e-4)
        assert_close(TestSharpness.f(inputs, 1.0), inputs, rtol=1e-4, atol=1e-4)
        assert_close(TestSharpness.f(inputs, 0.8), expected, rtol=1e-4, atol=1e-4)

    def test_value_batch(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray(arr)
        # en = ImageEnhance.Sharpness(img).enhance(0.8)
        # np.array(en) / 255.
        expected_08 = torch.tensor(
            [
                [[[0.4963, 0.7682, 0.0885], [0.1320, 0.3305, 0.6341], [0.4901, 0.8964, 0.4556]]],
                [[[0.6323, 0.3489, 0.4017], [0.0223, 0.2052, 0.2939], [0.5185, 0.6977, 0.8000]]],
            ],
            device=device,
            dtype=dtype,
        )

        expected_08_13 = torch.tensor(
            [
                [[[0.4963, 0.7682, 0.0885], [0.1320, 0.3305, 0.6341], [0.4901, 0.8964, 0.4556]]],
                [[[0.6323, 0.3489, 0.4017], [0.0223, 0.1143, 0.2939], [0.5185, 0.6977, 0.8000]]],
            ],
            device=device,
            dtype=dtype,
        )

        # If factor == 1, shall return original
        tol_val: float = utils._get_precision(device, dtype)
        assert_close(TestSharpness.f(inputs, 1), inputs, rtol=tol_val, atol=tol_val)
        assert_close(TestSharpness.f(inputs, torch.tensor([1.0, 1.0])), inputs, rtol=tol_val, atol=tol_val)
        assert_close(TestSharpness.f(inputs, 0.8), expected_08, rtol=tol_val, atol=tol_val)
        assert_close(TestSharpness.f(inputs, torch.tensor([0.8, 1.3])), expected_08_13, rtol=tol_val, atol=tol_val)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(TestSharpness.f, (inputs, 0.8), raise_exception=True)

    @pytest.mark.skip(reason="union type input")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = TestSharpness.f
        op_script = torch.jit.script(TestSharpness.f)
        img = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(img, 0.8)
        actual = op_script(img, 0.8)
        assert_close(actual, expected)

    # TODO: update with module when exists
    @pytest.mark.skip(reason="Not having it yet.")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        ops = TestSharpness.f
        mod = TestSharpness.f
        assert_close(ops(img), mod(img))


@pytest.mark.skipif(kornia.xla_is_available(), reason="issues with xla device")
class TestSolarize(BaseTester):
    f = kornia.enhance.solarize

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(TestSolarize.f(img, 0.8), Tensor)

    @pytest.mark.parametrize(
        "shape, thresholds, additions",
        [
            ((1, 1, 4, 5), 0.8, 0.4),
            ((4, 5), 0.8, 0.4),
            ((2, 4, 5), 0.8, None),
            ((2, 3, 2, 3, 4, 5), torch.tensor(0.8), None),
            ((2, 5, 4, 5), torch.tensor([0.8, 0.7]), None),
            ((2, 3, 4, 5), torch.tensor([0.8, 0.7]), torch.tensor([0.0, 0.4])),
        ],
    )
    def test_cardinality(self, shape, thresholds, additions, device, dtype):
        inputs = torch.ones(*shape, device=device, dtype=dtype)
        assert TestSolarize.f(inputs, thresholds, additions).shape == torch.Size(shape)

    # TODO(jian): add better assertions
    def test_exception(self, device, dtype):
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert TestSolarize.f([1.0], 0.0)

        with pytest.raises(TypeError):
            assert TestSolarize.f(img, 1)

        with pytest.raises(TypeError):
            assert TestSolarize.f(img, 0.8, 1)

    # TODO: add better cases
    def test_value(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(1, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray((255*inputs[0,0]).byte().numpy())
        # en = ImageOps.Solarize(img, 128)
        # np.array(en) / 255.
        expected = torch.tensor(
            [
                [
                    [
                        [0.49411765, 0.23529412, 0.08627451],
                        [0.12941176, 0.30588235, 0.36862745],
                        [0.48627451, 0.10588235, 0.45490196],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        # TODO(jian): precision is very bad compared to PIL
        assert_close(TestSolarize.f(inputs, 0.5), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(TestSolarize.f, (inputs, 0.8), raise_exception=True)

    # TODO: implement me
    @pytest.mark.skip(reason="union type input")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = TestSolarize.f
        op_script = torch.jit.script(op)
        img = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(img, 0.8)
        actual = op_script(img, 0.8)
        assert_close(actual, expected)

    # TODO: update with module when exists
    @pytest.mark.skip(reason="Not having it yet.")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        ops = TestSolarize.f
        mod = TestSolarize.f
        assert_close(ops(img), mod(img))


class TestPosterize(BaseTester):
    f = kornia.enhance.posterize

    def test_smoke(self, device, dtype):
        B, C, H, W = 2, 3, 4, 5
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        assert isinstance(TestPosterize.f(img, 8), Tensor)

    @pytest.mark.parametrize(
        "shape, bits",
        [
            ((1, 4, 5), 8),
            ((2, 3, 4, 5), 1),
            ((2, 3, 4, 5, 4, 5), 0),
            ((1, 4, 5), torch.tensor(8)),
            ((3, 4, 5), torch.tensor(8)),
            ((2, 5, 4, 5), torch.tensor([0, 8])),
            ((3, 3, 4, 5), torch.tensor([0, 1, 8])),
        ],
    )
    def test_cardinality(self, shape, bits, device, dtype):
        inputs = torch.ones(*shape, device=device, dtype=dtype)
        assert TestPosterize.f(inputs, bits).shape == torch.Size(shape)

    # TODO(jian): add better assertions
    def test_exception(self, device, dtype):
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert TestPosterize.f([1.0], 0.0)

        with pytest.raises(TypeError):
            assert TestPosterize.f(img, 1.0)

    # TODO(jian): add better cases
    @pytest.mark.skipif(kornia.xla_is_available(), reason="issues with xla device")
    def test_value(self, device, dtype):
        torch.manual_seed(0)

        inputs = torch.rand(1, 1, 3, 3).to(device=device, dtype=dtype)

        # Output generated is similar (1e-2 due to the uint8 conversions) to the below output:
        # img = PIL.Image.fromarray((255*inputs[0,0]).byte().numpy())
        # en = ImageOps.posterize(img, 1)
        # np.array(en) / 255.
        expected = torch.tensor(
            [[[[0.0, 0.50196078, 0.0], [0.0, 0.0, 0.50196078], [0.0, 0.50196078, 0.0]]]], device=device, dtype=dtype
        )

        assert_close(TestPosterize.f(inputs, 1), expected)
        assert_close(TestPosterize.f(inputs, 0), torch.zeros_like(inputs))
        assert_close(TestPosterize.f(inputs, 8), inputs)

    @pytest.mark.skip(reason="IndexError: tuple index out of range")
    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=dtype)
        inputs = tensor_to_gradcheck_var(inputs)
        assert gradcheck(TestPosterize.f, (inputs, 0), raise_exception=True)

    # TODO: implement me
    @pytest.mark.skip(reason="union type input")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        op = TestPosterize.f
        op_script = torch.jit.script(op)
        img = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(img, 8)
        actual = op_script(img, 8)
        assert_close(actual, expected)

    # TODO: update with module when exists
    @pytest.mark.skip(reason="Not having it yet.")
    @pytest.mark.nn
    def test_module(self, device, dtype):
        img = torch.ones(2, 3, 4, 4, device=device, dtype=dtype)
        ops = TestPosterize.f
        mod = TestPosterize.f
        assert_close(ops(img), mod(img))
