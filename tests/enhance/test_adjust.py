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

from math import prod

import pytest
import torch
from torch import Tensor

import kornia
from kornia import enhance
from kornia.constants import pi
from kornia.core._compat import torch_version_ge
from kornia.core.check import are_checks_enabled, disable_checks, enable_checks
from kornia.core.exceptions import AsyncCheckError

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


def _sync(device) -> None:
    # MPS dispatches asynchronously, so a kernel error raised by the forward under test would
    # otherwise surface inside an unrelated later test.
    if device.type == "mps":
        torch.mps.synchronize()


class TestFloatScalarPrecision(BaseTester):
    @pytest.mark.parametrize(
        "operation,kwargs",
        [
            (kornia.enhance.adjust_hue_raw, {"factor": 0.1}),
            (kornia.enhance.adjust_gamma, {"gamma": 0.1}),
            (kornia.enhance.adjust_gamma, {"gamma": 1.0, "gain": 0.1}),
            (kornia.enhance.solarize, {"thresholds": 0.1}),
            (kornia.enhance.solarize, {"thresholds": 0.5, "additions": 0.1}),
        ],
    )
    def test_float_matches_image_dtype_tensor(self, device, dtype, operation, kwargs):
        image = torch.tensor([0.05, 0.1], device=device, dtype=dtype).repeat(1, 3, 1, 1)
        tensors = {key: image.new_tensor(value) for key, value in kwargs.items()}
        self.assert_close(operation(image, **kwargs), operation(image, **tensors), rtol=0, atol=0)


class TestInvert(BaseTester):
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
        self.assert_close(out, torch.zeros_like(out))

    def test_max_val_255(self, device, dtype):
        img = 255.0 * torch.ones(1, 3, 4, 4, device=device, dtype=dtype)
        out = kornia.enhance.invert(img, torch.tensor(255.0))
        self.assert_close(out, torch.zeros_like(out))

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 1, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        max_val = torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.enhance.invert, (img, max_val))

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.invert
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.invert
        op_mod = kornia.enhance.Invert()
        self.assert_close(op(img), op_mod(img))


class TestAdjustSaturation(BaseTester):
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
        self.assert_close(f(data), expected)

    def test_saturation_with_gray_subtraction_one(self, device, dtype):
        data = torch.tensor(
            [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustSaturationWithGraySubtraction(1.0)
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

    def test_saturation_matches_hsv_reference_per_batch_factor(self, device, dtype):
        # Keep adjust_saturation's HSV semantics while allowing its implementation to avoid
        # materializing an HSV image. The values avoid hue-sector boundaries, where equivalent
        # formulas can legitimately differ by a small floating-point rounding error.
        image = torch.tensor(
            [
                [[[0.2, 0.4]], [[0.8, 0.7]], [[0.5, 0.1]]],
                [[[0.9, 0.3]], [[0.2, 0.8]], [[0.6, 0.5]]],
            ],
            device=device,
            dtype=dtype,
        )
        factor = torch.tensor([0.0, 1.5], device=device, dtype=dtype)

        hsv = kornia.color.rgb_to_hsv(image)
        expected_hsv = hsv.clone()
        expected_hsv[:, 1:2] = (hsv[:, 1:2] * factor[:, None, None, None]).clamp(0, 1)
        expected = kornia.color.hsv_to_rgb(expected_hsv)

        self.assert_close(kornia.enhance.adjust_saturation(image, factor), expected, low_tolerance=True)

    def test_zero_delta_gradient_is_finite(self, device, dtype):
        image = torch.tensor([0.0, 0.5], device=device, dtype=dtype).view(2, 1, 1, 1).expand(2, 3, 2, 2).clone()
        image.requires_grad_()
        result = kornia.enhance.adjust_saturation(image, 1.5)
        (gradient,) = torch.autograd.grad(result.sum(), image)
        assert torch.isfinite(result).all()
        assert torch.isfinite(gradient).all()

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
    def test_integer_input_matches_hsv_reference(self, dtype):
        image = torch.tensor([1, 0, 0], dtype=dtype).view(1, 3, 1, 1)
        expected = kornia.color.hsv_to_rgb(kornia.enhance.adjust_saturation_raw(kornia.color.rgb_to_hsv(image), 0.5))
        self.assert_close(kornia.enhance.adjust_saturation(image, 0.5), expected)

    def test_per_channel_factor_raises(self, device, dtype):
        image = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            kornia.enhance.adjust_saturation(image, torch.ones(2, 3, device=device, dtype=dtype))

    @pytest.mark.device_agnostic
    def test_tied_extrema_use_symmetric_subgradient(self):
        image = torch.tensor([0.8, 0.8, 0.2], dtype=torch.float64).view(1, 3, 1, 1).requires_grad_()
        result = kornia.enhance.adjust_saturation(image, 0.5)
        (gradient,) = torch.autograd.grad(result.sum(), image)
        self.assert_close(gradient.flatten(), torch.tensor([1.25, 1.25, 0.5], dtype=torch.float64))

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
        self.assert_close(f(data), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_saturation, (img, 2.0))

    def test_gradcheck_factor(self, device):
        img = torch.rand(2, 3, 2, 3, device=device, dtype=torch.float64)
        factor = torch.tensor([0.8, 1.2], device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_saturation, (img, factor))

    def test_gradcheck_with_gray_subtraction(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_saturation_with_gray_subtraction, (img, 2.0))


class TestAdjustHue(BaseTester):
    def test_black_pixels(self, device, dtype):
        data = torch.tensor([0.75, 0.5, 0.25], device=device, dtype=dtype).view(1, 3, 1, 1).repeat(1, 1, 2, 2)
        data[..., 0, 0] = 0.0
        data.requires_grad_()

        result = kornia.enhance.adjust_hue(data, 0.1)
        assert torch.isfinite(result).all()
        self.assert_close(result[..., 0, 0], torch.zeros_like(result[..., 0, 0]))
        (gradient,) = torch.autograd.grad(result.sum(), data)
        assert torch.isfinite(gradient).all()

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
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
        self.assert_close(result, result.flip(0))

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_hue, (img, 2.0))

    def test_hue_raw_wraps_negative_sum_5326(self, device, dtype):
        data = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[0.0, 0.0]]]], device=device, dtype=dtype).view(
            1, 3, 1, 2
        )

        hsv = kornia.color.rgb_to_hsv(data)
        out = kornia.enhance.adjust_hue_raw(hsv, -0.5)

        self.assert_close(out[:, 0], torch.remainder(hsv[:, 0] - 0.5, 2 * pi))
        assert (out[:, 0] >= 0).all()

    def test_hue_raw_wrap_never_returns_the_period_5326(self, device, dtype):
        # 0 - eps / 16 is below 0, but 2π - eps / 16 rounds to 2π in this dtype; the wrapped hue must be 0, not 2π.
        hsv = torch.tensor([0.0, 0.5, 0.5], device=device, dtype=dtype).view(1, 3, 1, 1)
        factor = torch.tensor(-torch.finfo(dtype).eps / 16, device=device, dtype=dtype)
        hue = kornia.enhance.adjust_hue_raw(hsv, factor)[0, 0, 0, 0]
        assert hue.item() == 0.0


class TestAdjustGamma(BaseTester):
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 3, 3), (4, 3, 3, 1, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.rand(shape, device=device, dtype=dtype)
        out = kornia.enhance.adjust_gamma(img, 1.0)
        assert out.shape == shape

    @pytest.mark.parametrize("gamma,gain", [(0.1, 0.1234567), (0.5, 0.9), (2.0, 1.0), (1e-50, 1e-50)])
    def test_scalar_and_tensor_parameters(self, device, dtype, gamma, gain):
        image = torch.linspace(0, 1, 33, device=device, dtype=dtype).reshape(1, 3, 11)
        expected = kornia.enhance.adjust_gamma(image, image.new_tensor([gamma]), image.new_tensor([gain]))
        self.assert_close(kornia.enhance.adjust_gamma(image, gamma, gain), expected)

    @pytest.mark.parametrize("parameter", ["gamma", "gain"])
    def test_negative_scalar_rounds_to_zero(self, device, dtype, parameter):
        if dtype == torch.float64:
            pytest.skip("The negative value is representable in float64")
        image = torch.full((1, 2, 2), 0.5, device=device, dtype=dtype)
        kwargs = {"gamma": 1.0, "gain": 1.0, parameter: -1e-50}
        expected = kornia.enhance.adjust_gamma(image, **{**kwargs, parameter: 0.0})
        self.assert_close(kornia.enhance.adjust_gamma(image, **kwargs), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("gamma", [1e-50, 0.99999999, 1.00000001, 1.99999999])
    def test_rounded_scalar_exponent_at_zero(self, device, gamma):
        image = torch.tensor([0.0, 0.5, 1.0], device=device, dtype=torch.float32, requires_grad=True)
        expected = kornia.enhance.adjust_gamma(image, image.new_tensor([gamma]))
        actual = kornia.enhance.adjust_gamma(image, gamma)
        # MPS scalar/tensor power kernels can round nonzero pixels differently.
        self.assert_close(actual, expected)
        self.assert_close(actual[0], expected[0], rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual.sum(), image)[0]
        expected_grad = torch.autograd.grad(expected.sum(), image)[0]
        self.assert_close(actual_grad, expected_grad)
        self.assert_close(actual_grad[0], expected_grad[0], rtol=0, atol=0)

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("value", [-0.1, float("nan")])
    @pytest.mark.parametrize("parameter", ["gamma", "gain"])
    def test_invalid_scalar_parameters(self, parameter, value):
        image = torch.ones(1, 1, 2, 2)
        with pytest.raises(RuntimeError, match="must be non-negative"):
            kornia.enhance.adjust_gamma(image, **{"gamma": 1.0, "gain": 1.0, parameter: value})

    def test_tensor_parameter_gradients(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 gradcheck")
        image = torch.full((2, 1, 2, 2), 0.4, device=device, dtype=torch.float64, requires_grad=True)
        gamma = torch.tensor([0.8, 1.2], device=device, dtype=torch.float64, requires_grad=True)
        gain = torch.tensor([0.7, 0.9], device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kornia.enhance.adjust_gamma, (image, gamma, gain))

    @pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 0.9, 2.0])
    @pytest.mark.parametrize("scalar", [True, False])
    def test_black_pixel_gradient(self, device, dtype, gamma, scalar):
        # For 0 < gamma < 1, x ** gamma has an infinite derivative at x = 0; the gradient there is 0, as for the other
        # exponents. x = 1 is left out: the derivative of the output clamp at its bound differs across torch releases.
        image = torch.tensor([0.0, 0.25, 0.5], device=device, dtype=dtype).reshape(1, 1, 3).requires_grad_()
        param = gamma if scalar else torch.tensor([gamma], device=device, dtype=dtype)
        out = kornia.enhance.adjust_gamma(image, param)
        (grad,) = torch.autograd.grad(out.sum(), image)
        # in the image dtype, as adjust_gamma rounds the exponent
        exponent = torch.tensor(gamma, device=device, dtype=dtype)
        expected = image.detach().pow(exponent)
        expected_grad = exponent * image.detach().pow(exponent - 1)
        expected_grad[..., 0] = 0.0
        self.assert_close(out, expected)
        assert torch.isfinite(grad).all()
        self.assert_close(grad, expected_grad)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_scalar_export(self, device, dtype):
        image = torch.full((1, 2, 2), 0.4, device=device, dtype=dtype)
        module = kornia.enhance.AdjustGamma(0.7, 0.9)
        exported = torch.export.export(module, (image,), strict=True).module()
        self.assert_close(exported(image), module(image))

    def test_scalar_jit(self, device, dtype):
        image = torch.full((1, 2, 2), 0.4, device=device, dtype=dtype)
        module = kornia.enhance.AdjustGamma(0.7, 0.9)
        scripted = torch.jit.script(module)
        self.assert_close(scripted(image), module(image))

    def test_gamma_zero(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.ones_like(data)

        f = kornia.enhance.AdjustGamma(0.0)
        self.assert_close(f(data), expected)

    def test_gamma_one(self, device, dtype):
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustGamma(1.0)
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.ones(batch_size, channels, height, width, device=device, dtype=torch.float)
        self.gradcheck(kornia.enhance.adjust_gamma, (img, 1.0, 2.0))

    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.adjust_gamma
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img, 1.5), op_optimized(img, 1.5))


class TestAdjustContrast(BaseTester):
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

    @pytest.mark.parametrize("shape", [(4, 2, 3), (2, 1, 2, 3), (2, 3, 4, 2, 3)])
    def test_non_rgb_mean_with_leading_dimensions(self, device, dtype, shape):
        numel = torch.Size(shape).numel()
        image = torch.arange(numel, device=device, dtype=dtype).reshape(shape) / numel
        factor = torch.linspace(0.25, 0.75, image[..., 0, 0, 0].numel(), device=device, dtype=dtype).reshape(
            image.shape[:-3]
        )
        factor_broadcast = factor.reshape(*factor.shape, 1, 1, 1)
        expected = image * factor_broadcast + image.mean((-3, -2, -1), keepdim=True) * (1 - factor_broadcast)

        actual = kornia.enhance.adjust_contrast_with_mean_subtraction(image, factor)

        assert actual.shape == shape
        self.assert_close(actual, expected)

    def test_factor_zero(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.zeros_like(data)

        f = kornia.enhance.AdjustContrast(0.0)
        self.assert_close(f(data), expected)

    def test_factor_zero_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [
                [[0.6210, 0.6210], [0.6210, 0.6210]],
                [[0.6210, 0.6210], [0.6210, 0.6210]],
                [[0.6210, 0.6210], [0.6210, 0.6210]],
            ],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(0.0)
        self.assert_close(f(data), expected)

    def test_factor_one_acumulative(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustContrast(1.0)
        self.assert_close(f(data), expected)

    def test_negative_factor_errors(self, device, dtype):
        if device.type != "cpu":
            pytest.skip("the value check raises during the call only on CPU; see test_mps_checks_the_factor")
        img = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        factor = torch.tensor([-0.5], device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="non-negative"):
            kornia.enhance.adjust_contrast(img, factor)

    def test_mps_checks_the_factor(self, device, dtype):
        # The factor is checked on MPS as well: from torch 2.13 by the asynchronous assert, which raises at the next
        # synchronization, and before, without that kernel, on the host during the call.
        if device.type != "mps":
            pytest.skip("pins the MPS behavior")
        img = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        factor = torch.tensor([-0.5], device=device, dtype=dtype)
        if torch_version_ge(2, 13):
            kornia.enhance.adjust_contrast(img, factor)
            with pytest.raises(RuntimeError, match="non-negative"):
                torch.mps.synchronize()
        else:
            with pytest.raises(AsyncCheckError, match="non-negative"):
                kornia.enhance.adjust_contrast(img, factor)

    def test_factor_check_is_skipped_when_checks_are_disabled(self, device, dtype):
        img = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        factor = torch.tensor([-0.5], device=device, dtype=dtype)
        were_checks_enabled = are_checks_enabled()
        disable_checks()
        try:
            out = kornia.enhance.adjust_contrast(img, factor, clip_output=False)
        finally:
            if were_checks_enabled:
                enable_checks()
        self.assert_close(out, img * -0.5)

    def test_vmap_over_the_factor(self, device, dtype):
        # vmap has no batching rule for the value assert, which is skipped under functorch transforms.
        img = torch.rand(3, 4, 4, device=device, dtype=dtype)
        factors = torch.tensor([0.5, 1.0, 1.5], device=device, dtype=dtype)
        out = torch.vmap(lambda factor: kornia.enhance.adjust_contrast(img, factor))(factors)
        expected = torch.stack([kornia.enhance.adjust_contrast(img, factor) for factor in factors])
        self.assert_close(out, expected)

    def test_factor_one_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = data.clone()

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(1.0)
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

    def test_factor_two_with_mean_subtraction(self, device, dtype):
        # prepare input data
        data = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.25, 0.25], [0.25, 0.25]]],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        expected = torch.tensor(
            [
                [[1.0000, 1.0000], [1.0000, 1.0000]],
                [[0.3790, 0.3790], [0.3790, 0.3790]],
                [[0.0000, 0.0000], [0.0000, 0.0000]],
            ],
            device=device,
            dtype=dtype,
        )  # 3x2x2

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(2.0)
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
        self.assert_close(f(data), expected)

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
                    [[0.1068, 0.1068, 0.1068], [0.6318, 0.6318, 0.1818]],
                ],
                [
                    [[0.0000, 0.0000, 1.0000], [0.0000, 0.0000, 0.2079]],
                    [[0.3079, 0.3079, 0.5079], [0.3079, 0.3079, 0.5079]],
                    [[0.9079, 0.9079, 0.0000], [0.3079, 0.1079, 0.0000]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x3

        factor = torch.tensor([1.5, 2.0], device=device, dtype=dtype)

        f = kornia.enhance.AdjustContrastWithMeanSubtraction(factor)

        self.assert_close(f(data), expected, low_tolerance=True)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_contrast, (img, 2.0))

    def test_gradcheck_with_mean_subtraction(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_contrast_with_mean_subtraction, (img, 2.0))

    # Regression for #4806: every channel count must use each image's own mean.
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_mean_subtraction_is_batch_independent_4806(self, device, dtype, channels):
        bright = torch.full((1, channels, 2, 2), 0.8, device=device, dtype=dtype)
        dark = torch.full((1, channels, 2, 2), 0.2, device=device, dtype=dtype)
        factor = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
        alone = kornia.enhance.adjust_contrast_with_mean_subtraction(dark, factor[:1])
        batched = kornia.enhance.adjust_contrast_with_mean_subtraction(torch.cat([bright, dark]), factor)[1:]
        self.assert_close(alone, dark)
        self.assert_close(batched, alone)

    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.adjust_contrast
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img, 1.5), op_optimized(img, 1.5))


class TestAdjustBrightness(BaseTester):
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
        self.assert_close(f(data), data)

    def test_factor_saturat(self, device, dtype):
        # prepare input data
        data = 0.5 * torch.ones(1, 4, 3, 2, device=device, dtype=dtype)
        ones = torch.ones_like(data)

        f = kornia.enhance.AdjustBrightness(0.6)
        self.assert_close(f(data), ones)

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
                [
                    [[0.2500, 0.2500], [0.2500, 0.2500]],
                    [[0.1250, 0.1250], [0.1250, 0.1250]],
                    [[0.0625, 0.0625], [0.0625, 0.0625]],
                ],
                [
                    [[0.0000, 0.0000], [0.0000, 0.0000]],
                    [[0.0300, 0.0300], [0.0300, 0.0300]],
                    [[0.0600, 0.0600], [0.0600, 0.0600]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        factor = torch.tensor([0.25, 0.1], device=device, dtype=dtype)

        f = kornia.enhance.AdjustBrightnessAccumulative(factor)
        self.assert_close(f(data), expected)

    def test_gradcheck_additive(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_brightness, (img, 1.0))

    def test_gradcheck_accumulative(self, device):
        batch_size, channels, height, width = 2, 3, 4, 5
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_brightness_accumulative, (img, 2.0))

    def test_accumulative_identity_factor_clamps_by_default(self, device, dtype):
        # The multiplicative identity 1 still clamps into [0, 1] unless clip_output=False, and the module
        # form always clamps.
        values = torch.tensor([-0.5, 0.5, 2.0], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        expected = values.clamp(0.0, 1.0)
        self.assert_close(kornia.enhance.adjust_brightness_accumulative(values, 1.0), expected)
        self.assert_close(kornia.enhance.AdjustBrightnessAccumulative(1.0)(values), expected)
        self.assert_close(kornia.enhance.adjust_brightness_accumulative(values, 1.0, clip_output=False), values)
        self.assert_close(kornia.enhance.adjust_brightness_accumulative(values, 0.0), torch.zeros_like(values))


class TestAdjustSigmoid(BaseTester):
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4)])
    def test_shape_sigmoid(self, shape, device):
        inputs = torch.ones(*shape, device=device)
        f = kornia.enhance.adjust_sigmoid
        assert f(inputs).shape == torch.Size(shape)

    def test_sigmoid(self, device, dtype):
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
                [
                    [[0.99330715, 0.99330715], [0.99330715, 0.99330715]],
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.07585818, 0.07585818], [0.07585818, 0.07585818]],
                ],
                [
                    [[0.00669285, 0.00669285], [0.00669285, 0.00669285]],
                    [[0.11920292, 0.11920292], [0.11920292, 0.11920292]],
                    [[0.73105858, 0.73105858], [0.73105858, 0.73105858]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        f = kornia.enhance.AdjustSigmoid()
        self.assert_close(f(data), expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.adjust_sigmoid
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))

    def test_gradcheck(self, device):
        bs, channels, height, width = 1, 2, 3, 3
        inputs = torch.ones(bs, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_sigmoid, inputs)


class TestAdjustLog(BaseTester):
    @pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4)])
    def test_shape_sigmoid(self, shape, device):
        inputs = torch.ones(*shape, device=device)
        f = kornia.enhance.adjust_log

        assert f(inputs).shape == torch.Size(shape)

    def test_log(self, device, dtype):
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
                [
                    [[1, 1], [1, 1]],
                    [[0.5849625, 0.5849625], [0.5849625, 0.5849625]],
                    [[0.32192809, 0.32192809], [0.32192809, 0.32192809]],
                ],
                [
                    [[0, 0], [0, 0]],
                    [[0.37851162, 0.37851162], [0.37851162, 0.37851162]],
                    [[0.67807191, 0.67807191], [0.67807191, 0.67807191]],
                ],
            ],
            device=device,
            dtype=dtype,
        )  # 2x3x2x2

        f = kornia.enhance.AdjustLog()
        self.assert_close(f(data), expected)

    @pytest.mark.slow
    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.enhance.adjust_log
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))

    def test_gradcheck(self, device):
        bs, channels, height, width = 1, 2, 3, 3
        inputs = torch.ones(bs, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.adjust_log, (inputs, 0.1))


class TestEqualize(BaseTester):
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

        self.assert_close(f(inputs), expected, low_tolerance=True)

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

        self.assert_close(f(inputs), expected, low_tolerance=True)

    def test_counts_beyond_half_precision_range(self, device, dtype):
        # 90000 pixels: the histogram total overflows float16 (inf above 65504) and bfloat16 rounds the counts.
        inputs = torch.full((1, 1, 300, 300), 0.8, device=device, dtype=dtype)
        inputs[..., :100, :] = 0.5
        expected = torch.ones_like(inputs)
        expected[..., :100, :] = 0.0
        self.assert_close(kornia.enhance.equalize(inputs), expected, rtol=0.0, atol=0.0)

    def test_gradcheck(self, device):
        bs, channels, height, width = 1, 2, 3, 3
        inputs = torch.ones(bs, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.equalize, (inputs,), fast_mode=False)

    @pytest.mark.parametrize("scale, shift", [(2.0, 0.0), (1.0, -1.0)])
    def test_out_of_range_input_names_the_range(self, scale, shift, device, dtype):
        # kornia#4431: an input the 256-bin lookup cannot index used to fail with a raw
        # "index 259 is out of bounds" from the gather. MPS range-checks it too (kornia#4600): from
        # torch 2.13 the assert is asynchronous there, so the message arrives at the sync.
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        x = torch.linspace(0, 1, 64, device=device, dtype=dtype).reshape(1, 1, 8, 8) * scale + shift
        with pytest.raises(RuntimeError, match=r"expects input values in \[0, 1\]"):
            _sync(kornia.enhance.equalize(x).device)

    def test_input_the_lookup_can_index_is_still_accepted(self, device, dtype):
        # The check covers exactly the values that crashed, so a hair above 1 keeps working.
        x = torch.linspace(0, 1, 64, device=device, dtype=dtype).reshape(1, 1, 8, 8) * 1.0001
        assert kornia.enhance.equalize(x).shape == x.shape

    def test_dynamo_fullgraph(self, device, dtype):
        # The range check must not reintroduce a graph break (#3842 removed an ``.item()`` check).
        x = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.enhance.equalize, fullgraph=True, backend="eager")
        self.assert_close(compiled(x), kornia.enhance.equalize(x))

    def test_dynamo_fullgraph_out_of_range_input_names_the_range(self, device, dtype):
        # The compiled graph has to carry the same check: on MPS kornia#4600 is only fixed while
        # compiling if the asynchronous assert is traced, because a host read cannot be.
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        if device.type == "mps" and not torch_version_ge(2, 13):
            pytest.skip("no MPS kernel for _assert_async before torch 2.13, so the check is skipped here")
        x = torch.linspace(0, 1, 64, device=device, dtype=dtype).reshape(1, 1, 8, 8) * 2.0
        torch._dynamo.reset()
        compiled = torch.compile(kornia.enhance.equalize, fullgraph=True, backend="eager")
        with pytest.raises(RuntimeError, match=r"expects input values in \[0, 1\]"):
            _sync(compiled(x).device)

    def test_jit(self, device, dtype):
        batch_size, channels, height, width = 1, 2, 3, 3
        inp = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)

        op = kornia.enhance.equalize
        op_script = torch.jit.script(op)

        self.assert_close(op(inp), op_script(inp))

    @staticmethod
    def build_input(batch_size, channels, height, width, device, dtype, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height).to(device, dtype)
        image = torch.stack([channel] * channels).to(device, dtype)
        return torch.stack([image] * batch_size).to(device, dtype)


class TestEqualize3D(BaseTester):
    @pytest.mark.parametrize("scale, shift", [(1.5, 0.0), (1.0, -0.5)])
    def test_out_of_range_input_names_the_range(self, scale, shift, device, dtype):
        # kornia#4432: the 3D path shares the lookup, so it shares the MPS fix too (kornia#4600).
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        torch.manual_seed(0)
        x = torch.rand(1, 1, 5, 7, 9, device=device, dtype=dtype) * scale + shift
        with pytest.raises(RuntimeError, match=r"expects input values in \[0, 1\]"):
            _sync(kornia.enhance.equalize3d(x).device)

    def test_at_most_255_voxels_per_channel_is_unchanged(self, device, dtype):
        # As the docstring states: the lookup step is an integer division by 255.
        torch.manual_seed(0)
        x = torch.rand(2, 3, 3, 5, 17, device=device, dtype=dtype) * 0.5
        self.assert_close(kornia.enhance.equalize3d(x), x)

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

        self.assert_close(f(inputs3d), expected, low_tolerance=True)

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

        self.assert_close(f(inputs3d), expected, low_tolerance=True)

    def test_counts_beyond_half_precision_range(self, device, dtype):
        # 90000 voxels per channel: the histogram total overflows float16 and bfloat16 rounds the counts.
        inputs3d = torch.full((1, 1, 4, 150, 150), 0.8, device=device, dtype=dtype)
        inputs3d[..., :1, :, :] = 0.5
        expected = torch.ones_like(inputs3d)
        expected[..., :1, :, :] = 0.0
        self.assert_close(kornia.enhance.equalize3d(inputs3d), expected, rtol=0.0, atol=0.0)

    def test_gradcheck(self, device):
        bs, channels, depth, height, width = 1, 2, 3, 4, 5
        inputs3d = torch.ones(bs, channels, depth, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.enhance.equalize3d, (inputs3d,), fast_mode=False)

    def test_jit(self, device, dtype):
        batch_size, channels, depth, height, width = 1, 2, 1, 3, 3
        inp = torch.ones(batch_size, channels, depth, height, width, device=device, dtype=dtype)

        op = kornia.enhance.equalize3d
        op_script = torch.jit.script(op)

        self.assert_close(op(inp), op_script(inp))

    @staticmethod
    def build_input(batch_size, channels, depth, height, width, device, dtype, row=None):
        if row is None:
            row = torch.arange(width) / float(width)

        channel = torch.stack([row] * height).to(device, dtype)
        image = torch.stack([channel] * channels).to(device, dtype)
        image3d = torch.stack([image] * depth).transpose(0, 1).to(device, dtype)
        return torch.stack([image3d] * batch_size).to(device, dtype)


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
        self.assert_close(TestSharpness.f(inputs, 1.0), inputs, low_tolerance=True)
        self.assert_close(TestSharpness.f(inputs, 0.8), expected, low_tolerance=True)

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
        # tol_val: float = utils._get_precision(device, dtype)
        self.assert_close(TestSharpness.f(inputs, 1), inputs, low_tolerance=True)
        self.assert_close(TestSharpness.f(inputs, torch.tensor([1.0, 1.0])), inputs, low_tolerance=True)
        self.assert_close(TestSharpness.f(inputs, 0.8), expected_08, low_tolerance=True)
        self.assert_close(TestSharpness.f(inputs, torch.tensor([0.8, 1.3])), expected_08_13, low_tolerance=True)

    def test_gradcheck(self, device):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(TestSharpness.f, (inputs, 0.8))

    @pytest.mark.skip(reason="union type input")
    def test_jit(self, device, dtype):
        op = TestSharpness.f
        op_script = torch.jit.script(TestSharpness.f)
        img = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(img, 0.8)
        actual = op_script(img, 0.8)
        self.assert_close(actual, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(3, 3, 5, 5, device=device, dtype=dtype)
        op = kornia.enhance.sharpness
        op_optimized = torch_optimizer(op)
        # scalar factor (whole batch) and per-sample factor
        self.assert_close(op(img, 0.5), op_optimized(img, 0.5))
        factor = torch.tensor([0.3, 0.7, 1.5], device=device, dtype=dtype)
        self.assert_close(op(img, factor), op_optimized(img, factor))


@pytest.mark.skipif(kornia.core.utils.xla_is_available(), reason="issues with xla device")
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

    @pytest.mark.parametrize("addition", [0.5, -0.5])
    def test_additions_closed_range_4605(self, device, dtype, addition):
        # RandomSolarize admits the closed [-0.5, 0.5], and its sampler can draw an endpoint exactly.
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        shifted = (img + addition).clamp(0.0, 1.0)
        expected = torch.where(shifted >= 0.5, 1.0 - shifted, shifted)
        self.assert_close(TestSolarize.f(img, 0.5, addition), expected)
        # Both entries are asserted: checking only sample 0 leaves a 1-D `additions` broadcast from the
        # first entry indistinguishable from a real per-sample dispatch.
        per_sample = torch.tensor([addition, 0.0], device=device, dtype=dtype)
        out = TestSolarize.f(img, 0.5, per_sample)
        self.assert_close(out[0], expected[0])
        unshifted = img[1].clamp(0.0, 1.0)
        self.assert_close(out[1], torch.where(unshifted >= 0.5, 1.0 - unshifted, unshifted))

    # `0.5 + 2 ** -24` is the next float32 above the bound, so the pin constrains it to one ulp
    # rather than to the ~1700 an 0.5001 leaves.
    @pytest.mark.parametrize("addition", [0.5 + 2**-24, -(0.5 + 2**-24), 0.5001, -0.5001])
    def test_additions_outside_closed_range_raise_4605(self, device, addition):
        if device.type != "cpu":
            pytest.skip("CPU only: the value check is an async device assert elsewhere")
        img = torch.rand(2, 3, 4, 5, device=device)
        with pytest.raises(RuntimeError, match=r"closed range \[-0\.5, 0\.5\]"):
            TestSolarize.f(img, 0.5, addition)

    @pytest.mark.parametrize("addition", [0.5001, -0.5001])
    def test_float_additions_checked_before_rounding(self, device, dtype, addition):
        # A float is checked as given, on the host, for every device: float16/bfloat16 round 0.5001 to 0.5,
        # and a check on an MPS tensor is skipped.
        img = torch.ones(2, 3, 4, 5, device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match=r"closed range \[-0\.5, 0\.5\]"):
            TestSolarize.f(img, 0.5, addition)

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
        self.assert_close(TestSolarize.f(inputs, 0.5), expected, rtol=1e-2, atol=1e-2)

    def test_gradcheck(self, device):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(TestSolarize.f, (inputs, 0.8))

    # TODO: implement me
    @pytest.mark.skip(reason="union type input")
    def test_jit(self, device, dtype):
        op = TestSolarize.f
        op_script = torch.jit.script(op)
        img = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(img, 0.8)
        actual = op_script(img, 0.8)
        self.assert_close(actual, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(3, 3, 5, 5, device=device, dtype=dtype)
        op = kornia.enhance.solarize
        op_optimized = torch_optimizer(op)
        # scalar threshold/addition (whole batch)
        self.assert_close(op(img, 0.5, 0.1), op_optimized(img, 0.5, 0.1))
        # per-sample 1-d threshold/addition — the batch-wise path that must stay fullgraph-safe
        thresholds = torch.tensor([0.3, 0.6, 0.8], device=device, dtype=dtype)
        additions = torch.tensor([-0.2, 0.0, 0.2], device=device, dtype=dtype)
        self.assert_close(op(img, thresholds, additions), op_optimized(img, thresholds, additions))

    def test_dynamo_fullgraph(self, device, dtype):
        img = torch.rand(3, 3, 5, 5, device=device, dtype=dtype)
        thresholds = torch.tensor([0.3, 0.6, 0.8], device=device, dtype=dtype)
        additions = torch.tensor([-0.2, 0.0, 0.2], device=device, dtype=dtype)
        expected = kornia.enhance.solarize(img, thresholds, additions)
        torch._dynamo.reset()
        compiled = torch.compile(kornia.enhance.solarize, fullgraph=True)
        self.assert_close(compiled(img, thresholds, additions), expected)


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

        # invalid input type
        with pytest.raises(TypeError):
            TestPosterize.f([1.0], 0)

        # bits must be int or tensor
        with pytest.raises(TypeError):
            TestPosterize.f(img, 1.0)

        # batch size mismatch between bits tensor and input
        with pytest.raises(AssertionError):
            TestPosterize.f(img, torch.tensor([1, 2, 3], device=device))

    # TODO(jian): add better cases
    @pytest.mark.skipif(kornia.core.utils.xla_is_available(), reason="issues with xla device")
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

        self.assert_close(TestPosterize.f(inputs, 1), expected)
        self.assert_close(TestPosterize.f(inputs, 0), torch.zeros_like(inputs))
        self.assert_close(TestPosterize.f(inputs, 8), inputs)

    @pytest.mark.skip(reason="IndexError: tuple index out of range")
    def test_gradcheck(self, device):
        bs, channels, height, width = 2, 3, 4, 5
        inputs = torch.rand(bs, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(TestPosterize.f, (inputs, 0))

    # TODO: implement me
    @pytest.mark.skip(reason="union type input")
    def test_jit(self, device, dtype):
        op = TestPosterize.f
        op_script = torch.jit.script(op)
        img = torch.rand(2, 1, 3, 3).to(device=device, dtype=dtype)
        expected = op(img, 8)
        actual = op_script(img, 8)
        self.assert_close(actual, expected)

    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        op = kornia.enhance.posterize
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img, 3), op_optimized(img, 3))


_FACTOR_OPS = [
    kornia.enhance.adjust_saturation,
    kornia.enhance.adjust_saturation_raw,
    kornia.enhance.adjust_saturation_with_gray_subtraction,
    kornia.enhance.adjust_hue,
    kornia.enhance.adjust_hue_raw,
    kornia.enhance.adjust_contrast,
    kornia.enhance.adjust_contrast_with_mean_subtraction,
    kornia.enhance.adjust_brightness,
    kornia.enhance.adjust_brightness_accumulative,
    kornia.enhance.adjust_gamma,
]


class TestFactorBroadcast(BaseTester):
    """A factor carrying more dimensions than the image must raise.

    The other factor ops used to spin forever while broadcasting it; ``adjust_gamma`` returned a larger tensor.
    """

    @pytest.mark.parametrize("op", _FACTOR_OPS)
    @pytest.mark.parametrize("img_shape", [(3, 4, 4), (2, 3, 4, 4)])
    def test_factor_with_more_dims_raises(self, device, dtype, op, img_shape):
        img = torch.rand(img_shape, device=device, dtype=dtype)
        factor = torch.ones(2, *([1] * len(img_shape)), device=device, dtype=dtype)
        with pytest.raises(ValueError):
            op(img, factor)

    @pytest.mark.parametrize("op", _FACTOR_OPS)
    def test_per_image_factor_with_equal_rank(self, device, dtype, op):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        values = [0.3, 0.7]
        factor = torch.tensor(values, device=device, dtype=dtype).view(2, 1, 1, 1)
        out = op(img, factor)
        for i, v in enumerate(values):
            self.assert_close(out[i : i + 1], op(img[i : i + 1], v))

    @pytest.mark.parametrize("op", [kornia.enhance.adjust_brightness, kornia.enhance.adjust_contrast])
    def test_per_channel_factor_with_equal_rank(self, device, dtype, op):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        values = [0.2, 0.5, 0.8]
        factor = torch.tensor(values, device=device, dtype=dtype).view(1, 3, 1, 1)
        out = op(img, factor)
        for c, v in enumerate(values):
            self.assert_close(out[:, c : c + 1], op(img[:, c : c + 1], v))

    def test_factor_with_fewer_dims_still_broadcasts(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        factor = torch.tensor([0.25, 0.75], device=device, dtype=dtype)
        out = kornia.enhance.adjust_brightness(img, factor)
        assert out.shape == img.shape

    def test_overranked_gain_raises(self, device, dtype):
        img = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        gain = torch.ones(2, 1, 1, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            kornia.enhance.adjust_gamma(img, 1.0, gain)


_FUNCTIONS = ["equalize", "equalize3d", "posterize", "sharpness"]


def _arguments(name):
    if name == "posterize":
        return (4,)
    if name == "sharpness":
        return (0.5,)
    return ()


def _spatial_shape(name):
    return (3, 5, 7) if name == "equalize3d" else (5, 7)


def _input(shape, device, dtype):
    return (torch.arange(prod(shape)).remainder(128).to(device=device, dtype=dtype) / 128).reshape(shape)


def _weights(input):
    return (torch.arange(input.numel()).remainder(17).to(input) + 1).reshape(input.shape)


def _equalize_reference(input):
    # Independent per-plane histogram: bincount instead of the batched scatter_add implementation, counting each
    # pixel in the 8-bit level its lookup reads, as PIL does.
    count_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    planes = input.reshape(input.shape[0] * input.shape[1], -1)
    outputs = []
    for plane in planes:
        scaled = plane * 255.0
        indices = scaled.long().clamp(0, 255)
        hist = torch.bincount(indices, minlength=256).to(count_dtype)
        last_count = hist[hist.nonzero()[-1, 0]]
        step = torch.div(hist.sum() - last_count, 255, rounding_mode="trunc")
        lut = torch.div(
            hist.cumsum(0) + torch.div(step, 2, rounding_mode="trunc"), step.clamp(min=1), rounding_mode="trunc"
        )
        lut = torch.cat([torch.zeros_like(lut[:1]), lut[:-1]]).clamp(0, 255).to(input.dtype)
        outputs.append(torch.where(step == 0, scaled, lut[scaled.long()]) / 255.0)
    return torch.stack(outputs).reshape(input.shape)


class TestEmptyIntensity(BaseTester):
    @pytest.mark.parametrize("name", _FUNCTIONS)
    @pytest.mark.parametrize("leading", [(0,), (2, 0), (0, 2), (2, 1, 0)])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_empty_shape_and_backward_4429(self, name, leading, transpose, device, dtype):
        image = torch.empty((*leading, 3, *_spatial_shape(name)), device=device, dtype=dtype)
        if transpose:
            image = image.transpose(-1, -2)
        image.requires_grad_()
        output = getattr(enhance, name)(image, *_arguments(name))
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == image.dtype
        assert output.numel() == 0
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("name", _FUNCTIONS)
    @pytest.mark.parametrize("leading", [None, (), (1,), (2,), (2, 1, 2)])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_nonempty_rank_interpretation_4429(self, name, leading, transpose, device, dtype):
        spatial = _spatial_shape(name)
        shape = spatial if leading is None else (*leading, 3, *spatial)
        image = _input(shape, device, dtype)
        if transpose:
            image = image.transpose(-1, -2)
        channels = 1 if leading is None else 3
        canonical = image.reshape(-1, channels, *image.shape[-len(spatial) :])
        fn = getattr(enhance, name)
        expected = fn(canonical, *_arguments(name)).reshape(image.shape)
        output = fn(image, *_arguments(name))
        assert output.shape == image.shape
        self.assert_close(output, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("name", _FUNCTIONS)
    def test_invalid_planes_4429(self, name, device, dtype):
        shape = (0, 3, *_spatial_shape(name))
        for axis in range(1, len(shape)):
            invalid = list(shape)
            invalid[axis] = 0
            with pytest.raises(ValueError):
                getattr(enhance, name)(torch.empty(invalid, device=device, dtype=dtype), *_arguments(name))
        # In a minimum-rank tensor, the first dimension is spatial, not an empty batch.
        invalid = (0, *_spatial_shape(name)[1:])
        with pytest.raises(ValueError):
            getattr(enhance, name)(torch.empty(invalid, device=device, dtype=dtype), *_arguments(name))

    @pytest.mark.parametrize("name", _FUNCTIONS)
    def test_invalid_rank_and_type_4429(self, name, device, dtype):
        for shape in [(), (3,)]:
            with pytest.raises(ValueError):
                getattr(enhance, name)(torch.ones(shape, device=device, dtype=dtype), *_arguments(name))
        with pytest.raises(TypeError):
            getattr(enhance, name)([1.0], *_arguments(name))

    @pytest.mark.parametrize("batch", [0, 2])
    @pytest.mark.parametrize("form", ["scalar", "one", "sample", "channel"])
    def test_posterize_parameter_forms_4429(self, batch, form, device, dtype):
        image = _input((2, batch, 3, 5, 7), device, dtype).requires_grad_()
        count = 2 * batch
        if form == "scalar":
            bits = 4
            mask = torch.zeros((count, 3, 1, 1), device=device, dtype=dtype)
        elif form == "one":
            bits = torch.tensor([8])
            mask = torch.ones((count, 3, 1, 1), device=device, dtype=dtype)
        elif form == "sample":
            bits = torch.arange(count).remainder(3) * 4
            mask = (bits == 8).to(image).reshape(count, 1, 1, 1).expand(count, 3, 1, 1)
        else:
            bits = (torch.arange(count * 3).remainder(3) * 4).reshape(count, 3)
            mask = (bits == 8).to(image).reshape(count, 3, 1, 1)
        output = enhance.posterize(image, bits)
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == dtype
        weights = _weights(image)
        (output * weights).sum().backward()
        assert image.grad is not None
        expected_grad = (mask.expand(count, 3, 5, 7) * weights.reshape(count, 3, 5, 7)).reshape(image.shape)
        self.assert_close(image.grad, expected_grad, rtol=0, atol=0)
        if batch:
            canonical = image.detach().reshape(count, 3, 5, 7)
            expected = []
            for index in range(count):
                per_image_bits = bits if form in ("scalar", "one") else bits[index : index + 1]
                expected.append(enhance.posterize(canonical[index : index + 1], per_image_bits))
            self.assert_close(output, torch.cat(expected).reshape(image.shape), rtol=0, atol=0)

    @pytest.mark.parametrize("batch", [0, 2])
    @pytest.mark.parametrize("form", ["scalar", "sample"])
    def test_sharpness_parameter_forms_4429(self, batch, form, device, dtype):
        image = _input((2, batch, 3, 5, 7), device, dtype).requires_grad_()
        count = 2 * batch
        factor = 0.5 if form == "scalar" else torch.linspace(0.2, 1.5, count)
        output = enhance.sharpness(image, factor)
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == dtype
        (output * _weights(image)).sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape
        if batch:
            canonical = image.detach().reshape(count, 3, 5, 7)
            expected = [
                enhance.sharpness(canonical[index : index + 1], factor if form == "scalar" else factor[index])
                for index in range(count)
            ]
            self.assert_close(output, torch.cat(expected).reshape(image.shape))

    def test_empty_parameters_are_validated_4429(self, device, dtype):
        image = torch.empty(0, 3, 6, 8, device=device, dtype=dtype)
        with pytest.raises(TypeError, match="bits type"):
            enhance.posterize(image, 4.0)
        for bits in [torch.tensor([4, 8]), torch.empty(0, 2, dtype=torch.long)]:
            with pytest.raises(AssertionError, match="equal between bits and input"):
                enhance.posterize(image, bits)
        for factor in [torch.tensor([0.5]), torch.empty(0, 1)]:
            with pytest.raises(AssertionError, match="batch size shall match"):
                enhance.sharpness(image, factor)

    @pytest.mark.parametrize("spatial", [(2, 8), (6, 2)])
    def test_small_empty_sharpness_is_rejected_4429(self, spatial, device, dtype):
        with pytest.raises(RuntimeError, match="Kernel size"):
            enhance.sharpness(torch.empty(0, 3, *spatial, device=device, dtype=dtype), 0.5)

    @pytest.mark.parametrize("volume", [False, True])
    @pytest.mark.parametrize("plane_size", [255, 256, 429])
    def test_equalize_independent_planes_4429(self, volume, plane_size, device, dtype):
        spatial = {255: (3, 5, 17), 256: (2, 8, 16), 429: (3, 11, 13)}[plane_size]
        if not volume:
            spatial = {255: (5, 51), 256: (16, 16), 429: (13, 33)}[plane_size]
        indices = torch.arange(plane_size)
        levels = torch.tensor([0.0, 0.125, 0.375, 0.875])
        channel0 = levels[indices.remainder(4)]
        channel1 = levels[(indices * indices + 3 * indices + 1).remainder(4)]
        image = torch.stack([channel0, channel1]).reshape(1, 2, *spatial).to(device=device, dtype=dtype)
        image.requires_grad_()
        reference_input = image.detach().clone().requires_grad_()
        fn = enhance.equalize3d if volume else enhance.equalize
        output = fn(image)
        expected = _equalize_reference(reference_input)
        self.assert_close(output, expected, rtol=0, atol=0)
        weights = _weights(image)
        (output * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assert_close(image.grad, reference_input.grad, rtol=0, atol=0)
        if plane_size == 429:
            assert not torch.equal(output, image)
            if volume:
                per_slice = torch.stack([enhance.equalize(image[:, :, depth]) for depth in range(spatial[0])], 2)
                assert not torch.equal(output, per_slice)

    @pytest.mark.parametrize("name", _FUNCTIONS)
    @pytest.mark.parametrize("batch", [0, 2])
    def test_dynamo_shape_and_values_4429(self, name, batch, device, dtype, torch_optimizer):
        image = _input((2, batch, 3, *_spatial_shape(name)), device, dtype)
        fn = getattr(enhance, name)
        expected = fn(image, *_arguments(name))
        optimized = torch_optimizer(fn)
        output = optimized(image, *_arguments(name))
        assert output.shape == image.shape
        assert output.device == image.device
        assert output.dtype == dtype
        self.assert_close(output, expected)


class TestClipUnitFreshBuffer(BaseTester):
    # adjust_brightness and adjust_contrast clamp their own fresh result in place when no gradient is recorded, so a
    # batch-32 call allocates one image-sized temporary fewer. The caller's tensor is never modified, and the values
    # and gradients equal the out-of-place clamp.
    _REFERENCE = {
        "adjust_brightness": lambda image, factor: torch.clamp(image + factor.view(-1, 1, 1, 1), 0.0, 1.0),
        "adjust_contrast": lambda image, factor: torch.clamp(image * factor.view(-1, 1, 1, 1), 0.0, 1.0),
    }

    @pytest.mark.parametrize("op", sorted(_REFERENCE))
    def test_input_unmodified_and_values_match_with_and_without_a_gradient(self, device, dtype, op):
        fn = getattr(enhance, op)
        reference = self._REFERENCE[op]
        image = torch.linspace(-0.5, 1.5, 2 * 3 * 4 * 4, device=device, dtype=dtype).reshape(2, 3, 4, 4)
        original = image.clone()
        factor = torch.tensor([0.3, 1.4], device=device, dtype=dtype)
        with torch.no_grad():
            out_no_grad = fn(image, factor)
        out_plain = fn(image, factor)
        leaf = image.clone().requires_grad_()
        out_grad = fn(leaf, factor)
        assert torch.equal(image, original)
        assert torch.equal(out_no_grad, out_plain)
        assert torch.equal(out_no_grad, out_grad.detach())
        self.assert_close(out_no_grad, reference(image, factor))

        ref_leaf = image.clone().requires_grad_()
        reference(ref_leaf, factor).sum().backward()
        out_grad.sum().backward()
        assert leaf.grad is not None
        self.assert_close(leaf.grad, ref_leaf.grad)

    @pytest.mark.parametrize("op", sorted(_REFERENCE))
    @pytest.mark.parametrize("batch", [0, 3])
    def test_vmap_keeps_the_batching_rule(self, device, dtype, op, batch):
        # vmap has a batching rule for clamp but not for clamp_, whose fallback loops over the batch and rejects an
        # empty one, so a transformed call keeps the out-of-place clamp. With the fallback disabled, reaching it raises.
        fn = getattr(enhance, op)
        image = torch.rand(batch, 3, 4, 4, device=device, dtype=dtype)
        functorch = torch._C._functorch
        fallback_enabled = functorch._is_vmap_fallback_enabled()
        functorch._set_vmap_fallback_enabled(False)
        try:
            out = torch.vmap(lambda frame: fn(frame, 0.2))(image)
        finally:
            functorch._set_vmap_fallback_enabled(fallback_enabled)
        self.assert_close(out, fn(image, 0.2))

    @pytest.mark.parametrize("op", sorted(_REFERENCE))
    def test_jit(self, device, dtype, op):
        # TorchScript cannot compile the functorch query, so the scripted functions and modules skip the in-place clamp.
        fn = getattr(enhance, op)
        module = {"adjust_brightness": enhance.AdjustBrightness, "adjust_contrast": enhance.AdjustContrast}[op]
        image = torch.linspace(-0.5, 1.5, 2 * 3 * 4 * 4, device=device, dtype=dtype).reshape(2, 3, 4, 4)
        original = image.clone()
        factor = torch.tensor([0.3, 1.4], device=device, dtype=dtype)
        expected = self._REFERENCE[op](image, factor)
        self.assert_close(torch.jit.script(fn)(image, factor), expected)
        self.assert_close(torch.jit.script(module(factor))(image), expected)
        assert torch.equal(image, original)

    @pytest.mark.parametrize("op", sorted(_REFERENCE))
    def test_integer_image_with_checks_disabled_keeps_the_promoting_clamp(self, device, op):
        # With checks disabled an integer image reaches the clamp. Its float bounds promote the result to float32,
        # which an in-place clamp cannot do, so that image keeps the out-of-place clamp and its float32 output.
        fn = getattr(enhance, op)
        image = torch.randint(0, 256, (2, 3, 4, 4), device=device, dtype=torch.uint8)
        original = image.clone()
        factor = torch.tensor([0.3, 1.4], device=device)
        was_enabled = are_checks_enabled()
        disable_checks()
        try:
            out = fn(image, factor)
        finally:
            if was_enabled:
                enable_checks()
        assert torch.equal(image, original)
        assert out.dtype == torch.float32
        self.assert_close(out, self._REFERENCE[op](image, factor.to(torch.uint8)))
