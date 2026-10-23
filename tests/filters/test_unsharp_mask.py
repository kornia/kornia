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

from kornia.filters import UnsharpMask, gaussian_blur2d, unsharp_mask

from testing.base import BaseTester, supports_reflect_padding


class Testunsharp(BaseTester):
    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [3, (5, 3)])
    @pytest.mark.parametrize("sigma", [(3.0, 1.0), (0.5, 0.1)])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_smoke(self, shape, kernel_size, sigma, params_as_tensor, device, dtype):
        if params_as_tensor is True:
            sigma = torch.tensor([sigma], device=device, dtype=dtype).repeat(shape[0], 1)

        data = torch.ones(shape, device=device, dtype=dtype)
        actual = unsharp_mask(data, kernel_size, sigma, "replicate")
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_cardinality(self, shape, device, dtype):
        kernel_size = (5, 7)
        sigma = (1.5, 2.1)

        data = torch.ones(shape, device=device, dtype=dtype)
        actual = unsharp_mask(data, kernel_size, sigma, "replicate")
        assert actual.shape == shape

    @pytest.mark.parametrize("input_dtype", [torch.uint8, torch.int16, torch.bool])
    def test_integer_input_matches_float32_5155(self, input_dtype: torch.dtype, device: torch.device) -> None:
        # An integer image is sharpened in float32 on its own scale, like gaussian_blur2d.
        generator = torch.Generator().manual_seed(0)
        low = -128 if input_dtype == torch.int16 else 0
        high = 2 if input_dtype == torch.bool else 256
        data = torch.randint(low, high, (2, 3, 9, 13), generator=generator).to(device, input_dtype)

        expected = unsharp_mask(data.float(), (5, 5), (1.5, 1.5))
        actual = unsharp_mask(data, (5, 5), (1.5, 1.5))

        assert actual.dtype == torch.float32
        assert actual.device == data.device
        self.assert_close(actual, expected)
        self.assert_close(UnsharpMask((5, 5), (1.5, 1.5))(data), expected)

    @pytest.mark.skip(reason="nothing to test")
    def test_exception(self): ...

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        data = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = (3, 3)
        sigma = (1.5, 2.1)
        actual = unsharp_mask(data, kernel_size, sigma, "replicate")
        assert actual.is_contiguous()

    def test_gradcheck(self, device):
        # test parameters
        shape = (1, 3, 5, 5)
        kernel_size = (3, 3)
        sigma = (1.5, 2.1)

        # evaluate function gradient
        data = torch.rand(shape, device=device, dtype=torch.float64)
        self.gradcheck(unsharp_mask, (data, kernel_size, sigma, "replicate"))

    def test_module(self, device, dtype):
        params = [(3, 3), (1.5, 1.5)]
        op = unsharp_mask
        op_module = UnsharpMask(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("sigma", [(3.0, 1.0), (0.5, 0.1)])
    @pytest.mark.parametrize("params_as_tensor", [True, False])
    def test_dynamo(self, sigma, params_as_tensor, device, dtype, torch_optimizer):
        if params_as_tensor is True:
            sigma = torch.tensor([sigma], device=device, dtype=dtype)

        data = torch.ones(1, 3, 10, 10, device=device, dtype=dtype)
        op = UnsharpMask(3, sigma)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestConventionsUnsharpMask(BaseTester):
    """Pins for the formula and the range of :func:`unsharp_mask`."""

    @pytest.mark.parametrize("border_type", ["reflect", "constant"])
    def test_convention_unsharp_mask_is_twice_the_image_minus_its_gaussian_blur(self, border_type, device, dtype):
        # out = 2 * x - gaussian_blur2d(x, kernel_size, sigma, border_type) = x + 1 * (x - blur): the gain is fixed at 1
        # (scikit-image's amount=1) and kernel_size, sigma and border_type are gaussian_blur2d's.
        if border_type == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        image = torch.rand(2, 3, 9, 13).to(device=device, dtype=dtype)
        expected = 2 * image - gaussian_blur2d(image, (3, 5), (1.0, 2.0), border_type)
        self.assert_close(unsharp_mask(image, (3, 5), (1.0, 2.0), border_type), expected)
        self.assert_close(UnsharpMask((3, 5), (1.0, 2.0), border_type)(image), expected)

    def test_convention_unsharp_mask_output_is_not_clamped(self, device, dtype):
        # The output is not clipped to the input range: a 0 | 1 step overshoots on both sides of the edge
        # (scikit-image clips by default, preserve_range=False).
        # Snippet used to generate expected:
        #   step = torch.zeros(1, 1, 7, 10); step[..., 5:] = 1; print(unsharp_mask(step, (5, 5), (1.5, 1.5))[0, 0, 3])
        if not supports_reflect_padding(device, dtype):
            pytest.skip(f"this torch build has no reflect padding kernel for {dtype} on {device.type}")
        step = torch.zeros(1, 1, 7, 10, device=device, dtype=dtype)
        step[..., 5:] = 1.0
        out = unsharp_mask(step, (5, 5), (1.5, 1.5))
        expected = [0.0, 0.0, 0.0, -0.120078, -0.353959, 1.353959, 1.120078, 1.0, 1.0, 1.0]
        self.assert_close(out[0, 0, 3], torch.tensor(expected, device=device, dtype=dtype))
        assert out.min() < -0.3
        assert out.max() > 1.3


def _mixed_precision_region_6182(device):
    # MPS autocasts to float16, and so does a CUDA device without bfloat16, where bfloat16 autocast raises.
    if device.type not in ("cpu", "cuda", "mps"):
        pytest.skip("CPU, CUDA and MPS autocast")
    bfloat16_ok = device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_bf16_supported())
    return torch.bfloat16 if bfloat16_ok else torch.float16


class TestUnsharpMaskMixedPrecision(BaseTester):
    """``unsharp_mask`` and ``UnsharpMask`` run inside a mixed-precision region (#6182).

    No "autocast" in the test names: the repository conftest skips those on MPS.
    """

    @pytest.mark.parametrize("as_module", [False, True])
    def test_a_mixed_precision_region_keeps_the_image_dtype_6182(self, device, as_module):
        # The blur, a convolution, came back in the autocast dtype and met the float32 image in torch.lerp, which raised
        # "expected dtype ... for `end` but got dtype float". The blur is now cast back, as 2 * image - blur would be.
        region_dtype = _mixed_precision_region_6182(device)
        generator = torch.Generator().manual_seed(6182)
        image = torch.rand(2, 3, 16, 20, generator=generator).to(device)
        op = UnsharpMask((5, 5), (1.5, 1.5)) if as_module else lambda x: unsharp_mask(x, (5, 5), (1.5, 1.5))
        with torch.autocast(device.type, dtype=region_dtype):
            actual = op(image)
            blur = gaussian_blur2d(image, (5, 5), (1.5, 1.5))
        assert blur.dtype == region_dtype
        assert actual.dtype == torch.float32
        self.assert_close(actual, torch.lerp(blur.float(), image, weight=2.0), atol=0, rtol=0)
        self.assert_close(actual, unsharp_mask(image, (5, 5), (1.5, 1.5)), atol=5e-2, rtol=0)

    def test_half_precision_image_in_the_other_half_precision_region_6182(self, device):
        # A float16 image inside a bfloat16 region also raised; it now keeps float16.
        if device.type != "cpu":
            pytest.skip("measured on the CPU")
        image = torch.rand(1, 3, 12, 12, generator=torch.Generator().manual_seed(6182)).to(device, torch.float16)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = unsharp_mask(image, (3, 3), (1.0, 1.0))
        assert out.dtype == torch.float16
        self.assert_close(out.float(), unsharp_mask(image.float(), (3, 3), (1.0, 1.0)), atol=5e-2, rtol=0)

    def test_gradient_in_a_mixed_precision_region_6182(self, device):
        region_dtype = _mixed_precision_region_6182(device)
        image = torch.rand(1, 3, 12, 12, device=device, requires_grad=True)
        with torch.autocast(device.type, dtype=region_dtype):
            out = unsharp_mask(image, (3, 3), (1.0, 1.0))
        out.sum().backward()
        assert out.dtype == torch.float32
        assert image.grad is not None
        assert image.grad.dtype == torch.float32
        assert torch.isfinite(image.grad).all()
        # The gradient also flows back through the blur: it matches that of 2 * image - blur outside the region.
        (expected,) = torch.autograd.grad((2 * image - gaussian_blur2d(image, (3, 3), (1.0, 1.0))).sum(), image)
        self.assert_close(image.grad, expected, atol=5e-2, rtol=0)
