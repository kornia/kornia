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

import importlib

import pytest
import torch

from kornia.core.exceptions import BaseError
from kornia.filters import Laplacian, filter2d, get_laplacian_kernel1d, get_laplacian_kernel2d, laplacian
from kornia.filters.kernels import normalize_kernel2d

from testing.base import (
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    assert_close,
    dynamo_is_available,
    supports_reflect_padding,
)

laplacian_module = importlib.import_module("kornia.filters.laplacian")


@pytest.mark.parametrize("window_size", [5, 11])
def test_get_laplacian_kernel1d(window_size, device, dtype):
    actual = get_laplacian_kernel1d(window_size, device=device, dtype=dtype)
    expected = torch.zeros(1, device=device, dtype=dtype)

    assert actual.shape == (window_size,)
    assert_close(actual.sum(), expected.sum())


@pytest.mark.parametrize("window_size", [5, 11, (3, 3)])
def test_get_laplacian_kernel2d(window_size, device, dtype):
    actual = get_laplacian_kernel2d(window_size, device=device, dtype=dtype)
    expected = torch.zeros(1, device=device, dtype=dtype)
    expected_shape = window_size if isinstance(window_size, tuple) else (window_size, window_size)

    assert actual.shape == expected_shape
    assert_close(actual.sum(), expected.sum())


def test_get_laplacian_kernel1d_exact(device, dtype):
    actual = get_laplacian_kernel1d(5, device=device, dtype=dtype)
    expected = torch.tensor([1.0, 1.0, -4.0, 1.0, 1.0], device=device, dtype=dtype)
    assert_close(expected, actual)


def test_get_laplacian_kernel2d_exact(device, dtype):
    actual = get_laplacian_kernel2d(7, device=device, dtype=dtype)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -48.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    assert_close(expected, actual)


def test_get_laplacian_kernel1d_rejects_a_single_tap(device, dtype):
    # A single tap is the all-zero kernel [0.0]: its centre is 1 - 1. The 1-D builder does not normalize, so the
    # message must not talk about it, and it reports the size as given.
    with pytest.raises(BaseError, match="size of at least 3 along one axis") as error:
        get_laplacian_kernel1d(1, device=device, dtype=dtype)
    assert "normaliz" not in str(error.value)
    assert str(error.value).endswith("Got 1")


@pytest.mark.parametrize("kernel_size", [1, (1, 1), [1, 1]])
def test_get_laplacian_kernel2d_rejects_a_single_tap(kernel_size, device, dtype):
    # A 1x1 kernel is the all-zero kernel [[0.0]]: its centre is 1 - 1 * 1.
    with pytest.raises(BaseError, match="size of at least 3 along one axis"):
        get_laplacian_kernel2d(kernel_size, device=device, dtype=dtype)


def test_get_laplacian_kernel2d_accepts_a_single_tap_along_one_axis(device, dtype):
    # (1, 3) and (3, 1) are the meaningful 1-D second difference, not the degenerate 1x1 kernel.
    actual = get_laplacian_kernel2d((1, 3), device=device, dtype=dtype)
    assert_close(actual, torch.tensor([[1.0, -2.0, 1.0]], device=device, dtype=dtype))
    actual = get_laplacian_kernel2d((3, 1), device=device, dtype=dtype)
    assert_close(actual, torch.tensor([[1.0], [-2.0], [1.0]], device=device, dtype=dtype))


class TestLaplacian(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, 5, (5, 7), (1, 3), (3, 1)])
    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_matches_dense_kernel(self, kernel_size, border_type, normalized, device, dtype):
        data = torch.rand(2, 3, 11, 13, device=device, dtype=dtype)
        kernel = get_laplacian_kernel2d(kernel_size, device=device, dtype=dtype)[None]
        if normalized:
            kernel = normalize_kernel2d(kernel)

        expected = filter2d(data, kernel, border_type)
        self.assert_close(laplacian(data, kernel_size, border_type, normalized), expected)

    @pytest.mark.parametrize("has_mkldnn", [False, True])
    @pytest.mark.parametrize("shape", [(1, 2, 9, 10), (1, 4, 1024, 1024)])
    def test_slices_dispatch(self, monkeypatch, has_mkldnn, shape, device, dtype):
        # CPU uses slices except on large inputs with oneDNN; eager CUDA keeps cuDNN's convolution.
        monkeypatch.setattr(laplacian_module, "_HAS_MKLDNN", has_mkldnn)
        conv = laplacian_module.filter2d
        calls = 0

        def counted_conv(*args, **kwargs):
            nonlocal calls
            calls += 1
            return conv(*args, **kwargs)

        monkeypatch.setattr(laplacian_module, "filter2d", counted_conv)
        data = torch.rand(shape, device=device, dtype=dtype)
        laplacian(data, 5)
        small = not has_mkldnn or data.numel() < 1 << 22
        uses_slices = device.type == "cpu" and dtype in (torch.float32, torch.float64) and small
        assert calls == int(not uses_slices)

    @pytest.mark.parametrize("value", ["large", "inf", "nan"])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_extreme_cpu_values_match_convolution(self, monkeypatch, value, normalized, device, dtype):
        if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
            pytest.skip("The CPU slice path supports float32/float64")
        data = torch.zeros(1, 1, 7, 9, device=device, dtype=dtype)
        data[..., 3, 4] = torch.finfo(dtype).max / 2 if value == "large" else float(value)
        kernel = get_laplacian_kernel2d(3, device=device, dtype=dtype)[None]
        if normalized:
            kernel = normalize_kernel2d(kernel)
        expected = filter2d(data, kernel, "constant")
        actual = laplacian(data, 3, "constant", normalized)
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual.isposinf(), expected.isposinf())
        self.assert_close(actual.isneginf(), expected.isneginf())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num())

    def test_vmap(self, device, dtype):
        data = torch.rand(2, 1, 3, 7, 9, device=device, dtype=dtype)
        expected = torch.stack([laplacian(image, 3) for image in data])
        actual = torch.vmap(lambda image: laplacian(image, 3))(data)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("slices", [True, False])
    @pytest.mark.parametrize("kernel_size", [1, (1, 1), [1, 1]])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_kernel_size_one_is_rejected(self, monkeypatch, kernel_size, normalized, slices, device, dtype):
        # The 1x1 Laplacian is the all-zero kernel: zeros unnormalized, 0 / 0 = NaN normalized. The size is rejected
        # up front, whichever of the slice and convolution paths would have run.
        monkeypatch.setattr(laplacian_module, "_laplacian_slices_eligible", lambda _input: slices)
        data = torch.rand(1, 1, 3, 5, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="size of at least 3 along one axis"):
            laplacian(data, kernel_size, normalized=normalized)

    @pytest.mark.parametrize("kernel_size", [1, (1, 1), [1, 1]])
    def test_module_rejects_kernel_size_one(self, kernel_size):
        with pytest.raises(BaseError, match="size of at least 3 along one axis"):
            Laplacian(kernel_size)

    @pytest.mark.parametrize(
        "kernel_size, message",
        [
            (2, "Kernel size must be an odd integer bigger than 0"),
            (0, "Kernel size must be an odd integer bigger than 0"),
            ((2, 2), "Kernel size must be an odd integer bigger than 0"),
            ((1, 2), "Kernel size must be an odd integer bigger than 0"),
            ((-1, 1), "Kernel size must be an odd integer bigger than 0"),
            ((3, 3, 3), "2D Kernel size should have a length of 2"),
        ],
    )
    def test_module_rejects_invalid_kernel_size_at_construction(self, kernel_size, message):
        # The module validates the whole size when it is built, as laplacian() does when it is called.
        with pytest.raises(BaseError, match=message):
            Laplacian(kernel_size)
        data = torch.rand(1, 1, 5, 5)
        with pytest.raises(BaseError, match=message):
            laplacian(data, kernel_size)

    @pytest.mark.parametrize("kernel_size", [(1, 3), (3, 1)])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_kernel_size_one_along_one_axis_is_a_second_difference(self, kernel_size, normalized, device, dtype):
        # Squares have the constant second difference 2, except at the reflected last sample: 2 * 16 - 50 = -18.
        row = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], device=device, dtype=dtype)
        expected = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, -18.0], device=device, dtype=dtype)
        data = row.view(1, 1, 1, 6)
        if kernel_size == (3, 1):
            data = data.view(1, 1, 6, 1)
            expected = expected.view(1, 1, 6, 1)
        else:
            expected = expected.view(1, 1, 1, 6)
        if normalized:
            expected = expected / 4.0  # the kernel [1, -2, 1] has an absolute sum of 4
        self.assert_close(laplacian(data, kernel_size, normalized=normalized), expected)

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (11, 7), (3, 3)])
    @pytest.mark.parametrize("normalized", [True, False])
    def test_smoke(self, shape, kernel_size, normalized, device, dtype):
        data = torch.rand(shape, device=device, dtype=dtype)
        actual = laplacian(data, kernel_size, "reflect", normalized)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == shape

    @pytest.mark.parametrize("shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    @pytest.mark.parametrize("kernel_size", [5, (11, 7), 3])
    def test_cardinality(self, shape, kernel_size, device, dtype):
        sample = torch.rand(shape, device=device, dtype=dtype)
        actual = laplacian(sample, kernel_size)
        assert actual.shape == shape

    @pytest.mark.skip(reason="Nothing to test.")
    def test_exception(self): ...

    def test_noncontiguous(self, device, dtype):
        batch_size = 3
        sample = torch.rand(3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        kernel_size = 3
        actual = laplacian(sample, kernel_size)
        assert actual.is_contiguous()

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_export(self, device, dtype):
        inp = torch.rand(1, 2, 7, 9, device=device, dtype=dtype)
        op = Laplacian((5, 7))
        exported = torch.export.export(op, (inp,), strict=True)
        self.assert_close(exported.module()(inp), op(inp))

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    def test_cpu_autocast(self, autocast_dtype):
        # Float32 inputs intentionally exercise CPU autocast's convolution cast.
        inp = torch.rand(1, 1, 7, 9, dtype=torch.float32)
        weights = get_laplacian_kernel2d(3, dtype=torch.float32)[None]
        weights = normalize_kernel2d(weights)
        with torch.autocast("cpu", dtype=autocast_dtype):
            expected = filter2d(inp, weights)
            actual = laplacian(inp, 3)
        assert actual.dtype == expected.dtype == autocast_dtype
        self.assert_close(actual, expected)

    def test_gradcheck(self, device):
        # test parameters
        batch_shape = (1, 2, 5, 7)
        kernel_size = 3

        # evaluate function gradient
        sample = torch.rand(batch_shape, device=device, dtype=torch.float64)
        self.gradcheck(laplacian, (sample, kernel_size))

    def test_module(self, device, dtype):
        params = [3]
        op = laplacian
        op_module = Laplacian(*params)

        img = torch.ones(1, 3, 5, 5, device=device, dtype=dtype)
        self.assert_close(op(img, *params), op_module(img))

    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = Laplacian(kernel_size)
        op_optimized = torch_optimizer(op)

        self.assert_close(op(data), op_optimized(data))


class TestConventionsLaplacian(BaseTester):
    @staticmethod
    def _require_reflect_padding(device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("torch has no reflect padding kernel for this device and dtype")

    def test_convention_laplacian_kernel_sign_and_normalization(self, device, dtype):
        self._require_reflect_padding(device, dtype)
        # delta off centre in a 7x10 image; kernel_size is (kH, kW) = (3, 5), all ones with centre 1 - kH * kW
        delta = torch.zeros(1, 1, 7, 10, device=device, dtype=dtype)
        delta[0, 0, 2, 6] = 1.0
        raw = laplacian(delta, (3, 5), normalized=False)
        expected = torch.zeros_like(delta)
        expected[0, 0, 1:4, 4:9] = 1.0
        expected[0, 0, 2, 6] = -14.0  # negative at a bright peak
        self.assert_close(raw, expected)
        # the default normalized=True divides by the kernel's absolute sum, 2 * (kH * kW - 1) = 28
        self.assert_close(laplacian(delta, (3, 5)), expected / 28)
        # relabel: transposing the image and the kernel size transposes the output
        self.assert_close(laplacian(delta.transpose(-1, -2), (5, 3), normalized=False), raw.transpose(-1, -2))
        # normalized is not derivative units: x^2/2 + y^2/2 has a Laplacian of 2, the k=3 output is 6 / 16
        ys, xs = torch.meshgrid(
            torch.arange(7, device=device, dtype=dtype), torch.arange(10, device=device, dtype=dtype), indexing="ij"
        )
        bowl = ((xs - 4) ** 2 / 2 + (ys - 2) ** 2 / 2)[None, None]
        self.assert_close(
            laplacian(bowl, 3, normalized=False)[0, 0, 3, 5], torch.tensor(6.0, device=device, dtype=dtype)
        )
        self.assert_close(laplacian(bowl, 3)[0, 0, 3, 5], torch.tensor(0.375, device=device, dtype=dtype))

    def test_convention_laplacian_default_border_is_reflect(self, device, dtype):
        self._require_reflect_padding(device, dtype)
        # an x-ramp's Laplacian is 0 inside; at x = 0 torch's reflect gives 0.375, replicate 0.1875
        ramp = (torch.arange(10, device=device, dtype=dtype) + 1).expand(1, 1, 7, 10)
        out = laplacian(ramp, 3)
        self.assert_close(out, laplacian(ramp, 3, border_type="reflect"))
        self.assert_close(out[0, 0, 3, [0, 5]], torch.tensor([0.375, 0.0], device=device, dtype=dtype))

    def test_convention_laplacian_rejects_kernel_size_one_5175(self, device, dtype):
        """kernel_size needs 3 or more taps along one axis: 1 and (1, 1) raise, (1, 3) is accepted."""
        self._require_reflect_padding(device, dtype)
        data = torch.rand(1, 1, 5, 6, device=device, dtype=dtype)
        for kernel_size in (1, (1, 1)):
            with pytest.raises(BaseError):
                laplacian(data, kernel_size)
            with pytest.raises(BaseError):
                Laplacian(kernel_size)
        # a 1 x 3 kernel is the 1-D stencil [1, -2, 1] along x, divided by its absolute sum 4
        ramp = (torch.arange(6, device=device, dtype=dtype) ** 2).expand(1, 1, 5, 6)
        self.assert_close(laplacian(ramp, (1, 3))[0, 0, 2, 1:-1], torch.full((4,), 0.5, device=device, dtype=dtype))

    def test_wart_laplacian_integer_input_returns_zeros_5155(self, device, dtype):
        """#5155: the kernel takes the integer input's dtype, so the normalised taps truncate to 0."""
        self._require_reflect_padding(device, dtype)
        if device.type == "mps":
            pytest.skip("#5155: MPS rejects integer convolution instead of returning zeros")
        img = torch.full((1, 1, 5, 7), 100, device=device, dtype=torch.uint8)
        img[0, 0, 2, 3] = 180
        # in a float dtype: (8 * 100 - 8 * 180) / 16 = -40 at the bright pixel, (180 - 100) / 16 = 5 beside it
        reference = laplacian(img.to(dtype), 3)[0, 0, [2, 1], [3, 3]]
        self.assert_close(reference, torch.tensor([-40.0, 5.0], device=device, dtype=dtype))
        # every tap of the normalised kernel is below 1 in magnitude (1/16 and -8/16 in int16; 1/256 and 248/256 in
        # uint8, whose centre wraps) and truncates to 0, so the wrap does not decide the result
        for int_dtype in (torch.uint8, torch.int16):
            assert laplacian(img.to(int_dtype), 3).count_nonzero().item() == 0
        # unnormalised, the taps survive but the sum is uint8 arithmetic: -640 at the bright pixel reads -640 mod 256
        assert laplacian(img, 3, normalized=False)[0, 0, 2, 3].item() == 128

    def test_convention_laplacian_border_type_is_case_insensitive_5156(self, device, dtype):
        """border_type is case-insensitive: 'REFLECT' and 'Reflect' pad as 'reflect' does, and so for every mode."""
        self._require_reflect_padding(device, dtype)
        generator = torch.Generator().manual_seed(0)
        data = torch.rand(1, 1, 7, 10, generator=generator).to(device=device, dtype=dtype)
        outputs = {}
        for border_type in ("reflect", "circular", "constant"):
            outputs[border_type] = laplacian(data, 3, border_type=border_type)
            for spelling in (border_type.upper(), border_type.capitalize()):
                self.assert_close(laplacian(data, 3, border_type=spelling), outputs[border_type])
        # the modes differ on the border, so a spelling that fell back to another mode would be seen
        assert not torch.allclose(outputs["reflect"], outputs["circular"])
        assert not torch.allclose(outputs["reflect"], outputs["constant"])
