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

from __future__ import annotations

import importlib
import warnings

import pytest
import torch
import torch.autograd.forward_ad as fwAD

from kornia.filters import filter2d_separable, get_gaussian_kernel1d
from kornia.losses import ssim_loss
from kornia.metrics import ssim
from kornia.metrics.ssim import _ssim_cpu, _ssim_cpu_eligible

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

ssim_module = importlib.import_module("kornia.metrics.ssim")


def _require_cpu_float32(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cpu" or dtype != torch.float32:
        pytest.skip("the channels-last SSIM path runs on CPU float32 only")


def _nchw_reference(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    padding: str = "same",
) -> torch.Tensor:
    """The SSIM map of NCHW convolutions and out-of-place arithmetic, the path the CPU helper replaces."""
    kernel = get_gaussian_kernel1d(window_size, 1.5, device=img1.device, dtype=img1.dtype)
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    p = window_size // 2

    def blur(image: torch.Tensor) -> torch.Tensor:
        out = filter2d_separable(image, kernel, kernel)
        return out[..., p : out.shape[-2] - p, p : out.shape[-1] - p] if padding == "valid" else out

    mu1, mu2 = blur(img1), blur(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = blur(img1**2) - mu1_sq
    sigma2_sq = blur(img2**2) - mu2_sq
    sigma12 = blur(img1 * img2) - mu1_mu2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return num / (den + eps)


def _pair(shape, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    img1 = torch.rand(shape, generator=generator, dtype=dtype)
    img2 = (img1 + 0.1 * torch.randn(shape, generator=generator, dtype=dtype)).clamp(0, 1)
    return img1, img2


def _count_helper_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0].shape)
        return _ssim_cpu(*args, **kwargs)

    monkeypatch.setattr(ssim_module, "_ssim_cpu", counting)
    return calls


def _forbid_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("the channels-last SSIM path must not run here")

    monkeypatch.setattr(ssim_module, "_ssim_cpu", fail)


class TestSsimCpu(BaseTester):
    def test_contiguous_float32_dispatches_once(self, monkeypatch, device, dtype):
        _require_cpu_float32(device, dtype)
        calls = _count_helper_calls(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 40))

        actual = ssim(img1, img2, 11)

        assert len(calls) == 1
        self.assert_close(actual, _nchw_reference(img1, img2, 11))

    def test_ssim_loss_dispatches_once(self, monkeypatch, device, dtype):
        _require_cpu_float32(device, dtype)
        calls = _count_helper_calls(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 40))

        actual = ssim_loss(img1, img2, 11)

        assert len(calls) == 1
        expected = torch.clamp((1.0 - _nchw_reference(img1, img2, 11)) / 2, min=0, max=1).mean()
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("padding", ["same", "valid"])
    @pytest.mark.parametrize("window_size", [1, 3, 5, 11])
    @pytest.mark.parametrize("shape", [(1, 3, 32, 40), (2, 3, 17, 23), (5, 1, 16, 16), (2, 4, 24, 13), (0, 3, 16, 16)])
    def test_matches_nchw_convolutions(self, shape, window_size, padding, device, dtype):
        _require_cpu_float32(device, dtype)
        img1, img2 = _pair(shape)
        assert _ssim_cpu_eligible(img1, img2)

        actual = ssim(img1, img2, window_size, padding=padding)

        assert actual.is_contiguous()
        self.assert_close(actual, _nchw_reference(img1, img2, window_size, padding=padding))

    @pytest.mark.parametrize("max_val, eps", [(255.0, 1e-12), (1.0, 0.0)])
    def test_matches_nchw_convolutions_for_other_constants(self, max_val, eps, device, dtype):
        _require_cpu_float32(device, dtype)
        img1, img2 = _pair((2, 3, 32, 32))
        img1, img2 = (img1 * max_val).round(), (img2 * max_val).round()

        actual = ssim(img1, img2, 11, max_val=max_val, eps=eps)

        self.assert_close(actual, _nchw_reference(img1, img2, 11, max_val=max_val, eps=eps))

    def test_preserves_nonfinite_pattern(self, device, dtype):
        _require_cpu_float32(device, dtype)
        generator = torch.Generator().manual_seed(0)
        img1, img2 = _pair((2, 3, 64, 64))
        positions = torch.randperm(img1.numel(), generator=generator)[:12]
        img1.view(-1)[positions[:4]] = float("nan")
        img1.view(-1)[positions[4:8]] = float("inf")
        img2.view(-1)[positions[8:]] = -float("inf")

        actual = ssim(img1, img2, 11)
        expected = _nchw_reference(img1, img2, 11)

        assert expected.isnan().any()
        assert torch.equal(actual.isnan(), expected.isnan())
        assert torch.equal(actual.isinf(), expected.isinf())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num())

    def test_identical_images(self, device, dtype):
        _require_cpu_float32(device, dtype)
        img, _ = _pair((2, 3, 32, 32))

        actual = ssim(img, img, 11)

        self.assert_close(actual, _nchw_reference(img, img, 11))
        assert (actual > 0.999).all()

    @pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.uint8])
    def test_half_and_integer_images_are_computed_in_float32(self, monkeypatch, input_dtype, device, dtype):
        _require_cpu_float32(device, dtype)
        calls = _count_helper_calls(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        if input_dtype == torch.uint8:
            img1, img2, max_val = (img1 * 255).round().to(input_dtype), (img2 * 255).round().to(input_dtype), 255.0
        else:
            img1, img2, max_val = img1.to(input_dtype), img2.to(input_dtype), 1.0

        actual = ssim(img1, img2, 11, max_val=max_val)

        assert len(calls) == 1
        assert actual.dtype == (torch.float32 if input_dtype == torch.uint8 else input_dtype)
        assert actual.is_contiguous()
        expected = _nchw_reference(img1.float(), img2.float(), 11, max_val=max_val).to(actual.dtype)
        self.assert_close(actual, expected)

    def test_too_small_image_raises_the_padding_error(self, device, dtype):
        _require_cpu_float32(device, dtype)
        img1, img2 = _pair((1, 3, 5, 6))
        kernel = get_gaussian_kernel1d(11, 1.5)
        with pytest.raises(RuntimeError) as expected:
            filter2d_separable(img1, kernel, kernel)
        with pytest.raises(RuntimeError) as actual:
            ssim(img1, img2, 11)
        assert str(actual.value) == str(expected.value)

    @pytest.mark.parametrize("layout", ["channels_last", "transposed", "mixed"])
    def test_noncontiguous_images_keep_the_nchw_path(self, monkeypatch, layout, device, dtype):
        _require_cpu_float32(device, dtype)
        _forbid_helper(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        expected = _nchw_reference(img1, img2, 11)
        if layout == "channels_last":
            img1 = img1.contiguous(memory_format=torch.channels_last)
            img2 = img2.contiguous(memory_format=torch.channels_last)
        elif layout == "transposed":
            img1 = img1.transpose(-1, -2).contiguous().transpose(-1, -2)
            img2 = img2.transpose(-1, -2).contiguous().transpose(-1, -2)
        else:
            img2 = img2.contiguous(memory_format=torch.channels_last)
        assert not _ssim_cpu_eligible(img1, img2)

        self.assert_close(ssim(img1, img2, 11), expected)

    def test_float64_keeps_the_nchw_path(self, monkeypatch, device, dtype):
        if device.type != "cpu":
            pytest.skip("the channels-last SSIM path runs on CPU float32 only")
        _forbid_helper(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32), dtype=torch.float64)
        assert not _ssim_cpu_eligible(img1, img2)

        self.assert_close(ssim(img1, img2, 11), _nchw_reference(img1, img2, 11))

    @pytest.mark.parametrize("requires_grad", [(True, True), (True, False), (False, True)])
    def test_reverse_mode_gradients_keep_the_nchw_path(self, monkeypatch, requires_grad, device, dtype):
        # Under torch.compile a channels-last reverse pass was slower than Inductor's own layout choice.
        _require_cpu_float32(device, dtype)
        _forbid_helper(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        img1, img2 = img1.requires_grad_(requires_grad[0]), img2.requires_grad_(requires_grad[1])
        assert not _ssim_cpu_eligible(img1, img2)
        ref1 = img1.detach().clone().requires_grad_(requires_grad[0])
        ref2 = img2.detach().clone().requires_grad_(requires_grad[1])
        weights = torch.rand(2, 3, 32, 32)
        inputs = [image for image, flag in zip((img1, img2), requires_grad) if flag]
        references = [image for image, flag in zip((ref1, ref2), requires_grad) if flag]

        actual = torch.autograd.grad((ssim(img1, img2, 11) * weights).sum(), inputs)
        expected = torch.autograd.grad((_nchw_reference(ref1, ref2, 11) * weights).sum(), references)

        for actual_grad, expected_grad in zip(actual, expected):
            assert actual_grad.is_contiguous()
            self.assert_close(actual_grad, expected_grad)

    def test_ssim_loss_backward_keeps_the_nchw_path(self, monkeypatch, device, dtype):
        _require_cpu_float32(device, dtype)
        _forbid_helper(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        img1.requires_grad_()
        reference = img1.detach().clone().requires_grad_()

        ssim_loss(img1, img2, 11).backward()
        torch.clamp((1.0 - _nchw_reference(reference, img2, 11)) / 2, min=0, max=1).mean().backward()

        self.assert_close(img1.grad, reference.grad)

    @pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
    def test_requires_grad_images_without_recorded_gradients_take_the_path(self, monkeypatch, mode, device, dtype):
        _require_cpu_float32(device, dtype)
        calls = _count_helper_calls(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        expected = _nchw_reference(img1, img2, 11)
        img1.requires_grad_()
        context = torch.no_grad() if mode == "no_grad" else torch.inference_mode()
        with context:
            actual = ssim(img1, img2, 11)

        assert len(calls) == 1
        self.assert_close(actual, expected)

    def test_forward_mode_ad_matches_nchw_convolutions(self, monkeypatch, device, dtype):
        # A dual tensor records no reverse-mode gradient, so the in-place combination carries the tangents.
        _require_cpu_float32(device, dtype)
        calls = _count_helper_calls(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        tangent = torch.rand_like(img1)
        with fwAD.dual_level():
            actual = fwAD.unpack_dual(ssim(fwAD.make_dual(img1, tangent), img2, 11))
        with fwAD.dual_level():
            expected = fwAD.unpack_dual(_nchw_reference(fwAD.make_dual(img1, tangent), img2, 11))

        assert len(calls) == 1
        self.assert_close(actual.primal, expected.primal)
        self.assert_close(actual.tangent, expected.tangent)

    @pytest.mark.parametrize("in_dims", [(0, 0), (0, None), (None, 0)])
    def test_public_vmap_has_no_user_warning(self, in_dims, device, dtype):
        _require_cpu_float32(device, dtype)
        img1, img2 = _pair((3, 2, 3, 32, 32))
        args = tuple(image if dim == 0 else image[0] for image, dim in zip((img1, img2), in_dims))

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = torch.vmap(lambda a, b: ssim(a, b, 11), in_dims=in_dims)(*args)
        assert not any(issubclass(warning.category, UserWarning) for warning in recorded)

        batched = [arg if dim == 0 else arg.expand(3, *arg.shape) for arg, dim in zip(args, in_dims)]
        expected = torch.stack([ssim(a, b, 11) for a, b in zip(*batched)])
        self.assert_close(actual, expected)

    def test_autocast_keeps_float32(self, device, dtype):
        _require_cpu_float32(device, dtype)
        img1, img2 = _pair((2, 3, 32, 32))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = ssim(img1, img2, 11)
        assert actual.dtype == torch.float32
        self.assert_close(actual, _nchw_reference(img1, img2, 11))

    def test_jit_trace_keeps_the_nchw_path(self, monkeypatch, device, dtype):
        _require_cpu_float32(device, dtype)
        _forbid_helper(monkeypatch)
        img1, img2 = _pair((2, 3, 32, 32))
        traced = torch.jit.trace(lambda a, b: ssim(a, b, 11), (img1, img2), check_trace=False)
        other1, other2 = _pair((2, 3, 32, 32), seed=1)
        self.assert_close(traced(other1, other2), _nchw_reference(other1, other2, 11))

    def test_dynamo_on_an_eligible_input(self, device, dtype, torch_optimizer, optimizer_backend):
        _require_cpu_float32(device, dtype)
        if optimizer_backend == "jit":
            pytest.skip("ssim is not scriptable")
        img1, img2 = _pair((2, 3, 32, 40))
        assert _ssim_cpu_eligible(img1, img2)
        optimized = torch_optimizer(ssim)
        for padding in ("same", "valid"):
            actual = optimized(img1, img2, 11, padding=padding)
            assert actual.is_contiguous()
            self.assert_close(actual, _nchw_reference(img1, img2, 11, padding=padding))


class _Exportable(torch.nn.Module):
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim(img1, img2, 5)


@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
class TestSsimCpuExport(BaseTester):
    def test_dynamic_shapes_keep_the_nchw_path(self, monkeypatch, device, dtype):
        # Exported programs keep main's graph: no channels-last copies, for any batch and image size.
        _require_cpu_float32(device, dtype)
        _forbid_helper(monkeypatch)
        dim = torch.export.Dim
        sizes = {0: dim("B", min=1, max=64), 2: dim("H", min=8, max=4096), 3: dim("W", min=8, max=4096)}
        program = torch.export.export(_Exportable(), _pair((2, 3, 32, 40)), dynamic_shapes=(sizes, sizes))
        monkeypatch.undo()

        exported = program.module()
        for shape in ((2, 3, 32, 40), (1, 3, 9, 70), (5, 3, 64, 48)):
            img1, img2 = _pair(shape)
            self.assert_close(exported(img1, img2), _nchw_reference(img1, img2, 5))
