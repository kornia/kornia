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
import warnings
from typing import Optional, Union

import pytest
import torch
import torch.nn.functional as F

from kornia.filters import bilateral_blur, joint_bilateral_blur
from kornia.filters.bilateral import _bilateral_blur_cpu, _bilateral_blur_cpu_eligible
from kornia.filters.kernels import get_gaussian_kernel2d

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

bilateral_module = importlib.import_module("kornia.filters.bilateral")

# One image with 3 x 148 x 148 = 65,712 values: just above the tap loop's threshold.
_ELIGIBLE = (1, 3, 148, 148)


def _require_native_cpu(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cpu" or dtype not in (torch.float32, torch.float64):
        pytest.skip("the CPU tap-loop bilateral path supports native CPU float32 and float64 only")


def _cpu_tolerance(dtype: torch.dtype) -> float:
    # The per-tap weights are the unfold path's; only the order of the kH * kW-term sums differs.
    return 1e-6 if dtype == torch.float32 else 1e-13


def _unfold_reference(
    input: torch.Tensor,
    guidance: Optional[torch.Tensor],
    kernel_size: Union[tuple[int, int], int],
    sigma_color: Union[float, torch.Tensor],
    sigma_space: Union[tuple[float, float], torch.Tensor],
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> torch.Tensor:
    """The unfold-based bilateral filter that the tap loop replaces on large CPU inputs, for floating inputs."""
    ky, kx = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    pad_y, pad_x = (ky - 1) // 2, (kx - 1) // 2
    if isinstance(sigma_color, torch.Tensor):
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1, 1)
    if isinstance(sigma_space, torch.Tensor):
        sigma_space = sigma_space.to(device=input.device, dtype=input.dtype)
    padded_input = F.pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1)
    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        guidance = guidance.to(input.dtype)
        padded_guidance = F.pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1)
    diff = unfolded_guidance - guidance[..., None, None]
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only accepts l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()
    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    kernel = space_kernel.view(-1, 1, 1, 1, ky, kx) * color_kernel
    return (unfolded_input * kernel).sum((-2, -1)) / kernel.sum((-2, -1))


def _counting(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0].shape)
        return _bilateral_blur_cpu(*args, **kwargs)

    monkeypatch.setattr(bilateral_module, "_bilateral_blur_cpu", counting)
    return calls


def _forbid_tap_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("the tap loop must not run here")

    monkeypatch.setattr(bilateral_module, "_bilateral_blur_cpu", fail)


def _assert_same_nonfinite(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual == float("inf"), expected == float("inf"))
    assert torch.equal(actual == -float("inf"), expected == -float("inf"))


class TestBilateralBlurCpuTapLoop(BaseTester):
    @pytest.mark.parametrize("border_type", ["constant", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    @pytest.mark.parametrize("channels", [1, 2, 3])
    def test_matches_unfold_path(self, border_type, color_distance_type, channels, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _counting(monkeypatch)
        image = torch.rand(2, channels, 181, 183, device=device, dtype=dtype)

        actual = bilateral_blur(image, (5, 7), 0.1, (1.5, 2.0), border_type, color_distance_type)
        expected = _unfold_reference(image, None, (5, 7), 0.1, (1.5, 2.0), border_type, color_distance_type)

        assert len(calls) == 1
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("kernel_size", [1, 3, (3, 5), (7, 3), 9])
    def test_kernel_sizes(self, kernel_size, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(*_ELIGIBLE, device=device, dtype=dtype)

        actual = bilateral_blur(image, kernel_size, 0.3, (1.0, 2.5))
        expected = _unfold_reference(image, None, kernel_size, 0.3, (1.0, 2.5))

        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("channels, guidance_channels", [(3, 1), (1, 3), (2, 3), (3, 3)])
    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_joint_matches_unfold_path(
        self, channels, guidance_channels, color_distance_type, monkeypatch, device, dtype
    ):
        _require_native_cpu(device, dtype)
        calls = _counting(monkeypatch)
        image = torch.rand(2, channels, 131, 257, device=device, dtype=dtype)
        guidance = torch.rand(2, guidance_channels, 131, 257, device=device, dtype=dtype)

        actual = joint_bilateral_blur(image, guidance, 5, 0.2, (1.5, 1.5), "reflect", color_distance_type)
        expected = _unfold_reference(image, guidance, 5, 0.2, (1.5, 1.5), "reflect", color_distance_type)

        assert len(calls) == 1
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("guidance_dtype", [torch.float16, torch.float32, torch.float64, torch.uint8])
    def test_joint_guidance_takes_the_input_dtype(self, guidance_dtype, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _counting(monkeypatch)
        image = torch.rand(2, 3, 131, 257, device=device, dtype=dtype)
        guidance = (torch.rand(2, 1, 131, 257, device=device) * 255).to(guidance_dtype)

        actual = joint_bilateral_blur(image, guidance, 5, 30.0, (1.5, 1.5))

        assert len(calls) == 1
        assert actual.dtype == dtype
        tolerance = _cpu_tolerance(dtype)
        expected = _unfold_reference(image, guidance.to(dtype), 5, 30.0, (1.5, 1.5))
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("chunk_images", [1, 2, 3])
    @pytest.mark.parametrize("per_sample", [False, True], ids=["shared_sigma", "per_sample_sigma"])
    def test_batch_chunks(self, chunk_images, per_sample, monkeypatch, device, dtype):
        # Eager calls work through the batch in chunks; per-sample sigmas must follow their samples across chunks,
        # including a last chunk that is shorter than the others.
        _require_native_cpu(device, dtype)
        image = torch.rand(5, 3, 64, 96, device=device, dtype=dtype)
        guidance = torch.rand(5, 2, 64, 96, device=device, dtype=dtype)
        rows = 5 if per_sample else 1
        sigma_color = torch.rand(rows, device=device, dtype=dtype) + 0.05
        sigma_space = torch.rand(rows, 2, device=device, dtype=dtype) + 0.5
        monkeypatch.setattr(bilateral_module, "_BILATERAL_CPU_CHUNK_BYTES", chunk_images * image[0].nbytes)

        tolerance = _cpu_tolerance(dtype)
        for guide in (None, guidance):
            actual = _bilateral_blur_cpu(image, guide, (3, 5), sigma_color, sigma_space, "reflect", "l1")
            expected = _unfold_reference(image, guide, (3, 5), sigma_color, sigma_space, "reflect", "l1")
            self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_chunks_do_not_change_values(self, monkeypatch, device, dtype):
        # Chunking only changes which samples each operation sees; every value is computed the same way.
        _require_native_cpu(device, dtype)
        image = torch.rand(6, 3, 61, 67, device=device, dtype=dtype)
        sigma_color = torch.rand(6, device=device, dtype=dtype) + 0.05

        whole = _bilateral_blur_cpu(image, None, 5, sigma_color, (1.5, 1.5), "reflect", "l1")
        monkeypatch.setattr(bilateral_module, "_BILATERAL_CPU_CHUNK_BYTES", 2 * image[0].nbytes)
        chunked = _bilateral_blur_cpu(image, None, 5, sigma_color, (1.5, 1.5), "reflect", "l1")

        self.assert_close(chunked, whole, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("color_distance_type", ["l1", "l2"])
    def test_functional_form_computes_the_same_values(self, color_distance_type, monkeypatch, device, dtype):
        # Compiled calls run the functional form of the loop over the whole batch; its operations are the in-place
        # form's, so the values are identical.
        _require_native_cpu(device, dtype)
        image = torch.rand(4, 3, 61, 67, device=device, dtype=dtype)
        guidance = torch.rand(4, 2, 61, 67, device=device, dtype=dtype)
        sigma_color = torch.rand(4, device=device, dtype=dtype) + 0.05
        monkeypatch.setattr(bilateral_module, "_BILATERAL_CPU_CHUNK_BYTES", image[0].nbytes)
        args = (5, sigma_color, (1.5, 1.5), "reflect", color_distance_type)

        in_place = [_bilateral_blur_cpu(image, guide, *args) for guide in (None, guidance)]
        monkeypatch.setattr(bilateral_module, "is_compiling", lambda: True)
        functional = [_bilateral_blur_cpu(image, guide, *args) for guide in (None, guidance)]

        for actual, expected in zip(functional, in_place):
            self.assert_close(actual, expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("joint", [False, True])
    def test_preserves_nonfinite_pattern(self, joint, device, dtype):
        _require_native_cpu(device, dtype)
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(2, 3, 128, 160, generator=generator, dtype=dtype)
        flat = image.view(-1)
        positions = torch.randperm(flat.numel(), generator=generator)[:60]
        flat[positions[:20]] = float("nan")
        flat[positions[20:40]] = float("inf")
        flat[positions[40:]] = -float("inf")
        image = image.to(device)
        # A finite guidance gives the non-finite input values positive weights, so infinities reach the output.
        guidance = torch.rand(2, 1, 128, 160, generator=generator, dtype=dtype).to(device) if joint else None

        for sigma_color in (0.1, 10.0):
            actual = _bilateral_blur_cpu(image, guidance, 5, sigma_color, (1.5, 1.5), "reflect", "l1")
            expected = _unfold_reference(image, guidance, 5, sigma_color, (1.5, 1.5), "reflect", "l1")

            _assert_same_nonfinite(actual, expected)
            tolerance = _cpu_tolerance(dtype)
            self.assert_close(actual.nan_to_num(), expected.nan_to_num(), rtol=tolerance, atol=tolerance)
        if joint:
            assert actual.isinf().any()

    def test_large_finite_values_stay_finite(self, device, dtype):
        # The weights of a pixel sum to one up to rounding, so no partial weighted sum of values within half the
        # dtype's range can overflow, in either summation order.
        _require_native_cpu(device, dtype)
        huge = torch.finfo(dtype).max / 2
        image = torch.zeros(1, 1, 256, 256, device=device, dtype=dtype)
        image[..., 100:, :] = huge
        image[..., :, 200:] = -huge
        guidance = torch.zeros(1, 1, 256, 256, device=device, dtype=dtype)

        actual = _bilateral_blur_cpu(image, guidance, 5, 1.0, (1.5, 1.5), "reflect", "l1")
        expected = _unfold_reference(image, guidance, 5, 1.0, (1.5, 1.5), "reflect", "l1")

        assert torch.isfinite(expected).all()
        assert torch.isfinite(actual).all()
        self.assert_close(actual / huge, expected / huge, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "shape, eligible",
        [
            ((1, 1, 256, 255), False),
            ((1, 1, 256, 256), True),
            ((1, 3, 147, 148), False),
            ((1, 3, 148, 148), True),
            ((4, 1, 128, 128), True),
            ((1, 2, 3, 20000), True),
        ],
    )
    def test_threshold_and_odd_shapes(self, shape, eligible, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(*shape, device=device, dtype=dtype)
        assert _bilateral_blur_cpu_eligible(image, None, 0.1, (1.5, 1.5)) == eligible

        tolerance = _cpu_tolerance(dtype)
        self.assert_close(
            bilateral_blur(image, 3, 0.1, (1.5, 1.5), "replicate"),
            _unfold_reference(image, None, 3, 0.1, (1.5, 1.5), "replicate"),
            rtol=tolerance,
            atol=tolerance,
        )

    def test_small_input_uses_unfold(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _forbid_tap_loop(monkeypatch)
        image = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)

        self.assert_close(bilateral_blur(image, 5, 0.1, (1.5, 1.5)), _unfold_reference(image, None, 5, 0.1, (1.5, 1.5)))

    def test_other_devices_and_dtypes_use_unfold(self, monkeypatch, device, dtype):
        if device.type == "cpu" and dtype in (torch.float32, torch.float64):
            pytest.skip("covered by the tap-loop tests")
        _forbid_tap_loop(monkeypatch)
        image = torch.rand(1, 3, 160, 160, device=device, dtype=dtype)

        self.assert_close(bilateral_blur(image, 5, 0.1, (1.5, 1.5)), _unfold_reference(image, None, 5, 0.1, (1.5, 1.5)))

    @pytest.mark.parametrize("which", ["input", "guidance", "sigma_color", "sigma_space"])
    def test_reverse_mode_gradients_use_unfold(self, which, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        _forbid_tap_loop(monkeypatch)
        tensors = {
            "input": torch.rand(*_ELIGIBLE, device=device, dtype=dtype),
            "guidance": torch.rand(1, 1, 148, 148, device=device, dtype=dtype),
            "sigma_color": torch.tensor([0.2], device=device, dtype=dtype),
            "sigma_space": torch.tensor([[1.5, 2.0]], device=device, dtype=dtype),
        }
        if which != "guidance":
            tensors["guidance"] = None
        tensors[which].requires_grad_()
        reference = {
            name: None if tensor is None else tensor.detach().clone().requires_grad_(name == which)
            for name, tensor in tensors.items()
        }
        weights = torch.rand(*_ELIGIBLE, device=device, dtype=dtype)

        def run(op, args):
            return (op(args["input"], args["guidance"], 5, args["sigma_color"], args["sigma_space"]) * weights).sum()

        if tensors["guidance"] is None:
            actual = (
                bilateral_blur(tensors["input"], 5, tensors["sigma_color"], tensors["sigma_space"]) * weights
            ).sum()
        else:
            actual = run(joint_bilateral_blur, tensors)
        actual.backward()
        run(_unfold_reference, reference).backward()

        self.assert_close(tensors[which].grad, reference[which].grad)

    @pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
    def test_requires_grad_input_without_recorded_gradients_takes_the_tap_loop(self, mode, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _counting(monkeypatch)
        image = torch.rand(*_ELIGIBLE, device=device, dtype=dtype, requires_grad=True)
        sigma_color = torch.tensor([0.2], device=device, dtype=dtype, requires_grad=True)

        context = torch.no_grad() if mode == "no_grad" else torch.inference_mode()
        with context:
            actual = bilateral_blur(image, 5, sigma_color, (1.5, 1.5))

        assert len(calls) == 1
        tolerance = _cpu_tolerance(dtype)
        expected = _unfold_reference(image.detach(), None, 5, sigma_color.detach(), (1.5, 1.5))
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_jit_trace_uses_unfold(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(*_ELIGIBLE, device=device, dtype=dtype)
        _forbid_tap_loop(monkeypatch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # check_trace would rerun the function eagerly, where the tap loop is the right path
            traced = torch.jit.trace(lambda x: bilateral_blur(x, 5, 0.1, (1.5, 1.5)), (image,), check_trace=False)
        self.assert_close(traced(image), _unfold_reference(image, None, 5, 0.1, (1.5, 1.5)))

    def test_layouts_give_contiguous_outputs(self, device, dtype):
        # The unfold path returned contiguous outputs for any input layout; callers may .view() them.
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 3, 148, 150, device=device, dtype=dtype)
        transposed = image.transpose(-1, -2).contiguous().transpose(-1, -2)
        channels_last = image.contiguous(memory_format=torch.channels_last)
        guidance = torch.rand(2, 1, 148, 150, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)

        tolerance = _cpu_tolerance(dtype)
        for variant in (transposed, channels_last):
            assert _bilateral_blur_cpu_eligible(variant, None, 0.1, (1.5, 1.5))
            for actual, guide in (
                (bilateral_blur(variant, 5, 0.1, (1.5, 1.5)), None),
                (joint_bilateral_blur(variant, guidance, 5, 0.1, (1.5, 1.5)), guidance),
            ):
                assert actual.is_contiguous()
                assert actual.dtype == dtype
                expected = _unfold_reference(image, guide, 5, 0.1, (1.5, 1.5))
                self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        # a chunked batch is assembled in a contiguous buffer as well
        batch = torch.rand(8, 3, 160, 160, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
        assert bilateral_blur(batch, 3, 0.1, (1.5, 1.5)).is_contiguous()

    def test_integer_input_is_filtered_in_float32(self, monkeypatch, device, dtype):
        _require_native_cpu(device, dtype)
        calls = _counting(monkeypatch)
        image = torch.randint(0, 256, _ELIGIBLE, device=device, dtype=torch.uint8)

        actual = bilateral_blur(image, 5, 25.0, (1.5, 1.5))

        assert len(calls) == 1
        assert actual.dtype == torch.float32
        expected = _unfold_reference(image.float(), None, 5, 25.0, (1.5, 1.5))
        self.assert_close(actual, expected, rtol=1e-6, atol=1e-4)

    @pytest.mark.parametrize("channels", [1, 3])
    def test_invalid_color_distance_type_raises(self, channels, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(1, channels, 256, 256, device=device, dtype=dtype)

        with pytest.raises(ValueError, match="color_distance_type only accepts l1 or l2"):
            bilateral_blur(image, 5, 0.1, (1.5, 1.5), color_distance_type="l3")

    def test_reflect_padding_larger_than_the_image_raises_as_before(self, device, dtype):
        _require_native_cpu(device, dtype)
        image = torch.rand(1, 1, 2, 40000, device=device, dtype=dtype)

        with pytest.raises(RuntimeError) as expected:
            _unfold_reference(image, None, 5, 0.1, (1.5, 1.5), color_distance_type="l3")
        with pytest.raises(RuntimeError) as actual:
            bilateral_blur(image, 5, 0.1, (1.5, 1.5), color_distance_type="l3")
        assert str(actual.value) == str(expected.value)

    def test_cpu_autocast_does_not_change_the_result(self, device, dtype):
        # No operation of either path is on an autocast list except the shared padding, so autocast needs no clause.
        _require_native_cpu(device, dtype)
        image = torch.rand(*_ELIGIBLE, device=device, dtype=dtype)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = bilateral_blur(image, 5, 0.1, (1.5, 1.5))
        expected = _unfold_reference(image, None, 5, 0.1, (1.5, 1.5))

        assert actual.dtype == dtype
        tolerance = _cpu_tolerance(dtype)
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_public_vmap_has_no_user_warning(self, device, dtype):
        _require_native_cpu(device, dtype)
        images = torch.rand(2, 1, 3, 148, 148, device=device, dtype=dtype)
        guidance = torch.rand(2, 1, 1, 148, 148, device=device, dtype=dtype)
        tolerance = _cpu_tolerance(dtype)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = torch.vmap(lambda x: bilateral_blur(x, 5, 0.1, (1.5, 1.5)))(images)
            joint = torch.vmap(lambda x, g: joint_bilateral_blur(x, g, 5, 0.1, (1.5, 1.5)))(images, guidance)
        assert not any(issubclass(warning.category, UserWarning) for warning in recorded)
        expected = torch.stack([_unfold_reference(image, None, 5, 0.1, (1.5, 1.5)) for image in images])
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        expected = torch.stack(
            [_unfold_reference(image, guide, 5, 0.1, (1.5, 1.5)) for image, guide in zip(images, guidance)]
        )
        self.assert_close(joint, expected, rtol=tolerance, atol=tolerance)

    def test_forward_mode_gradients_match_unfold(self, monkeypatch, device, dtype):
        # The tap loop runs for dual inputs, in place on its own temporaries and in chunks, and must carry their
        # tangents as the unfold path does.
        _require_native_cpu(device, dtype)
        import torch.autograd.forward_ad as fwAD

        calls = _counting(monkeypatch)
        monkeypatch.setattr(bilateral_module, "_BILATERAL_CPU_CHUNK_BYTES", 2 * 3 * 148 * 148 * image_bytes(dtype))
        image = torch.rand(3, 3, 148, 148, device=device, dtype=dtype)
        image_tangent = torch.rand_like(image)
        sigma_color = torch.tensor([0.2, 0.3, 0.4], device=device, dtype=dtype)
        sigma_tangent = torch.rand_like(sigma_color)

        with fwAD.dual_level():
            output = bilateral_blur(
                fwAD.make_dual(image, image_tangent), 3, fwAD.make_dual(sigma_color, sigma_tangent), (1.5, 1.5)
            )
            actual = fwAD.unpack_dual(output).tangent
        assert len(calls) == 1
        _, expected = torch.func.jvp(
            lambda x, s: _unfold_reference(x, None, 3, s, (1.5, 1.5)),
            (image, sigma_color),
            (image_tangent, sigma_tangent),
        )
        tolerance = 1e-5 if dtype == torch.float32 else 1e-12
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

        _, actual = torch.func.jvp(
            lambda x, s: bilateral_blur(x, 3, s, (1.5, 1.5)), (image, sigma_color), (image_tangent, sigma_tangent)
        )
        self.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_gradcheck_tap_loop(self, device):
        if device.type != "cpu":
            pytest.skip("the CPU tap-loop bilateral path runs on the CPU only")
        image = torch.rand(2, 2, 6, 7, device=device, dtype=torch.float64)
        guidance = torch.rand(2, 1, 6, 7, device=device, dtype=torch.float64)
        sigma_color = torch.rand(2, device=device, dtype=torch.float64) + 0.5
        sigma_space = torch.rand(2, 2, device=device, dtype=torch.float64) + 0.5
        self.gradcheck(_bilateral_blur_cpu, (image, guidance, 3, sigma_color, sigma_space, "reflect", "l1"))
        self.gradcheck(_bilateral_blur_cpu, (image, None, (3, 5), 0.7, sigma_space, "circular", "l2"))

    def test_dynamo_on_an_eligible_input(self, device, dtype, torch_optimizer):
        _require_native_cpu(device, dtype)
        image = torch.rand(2, 3, 148, 150, device=device, dtype=dtype)
        guidance = torch.rand(2, 1, 148, 150, device=device, dtype=dtype)
        sigma_color = torch.tensor([0.1, 0.3], device=device, dtype=dtype)
        assert _bilateral_blur_cpu_eligible(image, guidance, sigma_color, (1.5, 1.5))

        def op(x, g, s):
            first = bilateral_blur(x, 5, s, (1.5, 1.5))
            return first + joint_bilateral_blur(x, g, (3, 5), 0.2, (1.0, 2.0), "reflect", "l2")

        tolerance = 1e-5 if dtype == torch.float32 else 1e-12
        compiled = torch_optimizer(op, fullgraph=True)
        expected = op(image, guidance, sigma_color)
        self.assert_close(compiled(image, guidance, sigma_color), expected, rtol=tolerance, atol=tolerance)

    def test_dynamo_dynamic_batch_on_eligible_inputs(self, device, dtype, torch_optimizer):
        # The compiled tap loop runs over the whole batch, so one dynamic graph serves every eligible batch.
        _require_native_cpu(device, dtype)
        from torch._dynamo.testing import CompileCounter

        def op(x, sigma_color, sigma_space):
            return bilateral_blur(x, 3, sigma_color, sigma_space, "reflect")

        counter = CompileCounter()
        compiled = torch_optimizer(op, backend=counter, fullgraph=True, dynamic=True)
        for batch in (4, 5, 6):
            image = torch.rand(batch, 3, 129, 131, device=device, dtype=dtype)
            sigma_color = torch.rand(batch, device=device, dtype=dtype) + 0.5
            sigma_space = torch.rand(batch, 2, device=device, dtype=dtype) + 0.5
            assert _bilateral_blur_cpu_eligible(image, None, sigma_color, sigma_space)
            tolerance = _cpu_tolerance(dtype)
            expected = op(image, sigma_color, sigma_space)
            self.assert_close(compiled(image, sigma_color, sigma_space), expected, rtol=tolerance, atol=tolerance)
        assert counter.frame_count == 1


def image_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class _Exportable(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return bilateral_blur(image, 5, 0.1, (1.5, 1.5))


@pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
class TestBilateralBlurCpuExport(BaseTester):
    # torch.export keeps the unfold path: the tap loop's size check would otherwise become a guard of the exported
    # program, which then fails at run time for an image on the other side of the threshold from the example.
    @pytest.mark.parametrize("example, other", [((2, 3, 256, 256), (3, 3, 32, 48)), ((2, 3, 32, 32), (3, 3, 300, 310))])
    def test_dynamic_sizes_run_on_both_sides_of_the_threshold(self, example, other, device, dtype):
        _require_native_cpu(device, dtype)
        dim = torch.export.Dim
        dynamic = {"image": {0: dim("B", min=1, max=64), 2: dim("H", min=8, max=4096), 3: dim("W", min=8, max=4096)}}
        program = torch.export.export(
            _Exportable(), (torch.rand(*example, device=device, dtype=dtype),), dynamic_shapes=dynamic
        )
        assert any(node.target == torch.ops.aten.unfold.default for node in program.graph.nodes)

        exported = program.module()
        for shape in (example, other):
            image = torch.rand(*shape, device=device, dtype=dtype)
            self.assert_close(exported(image), _unfold_reference(image, None, 5, 0.1, (1.5, 1.5)))

    def test_dynamic_batch_of_small_grayscale_images(self, device, dtype):
        # (B, 1, 128, 128) reaches the threshold from B = 4 on, so a size check would constrain the dynamic batch.
        _require_native_cpu(device, dtype)
        dynamic = {"image": {0: torch.export.Dim("B", min=1, max=64)}}
        program = torch.export.export(
            _Exportable(), (torch.rand(8, 1, 128, 128, device=device, dtype=dtype),), dynamic_shapes=dynamic
        )
        image = torch.rand(2, 1, 128, 128, device=device, dtype=dtype)
        self.assert_close(program.module()(image), _unfold_reference(image, None, 5, 0.1, (1.5, 1.5)))
