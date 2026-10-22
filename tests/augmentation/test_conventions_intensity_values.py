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

import math
import warnings

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation._2d.intensity.base import _clamp_unit_range
from kornia.augmentation.random_generator import RectangleEraseGenerator
from kornia.core.exceptions import BaseError, ShapeError
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    equalize,
    equalize_clahe,
    normalize_min_max,
    posterize,
    solarize,
)
from kornia.geometry.transform import resize

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


@pytest.fixture(autouse=True)
def _restore_global_rng(restore_torch_rng):
    # Every pin below seeds the global RNG so its draw is reproducible.  ``torch.manual_seed`` also
    # reseeds the CUDA and MPS generators, so the root fixture (#4446) snapshots and restores all of
    # them; ``fork_rng(devices=[])`` would restore the CPU generator only and shift the draw of an
    # unseeded later test on an accelerator leg.
    return


# One constructor per 2D intensity class for the value-range pins below.  RandomDissolving (its
# constructor downloads a checkpoint), RandomClahe and RandomJPEG have pins of their own.
_INTENSITY_FACTORIES = {
    "ColorJiggle": lambda: K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0),
    "ColorJitter": lambda: K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
    "Denormalize": lambda: K.Denormalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5]), p=1.0),
    "Normalize": lambda: K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5]), p=1.0),
    "RandomAutoContrast": lambda: K.RandomAutoContrast(p=1.0),
    "RandomBoxBlur": lambda: K.RandomBoxBlur(p=1.0),
    "RandomBrightness": lambda: K.RandomBrightness((1.5, 1.5), p=1.0),
    "RandomChannelDropout": lambda: K.RandomChannelDropout(p=1.0),
    "RandomChannelShuffle": lambda: K.RandomChannelShuffle(p=1.0),
    "RandomContrast": lambda: K.RandomContrast((1.5, 1.5), p=1.0),
    "RandomEqualize": lambda: K.RandomEqualize(p=1.0),
    "RandomErasing": lambda: K.RandomErasing(p=1.0),
    "RandomGamma": lambda: K.RandomGamma((2.0, 2.0), (1.0, 1.0), p=1.0),
    "RandomGaussianBlur": lambda: K.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0),
    "RandomGaussianIllumination": lambda: K.RandomGaussianIllumination(p=1.0),
    "RandomGaussianNoise": lambda: K.RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
    "RandomGrayscale": lambda: K.RandomGrayscale(p=1.0),
    "RandomHue": lambda: K.RandomHue((0.25, 0.25), p=1.0),
    "RandomInvert": lambda: K.RandomInvert(p=1.0),
    "RandomLinearCornerIllumination": lambda: K.RandomLinearCornerIllumination(p=1.0),
    "RandomLinearIllumination": lambda: K.RandomLinearIllumination(p=1.0),
    "RandomMedianBlur": lambda: K.RandomMedianBlur(p=1.0),
    "RandomMotionBlur": lambda: K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), p=1.0),
    "RandomPlanckianJitter": lambda: K.RandomPlanckianJitter(p=1.0),
    "RandomPlasmaBrightness": lambda: K.RandomPlasmaBrightness(p=1.0),
    "RandomPlasmaContrast": lambda: K.RandomPlasmaContrast(p=1.0),
    "RandomPlasmaShadow": lambda: K.RandomPlasmaShadow(p=1.0),
    # Exercise the uint8 path here; bits=8 has its own input-clamp regression below.
    "RandomPosterize": lambda: K.RandomPosterize(bits=(3.0, 7.0), p=1.0),
    "RandomRGBShift": lambda: K.RandomRGBShift(p=1.0),
    "RandomRain": lambda: K.RandomRain(number_of_drops=(3, 3), drop_height=(1, 2), drop_width=(1, 2), p=1.0),
    "RandomSaltAndPepperNoise": lambda: K.RandomSaltAndPepperNoise(p=1.0),
    "RandomSaturation": lambda: K.RandomSaturation((2.0, 2.0), p=1.0),
    "RandomSharpness": lambda: K.RandomSharpness(1.0, p=1.0),
    "RandomSnow": lambda: K.RandomSnow(p=1.0),
    "RandomSolarize": lambda: K.RandomSolarize(0.1, 0.1, p=1.0),
}

# Unit-range arithmetic clamps before operating (#4430). AutoContrast instead rescales arbitrary
# ranges; the other classes listed below perform range-independent operations.
_RANGE_DEPENDENT = (
    "ColorJiggle",
    "ColorJitter",
    "RandomBrightness",
    "RandomContrast",
    "RandomEqualize",
    "RandomGamma",
    "RandomGaussianIllumination",
    "RandomHue",
    "RandomLinearCornerIllumination",
    "RandomLinearIllumination",
    "RandomPlanckianJitter",
    "RandomPlasmaBrightness",
    "RandomPlasmaContrast",
    "RandomPlasmaShadow",
    "RandomPosterize",
    "RandomRGBShift",
    "RandomSaturation",
    "RandomSharpness",
    "RandomSnow",
    "RandomSolarize",
)
_RANGE_INDEPENDENT = (
    "Denormalize",
    "Normalize",
    "RandomAutoContrast",
    "RandomBoxBlur",
    "RandomChannelDropout",
    "RandomChannelShuffle",
    "RandomErasing",
    "RandomGaussianBlur",
    "RandomGaussianNoise",
    "RandomGrayscale",
    "RandomInvert",
    "RandomMedianBlur",
    "RandomMotionBlur",
    "RandomRain",
    "RandomSaltAndPepperNoise",
)

# Selected draws exercise clamping before the operation on a constant negative image (#4430).
_NEGATIVE_INPUT_FACTORIES = {
    **{
        name: _INTENSITY_FACTORIES[name]
        for name in (
            "ColorJitter",
            "RandomBrightness",
            "RandomContrast",
            "RandomGaussianIllumination",
            "RandomLinearIllumination",
            "RandomLinearCornerIllumination",
            "RandomPlasmaShadow",
            "RandomSnow",
            "RandomSharpness",
            "RandomSolarize",
        )
    },
    "RandomBrightness-default": lambda: K.RandomBrightness(p=1.0),
    "ColorJiggle-contrast": lambda: K.ColorJiggle(0.0, 0.3, 0.0, 0.0, p=1.0),
}

# Fixture seed for the shared out-of-range images, and the seed drawn before each construct-and-forward.
_FIXTURE_SEED = 1234
_FORWARD_SEED = 0

_HALF = (torch.float16, torch.bfloat16)


def _out_of_range_fixtures(device, dtype) -> dict[str, torch.Tensor]:
    # One float32 CPU draw cast into the run's device/dtype, so every leg sees the same numbers.
    torch.manual_seed(_FIXTURE_SEED)
    base = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
    return {"[0, 2]": base * 2.0, "[-1, 0]": base - 1.0}


def _sync(device) -> None:
    # MPS dispatches asynchronously, so a kernel error raised by the forward under test would
    # otherwise surface inside an unrelated later test.
    if device.type == "mps":
        torch.mps.synchronize()


def _run(name: str, image: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    torch.manual_seed(_FORWARD_SEED if seed is None else seed)
    out = _INTENSITY_FACTORIES[name]()(image)
    _sync(image.device)
    return out


class TestIntensityValueRangeConventions(BaseTester):
    @pytest.mark.parametrize("name", sorted(_INTENSITY_FACTORIES))
    def test_convention_value_range_on_out_of_range_fixtures(self, device, dtype, name):
        assert set(_RANGE_DEPENDENT).isdisjoint(_RANGE_INDEPENDENT)
        assert set(_RANGE_DEPENDENT) | set(_RANGE_INDEPENDENT) == set(_INTENSITY_FACTORIES)
        if name in ("RandomBoxBlur", "RandomGaussianBlur", "RandomMedianBlur") and not supports_reflect_padding(
            device, dtype
        ):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        outside = False
        moved = False
        for image in _out_of_range_fixtures(device, dtype).values():
            torch.manual_seed(_FORWARD_SEED)
            aug = _INTENSITY_FACTORIES[name]()
            out = aug(image)
            moved |= not torch.equal(out, image)
            assert out.dtype == image.dtype
            assert out.device == image.device
            assert torch.equal(out, aug(image, params=aug._params))
            if name in _RANGE_DEPENDENT:
                assert torch.equal(out, aug(image.clamp(0.0, 1.0), params=aug._params))
                assert bool(((out >= 0.0) & (out <= 1.0)).all())
            elif name == "RandomAutoContrast":
                self.assert_close(out, normalize_min_max(image))
            else:
                outside |= bool(((out < 0.0) | (out > 1.0)).any())
        assert moved, name
        if name in _RANGE_INDEPENDENT and name != "RandomAutoContrast":
            assert outside, name

    # `clip_output` is live on both classes: the default clamps, `clip_output=False` returns the raw result.
    # Snippet used to generate expected:
    #   x = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8) * 2
    #   print(cls((1.5, 1.5), p=1.0)(x).max(), cls((1.5, 1.5), clip_output=False, p=1.0)(x).max())
    @pytest.mark.parametrize("name", ["RandomBrightness", "RandomContrast"])
    def test_convention_brightness_and_contrast_clip_output_is_live(self, device, dtype, name):
        cls = getattr(K, name)
        ramp = (torch.linspace(0, 1, 48).reshape(1, 1, 6, 8) * 2).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        clipped = cls((1.5, 1.5), p=1.0)(ramp)
        torch.manual_seed(_FORWARD_SEED)
        raw = cls((1.5, 1.5), clip_output=False, p=1.0)(ramp)
        self.assert_close(clipped.max(), clipped.new_tensor(1.0))
        self.assert_close(raw.max(), raw.new_tensor(2.5 if name == "RandomBrightness" else 3.0))
        negative = _out_of_range_fixtures(device, dtype)["[-1, 0]"]
        torch.manual_seed(_FORWARD_SEED)
        assert float(cls((1.5, 1.5), p=1.0)(negative).min()) >= 0.0
        torch.manual_seed(_FORWARD_SEED)
        assert float(cls((1.5, 1.5), clip_output=False, p=1.0)(negative).min()) < 0.0

    @pytest.mark.parametrize(
        ("name", "factor", "value", "expected"),
        [("RandomBrightness", 1.5, -0.1, 0.4), ("RandomContrast", 0.5, 1.5, 0.75)],
    )
    def test_convention_unclipped_arithmetic_can_bring_input_into_range(
        self, device, dtype, name, factor, value, expected
    ):
        # No clamp does not imply an out-of-range output: -0.1 + (1.5 - 1) = 0.4;
        # 1.5 * 0.5 = 0.75. These complement the out-of-range outputs above.
        image = torch.full((1, 3, 2, 2), value, device=device, dtype=dtype)
        aug = getattr(K, name)((factor, factor), clip_output=False, p=1.0)
        out = aug(image)
        self.assert_close(out, torch.full_like(image, expected))

    # At the default clip_output=True the input is clamped first: clamp(-0.1) + (1.5 - 1) = 0.5 rather than 0.4,
    # and clamp(1.5) * 0.5 = 0.5 rather than 0.75.
    @pytest.mark.parametrize(
        ("name", "factor", "value", "expected"),
        [("RandomBrightness", 1.5, -0.1, 0.5), ("RandomContrast", 0.5, 1.5, 0.5)],
    )
    def test_convention_clipped_arithmetic_clamps_the_input_first_4430(
        self, device, dtype, name, factor, value, expected
    ):
        image = torch.full((1, 3, 2, 2), value, device=device, dtype=dtype)
        out = getattr(K, name)((factor, factor), p=1.0)(image)
        self.assert_close(out, torch.full_like(image, expected))

    @pytest.mark.parametrize("name", sorted(_NEGATIVE_INPUT_FACTORIES))
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_convention_intensity_clamps_negative_input_before_arithmetic_4430(self, device, dtype, name, seed):
        image = torch.full((2, 3, 6, 8), -1.0, device=device, dtype=dtype)
        torch.manual_seed(seed)
        aug = _NEGATIVE_INPUT_FACTORIES[name]()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = aug(image)
            _sync(image.device)
        assert torch.equal(out, aug(torch.zeros_like(image), params=aug._params))
        assert not caught

    @pytest.mark.parametrize(
        ("name", "level", "expected"),
        [
            ("RandomPlasmaContrast", 0.5, 0.0),
            ("RandomPlasmaContrast", 0.2, 0.1),
            ("RandomPlasmaContrast", 0.0, 0.5),
            ("RandomPlasmaBrightness", 0.5, 0.0),
            ("RandomPlasmaBrightness", 0.75, 0.5),
            ("RandomPlasmaBrightness", 1.5, 1.0),
        ],
    )
    def test_convention_random_plasma_clamps_before_applying_map_4430(self, device, dtype, name, level, expected):
        image = torch.full((2, 3, 6, 8), -1.0, device=device, dtype=dtype)
        aug = getattr(K, name)(p=1.0)
        params = aug.forward_parameters(image.shape)
        params["plasma"] = torch.full_like(params["plasma"], level)
        if "intensity" in params:
            params["intensity"] = torch.ones_like(params["intensity"])
        out = aug(image, params=params)
        self.assert_close(out, torch.full_like(image, expected))

    @pytest.mark.parametrize("scale", ["[0, 2]", "[-1, 0]"])
    def test_convention_random_posterize_clamps_before_uint8_conversion_4430(self, device, dtype, scale):
        ramp = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32).to(device=device, dtype=dtype)
        image = ramp * 2.0 if scale == "[0, 2]" else ramp - 1.0
        out = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(image)
        assert torch.equal(out, posterize(image.clamp(0.0, 1.0), 3))
        if scale == "[0, 2]":
            assert len(out.unique()) == 8
            self.assert_close(out.max(), out.new_tensor(224 / 255))

    def test_convention_random_posterize_eight_bits_still_clamps_input_4430(self, device, dtype):
        image = torch.tensor([1.7, -0.4, 2.5, 0.3], device=device, dtype=dtype).reshape(1, 1, 1, 4)
        out = K.RandomPosterize(bits=(8.0, 8.0), p=1.0)(image)
        assert torch.equal(out, image.clamp(0.0, 1.0))

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_convention_random_gamma_clamps_before_power_4430(self, device, dtype, gamma):
        image = _out_of_range_fixtures(device, dtype)["[-1, 0]"].requires_grad_()
        out = K.RandomGamma((gamma, gamma), (1.0, 1.0), p=1.0)(image)
        assert bool(out.isfinite().all())
        self.assert_close(out, torch.full_like(image, 1.0 if gamma == 0.0 else 0.0))
        out.sum().backward()
        self.assert_close(image.grad, torch.zeros_like(image))

    def test_convention_random_equalize_clamps_before_histogram_4430(self, device, dtype):
        ramp = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).to(device=device, dtype=dtype)
        for image in (ramp * 2.0, ramp - 1.0, ramp, ramp * 1.0001, ramp - 0.003):
            out = K.RandomEqualize(p=1.0)(image)
            assert torch.equal(out, equalize(image.clamp(0.0, 1.0)))

    # Issue #4576: a skipped row must not trigger a value check or contribute a NaN gradient.
    def test_convention_p_gate_skipped_sample_has_no_effect_4576(self, device, dtype):
        ramp = torch.linspace(0, 1, 1024).reshape(1, 1, 32, 32)
        image = torch.cat([ramp, ramp * 2.0]).to(device=device, dtype=dtype)
        for cls in (K.RandomEqualize, K.RandomClahe):
            torch.manual_seed(_FORWARD_SEED)
            out = cls(p=0.0)(image)
            assert torch.equal(out, image)
        volume = image.reshape(2, 1, 4, 16, 16)
        assert torch.equal(K.RandomEqualize3D(p=0.0)(volume), volume)
        values = torch.tensor([0.0, 0.25, 1.0]).reshape(1, 1, 1, 3).repeat(2, 1, 1, 1)
        values = values.to(device=device, dtype=dtype).requires_grad_()
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGamma((0.5, 0.5), (1.0, 1.0), p=0.0)(values)
        assert torch.equal(out, values)
        out.sum().backward()
        assert torch.isfinite(values.grad).all()
        self.assert_close(values.grad, torch.ones_like(values.grad))

    @pytest.mark.parametrize("cls", [K.RandomEqualize, K.RandomClahe])
    def test_mixed_p_gate_skips_invalid_image_4576(self, cls, device, dtype):
        ramp = torch.linspace(0, 1, 1024, device=device, dtype=dtype).reshape(1, 1, 32, 32)
        image = torch.cat([ramp, ramp * 2.0])
        aug = cls(p=0.5)
        params = aug.forward_parameters(image.shape)
        params["batch_prob"] = torch.tensor([1.0, 0.0], device=params["batch_prob"].device)
        out = aug(image, params=params)
        expected = cls(p=1.0)(image[:1])
        self.assert_close(out[:1], expected)
        assert torch.equal(out[1:], image[1:])

    def test_mixed_p_gate_skipped_gamma_gradient_is_identity_4576(self, device, dtype):
        image = torch.tensor([[[[0.25, 0.5, 1.0]]], [[[0.0, 0.25, 0.5]]]], device=device, dtype=dtype).requires_grad_()
        aug = K.RandomGamma((0.5, 0.5), (1.0, 1.0), p=0.5)
        params = aug.forward_parameters(image.shape)
        params["batch_prob"] = torch.tensor([1.0, 0.0], device=params["batch_prob"].device)
        out = aug(image, params=params)
        self.assert_close(out[0], image[0].sqrt())
        assert torch.equal(out[1], image[1])
        out.sum().backward()
        assert torch.isfinite(image.grad).all()
        # Exclude x=1, where adjust_gamma's clamp derivative differs across torch releases.
        self.assert_close(image.grad[0, ..., :2], 0.5 / image[0, ..., :2].detach().sqrt())
        self.assert_close(image.grad[1], torch.ones_like(image.grad[1]))

    @pytest.mark.parametrize(
        ("factory", "substituted"),
        [(lambda: K.RandomGamma((0.5, 0.5), p=0.5), True), (lambda: K.RandomBrightness((0.9, 1.1), p=0.5), False)],
    )
    def test_p_gate_substitutes_skipped_samples_only_where_needed_4576(self, factory, substituted, device, dtype):
        image = torch.linspace(0, 1, 384, device=device, dtype=dtype).reshape(2, 3, 8, 8)
        aug = factory()
        params = aug.forward_parameters(image.shape)
        params["batch_prob"] = torch.tensor([1.0, 0.0])
        seen = []
        original = aug.apply_transform

        def spy(input, *args, **kwargs):
            seen.append(input)
            return original(input, *args, **kwargs)

        aug.apply_transform = spy
        aug(image, params=params)
        assert torch.equal(seen[0][0], image[0])
        assert torch.equal(seen[0][1], torch.full_like(image[1], 0.5) if substituted else image[1])


class TestIntensityColourConventions(BaseTester):
    # RandomGamma is `clamp(gain * x ** gamma, 0, 1)` with no `clip_output` to disable the clamp.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8)
    #   torch.manual_seed(0); y = K.RandomGamma((gamma, gamma), (gain, gain), p=1.0)(g)
    #   print((y - gain * g**gamma).abs().max())
    @pytest.mark.parametrize(("gamma", "gain", "unclamped_gap"), [(1.0, 2.0, 1.0), (0.5, 1.5, 0.5)])
    def test_convention_random_gamma_is_clamped_gain_times_power(self, device, dtype, gamma, gain, unclamped_gap):
        ramp = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGamma((gamma, gamma), (gain, gain), p=1.0)(ramp)
        raw = gain * ramp**gamma
        self.assert_close(out, raw.clamp(0.0, 1.0))
        assert float((out - raw).abs().max()) > unclamped_gap / 2

    def test_convention_random_gamma_negative_exponent_is_unchecked_on_mps(self, device, dtype):
        if device.type != "mps":
            pytest.skip("MPS only: CPU rejects negative gamma; CUDA would invalidate the context with a device assert")
        # kornia does not run the non-negativity check for an MPS image, so gamma=-1 is evaluated as
        # clamp(gain / input, 0, 1).
        image = torch.tensor([0.2, 0.5, 0.8], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        out = K.RandomGamma((-1.0, -1.0), (0.1, 0.1), p=1.0)(image)
        self.assert_close(out, image.new_tensor([0.5, 0.2, 0.125]).reshape_as(image))

    def test_convention_random_saturation_clamps_input_and_rgb_4430(self, device, dtype):
        pixels = [[-0.1, 0.5, 0.5], [1.5, 0.0, 0.0], [-0.1, -0.5, -0.9], [2.0, 1.0, 0.5]]
        image = torch.tensor(pixels, device=device, dtype=dtype).reshape(4, 3, 1, 1)
        out = K.RandomSaturation((1.0, 1.0), p=1.0)(image)
        self.assert_close(out, image.clamp(0.0, 1.0))

    # RandomBrightness passes `factor - 1` to adjust_brightness, whose identity is 0.0, while RandomContrast
    # and RandomSaturation pass their factor through unchanged.
    # Snippet used to generate expected:
    #   c = torch.full((1, 1, 2, 2), 0.4); torch.manual_seed(0); print(K.RandomBrightness((1.5, 1.5), p=1.0)(c))
    def test_convention_brightness_factor_is_one_centred(self, device, dtype):
        constant = torch.full((1, 1, 2, 2), 0.4, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        brightened = K.RandomBrightness((1.5, 1.5), p=1.0)(constant)
        self.assert_close(brightened, adjust_brightness(constant, 0.5))
        self.assert_close(brightened, torch.full_like(constant, 0.9))
        assert float((brightened - adjust_brightness(constant, 1.5)).abs().max()) > 0.05

        torch.manual_seed(_FORWARD_SEED)
        contrasted = K.RandomContrast((1.5, 1.5), p=1.0)(constant)
        self.assert_close(contrasted, adjust_contrast(constant, 1.5))
        self.assert_close(contrasted, torch.full_like(constant, 0.6))

        pixel = torch.tensor([[0.6], [0.4], [0.2]], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        torch.manual_seed(_FORWARD_SEED)
        saturated = K.RandomSaturation((2.0, 2.0), p=1.0)(pixel)
        self.assert_close(saturated, adjust_saturation(pixel, 2.0))
        self.assert_close(saturated, pixel.new_tensor([0.6, 0.3, 0.0]).reshape(1, 3, 1, 1))

    # Each class's identity factor is 1.0, checked on two distinct samples.
    @pytest.mark.parametrize("cls", [K.RandomBrightness, K.RandomContrast, K.RandomSaturation])
    def test_convention_point_range_factor_one_is_identity(self, device, dtype, cls):
        if dtype in _HALF and cls is K.RandomSaturation:
            pytest.skip("adjust_saturation's HSV round trip is lossier than the half-precision tolerance")
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(cls((1.0, 1.0), p=1.0)(image), image)

    # With matching bounds the two classes draw identical parameters, `order` included, from the same seed;
    # replaying ColorJitter with ColorJiggle's parameters reproduces ColorJitter's output, so the gap between
    # the two outputs is the primitives, not the draw.
    # Snippet used to generate expected:
    #   torch.manual_seed(7); a = K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0); ya = a(x)
    #   torch.manual_seed(7); b = K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0); print((ya - b(x)).abs().max())
    def test_convention_color_jiggle_and_jitter_matching_ranges_draw_identical_params(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(7)
        jiggle = K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0)
        jiggled = jiggle(image)
        torch.manual_seed(7)
        jitter = K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0)
        jittered = jitter(image)
        assert sorted(jiggle._params) == sorted(jitter._params)
        assert "order" in jiggle._params
        for key, value in jiggle._params.items():
            assert torch.equal(value, jitter._params[key]), f"{key} differs between the two classes"
        assert float((jiggled - jittered).abs().max()) > 0.05
        replayed = jitter(image, params=jiggle._params)
        self.assert_close(replayed, jittered)
        assert float((jiggled - replayed).abs().max()) > 0.05

    @pytest.mark.device_agnostic
    def test_convention_color_jiggle_and_jitter_have_different_brightness_bounds(self):
        # A scalar brightness above 1 overshoots ColorJiggle's [0, 2] bound and is rejected there, while
        # ColorJitter reads it as [0, 1 + brightness].
        with pytest.raises(ValueError, match="brightness out of bounds"):
            K.ColorJiggle(brightness=1.5)
        torch.manual_seed(7)
        jitter_brightness = K.ColorJitter(brightness=1.5).forward_parameters((256, 3, 2, 2))["brightness_factor"]
        assert bool(((jitter_brightness >= 0) & (jitter_brightness <= 2.5)).all())
        assert bool((jitter_brightness > 2).any())
        # At a scalar within both bounds the two draw the same factors from the same seed.
        torch.manual_seed(7)
        jiggle_params = K.ColorJiggle(brightness=0.5).forward_parameters((4, 3, 2, 2))
        torch.manual_seed(7)
        jitter_params = K.ColorJitter(brightness=0.5).forward_parameters((4, 3, 2, 2))
        for key in jiggle_params:
            assert torch.equal(jiggle_params[key], jitter_params[key]), key

    @pytest.mark.parametrize("hue", [0.0, (0.1, 0.1)])
    def test_convention_color_jiggle_clamps_even_with_neutral_steps_4430(self, device, dtype, hue):
        image = torch.full((2, 3, 6, 8), 2.0, device=device, dtype=dtype)
        out = K.ColorJiggle(0.0, 0.0, 0.0, hue, p=1.0)(image)
        self.assert_close(out, torch.ones_like(image), atol=0, rtol=0)
        chroma = torch.tensor([2.0, 1.5, 0.5], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        aug = K.ColorJiggle(0.0, 0.0, 0.0, hue, p=1.0)
        rotated = aug(chroma)
        self.assert_close(rotated, aug(chroma.clamp(0.0, 1.0), params=aug._params))
        self.assert_close(rotated.amax(1), torch.ones_like(rotated.amax(1)))

    # Both classes are the identity at (0, 0, 0, 0) on an in-range image.
    @pytest.mark.parametrize("cls", [K.ColorJiggle, K.ColorJitter])
    def test_convention_color_jiggle_and_jitter_are_identity_at_zero(self, device, dtype, cls):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(cls(0.0, 0.0, 0.0, 0.0, p=1.0)(image), image)

    # Issue #4785: the neutral brightness step is skipped even if it occurs in a fixed order.
    # The enclosing input clamp now runs independently of which colour steps are enabled (#4430).
    @pytest.mark.parametrize(("step", "kwargs"), [(1, {"contrast": (1.9, 1.9)}), (2, {"saturation": (1.9, 1.9)})])
    def test_convention_color_jitter_default_brightness_step_is_skipped_4785(self, device, dtype, step, kwargs):
        torch.manual_seed(_FIXTURE_SEED)
        negative = (torch.rand(1, 3, 8, 8) - 1.0).to(device=device, dtype=dtype)
        step_only = K.ColorJitter(p=1.0, order=(step,), **kwargs)(negative)
        assert float(step_only.min()) >= 0.0
        assert float(step_only.max()) == 0.0
        brightness_first = K.ColorJitter(p=1.0, order=(0, step), **kwargs)(negative)
        self.assert_close(brightness_first, step_only, atol=0, rtol=0)
        step_first = K.ColorJitter(p=1.0, order=(step, 0), **kwargs)(negative)
        self.assert_close(step_first, step_only, atol=0, rtol=0)
        # The enclosing input clamp applies even when every colour step is neutral.
        above = torch.full((1, 3, 2, 2), 2.0, device=device, dtype=dtype)
        self.assert_close(K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0, order=(1, 2, 3))(above), above.clamp(0.0, 1.0))
        self.assert_close(K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0)(above), above.clamp(0.0, 1.0))

    # A fixed constructor `order` ignores a forward `order=` tensor; without one, a forward `order=` replaces
    # ColorJiggle's drawn order, so leaving the hue step out lets a one-channel image through.  ColorJitter's
    # saturation step rejects four channels, clamps a three-channel image and passes a one-channel image.
    def test_convention_color_jitter_order_override_and_channel_counts(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4).to(device=device, dtype=dtype)
        fixed = K.ColorJitter(brightness=(0.5, 0.5), order=(1,), p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        assert torch.equal(fixed(image, order=torch.tensor([0])), fixed(image))
        fixed_jiggle = K.ColorJiggle(brightness=(0.5, 0.5), order=(1,), p=1.0)
        assert torch.equal(fixed_jiggle(image, order=torch.tensor([0])), fixed_jiggle(image))
        assert float((K.ColorJitter(brightness=(0.5, 0.5), p=1.0, order=(0,))(image) - image).abs().max()) > 0.1

        gray = image[:, :1].contiguous()
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match="shape of"):
            K.ColorJiggle(hue=0.1, p=1.0)(gray)
        torch.manual_seed(_FORWARD_SEED)
        assert K.ColorJiggle(hue=0.1, p=1.0)(gray, order=torch.tensor([0, 1, 2])).shape == gray.shape

        saturation = K.ColorJitter(saturation=(0.5, 0.5), order=(2,), p=1.0)
        two = torch.full((2, 1, 3, 3), 2.0, device=device, dtype=dtype)
        self.assert_close(saturation(two), two.clamp(0.0, 1.0))
        self.assert_close(saturation(two.expand(2, 3, 3, 3)), torch.ones_like(two.expand(2, 3, 3, 3)))
        four_channel = torch.rand(2, 4, 3, 3).to(device=device, dtype=dtype)
        with pytest.raises(BaseError, match="Not a color or gray tensor"):
            saturation(four_channel)

    # Issue #4785: ColorJitter's brightness step multiplies (identity 1), so a factor of 0 gives a black image
    # whatever the other samples drew: a batch whose factors are all 0 comes back black, like the same sample in
    # a mixed batch, and a fixed order that includes the step does the same.
    def test_convention_color_jitter_zero_brightness_factor_is_black_4785(self, device, dtype):
        image = torch.full((2, 3, 2, 2), 0.5, device=device, dtype=dtype)
        aug = K.ColorJitter(brightness=(0.0, 0.0), p=1.0)
        assert float(aug(image).abs().max()) == 0.0
        fixed = K.ColorJitter(brightness=(0.0, 0.0), p=1.0, order=(0,))
        assert float(fixed(image).abs().max()) == 0.0
        params = aug.forward_parameters(image.shape)
        params["brightness_factor"] = torch.tensor([0.0, 1.0])
        mixed = aug(image, params=params)
        assert float(mixed[0].abs().max()) == 0.0
        self.assert_close(mixed[1], image[1])

    # RandomHue's argument is turns of the hue circle; the class multiplies it by 2*pi for adjust_hue.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); y = K.RandomHue((0.25, 0.25), p=1.0)(x)
    #   print((y - adjust_hue(x, 0.25 * 2 * math.pi)).abs().max(), (y - adjust_hue(x, 0.25)).abs().max())
    def test_convention_random_hue_is_turns_of_the_hue_circle(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomHue((0.25, 0.25), p=1.0)(image)
        self.assert_close(out, adjust_hue(image, 0.25 * 2 * math.pi))
        assert float((out - adjust_hue(image, 0.25)).abs().max()) > 0.5

    def test_convention_random_hue_keeps_clamped_extremes_4430(self, device, dtype):
        pixels = torch.tensor([[1.5, 0.2, 0.3], [0.5, -0.1, 0.2], [-0.2, -0.5, -0.9]], device=device, dtype=dtype)
        out = K.RandomHue((0.25, 0.25), p=1.0)(pixels.reshape(3, 3, 1, 1)).reshape(3, 3)
        self.assert_close(out.amax(1), pixels.clamp(0.0, 1.0).amax(1))
        self.assert_close(out.amin(1), pixels.clamp(0.0, 1.0).amin(1))

    # RandomGrayscale keeps the channel count, writes the same value into every channel and uses the BT.601
    # luma weights by default (pure red -> 0.299); custom weights replace them.
    def test_convention_random_grayscale_keeps_channels_and_weights(self, device, dtype):
        red = torch.zeros(1, 3, 6, 8, device=device, dtype=dtype)
        red[0, 0] = 1.0
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGrayscale(p=1.0)(red)
        assert out.shape == red.shape
        self.assert_close(out, out[:, :1].expand_as(out))
        self.assert_close(out[0, 0, 0, 0], out.new_tensor(0.299))
        weights = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        weighted = K.RandomGrayscale(rgb_weights=weights, p=1.0)(red)
        self.assert_close(weighted[0, 0, 0, 0], weighted.new_tensor(1.0))
        # Channel mixing can put out-of-range input back in range without clamping it.
        reduced = K.RandomGrayscale(p=1.0)(2 * red)
        self.assert_close(reduced, torch.full_like(red, 0.598))
        # The weights are applied as given, not normalized.
        half = torch.full((1, 3, 2, 2), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        unnormalized = K.RandomGrayscale(rgb_weights=torch.ones(3, device=device, dtype=dtype), p=1.0)(half)
        self.assert_close(unnormalized, torch.full_like(half, 1.5))
        wide = torch.full((1, 4, 2, 2), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        wide_out = K.RandomGrayscale(rgb_weights=torch.ones(4, device=device, dtype=dtype), p=1.0)(wide)
        self.assert_close(wide_out, torch.full_like(wide, 2.0))

    # A non-RGB channel count is not rejected: every channel becomes the plain mean over channels.
    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_convention_random_grayscale_non_rgb_is_the_channel_mean(self, device, dtype, channels):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(1, channels, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGrayscale(p=1.0)(image)
        assert out.shape == image.shape
        self.assert_close(out, image.mean(1, keepdim=True).expand_as(image))

    # Output channel `c` of sample `b` is input channel `channels[b][c]`, drawn per sample unless
    # same_on_batch; per-sample constant planes 0..11 make the relabelling visible.
    def test_convention_random_channel_shuffle_indexes_the_source_channel(self, device, dtype):
        planes = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1).expand(4, 3, 6, 8)
        image = planes.contiguous().to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomChannelShuffle(p=1.0)
        out = aug(image)
        order = aug._params["channels"]
        assert order.shape == (4, 3)
        identity = torch.arange(3, device=order.device)
        assert not bool((order == identity).all()), "the seeded draw is the identity on every sample"
        assert not bool((order == order[0]).all()), "the seeded draw is one permutation for every sample"
        for b in range(4):
            for c in range(3):
                self.assert_close(out[b, c], image[b, int(order[b, c])])
        # same_on_batch collapses the draw to a single permutation.
        torch.manual_seed(_FORWARD_SEED)
        shared = K.RandomChannelShuffle(p=1.0, same_on_batch=True)
        shared(image)
        assert bool((shared._params["channels"] == shared._params["channels"][0]).all())

    # `fill_value` is written into exactly the channels named by the drawn `channel_idx`, one draw per sample.
    @pytest.mark.parametrize(("num_drop_channels", "fill_value"), [(1, 0.0), (2, 0.7)])
    def test_convention_random_channel_dropout_fills_the_named_channels(
        self, device, dtype, num_drop_channels, fill_value
    ):
        image = torch.ones(2, 3, 6, 8, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomChannelDropout(num_drop_channels=num_drop_channels, fill_value=fill_value, p=1.0)
        out = aug(image)
        index = aug._params["channel_idx"]
        assert index.shape == (2, num_drop_channels)
        for b in range(2):
            dropped = {int(c) for c in index[b]}
            assert len(dropped) == num_drop_channels
            for c in range(3):
                expected = fill_value if c in dropped else 1.0
                self.assert_close(out[b, c], torch.full_like(out[b, c], expected))
        # The draw is per sample; two samples can coincide by chance, so distinctness is checked on B=8.
        torch.manual_seed(_FORWARD_SEED)
        wide = K.RandomChannelDropout(num_drop_channels=num_drop_channels, fill_value=fill_value, p=1.0)
        wide(torch.ones(8, 3, 6, 8, device=device, dtype=dtype))
        wide_index = wide._params["channel_idx"]
        assert wide_index.shape == (8, num_drop_channels)
        assert len({tuple(row) for row in wide_index.tolist()}) > 1, "the seeded per-sample draws coincide"
        torch.manual_seed(_FORWARD_SEED)
        batched = K.RandomChannelDropout(
            num_drop_channels=num_drop_channels, fill_value=fill_value, p=1.0, same_on_batch=True
        )
        batched(image)
        shared = batched._params["channel_idx"]
        assert shared.shape == (2, num_drop_channels)
        assert shared[0].tolist() == shared[1].tolist()

    # RandomInvert is `max_val - x` with no clamp.
    @pytest.mark.parametrize(("max_val", "expected_range"), [(1.0, (-1.0, 1.0)), (2.0, (0.0, 2.0))])
    def test_convention_random_invert_subtracts_from_max_val(self, device, dtype, max_val, expected_range):
        ramp = torch.linspace(0, 2, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomInvert(max_val=max_val, p=1.0)(ramp)
        self.assert_close(out, max_val - ramp)
        self.assert_close(out.min(), out.new_tensor(expected_range[0]))
        self.assert_close(out.max(), out.new_tensor(expected_range[1]))

    # The addition comes first and the sum is clamped into [0, 1]; only then is everything at or above the
    # threshold inverted.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); y = K.RandomSolarize((t, t), (a, a), p=1.0)(g)
    #   z = (g + a).clamp(0, 1); print((y - torch.where(z < t, z, 1.0 - z)).abs().max())
    @pytest.mark.parametrize(("threshold", "addition"), [(0.5, 0.0), (0.5, 0.25), (0.0, 0.0)])
    def test_convention_random_solarize_adds_before_inverting(self, device, dtype, threshold, addition):
        ramp = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSolarize((threshold, threshold), (addition, addition), p=1.0)(ramp)
        shifted = (ramp + addition).clamp(0.0, 1.0)
        self.assert_close(out, torch.where(shifted < threshold, shifted, 1.0 - shifted))
        if (threshold, addition) == (0.5, 0.25):
            self.assert_close(out.flatten()[1], out.new_tensor(1.0 / 47.0 + 0.25))

    # Outside [0, 1] the result depends on the drawn addition and threshold: a positive addition lifts a
    # negative pixel above zero before the clamp, a negative one brings 1.1 below 1 before the inversion
    # (1 - 0.7 = 0.3), and a threshold of 0 inverts the clamped zeros to ones.
    @pytest.mark.parametrize(
        ("value", "threshold", "addition", "expected"),
        [
            (-0.001, 0.5, -0.1, 0.0),
            (-0.001, 0.5, 0.1, 0.1),
            (1.1, 0.5, -0.4, 0.4),
            (1.1, 0.5, 0.0, 0.0),
            (-0.001, 0.0, 0.0, 1.0),
        ],
    )
    def test_convention_random_solarize_out_of_range_depends_on_the_draw(
        self, device, dtype, value, threshold, addition, expected
    ):
        image = torch.full((1, 3, 6, 8), value, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSolarize(thresholds=(threshold, threshold), additions=(addition, addition), p=1.0)(image)
        self.assert_close(out, torch.full_like(image, expected))

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_convention_random_solarize_clamps_before_addition_4430(self, device, dtype, seed):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.empty(2, 3, 6, 8).uniform_(1.5, 3.0).to(device=device, dtype=dtype)
        torch.manual_seed(seed)
        aug = K.RandomSolarize(p=1.0)
        out = aug(image)
        expected = solarize(
            torch.ones_like(image), aug._params["thresholds"].to(image), aug._params["additions"].to(image)
        )
        assert torch.equal(out, expected)
        # A negative addition now changes a formerly all-zero result.
        self.assert_close(K.RandomSolarize((0.5, 0.5), (-0.25, -0.25), p=1.0)(image), torch.full_like(image, 0.25))

    # A scalar `thresholds` is a half-width around 0.5 and a scalar `additions` one around 0.
    @pytest.mark.device_agnostic
    def test_convention_random_solarize_scalar_is_a_centred_range(self):
        torch.manual_seed(_FORWARD_SEED)
        params = K.RandomSolarize(0.1, 0.1, p=1.0).forward_parameters((256, 1, 6, 8))
        assert params["thresholds"].shape == (256,)
        assert params["additions"].shape == (256,)
        assert float(params["thresholds"].min()) >= 0.4
        assert float(params["thresholds"].max()) <= 0.6
        assert float(params["thresholds"].max()) > 0.5 > float(params["thresholds"].min())
        assert float(params["additions"].min()) >= -0.1
        assert float(params["additions"].max()) <= 0.1
        assert float(params["additions"].max()) > 0.0 > float(params["additions"].min())
        # The draw reaches the documented half-width, not a narrower one.
        assert float(params["thresholds"].max()) > 0.58
        assert float(params["thresholds"].min()) < 0.42
        assert float(params["additions"].max()) > 0.09
        assert float(params["additions"].min()) < -0.09

    # `bits=(k, k)` leaves exactly 2**k distinct levels on a 256-level ramp.
    @pytest.mark.parametrize("bits", [0, 3, 8])
    def test_convention_random_posterize_bits_are_levels(self, device, dtype, bits):
        ramp = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32)
        image = ramp.to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(float(bits), float(bits)), p=1.0)(image)
        assert len(out.unique()) == 2**bits

    # An `int` argument is the lower bound of [x, 8] -- the opposite of RandomSharpness's scalar.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("bits", [3, 6])
    def test_convention_random_posterize_int_argument_is_a_lower_bound(self, bits):
        torch.manual_seed(_FORWARD_SEED)
        drawn = K.RandomPosterize(bits=bits, p=1.0).forward_parameters((256, 1, 4, 4))["bits_factor"]
        assert drawn.dtype == torch.int32
        assert int(drawn.min()) == bits
        assert int(drawn.max()) == 8

    # The factor blends linearly between the blurred image (0) and the input (1); above 1 it sharpens.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); print((K.RandomSharpness((f, f), p=1.0)(x) - x).abs().max())
    def test_convention_random_sharpness_factor_is_identity_at_one(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        out = {}
        for factor in (0.0, 0.5, 1.0, 2.0):
            torch.manual_seed(_FORWARD_SEED)
            out[factor] = K.RandomSharpness((factor, factor), p=1.0)(image)
        assert float((out[0.0] - image).abs().max()) > 0.1
        self.assert_close(out[0.5] - image, 0.5 * (out[0.0] - image))
        self.assert_close(out[1.0], image)
        assert float((out[2.0] - image).abs().max()) > 0.1
        assert float(out[2.0].min()) >= 0.0
        assert float(out[2.0].max()) <= 1.0
        # The one-pixel border is copied from the input at every factor: neither blurred nor sharpened.
        border = torch.ones(image.shape, dtype=torch.bool, device=image.device)
        border[..., 1:-1, 1:-1] = False
        for factor in (0.0, 2.0):
            assert torch.equal(out[factor][border], image[border])
        # ... but the final clamp into [0, 1] reaches the copied border too.
        outside = torch.full((2, 1, 4, 4), 0.5, device=device, dtype=dtype)
        outside[:, :, 0, 0] = 1.5
        outside[:, :, 0, 1] = -0.25
        torch.manual_seed(_FORWARD_SEED)
        corner = K.RandomSharpness((1.0, 1.0), p=1.0)(outside)[:, 0, 0, :2]
        self.assert_close(corner, outside.new_tensor([[1.0, 0.0], [1.0, 0.0]]))

    # A scalar `sharpness` is the upper bound of [0, x], so the default 0.5 never reaches the identity.
    @pytest.mark.device_agnostic
    def test_convention_random_sharpness_scalar_is_an_upper_bound(self):
        torch.manual_seed(_FORWARD_SEED)
        drawn = K.RandomSharpness(0.5, p=1.0).forward_parameters((256, 1, 6, 8))["sharpness"]
        assert drawn.shape == (256,)
        assert float(drawn.min()) >= 0.0
        assert float(drawn.max()) <= 0.5
        # The draw reaches the documented bound, not a shrunk range.
        assert float(drawn.max()) > 0.45
        assert float(drawn.min()) < 0.05

    # One additive shift per channel per sample, clamped into [0, 1] by shift_rgb; the red shift is replaced
    # by +1.0 so the clamp is exercised without depending on the draw.
    def test_convention_random_rgb_shift_is_per_channel_per_sample(self, device, dtype):
        image = torch.full((2, 3, 6, 8), 0.5, device=device, dtype=dtype)
        aug = K.RandomRGBShift(r_shift_limit=1.0, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        params = aug.forward_parameters(image.shape)
        for key in ("r_shift", "g_shift", "b_shift"):
            assert params[key].shape == (2,), f"{key} is not one shift per sample"
        # The two samples draw independently, and the zero-limit channels stay at exactly 0.
        assert float(params["r_shift"][0]) != float(params["r_shift"][1])
        self.assert_close(params["g_shift"], torch.zeros_like(params["g_shift"]), atol=0, rtol=0)
        self.assert_close(params["b_shift"], torch.zeros_like(params["b_shift"]), atol=0, rtol=0)
        params["r_shift"] = torch.ones_like(params["r_shift"])
        out = aug(image, params=params)
        self.assert_close(out[:, 0], torch.ones_like(out[:, 0]))
        self.assert_close(out[:, 1:], image[:, 1:])

    # A limit is the half-width of a symmetric range; a limit of 0 adds nothing but the clamp still applies.
    def test_convention_random_rgb_shift_limit_is_a_half_width(self, device, dtype):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomRGBShift(r_shift_limit=0.5, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)
        shifts = aug.forward_parameters((512, 3, 2, 2))["r_shift"]
        assert float(shifts.min()) < -0.4
        assert float(shifts.max()) > 0.4
        assert float(shifts.abs().max()) <= 0.5
        image = torch.tensor([1.5, -0.25], device=device, dtype=dtype).reshape(1, 1, 1, 2).expand(1, 3, 1, 2)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomRGBShift(r_shift_limit=0.0, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)(image.contiguous())
        self.assert_close(out, image.clamp(0.0, 1.0))

    def test_convention_random_rgb_shift_clamps_before_shifting_4430(self, device, dtype):
        image = torch.full((2, 3, 6, 8), -0.5, device=device, dtype=dtype)
        aug = K.RandomRGBShift((0.25, 0.25), (0.25, 0.25), (0.25, 0.25), p=1.0)
        self.assert_close(aug(image), torch.full_like(image, 0.25))

    # A (low, high) limit is the sampling range itself, as in albumentations' RGBShift (#4782); a number stays
    # a half-width, and a malformed pair is rejected by name.
    @pytest.mark.device_agnostic
    def test_convention_random_rgb_shift_tuple_limit_is_a_range_4782(self):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomRGBShift((0.1, 0.2), [-0.3, -0.1], 0.05, p=1.0)
        aug(torch.rand(64, 3, 2, 2))
        for name, (low, high) in {"r_shift": (0.1, 0.2), "g_shift": (-0.3, -0.1), "b_shift": (-0.05, 0.05)}.items():
            drawn = aug._params[name]
            assert bool((drawn >= low).all()), name
            assert bool((drawn <= high).all()), name
        # A 1-D tensor is a pair too, and a pair's domain is as unbounded as a number's.
        K.RandomRGBShift(torch.tensor([0.1, 0.2]), (-2.0, 2.0), 2.0)
        with pytest.raises(ValueError, match=r"r_shift\[0\] should be smaller than r_shift\[1\]"):
            K.RandomRGBShift((0.2, 0.1))
        with pytest.raises(ValueError, match="If g_shift is a range, it must be finite"):
            K.RandomRGBShift(g_shift_limit=(0.0, float("inf")))

    # `pl` is a persistent buffer holding the table the mode selects -- 25 rows for blackbody, 23 for CIED --
    # and `select_from` narrows it.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("kwargs", "rows"), [({"mode": "blackbody"}, 25), ({"mode": "CIED"}, 23), ({"select_from": [0, 1]}, 2)]
    )
    def test_convention_random_planckian_jitter_table_follows_mode(self, kwargs, rows):
        aug = K.RandomPlanckianJitter(p=1.0, **kwargs)
        assert tuple(aug.pl.shape) == (rows, 2)
        assert [name for name, _ in aug.named_buffers()] == ["pl"]

    # Clamp before scaling: even a darkening illuminant must not bring an overshoot back into range.
    def test_convention_random_planckian_jitter_clamps_before_channel_scaling_4430(self, device, dtype):
        image = torch.tensor([[1.5, -0.5], [1.7, -0.5], [1.5, -0.5]], device=device, dtype=dtype).reshape(1, 3, 1, 2)
        aug = K.RandomPlanckianJitter(mode="blackbody", select_from=[0], p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        red_gain, blue_gain = (float(value) for value in aug.pl[0])
        clamped = image.clamp(0.0, 1.0)
        scaled = torch.stack([clamped[:, 0] * red_gain, clamped[:, 1], clamped[:, 2] * blue_gain], dim=1)
        self.assert_close(out, scaled.clamp(0.0, 1.0))
        assert float(out[0, 1, 0, 0]) == 1.0
        assert float(out[0, 2, 0, 0]) < 0.01
        assert bool((out[..., 1] == 0).all())

    # Issue #4574: the coefficient table follows the input dtype, so a half-precision input is not promoted.
    def test_convention_random_planckian_jitter_preserves_dtype_4574(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPlanckianJitter(p=1.0)(image)

        assert out.dtype == dtype
        assert out.device == image.device

        if dtype in _HALF:
            # A module cast to the other half dtype does not decide the output dtype.
            other = torch.bfloat16 if dtype == torch.float16 else torch.float16
            torch.manual_seed(_FORWARD_SEED)
            assert K.RandomPlanckianJitter(p=1.0).to(device=device, dtype=other)(image).dtype == dtype

    # Issue #4574, the wider direction: a module cast wider than the input does not widen the output.
    def test_convention_random_planckian_jitter_wider_module_dtype_4574(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")

        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float32)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPlanckianJitter(p=1.0).to(device=device, dtype=torch.float64)(image)

        assert out.dtype == torch.float32
        assert out.device == image.device

    # The illuminant table is an RGB ratio, so a non-RGB input is rejected rather than broadcast.
    def test_convention_random_planckian_jitter_requires_three_channels(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ShapeError, match=r"expected 3, got 1"):
            _sync(K.RandomPlanckianJitter(p=1.0)(image).device)

    # Issue #4428: `pl`'s shape depends on `mode`, so a blackbody state_dict does not load into a CIED
    # instance.
    @pytest.mark.device_agnostic
    def test_wart_random_planckian_jitter_state_dict_is_mode_specific_4428(self):
        blackbody = K.RandomPlanckianJitter(mode="blackbody")
        cied = K.RandomPlanckianJitter(mode="CIED")
        assert "pl" in blackbody.state_dict()
        with pytest.raises(RuntimeError):
            cied.load_state_dict(blackbody.state_dict())
        # The same-mode round trip is fine, so the defect is the shape and not the buffer.
        K.RandomPlanckianJitter(mode="blackbody").load_state_dict(blackbody.state_dict())

    # RandomAutoContrast is kornia.enhance.normalize_min_max, a per-sample, per-channel rescale; six channels
    # with six distinct ranges rule out a per-batch or per-image rescale.
    def test_convention_random_auto_contrast_is_normalize_min_max(self, device, dtype):
        torch.manual_seed(5)
        scale = torch.tensor([0.2, 0.5, 1.0]).reshape(1, 3, 1, 1)
        offset = torch.tensor([0.1, 0.3]).reshape(2, 1, 1, 1)
        image = (torch.rand(2, 3, 6, 8) * scale + offset).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomAutoContrast(p=1.0)(image)
        self.assert_close(out, normalize_min_max(image))
        for b in range(2):
            for c in range(3):
                self.assert_close(out[b, c].min(), out.new_tensor(0.0))
                self.assert_close(out[b, c].max(), out.new_tensor(1.0))
        # The documented `+ 1e-6` in the denominator: a range of 1e-5 peaks at 1e-5 / 1.1e-5.
        if dtype in (torch.float32, torch.float64):
            narrow = torch.full((1, 1, 2, 2), 0.2, device=device, dtype=dtype)
            narrow[0, 0, 0, 0] += 1e-5
            torch.manual_seed(_FORWARD_SEED)
            peak = K.RandomAutoContrast(p=1.0)(narrow).max()
            assert abs(float(peak) - 1e-5 / 1.1e-5) < 1e-3

    # RandomJPEG's decode hard-clamps into [0, 1], so an input far outside it comes back inside, reaching both
    # ends.
    def test_convention_random_jpeg_output_is_clamped_to_unit_range(self, device, dtype):
        if not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        image = (torch.rand(2, 3, 32, 32) * 3.0 - 1.0).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomJPEG(p=1.0)(image)
        assert out.dtype == dtype
        assert not bool(out.isnan().any())
        assert float(out.min()) == 0.0
        assert float(out.max()) == 1.0

    @pytest.mark.parametrize("negative", [True, False])
    def test_convention_random_jpeg_clamps_before_codec_4430(self, device, dtype, negative):
        if not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        field = (torch.rand(1, 1, 32, 32) * 2.0 + 0.03125).expand(1, 3, -1, -1).contiguous()
        image = (-field if negative else 1.0 + field).to(device=device, dtype=dtype)
        aug = K.RandomJPEG(jpeg_quality=(50.0, 50.0), p=1.0)
        out = aug(image)
        assert torch.equal(out, aug(image.clamp(0.0, 1.0), params=aug._params))
        assert bool(torch.isfinite(out).all())

    # The documented parameter bounds are enforced, and `stage` records where: at construction, or on the
    # forward pass for the checks that need the image.  `match` pins kornia's message, so a case cannot be
    # satisfied by a different error of the same class; `bits=9` is the example -- a scalar becomes [9, 8]
    # and trips the "lo > hi" arm, so the (0, 8] bound needs the explicit range row.
    @pytest.mark.parametrize(
        ("case", "stage", "error", "match"),
        [
            (
                "sharpness_negative",
                "construction",
                ValueError,
                r"If sharpness is a single number, it must be non negative",
            ),
            ("posterize_bits_above_eight", "construction", ValueError, r"bits\[0\] should be smaller than bits\[1\]"),
            (
                "posterize_bits_range_above_eight",
                "construction",
                ValueError,
                r"bits out of bounds\. Expected inside \(0, 8\)",
            ),
            ("posterize_bits_negative", "construction", ValueError, r"bits out of bounds\. Expected inside \(0, 8\)"),
            ("gamma_negative", "forward", RuntimeError, r"Gamma must be non-negative"),
            ("gain_negative", "forward", RuntimeError, r"Gain must be non-negative"),
            (
                "brightness_above_two",
                "construction",
                ValueError,
                r"brightness out of bounds\. Expected inside \(0\.0, 2\.0\)",
            ),
            ("contrast_negative", "construction", ValueError, r"contrast out of bounds\. Expected inside \(0, inf\)"),
            (
                "saturation_negative",
                "construction",
                ValueError,
                r"saturation out of bounds\. Expected inside \(0, inf\)",
            ),
            ("hue_above_half", "construction", ValueError, r"hue out of bounds\. Expected inside \(-0\.5, 0\.5\)"),
            ("gaussian_blur_sigma_zero", "forward", BaseError, r"sigma must be positive"),
            ("gaussian_blur_even_kernel", "forward", BaseError, r"Kernel size must be an odd integer bigger than 0"),
            ("rain_drop_height_zero", "forward", BaseError, r"Height of drop should be greater than zero"),
            (
                "motion_blur_kernel_range_draws_one",
                "forward",
                BaseError,
                r"Kernel size must be an odd integer bigger than 2",
            ),
            (
                "motion_blur_even_bound_rounds_up",
                "forward",
                BaseError,
                r"Kernel size must be an odd integer bigger than 2",
            ),
            ("channel_dropout_more_than_the_channels", "forward", BaseError, r"Invalid value in .num_drop_channels."),
            (
                "channel_dropout_fill_value_above_one",
                "construction",
                BaseError,
                r"Invalid value in .fill_value.\. Must be a number between 0 and 1",
            ),
            (
                "color_jiggle_brightness_range_above_two",
                "construction",
                ValueError,
                r"brightness out of bounds\. Expected inside \(0, 2\)",
            ),
            # torch's indexing error, so only the type is pinned.
            ("planckian_select_past_the_table", "construction", IndexError, None),
            # The scalar rows name the unclamped `[center - x, center + x]`, which only the scalar check
            # reports; the range check behind it reports the floored range, so a bound-only regex passes either way.
            (
                "hue_scalar_above_half",
                "construction",
                ValueError,
                r"hue out of bounds\. Expected inside \(-0\.5, 0\.5\), got tensor\(\[-0\.60*,\s*0\.60*\]\)",
            ),
            (
                "brightness_scalar_above_two",
                "construction",
                ValueError,
                r"brightness out of bounds\. Expected inside \(0\.0, 2\.0\), got tensor\(\[-2\.0*,\s*4\.0*\]\)",
            ),
            (
                "solarize_scalar_threshold_above_half",
                "construction",
                ValueError,
                r"thresholds out of bounds\. Expected inside \(0\.0, 1\.0\), got tensor\(\[-1\.50*,\s*2\.50*\]\)",
            ),
        ],
    )
    def test_convention_intensity_constructors_reject_out_of_bounds(self, device, dtype, case, stage, error, match):
        if case in ("gamma_negative", "gain_negative") and device.type != "cpu":
            pytest.skip("CPU only: on CUDA the value assert is a device-side assert; kornia skips it on MPS")
        factories = {
            "sharpness_negative": lambda: K.RandomSharpness(-1.0, p=1.0),
            "posterize_bits_above_eight": lambda: K.RandomPosterize(bits=9, p=1.0),
            "posterize_bits_negative": lambda: K.RandomPosterize(bits=-1, p=1.0),
            "gamma_negative": lambda: K.RandomGamma((-1.0, -1.0), (1.0, 1.0), p=1.0),
            # `gain` has the identical forward-time check and no construction bound of its own.
            "gain_negative": lambda: K.RandomGamma((1.0, 1.0), (-1.0, -1.0), p=1.0),
            "brightness_above_two": lambda: K.RandomBrightness((3.0, 3.0), p=1.0),
            "contrast_negative": lambda: K.RandomContrast((-1.0, -1.0), p=1.0),
            "saturation_negative": lambda: K.RandomSaturation((-1.0, -1.0), p=1.0),
            "hue_above_half": lambda: K.RandomHue((0.6, 0.6), p=1.0),
            "hue_scalar_above_half": lambda: K.RandomHue(0.6, p=1.0),
            "brightness_scalar_above_two": lambda: K.RandomBrightness(3.0, p=1.0),
            "solarize_scalar_threshold_above_half": lambda: K.RandomSolarize(2.0, 0.1, p=1.0),
            # The constructor admits sigma 0; gaussian_blur2d rejects it on the forward pass.
            "gaussian_blur_sigma_zero": lambda: K.RandomGaussianBlur((3, 3), (0.0, 0.0), p=1.0),
            # The documented "greater than zero" is checked against the image on the forward pass.
            "rain_drop_height_zero": lambda: K.RandomRain(
                number_of_drops=(1, 1), drop_height=(0, 0), drop_width=(1, 1), p=1.0
            ),
            # The constructor admits an even entry; gaussian_blur2d rejects it on the forward pass.
            "gaussian_blur_even_kernel": lambda: K.RandomGaussianBlur((4, 4), (1.0, 1.0), p=1.0),
            # Only odd sizes are drawn, so (1, 1) draws 1, which the motion kernel rejects.
            "motion_blur_kernel_range_draws_one": lambda: K.RandomMotionBlur((1, 1), 35.0, 0.5, p=1.0),
            # The bound is the input's channel count, which only the forward pass knows.
            "channel_dropout_more_than_the_channels": lambda: K.RandomChannelDropout(num_drop_channels=4, p=1.0),
            # `select_from` indexes the 25-row blackbody table at construction.
            "planckian_select_past_the_table": lambda: K.RandomPlanckianJitter(select_from=[25], p=1.0),
            # A scalar `bits=9` becomes the range [9, 8] and trips the "lo > hi" arm, so the documented
            # (0, 8] upper bound needs an explicit range to be pinned at all.
            "posterize_bits_range_above_eight": lambda: K.RandomPosterize(bits=(3.0, 9.0), p=1.0),
            # An even bound is rounded up to the next odd size, so (0, 2) draws 1 and the kernel rejects it.
            "motion_blur_even_bound_rounds_up": lambda: K.RandomMotionBlur((0, 2), 35.0, 0.5, p=1.0),
            "channel_dropout_fill_value_above_one": lambda: K.RandomChannelDropout(fill_value=2.0, p=1.0),
            # ColorJiggle's brightness bound is (0, 2); ColorJitter accepts the same range.
            "color_jiggle_brightness_range_above_two": lambda: K.ColorJiggle(brightness=(0.0, 3.0), p=1.0),
        }
        if stage == "construction":
            with pytest.raises(error, match=match):
                factories[case]()
            return
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = factories[case]()
        with pytest.raises(error, match=match):
            _sync(aug(image).device)

    # Issue #4605: RandomSolarize admits the closed [-0.5, 0.5] for `additions`, and an endpoint is applied.
    @pytest.mark.parametrize("addition", [0.5, -0.5])
    def test_convention_random_solarize_applies_an_additions_endpoint_4605(self, device, dtype, addition):
        if device.type == "cuda":
            pytest.skip("not on CUDA: value asserts are kept out of the shared CUDA process")
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        aug = K.RandomSolarize((0.5, 0.5), (addition, addition), p=1.0)
        out = aug(image)
        self.assert_close(out, solarize(image, 0.5, addition))

    # `select_from` indexes the table like a Python sequence, so a negative entry counts from the end.
    @pytest.mark.device_agnostic
    def test_convention_random_planckian_jitter_select_from_is_a_python_index(self):
        assert torch.equal(
            K.RandomPlanckianJitter(select_from=[-1], p=1.0).pl, K.RandomPlanckianJitter(select_from=[24], p=1.0).pl
        )

    # A scalar magnitude is `center +- x`, floored at the parameter's lower bound and rejected past its upper
    # bound.  Each regex names the unclamped range, which only the scalar check reports.
    @pytest.mark.device_agnostic
    def test_convention_scalar_magnitude_floors_low_and_rejects_high(self):
        for ctor, message in (
            (lambda: K.RandomHue(0.7, p=1.0), r"hue out of bounds\. .*got tensor\(\[-0\.70*,\s*0\.70*\]\)"),
            (
                lambda: K.RandomBrightness(3.0, p=1.0),
                r"brightness out of bounds\. .*got tensor\(\[-2\.0*,\s*4\.0*\]\)",
            ),
            (
                lambda: K.ColorJiggle(brightness=1.5, p=1.0),
                r"brightness out of bounds\. .*got tensor\(\[-0\.50*,\s*2\.50*\]\)",
            ),
            (
                lambda: K.RandomSolarize(2.0, 0.1, p=1.0),
                r"thresholds out of bounds\. .*got tensor\(\[-1\.50*,\s*2\.50*\]\)",
            ),
            (
                lambda: K.RandomMotionBlur(3, 45.0, 2.0, p=1.0),
                r"direction out of bounds\. .*got tensor\(\[-2\.0*,\s*2\.0*\]\)",
            ),
            (
                lambda: K.RandomJPEG(90.0, p=1.0),
                r"jpeg_quality out of bounds\. .*got tensor\(\[-40\.0*,\s*140\.0*\]\)",
            ),
        ):
            with pytest.raises(ValueError, match=message):
                ctor()
        assert K.RandomContrast(1.5, p=1.0).contrast.tolist() == [0.0, 2.5]
        assert K.RandomBrightness(0.5, p=1.0).brightness.tolist() == [0.5, 1.5]
        torch.manual_seed(_FORWARD_SEED)
        sharpness = K.RandomSharpness(0.5, p=1.0).forward_parameters((2000, 1, 4, 4))["sharpness"]
        assert float(sharpness.min()) >= 0.0
        assert float(sharpness.max()) <= 0.5
        assert float(sharpness.min()) < 0.1  # the lower end is the floored -x, not a centred x / 2
        jpeg = K.RandomJPEG(50.0, p=1.0)._param_generator
        assert (float(jpeg.jpeg_quality_sampler.low), float(jpeg.jpeg_quality_sampler.high)) == (1.0, 100.0)

    def test_convention_random_clahe_clamps_before_histogram_4430(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = (torch.rand(1, 3, 16, 16) * 2).to(device=device, dtype=dtype)
        aug = K.RandomClahe(p=1.0)
        out = aug(image)
        assert torch.equal(out, aug(image.clamp(0.0, 1.0), params=aug._params))
        if device.type != "cuda":
            with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]"):
                equalize_clahe(image)

    @pytest.mark.parametrize("value", [1.0, 0.0, 1.0 + 0.5 / 255, 1.0 + 1.5 / 255, -0.5 / 255, -1.5 / 255])
    def test_convention_random_clahe_clamps_small_overshoots_4430(self, device, dtype, value):
        image = torch.full((1, 1, 16, 16), 0.5, device=device, dtype=dtype)
        image[0, 0, 0, 0] = value
        aug = K.RandomClahe(p=1.0)
        out = aug(image)
        assert bool(torch.isfinite(out).all())
        assert torch.equal(out, aug(image.clamp(0.0, 1.0), params=aug._params))

    # Issue #2531: each axis of `grid_size` is tiled on its own, with tiles of ceil(size / grid) pixels, odd or
    # even (#5812), so a grid works whether or not it divides the image, and only an axis the grid does not
    # divide is padded.  The padding is reflect at the end of each axis, so the pre-padded image cropped back
    # reproduces the output bitwise.
    @pytest.mark.parametrize(
        ("grid_size", "size", "pad"),
        [
            ((3, 3), 10, (2, 2)),
            ((4, 4), 18, (2, 2)),
            ((4, 4), 20, (0, 0)),
            ((5, 5), 20, (0, 0)),
            ((4, 5), 18, (2, 2)),
            ((3, 5), 30, (0, 0)),
            ((1, 2), 11, (0, 1)),
        ],
    )
    def test_convention_random_clahe_grid_pads_each_axis_to_whole_tiles_2531(self, device, dtype, grid_size, size, pad):
        pad_rows, pad_cols = pad
        if (pad_rows or pad_cols) and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        image = torch.linspace(0, 1, size * size, device=device, dtype=dtype).reshape(1, 1, size, size)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomClahe(grid_size=grid_size, p=1.0)(image)
        assert out.shape == image.shape
        assert bool(out.isfinite().all())
        assert not torch.equal(out, image)
        if pad_rows or pad_cols:
            padded = torch.nn.functional.pad(image, [0, pad_cols, 0, pad_rows], mode="reflect")
            torch.manual_seed(_FORWARD_SEED)
            reference = K.RandomClahe(grid_size=grid_size, p=1.0)(padded)[..., :size, :size]
            assert torch.equal(out, reference)

    # At the default grid (8, 8) an image smaller than the grid gets kornia's named ValueError, and 9 x 9 is
    # admitted.  The 8 x 8 case in between is covered by the #4783 pin below.
    def test_convention_random_clahe_default_grid_rejects_a_smaller_image(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match=r"Cannot compute tiles on the image according to the given grid size"):
            K.RandomClahe(p=1.0)(torch.rand(1, 1, 7, 7, device=device, dtype=dtype))
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomClahe(p=1.0)(torch.rand(1, 1, 9, 9, device=device, dtype=dtype)).shape == (1, 1, 9, 9)

    # Issue #4783: an image exactly as large as the grid used to pass the named size check and fail in the
    # reflect padding with torch's raw error.  It now gets kornia's ValueError, naming the smallest image
    # the grid admits.
    def test_convention_random_clahe_grid_sized_image_is_rejected_by_name_4783(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match=r"Cannot compute tiles") as info:
            _sync(K.RandomClahe(p=1.0)(torch.rand(1, 1, 8, 8, device=device, dtype=dtype)).device)
        # The error names the argument the caller passed, unlike torch's padding error.
        assert "Cannot compute tiles" in str(info.value)
        assert "smallest image this grid admits is (9, 9)" in str(info.value)

    # Issue #4572: each image is equalized with its own `clip_limit_factor` draw, not the first sample's.
    @pytest.mark.device_agnostic
    def test_convention_random_clahe_applies_each_clip_limit_to_its_own_image_4572(self):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(8, 1, 32, 32) ** 3
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomClahe(clip_limit=(0.5, 40.0), grid_size=(2, 2), p=1.0)
        out = aug(image)
        clip = aug._params["clip_limit_factor"]
        assert float((clip - clip[0]).abs().max()) > 1.0, clip.tolist()
        for row in range(image.shape[0]):
            own = equalize_clahe(image[row : row + 1], float(clip[row]), (2, 2))
            assert torch.equal(out[row : row + 1], own), row
        # ... and not with the first sample's.
        first = int((clip - clip[0]).abs().argmax())
        assert not torch.equal(out[first : first + 1], equalize_clahe(image[first : first + 1], float(clip[0]), (2, 2)))

    # With `same_on_batch=True` every image is equalized with the one shared draw.
    @pytest.mark.device_agnostic
    def test_convention_random_clahe_same_on_batch_shares_one_clip_limit(self):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(8, 1, 32, 32) ** 3
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomClahe(clip_limit=(0.5, 40.0), grid_size=(2, 2), p=1.0, same_on_batch=True)
        out = aug(image)
        clip = aug._params["clip_limit_factor"]
        assert bool((clip == clip[0]).all())
        assert torch.equal(out, equalize_clahe(image, float(clip[0]), (2, 2)))
        for row in range(image.shape[0]):
            assert torch.equal(out[row : row + 1], equalize_clahe(image[row : row + 1], float(clip[0]), (2, 2)))

    # Issue #4560: every class whose path goes through rgb_to_hsv is finite for a pixel whose largest
    # channel is 0, in every dtype.
    @pytest.mark.parametrize(
        "name", ["RandomHue", "RandomSaturation", "ColorJiggle", "ColorJiggleSaturation", "ColorJitter"]
    )
    def test_convention_hsv_path_zero_max_pixel_is_finite_4560(self, device, dtype, name):
        factories = {
            "RandomHue": lambda: K.RandomHue((0.1, 0.1), p=1.0),
            "RandomSaturation": lambda: K.RandomSaturation((1.5, 1.5), p=1.0),
            "ColorJiggle": lambda: K.ColorJiggle(0.0, 0.0, 0.0, (0.1, 0.1), p=1.0),
            # ColorJiggle's saturation step is an HSV round trip too.
            "ColorJiggleSaturation": lambda: K.ColorJiggle(0.0, 0.0, (1.5, 1.5), 0.0, p=1.0),
            "ColorJitter": lambda: K.ColorJitter(0.0, 0.0, 0.0, (0.1, 0.1), p=1.0),
        }
        # A black pixel, and a pixel whose largest channel is 0 without being black.
        pixels = torch.tensor([[0.0, 0.0, 0.0], [0.0, -0.5, -0.5]], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = factories[name]()(pixels.T.reshape(1, 3, 1, 2))
        assert bool(out.isfinite().all()), (name, dtype, out.tolist())

    # An unbatched (C, H, W) input is promoted to (1, C, H, W), and `keepdim=True` returns the unbatched
    # shape, across a colour, a normalization and a weather class.
    @pytest.mark.parametrize("name", ["RandomBrightness", "Normalize", "RandomRain"])
    def test_convention_intensity_promotes_chw_to_bchw(self, device, dtype, name):
        factories = {
            "RandomBrightness": lambda keepdim: K.RandomBrightness((1.5, 1.5), p=1.0, keepdim=keepdim),
            "Normalize": lambda keepdim: K.Normalize(mean=0.5, std=0.25, p=1.0, keepdim=keepdim),
            "RandomRain": lambda keepdim: K.RandomRain(
                number_of_drops=(2, 2), drop_height=(2, 2), drop_width=(1, 1), p=1.0, keepdim=keepdim
            ),
        }
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert factories[name](False)(image).shape == (1, 3, 6, 8)
        torch.manual_seed(_FORWARD_SEED)
        assert factories[name](True)(image).shape == (3, 6, 8)

    # RandomClahe's default fast path leaves the autograd graph; `slow_and_differentiable=True` keeps it.
    @pytest.mark.device_agnostic
    def test_convention_random_clahe_is_differentiable_only_when_asked(self):
        image = (torch.rand(2, 3, 16, 16) * 0.8 + 0.1).requires_grad_()
        torch.manual_seed(_FORWARD_SEED)
        fast = K.RandomClahe(p=1.0)(image)
        assert not fast.requires_grad
        assert fast.grad_fn is None
        torch.manual_seed(_FORWARD_SEED)
        slow = K.RandomClahe(p=1.0, slow_and_differentiable=True)(image)
        assert slow.requires_grad
        slow.sum().backward()
        assert image.grad is not None
        assert float(image.grad.abs().sum()) > 0.0

    # The uint8 round trip is a step function, so the gradient below bits=8 is structurally zero even
    # though requires_grad stays True; bits=8 skips the round trip and is the identity.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(("bits", "expected"), [(0, 0.0), (3, 0.0), (7, 0.0), (8, 1536.0)])
    def test_convention_random_posterize_gradient_is_zero_below_eight_bits(self, bits, expected):
        image = (torch.rand(2, 3, 16, 16) * 0.8 + 0.1).requires_grad_()
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize((bits, bits), p=1.0)(image)
        assert out.requires_grad, "the output keeps requires_grad whatever the gradient is"
        out.sum().backward()
        assert float(image.grad.abs().sum()) == expected

    # A non-integral `bits` is accepted and rounded half to even.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(("bits", "drawn"), [(2.5, 2), (3.5, 4), (0.4, 0), (0.6, 1)])
    def test_convention_random_posterize_rounds_a_float_bits_half_to_even(self, bits, drawn):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomPosterize((bits, bits), p=1.0)
        aug(torch.rand(1, 3, 16, 16))
        assert [int(v) for v in aug._params["bits_factor"].tolist()] == [drawn]

    # ColorJiggle skips a step when every drawn factor is neutral, so the channel count only has to suit the
    # steps that run.
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_convention_color_jiggle_skips_neutral_steps(self, device, dtype, channels):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, channels, 5, 5).to(device=device, dtype=dtype)
        for factors in ((0.0, 0.0, 0.0, 0.0), (0.2, 0.0, 0.0, 0.0), (0.0, 0.2, 0.0, 0.0)):
            torch.manual_seed(_FORWARD_SEED)
            assert K.ColorJiggle(*factors, p=1.0)(image).shape == image.shape
        # The steps that do run still need three channels.
        for factors in ((0.0, 0.0, 0.2, 0.0), (0.0, 0.0, 0.0, 0.1)):
            torch.manual_seed(_FORWARD_SEED)
            if channels == 3:
                assert K.ColorJiggle(*factors, p=1.0)(image).shape == image.shape
            else:
                with pytest.raises(ValueError, match="shape of"):
                    K.ColorJiggle(*factors, p=1.0)(image)

    # Issue #4813: ColorJitter skips a step whose factors are all neutral, so the hue and saturation steps accept
    # a channel count that the configuration never asks them to touch, as ColorJiggle and torchvision do, in the
    # random and the fixed order alike.  A non-neutral hue still needs three channels.
    @pytest.mark.parametrize("order", [None, (0, 1, 2, 3)])
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_color_jitter_skips_neutral_steps_4813(self, device, dtype, channels, order):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, channels, 5, 5).to(device=device, dtype=dtype)
        for factors in ((0.0, 0.0, 0.0, 0.0), (0.2, 0.0, 0.0, 0.0), (0.0, 0.2, 0.0, 0.0)):
            torch.manual_seed(_FORWARD_SEED)
            assert K.ColorJitter(*factors, p=1.0, order=order)(image).shape == image.shape
        if channels != 3:
            with pytest.raises(ValueError, match="shape of"):
                K.ColorJitter(hue=0.1, p=1.0, order=order)(image)

    # The erasing box is clamped from below as well as above: scale=(0, 0) still erases one pixel and a 1x1
    # image is always erased in full.
    @pytest.mark.device_agnostic
    def test_convention_random_erasing_box_is_never_empty(self):
        torch.manual_seed(_FORWARD_SEED)
        degenerate = K.RandomErasing(scale=(0.0, 0.0), ratio=(1.0, 1.0), p=1.0)
        out = degenerate(torch.ones(1, 1, 10, 10))
        assert int((out == 0).sum()) == 1
        assert degenerate._params["heights"].tolist() == [1.0]
        assert degenerate._params["widths"].tolist() == [1.0]
        torch.manual_seed(_FORWARD_SEED)
        single = K.RandomErasing(p=1.0)(torch.ones(1, 1, 1, 1))
        assert float(single.sum()) == 0.0

    # A `ratio` straddling 1 is a 50/50 mixture of [ratio[0], 1] and [1, ratio[1]], not uniform over it.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); p = g((200000, 3, 400, 400)); print((p["heights"] > p["widths"]).float().mean())
    @pytest.mark.device_agnostic
    def test_convention_random_erasing_straddling_ratio_is_a_fair_mixture(self):
        generator = RectangleEraseGenerator(scale=(0.25, 0.25), ratio=(0.3, 3.3))
        generator.set_rng_device_and_dtype(torch.device("cpu"), torch.float32)
        torch.manual_seed(_FORWARD_SEED)
        params = generator((200000, 3, 400, 400))
        tall = float((params["heights"] > params["widths"]).float().mean())
        assert abs(tall - 0.5) < 0.01, "the straddling draw is a fair coin between the two sub-intervals"
        uniform = (3.3 - 1.0) / (3.3 - 0.3)
        assert abs(tall - uniform) > 0.2, "a uniform draw over the interval would be far from a fair coin"
        # The realised box follows the draw only while the [1, H] x [1, W] clamp does not bite.
        for height, width, expected in ((4, 256, 0.0), (256, 4, 1.0)):
            torch.manual_seed(_FORWARD_SEED)
            clamped = generator((20000, 3, height, width))
            realised = float((clamped["heights"] > clamped["widths"]).float().mean())
            assert realised == expected, (height, width, realised)

    # With every step neutral both classes return a copy, on the Python guards (a sampled order) and on the
    # torch.cond path (a fixed order) alike, so writing into the output leaves the input unchanged, as with p=0.
    @pytest.mark.parametrize("cls", [K.ColorJiggle, K.ColorJitter])
    @pytest.mark.parametrize("order", [None, (0, 1, 2, 3)])
    def test_convention_color_neutral_steps_return_a_copy(self, device, dtype, cls, order):
        image = torch.full((2, 3, 2, 2), 0.5, device=device, dtype=dtype)
        original = image.clone()
        output = cls(0.0, 0.0, 0.0, 0.0, p=1.0, order=order)(image)
        self.assert_close(output, original)
        assert output.untyped_storage().data_ptr() != image.untyped_storage().data_ptr()
        output.fill_(-1.0)
        self.assert_close(image, original)


class TestIntensityInputClamping(BaseTester):
    @pytest.mark.parametrize("forward_mode", ["jvp", "dual"])
    @pytest.mark.parametrize("grad_enabled", [False, True])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_input_clamp_preserves_forward_tangents(self, device, dtype, forward_mode, grad_enabled, requires_grad):
        image = torch.tensor([-0.5, -0.0, 0.0, 0.25, 1.0, 1.5], device=device, dtype=dtype)
        image.requires_grad_(requires_grad)
        tangent = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 3.0, 4.0, 5.0, 6.0, 0.0], device=device, dtype=dtype)
        # no_grad disables reverse-mode recording, but forward AD must still pass the tangent at both bounds.
        with torch.set_grad_enabled(grad_enabled):
            if forward_mode == "jvp":
                values, actual = torch.func.jvp(_clamp_unit_range, (image,), (tangent,))
            else:
                with torch.autograd.forward_ad.dual_level():
                    dual = torch.autograd.forward_ad.make_dual(image, tangent)
                    values, actual = torch.autograd.forward_ad.unpack_dual(_clamp_unit_range(dual))
        self.assert_close(values, image.clamp(0.0, 1.0))
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("grad_enabled", [False, True])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_input_clamp_preserves_signed_zero_and_nan(self, device, dtype, grad_enabled, requires_grad):
        image = torch.tensor([-0.0, 0.0, float("nan"), -float("inf"), float("inf")], device=device, dtype=dtype)
        image.requires_grad_(requires_grad)
        with torch.set_grad_enabled(grad_enabled):
            actual = _clamp_unit_range(image)
        assert torch.equal(torch.isnan(actual), torch.isnan(image))
        finite = ~torch.isnan(image)
        self.assert_close(actual[finite], image.clamp(0.0, 1.0)[finite])
        assert torch.equal(torch.signbit(actual[:2]), torch.signbit(image[:2]))

    @pytest.mark.parametrize("name", ["ColorJiggle", "RandomGamma", "RandomPosterize"])
    def test_bicubic_overshoot_is_clamped_before_intensity(self, device, dtype, name):
        torch.manual_seed(0)
        binary = (torch.rand(1, 3, 32, 32) > 0.5).float().to(device)
        # Resize in float32: older CPU torch releases have no half bicubic kernel.
        image = resize(binary, (47, 47), interpolation="bicubic").to(dtype)
        assert bool(((image < 0.0) | (image > 1.0)).any())
        aug = K.RandomGamma((0.5, 0.5), (0.5, 0.5), p=1.0) if name == "RandomGamma" else getattr(K, name)(p=1.0)
        out = aug(image)
        assert bool(torch.isfinite(out).all())
        assert torch.equal(out, aug(image.clamp(0.0, 1.0), params=aug._params))

    @pytest.mark.parametrize("name", _RANGE_DEPENDENT + ("RandomClahe", "RandomJPEG"))
    @pytest.mark.parametrize("gate", [(0.0, 0.0), (1.0, 0.0)])
    def test_clamp_respects_probability_gate_and_replay(self, device, dtype, name, gate):
        if name == "RandomJPEG" and not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        image = torch.linspace(-0.5, 1.5, 2 * 3 * 16 * 16, device=device, dtype=dtype).reshape(2, 3, 16, 16)
        aug = _INTENSITY_FACTORIES[name]() if name in _INTENSITY_FACTORIES else getattr(K, name)(p=1.0)
        params = aug.forward_parameters(image.shape)
        expected_applied = aug(image.clamp(0.0, 1.0), params=params)
        aug.p = 0.5
        params["batch_prob"] = torch.tensor(gate, device=params["batch_prob"].device)
        out = aug(image, params=params)
        expected = torch.where(image.new_tensor(gate).bool().view(2, 1, 1, 1), expected_applied, image)
        assert torch.equal(out, expected)
        assert torch.equal(out, aug(image, params=aug._params))
        assert out.dtype == dtype
        assert out.device == device

    # Valid images often hold exact 0 and 1, and the clamps keep an identity gradient there on every torch
    # version: torch.clamp's own derivative at its bounds is 1 on torch 2.5.1 and 2.9.1 but 0 on 2.14.
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: K.ColorJiggle(0.0, 0.0, 0.0, 0.0, p=1.0),
            lambda: K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0),
            lambda: K.RandomPosterize(bits=(8.0, 8.0), p=1.0),
        ],
        ids=["ColorJiggle", "ColorJitter", "RandomPosterize"],
    )
    def test_clamp_keeps_gradient_at_unit_range_bounds_4430(self, device, dtype, factory):
        image = torch.tensor([0.0, 0.25, 0.75, 1.0], device=device, dtype=dtype).reshape(1, 1, 2, 2)
        image = image.expand(1, 3, 2, 2).contiguous().requires_grad_()
        out = factory()(image)
        self.assert_close(out, image.detach())
        out.sum().backward()
        self.assert_close(image.grad, torch.ones_like(image))

    # Without a recorded gradient the input clamp is a single torch.clamp; with one it selects the input where the
    # clamp left it unchanged. Both must return the same values, and the recorded gradient is 1 inside [0, 1]
    # and 0 outside.
    @pytest.mark.parametrize(
        "factory",
        [lambda: K.RandomBrightness(brightness=(1.0, 1.0), p=1.0), lambda: K.ColorJiggle(0.0, 0.0, 0.0, 0.0, p=1.0)],
        ids=["RandomBrightness", "ColorJiggle"],
    )
    def test_unit_range_input_clamp_matches_with_and_without_a_gradient(self, device, dtype, factory):
        values = torch.tensor([-0.5, -0.0, 0.0, 0.25, 0.75, 1.0, 1.5, 2.0], device=device, dtype=dtype)
        image = values.reshape(1, 1, 2, 4).expand(1, 3, 2, 4).contiguous()
        expected = image.clamp(0.0, 1.0)
        with torch.no_grad():
            assert torch.equal(factory()(image), expected)
        assert torch.equal(factory()(image), expected)
        leaf = image.clone().requires_grad_()
        out = factory()(leaf)
        assert torch.equal(out.detach(), expected)
        out.sum().backward()
        interior = (image > 0.0) & (image < 1.0)
        outside = (image < 0.0) | (image > 1.0)
        assert torch.equal(leaf.grad[interior], torch.ones_like(leaf.grad[interior]))
        assert torch.equal(leaf.grad[outside], torch.zeros_like(leaf.grad[outside]))

    @pytest.mark.parametrize("gamma", [0.5, 1.5])
    def test_gamma_gradient_outside_range_is_zero(self, device, dtype, gamma):
        image = torch.tensor([-0.5, 0.25, 0.5, 1.5], device=device, dtype=dtype).reshape(1, 1, 1, 4)
        image.requires_grad_()
        out = K.RandomGamma((gamma, gamma), p=1.0)(image)
        self.assert_close(out, image.detach().clamp(0.0, 1.0).pow(gamma))
        out.sum().backward()
        expected_grad = gamma * image.detach()[..., 1:3].pow(gamma - 1)
        self.assert_close(image.grad[..., 1:3], expected_grad)
        assert torch.equal(image.grad[..., [0, 3]], torch.zeros_like(image.grad[..., [0, 3]]))

    @pytest.mark.parametrize("name", ["ColorJiggle", "ColorJitter", "RandomGamma", "RandomPosterize"])
    def test_keepdim_and_input_storage(self, device, dtype, name):
        image = torch.linspace(-0.5, 1.5, 3 * 8 * 8, device=device, dtype=dtype).reshape(3, 8, 8)
        original = image.clone()
        aug = getattr(K, name)(p=1.0, keepdim=True)
        out = aug(image)
        assert out.shape == image.shape
        assert torch.equal(image, original)
        assert torch.equal(out, aug(image.clamp(0.0, 1.0), params=aug._params))

    @pytest.mark.parametrize("name", ["RandomGamma", "RandomPosterize", "RandomPlanckianJitter", "ColorJiggle"])
    def test_dynamo_input_clamp(self, device, dtype, name, torch_optimizer):
        image = torch.linspace(-0.5, 1.5, 3 * 8 * 8, device=device, dtype=dtype).reshape(1, 3, 8, 8)
        if name == "ColorJiggle":
            aug = K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0, order=(0, 1, 2, 3))
        else:
            aug = getattr(K, name)(p=1.0)
        params = aug.forward_parameters(image.shape)
        expected = aug(image, params=params)
        optimized = torch_optimizer(aug, fullgraph=True)
        self.assert_close(optimized(image, params=params), expected)

    def test_dissolving_clamps_at_model_boundary(self, device, dtype, monkeypatch):
        class StubDissolving(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.seen = None

            def forward(self, image, step):
                self.seen = image
                return image * 2.0 - 0.5

        monkeypatch.setattr("kornia.augmentation._2d.intensity.dissolving.StableDiffusionDissolving", StubDissolving)
        image = torch.tensor([-0.5, 0.25, 0.75, 1.5], device=device, dtype=dtype).reshape(1, 1, 2, 2)
        aug = K.RandomDissolving(p=1.0)
        out = aug(image)
        assert torch.equal(aug._dslv.seen, image.clamp(0.0, 1.0))
        assert torch.equal(out, (image.clamp(0.0, 1.0) * 2 - 0.5).clamp(0.0, 1.0))
