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
from kornia.augmentation.random_generator import RectangleEraseGenerator
from kornia.core.exceptions import BaseError, ShapeError
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    equalize_clahe,
    normalize_min_max,
    posterize,
    solarize,
)

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


@pytest.fixture(autouse=True)
def _restore_global_rng(restore_torch_rng):
    # Every pin below seeds the global RNG so its draw is reproducible.  ``torch.manual_seed`` also
    # reseeds the CUDA and MPS generators, so the root fixture (#4446) snapshots and restores all of
    # them; ``fork_rng(devices=[])`` would restore the CPU generator only and shift the draw of an
    # unseeded later test on an accelerator leg.
    yield


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
    # A range, not the scalar `bits=3` (the lower bound of [3, 8]): a draw of 8 skips the uint8 round
    # trip and keeps the fixture's range, as test_wart_random_posterize_eight_bits_skips_the_round_trip_4430
    # pins.
    "RandomPosterize": lambda: K.RandomPosterize(bits=(3.0, 7.0), p=1.0),
    "RandomRGBShift": lambda: K.RandomRGBShift(p=1.0),
    "RandomRain": lambda: K.RandomRain(number_of_drops=(3, 3), drop_height=(1, 2), drop_width=(1, 2), p=1.0),
    "RandomSaltAndPepperNoise": lambda: K.RandomSaltAndPepperNoise(p=1.0),
    "RandomSaturation": lambda: K.RandomSaturation((2.0, 2.0), p=1.0),
    "RandomSharpness": lambda: K.RandomSharpness(1.0, p=1.0),
    "RandomSnow": lambda: K.RandomSnow(p=1.0),
    "RandomSolarize": lambda: K.RandomSolarize(0.1, 0.1, p=1.0),
}

# The four observed outcomes for these constructor arguments and fixtures (#4430).  They are not
# class-wide policies: a blur can attenuate an out-of-range impulse into [0, 1], and a hue-only
# ColorJiggle keeps an out-of-range maximum.
_BOUNDED_ON_FIXTURES = (
    "ColorJiggle",
    "ColorJitter",
    "RandomAutoContrast",
    "RandomBrightness",
    "RandomContrast",
    "RandomGamma",
    "RandomGaussianIllumination",
    "RandomLinearCornerIllumination",
    "RandomLinearIllumination",
    "RandomPlasmaBrightness",
    "RandomPlasmaContrast",
    "RandomPlasmaShadow",
    "RandomPosterize",
    "RandomRGBShift",
    "RandomSharpness",
    "RandomSolarize",
)
# ``output.clamp(max=1.0)`` only, so a negative input stays negative.
_UPPER_BOUNDED_ON_FIXTURES = ("RandomPlanckianJitter",)
_OUT_OF_RANGE_ON_FIXTURES = (
    "Denormalize",
    "Normalize",
    "RandomBoxBlur",
    "RandomChannelDropout",
    "RandomChannelShuffle",
    "RandomErasing",
    "RandomGaussianBlur",
    "RandomGaussianNoise",
    "RandomGrayscale",
    "RandomHue",
    "RandomInvert",
    "RandomMedianBlur",
    "RandomMotionBlur",
    "RandomRain",
    "RandomSaltAndPepperNoise",
    "RandomSaturation",
    "RandomSnow",
)
# The only factory that rejects these fixtures.
_REJECTS_OUT_OF_RANGE_FIXTURES = ("RandomEqualize",)

# These factories return zeros for a constant -1 image on every draw (#4430): no admissible draw lifts it
# above zero.  The plasma brightness and contrast maps are unbounded draws, so those two are pinned by replay
# instead.  RandomPosterize is not here: its uint8 conversion of a negative float is platform-dependent
# (test_wart_random_posterize_out_of_range_wraps_4430).
_COLLAPSES_ON_CONSTANT_MINUS_ONE = {
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
    # Issue #4430: the four outcomes above, per factory.  `clip_output=False` variants of RandomBrightness
    # and RandomContrast, which pass the range through, are pinned separately below.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); base = torch.rand(2, 3, 6, 8)
    #   for x in (base * 2.0, base - 1.0): torch.manual_seed(0); print(name, factory()(x).aminmax())
    @pytest.mark.parametrize("name", sorted(_INTENSITY_FACTORIES))
    def test_convention_value_range_on_out_of_range_fixtures(self, device, dtype, name):
        groups = (
            _BOUNDED_ON_FIXTURES,
            _UPPER_BOUNDED_ON_FIXTURES,
            _OUT_OF_RANGE_ON_FIXTURES,
            _REJECTS_OUT_OF_RANGE_FIXTURES,
        )
        assert set().union(*groups) == set(_INTENSITY_FACTORIES)
        if name in _REJECTS_OUT_OF_RANGE_FIXTURES and device.type == "cuda":
            pytest.skip("CUDA: the value assert is a device-side assert that invalidates the context")
        if name in ("RandomBoxBlur", "RandomGaussianBlur") and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        fixtures = _out_of_range_fixtures(device, dtype)
        if name in _REJECTS_OUT_OF_RANGE_FIXTURES:
            # kornia's own value check, which names the range.
            for image in fixtures.values():
                with pytest.raises(RuntimeError, match=r"\[0, 1\]"):
                    _run(name, image)
            return
        # A clamped output is exactly 0.0 or 1.0 and a pass-through one sits near 2.0 or -1.0, so this
        # tolerance still separates the groups in bfloat16.
        tol = 1e-3
        outputs = {tag: _run(name, image) for tag, image in fixtures.items()}
        ranges = {tag: out.aminmax() for tag, out in outputs.items()}
        # The fixtures are already out of range, so the pass-through branch would accept the identity:
        # require the class to change a fixture on some draw (a single draw can be neutral).
        moved = False
        for seed in range(8):
            for image in fixtures.values():
                if not torch.equal(_run(name, image, seed=seed), image):
                    moved = True
                    break
            if moved:
                break
        assert moved, f"{name} returned the out-of-range fixtures unchanged on every seed tried"
        if name in _BOUNDED_ON_FIXTURES:
            for tag, (low, high) in ranges.items():
                assert float(low) >= -tol, f"{name} on {tag} left the lower end unclamped"
                assert float(high) <= 1 + tol, f"{name} on {tag} left the upper end unclamped"
            # Reach: the bound is [0, 1], not a narrower interval.  The exceptions are draws that end
            # inside it: ColorJiggle's hue-only draws, a brightness factor below 1, RandomPosterize's top
            # level 252/255 and RandomSolarize's inversion.
            if name != "ColorJiggle":
                assert float(ranges["[-1, 0]"][0]) <= tol, f"{name} on [-1, 0] does not reach 0"
            if name not in ("ColorJiggle", "ColorJitter", "RandomPosterize", "RandomSolarize"):
                assert float(ranges["[0, 2]"][1]) >= 1 - tol, f"{name} on [0, 2] does not reach 1"
        elif name in _UPPER_BOUNDED_ON_FIXTURES:
            for tag, (_, high) in ranges.items():
                assert float(high) <= 1 + tol, f"{name} on {tag} left the upper end unclamped"
            # The lower end is checked on the fixture that has one: [0, 2] is non-negative anyway.
            negative_min = float(ranges["[-1, 0]"][0])
            assert negative_min < -tol, f"{name} also clamped the lower end (min {negative_min})"
        else:
            assert name in _OUT_OF_RANGE_ON_FIXTURES
            assert any(float(high) > 1 + tol or float(low) < -tol for low, high in ranges.values()), (
                f"{name} kept the output inside [0, 1] on both out-of-range fixtures"
            )

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

    # Issue #4430: these factories silently return an all-zero image for an all-negative input.  The pin
    # asserts the silence as well as the zeros, so a fix that rejects or warns flips it; a fix that instead
    # rules the clamp to be the policy renames it to a convention pin.
    # Snippet used to generate expected:
    #   x = torch.full((2, 3, 6, 8), -1.0); torch.manual_seed(seed); print(factory()(x).abs().max())
    @pytest.mark.parametrize("name", sorted(_COLLAPSES_ON_CONSTANT_MINUS_ONE))
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_wart_intensity_negative_input_collapses_to_zero_4430(self, device, dtype, name, seed):
        image = torch.full((2, 3, 6, 8), -1.0, device=device, dtype=dtype)
        torch.manual_seed(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _COLLAPSES_ON_CONSTANT_MINUS_ONE[name]()(image)
            _sync(image.device)
        assert float(out.abs().max()) == 0.0
        assert not caught

    # Issue #4430, the plasma classes: whether a constant -1 image collapses depends on the drawn map.
    # RandomPlasmaContrast, ``(x - 0.5) * 4 * plasma + 0.5``, zeroes it wherever the map is at least 1/12;
    # RandomPlasmaBrightness, ``x + (2 * plasma - 1) * intensity``, at intensity 1 zeroes it wherever the map
    # is at most 1.  The draw is pinned by replaying ``params`` with a constant map.
    @pytest.mark.parametrize(
        ("name", "level", "collapses"),
        [
            ("RandomPlasmaContrast", 0.5, True),
            ("RandomPlasmaContrast", 0.0, False),
            ("RandomPlasmaBrightness", 0.5, True),
            ("RandomPlasmaBrightness", 1.5, False),
        ],
    )
    def test_wart_random_plasma_negative_collapse_depends_on_the_map_4430(self, device, dtype, name, level, collapses):
        image = torch.full((2, 3, 6, 8), -1.0, device=device, dtype=dtype)
        aug = getattr(K, name)(p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        aug(image)
        params = dict(aug._params)
        params["plasma"] = torch.full_like(params["plasma"], level)
        if "intensity" in params:
            params["intensity"] = torch.ones_like(params["intensity"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = aug(image, params=params)
            _sync(image.device)
        assert (float(out.abs().max()) == 0.0) is collapses
        assert not caught

    # Issue #4430: RandomPosterize posterizes the `uint8` conversion of the raw float, not of the clamped
    # input.  What that conversion does outside [0, 255] is platform-dependent (it wraps or saturates), so
    # the pin asserts the round-trip identity rather than platform literals.
    # Snippet used to generate expected:
    #   g = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32)
    #   torch.manual_seed(0); y = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(g * 2.0); print(len(y.unique()), y.max())
    @pytest.mark.parametrize("scale", ["[0, 2]", "[-1, 0]"])
    def test_wart_random_posterize_out_of_range_wraps_4430(self, device, dtype, scale):
        ramp = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32).to(device=device, dtype=dtype)
        image = ramp * 2.0 if scale == "[0, 2]" else ramp - 1.0
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(image)
        _sync(image.device)
        # `codes` reproduces the one `uint8` conversion inside `posterize` that sees the raw value.
        codes = (image * 255).to(torch.uint8)
        expected = posterize(codes.to(dtype) / 255.0, 3)
        clamped = posterize(image.clamp(0.0, 1.0), 3)
        assert torch.equal(out, expected)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        # A clamp-first fix would return `clamped`; where the running conversion saturates, the two agree.
        if not torch.equal(codes, (image.clamp(0.0, 1.0) * 255).to(torch.uint8)):
            assert not torch.equal(out, clamped)
        if scale == "[0, 2]":
            assert len(out.unique()) == 8
            self.assert_close(out.max(), out.new_tensor(224 / 255))

    # Issue #4430, the other posterize path: a sample that draws 8 bits is returned unchanged, with no
    # `uint8` round trip to bound it, so its out-of-range values survive.
    def test_wart_random_posterize_eight_bits_skips_the_round_trip_4430(self, device, dtype):
        image = torch.tensor([1.7, -0.4, 2.5, 0.3], device=device, dtype=dtype).reshape(1, 1, 1, 4)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(8.0, 8.0), p=1.0)(image)
        _sync(image.device)
        assert torch.equal(out, image)

    # Issue #4430: on a negative input `x ** gamma` is NaN for a non-integer gamma and the clamp keeps it;
    # an integer gamma stays finite, and the default gamma=1 clamps every negative value to 0.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); neg = torch.rand(2, 3, 6, 8) - 1.0
    #   torch.manual_seed(0); print(K.RandomGamma((gamma, gamma), (1.0, 1.0), p=1.0)(neg).isnan().all())
    @pytest.mark.parametrize(("gamma", "expected"), [(0.5, "nan"), (1.0, "zero"), (2.0, "positive")])
    def test_wart_random_gamma_negative_input_depends_on_gamma_4430(self, device, dtype, gamma, expected):
        image = _out_of_range_fixtures(device, dtype)["[-1, 0]"]
        assert float(image.max()) < 0.0, "the fixture must be strictly negative for the power to be NaN"
        torch.manual_seed(_FORWARD_SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = K.RandomGamma((gamma, gamma), (1.0, 1.0), p=1.0)(image)
            _sync(image.device)
        assert not caught
        if expected == "nan":
            assert bool(out.isnan().all()), f"gamma={gamma} on a negative input should be NaN throughout"
        else:
            assert bool(out.isfinite().all()), f"an integer gamma={gamma} should stay finite"
            assert (float(out.abs().max()) > 0.0) is (expected == "positive")

    # RandomEqualize rejects an out-of-range input with kornia's error naming the [0, 1] range.  The guard
    # is `input * 255` inside (-1, 256), so less than one 8-bit code outside the range is admitted.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    #   for x in (g, g * 1.0001, g - 0.003, g * 2, g - 1): torch.manual_seed(0); K.RandomEqualize(p=1.0)(x)
    def test_convention_random_equalize_rejects_out_of_range_with_named_range(self, device, dtype):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        match = r"\[0, 1\]"
        ramp = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).to(device=device, dtype=dtype)
        for image in (ramp * 2.0, ramp - 1.0):
            torch.manual_seed(_FORWARD_SEED)
            with pytest.raises(RuntimeError, match=match):
                _sync(K.RandomEqualize(p=1.0)(image).device)
        for image in (ramp, ramp * 1.0001, ramp - 0.003):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomEqualize(p=1.0)(image)
            assert out.shape == image.shape
            assert torch.isfinite(out).all()

    # Issue #4576: below p=1 the base computes the transform for the whole batch and then selects the
    # rows the gate keeps (`_AugmentationBase.transform_inputs`), so a row the gate skips still reaches
    # the value check and the backward pass.  At p=0.0 RandomEqualize and RandomClahe raise for an image
    # they never touch, and RandomGamma returns the input values but a NaN gradient where the discarded
    # power's derivative is infinite.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 1024).reshape(1, 1, 32, 32); x = torch.cat([g, g * 2])
    #   torch.manual_seed(0); K.RandomEqualize(p=0.0)(x)  # and K.RandomClahe(p=0.0)(x)
    #   v = torch.tensor([0.0, 0.25, 1.0]).reshape(1, 1, 1, 3).repeat(2, 1, 1, 1).requires_grad_()
    #   torch.manual_seed(0); y = K.RandomGamma((0.5, 0.5), (1.0, 1.0), p=0.0)(v); y.sum().backward()
    #   print(torch.equal(y, v), v.grad.flatten().tolist())  # -> True [nan, 1.0, 1.0, nan, 1.0, 1.0]
    def test_wart_p_gate_computes_the_skipped_samples_4576(self, device, dtype):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value asserts are device-side asserts that poison the context")
        ramp = torch.linspace(0, 1, 1024).reshape(1, 1, 32, 32)
        image = torch.cat([ramp, ramp * 2.0]).to(device=device, dtype=dtype)
        for cls in (K.RandomEqualize, K.RandomClahe):
            torch.manual_seed(_FORWARD_SEED)
            # It has to be the value check that fires, not merely some RuntimeError: the claim is that
            # the transform ran on a sample `p=0.0` was supposed to skip.
            with pytest.raises(RuntimeError, match=r"\[0, 1\]"):
                _sync(cls(p=0.0)(image).device)
        values = torch.tensor([0.0, 0.25, 1.0]).reshape(1, 1, 1, 3).repeat(2, 1, 1, 1)
        values = values.to(device=device, dtype=dtype).requires_grad_()
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGamma((0.5, 0.5), (1.0, 1.0), p=0.0)(values)
        assert torch.equal(out, values)
        out.sum().backward()
        assert bool(values.grad[..., 0].isnan().all())
        self.assert_close(values.grad[..., 1:], torch.ones_like(values.grad[..., 1:]))


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

    def test_convention_random_saturation_clamps_hsv_without_a_final_rgb_clamp(self, device, dtype):
        # Clamping the HSV saturation removes a negative RGB channel even at factor 1 and turns an
        # all-negative pixel gray at its largest channel; pixels above 1 with no negative channel keep
        # their values, which a final RGB clamp would remove.
        pixels = [[-0.1, 0.5, 0.5], [1.5, 0.0, 0.0], [-0.1, -0.5, -0.9], [2.0, 1.0, 0.5]]
        image = torch.tensor(pixels, device=device, dtype=dtype).reshape(4, 3, 1, 1)
        out = K.RandomSaturation((1.0, 1.0), p=1.0)(image)
        expected = [[0.0, 0.5, 0.5], [1.5, 0.0, 0.0], [-0.1, -0.1, -0.1], [2.0, 1.0, 0.5]]
        self.assert_close(out, image.new_tensor(expected).reshape_as(image))

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
    def test_convention_color_jiggle_can_leave_output_above_unit_range(self, device, dtype, hue):
        # Disabled brightness/contrast steps do not clamp, and an HSV hue rotation preserves value.
        image = torch.full((2, 3, 6, 8), 2.0, device=device, dtype=dtype)
        out = K.ColorJiggle(0.0, 0.0, 0.0, hue, p=1.0)(image)
        self.assert_close(out, image, atol=0, rtol=0)
        # A chromatic pixel: the rotation keeps its out-of-range maximum.
        chroma = torch.tensor([2.0, 1.5, 0.5], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        torch.manual_seed(_FORWARD_SEED)
        rotated = K.ColorJiggle(0.0, 0.0, 0.0, hue, p=1.0)(chroma)
        self.assert_close(rotated.amax(1), chroma.amax(1))
        assert torch.equal(rotated, chroma) is (hue == 0.0)

    # Both classes are the identity at (0, 0, 0, 0) on an in-range image.
    @pytest.mark.parametrize("cls", [K.ColorJiggle, K.ColorJitter])
    def test_convention_color_jiggle_and_jitter_are_identity_at_zero(self, device, dtype, cls):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(cls(0.0, 0.0, 0.0, 0.0, p=1.0)(image), image)

    # Issue #4785, ColorJitter: the brightness guard tests the factor against 0, not the multiplier's neutral 1,
    # so the factor 1 drawn by the default ``brightness=0.0`` still runs the clamping step.  Run first, it
    # clamps an all-negative input to zero; a contrast or saturation factor above 1 applied before it lifts
    # part of the image first.  A fix that skips a factor of 1 flips the brightness-first leg; removing the guard
    # altogether leaves it green, because the clamp is then #4430's.  A fixed order without index 0 skips the
    # step and its clamp.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); neg = torch.rand(1, 3, 8, 8) - 1.0
    #   print(K.ColorJitter(p=1.0, order=order, contrast=(1.9, 1.9))(neg).aminmax())
    @pytest.mark.parametrize(("step", "kwargs"), [(1, {"contrast": (1.9, 1.9)}), (2, {"saturation": (1.9, 1.9)})])
    def test_wart_color_jitter_default_brightness_step_clamps_4785(self, device, dtype, step, kwargs):
        torch.manual_seed(_FIXTURE_SEED)
        negative = (torch.rand(1, 3, 8, 8) - 1.0).to(device=device, dtype=dtype)
        brightness_first = K.ColorJitter(p=1.0, order=(0, step), **kwargs)(negative)
        assert float(brightness_first.abs().max()) == 0.0
        step_first = K.ColorJitter(p=1.0, order=(step, 0), **kwargs)(negative)
        assert float(step_first.min()) >= 0.0
        assert float(step_first.max()) > 0.25
        # Leaving index 0 out of a fixed order skips the brightness step and its clamp.
        above = torch.full((1, 3, 2, 2), 2.0, device=device, dtype=dtype)
        self.assert_close(K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0, order=(1, 2, 3))(above), above)

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
        self.assert_close(saturation(two), two)
        self.assert_close(saturation(two.expand(2, 3, 3, 3)), torch.ones_like(two.expand(2, 3, 3, 3)))
        four_channel = torch.rand(2, 4, 3, 3).to(device=device, dtype=dtype)
        with pytest.raises(BaseError, match="Not a color or gray tensor"):
            saturation(four_channel)

    # Issue #4785: ColorJitter's brightness step multiplies (identity 1) but is guarded as if 0 were
    # neutral, so a batch whose factors are all 0 comes back unchanged while the same sample in a mixed
    # batch comes back black.  The fix makes the all-zero batch black and flips the first assertion.
    def test_wart_color_jitter_zero_brightness_factor_is_skipped_4785(self, device, dtype):
        image = torch.full((2, 3, 2, 2), 0.5, device=device, dtype=dtype)
        aug = K.ColorJitter(brightness=(0.0, 0.0), p=1.0)
        assert torch.equal(aug(image), image)
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

    # RandomHue has no clamp, and a hue rotation keeps each pixel's largest and smallest channel, so a pixel
    # outside [0, 1] keeps a channel outside it.  A pixel whose largest channel is exactly 0 comes back as zeros.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); print(K.RandomHue((0.25, 0.25), p=1.0)(x.reshape(4, 3, 1, 1)).reshape(4, 3))
    def test_convention_random_hue_keeps_out_of_range_extremes(self, device, dtype):
        pixels = torch.tensor([[1.5, 0.2, 0.3], [0.5, -0.1, 0.2], [-0.2, -0.5, -0.9]], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomHue((0.25, 0.25), p=1.0)(pixels.reshape(3, 3, 1, 1)).reshape(3, 3)
        self.assert_close(out.amax(1), pixels.amax(1))
        self.assert_close(out.amin(1), pixels.amin(1))
        zero_max = torch.tensor([0.0, -0.5, -0.5], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        torch.manual_seed(_FORWARD_SEED)
        collapsed = K.RandomHue((0.25, 0.25), p=1.0)(zero_max)
        self.assert_close(collapsed, torch.zeros_like(collapsed))

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
            (-0.001, 0.5, 0.1, 0.099),
            (1.1, 0.5, -0.4, 0.3),
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

    # Issue #4430, the upper end: `clamp(x + a, 0, 1)` sends every x >= 1.5 to exactly 1.0 for any admissible
    # addition, which the inversion returns as 0, so such an input is silently all-zero on every draw.
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_wart_random_solarize_at_least_one_point_five_collapses_on_every_draw_4430(self, device, dtype, seed):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.empty(2, 3, 6, 8).uniform_(1.5, 3.0).to(device=device, dtype=dtype)
        torch.manual_seed(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = K.RandomSolarize(p=1.0)(image)
        assert bool((out == 0).all())
        assert not caught
        # A fixture that straddles the bound does not collapse.
        torch.manual_seed(_FIXTURE_SEED)
        straddling = (torch.rand(2, 3, 6, 8) * 2.0).to(device=device, dtype=dtype)
        torch.manual_seed(seed)
        assert K.RandomSolarize(p=1.0)(straddling).unique().numel() > 1

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
        assert float(shifts.min()) < -0.4 and float(shifts.max()) > 0.4
        assert float(shifts.abs().max()) <= 0.5
        image = torch.tensor([1.5, -0.25], device=device, dtype=dtype).reshape(1, 1, 1, 2).expand(1, 3, 1, 2)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomRGBShift(r_shift_limit=0.0, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)(image.contiguous())
        self.assert_close(out, image.clamp(0.0, 1.0))

    # Issue #4430, RandomRGBShift: the clamp on the sum makes an image at or below `-limit` silently all-zero
    # on every draw.
    def test_wart_random_rgb_shift_at_or_below_minus_limit_collapses_on_every_draw_4430(self, device, dtype):
        image = torch.full((2, 3, 6, 8), -0.5, device=device, dtype=dtype)
        for seed in range(20):
            torch.manual_seed(seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = K.RandomRGBShift(p=1.0)(image)
            assert float(out.abs().max()) == 0.0, seed
            assert not caught

    # Issue #4782: a tuple limit is neither read as a range nor rejected by name; it dies on the unary
    # minus that builds `(-limit, limit)`.  A fix that accepts the tuple, or raises a kornia error, flips it.
    @pytest.mark.device_agnostic
    def test_wart_random_rgb_shift_tuple_limit_raises_a_raw_type_error_4782(self):
        with pytest.raises(TypeError) as info:
            K.RandomRGBShift((0.1, 0.5), p=1.0)
        # A kornia rejection would name the argument; the raw error from the unary minus does not.
        assert not any(word in str(info.value) for word in ("limit", "shift"))

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

    # The selected row scales red and blue only, then `clamp(max=1.0)` applies to all three channels:
    # green above 1 is cut back although it is not scaled, and negatives stay negative.
    # Snippet used to generate expected:
    #   x = torch.tensor([[1.5, -0.5], [1.7, -0.5], [1.5, -0.5]]).reshape(1, 3, 1, 2)
    #   aug = K.RandomPlanckianJitter(mode="blackbody", select_from=[0], p=1.0); print(aug.pl[0], aug(x))
    def test_convention_random_planckian_jitter_scales_red_and_blue_then_clamps_above(self, device, dtype):
        image = torch.tensor([[1.5, -0.5], [1.7, -0.5], [1.5, -0.5]], device=device, dtype=dtype).reshape(1, 3, 1, 2)
        aug = K.RandomPlanckianJitter(mode="blackbody", select_from=[0], p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        red_gain, blue_gain = (float(value) for value in aug.pl[0])
        scaled = torch.stack([image[:, 0] * red_gain, image[:, 1], image[:, 2] * blue_gain], dim=1)
        self.assert_close(out, scaled.clamp(max=1.0))
        assert float(out[0, 1, 0, 0]) == 1.0 and float(out[0, 2, 0, 0]) < 0.01
        assert bool((out[..., 1] < 0).all())

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
    def test_convention_random_jpeg_out_of_range_input_need_not_be_solid(self, device, dtype, negative):
        if not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        # A nonconstant grayscale field distinguishes clamping the decoded image from clamping
        # the input. Keep its values away from the endpoints even after conversion to bfloat16.
        torch.manual_seed(_FIXTURE_SEED)
        field = (torch.rand(1, 1, 32, 32) * 2.0 + 0.03125).expand(1, 3, -1, -1).contiguous()
        image = (-field if negative else 1.0 + field).to(device=device, dtype=dtype)
        assert bool((image < 0.0).all()) if negative else bool((image > 1.0).all())
        aug = K.RandomJPEG(jpeg_quality=(50.0, 50.0), p=1.0)
        out = aug(image)
        assert bool(torch.isfinite(out).all())
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert bool(((out > 0.0) & (out < 1.0)).any())
        # Quantization can also give a constant black input small positive values, so the range
        # assertion alone would not catch a misplaced input clamp.
        assert not torch.equal(out, aug(image.clamp(0.0, 1.0)))

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
        assert float(sharpness.min()) >= 0.0 and float(sharpness.max()) <= 0.5
        assert float(sharpness.min()) < 0.1  # the lower end is the floored -x, not a centred x / 2
        jpeg = K.RandomJPEG(50.0, p=1.0)._param_generator
        assert (float(jpeg.jpeg_quality_sampler.low), float(jpeg.jpeg_quality_sampler.high)) == (1.0, 100.0)

    # Issue #4564: RandomClahe rejects an out-of-[0, 1] input with kornia's error naming `equalize_clahe` and
    # the range, on MPS as on the CPU.
    def test_convention_random_clahe_out_of_range_error_names_the_range_4564(self, device, dtype):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the index error is a device-side assert that poisons the context")
        torch.manual_seed(_FIXTURE_SEED)
        image = (torch.rand(1, 3, 16, 16) * 2).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]") as info:
            _sync(K.RandomClahe(p=1.0)(image).device)
        assert "image / 255.0" in str(info.value)

    # The rejection is one 8-bit code wide either side of [0, 1]: the guard is the 256-entry lookup indexed
    # with `(input * 255).long()`, so the admitted band is (-1/255, 1 + 1/255).
    @pytest.mark.parametrize(
        ("value", "admitted"),
        [
            (1.0, True),
            (0.0, True),
            (1.0 + 0.5 / 255, True),
            (1.0 + 1.5 / 255, False),
            (-0.5 / 255, True),
            (-1.5 / 255, False),
        ],
    )
    def test_convention_random_clahe_admitted_band_is_one_code_wide(self, device, dtype, value, admitted):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        image = torch.full((1, 1, 16, 16), 0.5, device=device, dtype=dtype)
        image[0, 0, 0, 0] = value
        torch.manual_seed(_FORWARD_SEED)
        if admitted:
            out = K.RandomClahe(p=1.0)(image)
            _sync(image.device)
            assert out.isfinite().all()
        else:
            with pytest.raises(RuntimeError, match=r"equalize_clahe expects input values in \[0, 1\]"):
                _sync(K.RandomClahe(p=1.0)(image).device)

    # Issue #2531: each axis of `grid_size` is tiled on its own and padded up to a whole number of tiles,
    # so a grid works whether or not it divides the image, and an exactly dividing grid can still pad (the
    # tile is rounded up to an even size).  The padding is reflect at the end of each axis, so the
    # pre-padded image cropped back reproduces the output bitwise.
    @pytest.mark.parametrize(
        ("grid_size", "size", "pad"),
        [
            ((3, 3), 10, (2, 2)),
            ((4, 4), 20, (4, 4)),
            ((5, 5), 20, (0, 0)),
            ((4, 5), 20, (4, 0)),
            ((3, 5), 30, (0, 0)),
            ((1, 2), 10, (0, 2)),
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
    # admitted.  The 8 x 8 case in between is the #4783 wart below.
    def test_convention_random_clahe_default_grid_rejects_a_smaller_image(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match=r"Cannot compute tiles on the image according to the given grid size"):
            K.RandomClahe(p=1.0)(torch.rand(1, 1, 7, 7, device=device, dtype=dtype))
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomClahe(p=1.0)(torch.rand(1, 1, 9, 9, device=device, dtype=dtype)).shape == (1, 1, 9, 9)

    # Issue #4783: an image exactly as large as the grid passes the named size check and then fails in
    # the reflect padding with torch's raw error.  A fix that handles it, or names it, flips this pin.
    def test_wart_random_clahe_grid_sized_image_raises_a_raw_padding_error_4783(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(RuntimeError) as info:
            _sync(K.RandomClahe(p=1.0)(torch.rand(1, 1, 8, 8, device=device, dtype=dtype)).device)
        # A kornia error would name the grid or its tiles; torch's padding error does not.
        assert not any(word in str(info.value).lower() for word in ("grid", "tile"))

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
        assert not fast.requires_grad and fast.grad_fn is None
        torch.manual_seed(_FORWARD_SEED)
        slow = K.RandomClahe(p=1.0, slow_and_differentiable=True)(image)
        assert slow.requires_grad
        slow.sum().backward()
        assert image.grad is not None and float(image.grad.abs().sum()) > 0.0

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
