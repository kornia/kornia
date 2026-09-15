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

import pytest
import torch

import kornia.augmentation as K
from kornia.core.exceptions import ShapeError
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, normalize_min_max

from testing.base import BaseTester


@pytest.fixture(autouse=True)
def _restore_global_rng():
    # Every pin below seeds the global RNG so its draw is reproducible; fork it so the seeding
    # cannot shift the draw of any unseeded pre-existing test in this directory.
    with torch.random.fork_rng(devices=[]):
        yield


# Constructor arguments are verbatim from the 6c audit probe (``audit-6c.py::mk()``), so the
# executed literals below are the ones the fact table records.  ``RandomDissolving`` is the one 2D
# intensity class with no executed row: constructing it prompts on stdin for ``diffusers`` and its
# first forward downloads a Stable-Diffusion checkpoint, so it is excluded here as it was there.
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
    "RandomPosterize": lambda: K.RandomPosterize(bits=3, p=1.0),
    "RandomRGBShift": lambda: K.RandomRGBShift(p=1.0),
    "RandomRain": lambda: K.RandomRain(number_of_drops=(3, 3), drop_height=(1, 2), drop_width=(1, 2), p=1.0),
    "RandomSaltAndPepperNoise": lambda: K.RandomSaltAndPepperNoise(p=1.0),
    "RandomSaturation": lambda: K.RandomSaturation((2.0, 2.0), p=1.0),
    "RandomSharpness": lambda: K.RandomSharpness(1.0, p=1.0),
    "RandomSnow": lambda: K.RandomSnow(p=1.0),
    "RandomSolarize": lambda: K.RandomSolarize(0.1, 0.1, p=1.0),
}

# Row 6c-01: the package splits four ways on an input outside [0, 1].  Issue #4430.
# "Clamps both ends" names the observed output range, not the mechanism: see the caveats in the
# header comment of test_convention_value_range_policy before citing this group in prose.
_CLAMPS_BOTH_ENDS = (
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
# Row 6c-04: ``output.clamp(max=1.0)`` only, so a negative input stays negative.
_CLAMPS_UPPER_END_ONLY = ("RandomPlanckianJitter",)
_PASSES_RANGE_THROUGH = (
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
# Row 6c-02, in the state #4489 left it: the only class that rejects the input instead.
_REJECTS_OUT_OF_RANGE = ("RandomEqualize",)

# Row 6c-03: ten of the clamping classes do not merely clamp -- a wholly negative input comes back
# as an all-zero image, with no warning.  Issue #4430.
_COLLAPSES_ON_NEGATIVE_INPUT = (
    "ColorJitter",
    "RandomContrast",
    "RandomGaussianIllumination",
    "RandomLinearIllumination",
    "RandomLinearCornerIllumination",
    "RandomPlasmaShadow",
    "RandomPosterize",
    "RandomSnow",
    "RandomSharpness",
    "RandomSolarize",
)

# Fixture seed for the shared out-of-range images, and the seed drawn immediately before each
# construct-and-forward, exactly as ``audit-6c.py::r1`` does it.
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
    # otherwise surface inside an unrelated later test (audit-6c.py::sync, audit review C3).
    if device.type == "mps":
        torch.mps.synchronize()


def _run(name: str, image: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(_FORWARD_SEED)
    out = _INTENSITY_FACTORIES[name]()(image)
    _sync(image.device)
    return out


class TestIntensityValueRangeConventions(BaseTester):
    # Row 6c-01 (issue #4430): the 35 executable 2D intensity classes split four ways on an input
    # outside [0, 1] -- 16 keep the output inside [0, 1], 1 bounds the upper end only, 17 pass the
    # range through and 1 rejects the input.  Membership is the claim; the individual minima and
    # maxima are fixture-bound and stay in comments.
    # Three caveats the audit attaches to this row, so that "16 clamp both ends" is not written as
    # prose without them (executed 2026-09-15, torch 2.14.0, cpu, on this test's own fixtures):
    #   * `RandomBrightness` and `RandomContrast` are bounded only by their default
    #     `clip_output=True`.  With `clip_output=False` the range is passed through:
    #     `RandomBrightness((1.5, 1.5), clip_output=False)` on `[0, 2]` gives `max=2.496` and
    #     `RandomContrast((1.5, 1.5), clip_output=False)` gives `max=2.99399` (audit row 6c-35,
    #     `R1 RandomBrightness(clip=False) in=[0,2] -> max=2.496`).  The `(clip=False)` variants are
    #     deliberately outside `_INTENSITY_FACTORIES`, as they were outside the audit's count.
    #   * `RandomPosterize`'s `[0, 1]` output is a `uint8` round-trip that wraps or floors, not a
    #     clamp -- see test_wart_random_posterize_out_of_range_wraps_4430 (audit row 6c-38).
    #   * `RandomAutoContrast` is a per-sample, per-channel min-max rescale, not a clamp: its
    #     `clip_output` flag is dead because `normalize_min_max` already returns `[0, 1]`
    #     (audit row 6c-35, #4436; pinned by test_convention_random_auto_contrast_is_normalize_min_max
    #     and, for the dead flag, by TestRandomAutoContrast::test_clip_output_does_not_change_the_output).
    # Snippet used to generate expected: audit-6c.py::r1 re-executed on 90650596 as
    #   torch.manual_seed(1234); base = torch.rand(2, 3, 6, 8)
    #   for x in (base * 2.0, base - 1.0):
    #       for name, ctor in mk(): torch.manual_seed(0); print(name, ctor()(x).aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu), e.g. `RandomBrightness in=[0,2] -> min=0.501276
    # max=1`, `RandomSnow in=[0,2] -> min=0.00127578 max=1.996`, `RandomPlanckianJitter
    # in=[-1,0] -> min=-1.41941 max=-0.00200176`.
    @pytest.mark.parametrize("name", sorted(_INTENSITY_FACTORIES))
    def test_convention_value_range_policy(self, device, dtype, name):
        groups = (_CLAMPS_BOTH_ENDS, _CLAMPS_UPPER_END_ONLY, _PASSES_RANGE_THROUGH, _REJECTS_OUT_OF_RANGE)
        assert tuple(len(group) for group in groups) == (16, 1, 17, 1)
        assert set().union(*groups) == set(_INTENSITY_FACTORIES) and len(_INTENSITY_FACTORIES) == 35
        if dtype == torch.float16 and name in ("ColorJiggle", "ColorJitter"):
            pytest.skip(
                "float16 only (#4560): the contrast step collapses the [-1, 0] fixture to exact zeros "
                "and rgb_to_hsv's eps=1e-8 then underflows for a black pixel, so adjust_hue returns "
                "NaN (adjust_hue(torch.zeros(1, 3, 2, 2, dtype=torch.float16), 0.1) is already NaN); "
                "float32, float64 and bfloat16 are finite"
            )
        fixtures = _out_of_range_fixtures(device, dtype)
        if name in _REJECTS_OUT_OF_RANGE:
            # The message is pinned separately; on MPS the range check is skipped
            # (kornia/enhance/adjust.py:37-47) and the raw gather raises instead, so only the
            # rejection itself is device-independent.
            for image in fixtures.values():
                with pytest.raises(RuntimeError):
                    _run(name, image)
            return
        # A clamped output is exactly 0.0 or 1.0, and the pass-through outputs sit near 2.0 or
        # -1.0, so a tolerance this loose still separates the groups in bfloat16.
        tol = 1e-3
        ranges = {tag: _run(name, image).aminmax() for tag, image in fixtures.items()}
        if name in _CLAMPS_BOTH_ENDS:
            for tag, (low, high) in ranges.items():
                assert float(low) >= -tol, f"{name} on {tag} left the lower end unclamped"
                assert float(high) <= 1 + tol, f"{name} on {tag} left the upper end unclamped"
        elif name in _CLAMPS_UPPER_END_ONLY:
            for tag, (_, high) in ranges.items():
                assert float(high) <= 1 + tol, f"{name} on {tag} left the upper end unclamped"
            # The lower end is checked on the fixture that has one: [0, 2] is non-negative anyway.
            negative_min = float(ranges["[-1, 0]"][0])
            assert negative_min < -tol, f"{name} also clamped the lower end (min {negative_min})"
        else:
            assert name in _PASSES_RANGE_THROUGH
            assert any(float(high) > 1 + tol or float(low) < -tol for low, high in ranges.values()), (
                f"{name} kept the output inside [0, 1] on both out-of-range fixtures"
            )

    # Row 6c-03 (issue #4430): ten of the clamping classes return an all-zero image on a wholly
    # negative input.  #4430 offers two coherent outcomes (clamp, or reject as RandomEqualize now
    # does), so this is a wart pin on today's behavior, not a strict xfail on a settled contract.
    # Snippet used to generate expected: the r1 snippet above, reading the `in=[-1,0]` rows;
    # executed 2026-09-15 (torch 2.14.0, cpu) -- all ten print `min=0 max=0`.
    @pytest.mark.parametrize("name", _COLLAPSES_ON_NEGATIVE_INPUT)
    def test_wart_intensity_negative_input_collapses_to_zero_4430(self, device, dtype, name):
        if dtype == torch.float16 and name == "ColorJitter":
            pytest.skip(
                "float16 only (#4560): the collapsed image reaches adjust_hue, whose rgb_to_hsv gives a "
                "NaN saturation for a black pixel there (eps=1e-8 underflows in float16)"
            )
        if dtype == torch.float64 and name == "RandomPosterize":
            pytest.skip(
                "float64 only: posterize's `(x * 255).to(torch.uint8)` wraps the negative fixture to "
                "non-zero codes instead of saturating to 0 (mechanism measured in "
                "test_wart_random_posterize_out_of_range_wraps_4430), so the collapse is a "
                "float32/float16/bfloat16 observation"
            )
        image = _out_of_range_fixtures(device, dtype)["[-1, 0]"]
        assert float(_run(name, image).abs().max()) == 0.0

    # Row 6c-38 (issue #4430): RandomPosterize's uint8 round-trip wraps rather than clamps, so an
    # input above 1 comes back as a full-range posterized image and a negative input as a constant.
    # Snippet used to generate expected:
    #   g = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32)
    #   for x in (g * 2.0, g - 1.0):
    #       torch.manual_seed(0); y = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(x)
    #       print(len(y.unique()), y.min().item(), y.max().item())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `8 0.0 0.8784313797950745` and `1 0.0 0.0`.
    @pytest.mark.parametrize("scale", ["[0, 2]", "[-1, 0]"])
    def test_wart_random_posterize_out_of_range_wraps_4430(self, device, dtype, scale):
        if dtype == torch.float64 and scale == "[-1, 0]":
            pytest.skip(
                "float64 only: `(x * 255).to(torch.uint8)` is out of the uint8 domain for a negative x, "
                "and torch's two conversion paths disagree. The input values are bit-identical in the "
                "two dtypes (x[254] == -0.003921568393707275 in both, x[254] * 255 == "
                "-0.9999999403953552 in both), so this is a kernel split, not a value difference: "
                "measured on ((arange(256, dtype=float32) / 255).to(d) - 1.0) * 255, float32 saturates "
                "every negative element to code 0 once the tensor reaches 8 elements (1 distinct code) "
                "while float64 wraps modulo 256 at every size (255 distinct codes, 1..255); below 8 "
                "elements float32 wraps too, and float16/bfloat16 follow float32. So the negative ramp "
                "keeps 8 posterize levels in float64 and collapses to a constant 0 elsewhere. The wrap "
                "itself is what this pin records"
            )
        ramp = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32).to(device=device, dtype=dtype)
        image = ramp * 2.0 if scale == "[0, 2]" else ramp - 1.0
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(image)
        _sync(image.device)
        if scale == "[0, 2]":
            assert len(out.unique()) == 8
            self.assert_close(out.max(), out.new_tensor(224 / 255))
        else:
            assert len(out.unique()) == 1
            assert float(out.max()) == 0.0

    # Row 6c-02 in the state #4489 left it (#4431 closed): RandomEqualize is the one 2D intensity
    # class that rejects an out-of-range input, and its message names the [0, 1] range it needs.
    # The guard is `input * 255` inside (-1, 256), so `[0, 1.0001]` -- the audit's executed
    # instance -- is still admitted; only past 256/255 does the histogram index go out of bounds.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    #   for x in (g, g * 1.0001, g * 2, g - 1):
    #       torch.manual_seed(0); K.RandomEqualize(p=1.0)(x)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> the last two raise `RuntimeError: equalize expects
    # input values in [0, 1]. Scale the image into that range first, for example image / 255.0 for
    # 8-bit data.`, the first two return a 64-level image.
    def test_convention_random_equalize_rejects_out_of_range_with_named_range(self, device, dtype):
        if device.type == "mps":
            pytest.skip(
                "MPS only: _assert_async_value_check skips the range check there because "
                "aten::_assert_async has no MPS kernel (kornia/enhance/adjust.py:37-47), so the raw "
                "gather raises without naming the range"
            )
        ramp = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).to(device=device, dtype=dtype)
        for image in (ramp * 2.0, ramp - 1.0):
            torch.manual_seed(_FORWARD_SEED)
            with pytest.raises(RuntimeError, match=r"\[0, 1\]"):
                K.RandomEqualize(p=1.0)(image)
        for image in (ramp, ramp * 1.0001):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomEqualize(p=1.0)(image)
            assert out.shape == image.shape
            assert torch.isfinite(out).all()


class TestIntensityColourConventions(BaseTester):
    # Row 6c-05: RandomGamma is `clamp(gain * x ** gamma, 0, 1)`; the clamp is not optional, so the
    # class has no `clip_output` escape hatch the way RandomBrightness and RandomContrast do.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8)
    #   torch.manual_seed(0); y = K.RandomGamma((gamma, gamma), (gain, gain), p=1.0)(g)
    #   print((y - (gain * g ** gamma).clamp(0, 1)).abs().max(), (y - gain * g ** gamma).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0 / 1` at gamma=1.0 gain=2.0 and
    # `5.96046e-08 / 0.5` at gamma=0.5 gain=1.5.
    @pytest.mark.parametrize(("gamma", "gain", "unclamped_gap"), [(1.0, 2.0, 1.0), (0.5, 1.5, 0.5)])
    def test_convention_random_gamma_is_clamped_gain_times_power(self, device, dtype, gamma, gain, unclamped_gap):
        ramp = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGamma((gamma, gamma), (gain, gain), p=1.0)(ramp)
        raw = gain * ramp**gamma
        self.assert_close(out, raw.clamp(0.0, 1.0))
        assert float((out - raw).abs().max()) > unclamped_gap / 2

    # Rows 6c-06 and 6c-49: RandomBrightness re-bases its factor -- it passes `factor - 1` to
    # kornia.enhance.adjust_brightness, whose identity is 0.0 -- while RandomContrast and
    # RandomSaturation pass their factor through unchanged.
    # Snippet used to generate expected:
    #   c = torch.full((1, 1, 2, 2), 0.4); px = torch.tensor([[0.6], [0.4], [0.2]]).reshape(1, 3, 1, 1)
    #   torch.manual_seed(0); print(K.RandomBrightness((1.5, 1.5), p=1.0)(c).flatten()[0])
    #   print(adjust_brightness(c, 1.5).flatten()[0], adjust_brightness(c, 0.5).flatten()[0])
    #   torch.manual_seed(0); print(K.RandomContrast((1.5, 1.5), p=1.0)(c).flatten()[0])
    #   torch.manual_seed(0); print(K.RandomSaturation((2.0, 2.0), p=1.0)(px).flatten())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0.9 | 1 | 0.9`, `0.6`, `[0.6, 0.3, 0]`.
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

    # Rows 6c-06 and 6c-49, the other side of the same claim: each class's identity factor is 1.0,
    # which is what makes the RandomBrightness offset invisible until it is compared to the
    # primitive.  Checked on an asymmetric batch of two distinct samples.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 6, 8)
    #   torch.manual_seed(0); print((cls((1.0, 1.0), p=1.0)(x) - x).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0`, `0`, `4.17233e-07`.
    @pytest.mark.parametrize("cls", [K.RandomBrightness, K.RandomContrast, K.RandomSaturation])
    def test_convention_point_range_factor_one_is_identity(self, device, dtype, cls):
        if dtype in _HALF and cls is K.RandomSaturation:
            pytest.skip(
                "half precision only: adjust_saturation's RGB -> HSV -> RGB round trip is lossy there, so "
                "the identity holds only to 2.9e-3 in float16 and 3.1e-2 in bfloat16 (measured), not to "
                "the dtype tolerance; RandomBrightness and RandomContrast are exact in both"
            )
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(cls((1.0, 1.0), p=1.0)(image), image)

    # Rows 6c-07 and 6c-08: ColorJiggle and ColorJitter draw byte-identical parameters from the
    # same seed -- including the `order` key -- and differ only in the four primitives they call
    # (adjust_*_accumulative / _with_mean_subtraction / _with_gray_subtraction vs the plain ones).
    # Replaying ColorJitter with ColorJiggle's own parameters reproduces the same gap, which is
    # what rules out a sampling-order difference.  The gap is fixture-bound (max|diff| 0.124188
    # here), so only the inequality is asserted.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 6, 8)
    #   torch.manual_seed(7); a = K.ColorJiggle(0.2, 0.2, 0.2, 0.1, p=1.0); ya = a(x)
    #   torch.manual_seed(7); b = K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0); yb = b(x)
    #   print((ya - yb).abs().max(), (ya - b(x, params=a._params)).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0.124188` twice, with `params equal: True` and
    # the drawn `order` `[0, 3, 2, 1]` on both.
    def test_convention_color_jiggle_and_jitter_draw_identical_params(self, device, dtype):
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
        # Replaying ColorJitter with ColorJiggle's own parameters reproduces ColorJitter's output,
        # so the gap above is the four primitives and not the draw.
        replayed = jitter(image, params=jiggle._params)
        self.assert_close(replayed, jittered)
        assert float((jiggled - replayed).abs().max()) > 0.05

    # Row 6c-08: both classes are the identity at (0, 0, 0, 0), so the gap above is the primitives
    # and not a stray factor.  Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 6, 8)
    #   torch.manual_seed(0); print((cls(0, 0, 0, 0, p=1.0)(x) - x).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0` for both.
    @pytest.mark.parametrize("cls", [K.ColorJiggle, K.ColorJitter])
    def test_convention_color_jiggle_and_jitter_are_identity_at_zero(self, device, dtype, cls):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(cls(0.0, 0.0, 0.0, 0.0, p=1.0)(image), image)

    # Row 6c-10: RandomHue's argument is turns of the hue circle, restricted to (-0.5, 0.5); the
    # class multiplies it by 2*pi before calling kornia.enhance.adjust_hue, which takes radians.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 6, 8)
    #   torch.manual_seed(0); y = K.RandomHue((0.25, 0.25), p=1.0)(x)
    #   print((y - adjust_hue(x, 0.25 * 2 * math.pi)).abs().max(), (y - adjust_hue(x, 0.25)).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0` and `0.87243`.
    def test_convention_random_hue_is_turns_of_the_hue_circle(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomHue((0.25, 0.25), p=1.0)(image)
        self.assert_close(out, adjust_hue(image, 0.25 * 2 * math.pi))
        assert float((out - adjust_hue(image, 0.25)).abs().max()) > 0.5
        # The (-0.5, 0.5) bound that goes with those units is pinned in
        # test_convention_intensity_constructors_reject_out_of_bounds.

    # Row 6c-11: RandomGrayscale keeps the channel count and writes the same value into every
    # channel; the default weights are the ITU-R BT.601 luma weights, so a pure red pixel becomes
    # 0.299, and custom weights replace them wholesale.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 3, 6, 8); x[0, 0] = 1.0
    #   torch.manual_seed(0); print(K.RandomGrayscale(p=1.0)(x)[0, 0, 0, 0])
    #   torch.manual_seed(0)
    #   print(K.RandomGrayscale(rgb_weights=torch.tensor([1.0, 0.0, 0.0]), p=1.0)(x)[0, 0, 0, 0])
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0.299` and `1`.
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

    # Row 6c-11, one parameter away from the pin above: a non-RGB channel count is not rejected --
    # the channel count is preserved and every channel becomes the plain mean over channels.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(1, C, 6, 8)
    #   torch.manual_seed(0); y = K.RandomGrayscale(p=1.0)(x)
    #   print(y.shape, (y - x.mean(1, keepdim=True)).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(1, C, 6, 8)` and `0` for C in (1, 2, 4).
    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_convention_random_grayscale_non_rgb_is_the_channel_mean(self, device, dtype, channels):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(1, channels, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGrayscale(p=1.0)(image)
        assert out.shape == image.shape
        self.assert_close(out, image.mean(1, keepdim=True).expand_as(image))

    # Row 6c-14: out channel `c` is in channel `channels[b][c]` -- the drawn index reads the source,
    # it does not name the destination -- and the permutation is drawn per sample.  The claim is a
    # labelling, so it is checked against the drawn permutation of every sample rather than against
    # one literal, on per-sample constant planes 0..11 that make the relabelling visible.
    # Snippet used to generate expected:
    #   x = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1).expand(4, 3, 6, 8)
    #   torch.manual_seed(0); a = K.RandomChannelShuffle(p=1.0); y = a(x.contiguous())
    #   print(a._params["channels"].tolist(), x[:, :, 0, 0].tolist(), y[:, :, 0, 0].tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> channels `[[2, 0, 1], [0, 1, 2], [2, 0, 1],
    # [1, 2, 0]]` sending `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]` to
    # `[[2, 0, 1], [3, 4, 5], [8, 6, 7], [10, 11, 9]]`.
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
        for b in range(4):
            for c in range(3):
                self.assert_close(out[b, c], image[b, int(order[b, c])])

    # Row 6c-14, one parameter away: same_on_batch collapses the draw to a single permutation.
    # Snippet used to generate expected: as above with `same_on_batch=True`; executed 2026-09-15
    # (torch 2.14.0, cpu) -> `[[2, 0, 1], [2, 0, 1], [2, 0, 1], [2, 0, 1]]`.
    def test_convention_random_channel_shuffle_same_on_batch(self, device, dtype):
        planes = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1).expand(4, 3, 6, 8)
        image = planes.contiguous().to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomChannelShuffle(p=1.0, same_on_batch=True)
        aug(image)
        order = aug._params["channels"]
        assert order.shape == (4, 3)
        assert bool((order == order[0]).all())

    # Row 6c-15: RandomChannelDropout writes `fill_value` (0 by default) into exactly the channels
    # named by the drawn `channel_idx`, one independent draw per sample, and leaves the rest alone.
    # Snippet used to generate expected:
    #   x = torch.ones(2, 3, 6, 8)
    #   torch.manual_seed(0); a = K.RandomChannelDropout(num_drop_channels=n, fill_value=v, p=1.0)
    #   print(a(x).mean((2, 3)).tolist(), a._params["channel_idx"].tolist())
    #   torch.manual_seed(0)
    #   b = K.RandomChannelDropout(num_drop_channels=n, fill_value=v, p=1.0, same_on_batch=True)
    #   b(x); print(b._params["channel_idx"].tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> default: `[[1, 1, 0], [0, 1, 1]]` with idx
    # `[[2], [0]]`; n=2, v=0.7: `[[0.7, 1, 0.7], [0.7, 0.7, 1]]` with idx `[[2, 0], [0, 1]]`.  The
    # two samples therefore differ; `same_on_batch=True` collapses them to `[[2], [2]]` and
    # `[[2, 0], [2, 0]]`.
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
        # The draw is per sample: validating the output against the reported channel_idx alone would
        # also pass for an implementation that dropped the same channels across the whole batch.
        assert index[0].tolist() != index[1].tolist(), "the seeded per-sample draws coincide"
        torch.manual_seed(_FORWARD_SEED)
        batched = K.RandomChannelDropout(
            num_drop_channels=num_drop_channels, fill_value=fill_value, p=1.0, same_on_batch=True
        )
        batched(image)
        shared = batched._params["channel_idx"]
        assert shared.shape == (2, num_drop_channels)
        assert shared[0].tolist() == shared[1].tolist()

    # Row 6c-40: RandomInvert is `max_val - x`, with no clamp, so an input outside [0, 1] comes
    # back reflected around `max_val` rather than clipped.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 2, 48).reshape(1, 1, 6, 8)
    #   torch.manual_seed(0); y = K.RandomInvert(max_val=m, p=1.0)(g); print(y.aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(-1, 1)` at max_val=1.0 and `(0, 2)` at 2.0.
    @pytest.mark.parametrize(("max_val", "expected_range"), [(1.0, (-1.0, 1.0)), (2.0, (0.0, 2.0))])
    def test_convention_random_invert_subtracts_from_max_val(self, device, dtype, max_val, expected_range):
        ramp = torch.linspace(0, 2, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomInvert(max_val=max_val, p=1.0)(ramp)
        self.assert_close(out, max_val - ramp)
        self.assert_close(out.min(), out.new_tensor(expected_range[0]))
        self.assert_close(out.max(), out.new_tensor(expected_range[1]))

    # Row 6c-39: `additions` is added to the whole image first and the sum is clamped into [0, 1];
    # only then is everything at or above `thresholds` inverted.  The order matters: adding after
    # the inversion would move the dark pixels the other way.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8)
    #   torch.manual_seed(0); y = K.RandomSolarize((t, t), (a, a), p=1.0)(g)
    #   z = (g + a).clamp(0, 1); print((y - torch.where(z < t, z, 1.0 - z)).abs().max(), y.aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> formula diff `0` for every pair below, and
    # threshold=0.5 addition=0.25 sends the input 0.0212766 to 0.271277.
    @pytest.mark.parametrize(("threshold", "addition"), [(0.5, 0.0), (0.5, 0.25), (0.0, 0.0)])
    def test_convention_random_solarize_adds_before_inverting(self, device, dtype, threshold, addition):
        ramp = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSolarize((threshold, threshold), (addition, addition), p=1.0)(ramp)
        shifted = (ramp + addition).clamp(0.0, 1.0)
        self.assert_close(out, torch.where(shifted < threshold, shifted, 1.0 - shifted))
        if (threshold, addition) == (0.5, 0.25):
            self.assert_close(out.flatten()[1], out.new_tensor(1.0 / 47.0 + 0.25))

    # Row 6c-50: a scalar `thresholds` is a half-width around the function default 0.5, while a
    # scalar `additions` is a symmetric range about zero -- the class default centres on
    # kornia.enhance.solarize's default rather than equalling it.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); p = K.RandomSolarize(0.1, 0.1, p=1.0).forward_parameters((256, 1, 6, 8))
    #   print(p["thresholds"].aminmax(), p["additions"].aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> thresholds in `[0.400247, 0.599413]` and
    # additions in `[-0.0991148, 0.0999676]`.
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

    # Row 6c-37: `bits=(k, k)` leaves exactly 2**k distinct levels on a 256-level ramp, so k=0 is a
    # constant image and k=8 is the identity.
    # Snippet used to generate expected:
    #   g = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32)
    #   torch.manual_seed(0); print(len(K.RandomPosterize(bits=(k, k), p=1.0)(g).unique()))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> 1, 2, 4, 8, 16, 32, 64, 128, 256 for k = 0..8.
    @pytest.mark.parametrize("bits", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    def test_convention_random_posterize_bits_are_levels(self, device, dtype, bits):
        ramp = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32)
        image = ramp.to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(float(bits), float(bits)), p=1.0)(image)
        assert len(out.unique()) == 2**bits

    # Row 6c-37: an `int` argument is the LOWER bound of [x, 8] and the draw is an int32 -- the
    # mirror image of RandomSharpness, where a scalar is the upper bound (row 6c-36).
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   p = K.RandomPosterize(bits=k, p=1.0).forward_parameters((256, 1, 4, 4))["bits_factor"]
    #   print(p.dtype, p.min().item(), p.max().item())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `torch.int32 3 8` and `torch.int32 6 8`.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("bits", [3, 6])
    def test_convention_random_posterize_int_argument_is_a_lower_bound(self, bits):
        torch.manual_seed(_FORWARD_SEED)
        drawn = K.RandomPosterize(bits=bits, p=1.0).forward_parameters((256, 1, 4, 4))["bits_factor"]
        assert drawn.dtype == torch.int32
        assert int(drawn.min()) == bits
        assert int(drawn.max()) == 8

    # Row 6c-36: the sharpness factor blends between the fully blurred image (0.0) and the input
    # (1.0), so 0.5 moves exactly half as far as 0.0 and values above 1 sharpen.  The scale is
    # linear in the factor, which is what makes "0 = blurred, 1 = identity" checkable.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 6, 8)
    #   for f in (0.0, 0.5, 1.0, 2.0):
    #       torch.manual_seed(0); print(f, (K.RandomSharpness((f, f), p=1.0)(x) - x).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0.3818`, `0.1909`, `0`, `0.257182`, and the
    # factor-2.0 output stays inside [0, 1] (`min/max 0 / 1`).
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

    # Row 6c-36: a scalar `sharpness` is the UPPER bound of [0, x], so the class default of 0.5
    # never reaches the identity at 1.0 and therefore never sharpens.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   p = K.RandomSharpness(0.5, p=1.0).forward_parameters((256, 1, 6, 8))["sharpness"]
    #   print(p.aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `[0.000618011, 0.498533]`.
    @pytest.mark.device_agnostic
    def test_convention_random_sharpness_scalar_is_an_upper_bound(self):
        torch.manual_seed(_FORWARD_SEED)
        drawn = K.RandomSharpness(0.5, p=1.0).forward_parameters((256, 1, 6, 8))["sharpness"]
        assert drawn.shape == (256,)
        assert float(drawn.min()) >= 0.0
        assert float(drawn.max()) <= 0.5

    # Row 6c-34: one additive shift per channel per sample, clamped into [0, 1] by
    # kornia.enhance.shift_rgb.  The shift is replaced by an explicit +1.0 on red so the clamp is
    # exercised without depending on the draw.
    # Snippet used to generate expected:
    #   x = torch.full((2, 3, 6, 8), 0.5)
    #   torch.manual_seed(0)
    #   a = K.RandomRGBShift(r_shift_limit=1.0, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)
    #   p = a.forward_parameters(x.shape)
    #   print(p["r_shift"].tolist(), {k: tuple(v.shape) for k, v in p.items()})
    #   print(a(x, params=p).mean((2, 3)).tolist())          # the natural draw
    #   p["r_shift"] = torch.ones_like(p["r_shift"])         # the injection the pin asserts on
    #   print(a(x, params=p).mean((2, 3)).tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> drawn `r_shift [-0.00748682, 0.536444]` (one per
    # sample, and `g_shift`/`b_shift` both `[0, 0]`), all three of shape `(2,)`; the natural draw
    # gives `[[0.492513, 0.5, 0.5], [1, 0.5, 0.5]]` and the injected `+1.0` gives
    # `[[1, 0.5, 0.5], [1, 0.5, 0.5]]`.
    def test_convention_random_rgb_shift_is_per_channel_per_sample(self, device, dtype):
        image = torch.full((2, 3, 6, 8), 0.5, device=device, dtype=dtype)
        aug = K.RandomRGBShift(r_shift_limit=1.0, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        params = aug.forward_parameters(image.shape)
        for key in ("r_shift", "g_shift", "b_shift"):
            assert params[key].shape == (2,), f"{key} is not one shift per sample"
        # The two samples draw independently, so the shape above is not the whole claim: the seeded
        # draw is `[-0.00748682, 0.536444]`, and the zero-limit channels stay at exactly 0.
        assert float(params["r_shift"][0]) != float(params["r_shift"][1])
        self.assert_close(params["g_shift"], torch.zeros_like(params["g_shift"]), atol=0, rtol=0)
        self.assert_close(params["b_shift"], torch.zeros_like(params["b_shift"]), atol=0, rtol=0)
        params["r_shift"] = torch.ones_like(params["r_shift"])
        out = aug(image, params=params)
        self.assert_close(out[:, 0], torch.ones_like(out[:, 0]))
        self.assert_close(out[:, 1:], image[:, 1:])

    # Row 6c-33: `pl` is a persistent buffer holding the illuminant table the mode selects -- 25
    # rows for blackbody, 23 for CIED -- and `select_from` narrows the table itself.
    # Snippet used to generate expected:
    #   print(K.RandomPlanckianJitter(mode=m, p=1.0).pl.shape)
    #   print(K.RandomPlanckianJitter(select_from=[0, 1], p=1.0).pl.shape)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(25, 2)`, `(23, 2)`, `(2, 2)`.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("kwargs", "rows"), [({"mode": "blackbody"}, 25), ({"mode": "CIED"}, 23), ({"select_from": [0, 1]}, 2)]
    )
    def test_convention_random_planckian_jitter_table_follows_mode(self, kwargs, rows):
        aug = K.RandomPlanckianJitter(p=1.0, **kwargs)
        assert tuple(aug.pl.shape) == (rows, 2)
        assert [name for name, _ in aug.named_buffers()] == ["pl"]

    # Row 6c-32/6c-33: the illuminant table is an RGB ratio, so a non-RGB input is rejected rather
    # than broadcast.  Snippet used to generate expected:
    #   torch.manual_seed(0); K.RandomPlanckianJitter(p=1.0)(torch.rand(1, 1, 6, 8))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `ShapeError: Shape mismatch at dimension 0:
    # expected 3, got 1.`
    def test_convention_random_planckian_jitter_requires_three_channels(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ShapeError):
            # The sync is what surfaces an MPS kernel error inside this block rather than later.
            _sync(K.RandomPlanckianJitter(p=1.0)(image).device)

    # Row 6c-48 (issue #4428): because `pl` is a persistent buffer whose shape depends on `mode`, a
    # state_dict is mode-specific -- a blackbody checkpoint cannot be loaded into a CIED instance.
    # The other eleven 6c range buffers at least load; they are simply inert (6a pins that half of
    # #4428 on RandomBrightness).
    # Snippet used to generate expected:
    #   K.RandomPlanckianJitter(mode="CIED").load_state_dict(
    #       K.RandomPlanckianJitter(mode="blackbody").state_dict())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `RuntimeError: Error(s) in loading state_dict for
    # RandomPlanckianJitter: size mismatch for pl: copying a param with shape torch.Size([25, 2])
    # from checkpoint, the shape in current model is torch.Size([23, 2]).`
    @pytest.mark.device_agnostic
    def test_wart_random_planckian_jitter_state_dict_is_mode_specific_4428(self):
        blackbody = K.RandomPlanckianJitter(mode="blackbody")
        cied = K.RandomPlanckianJitter(mode="CIED")
        assert "pl" in blackbody.state_dict()
        with pytest.raises(RuntimeError, match="size mismatch for pl"):
            cied.load_state_dict(blackbody.state_dict())
        # The same-mode round trip is fine, so the defect is the shape and not the buffer.
        K.RandomPlanckianJitter(mode="blackbody").load_state_dict(blackbody.state_dict())

    # Row 6c-35: RandomAutoContrast is exactly kornia.enhance.normalize_min_max -- a per-sample,
    # per-channel rescale onto [0, 1] -- checked on a fixture whose six channels have six distinct
    # ranges so a per-batch or per-image rescale would not reproduce it.
    # Snippet used to generate expected:
    #   torch.manual_seed(5)
    #   x = torch.rand(2, 3, 6, 8) * torch.tensor([0.2, 0.5, 1.0]).reshape(1, 3, 1, 1) \
    #       + torch.tensor([0.1, 0.3]).reshape(2, 1, 1, 1)
    #   torch.manual_seed(0)
    #   print((K.RandomAutoContrast(p=1.0)(x) - kornia.enhance.normalize_min_max(x)).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0`, from per-channel input ranges
    # `[(0.106653, 0.298837), (0.164935, 0.586581), (0.118316, 1.08595)]` on the first sample.
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

    # Row 6c-45: the documented parameter bounds are enforced.  Seven of the eight raise at
    # construction; RandomGamma raises from the forward, so every case is constructed and run.
    # The exception type is what the audit recorded, so the type is the pin and the message is not.
    # Snippet used to generate expected:
    #   ctor()(torch.rand(2, 3, 6, 8))  # for each case below
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `ValueError: If sharpness is a single number, it
    # must be non negative.`, `ValueError: bits[0] should be smaller than bits[1]`, `ValueError:
    # bits out of bounds. Expected inside (0, 8)`, `RuntimeError: Gamma must be non-negative.`,
    # `ValueError: brightness out of bounds. Expected inside (0.0, 2.0)`, `ValueError: contrast out
    # of bounds. Expected inside (0, inf)`, `ValueError: saturation out of bounds. Expected inside
    # (0, inf)`, `ValueError: hue out of bounds. Expected inside (-0.5, 0.5)`.
    @pytest.mark.parametrize(
        ("case", "error"),
        [
            ("sharpness_negative", ValueError),
            ("posterize_bits_above_eight", ValueError),
            ("posterize_bits_negative", ValueError),
            ("gamma_negative", RuntimeError),
            ("brightness_above_two", ValueError),
            ("contrast_negative", ValueError),
            ("saturation_negative", ValueError),
            ("hue_above_half", ValueError),
        ],
    )
    def test_convention_intensity_constructors_reject_out_of_bounds(self, device, dtype, case, error):
        if case == "gamma_negative" and device.type == "mps":
            pytest.skip(
                "MPS only: _assert_async_value_check skips the non-negativity check there because "
                "aten::_assert_async has no MPS kernel (kornia/enhance/adjust.py:37-47), so a negative "
                "gamma returns a constant 1 instead of raising"
            )
        factories = {
            "sharpness_negative": lambda: K.RandomSharpness(-1.0, p=1.0),
            "posterize_bits_above_eight": lambda: K.RandomPosterize(bits=9, p=1.0),
            "posterize_bits_negative": lambda: K.RandomPosterize(bits=-1, p=1.0),
            "gamma_negative": lambda: K.RandomGamma((-1.0, -1.0), (1.0, 1.0), p=1.0),
            "brightness_above_two": lambda: K.RandomBrightness((3.0, 3.0), p=1.0),
            "contrast_negative": lambda: K.RandomContrast((-1.0, -1.0), p=1.0),
            "saturation_negative": lambda: K.RandomSaturation((-1.0, -1.0), p=1.0),
            # A scalar `hue` is silently clamped into the bound instead of rejected, so the pin uses
            # the explicit range the audit executed.
            "hue_above_half": lambda: K.RandomHue((0.6, 0.6), p=1.0),
        }
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(error):
            # The sync is what surfaces an MPS kernel error inside this block rather than later.
            _sync(factories[case]()(image).device)

    # Row 6c-46: an unbatched (C, H, W) input is promoted to (1, C, H, W), and `keepdim=True`
    # returns the unbatched shape again.  Checked across four classes of the sub-batch so the claim
    # is the package convention rather than one implementation.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(3, 6, 8)
    #   torch.manual_seed(0); print(ctor(keepdim=False)(x).shape, ctor(keepdim=True)(x).shape)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(1, 3, 6, 8)` and `(3, 6, 8)` for all four.
    @pytest.mark.parametrize("name", ["RandomBrightness", "RandomGrayscale", "RandomSolarize", "RandomPlanckianJitter"])
    def test_convention_intensity_promotes_chw_to_bchw(self, device, dtype, name):
        factories = {
            "RandomBrightness": lambda keepdim: K.RandomBrightness((1.5, 1.5), p=1.0, keepdim=keepdim),
            "RandomGrayscale": lambda keepdim: K.RandomGrayscale(p=1.0, keepdim=keepdim),
            "RandomSolarize": lambda keepdim: K.RandomSolarize(0.1, 0.1, p=1.0, keepdim=keepdim),
            "RandomPlanckianJitter": lambda keepdim: K.RandomPlanckianJitter(p=1.0, keepdim=keepdim),
        }
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert factories[name](False)(image).shape == (1, 3, 6, 8)
        torch.manual_seed(_FORWARD_SEED)
        assert factories[name](True)(image).shape == (3, 6, 8)
