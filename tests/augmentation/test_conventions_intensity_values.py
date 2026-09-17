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
from kornia.augmentation.random_generator import RectangleEraseGenerator
from kornia.core._compat import torch_version_ge
from kornia.core.exceptions import BaseError, ImageError, ShapeError
from kornia.enhance import (
    AdjustBrightnessAccumulative,
    adjust_brightness,
    adjust_brightness_accumulative,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    equalize_clahe,
    normalize_min_max,
    posterize,
)

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


@pytest.fixture(autouse=True)
def _restore_global_rng(restore_torch_rng):
    # Every pin below seeds the global RNG so its draw is reproducible.  ``torch.manual_seed`` also
    # reseeds the CUDA and MPS generators, so the root fixture (#4446) snapshots and restores all of
    # them; ``fork_rng(devices=[])`` would restore the CPU generator only and shift the draw of an
    # unseeded later test on an accelerator leg.
    yield


# Constructor arguments are verbatim from the 6c audit probe, so the executed literals below are
# the ones the fact table records.  Three classes on the intensity page have no row here:
# ``RandomDissolving``, because constructing it can prompt for ``diffusers`` and download a
# Stable-Diffusion checkpoint, and ``RandomClahe`` and ``RandomJPEG``, which are outside
# ``kornia.augmentation.__all__`` and are pinned by their own tests below.
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
    # A range, not the scalar `bits=3`, which is the LOWER bound of [3, 8]: a draw of 8 skips the
    # uint8 round trip and keeps the fixture's range, so the scalar form put this factory outside
    # `_BOUNDED_ON_FIXTURES` on 37 of seeds 0..199 (first at seed 19, `bits_factor [8, 4]`,
    # max 1.9748).  The 8-bit behaviour stays pinned by
    # test_wart_random_posterize_eight_bits_skips_the_round_trip_4430.
    "RandomPosterize": lambda: K.RandomPosterize(bits=(3.0, 7.0), p=1.0),
    "RandomRGBShift": lambda: K.RandomRGBShift(p=1.0),
    "RandomRain": lambda: K.RandomRain(number_of_drops=(3, 3), drop_height=(1, 2), drop_width=(1, 2), p=1.0),
    "RandomSaltAndPepperNoise": lambda: K.RandomSaltAndPepperNoise(p=1.0),
    "RandomSaturation": lambda: K.RandomSaturation((2.0, 2.0), p=1.0),
    "RandomSharpness": lambda: K.RandomSharpness(1.0, p=1.0),
    "RandomSnow": lambda: K.RandomSnow(p=1.0),
    "RandomSolarize": lambda: K.RandomSolarize(0.1, 0.1, p=1.0),
}

# Row 6c-01: four observed outcomes for these constructor arguments, seeds and fixtures (#4430).
# These are not class-wide policies: a different input or configuration can change the group.
# For example, a blur can attenuate an out-of-range impulse into [0, 1], and hue-only ColorJiggle
# can preserve an out-of-range maximum.
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
# Row 6c-04: ``output.clamp(max=1.0)`` only, so a negative input stays negative.
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
# Row 6c-02, in the state #4489 left it: the only audited factory that rejects these fixtures.
_REJECTS_AUDIT_FIXTURES = ("RandomEqualize",)

# Row 6c-03: these nine factories return zeros for the negative fixture at seed 0 (#4430).
# Positive illumination or solarize additions can instead lift negative inputs above zero, and so can
# a ColorJitter contrast or saturation factor above 1 drawn ahead of its brightness step.
# `RandomPosterize` is deliberately not here.  It reaches zero on some platforms only: its
# `(x * 255).to(torch.uint8)` conversion saturates a negative float to code 0 on macOS arm64 with
# torch 2.14.0 (the vectorised path, from 8 elements up) and on MPS, but wraps modulo 256 on Linux
# with torch 2.14.0, on torch 2.5.1 in every dtype and on torch 2.9.1 in the half dtypes, where the
# same input comes back as a full-range posterized image.  The collapse is therefore a property of
# the running conversion kernel, not of kornia; what holds everywhere is pinned by
# test_wart_random_posterize_out_of_range_wraps_4430 instead.
_COLLAPSES_ON_NEGATIVE_FIXTURE = (
    "ColorJitter",
    "RandomContrast",
    "RandomGaussianIllumination",
    "RandomLinearIllumination",
    "RandomLinearCornerIllumination",
    "RandomPlasmaShadow",
    "RandomSnow",
    "RandomSharpness",
    "RandomSolarize",
)

# Fixture seed for the shared out-of-range images, and the seed drawn immediately before each
# construct-and-forward, exactly as the audit probe did it.
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
    # Row 6c-01 (issue #4430): preserve the audit observations for these 35 specific factories
    # and fixtures. Both the group membership and numerical extrema depend on the configuration
    # and input; the counterexample tests below and in the ops file pin those qualifications.
    # Additional mechanism caveats (executed 2026-09-15, torch 2.14.0, cpu):
    #   * `RandomBrightness` and `RandomContrast` are bounded only by their default
    #     `clip_output=True`.  With `clip_output=False` the range is passed through:
    #     `RandomBrightness((1.5, 1.5), clip_output=False)` on `[0, 2]` gives `max=2.496` and
    #     `RandomContrast((1.5, 1.5), clip_output=False)` gives `max=2.99399` (audit row 6c-35,
    #     `R1 RandomBrightness(clip=False) in=[0,2] -> max=2.496`).  The `(clip=False)` variants are
    #     deliberately outside `_INTENSITY_FACTORIES`, as they were outside the audit's count.
    #   * `RandomPosterize`'s `[0, 1]` output is a `uint8` round-trip, not a clamp: the conversion
    #     wraps or saturates depending on the platform and torch version, and the output bears no
    #     relation to the clamped input -- see test_wart_random_posterize_out_of_range_wraps_4430
    #     (audit row 6c-38).  The `[0, 1]` bound holds on every platform for a sample that draws fewer
    #     than 8 bits, because a `uint8` code divided by 255 is in `[0, 1]` however the conversion got
    #     there.  A sample that draws 8 skips the round trip and keeps its range
    #     (test_wart_random_posterize_eight_bits_skips_the_round_trip_4430), so the group membership
    #     rests on this seed's draw of `bits_factor` `[5, 7]`, not on `bits=3` in general.
    #   * `RandomAutoContrast` is a per-sample, per-channel min-max rescale, not a clamp: its
    #     `clip_output` flag is dead because `normalize_min_max` already returns `[0, 1]`
    #     (audit row 6c-35, #4436; pinned by test_convention_random_auto_contrast_is_normalize_min_max
    #     and, for the dead flag, by TestRandomAutoContrast::test_clip_output_does_not_change_the_output).
    # Snippet used to generate expected (re-executed on 90650596):
    #   torch.manual_seed(1234); base = torch.rand(2, 3, 6, 8)
    #   for x in (base * 2.0, base - 1.0):
    #       for name, ctor in mk(): torch.manual_seed(0); print(name, ctor()(x).aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu), e.g. `RandomBrightness in=[0,2] -> min=0.501276
    # max=1`, `RandomSnow in=[0,2] -> min=0.00127578 max=1.996`, `RandomPlanckianJitter
    # in=[-1,0] -> min=-1.41941 max=-0.00200176`.
    @pytest.mark.parametrize("name", sorted(_INTENSITY_FACTORIES))
    def test_convention_value_range_on_audit_fixtures(self, device, dtype, name):
        groups = (_BOUNDED_ON_FIXTURES, _UPPER_BOUNDED_ON_FIXTURES, _OUT_OF_RANGE_ON_FIXTURES, _REJECTS_AUDIT_FIXTURES)
        assert tuple(len(group) for group in groups) == (16, 1, 17, 1)
        assert set().union(*groups) == set(_INTENSITY_FACTORIES) and len(_INTENSITY_FACTORIES) == 35
        if name in _REJECTS_AUDIT_FIXTURES and device.type == "cuda":
            # torch._assert_async on a false condition is a device-side assert on CUDA, which poisons
            # the context for every later test in the process.  On MPS the check is skipped by design
            # and the raw histogram gather raises instead (an AcceleratorError, a RuntimeError subclass).
            pytest.skip("CUDA: the value assert is a device-side assert that invalidates the context")
        if dtype == torch.float16 and name in ("ColorJiggle", "ColorJitter"):
            pytest.skip(
                "float16 only (#4560): the contrast step collapses the [-1, 0] fixture to exact zeros "
                "and rgb_to_hsv's eps=1e-8 then underflows for a black pixel, so adjust_hue returns "
                "NaN (adjust_hue(torch.zeros(1, 3, 2, 2, dtype=torch.float16), 0.1) is already NaN); "
                "float32, float64 and bfloat16 are finite"
            )
        if name in ("RandomBoxBlur", "RandomGaussianBlur") and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        fixtures = _out_of_range_fixtures(device, dtype)
        if name in _REJECTS_AUDIT_FIXTURES:
            # The rejection has to be the VALUE check, not any RuntimeError: on the CPU the message
            # names the range, and on MPS the check is skipped so the raw gather error surfaces (#4600) --
            # from torch 2.14; 2.5.1 and 2.9.1 leave the MPS gather unchecked and return silently.
            if device.type == "mps" and not torch_version_ge(2, 14):
                for image in fixtures.values():
                    out = _run(name, image)
                    assert bool(torch.isfinite(out).all())
                return
            rejection = r"\[0, 1\]" if device.type == "cpu" else "out of bounds"
            for image in fixtures.values():
                with pytest.raises(RuntimeError, match=rejection):
                    _run(name, image)
            return
        # A clamped output is exactly 0.0 or 1.0, and the pass-through outputs sit near 2.0 or
        # -1.0, so a tolerance this loose still separates the groups in bfloat16.
        tol = 1e-3
        outputs = {tag: _run(name, image) for tag, image in fixtures.items()}
        ranges = {tag: out.aminmax() for tag, out in outputs.items()}
        # Both fixtures are already outside [0, 1], so the _OUT_OF_RANGE_ON_FIXTURES branch below is
        # satisfied by the identity: replacing RandomInvert.apply_transform with `return input` left its
        # parametrization green.  Require the class to actually change a fixture, which is what makes "the
        # range is passed through" a claim about the transform rather than about the input.
        # The check is over several draws, not the one _FORWARD_SEED picks, because a neutral draw is
        # legitimate for some classes: RandomChannelShuffle draws the identity permutation about one time
        # in six, and RandomSaltAndPepperNoise's mask can select no pixel on a 6x8 image -- each returned
        # its input on about 11 of seeds 0..199.  An implementation that is the identity moves nothing on
        # any seed, so the weaker claim still catches it.
        moved = False
        for seed in range(8):
            for image in fixtures.values():
                if not torch.equal(_run(name, image, seed=seed), image):
                    moved = True
                    break
            if moved:
                break
        assert moved, f"{name} returned the audit fixtures unchanged on every seed tried"
        if name in _BOUNDED_ON_FIXTURES:
            for tag, (low, high) in ranges.items():
                assert float(low) >= -tol, f"{name} on {tag} left the lower end unclamped"
                assert float(high) <= 1 + tol, f"{name} on {tag} left the upper end unclamped"
            # Reach: the bound is [0, 1], not a narrower interval that would also satisfy the above (a clamp
            # into [0, 0.5] left every bounded class green).  The classes whose draw is fixed by the factory
            # reach 0 on the negative fixture and 1 on [0, 2]; the exceptions are what the draw can do after
            # the clamp, measured over seeds 0..199: ColorJiggle's hue-only draws leave the negative minimum
            # above 0 (24/200) and a brightness factor below 1 pulls the [0, 2] maximum back under 1 for
            # ColorJiggle (54/200) and ColorJitter (53/200); RandomPosterize tops out at 252/255 after the
            # uint8 round trip and RandomSolarize inverts everything above its threshold (200/200 each).
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

    # Row 6c-35, the executable half of the first caveat above: `clip_output` is live on both classes.
    # Snippet used to generate expected:
    #   x = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8) * 2
    #   for cls in (K.RandomBrightness, K.RandomContrast):
    #       print(cls((1.5, 1.5), p=1.0)(x).max(), cls((1.5, 1.5), clip_output=False, p=1.0)(x).max())
    #   torch.manual_seed(1234); neg = torch.rand(2, 3, 6, 8) - 1
    #   for cls in (K.RandomBrightness, K.RandomContrast):
    #       print(cls((1.5, 1.5), clip_output=False, p=1.0)(neg).aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `1 / 2.5` and `1 / 3`; on the negative input the
    # brightness shift of +0.5 gives `min=-0.471 max=0.498` and the contrast scale
    # `min=-1.49904 max=-0.00300264`, both unclamped where the default returns zeros.
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

    # Row 6c-01, the audit exclusions the anchor and conventions.rst name: the factories cover
    # the classes this file executes, and the classes they leave out are exactly the
    # three named there.  RandomDissolving is excluded by method (constructing it downloads a
    # Stable-Diffusion checkpoint, so it is never constructed here), and RandomClahe / RandomJPEG
    # are outside kornia.augmentation.__all__ -- but all three are importable from
    # kornia.augmentation and the last two are rendered on augmentation.intensity.rst, so a new
    # intensity class would falsify the anchor sentence silently without this census.
    # Snippet used to generate expected:
    #   base = K.IntensityAugmentationBase2D
    #   print(sorted(n for n, o in vars(K).items() if isinstance(o, type) and issubclass(o, base)
    #                and o is not base and o.__module__.startswith("kornia.augmentation._2d.intensity")))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> 38 concrete classes, 35 of them the keys of
    # _INTENSITY_FACTORIES and the other three `['RandomClahe', 'RandomDissolving', 'RandomJPEG']`.
    @pytest.mark.device_agnostic
    def test_convention_intensity_split_carve_outs_are_exactly_three(self):
        base = K.IntensityAugmentationBase2D
        concrete = {
            name
            for name, obj in vars(K).items()
            if isinstance(obj, type)
            and issubclass(obj, base)
            and obj is not base
            and obj.__module__.startswith("kornia.augmentation._2d.intensity")
        }
        assert len(concrete) == 38
        assert concrete - set(_INTENSITY_FACTORIES) == {"RandomClahe", "RandomDissolving", "RandomJPEG"}
        assert set(_INTENSITY_FACTORIES) - concrete == set()

    # Row 6c-03 (issue #4430): nine factories return zeros on this negative fixture at seed 0.
    # This is not a guarantee for other draws, and the seed is therefore pinned as the literal `0`
    # rather than read from `_FORWARD_SEED`: five of these nine depend on the draw, and overriding
    # `_FORWARD_SEED` across seeds 0..199 fails on 145 of them for RandomLinearCornerIllumination,
    # 143 for RandomSolarize and RandomLinearIllumination, 136 for RandomGaussianIllumination and 99
    # for ColorJitter.  The literal keeps the audit record honest -- and keeps a future reseed of the
    # file from turning this row red for a reason that is not a regression.
    # #4430 offers two outcomes (clamp, or reject as RandomEqualize now does), so this is a wart pin on
    # today's behavior, not a strict xfail on a settled contract.
    # These nine collapse through their own arithmetic, not through a dtype conversion, so the
    # result is the same on every platform, torch version and dtype (`RandomPosterize`, whose
    # collapse is a `uint8`-conversion artifact of one platform, is excluded above).
    # Snippet used to generate expected: the r1 snippet above, reading the `in=[-1,0]` rows;
    # executed 2026-09-15 on this worktree under torch 2.14.0 and under torch 2.5.1 (macOS arm64,
    # cpu) -- all nine print `min=0 max=0` in both runs.
    @pytest.mark.parametrize("name", _COLLAPSES_ON_NEGATIVE_FIXTURE)
    def test_wart_intensity_negative_input_collapses_to_zero_4430(self, device, dtype, name):
        if dtype == torch.float16 and name == "ColorJitter":
            pytest.skip(
                "float16 only (#4560): the collapsed image reaches adjust_hue, whose rgb_to_hsv gives a "
                "NaN saturation for a black pixel there (eps=1e-8 underflows in float16)"
            )
        image = _out_of_range_fixtures(device, dtype)["[-1, 0]"]
        assert float(_run(name, image, seed=0).abs().max()) == 0.0

    # The anchor's census (issue #4430): on a constant `-1.0` image at the audited constructor arguments,
    # 17 factories collapse, 15 of them on every one of seeds 0..4 and only ColorJiggle and
    # RandomPlasmaContrast on some.  Two of the 15 are artefacts of the constant image rather than of
    # its sign -- Denormalize maps -1 to exactly 0 at mean=std=0.5 and RandomAutoContrast returns zeros
    # for any constant image -- so neither collapses on the `[-1, 0)` audit fixture, where the draw
    # decides for 7 of the 12 that do.  The seeds are literal so a reseed of the file cannot turn the
    # census red for a reason that is not a regression.
    # Snippet used to generate expected:
    #   img = torch.full((2, 3, 6, 8), -1.0)
    #   for name, f in sorted(_INTENSITY_FACTORIES.items()):
    #       n = 0
    #       for s in range(5):
    #           torch.manual_seed(s)
    #           try: n += float(f()(img).abs().max()) == 0.0
    #           except RuntimeError: pass
    #       print(name, n)
    # executed 2026-09-16 (torch 2.14.0, cpu, float32) -> 5 for the fifteen names below, 2 for
    # ColorJiggle, 1 for RandomPlasmaContrast, 0 for the rest; RandomEqualize raises.
    _COLLAPSES_ON_CONSTANT_MINUS_ONE = (
        "ColorJitter",
        "Denormalize",
        "RandomAutoContrast",
        "RandomBrightness",
        "RandomContrast",
        "RandomGaussianIllumination",
        "RandomLinearCornerIllumination",
        "RandomLinearIllumination",
        "RandomPlasmaBrightness",
        "RandomPlasmaShadow",
        "RandomPosterize",
        "RandomRGBShift",
        "RandomSharpness",
        "RandomSnow",
        "RandomSolarize",
    )

    def test_wart_intensity_constant_negative_collapse_census_4430(self, device, dtype):
        image = torch.full((2, 3, 6, 8), -1.0, device=device, dtype=dtype)
        every, some = [], []
        for name in sorted(_INTENSITY_FACTORIES):
            if name == "RandomEqualize" or (dtype == torch.float16 and name in ("ColorJitter", "RandomSnow")):
                continue  # raises out of range; float16 ColorJitter (#4560) and RandomSnow (#4571) are NaN there
            if name in ("RandomBoxBlur", "RandomGaussianBlur") and not supports_reflect_padding(device, dtype):
                continue  # torch 2.5.1 has no half reflection_pad2d on the CPU
            zeros = sum(float(_run(name, image, seed=seed).abs().max()) == 0.0 for seed in range(5))
            (every if zeros == 5 else some if zeros else []).append(name)
        expected = set(self._COLLAPSES_ON_CONSTANT_MINUS_ONE)
        if dtype == torch.float16:
            expected -= {"ColorJitter", "RandomSnow"}
        assert set(every) == expected
        assert set(some) <= {"ColorJiggle", "RandomPlasmaContrast"}
        # The two constant-image artefacts do not collapse the non-constant audit fixture on any of 20 seeds.
        fixture = _out_of_range_fixtures(device, dtype)["[-1, 0]"]
        for name in ("Denormalize", "RandomAutoContrast"):
            assert all(float(_run(name, fixture, seed=seed).abs().max()) > 0.0 for seed in range(20))
        assert float(_run("RandomAutoContrast", torch.full_like(image, 0.5)).abs().max()) == 0.0
        self.assert_close(_run("Denormalize", torch.full_like(image, -0.5)), torch.full_like(image, 0.25))

    # Row 6c-38 (issue #4430): RandomPosterize does not clamp an out-of-range input.  It posterizes
    # the `uint8` round-trip of the raw float, so the output is the posterization of whatever codes
    # that conversion produced and bears no relation to the clamped input.
    # What `(x * 255).to(torch.uint8)` does with a value outside `[0, 255]` is a property of the
    # running kernel, not of kornia, and the platforms disagree: it wraps modulo 256 on Linux with
    # torch 2.14.0 and on torch 2.5.1/2.9.1, saturates negatives to code 0 on macOS arm64 with
    # torch 2.14.0 (the vectorised path, from 8 elements up; measured on
    # `(torch.linspace(-1, -0.004, n) * 255).to(torch.uint8)`, n = 7 gives 7 distinct codes and
    # n = 8 gives 1), and saturates both ends on MPS.  An earlier revision of this pin asserted the
    # macOS 2.14 collapse (`len(out.unique()) == 1`) and failed on five CI legs; the round-trip
    # identity below is what holds on all of them, so the platform-dependent literals are gone.
    # Snippet used to generate expected:
    #   print((torch.linspace(-1, 0, 48) * 255).to(torch.uint8).unique().numel())  # kernel probe
    #   g = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32)
    #   for x in (g * 2.0, g - 1.0):
    #       torch.manual_seed(0); y = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(x)
    #       print(len(y.unique()), y.min().item(), y.max().item())
    # executed 2026-09-15 on this worktree (macOS arm64, cpu, float32): probe `1`, then
    # `8 0.0 0.8784313797950745` and `1 0.0 0.0` under torch 2.14.0; probe `48`, then
    # `8 0.0 0.8784313797950745` and `8 0.0 0.8784313797950745` under torch 2.5.1.  The probe tells
    # a reader which of the two behaviours the kernel in front of them has; only the `[0, 2]`
    # literals, which agree in both runs, are asserted.
    @pytest.mark.parametrize("scale", ["[0, 2]", "[-1, 0]"])
    def test_wart_random_posterize_out_of_range_wraps_4430(self, device, dtype, scale):
        ramp = (torch.arange(256, dtype=torch.float32) / 255.0).reshape(1, 1, 8, 32).to(device=device, dtype=dtype)
        image = ramp * 2.0 if scale == "[0, 2]" else ramp - 1.0
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(3.0, 3.0), p=1.0)(image)
        _sync(image.device)
        # `bits_factor` is `tensor([3], dtype=torch.int32)` for `bits=(3.0, 3.0)`, and `posterize`
        # takes the same one-element path for a plain `int`.  `codes` reproduces the first of the
        # two `uint8` conversions inside `posterize` (kornia/enhance/adjust.py::_right_shift) --
        # the only one that ever sees an out-of-range value, since the second is handed a
        # non-negative `codes / 2 ** shift <= 255`.
        codes = (image * 255).to(torch.uint8)
        expected = posterize(codes.to(dtype) / 255.0, 3)
        clamped = posterize(image.clamp(0.0, 1.0), 3)
        assert torch.equal(out, expected)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        # A clamp-first implementation would return `clamped`; spelled out so the intent survives
        # an edit to `expected`.  The guard is a property of the kernel and the fixture, not of
        # `expected`: a wrapping conversion sends the out-of-range half of the ramp to codes the
        # clamped input cannot produce, while a saturating one sends them to exactly the clamped
        # input's codes, which makes the two implementations numerically identical and leaves
        # nothing to assert.
        if not torch.equal(codes, (image.clamp(0.0, 1.0) * 255).to(torch.uint8)):
            assert not torch.equal(out, clamped)
        if scale == "[0, 2]":
            assert len(out.unique()) == 8
            self.assert_close(out.max(), out.new_tensor(224 / 255))

    # Issue #4430, the other posterize path: kornia.enhance.posterize returns a sample that draws 8 bits
    # unchanged (`torch.where(bits == 8, input, out)`), with no `uint8` round trip to bound it, so its
    # out-of-range values survive.  The int form draws 8 for part of the batch -- its lower-bound pin,
    # test_convention_random_posterize_int_argument_is_a_lower_bound, reaches 8.
    # Snippet used to generate expected:
    #   print(K.RandomPosterize(bits=(8.0, 8.0), p=1.0)(torch.tensor([1.7, -0.4, 2.5, 0.3]).reshape(1, 1, 1, 4)))
    #   torch.manual_seed(0)
    #   print((K.RandomPosterize(bits=3, p=1.0).forward_parameters((100000, 1, 2, 2))["bits_factor"] == 8).sum())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> the input unchanged, and `9986` of the 100000 draws.
    def test_wart_random_posterize_eight_bits_skips_the_round_trip_4430(self, device, dtype):
        image = torch.tensor([1.7, -0.4, 2.5, 0.3], device=device, dtype=dtype).reshape(1, 1, 1, 4)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPosterize(bits=(8.0, 8.0), p=1.0)(image)
        _sync(image.device)
        assert torch.equal(out, image)

    # Row 6c-05 (issue #4430), the failure the grouping run cannot see: RandomGamma is in
    # _BOUNDED_ON_FIXTURES on the strength of `gamma=2.0`, but on a negative input `x ** gamma` is NaN
    # for every non-integer gamma and the clamp propagates it.  Only an integer gamma is finite, and
    # of those only an even one keeps a non-zero value -- an odd power keeps the sign and the clamp
    # floors it, which is the all-zero collapse #4430 is about.  #4430 leaves two coherent outcomes
    # open (clamp, or reject), so this is a wart pin on today's behavior, not a strict xfail.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); neg = torch.rand(2, 3, 6, 8) - 1.0
    #   for gamma in (0.5, 1.0, 1.5, 2.0, 3.0):
    #       torch.manual_seed(0); y = K.RandomGamma((gamma, gamma), (1.0, 1.0), p=1.0)(neg)
    #       print(gamma, bool(y.isnan().any()), bool(y.isnan().all()), y.nan_to_num().abs().max())
    # executed 2026-09-15 (torch 2.14.0) on cpu float32/float64/float16/bfloat16 and mps float32,
    # identical on every leg -> `0.5` and `1.5` NaN in every element; `1.0`, `2.0` and `3.0` finite,
    # with max|out| `0` at gamma 1.0 and 3.0 and `0.998725` at gamma 2.0 (`0.999023` in float16,
    # `1` in bfloat16, so only the inequality is asserted).  The fixture has no exact zero
    # (its maximum is -0.00200176 in float32), so the NaN is every element, not almost every one.
    @pytest.mark.parametrize(("gamma", "nan_expected"), [(0.5, True), (1.5, True), (2.0, False), (3.0, False)])
    def test_wart_random_gamma_negative_input_is_nan_4430(self, device, dtype, gamma, nan_expected):
        image = _out_of_range_fixtures(device, dtype)["[-1, 0]"]
        assert float(image.max()) < 0.0, "the fixture must be strictly negative for the power to be NaN"
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomGamma((gamma, gamma), (1.0, 1.0), p=1.0)(image)
        _sync(image.device)
        if nan_expected:
            assert bool(out.isnan().all()), f"gamma={gamma} on a negative input should be NaN throughout"
        else:
            assert bool(out.isfinite().all()), f"an integer gamma={gamma} should stay finite"
            # The even power is the control: it is the one integer gamma that survives the clamp,
            # and it is the value `_INTENSITY_FACTORIES` uses -- which is what hides the NaN there.
            non_zero = float(out.abs().max()) > 0.0
            assert non_zero is (gamma == 2.0), f"gamma={gamma} gave max|out| {float(out.abs().max())}"

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
    # 8-bit data.`, the first two return a 64-level image.  MPS skips the value check, and the same two
    # inputs raise the raw `gather: index ... is out of bounds` there (an AcceleratorError, which is a
    # RuntimeError).  The guard compares `input * 255` in the input's dtype, so `1.00390625` (1 + 1/256,
    # under one code above 1) is admitted in float32 and float64 but rounds onto 256 and raises in float16.
    def test_convention_random_equalize_rejects_out_of_range_with_named_range(self, device, dtype):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value assert is a device-side assert that poisons the context")
        # Only kornia's own check names the range; MPS skips it and surfaces torch's gather error from
        # torch 2.14 (2.5.1 and 2.9.1 leave the MPS gather unchecked and return the image silently).
        match = r"\[0, 1\]" if device.type == "cpu" else "out of bounds"
        ramp = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).to(device=device, dtype=dtype)
        for image in (ramp * 2.0, ramp - 1.0):
            torch.manual_seed(_FORWARD_SEED)
            if device.type == "mps" and not torch_version_ge(2, 14):
                assert bool(torch.isfinite(K.RandomEqualize(p=1.0)(image)).all())
                continue
            with pytest.raises(RuntimeError, match=match):
                _sync(K.RandomEqualize(p=1.0)(image).device)
        # Less than one 8-bit code outside the range is admitted at either end: `ramp - 0.003` has
        # `input * 255 > -1` everywhere, so its lookup index truncates to 0.
        for image in (ramp, ramp * 1.0001, ramp - 0.003):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomEqualize(p=1.0)(image)
            assert out.shape == image.shape
            assert torch.isfinite(out).all()
        # bfloat16 cannot represent 1 + 1/256 at all, so that dtype has no row.
        if dtype in (torch.float16, torch.float32, torch.float64):
            edge = ramp.clone()
            edge[0, 0, 0, 0] = 1.00390625
            torch.manual_seed(_FORWARD_SEED)
            if dtype == torch.float16:
                with pytest.raises(RuntimeError, match=match):
                    _sync(K.RandomEqualize(p=1.0)(edge).device)
            else:
                assert K.RandomEqualize(p=1.0)(edge).shape == edge.shape

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
    #   print(torch.equal(y, v), v.grad.flatten().tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu float16/bfloat16/float32/float64 and mps float32) -> both
    # classes raise (`equalize expects input values in [0, 1]`, `index ... is out of bounds`), then
    # `True [nan, 1.0, 1.0, nan, 1.0, 1.0]`.
    def test_wart_p_gate_computes_the_skipped_samples_4576(self, device, dtype):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the value asserts are device-side asserts that poison the context")
        ramp = torch.linspace(0, 1, 1024).reshape(1, 1, 32, 32)
        image = torch.cat([ramp, ramp * 2.0]).to(device=device, dtype=dtype)
        for cls in (K.RandomEqualize, K.RandomClahe):
            torch.manual_seed(_FORWARD_SEED)
            # It has to be the value check that fires, not merely some RuntimeError: the claim is that
            # the transform ran on a sample `p=0.0` was supposed to skip.
            gate_rejection = r"\[0, 1\]|out of bounds" if device.type == "cpu" else "out of bounds"
            if device.type == "mps" and not torch_version_ge(2, 14):
                # The transform still runs on the skipped samples, but 2.5.1 leaves the MPS gather
                # unchecked, so the out-of-range rows come back silently instead of raising.
                assert bool(torch.isfinite(cls(p=0.0)(image)).all())
                continue
            with pytest.raises(RuntimeError, match=gate_rejection):
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

    @pytest.mark.parametrize(("gain", "expected"), [(1.0, [1.0, 1.0, 1.0]), (0.1, [0.5, 0.2, 0.125])])
    def test_convention_random_gamma_negative_exponent_on_mps_depends_on_gain(self, device, dtype, gain, expected):
        if device.type != "mps":
            pytest.skip("MPS only: CPU rejects negative gamma; CUDA would invalidate the context with a device assert")
        # MPS skips the value check. At gamma=-1 the output is clamp(gain / input, 0, 1):
        # gain=1 saturates every pixel here, while gain=0.1 preserves three distinct values.
        image = torch.tensor([0.2, 0.5, 0.8], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        out = K.RandomGamma((-1.0, -1.0), (gain, gain), p=1.0)(image)
        self.assert_close(out, image.new_tensor(expected).reshape_as(image))

    def test_convention_random_saturation_clamps_hsv_without_a_final_rgb_clamp(self, device, dtype):
        # Clamping HSV saturation removes the first pixel's negative RGB channel even at factor=1, and
        # turns the all-negative third pixel gray at its largest channel (its saturation is negative).
        # The second and fourth pixels, above 1 with no negative channel, keep their values, which a final
        # RGB clamp would remove.  The second one's zero hue avoids amplifying half-precision hue
        # round-trip error at that larger value; the fourth comes back exact in float16 and bfloat16 too.
        pixels = [[-0.1, 0.5, 0.5], [1.5, 0.0, 0.0], [-0.1, -0.5, -0.9], [2.0, 1.0, 0.5]]
        image = torch.tensor(pixels, device=device, dtype=dtype).reshape(4, 3, 1, 1)
        out = K.RandomSaturation((1.0, 1.0), p=1.0)(image)
        expected = [[0.0, 0.5, 0.5], [1.5, 0.0, 0.0], [-0.1, -0.1, -0.1], [2.0, 1.0, 0.5]]
        self.assert_close(out, image.new_tensor(expected).reshape_as(image))

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

    # Rows 6c-07 and 6c-08: with matching effective ranges and default CPU float32 samplers, these
    # configurations draw byte-identical parameters from the same seed, including the random `order`.
    # The selected non-identity factors expose the different adjustment primitives.
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
        # Replaying ColorJitter with ColorJiggle's own parameters reproduces ColorJitter's output,
        # so the gap for this configuration is in application, not in the draw.
        replayed = jitter(image, params=jiggle._params)
        self.assert_close(replayed, jittered)
        assert float((jiggled - replayed).abs().max()) > 0.05

    @pytest.mark.device_agnostic
    def test_convention_color_jiggle_and_jitter_have_different_brightness_bounds(self):
        # A scalar brightness above 1 overshoots ColorJiggle's [0, 2] bound and is rejected there, while
        # ColorJitter, bounded by (0, inf), reads it as [0, 1 + brightness].
        with pytest.raises(ValueError, match="brightness out of bounds"):
            K.ColorJiggle(brightness=1.5)
        # `(jitter > 2).any()` needs a batch wide enough to be certain: one draw clears 2 with probability
        # 0.2, so at the `B = 4` an earlier revision of this pin used it failed on 13 of seeds 0..199, and
        # this literal seed is outside the `_FORWARD_SEED` sweep that would have caught it.  At `B = 256`
        # the failure probability is `0.8 ** 256`, about `1e-25`.
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
        # The constant fixture is achromatic, so the rotation is the identity on it whatever `hue` is
        # and only the missing clamp is exercised above.  A chromatic pixel makes the second half of
        # the claim -- that the rotation preserves the out-of-range value -- discriminating.
        # Snippet used to generate expected:
        #   torch.manual_seed(0)
        #   K.ColorJiggle(0.0, 0.0, 0.0, (0.1, 0.1), p=1.0)(torch.tensor([2.0, 1.5, 0.5]).reshape(1, 3, 1, 1))
        # executed 2026-09-16 (torch 2.14.0, cpu) -> `[1.5999999, 2.0, 0.5]`, whose maximum is the
        # input's `2.0`.
        chroma = torch.tensor([2.0, 1.5, 0.5], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        torch.manual_seed(_FORWARD_SEED)
        rotated = K.ColorJiggle(0.0, 0.0, 0.0, hue, p=1.0)(chroma)
        self.assert_close(rotated.amax(1), chroma.amax(1))
        assert torch.equal(rotated, chroma) is (hue == 0.0)

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

    # Issue #4430, the ColorJiggle half: which steps run decides whether the collapse is draw-dependent.
    # The all-zero default is the identity; contrast-only collapses on every draw; brightness-only and the
    # audit's four-factor configuration on some; hue-only never.  Literal seeds, as for the census above.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); neg = torch.rand(2, 3, 6, 8) - 1.0
    #   for args in ((0, 0.3, 0, 0), (0.3, 0, 0, 0), (0.2, 0.2, 0.2, 0.1), (0, 0, 0, 0.1)):
    #       n = 0
    #       for s in range(20):
    #           torch.manual_seed(s); n += float(K.ColorJiggle(*args, p=1.0)(neg).abs().max()) == 0.0
    #       print(args, n)
    # executed 2026-09-16 (torch 2.14.0, cpu, float32) -> 20, 7, 7, 0; `ColorJiggle(p=1.0)(neg)` equals `neg`.
    @pytest.mark.parametrize(
        ("args", "lo", "hi"),
        [
            ((0.0, 0.3, 0.0, 0.0), 20, 20),
            ((0.3, 0.0, 0.0, 0.0), 1, 19),
            ((0.2, 0.2, 0.2, 0.1), 1, 19),
            ((0.0, 0.0, 0.0, 0.1), 0, 0),
        ],
    )
    def test_convention_color_jiggle_negative_collapse_depends_on_which_steps_run(self, device, dtype, args, lo, hi):
        if dtype == torch.float16 and args[3] > 0.0:
            pytest.skip("float16 only (#4560): the hue step returns NaN for a black pixel")
        torch.manual_seed(_FIXTURE_SEED)
        image = (torch.rand(2, 3, 6, 8) - 1.0).to(device=device, dtype=dtype)
        assert torch.equal(K.ColorJiggle(p=1.0)(image), image)
        zeros = 0
        for seed in range(20):
            torch.manual_seed(seed)
            zeros += float(K.ColorJiggle(*args, p=1.0)(image).abs().max()) == 0.0
        assert lo <= zeros <= hi

    # Issue #4430, the ColorJitter half: the all-negative collapse depends on the drawn order.  The
    # brightness step, which the default scalar brightness runs, clamps what is still negative to zero,
    # but a contrast or saturation factor above 1 applied before it lifts part of the image first -- with
    # the random order the audit configuration ColorJitter(0.2, 0.2, 0.2, 0.1) keeps non-zero values for
    # 18 of seeds 0..39 on the audit fixture.  A fixed `order` makes both outcomes deterministic, and one
    # without index 0 skips the brightness clamp, so the default factors pass an out-of-range value.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); neg = torch.rand(1, 3, 8, 8) - 1.0
    #   for step, kw in ((1, dict(contrast=(1.9, 1.9))), (2, dict(saturation=(1.9, 1.9)))):
    #       for order in ((0, step), (step, 0)):
    #           print(order, K.ColorJitter(p=1.0, order=order, **kw)(neg).aminmax())
    #   print(K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0, order=(1, 2, 3))(torch.full((1, 3, 2, 2), 2.0)).unique())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> brightness first `0 / 0` for both; contrast first
    # `0 / 0.453` and saturation first `0 / 0.669` (0.449 and 0.664 in bfloat16); `[2.0]`.
    @pytest.mark.parametrize(("step", "kwargs"), [(1, {"contrast": (1.9, 1.9)}), (2, {"saturation": (1.9, 1.9)})])
    def test_convention_color_jitter_negative_collapse_depends_on_the_order(self, device, dtype, step, kwargs):
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

    # Row 6c-08, the order and channel-count details:
    # - a fixed constructor `order` ignores a forward `order=` tensor;
    # - ColorJiggle has no constructor order, and a forward `order=` tensor replaces the drawn one, so
    #   leaving the hue step out lets a one-channel image through;
    # - ColorJitter's saturation step (adjust_saturation_with_gray_subtraction) rejects four channels,
    #   clamps a three-channel image, and returns a one-channel image unchanged.
    # Snippet used to generate expected:
    #   x = torch.rand(2, 3, 4, 4); torch.manual_seed(0)
    #   a = K.ColorJitter(brightness=(0.5, 0.5), order=(1,), p=1.0)
    #   print(torch.equal(a(x, order=torch.tensor([0])), a(x)))
    #   print(K.ColorJiggle(hue=0.1, p=1.0)(torch.rand(2, 1, 4, 4), order=torch.tensor([0, 1, 2])).shape)
    #   s = K.ColorJitter(saturation=(0.5, 0.5), order=(2,), p=1.0)
    #   print(s(torch.full((2, 1, 3, 3), 2.0)).unique(), s(torch.full((2, 3, 3, 3), 2.0)).unique())
    #   s(torch.rand(2, 4, 3, 3))
    # executed 2026-09-15 (torch 2.14.0, cpu float16/bfloat16/float32/float64 and mps float32) -> `True`,
    # `(2, 1, 4, 4)` (without `order=`: `ValueError: Input size must have a shape of (*, 3, H, W)`),
    # `[2.]`, `[1.]`, and `ImageError: Not a color or gray tensor.`
    def test_convention_color_jitter_order_override_and_channel_counts(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4).to(device=device, dtype=dtype)
        fixed = K.ColorJitter(brightness=(0.5, 0.5), order=(1,), p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        assert torch.equal(fixed(image, order=torch.tensor([0])), fixed(image))
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

    # `adjust_brightness_accumulative`, ColorJitter's brightness primitive, multiplies by the factor and
    # clamps by default, so its identity factor `1` holds only for an image in [0, 1]; the module form
    # AdjustBrightnessAccumulative always clamps.
    # Snippet used to generate expected:
    #   x = torch.tensor([-0.5, 0.5, 2.0]).reshape(1, 3, 1, 1)
    #   print(adjust_brightness_accumulative(x, 1.0).flatten(), AdjustBrightnessAccumulative(1.0)(x).flatten())
    # executed 2026-09-15 (torch 2.14.0, cpu float16/bfloat16/float32/float64 and mps float32) -> `[0, 0.5, 1]`
    # twice.
    def test_convention_adjust_brightness_accumulative_one_is_identity_only_in_range(self, device, dtype):
        values = torch.tensor([-0.5, 0.5, 2.0], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        expected = values.clamp(0.0, 1.0)
        self.assert_close(adjust_brightness_accumulative(values, 1.0), expected)
        self.assert_close(AdjustBrightnessAccumulative(1.0)(values), expected)
        self.assert_close(adjust_brightness_accumulative(values, 1.0, clip_output=False), values)
        self.assert_close(adjust_brightness_accumulative(values, 0.0), torch.zeros_like(values))

    # Row 6c-10: RandomHue's argument is turns of the hue circle, restricted to the closed [-0.5, 0.5];
    # the class multiplies it by 2*pi before calling kornia.enhance.adjust_hue, which takes radians.
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
        # The [-0.5, 0.5] bound that goes with those units is pinned from outside in
        # test_convention_intensity_constructors_reject_out_of_bounds.

    # Row 6c-10, the range half: RandomHue has no clamp, and a hue rotation keeps each pixel's largest and
    # smallest channel values, so a pixel outside [0, 1] keeps a channel outside it.  The exception is a
    # pixel whose largest channel is exactly 0, whose HSV saturation divides by that zero: the round trip
    # returns zeros, and NaN in float16 (the #4560 underflow, which reaches a zero-max pixel that is not
    # black as well).
    # Snippet used to generate expected:
    #   x = torch.tensor([[1.5, 0.2, 0.3], [0.5, -0.1, 0.2], [-0.2, -0.5, -0.9], [0.0, -0.5, -0.5]])
    #   torch.manual_seed(0); print(K.RandomHue((0.25, 0.25), p=1.0)(x.reshape(4, 3, 1, 1)).reshape(4, 3))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `[[0.95, 1.5, 0.2], [0.5, 0.5, -0.1], [-0.9, -0.2, -0.85],
    # [-0, 0, -0]]`.
    def test_convention_random_hue_keeps_out_of_range_extremes(self, device, dtype):
        pixels = torch.tensor([[1.5, 0.2, 0.3], [0.5, -0.1, 0.2], [-0.2, -0.5, -0.9]], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomHue((0.25, 0.25), p=1.0)(pixels.reshape(3, 3, 1, 1)).reshape(3, 3)
        self.assert_close(out.amax(1), pixels.amax(1))
        self.assert_close(out.amin(1), pixels.amin(1))
        zero_max = torch.tensor([0.0, -0.5, -0.5], device=device, dtype=dtype).reshape(1, 3, 1, 1)
        torch.manual_seed(_FORWARD_SEED)
        collapsed = K.RandomHue((0.25, 0.25), p=1.0)(zero_max)
        if dtype == torch.float16:
            assert bool(collapsed.isnan().any())
        else:
            self.assert_close(collapsed, torch.zeros_like(collapsed))

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
        # Channel mixing can put out-of-range input back in range without clamping it.
        reduced = K.RandomGrayscale(p=1.0)(2 * red)
        self.assert_close(reduced, torch.full_like(red, 0.598))
        # The weights are applied AS GIVEN.  Every weights vector above sums to one, so an
        # implementation that normalized them would pass; these two do not sum to one.
        # Snippet used to generate expected:
        #   torch.manual_seed(0); half = torch.full((1, 3, 2, 2), 0.5)
        #   print(K.RandomGrayscale(rgb_weights=torch.ones(3), p=1.0)(half)[0, 0, 0, 0])
        #   torch.manual_seed(0); print(K.RandomGrayscale(p=1.0)(torch.full((1, 4, 2, 2), 0.5))[0, 0, 0, 0])
        # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> `1.5`; the 4-channel branch
        # averages instead and gives `0.5`, so `torch.ones(4)` there gives `2.0`.
        half = torch.full((1, 3, 2, 2), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        unnormalized = K.RandomGrayscale(rgb_weights=torch.ones(3, device=device, dtype=dtype), p=1.0)(half)
        self.assert_close(unnormalized, torch.full_like(half, 1.5))
        wide = torch.full((1, 4, 2, 2), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        wide_out = K.RandomGrayscale(rgb_weights=torch.ones(4, device=device, dtype=dtype), p=1.0)(wide)
        self.assert_close(wide_out, torch.full_like(wide, 2.0))

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
        assert not bool((order == order[0]).all()), "the seeded draw is one permutation for every sample"
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
        # also pass for an implementation that dropped the same channels across the whole batch.  Two
        # samples coincide by chance with probability 1/3 (n=1) or 1/6 (n=2) -- this assertion failed on
        # 64 of seeds 0..199 at B=2 -- so the distinctness is checked over a wider batch instead.
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

    # Row 6c-40: RandomInvert is `max_val - x`, with no clamp, so an input outside [0, 1] comes
    # back subtracted from `max_val` rather than clipped.  That is a reflection about `max_val / 2`,
    # not about `max_val`: `invert(torch.tensor([2.0]), torch.tensor(1.0))` is `-1.0`, where a
    # reflection around `max_val` would give `0.0` (executed 2026-09-15, torch 2.14.0, cpu).
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
    # only then is everything at or above `thresholds` inverted.  The order matters for the pixels
    # the inversion touches, not for the dark ones: adding after the inversion would give `1 - g + a`
    # where adding first gives `1 - g - a`, so it is the inverted (bright) pixels that move the other
    # way, by `2 * addition`.  Every pixel more than `addition` below the threshold is identical
    # under both orders, and only the band `[threshold - addition, threshold)` changes bucket.
    # Snippet used to generate expected:
    #   g = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8)
    #   torch.manual_seed(0); y = K.RandomSolarize((t, t), (a, a), p=1.0)(g)
    #   z = (g + a).clamp(0, 1); print((y - torch.where(z < t, z, 1.0 - z)).abs().max(), y.aminmax())
    #   w = (torch.where(g < t, g, 1.0 - g) + a).clamp(0, 1); print((y - w).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> formula diff `0` for every pair below, and
    # threshold=0.5 addition=0.25 sends the input 0.0212766 to 0.271277.  Against the other order at
    # that pair, max|diff| is `0.5` (= 2 * addition), reached on every pixel at or above the
    # threshold; the two agree exactly below `t - a = 0.25` and the first differing element is
    # g=0.255319, which is in the band that changes bucket.
    @pytest.mark.parametrize(("threshold", "addition"), [(0.5, 0.0), (0.5, 0.25), (0.0, 0.0)])
    def test_convention_random_solarize_adds_before_inverting(self, device, dtype, threshold, addition):
        ramp = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSolarize((threshold, threshold), (addition, addition), p=1.0)(ramp)
        shifted = (ramp + addition).clamp(0.0, 1.0)
        self.assert_close(out, torch.where(shifted < threshold, shifted, 1.0 - shifted))
        if (threshold, addition) == (0.5, 0.25):
            self.assert_close(out.flatten()[1], out.new_tensor(1.0 / 47.0 + 0.25))

    @pytest.mark.parametrize(("addition", "expected"), [(-0.1, 0.0), (0.1, 0.099)])
    def test_convention_random_solarize_negative_input_depends_on_addition(self, device, dtype, addition, expected):
        # The seed-0 audit collapse is conditional: a positive addition can lift a negative pixel above zero.
        image = torch.full((1, 3, 6, 8), -0.001, device=device, dtype=dtype)
        out = K.RandomSolarize(thresholds=(0.5, 0.5), additions=(addition, addition), p=1.0)(image)
        self.assert_close(out, torch.full_like(image, expected))

    @pytest.mark.parametrize(("addition", "expected"), [(-0.4, 0.3), (0.0, 0.0)])
    def test_convention_random_solarize_above_one_depends_on_addition(self, device, dtype, addition, expected):
        # Being above 1 is insufficient for unconditional collapse: a negative addition can bring
        # the value below 1 before inversion. At 1.1 - 0.4, the result is 1 - 0.7 = 0.3.
        image = torch.full((1, 3, 6, 8), 1.1, device=device, dtype=dtype)
        out = K.RandomSolarize(thresholds=(0.5, 0.5), additions=(addition, addition), p=1.0)(image)
        self.assert_close(out, torch.full_like(image, expected))

    # The #4430 collapse has an upper-end twin that the audit fixtures cannot see: `[0, 2]` is not
    # entirely above 1, so `RandomSolarize(p=1.0)` on it returns 151 distinct values.  An input drawn
    # from `[1.5, 3.0]` is all-zero on EVERY draw, because `clamp(x + a, 0, 1)` sends every `x >= 1.5`
    # to exactly `1.0` for any admissible `a in [-0.5, 0.5]` and `1 - 1.0 == 0`.  Swept over the 35
    # audited factories, `RandomSolarize` is the only one that does this.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.empty(2, 3, 6, 8).uniform_(1.5, 3.0)
    #   for seed in range(5): torch.manual_seed(seed); print((K.RandomSolarize(p=1.0)(x) == 0).all())
    # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> `True` on all five seeds for
    # RandomSolarize and on none of the other 34 factories.
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_wart_random_solarize_at_least_one_point_five_collapses_on_every_draw_4430(self, device, dtype, seed):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.empty(2, 3, 6, 8).uniform_(1.5, 3.0).to(device=device, dtype=dtype)
        torch.manual_seed(seed)
        assert bool((K.RandomSolarize(p=1.0)(image) == 0).all())
        # The audited `[0, 2]` fixture straddles the bound, so it does NOT collapse -- which is why
        # nothing in the row-6c-01 table sees this.
        torch.manual_seed(_FIXTURE_SEED)
        straddling = (torch.rand(2, 3, 6, 8) * 2.0).to(device=device, dtype=dtype)
        torch.manual_seed(seed)
        assert K.RandomSolarize(p=1.0)(straddling).unique().numel() > 1

    def test_convention_random_solarize_threshold_zero_inverts_the_clamped_zeros(self, device, dtype):
        # The other way the #4430 collapse fails: the clamped zeros are "at or above" a drawn threshold of 0,
        # so they are inverted and the all-negative image comes back as ones.
        image = torch.full((1, 3, 6, 8), -0.001, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSolarize(thresholds=(0.0, 0.0), additions=(0.0, 0.0), p=1.0)(image)
        self.assert_close(out, torch.ones_like(image))

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
        # A half-width of 0.05 instead of 0.1 satisfies every bound above, so also require the draw to
        # reach the documented half-width.  Over 256 draws each of these fails with probability
        # 0.9 ** 256 ~ 2e-12 when the half-width is right.
        assert float(params["thresholds"].max()) > 0.58
        assert float(params["thresholds"].min()) < 0.42
        assert float(params["additions"].max()) > 0.09
        assert float(params["additions"].min()) < -0.09

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
        # Containment alone cannot see a shrunk range: moving the generator's centre from 0.0 to -0.25
        # makes RandomSharpness(0.5) draw [0, 0.25], which satisfies every bound above.  Require the draw
        # to reach the documented bound.  Over 256 draws a true upper bound of 0.5 fails this with
        # probability 0.9 ** 256 ~ 2e-12.
        assert float(drawn.max()) > 0.45
        assert float(drawn.min()) < 0.05

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

    # Row 6c-34, the limit itself: a limit is the half-width of a symmetric range, so the draw takes both
    # signs, and a limit of 0 adds nothing but still leaves the channel to shift_rgb's clamp.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   aug = K.RandomRGBShift(r_shift_limit=0.5, g_shift_limit=0.0, b_shift_limit=0.0, p=1.0)
    #   print(aug.forward_parameters((512, 3, 2, 2))["r_shift"].aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `min=-0.4988 max=0.4998`.
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

    # Issue #4430, the RandomRGBShift half: the clamp on the sum makes a constant image at or below
    # `-limit` all-zero on every draw, and one between `-limit` and 0 on the draws whose shift does not
    # lift it.  Literal seeds, as for the census above.
    # Snippet used to generate expected:
    #   for v in (-0.5, -0.3):
    #       n = 0
    #       for s in range(100):
    #           torch.manual_seed(s)
    #           n += float(K.RandomRGBShift(p=1.0)(torch.full((2, 3, 6, 8), v)).abs().max()) == 0.0
    #       print(v, n)
    # executed 2026-09-16 (torch 2.14.0, cpu, float32) -> `-0.5 100`, `-0.3 24`.
    def test_wart_random_rgb_shift_at_or_below_minus_limit_collapses_on_every_draw_4430(self, device, dtype):
        zeros = {}
        for value in (-0.5, -0.3):
            image = torch.full((2, 3, 6, 8), value, device=device, dtype=dtype)
            zeros[value] = 0
            for seed in range(100):
                torch.manual_seed(seed)
                zeros[value] += float(K.RandomRGBShift(p=1.0)(image).abs().max()) == 0.0
        assert zeros[-0.5] == 100
        assert 5 <= zeros[-0.3] <= 50

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

    # Row 6c-04 and 6c-33, per channel: the selected row scales red and blue only, and the clamp that
    # follows is `clamp(max=1.0)` on all three channels -- green above 1 is cut back although it is not
    # scaled, a scaled value is cut only if it is still above 1, and negatives stay negative.
    # Snippet used to generate expected:
    #   x = torch.tensor([[1.5, -0.5], [1.7, -0.5], [1.5, -0.5]]).reshape(1, 3, 1, 2)
    #   aug = K.RandomPlanckianJitter(mode="blackbody", select_from=[0], p=1.0); print(aug.pl[0], aug(x))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> row `(1.6736, 0.0032)`; red `[1.0, -0.8368]`, green
    # `[1.0, -0.5]`, blue `[0.0048, -0.0016]`.
    def test_convention_random_planckian_jitter_scales_red_and_blue_then_clamps_above(self, device, dtype):
        image = torch.tensor([[1.5, -0.5], [1.7, -0.5], [1.5, -0.5]], device=device, dtype=dtype).reshape(1, 3, 1, 2)
        aug = K.RandomPlanckianJitter(mode="blackbody", select_from=[0], p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        red_gain, blue_gain = (float(value) for value in aug.pl[0])
        scaled = torch.stack([image[:, 0] * red_gain, image[:, 1], image[:, 2] * blue_gain], dim=1)
        # The coefficient table follows the input dtype, so the output is already in it and the
        # comparison needs no cast.
        self.assert_close(out, scaled.clamp(max=1.0))
        assert float(out[0, 1, 0, 0]) == 1.0 and float(out[0, 2, 0, 0]) < 0.01
        assert bool((out[..., 1] < 0).all())

    # Issue #4574: the coefficient table follows the input dtype, so half-precision inputs are no
    # longer promoted to float32, and a module cast does not decide the output dtype either.
    # Snippet used to generate expected:
    #   for dt in (torch.float16, torch.bfloat16):
    #       other = torch.bfloat16 if dt == torch.float16 else torch.float16
    #       torch.manual_seed(0); x = torch.rand(2, 3, 4, 4).to(dt)
    #       torch.manual_seed(0); a = K.RandomPlanckianJitter(p=1.0)(x).dtype
    #       torch.manual_seed(0); b = K.RandomPlanckianJitter(p=1.0).to(dtype=other)(x).dtype
    #       print(dt, a, b)
    # executed 2026-09-17 (torch 2.14.0, cpu) -> `float16 float16 float16` and `bfloat16 bfloat16
    # bfloat16`; on `30dfbf711`, this PR's base, both columns were `float32` for both inputs.
    def test_convention_random_planckian_jitter_preserves_dtype_4574(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPlanckianJitter(p=1.0)(image)

        assert out.dtype == dtype
        assert out.device == image.device

        if dtype in _HALF:
            # A module cast to the *other* half dtype returned float32 before the fix, because the
            # table's dtype decided the promotion; it now follows the input like every other cast.
            other = torch.bfloat16 if dtype == torch.float16 else torch.float16
            torch.manual_seed(_FORWARD_SEED)
            assert K.RandomPlanckianJitter(p=1.0).to(device=device, dtype=other)(image).dtype == dtype

    # Issue #4574, the wider direction: a module cast wider than the input does not widen the output.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); x = torch.rand(2, 3, 4, 4)
    #   torch.manual_seed(0); print(K.RandomPlanckianJitter(p=1.0).to(dtype=torch.float64)(x).dtype)
    # executed 2026-09-17 (torch 2.14.0, cpu) -> `float32`; on `30dfbf711` it was `float64`.
    def test_convention_random_planckian_jitter_wider_module_dtype_4574(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")

        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float32)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomPlanckianJitter(p=1.0).to(device=device, dtype=torch.float64)(image)

        assert out.dtype == torch.float32
        assert out.device == image.device

    # Row 6c-32/6c-33: the illuminant table is an RGB ratio, so a non-RGB input is rejected rather
    # than broadcast.  Snippet used to generate expected:
    #   torch.manual_seed(0); K.RandomPlanckianJitter(p=1.0)(torch.rand(1, 1, 6, 8))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `ShapeError: Shape mismatch at dimension 0:
    # expected 3, got 1.`
    def test_convention_random_planckian_jitter_requires_three_channels(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(1, 1, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        # `match` is part of the pin: requiring any other channel count raises the same ShapeError, so
        # without it `KORNIA_CHECK_SHAPE(input, ["*", "4", "H", "W"])` leaves this node green.
        with pytest.raises(ShapeError, match=r"expected 3, got 1"):
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
        # The denominator is `max - min + 1e-6`, so a channel whose range is not large next to 1e-6 stops
        # short of 1: a range of 1e-5 peaks at 1e-5 / 1.1e-5 = 0.90908 (float32/float64 only; the half
        # dtypes cannot hold 0.2 + 1e-5 apart from 0.2).
        if dtype in (torch.float32, torch.float64):
            narrow = torch.full((1, 1, 2, 2), 0.2, device=device, dtype=dtype)
            narrow[0, 0, 0, 0] += 1e-5
            torch.manual_seed(_FORWARD_SEED)
            peak = K.RandomAutoContrast(p=1.0)(narrow).max()
            assert abs(float(peak) - 1e-5 / 1.1e-5) < 1e-3

    # RandomJPEG, outside `kornia.augmentation.__all__`, carries one range sentence: `ycbcr_to_rgb`
    # hard-clamps the decoded RGB into [0, 1], and the codec's final soft clip, with bounds [0, 255], leaves
    # those values alone.  So an input far outside [0, 1] comes back inside it, and reaches both ends.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 32, 32) * 3 - 1
    #   torch.manual_seed(0); y = K.RandomJPEG(p=1.0)(x); print(y.aminmax(), y.isnan().any())
    # executed 2026-09-15 (torch 2.14.0, cpu float16/bfloat16/float32/float64 and mps float32) ->
    # `min=0, max=1`, `False`.
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
        # CPU/MPS float32 at quality 50: negative input peaks near 0.04625; positive input bottoms
        # near 0.95154. The half dtypes differ, but each keeps values strictly inside the range.
        assert bool(((out > 0.0) & (out < 1.0)).any())
        # Quantization can also give a constant black input small positive values, so the range
        # assertion alone would not catch a misplaced input clamp.
        assert not torch.equal(out, aug(image.clamp(0.0, 1.0)))

    # Row 6c-45: the documented parameter bounds are enforced, and where: `stage` records whether the
    # constructor raises or constructs and the forward pass raises, since the anchor's list of forward-time
    # checks is part of the contract.  Beyond the audit rows, RandomGaussianBlur's even kernel, a
    # RandomMotionBlur kernel range whose drawn odd size is 1, RandomChannelDropout dropping more channels
    # than the image has and a RandomPlanckianJitter row past the table are pinned here too.
    # The exception type is what the audit recorded, so the type is the pin and the message is not.
    # Snippet used to generate expected:
    #   ctor()(torch.rand(2, 3, 6, 8))  # for each case below
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `ValueError: If sharpness is a single number, it
    # must be non negative.`, `ValueError: bits[0] should be smaller than bits[1]`, `ValueError:
    # bits out of bounds. Expected inside (0, 8)`, `RuntimeError: Gamma must be non-negative.`,
    # `ValueError: brightness out of bounds. Expected inside (0.0, 2.0)`, `ValueError: contrast out
    # of bounds. Expected inside (0, inf)`, `ValueError: saturation out of bounds. Expected inside
    # (0, inf)`, `ValueError: hue out of bounds. Expected inside (-0.5, 0.5)`, `RuntimeError: The addition
    # must be in the open range (-0.5, 0.5).` (for 0.5 and -0.5; #4605), `BaseError: sigma must be positive` and
    # `BaseError: Height of drop should be greater than zero and less than image height.`; then
    # `BaseError: Kernel size must be an odd integer bigger than 0` (Gaussian (4, 4)), `... bigger than 2`
    # (motion (1, 1)), `BaseError: Invalid value in num_drop_channels` and `IndexError: index 25 is out
    # of bounds for dimension 0 with size 25`; the last one at construction, the three before it on the
    # forward pass.
    # `match` is part of the pin, not decoration: without it a case can be satisfied by a different
    # error of the same class.  `bits=9` is the example -- a scalar becomes the range [9, 8], so it trips
    # the "lo > hi" arm and says nothing about the documented (0, 8] upper bound; widening that bound to
    # (0, 16) left every pin in this file green until `posterize_bits_range_above_eight` was added.
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
            (
                "solarize_additions_at_half",
                "forward",
                RuntimeError,
                r"The addition must be in the open range \(-0\.5, 0\.5\)",
            ),
            (
                "solarize_additions_at_minus_half",
                "forward",
                RuntimeError,
                r"The addition must be in the open range \(-0\.5, 0\.5\)",
            ),
            ("gaussian_blur_sigma_zero", "forward", BaseError, r"sigma must be positive"),
            ("gaussian_blur_even_kernel", "forward", BaseError, r"Kernel size must be an odd integer bigger than 0"),
            ("median_blur_even_kernel", "forward", RuntimeError, r"is invalid for input of size"),
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
                r"Invalid value in .fill_value.\. Must be a float between 0 and 1",
            ),
            (
                "color_jiggle_brightness_range_above_two",
                "construction",
                ValueError,
                r"brightness out of bounds\. Expected inside \(0, 2\)",
            ),
            ("rgb_shift_tuple_limit", "construction", TypeError, r"bad operand type for unary -: 'tuple'"),
            (
                "planckian_select_past_the_table",
                "construction",
                IndexError,
                r"index 25 is out of bounds for dimension 0 with size 25",
            ),
            # The scalar rows name the reported tensor as well as the bound: it is the unclamped
            # `[center - x, center + x]`, which only `_range_bound`'s own scalar raise produces.  The
            # trailing `_joint_range_check` rejects the same inputs with the lower end already floored
            # (`[-0.5000, 0.6000]`, `[0., 4.]`, `[0.0000, 2.5000]`), so a bound-only regex passes either way.
            (
                "hue_scalar_above_half",
                "construction",
                ValueError,
                r"hue out of bounds\. Expected inside \(-0\.5, 0\.5\), got tensor\(\[-0\.6000,  0\.6000\]\)",
            ),
            (
                "brightness_scalar_above_two",
                "construction",
                ValueError,
                r"brightness out of bounds\. Expected inside \(0\.0, 2\.0\), got tensor\(\[-2\.,  4\.\]\)",
            ),
            (
                "solarize_scalar_threshold_above_half",
                "construction",
                ValueError,
                r"thresholds out of bounds\. Expected inside \(0\.0, 1\.0\), got tensor\(\[-1\.5000,  2\.5000\]\)",
            ),
        ],
    )
    def test_convention_intensity_constructors_reject_out_of_bounds(self, device, dtype, case, stage, error, match):
        if case in ("gamma_negative", "gain_negative") and device.type != "cpu":
            pytest.skip("CPU only: on CUDA the value assert is a device-side assert; MPS skips the check")
        # The solarize check runs on the CPU-drawn additions, so unlike the gamma check it raises for an
        # MPS image as well; CUDA stays out, like every value assert here.
        if case.startswith("solarize_additions") and device.type == "cuda":
            pytest.skip("not on CUDA: value asserts are kept out of the shared CUDA process")
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
            # The scalar forms overshoot the same bounds and raise the same error since #4563.
            "hue_scalar_above_half": lambda: K.RandomHue(0.6, p=1.0),
            "brightness_scalar_above_two": lambda: K.RandomBrightness(3.0, p=1.0),
            "solarize_scalar_threshold_above_half": lambda: K.RandomSolarize(2.0, 0.1, p=1.0),
            # The construction check is the closed [-0.5, 0.5]; kornia.enhance.solarize rejects the
            # ends on the forward pass, so this one constructs and raises like the gamma case.
            "solarize_additions_at_half": lambda: K.RandomSolarize((0.5, 0.5), (0.5, 0.5), p=1.0),
            "solarize_additions_at_minus_half": lambda: K.RandomSolarize((0.5, 0.5), (-0.5, -0.5), p=1.0),
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
            # An even entry raises a raw reshape error from median_blur's view, not a kornia check.
            "median_blur_even_kernel": lambda: K.RandomMedianBlur((4, 4), p=1.0),
            # An even bound is rounded up to the next odd size, so (0, 2) draws 1 and the kernel rejects it.
            "motion_blur_even_bound_rounds_up": lambda: K.RandomMotionBlur((0, 2), 35.0, 0.5, p=1.0),
            # `fill_value` is bounded to [0, 1] at construction, unlike RandomErasing's `value` type check.
            "channel_dropout_fill_value_above_one": lambda: K.RandomChannelDropout(fill_value=2.0, p=1.0),
            # ColorJiggle's brightness bound is (0, 2); ColorJitter accepts the same range.
            "color_jiggle_brightness_range_above_two": lambda: K.ColorJiggle(brightness=(0.0, 3.0), p=1.0),
            # Each *_shift_limit must be a scalar; a tuple dies on the unary minus that builds the range.
            "rgb_shift_tuple_limit": lambda: K.RandomRGBShift((0.1, 0.5), p=1.0),
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
            # The sync is what surfaces an MPS kernel error inside this block rather than later.
            _sync(aug(image).device)

    # Row 6c-45, the bound that never raises: RandomPlanckianJitter's `select_from` is used as a Python
    # index into its table, so a negative entry down to -25 selects a row from the end instead of being
    # rejected against the documented `[0-24]`.
    # Snippet used to generate expected:
    #   print(torch.equal(K.RandomPlanckianJitter(select_from=[-1]).pl, K.RandomPlanckianJitter(select_from=[24]).pl))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `True`.
    @pytest.mark.device_agnostic
    def test_convention_random_planckian_jitter_select_from_is_a_python_index(self):
        assert torch.equal(
            K.RandomPlanckianJitter(select_from=[-1], p=1.0).pl, K.RandomPlanckianJitter(select_from=[24], p=1.0).pl
        )

    # A scalar magnitude is ``center ± x`` floored at the parameter's lower bound and rejected past its
    # upper bound (the fix for #4563; the upper end used to be clamped silently, so RandomHue(0.7)
    # sampled from [-0.5, 0.5]).  RandomSharpness is the class that depends on the floor: its scalar
    # form is the centred `[-x, x]` floored at 0, which is what makes `sharpness=0.5` mean `[0, 0.5]`.
    # Snippet used to generate expected:
    #   for ctor in (lambda: K.RandomHue(0.7), lambda: K.RandomBrightness(3.0),
    #                lambda: K.ColorJiggle(brightness=1.5), lambda: K.RandomSolarize(2.0, 0.1),
    #                lambda: K.RandomMotionBlur(3, 45.0, 2.0), lambda: K.RandomJPEG(90.0)):
    #       try: ctor(); print("ok")
    #       except ValueError as e: print(e)
    #   print(K.RandomContrast(1.5).contrast, K.RandomBrightness(0.5).brightness)
    #   print(K.RandomSharpness(0.5).forward_parameters((2000, 1, 4, 4))["sharpness"].aminmax())
    #   print(K.RandomJPEG(50.0)._param_generator.jpeg_quality_sampler.low)
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `hue out of bounds. Expected inside (-0.5, 0.5), got
    # tensor([-0.7000,  0.7000]).`, `brightness out of bounds ... (0.0, 2.0), got tensor([-2., 4.])`,
    # `brightness out of bounds ... (0, 2), got tensor([-0.5000,  2.5000])`, `thresholds out of bounds ...
    # (0.0, 1.0), got tensor([-1.5000,  2.5000])`, `direction out of bounds ... (-1, 1), got
    # tensor([-2., 2.])`, `jpeg_quality out of bounds ... (1, 100), got tensor([-40., 140.])`;
    # `[0, 2.5]`, `[0.5, 1.5]`; sharpness in `[3.87e-05, 0.5]`; jpeg low `1`.
    @pytest.mark.device_agnostic
    def test_convention_scalar_magnitude_floors_low_and_rejects_high(self):
        for ctor, message in (
            (lambda: K.RandomHue(0.7, p=1.0), r"hue out of bounds\. .*got tensor\(\[-0\.7000,  0\.7000\]\)"),
            (lambda: K.RandomBrightness(3.0, p=1.0), r"brightness out of bounds\. .*got tensor\(\[-2\.,  4\.\]\)"),
            (
                lambda: K.ColorJiggle(brightness=1.5, p=1.0),
                r"brightness out of bounds\. .*got tensor\(\[-0\.5000,  2\.5000\]\)",
            ),
            (
                lambda: K.RandomSolarize(2.0, 0.1, p=1.0),
                r"thresholds out of bounds\. .*got tensor\(\[-1\.5000,  2\.5000\]\)",
            ),
            (
                lambda: K.RandomMotionBlur(3, 45.0, 2.0, p=1.0),
                r"direction out of bounds\. .*got tensor\(\[-2\.,  2\.\]\)",
            ),
            (lambda: K.RandomJPEG(90.0, p=1.0), r"jpeg_quality out of bounds\. .*got tensor\(\[-40\., 140\.\]\)"),
        ):
            # Each regex names the parameter and the unclamped `[center - x, center + x]`, so it can only be
            # satisfied by `_range_bound`'s scalar raise and not by the trailing range check behind it.
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

    # Issue #4564: RandomClahe rejects an out-of-[0, 1] input with the raw indexing error of the
    # histogram gather, naming neither the class nor the range RandomEqualize names (#4489).  Not on CUDA,
    # where the out-of-range index is a device-side assert that poisons the context; on MPS the error is
    # an AcceleratorError, a RuntimeError subclass, reading `gather: index ... is out of bounds`.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(1, 3, 16, 16) * 2
    #   torch.manual_seed(0); K.RandomClahe(p=1.0)(x)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `RuntimeError: index 430 is out of bounds for
    # dimension 5 with size 256`.
    def test_wart_random_clahe_out_of_range_error_is_raw_4564(self, device, dtype):
        if device.type == "cuda":
            pytest.skip("not on CUDA: the index error is a device-side assert that poisons the context")
        torch.manual_seed(_FIXTURE_SEED)
        image = (torch.rand(1, 3, 16, 16) * 2).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        if device.type == "mps" and not torch_version_ge(2, 14):
            # 2.5.1 leaves the MPS gather unchecked: the call returns an in-range image as if valid.
            out = K.RandomClahe(p=1.0)(image)
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
            return
        with pytest.raises(RuntimeError, match="out of bounds") as info:
            _sync(K.RandomClahe(p=1.0)(image).device)
        assert "RandomClahe" not in str(info.value) and "[0, 1]" not in str(info.value)

    # Divisibility is not the axis that decides whether `grid_size` works: a SQUARE grid is padded up to
    # a whole number of tiles, so it works whether or not it divides the image.  `_compute_tiles` is
    # called with `even_tile_size=True`, so an exactly dividing grid can still pad -- `(4, 5)` on
    # `20 x 20` pads 4 rows -- which is why the rectangular leg below carries the padding guard too.
    # Snippet used to generate expected:
    #   for gs, n in (((3, 3), 10), ((4, 4), 20), ((5, 5), 20), ((8, 8), 9)):
    #       torch.manual_seed(0); K.RandomClahe(grid_size=gs, p=1.0)(torch.rand(1, 1, n, n))
    # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> no raise for any of them.
    @pytest.mark.parametrize(("grid_size", "size", "pad"), [((3, 3), 10, 2), ((4, 4), 20, 4), ((5, 5), 20, 0)])
    def test_convention_random_clahe_square_grid_works_whether_or_not_it_tiles(
        self, device, dtype, grid_size, size, pad
    ):
        if pad and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        image = torch.linspace(0, 1, size * size, device=device, dtype=dtype).reshape(1, 1, size, size)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomClahe(grid_size=grid_size, p=1.0)(image)
        assert out.shape == image.shape
        assert bool(out.isfinite().all())
        # A pass-through would satisfy the two assertions above, so the transform has to have moved it.
        assert not torch.equal(out, image)
        if pad:
            # Shape, finiteness and "not the input" say nothing about HOW the image is padded, and the
            # inequality is satisfied by CLAHE itself: switching `_compute_tiles`' pad from "reflect" to
            # "constant" leaves all three green.  The padding is reflect and sits at the END of each axis,
            # so running the pre-padded image through the same call and cropping back reproduces this
            # output bitwise.
            padded = torch.nn.functional.pad(image, [0, pad, 0, pad], mode="reflect")
            torch.manual_seed(_FORWARD_SEED)
            reference = K.RandomClahe(grid_size=grid_size, p=1.0)(padded)[..., :size, :size]
            assert torch.equal(out, reference)

    # The other documented `grid_size` boundary, which nothing exercised: at the default `(8, 8)` the
    # smallest admissible square image is `9 x 9`.  `8 x 8` needs 8 rows of reflect padding on an
    # 8-row image, which torch refuses, and anything smaller than the grid gets the named ValueError.
    # Snippet used to generate expected:
    #   for n in (7, 8, 9): torch.manual_seed(0); K.RandomClahe(p=1.0)(torch.rand(1, 1, n, n))
    # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> ValueError("Cannot compute tiles on
    # the image according to the given grid size"), RuntimeError("Padding size should be less than the
    # corresponding input dimension"), and a `(1, 1, 9, 9)` output.
    @pytest.mark.parametrize(
        ("size", "error", "match"),
        [
            (7, ValueError, r"Cannot compute tiles on the image according to the given grid size"),
            (8, RuntimeError, r"Padding size should be less than the corresponding input dimension"),
        ],
    )
    def test_convention_random_clahe_default_grid_needs_nine_pixels(self, device, dtype, size, error, match):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(error, match=match):
            K.RandomClahe(p=1.0)(torch.rand(1, 1, size, size, device=device, dtype=dtype))
        # One pixel more on each axis is admitted, so the boundary is exactly where the block says.
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomClahe(p=1.0)(torch.rand(1, 1, 9, 9, device=device, dtype=dtype)).shape == (1, 1, 9, 9)

    # Issue #2531: every NON-square `grid_size` raises, on every image, whether or not it tiles.  The
    # `(4, 5)` / `20 x 20` row is the exactly dividing case, `(1, 2)` / `10 x 10` the one that pads, and
    # `(3, 5)` / `30 x 30` rules out a mismatch between odd and even grid entries as the cause.
    # Snippet used to generate expected:
    #   import itertools
    #   for gh, gw in itertools.product(range(1, 7), repeat=2):
    #       for h, w in ((24, 24), (30, 30), (24, 30), (35, 42), (60, 60)):
    #           torch.manual_seed(0); K.RandomClahe(grid_size=(gh, gw), p=1.0)(torch.rand(1, 1, h, w))
    # executed 2026-09-16 (torch 2.14.0, cpu) -> all 150 rows follow `raises == (gh != gw)`; a further
    # 40 random non-square grids from {1,2,3,4,5,7,8,9,12,16}^2 over three image shapes raised in
    # 120/120 cases and the 30 square controls all passed.
    @pytest.mark.parametrize(
        ("grid_size", "size", "pads"), [((4, 5), 20, True), ((3, 5), 30, False), ((1, 2), 10, True)]
    )
    def test_wart_random_clahe_non_square_grid_always_raises_2531(self, device, dtype, grid_size, size, pads):
        if pads and not supports_reflect_padding(device, dtype):
            # A padding row would die of "reflection_pad2d not implemented for 'Half'" on the torch 2.5.1
            # floor before reaching the lookup; the `(3, 5)` row pads nothing, so it keeps the pin alive
            # on every dtype.
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        if device.type == "cuda":
            # The broadcast of the lookup indices is computed on the host, so this should raise before
            # any kernel launches -- but that is untested on CUDA, and being wrong here poisons the
            # context for every later test, exactly as the #4564 pin above guards against.
            pytest.skip("not on CUDA: the raise site ahead of the device-side lookup is unverified there")
        image = torch.linspace(0, 1, size * size, device=device, dtype=dtype).reshape(1, 1, size, size)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(IndexError, match="shape mismatch: indexing tensors could not be broadcast together"):
            K.RandomClahe(grid_size=grid_size, p=1.0)(image)

    # Issue #4572: RandomClahe draws `clip_limit_factor` per sample but equalizes the whole batch with
    # the first sample's value (`float(params["clip_limit_factor"][0])`).
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 1, 32, 32) ** 3
    #   torch.manual_seed(0); aug = K.RandomClahe(clip_limit=(0.5, 40.0), grid_size=(2, 2), p=1.0); y = aug(x)
    #   c = aug._params["clip_limit_factor"]; print(c)
    #   for i in (1, 0):
    #       print(torch.equal(y[1:], equalize_clahe(x[1:], float(c[i]), (2, 2))))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `[20.1021, 30.8448]`, `False True`.
    @pytest.mark.device_agnostic
    def test_wart_random_clahe_applies_the_first_clip_limit_to_the_batch_4572(self):
        # B is 8, not 2: two draws from (0.5, 40.0) land within 1.0 of each other about 5% of the time,
        # and at B=2 this node failed on 12 of seeds 0..199.  Over 8 draws the spread was at least 9.4 on
        # every one of those seeds, and the sample farthest from the first is the one that shows the wart.
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(8, 1, 32, 32) ** 3
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomClahe(clip_limit=(0.5, 40.0), grid_size=(2, 2), p=1.0)
        out = aug(image)
        clip = aug._params["clip_limit_factor"]
        far = int((clip - clip[0]).abs().argmax())
        assert abs(float(clip[far]) - float(clip[0])) > 1.0
        # The whole batch is equalized with the first sample's clip limit, not with its own.
        assert torch.equal(out[far : far + 1], equalize_clahe(image[far : far + 1], float(clip[0]), (2, 2)))
        assert not torch.equal(out[far : far + 1], equalize_clahe(image[far : far + 1], float(clip[far]), (2, 2)))

    # Issue #4560: every class whose path goes through rgb_to_hsv returns NaN for a black pixel in
    # float16, because the conversion's `eps=1e-8` underflows to 0 there; bfloat16 keeps the exponent
    # range of float32 and is finite.  Pinned here so the four warnings that cite #4560 have an
    # executable anchor, and so the skip reasons above stay honest if the NaN ever disappears.
    # Snippet used to generate expected:
    #   for dt in (torch.float16, torch.float32, torch.bfloat16):
    #       print(dt, K.RandomHue((0.1, 0.1), p=1.0)(torch.zeros(1, 3, 4, 4, dtype=dt)).isnan().any())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `True`, `False`, `False`; the same for
    # RandomSaturation((1.5, 1.5)), ColorJiggle(0, 0, 0, (0.1, 0.1)) and ColorJitter(0, 0, 0, (0.1, 0.1)).
    # The pixel (0, -0.5, -0.5) is NaN in float16 through the three hue-step configurations and finite
    # through the two saturation-only ones (measured on cpu in all four dtypes and on mps float32).
    @pytest.mark.parametrize(
        "name", ["RandomHue", "RandomSaturation", "ColorJiggle", "ColorJiggleSaturation", "ColorJitter"]
    )
    def test_wart_hsv_path_black_pixel_is_nan_in_float16_4560(self, device, dtype, name):
        factories = {
            "RandomHue": lambda: K.RandomHue((0.1, 0.1), p=1.0),
            "RandomSaturation": lambda: K.RandomSaturation((1.5, 1.5), p=1.0),
            "ColorJiggle": lambda: K.ColorJiggle(0.0, 0.0, 0.0, (0.1, 0.1), p=1.0),
            # ColorJiggle's saturation step is adjust_saturation, an HSV round trip too, so the NaN needs
            # no hue step (ColorJitter's gray-subtraction saturation does not have it).
            "ColorJiggleSaturation": lambda: K.ColorJiggle(0.0, 0.0, (1.5, 1.5), 0.0, p=1.0),
            "ColorJitter": lambda: K.ColorJitter(0.0, 0.0, 0.0, (0.1, 0.1), p=1.0),
        }
        # A black pixel, and a pixel whose largest channel is 0 without being black: the hue step divides
        # by that zero maximum too, while the saturation step's NaN needs `max - min` to be 0 as well.
        pixels = torch.tensor([[0.0, 0.0, 0.0], [0.0, -0.5, -0.5]], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = factories[name]()(pixels.T.reshape(1, 3, 1, 2))
        half = dtype == torch.float16
        hue_path = name not in ("RandomSaturation", "ColorJiggleSaturation")
        assert out.isnan().any(1)[0, 0].tolist() == [half, half and hue_path]

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

    # Differentiability is a family-wide property, so it is pinned as one sweep: RandomClahe is the only
    # class of the 35 whose default output leaves the autograd graph, and RandomPosterize the only one
    # whose gradient is identically zero while still reporting requires_grad.
    # Snippet used to generate expected:
    #   x = (torch.rand(2, 3, 16, 16) * 0.8 + 0.1).requires_grad_()
    #   o = K.RandomClahe(p=1.0)(x); print(o.requires_grad, o.grad_fn)
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `False None`, and `True` with
    # slow_and_differentiable=True; RandomPosterize gives requires_grad True with x.grad.abs().sum() 0.0
    # below bits=8 and 1536.0 (= 2*3*16*16, the identity) at bits=8.
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

    # A non-integral `bits` is accepted and rounded half to even, which the "Integer" wording hid.
    # Snippet used to generate expected:
    #   for b in (2.5, 3.5, 0.4, 0.6):
    #       torch.manual_seed(0); a = K.RandomPosterize((b, b), p=1.0); a(torch.rand(1, 3, 16, 16))
    #       print(b, a._params["bits_factor"].tolist())
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `2`, `4`, `0`, `1`.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(("bits", "drawn"), [(2.5, 2), (3.5, 4), (0.4, 0), (0.6, 1)])
    def test_convention_random_posterize_rounds_a_float_bits_half_to_even(self, bits, drawn):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomPosterize((bits, bits), p=1.0)
        aug(torch.rand(1, 3, 16, 16))
        assert [int(v) for v in aug._params["bits_factor"].tolist()] == [drawn]

    # ColorJiggle skips a step whose drawn factor is neutral, where ColorJitter evaluates every step
    # under a torch.where.  So the channel count only has to suit the steps that actually run.
    # Snippet used to generate expected:
    #   for c in (1, 3, 4): torch.manual_seed(0); K.ColorJiggle(0, 0, 0, 0, p=1.0)(torch.rand(2, c, 5, 5))
    # executed 2026-09-16 (torch 2.14.0, cpu) -> all three accepted on ColorJiggle, C=1 and C=4 rejected
    # on ColorJitter; a saturation- or hue-only ColorJiggle rejects C=1 and C=4 as well.
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
        # ColorJitter computes every step, so the neutral configuration raises where ColorJiggle does not.
        torch.manual_seed(_FORWARD_SEED)
        if channels == 3:
            assert K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0)(image).shape == image.shape
        else:
            # WHICH step raises depends on the drawn `order`, so the message is not fixed.  At C=4 the
            # saturation primitive raises `ImageError: Not a color or gray tensor` when it comes first
            # and the hue step raises `ValueError: Input size must have a shape of (*, 3, H, W)` when it
            # does -- 102 against 98 over seeds 0..199, so pinning either one alone is a coin flip.  At
            # C=1 the saturation step passes a one-channel image through, so only the hue message occurs
            # (200/200).
            # Snippet used to generate expected:
            #   torch.manual_seed(1234); x = torch.rand(2, c, 5, 5)
            #   for s in range(200): torch.manual_seed(s); K.ColorJitter(0, 0, 0, 0, p=1.0)(x)
            # executed 2026-09-16 (torch 2.14.0, cpu).
            expected = r"Input size must have a shape of \(\*, 3, H, W\)"
            if channels != 1:
                expected = r"Not a color or gray tensor|" + expected
            with pytest.raises((ValueError, ImageError), match=expected):
                K.ColorJitter(0.0, 0.0, 0.0, 0.0, p=1.0)(image)

    # The erasing box is clamped from BELOW as well as above, so it is never empty: scale=(0, 0) still
    # erases one pixel and a 1x1 image is always erased in full.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); a = K.RandomErasing(scale=(0.0, 0.0), ratio=(1.0, 1.0), p=1.0)
    #   o = a(torch.ones(1, 1, 10, 10)); print(int((o == 0).sum()), a._params["heights"].tolist())
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `1 [1.0]`, and `1 [1.0]` for a 1x1 image.
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

    # When `ratio` straddles 1 the draw is a 50/50 mixture of [ratio[0], 1] and [1, ratio[1]], not uniform
    # over the interval, so tall and wide boxes are equally likely whatever the sub-intervals' widths.
    # A single-point range cannot see this -- it needs the ratio of the two branch populations.
    # Snippet used to generate expected:
    #   g = RectangleEraseGenerator(scale=(0.25, 0.25), ratio=(0.3, 3.3))
    #   g.set_rng_device_and_dtype(torch.device("cpu"), torch.float32)
    #   torch.manual_seed(0); p = g((200000, 3, 400, 400))
    #   print((p["heights"] > p["widths"]).float().mean())
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `0.4987`, against `(3.3 - 1) / (3.3 - 0.3) = 0.7667` for a
    # uniform draw; ratio=(1.0, 3.3) takes the single-sampler branch and gives 0.998.
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
        # The fair coin is a property of the DRAW.  The realised box only follows it while the
        # `[1, H] x [1, W]` clamp does not bite, and on a strongly non-square image it always does.
        # Snippet used to generate expected: this body with the shape replaced; executed 2026-09-16
        # (torch 2.14.0, cpu) -> tall fraction 0.4987 at 400 x 400, 0.0 at 4 x 256 and 1.0 at 256 x 4.
        for height, width, expected in ((4, 256, 0.0), (256, 4, 1.0)):
            torch.manual_seed(_FORWARD_SEED)
            clamped = generator((20000, 3, height, width))
            realised = float((clamped["heights"] > clamped["widths"]).float().mean())
            assert realised == expected, (height, width, realised)
