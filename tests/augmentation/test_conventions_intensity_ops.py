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

import copy
import inspect
import io
import itertools
import pickle

import pytest
import torch
from torch.distributions import Distribution

import kornia.augmentation as K
from kornia.constants import BorderType, Resample
from kornia.core.exceptions import BaseError, ImageError, ShapeError
from kornia.filters import box_blur

from testing.base import (
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    dynamo_is_available,
    supports_reflect_padding,
    supports_replicate_padding,
)


@pytest.fixture(autouse=True)
def _restore_global_rng(restore_torch_rng):
    # The pins below seed the global RNG; the root fixture (#4446) restores every generator afterwards.
    yield


# The seed drawn immediately before each construct-and-forward.
_FORWARD_SEED = 0
_FIXTURE_SEED = 1234


def _sync(device) -> None:
    # MPS dispatches asynchronously, so a kernel error raised by the forward under test would
    # otherwise surface inside an unrelated later test.
    if device.type == "mps":
        torch.mps.synchronize()


def _lit_extent(plane: torch.Tensor, threshold: float = 1e-3) -> tuple[list[int], list[int]]:
    """Return the sorted distinct rows and columns of a single ``(H, W)`` plane above ``threshold``."""
    indices = (plane.float().abs() > threshold).nonzero()
    return sorted({int(row) for row, _ in indices}), sorted({int(col) for _, col in indices})


def _impulse(device, dtype, height: int = 7, width: int = 9, row: int = 2, col: int = 3) -> torch.Tensor:
    """A non-square zero image with a single 1.0 off both centre lines (7x9 centres are row 3, col 4)."""
    image = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    image[0, 0, row, col] = 1.0
    return image


class TestBlurConventions(BaseTester):
    # ``kernel_size`` is ``(kH, kW)``, also checked under relabelling (transposed image and kernel).
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 3, 4] = 1.0
    #   for ks in ((1, 5), (5, 1)):
    #       torch.manual_seed(0); y = K.RandomBoxBlur(ks, p=1.0)(x); print(ks, (y[0, 0] > 0).nonzero().T.tolist())
    @pytest.mark.parametrize("name", ["RandomBoxBlur", "RandomGaussianBlur"])
    def test_convention_blur_kernel_size_is_height_then_width(self, device, dtype, name):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        factories = {
            "RandomBoxBlur": lambda ks: K.RandomBoxBlur(ks, p=1.0),
            "RandomGaussianBlur": lambda ks: K.RandomGaussianBlur(ks, (1.0, 1.0), p=1.0),
        }
        make = factories[name]

        # Hot pixel on the 7x9 centre.
        centred = _impulse(device, dtype, row=3, col=4)
        torch.manual_seed(_FORWARD_SEED)
        assert _lit_extent(make((1, 5))(centred)[0, 0]) == ([3], [2, 3, 4, 5, 6])
        torch.manual_seed(_FORWARD_SEED)
        assert _lit_extent(make((5, 1))(centred)[0, 0]) == ([1, 2, 3, 4, 5], [4])

        # The same claim with the impulse off both centre lines, so no literal is its own transpose.
        image = _impulse(device, dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert _lit_extent(make((1, 5))(image)[0, 0]) == ([2], [1, 2, 3, 4, 5])
        torch.manual_seed(_FORWARD_SEED)
        assert _lit_extent(make((5, 1))(image)[0, 0]) == ([0, 1, 2, 3, 4], [3])

        # Relabelling check: transpose the image and the kernel, and the support transposes too.
        transposed = image.transpose(-1, -2).contiguous()
        assert transposed.shape[-2:] == (9, 7)
        torch.manual_seed(_FORWARD_SEED)
        assert _lit_extent(make((5, 1))(transposed)[0, 0]) == ([1, 2, 3, 4, 5], [2])
        torch.manual_seed(_FORWARD_SEED)
        assert _lit_extent(make((1, 5))(transposed)[0, 0]) == ([3], [0, 1, 2, 3, 4])

    def test_convention_median_blur_border_median_is_over_zero_padding(self, device, dtype):
        # A constant-ones image: the 3x3 window at a corner holds 5 zeros and 4 ones, so the corner comes
        # back 0 while an edge pixel (3 zeros, 6 ones) stays 1.  Replicate padding would return 1 at both.
        # Snippet used to generate expected:
        #   torch.manual_seed(0); y = K.RandomMedianBlur((3, 3), p=1.0)(torch.ones(1, 1, 4, 4)); print(y[0, 0, 0, :2])
        #   torch.manual_seed(0); print(K.RandomMedianBlur((5, 5), p=1.0)(torch.ones(1, 1, 6, 6))[0, 0, 0])
        ones = torch.ones(1, 1, 4, 4, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomMedianBlur((3, 3), p=1.0)(ones)
        assert float(out[0, 0, 0, 0]) == 0.0 and float(out[0, 0, 0, 1]) == 1.0
        assert float(out[0, 0, 1, 1]) == 1.0
        # 5x5 on a 6x6 image: the top row's window holds 3 image rows, so 15 ones at columns 2 and 3
        # (median 1) but 12 at columns 1 and 4 and 9 at the corners (median 0).
        torch.manual_seed(_FORWARD_SEED)
        wide = K.RandomMedianBlur((5, 5), p=1.0)(torch.ones(1, 1, 6, 6, device=device, dtype=dtype))
        self.assert_close(wide[0, 0, 0], ones.new_tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))

    def test_convention_box_blur_even_kernel_is_accepted_and_off_centre(self, device, dtype):
        # Snippet used to generate expected:
        #   imp = torch.zeros(1, 1, 7, 9); imp[0, 0, 3, 4] = 1.0
        #   for ks in ((2, 2), (4, 4), (3, 4)):
        #       torch.manual_seed(0); nz = K.RandomBoxBlur(ks, p=1.0)(imp)[0, 0].nonzero()
        #       print(ks, nz[:, 0].unique().tolist(), nz[:, 1].unique().tolist())
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        impulse = torch.zeros(1, 1, 7, 9, device=device, dtype=dtype)
        impulse[0, 0, 3, 4] = 1.0
        for kernel_size, rows, cols in (
            ((2, 2), [2, 3], [3, 4]),
            ((4, 4), [1, 2, 3, 4], [2, 3, 4, 5]),
            ((3, 4), [2, 3, 4], [2, 3, 4, 5]),
        ):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomBoxBlur(kernel_size, p=1.0)(impulse)
            lit = out[0, 0].nonzero()
            assert lit[:, 0].unique().tolist() == rows and lit[:, 1].unique().tolist() == cols
            self.assert_close(out.sum(), impulse.sum())

    # A rank filter cannot be read off an impulse, so the detector is a one-row bar.  A (1, kW) window slides
    # along that row and keeps it; a (kH, 1) window spans five rows of which four are zero, so the median is
    # zero and the bar is erased.
    # Snippet used to generate expected:
    #   bar = torch.zeros(1, 1, 7, 9); bar[0, 0, 3, :] = 1.0
    #   for ks in ((1, 5), (5, 1)):
    #       torch.manual_seed(0); print(ks, K.RandomMedianBlur(ks, p=1.0)(bar)[0, 0].sum(-1).tolist())
    def test_convention_median_blur_kernel_size_is_height_then_width(self, device, dtype):
        bar = torch.zeros(1, 1, 7, 9, device=device, dtype=dtype)
        bar[0, 0, 3, :] = 1.0
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(K.RandomMedianBlur((1, 5), p=1.0)(bar)[0, 0].sum(-1), bar.new_tensor([0, 0, 0, 9, 0, 0, 0]))
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(K.RandomMedianBlur((5, 1), p=1.0)(bar)[0, 0].sum(-1), bar.new_zeros(7))

        # Off the centre row, and then transposed, so the surviving axis is not a symmetry of the fixture.
        off_centre = torch.zeros(1, 1, 7, 9, device=device, dtype=dtype)
        off_centre[0, 0, 2, :] = 1.0
        torch.manual_seed(_FORWARD_SEED)
        kept = K.RandomMedianBlur((1, 5), p=1.0)(off_centre)[0, 0].sum(-1)
        self.assert_close(kept, off_centre.new_tensor([0, 0, 9, 0, 0, 0, 0]))
        transposed = off_centre.transpose(-1, -2).contiguous()
        torch.manual_seed(_FORWARD_SEED)
        kept_t = K.RandomMedianBlur((5, 1), p=1.0)(transposed)[0, 0].sum(-2)
        self.assert_close(kept_t, transposed.new_tensor([0, 0, 9, 0, 0, 0, 0]))
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(K.RandomMedianBlur((1, 5), p=1.0)(transposed)[0, 0].sum(-2), transposed.new_zeros(7))

    # Wart (#4781, https://github.com/kornia/kornia/issues/4781): an even entry is constructed and then fails on
    # every image size with a raw torch error, not a kornia one.  Rejecting it with a kornia error or
    # supporting it flips this pin.
    @pytest.mark.parametrize("shape", [(2, 3, 32, 32), (2, 3, 1, 1)])
    def test_wart_random_median_blur_even_kernel_raises_a_raw_torch_error_4781(self, device, dtype, shape):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(*shape).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        even = K.RandomMedianBlur((4, 4), p=1.0)
        with pytest.raises(RuntimeError) as raised:
            _sync(even(image).device)
        assert type(raised.value) is RuntimeError

    # ``normalized`` is forwarded as box_blur's ``separable``; both branches are means and agree to rounding.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 7, 9)
    #   torch.manual_seed(0); yt = K.RandomBoxBlur((3, 3), normalized=True, p=1.0)(x)
    #   print(torch.equal(yt, box_blur(x, (3, 3), "reflect", separable=True)))
    def test_convention_random_box_blur_normalized_selects_separable_box_blur(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 7, 9).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        separable = K.RandomBoxBlur((3, 3), normalized=True, p=1.0)(image)
        torch.manual_seed(_FORWARD_SEED)
        joint = K.RandomBoxBlur((3, 3), normalized=False, p=1.0)(image)
        assert torch.equal(separable, box_blur(image, (3, 3), "reflect", separable=True))
        assert torch.equal(joint, box_blur(image, (3, 3), "reflect", separable=False))
        assert not torch.equal(separable, box_blur(image, (3, 3), "reflect", separable=False))
        # Both branches are the same mean, so they agree to float rounding.
        self.assert_close(separable, joint)
        # A mean leaves a constant image alone; a sum would multiply it by nine.
        constant = torch.full((2, 3, 7, 9), 0.375, device=device, dtype=dtype)
        for normalized in (True, False):
            torch.manual_seed(_FORWARD_SEED)
            self.assert_close(K.RandomBoxBlur((3, 3), normalized=normalized, p=1.0)(constant), constant)

    # One scalar sigma per sample (a (B,) tensor), and the defaults are gaussian_blur2d's own.
    # Snippet used to generate expected:
    #   aug = K.RandomGaussianBlur((3, 3), (0.5, 1.5), p=1.0)
    #   torch.manual_seed(0); aug(torch.rand(4, 3, 7, 9)); print(aug._params["sigma"].shape, aug.flags)
    def test_convention_random_gaussian_blur_draws_one_sigma_per_sample(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 7, 9).to(device=device, dtype=dtype)
        aug = K.RandomGaussianBlur((3, 3), (0.5, 1.5), p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        aug(image)
        sigma = aug._params["sigma"]
        assert sigma.shape == (4,)
        assert len(set(sigma.flatten().tolist())) == 4
        assert float(sigma.min()) >= 0.5 and float(sigma.max()) <= 1.5
        # Reach, not only containment: a draw shrunk to `[0.5, 0.6]` satisfies the bounds above.
        torch.manual_seed(_FORWARD_SEED)
        many = aug.forward_parameters((256, 3, 7, 9))["sigma"]
        assert float(many.min()) < 0.6 and float(many.max()) > 1.4
        assert aug.flags["separable"] is True
        assert aug.flags["border_type"] == BorderType.REFLECT
        assert aug.flags["kernel_size"] == (3, 3)

        # One parameter away: same_on_batch collapses the same key to a single repeated value.
        shared = K.RandomGaussianBlur((3, 3), (0.5, 1.5), p=1.0, same_on_batch=True)
        torch.manual_seed(_FORWARD_SEED)
        shared(image)
        assert len(set(shared._params["sigma"].flatten().tolist())) == 1

    # An even kernel_size is rejected with gaussian_blur2d's own "odd integer" error, whatever the image
    # size: a 2x2 image is below the reflect minimum, so a size check that ran first would blame the image.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); K.RandomGaussianBlur((4, 4), (0.1, 1.0), p=1.0)(torch.rand(1, 1, 2, 2))
    @pytest.mark.parametrize(
        ("kernel_size", "shape"), [((2, 2), (4, 3, 7, 9)), ((4, 4), (1, 1, 2, 2)), ((3, 4), (1, 1, 2, 2))]
    )
    def test_convention_random_gaussian_blur_rejects_even_kernel_size(self, device, dtype, kernel_size, shape):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(*shape).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(BaseError, match="odd integer"):
            _sync(K.RandomGaussianBlur(kernel_size, (0.1, 1.0), p=1.0)(image).device)

    # ``angle`` is counter-clockwise as displayed; the impulse sits off both centre lines, so no literal
    # is its own transpose.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 2, 3] = 1.0
    #   for a in (0.0, 45.0, -45.0, 90.0):
    #       torch.manual_seed(0)
    #       print(a, (K.RandomMotionBlur(5, (a, a), (0.0, 0.0), p=1.0)(x)[0, 0].abs() > 1e-6).nonzero().tolist())
    @pytest.mark.parametrize(
        ("angle", "support"),
        [
            (0.0, [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]),
            (45.0, [(1, 4), (2, 3), (3, 2)]),
            (-45.0, [(1, 2), (2, 3), (3, 4)]),
            (90.0, [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]),
        ],
    )
    def test_convention_random_motion_blur_angle_is_counter_clockwise(self, device, dtype, angle, support):
        image = _impulse(device, dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomMotionBlur(5, (angle, angle), (0.0, 0.0), p=1.0)(image)
        lit = [(int(row), int(col)) for row, col in (out[0, 0].float().abs() > 1e-6).nonzero()]
        assert sorted(lit) == support

    # ``direction`` re-weights the kernel along its own axis (-1 start, +1 end, 0 uniform), on two angles.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 2, 3] = 1.0
    #   for d in (-1.0, 0.0, 1.0):
    #       torch.manual_seed(0); print(K.RandomMotionBlur(5, (0.0, 0.0), (d, d), p=1.0)(x)[0, 0, 2].tolist())
    #       torch.manual_seed(0); print(K.RandomMotionBlur(5, (90.0, 90.0), (d, d), p=1.0)(x)[0, 0, :, 3].tolist())
    @pytest.mark.parametrize(
        ("direction", "along_row", "along_col"),
        [
            (-1.0, [0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0]),
            (0.0, [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]),
            (1.0, [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0], [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]),
        ],
    )
    def test_convention_random_motion_blur_direction_weights_the_kernel_axis(
        self, device, dtype, direction, along_row, along_col
    ):
        image = _impulse(device, dtype)
        torch.manual_seed(_FORWARD_SEED)
        horizontal = K.RandomMotionBlur(5, (0.0, 0.0), (direction, direction), p=1.0)(image)
        self.assert_close(horizontal[0, 0, 2], image.new_tensor(along_row))
        torch.manual_seed(_FORWARD_SEED)
        vertical = K.RandomMotionBlur(5, (90.0, 90.0), (direction, direction), p=1.0)(image)
        self.assert_close(vertical[0, 0, :, 3], image.new_tensor(along_col))

    # The weighted line is rotated with ``resample``; a "nearest" rotation at 45 degrees keeps three of the
    # five taps.  The impulse response is the kernel, so these are its taps.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 9, 9); x[0, 0, 4, 4] = 1.0
    #   for angle in (0.0, 45.0):
    #       torch.manual_seed(0); y = K.RandomMotionBlur(5, (angle, angle), (1.0, 1.0), p=1.0)(x)
    #       print(angle, sorted(y[y > 0].tolist()))
    def test_convention_random_motion_blur_nearest_rotation_resamples_the_taps(self, device, dtype):
        image = _impulse(device, dtype, height=9, width=9, row=4, col=4)
        taps = {}
        for angle in (0.0, 45.0):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomMotionBlur(5, (angle, angle), (1.0, 1.0), p=1.0)(image)
            taps[angle] = out[out.abs() > 1e-3].sort().values
        self.assert_close(taps[0.0], image.new_tensor([0.1, 0.2, 0.3, 0.4]))
        self.assert_close(taps[45.0], image.new_tensor([1.0, 2.0, 3.0]) / 6.0)

    # The class keeps kornia.filters.motion_blur's defaults.
    @pytest.mark.device_agnostic
    def test_convention_random_motion_blur_defaults_are_constant_and_nearest(self):
        aug = K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), p=1.0)
        assert aug.flags["border_type"] == BorderType.CONSTANT
        assert aug.flags["resample"] == Resample.NEAREST

    # Every odd size in a tuple range is drawn in near-equal shares, bounds included (#4599); an even upper
    # bound caps at the odd size below it, and a range holding no odd size rounds up (``(4, 4)`` draws 5).
    # The size is shared within a batch (#4671), so independent calls are sampled.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); aug = K.RandomMotionBlur(ks, (0., 0.), (0., 0.), p=1.)
    #   v = torch.cat([aug.forward_parameters((1, 1, 8, 8))["ksize_factor"] for _ in range(20000)])
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        ("kernel_size", "drawn"),
        [
            ((3, 5), [3, 5]),
            ((3, 7), [3, 5, 7]),
            ((5, 11), [5, 7, 9, 11]),
            ((3, 20), [3, 5, 7, 9, 11, 13, 15, 17, 19]),
            ((4, 7), [5, 7]),
            ((4, 4), [5]),
        ],
    )
    def test_convention_random_motion_blur_draws_every_odd_size_in_the_range_4599(self, kernel_size, drawn):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomMotionBlur(kernel_size, (0.0, 0.0), (0.0, 0.0), p=1.0)
        factors = torch.cat([aug.forward_parameters((1, 1, 8, 8))["ksize_factor"] for _ in range(20000)])
        counts = torch.stack([(factors == k).sum() for k in drawn])
        assert sorted(set(factors.tolist())) == drawn
        # Near-equal shares: each size within 10% of 20000 / len(drawn).
        expected = 20000 / len(drawn)
        assert bool(((counts - expected).abs() < 0.1 * expected).all()), counts.tolist()

    # Issue #4671: the stored sizes describe the shared kernel actually applied to every image.
    def test_convention_random_motion_blur_stored_kernel_sizes_match_impulse_widths_4671(self, device, dtype):
        image = torch.zeros(8, 1, 17, 17, device=device, dtype=dtype)
        image[:, :, 8, 8] = 1
        aug = K.RandomMotionBlur((3, 9), (0.0, 0.0), (0.0, 0.0), p=1.0)
        drawn = set()
        for seed in range(8):
            torch.manual_seed(seed)
            output = aug(image)
            # A horizontal, uniform blur spreads an impulse across exactly kernel_size pixels.
            widths = (output[:, 0, 8] != 0).sum(-1)
            assert widths.tolist() == aug._params["ksize_factor"].tolist()
            assert widths.unique().numel() == 1
            drawn.update(widths.tolist())
        assert len(drawn) > 1

    # At ``border_type="reflect"`` a "nearest" or "bilinear" kernel rotation stays inside the input's
    # extremes; "bicubic" gives the kernel negative lobes that overshoot both at an edge.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 15, 15); x[..., 7:] = 1.0
    #   torch.manual_seed(0)
    #   print(K.RandomMotionBlur(5, (80., 80.), (1., 1.), border_type="reflect", resample="bicubic", p=1.)(x).aminmax())
    def test_convention_random_motion_blur_reflect_bound_depends_on_resample(self, device, dtype):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        edge = torch.zeros(1, 1, 15, 15, device=device, dtype=dtype)
        edge[..., 7:] = 1.0
        for resample in ("nearest", "bilinear"):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomMotionBlur(5, (80.0, 80.0), (1.0, 1.0), border_type="reflect", resample=resample, p=1.0)(edge)
            assert float(out.min()) >= -1e-3 and float(out.max()) <= 1.0 + 1e-3, resample
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomMotionBlur(5, (80.0, 80.0), (1.0, 1.0), border_type="reflect", resample="bicubic", p=1.0)(edge)
        assert float(out.min()) < -0.02 and float(out.max()) > 1.02

    # The four filters do not clamp: a 1.1 impulse is attenuated below one, and the output is linear in the
    # input (for the median blur, a negative plateau stays negative), so no wrapper clamp is hiding in there.
    @pytest.mark.parametrize("name", ["RandomBoxBlur", "RandomGaussianBlur", "RandomMedianBlur", "RandomMotionBlur"])
    def test_convention_filters_attenuate_an_out_of_range_impulse_without_clamping(self, device, dtype, name):
        if name in ("RandomBoxBlur", "RandomGaussianBlur") and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        factories = {
            "RandomBoxBlur": lambda: K.RandomBoxBlur((3, 3), p=1.0),
            "RandomGaussianBlur": lambda: K.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0),
            "RandomMedianBlur": lambda: K.RandomMedianBlur((3, 3), p=1.0),
            "RandomMotionBlur": lambda: K.RandomMotionBlur(3, (0.0, 0.0), (0.0, 0.0), p=1.0),
        }
        make = factories[name]

        impulse = torch.zeros(1, 1, 7, 7, device=device, dtype=dtype)
        impulse[0, 0, 3, 3] = 1.1
        out = make()(impulse)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

        if name == "RandomMedianBlur":
            self.assert_close(out, torch.zeros_like(out))
            # A negative plateau has a negative median and must come back negative.
            negative = torch.full((1, 1, 7, 7), -0.5, device=device, dtype=dtype)
            torch.manual_seed(_FORWARD_SEED)
            assert float(make()(negative).min()) < -0.4, "the wrapper clamped the median into [0, 1]"
            return

        half_impulse = impulse * 0.5
        self.assert_close(out, make()(half_impulse) * 2.0)
        self.assert_close(make()(-half_impulse), -make()(half_impulse))

    # Issue #4559: RandomBoxBlur, RandomGaussianBlur and RandomSharpness raise a kornia `ValueError` naming
    # the class, the kernel and the input shape.  The two blurs reflect-pad, so an axis must be longer than the
    # kernel radius (`k // 2`, one pixel for the 3x3 default); RandomSharpness convolves without padding, so
    # it needs the full 3x3.  RandomMedianBlur and, at its default constant border, RandomMotionBlur accept
    # the same degenerate image.
    # Snippet used to generate expected:
    #   for shape in ((2, 3, 1, 8), (2, 3, 2, 2), (2, 3, 3, 3)):
    #       x = torch.rand(*shape)
    #       torch.manual_seed(0); K.RandomBoxBlur(p=1.0)(x)  # and the four other classes
    def test_convention_blur_and_sharpness_name_the_size_they_need_4559(self, device, dtype):
        # Only the legs that reflect-pad are skipped where reflection_pad2d is unavailable.
        reflect_ok = supports_reflect_padding(device, dtype)
        torch.manual_seed(_FIXTURE_SEED)
        thin = torch.rand(2, 3, 1, 8).to(device=device, dtype=dtype)
        small = torch.rand(2, 3, 2, 2).to(device=device, dtype=dtype)
        square = torch.rand(2, 3, 3, 3).to(device=device, dtype=dtype)
        blurs = {
            "RandomBoxBlur": lambda: K.RandomBoxBlur(p=1.0),
            "RandomGaussianBlur": lambda: K.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0),
        }
        for name, make in blurs.items():
            if not reflect_ok:
                continue
            torch.manual_seed(_FORWARD_SEED)
            # kornia's own error, raised before the filter runs, so it is device- and dtype-independent.
            with pytest.raises(ValueError, match=rf"{name} cannot filter an image this small"):
                _sync(make()(thin).device)
            torch.manual_seed(_FORWARD_SEED)
            assert make()(small).shape == small.shape, f"{name} should still accept a 2x2 image"
        # The threshold is per axis, half the kernel's extent along that axis: a (7, 3) kernel runs on a
        # 2-column image (2 > 3 // 2) and raises on a 3-row one (3 <= 7 // 2), in both classes.
        rectangular = {
            "RandomBoxBlur": lambda: K.RandomBoxBlur((7, 3), p=1.0),
            "RandomGaussianBlur": lambda: K.RandomGaussianBlur((7, 3), (1.0, 1.0), p=1.0),
        }
        narrow = torch.rand(2, 3, 20, 2).to(device=device, dtype=dtype)
        short = torch.rand(2, 3, 3, 20).to(device=device, dtype=dtype)
        for name, make in rectangular.items():
            if not reflect_ok:
                continue
            torch.manual_seed(_FORWARD_SEED)
            assert make()(narrow).shape == narrow.shape
            torch.manual_seed(_FORWARD_SEED)
            with pytest.raises(ValueError, match=rf"{name} cannot filter an image this small"):
                _sync(make()(short).device)
            # the message names the axis that is too short, not just the kernel
            with pytest.raises(ValueError, match="along height"):
                make()(short)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match="RandomSharpness cannot filter an image this small"):
            _sync(K.RandomSharpness(1.0, p=1.0)(small).device)
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomSharpness(1.0, p=1.0)(square).shape == square.shape
        # The two rank/kernel-rotation filters accept the same one-row image, so the failure is not
        # a package-wide "kernel larger than image" rule.
        for make in (lambda: K.RandomMedianBlur(p=1.0), lambda: K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), p=1.0)):
            torch.manual_seed(_FORWARD_SEED)
            assert make()(thin).shape == thin.shape

    # ``border_type="circular"`` needs each axis at least the kernel radius (``k // 2``): the two padding
    # blurs raise kornia's named error below it, while ``"constant"`` and ``"replicate"`` run there.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); thin = torch.rand(2, 3, 3, 12)
    #   for bt in ("constant", "replicate", "circular"):
    #       torch.manual_seed(0); K.RandomBoxBlur((9, 9), border_type=bt, p=1.)(thin)
    def test_convention_circular_padding_rejects_images_smaller_than_kernel_radius_4559(self, device, dtype):
        if not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        thin = torch.rand(2, 3, 3, 12).to(device=device, dtype=dtype)
        blurs = {
            "RandomBoxBlur": lambda border: K.RandomBoxBlur((9, 9), border_type=border, p=1.0),
            "RandomGaussianBlur": lambda border: K.RandomGaussianBlur((9, 9), (1.0, 1.0), border_type=border, p=1.0),
        }
        for name, make in blurs.items():
            torch.manual_seed(_FORWARD_SEED)
            with pytest.raises(ValueError, match=rf"{name} cannot filter an image this small"):
                _sync(make("circular")(thin).device)
            for border in ("constant", "replicate"):
                torch.manual_seed(_FORWARD_SEED)
                assert make(border)(thin).shape == thin.shape, f"{name} at {border} should accept a 3-row image"

    # Wart (#4784, https://github.com/kornia/kornia/issues/4784): RandomMotionBlur has the same minimum
    # size under ``"reflect"`` and ``"circular"`` but lets torch's raw padding error out, where RandomBoxBlur
    # raises a named ValueError.  A kornia check flips this pin; the controls below it do not move.
    @pytest.mark.parametrize(
        ("border", "kernel_size", "shape"), [("reflect", 3, (2, 3, 1, 8)), ("circular", 5, (2, 3, 1, 1))]
    )
    def test_wart_random_motion_blur_small_image_raises_a_raw_padding_error_4784(
        self, device, dtype, border, kernel_size, shape
    ):
        if border == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(*shape).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomMotionBlur(kernel_size, (45.0, 45.0), (0.0, 0.0), border_type=border, p=1.0)
        with pytest.raises(RuntimeError) as raised:
            _sync(aug(image).device)
        assert type(raised.value) is RuntimeError
        # The default constant border runs on the same image, and one radius less runs under circular.
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomMotionBlur(kernel_size, (45.0, 45.0), (0.0, 0.0), p=1.0)(image).shape == image.shape
        if border == "circular":
            torch.manual_seed(_FORWARD_SEED)
            smaller = K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), border_type=border, p=1.0)
            assert smaller(image).shape == image.shape

    # An even extent ``k`` is padded asymmetrically, so the minimum size follows the wider pad ``k // 2``:
    # reflect needs ``k // 2 + 1`` pixels and circular ``k // 2``.
    # Snippet used to generate expected:
    #   for k in range(1, 9):
    #       front, rear = (k - 1) // 2, k - 1 - (k - 1) // 2
    #       for mode in ("reflect", "circular"):
    #           min(s for s in range(1, 12) if F.pad(torch.rand(1, 1, s, s), [front, rear] * 2, mode=mode))
    @pytest.mark.parametrize(
        "kernel, border, too_small, big_enough",
        [
            ((4, 4), "reflect", (1, 1, 2, 2), (1, 1, 3, 3)),
            ((4, 4), "circular", (1, 1, 1, 1), (1, 1, 2, 2)),
            ((2, 2), "reflect", (1, 1, 1, 1), (1, 1, 2, 2)),
            ((6, 6), "reflect", (1, 1, 3, 3), (1, 1, 4, 4)),
            ((6, 6), "circular", (1, 1, 2, 2), (1, 1, 3, 3)),
            # the odd-extent circular boundary: `(k + 1) // 2` agrees with `k // 2` on every even row
            # above, and only an odd kernel at its exact minimum tells them apart
            ((5, 5), "circular", (1, 1, 1, 1), (1, 1, 2, 2)),
        ],
    )
    def test_convention_even_kernels_use_the_wider_asymmetric_pad_4559(
        self, kernel, border, too_small, big_enough, device, dtype
    ):
        if border == "reflect" and not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        small = torch.rand(*too_small).to(device=device, dtype=dtype)
        large = torch.rand(*big_enough).to(device=device, dtype=dtype)

        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match="RandomBoxBlur cannot filter an image this small"):
            _sync(K.RandomBoxBlur(kernel, border_type=border, p=1.0)(small).device)
        # one pixel more and it is kornia's job to run, not to refuse
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomBoxBlur(kernel, border_type=border, p=1.0)(large).shape == large.shape
        # the modes that invent their padding are untouched by the widening
        for invented in ("constant", "replicate"):
            if invented == "replicate" and not supports_replicate_padding(device, dtype):
                continue
            torch.manual_seed(_FORWARD_SEED)
            assert K.RandomBoxBlur(kernel, border_type=invented, p=1.0)(small).shape == small.shape

    # Zero padding pulls a border pixel toward 0: below the minimum of a positive image, above the maximum
    # of a negative one.
    # Snippet used to generate expected:
    #   neg = torch.full((1, 1, 7, 7), -1.0)
    #   torch.manual_seed(0); print(K.RandomBoxBlur((3, 3), border_type="constant", p=1.0)(neg).aminmax())
    @pytest.mark.parametrize("name", ["RandomBoxBlur", "RandomGaussianBlur", "RandomMotionBlur"])
    def test_convention_constant_border_pulls_the_border_toward_zero(self, device, dtype, name):
        factories = {
            "RandomBoxBlur": lambda: K.RandomBoxBlur((3, 3), border_type="constant", p=1.0),
            "RandomGaussianBlur": lambda: K.RandomGaussianBlur((3, 3), (1.0, 1.0), border_type="constant", p=1.0),
            "RandomMotionBlur": lambda: K.RandomMotionBlur(3, (0.0, 0.0), (0.0, 0.0), border_type="constant", p=1.0),
        }
        positive = torch.full((1, 1, 7, 7), 1.0, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        positive_out = factories[name]()(positive)
        low, high = positive_out.aminmax()
        assert float(low) < 1.0 and float(high) <= 1.0 + 1e-3, "a positive image is pulled below its minimum"
        negative = torch.full((1, 1, 7, 7), -1.0, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        negative_out = factories[name]()(negative)
        low, high = negative_out.aminmax()
        assert float(high) > -1.0, "a negative image is pulled above its maximum, not below its minimum"
        assert float(low) >= -1.0 - 1e-3
        # Toward zero and not another constant ``v``: ``out(+1) + out(-1) = 2 * (1 - s) * v`` vanishes only at 0.
        self.assert_close(positive_out + negative_out, torch.zeros_like(positive_out), atol=1e-6, rtol=0)


# Twice float32(0.9): the snow coefficient is drawn in float32, so this red channel's HLS lightness equals
# the drawn coefficient exactly in every dtype, which is what tells the coverage test's `<` from `<=`.
_SNOW_BOUNDARY_RED = 1.7999999523162842


class TestNoiseAndWeatherConventions(BaseTester):
    # ``scale`` is the target area fraction, ``ratio`` is height / width, the box is the half-open
    # ``[ys, ys + h) x [xs, xs + w)`` in ``_params``, and ``value`` is the literal fill.
    # Snippet used to generate expected:
    #   x = torch.ones(1, 1, 10, 20)
    #   torch.manual_seed(0); aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(2.0, 2.0), p=1.0); y = aug(x)
    #   print((y != 1).sum(), {k: v.flatten().tolist() for k, v in aug._params.items()})
    @pytest.mark.parametrize(("ratio", "value", "height", "width"), [(2.0, 0.0, 10, 5), (0.5, 0.3, 5, 10)])
    def test_convention_random_erasing_scale_is_area_and_ratio_is_height_over_width(
        self, device, dtype, ratio, value, height, width
    ):
        image = torch.ones(1, 1, 10, 20, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(ratio, ratio), value=value, p=1.0)
        out = aug(image)
        params = aug._params
        assert int(params["heights"][0]) == height
        assert int(params["widths"][0]) == width
        erased = out != 1.0
        assert int(erased.sum()) == 50 == int(0.25 * image.numel())
        # The box is exactly [ys, ys + h) x [xs, xs + w) -- half open, so the extents are h and w.
        rows, cols = _lit_extent(erased[0, 0].to(dtype))
        y_start, x_start = int(params["ys"][0]), int(params["xs"][0])
        assert rows == list(range(y_start, y_start + height))
        assert cols == list(range(x_start, x_start + width))
        self.assert_close(out[erased], out.new_full((50,), value))

    # The target box for scale 0.25 and ratio 3 on 10x20 is about 12.2 x 4.1: rounded to 12 x 4, then
    # clipped to the 10-row image, so 40 of the 50 requested pixels are erased.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(3.0, 3.0), p=1.0)
    #   y = aug(torch.ones(1, 1, 10, 20)); print((y != 1).sum(), aug._params["heights"], aug._params["widths"])
    def test_convention_random_erasing_box_is_rounded_and_clipped(self, device, dtype):
        image = torch.ones(1, 1, 10, 20, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(3.0, 3.0), p=1.0)
        out = aug(image)
        assert (int(aug._params["heights"][0]), int(aug._params["widths"][0])) == (10, 4)
        assert int((out != 1.0).sum()) == 40

    # A ``mask`` data key is erased in the same box, but filled with 0 rather than the image's ``value``.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   seq = K.AugmentationSequential(K.RandomErasing(scale=(0.25, 0.25), ratio=(2.0, 2.0), value=0.3, p=1.0),
    #                                  data_keys=["input", "mask"])
    #   oi, om = seq(torch.ones(1, 1, 10, 20), torch.ones(1, 1, 10, 20))
    def test_convention_random_erasing_erases_the_mask_in_the_same_box(self, device, dtype):
        image = torch.ones(1, 1, 10, 20, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 10, 20, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        pipeline = K.AugmentationSequential(
            K.RandomErasing(scale=(0.25, 0.25), ratio=(2.0, 2.0), value=0.3, p=1.0), data_keys=["input", "mask"]
        )
        out_image, out_mask = pipeline(image, mask)
        image_box = (out_image != 1.0)[0, 0]
        mask_box = (out_mask != 1.0)[0, 0]
        assert int(image_box.sum()) == 50
        assert torch.equal(image_box, mask_box)
        # The image carries `value`, the mask carries 0.
        self.assert_close(out_image[out_image != 1.0], out_image.new_full((50,), 0.3))
        self.assert_close(out_mask[mask_box[None, None]], out_mask.new_zeros(50))

    # The noise is added unclamped and recorded in full under ``_params["gaussian_noise"]``; ``mean`` is
    # an offset, not a target.
    def test_convention_random_gaussian_noise_is_additive_and_unclamped(self, device, dtype):
        constant = torch.full((2, 3, 7, 9), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomGaussianNoise(mean=0.0, std=1.0, p=1.0)
        noisy = aug(constant)
        assert float(noisy.min()) < 0.0
        assert float(noisy.max()) > 1.0
        assert aug._params["gaussian_noise"].shape == constant.shape
        self.assert_close(noisy, constant + aug._params["gaussian_noise"].to(device=device, dtype=dtype))
        # std=0 makes the addition exact, which is what pins `mean` as an offset and not a target.
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(K.RandomGaussianNoise(mean=2.0, std=0.0, p=1.0)(constant), constant + 2.0)

    # Parameter fields use the batched BCHW shape even when ``keepdim`` restores a CHW output.
    @pytest.mark.parametrize(
        "name",
        [
            "RandomGaussianNoise",
            "RandomGaussianIllumination",
            "RandomLinearIllumination",
            "RandomLinearCornerIllumination",
            "RandomPlasmaBrightness",
            "RandomPlasmaContrast",
            "RandomPlasmaShadow",
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_convention_intensity_parameter_fields_normalize_chw_shape(self, device, dtype, name, keepdim):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(3, 7, 9).to(device=device, dtype=dtype)
        aug, field_name, field_channels = _field_augmentation(name, keepdim=keepdim)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        assert out.shape == (image.shape if keepdim else (1, *image.shape))
        assert aug._params[field_name].shape == (1, field_channels, *image.shape[-2:])
        assert torch.equal(out, aug(image, params=aug._params))

    # At a partial probability the stored field keeps the full batch length; unselected rows pass through
    # and the parameter state replays exactly.
    @pytest.mark.parametrize(
        "name",
        [
            "RandomGaussianNoise",
            "RandomGaussianIllumination",
            "RandomLinearIllumination",
            "RandomLinearCornerIllumination",
            "RandomPlasmaBrightness",
            "RandomPlasmaContrast",
            "RandomPlasmaShadow",
        ],
    )
    def test_convention_intensity_parameter_fields_keep_original_batch_for_partial_probability(
        self, device, dtype, name
    ):
        # B=16 makes an all-or-nothing gate at p=0.5 unlikely (2 * 2 ** -16).
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(16, 3, 7, 9).to(device=device, dtype=dtype)
        aug, field_name, field_channels = _field_augmentation(name, p=0.5)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        selected = aug._params["batch_prob"].bool()
        assert 0 < int(selected.sum()) < image.shape[0]
        assert aug._params[field_name].shape == (image.shape[0], field_channels, *image.shape[-2:])
        assert torch.equal(out[~selected], image[~selected])
        assert torch.equal(out, aug(image, params=aug._params))

    def test_convention_gaussian_noise_same_on_batch_shares_one_normalized_field(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 7, 9).to(device=device, dtype=dtype)
        aug = K.RandomGaussianNoise(mean=0.0, std=1.0, p=1.0, same_on_batch=True)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        noise = aug._params["gaussian_noise"]
        assert noise.shape == (1, *image.shape[1:])
        self.assert_close(out, image + noise.to(device=device, dtype=dtype).expand_as(image))

    # ``amount`` is a per-pixel fraction (the rewrite mask is the same in every channel) and
    # ``salt_vs_pepper`` the salt share of the rewritten pixels; both are asserted within five binomial
    # standard deviations.
    @pytest.mark.parametrize("amount", [0.1, 0.3, 0.6])
    def test_convention_random_salt_and_pepper_amount_is_a_pixel_fraction(self, device, dtype, amount):
        image = torch.full((2, 3, 96, 160), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSaltAndPepperNoise(amount=(amount, amount), salt_vs_pepper=(0.25, 0.25), p=1.0)(image)
        changed = out != 0.5
        assert torch.equal(changed[:, 0], changed[:, 1])
        assert torch.equal(changed[:, 0], changed[:, 2])
        pixels = changed[:, 0].numel()
        fraction = float(changed[:, 0].float().mean())
        assert abs(fraction - amount) < 5.0 * (amount * (1.0 - amount) / pixels) ** 0.5
        # salt_vs_pepper splits the touched pixels; 0.25 means a quarter of them go to salt.
        salt = float((out[:, 0] == 1.0).float().sum())
        pepper = float((out[:, 0] == 0.0).float().sum())
        touched = salt + pepper
        assert abs(salt / touched - 0.25) < 5.0 * (0.25 * 0.75 / touched) ** 0.5

    # The poles are the literals 1.0 and 0.0 whatever the input's range.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   K.RandomSaltAndPepperNoise(amount=(0.3, 0.3), salt_vs_pepper=(1.0, 1.0), p=1.0)(torch.full((2, 3, 12, 20), 2.0))
    @pytest.mark.parametrize(("salt_vs_pepper", "written"), [(1.0, 1.0), (0.0, 0.0)])
    def test_convention_random_salt_and_pepper_writes_literal_zero_and_one(
        self, device, dtype, salt_vs_pepper, written
    ):
        image = torch.full((2, 3, 12, 20), 2.0, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSaltAndPepperNoise(amount=(0.3, 0.3), salt_vs_pepper=(salt_vs_pepper, salt_vs_pepper), p=1.0)(
            image
        )
        changed = out != 2.0
        assert int(changed.sum()) > 0
        self.assert_close(out[changed], out.new_full((int(changed.sum()),), written))
        # The other pole is never written at this setting.
        assert not bool((out == (1.0 - written)).any())

    # ``drop_height`` runs down rows and ``drop_width`` along columns, also on the transposed image.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); aug = K.RandomRain(number_of_drops=(1, 1), drop_height=(6, 6), drop_width=(1, 1), p=1.0)
    #   print((aug(torch.zeros(1, 1, 12, 30))[0, 0] > 0).nonzero().tolist())
    @pytest.mark.parametrize(("drop_height", "drop_width"), [(6, 1), (1, 6)])
    def test_convention_random_rain_drop_height_runs_down_rows(self, device, dtype, drop_height, drop_width):
        def extents(image: torch.Tensor) -> tuple[int, int]:
            torch.manual_seed(_FORWARD_SEED)
            aug = K.RandomRain(
                number_of_drops=(1, 1),
                drop_height=(drop_height, drop_height),
                drop_width=(drop_width, drop_width),
                p=1.0,
            )
            rows, cols = _lit_extent(aug(image)[0, 0])
            return len(rows), len(cols)

        tall_rows, tall_cols = extents(torch.zeros(1, 1, 12, 30, device=device, dtype=dtype))
        if drop_height > drop_width:
            assert tall_rows >= 5 and tall_cols <= 2
        else:
            assert tall_cols >= 5 and tall_rows <= 2
        # On a 30x12 image the arguments still mean rows and columns.
        wide_rows, wide_cols = extents(torch.zeros(1, 1, 30, 12, device=device, dtype=dtype))
        if drop_height > drop_width:
            assert wide_rows >= 5 and wide_cols <= 2
        else:
            assert wide_cols >= 5 and wide_rows <= 2

    # A drop is the literal 200/255, not a function of the image.
    def test_convention_random_rain_drop_value_is_two_hundred_over_255(self, device, dtype):
        image = torch.full((1, 1, 12, 30), 2.0, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomRain(number_of_drops=(5, 5), drop_height=(3, 3), drop_width=(1, 1), p=1.0)(image)
        self.assert_close(out.min(), out.new_tensor(200 / 255))
        self.assert_close(out.max(), out.new_tensor(2.0))

    # A drop as tall or as wide as the image is rejected with kornia's error on the forward pass (the
    # image size is only known there), and one pixel smaller runs.
    @pytest.mark.parametrize(("drop_height", "drop_width", "message"), [(12, 1, "Height of drop"), (1, 30, "Width of")])
    def test_convention_random_rain_rejects_a_drop_as_large_as_the_image(
        self, device, dtype, drop_height, drop_width, message
    ):
        image = torch.zeros(1, 1, 12, 30, device=device, dtype=dtype)
        # Constructing an oversized drop is allowed; only the forward, which knows the image, raises.
        aug = K.RandomRain(
            number_of_drops=(1, 1),
            drop_height=(drop_height, drop_height),
            drop_width=(drop_width, drop_width),
            p=1.0,
        )
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(BaseError, match=message):
            _sync(aug(image).device)
        # One pixel smaller on the same axis runs.
        inside_height, inside_width = min(drop_height, 11), min(drop_width, 29)
        torch.manual_seed(_FORWARD_SEED)
        smaller = K.RandomRain(
            number_of_drops=(1, 1),
            drop_height=(inside_height, inside_height),
            drop_width=(inside_width, inside_width),
            p=1.0,
        )
        assert smaller(image).shape == image.shape

    # Issue #4810: a drop of size n >= 2 paints n pixels over n + 1 rows, skipping one.  A fix that paints a
    # contiguous drop flips it.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   y = K.RandomRain(number_of_drops=(1, 1), drop_height=(h, h), drop_width=(0, 0), p=1.0)(torch.zeros(1, 1, 6, 10))
    #   print(sorted({r for r, _ in (y[0, 0] != 0).nonzero().tolist()}))
    @pytest.mark.parametrize(("height", "rows"), [(2, [0, 2]), (5, [0, 1, 2, 3, 5])])
    def test_wart_random_rain_drop_skips_a_row_4810(self, device, dtype, height, rows):
        image = torch.zeros(1, 1, 6, 10, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomRain(number_of_drops=(1, 1), drop_height=(height, height), drop_width=(0, 0), p=1.0)(image)
        lit = (out[0, 0] != 0).nonzero()
        assert (lit[:, 0] - lit[:, 0].min()).tolist() == rows

    # The three integer ranges are closed and uniform (#4567), including signed widths that straddle or
    # end at zero.  The 15% band is wide for the 16-bin height at 22000 draws; what it has to separate is
    # a factor of two (a truncating draw gives ``0`` twice its share).
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   p = K.RandomRain(number_of_drops=(2, 4), drop_height=(5, 20), drop_width=(-5, 5), p=1.0)
    #   print(torch.bincount(p.forward_parameters((22000, 1, 64, 64))["drop_width_factor"] + 5))
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("drop_width", [(-5, 5), (-1, 1), (-5, -1), (-3, 0)])
    def test_convention_random_rain_integer_ranges_are_closed_and_uniform(self, drop_width):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomRain(number_of_drops=(2, 4), drop_height=(5, 20), drop_width=drop_width, p=1.0)
        params = aug.forward_parameters((22000, 1, 64, 64))
        for key, low, high in (
            ("number_of_drops_factor", 2, 4),
            ("drop_height_factor", 5, 20),
            ("drop_width_factor", *drop_width),
        ):
            drawn = params[key].flatten()
            assert drawn.dtype == torch.long
            assert (int(drawn.min()), int(drawn.max())) == (low, high), key
            counts = torch.bincount(drawn - low, minlength=high - low + 1)
            expected = drawn.numel() / (high - low + 1)
            assert bool(((counts > 0.85 * expected) & (counts < 1.15 * expected)).all()), (key, counts.tolist())

    # The largest ``torch.rand`` value maps onto ``hi + 1`` in float32 (``5 + (1 - 2 ** -24) * 16 == 21.0``);
    # kornia clamps it back to ``hi``.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_largest_draw_stays_inside_the_closed_range(self, monkeypatch):
        largest = 1 - 2**-24
        real_rand = torch.rand
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.full_like(real_rand(*args, **kwargs), largest))
        aug = K.RandomRain(number_of_drops=(2, 4), drop_height=(5, 20), drop_width=(-5, 5), p=1.0)
        params = aug.forward_parameters((4, 1, 64, 64))
        # The patch has to reach the generator's own draws, or the pin below is vacuous.
        assert float(params["coordinates_factor"].min()) == largest
        assert params["number_of_drops_factor"].tolist() == [4, 4, 4, 4]
        assert params["drop_height_factor"].tolist() == [20, 20, 20, 20]
        assert params["drop_width_factor"].tolist() == [5, 5, 5, 5]

    # Each integer range is validated at construction: reversed, fractional, non-finite or not a pair is
    # a ValueError naming the argument.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(
        "bounds",
        [(6, 5), (0.5, 2.5), (1, 2.5), (float("nan"), 5), (5, float("nan")), (1, float("inf")), (1, 2, 3), (1,), 5],
    )
    @pytest.mark.parametrize("name", ["number_of_drops", "drop_height", "drop_width"])
    def test_convention_random_rain_rejects_an_invalid_range(self, name, bounds):
        ranges = {"number_of_drops": (1, 1), "drop_height": (1, 1), "drop_width": (1, 1)}
        ranges[name] = bounds
        with pytest.raises(ValueError, match=name):
            K.RandomRain(p=1.0, **ranges)

    # Integral floats and a tensor pair are the same closed range written another way.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_accepts_integral_float_and_tensor_ranges(self):
        for name in ("number_of_drops", "drop_height", "drop_width"):
            ranges = {"number_of_drops": (1, 1), "drop_height": (1, 1), "drop_width": (1, 1)}
            ranges[name] = (1.0, 2.0)
            K.RandomRain(p=1.0, **ranges)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomRain(p=1.0, number_of_drops=(1, 1), drop_height=torch.tensor([2, 4]), drop_width=(1, 1))
        drawn = aug.forward_parameters((2000, 1, 64, 64))["drop_height_factor"]
        assert sorted(set(drawn.flatten().tolist())) == [2, 3, 4]

    # A drop starts anywhere that keeps it inside the image, so over many draws every row and column is
    # painted, for a single pixel and for drops slanting either way (#4604).
    @pytest.mark.parametrize("drop_height, drop_width", [(1, 0), (2, 2), (2, -2)])
    def test_convention_random_rain_paints_every_row_and_column_4604(self, device, dtype, drop_height, drop_width):
        image = torch.zeros(1, 3, 6, 10, device=device, dtype=dtype)
        # `.to` so the coordinate sampler runs in the test's dtype, where a half `rand` can return 1.0 (#4553).
        aug = K.RandomRain(
            number_of_drops=(20, 20), drop_height=(drop_height, drop_height), drop_width=(drop_width, drop_width), p=1.0
        ).to(device=device, dtype=dtype)
        rows, cols = set(), set()
        for seed in range(300):
            torch.manual_seed(seed)
            lit = (aug(image)[0, 0] != 0).nonzero()
            rows |= set(lit[:, 0].tolist())
            cols |= set(lit[:, 1].tolist())
        assert sorted(rows) == list(range(6)) and sorted(cols) == list(range(10))

    # A single drop paints a bounding box of exactly ``(h, |w|)`` inside the image for every legal start
    # (#4604); a drop wrapped across opposite edges by a negative index would still satisfy the union pin.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_single_drop_box_is_its_size_4604(self):
        for height, width in itertools.product(range(2, 8), repeat=2):
            image = torch.zeros(1, 1, height, width)
            for drop_height in range(1, height):
                for drop_width in range(-(width - 1), width):
                    aug = K.RandomRain(
                        number_of_drops=(1, 1),
                        drop_height=(drop_height, drop_height),
                        drop_width=(drop_width, drop_width),
                        p=1.0,
                    )
                    expected = (drop_height, abs(drop_width)) if max(drop_height, abs(drop_width)) > 1 else (0, 0)
                    for seed in range(2):
                        torch.manual_seed(seed)
                        lit = (aug(image)[0, 0] != 0).nonzero()
                        rows, cols = lit[:, 0].tolist(), lit[:, 1].tolist()
                        case = (height, width, drop_height, drop_width, seed)
                        assert (max(rows) - min(rows), max(cols) - min(cols)) == expected, case
                        assert min(rows) >= 0 and max(rows) < height, case
                        assert min(cols) >= 0 and max(cols) < width, case

    # Every legal start row is equally likely, not merely reachable (#4604); single-pixel drops on a wide
    # image, so each painted cell is one start.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   aug = K.RandomRain(number_of_drops=(100, 100), drop_height=(1, 1), drop_width=(0, 0), p=1.0)
    #   print(torch.bincount((aug(torch.zeros(200, 1, 6, 2000))[:, 0] != 0).nonzero()[:, 1], minlength=6))
    @pytest.mark.device_agnostic
    def test_convention_random_rain_start_row_is_uniform_4604(self):
        aug = K.RandomRain(number_of_drops=(100, 100), drop_height=(1, 1), drop_width=(0, 0), p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        lit = (aug(torch.zeros(200, 1, 6, 2000))[:, 0] != 0).nonzero()
        counts = torch.bincount(lit[:, 1], minlength=6)
        expected = int(counts.sum()) / 6
        assert bool(((counts > 0.85 * expected) & (counts < 1.15 * expected)).all()), counts.tolist()

    # A start draw of exactly 1.0 (possible for a half ``rand``, #4553) is clamped to the last legal start:
    # the drop reaches the last row and the last column and stays inside (#4604).
    @pytest.mark.device_agnostic
    def test_convention_random_rain_start_draw_of_one_stays_inside_4604(self, monkeypatch):
        real_rand = torch.rand
        monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.ones_like(real_rand(*a, **kw)))
        aug = K.RandomRain(number_of_drops=(1, 1), drop_height=(2, 2), drop_width=(-2, -2), p=1.0)
        # The patch has to reach the generator's own draws, or the pin is vacuous.
        assert float(aug.forward_parameters((1, 1, 6, 10))["coordinates_factor"].min()) == 1.0
        lit = (aug(torch.zeros(1, 1, 6, 10))[0, 0] != 0).nonzero().tolist()
        assert sorted(lit) == [[3, 9], [5, 7]]

    # ``same_on_batch=True`` shares the drop count, the drop sizes and the coordinates; without it each
    # sample draws its own.
    def test_convention_random_rain_same_on_batch_draws_one_drop_count(self, device, dtype):
        # B=16 makes an accidentally constant unshared draw over {1, 2, 3} unlikely.
        batch_size = 16
        image = torch.zeros(batch_size, 1, 12, 30, device=device, dtype=dtype)
        drawn = {}
        shared = {}
        for same_on_batch in (True, False):
            torch.manual_seed(_FORWARD_SEED)
            aug = K.RandomRain(
                number_of_drops=(1, 50), drop_height=(1, 3), drop_width=(1, 3), p=1.0, same_on_batch=same_on_batch
            )
            aug(image)
            drawn[same_on_batch] = aug._params["number_of_drops_factor"].flatten().tolist()
            shared[same_on_batch] = aug._params
        assert len(set(drawn[True])) == 1
        assert len(drawn[True]) == batch_size
        assert len(set(drawn[False])) > 1
        for key in ("drop_height_factor", "drop_width_factor"):
            assert len(set(shared[True][key].flatten().tolist())) == 1, key
            assert len(set(shared[False][key].flatten().tolist())) > 1, key
        coordinates = shared[True]["coordinates_factor"]
        assert all(torch.equal(coordinates[0], coordinates[row]) for row in range(1, batch_size))
        assert not torch.equal(shared[False]["coordinates_factor"][0], shared[False]["coordinates_factor"][1])

    # ``snow_coefficient`` is validated against [0, 1] at construction (the channel-count rule is in the
    # precondition table below).
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("coefficient", [1.5, -0.5])
    def test_convention_random_snow_rejects_a_coefficient_outside_unit_range(self, coefficient):
        with pytest.raises(BaseError, match="between 0 and 1"):
            K.RandomSnow(snow_coefficient=(coefficient, coefficient), p=1.0)

    # ``snow_coefficient`` and ``brightness`` are drawn per sample; ``same_on_batch=True`` shares both.
    def test_convention_random_snow_draws_one_value_per_sample(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 7, 9).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomSnow(snow_coefficient=(0.2, 0.8), brightness=(1.5, 3.0), p=1.0)
        aug(image)
        torch.manual_seed(_FORWARD_SEED)
        many = aug.forward_parameters((256, 3, 7, 9))
        for key, low, high in (("snow_coefficient", 0.2, 0.8), ("brightness", 1.5, 3.0)):
            drawn = aug._params[key].flatten()
            assert drawn.shape == (4,)
            assert len(set(drawn.tolist())) == 4
            assert float(drawn.min()) >= low and float(drawn.max()) <= high
            # Reach, not only containment.
            width = high - low
            assert float(many[key].min()) < low + 0.1 * width and float(many[key].max()) > high - 0.1 * width
        torch.manual_seed(_FORWARD_SEED)
        shared = K.RandomSnow(snow_coefficient=(0.2, 0.8), brightness=(1.5, 3.0), p=1.0, same_on_batch=True)
        shared(image)
        for key in ("snow_coefficient", "brightness"):
            assert len(set(shared._params[key].flatten().tolist())) == 1

    # Only a covered pixel (lightness below the coefficient) is scaled and clamped: covered (1.5, 0, 0)
    # turns white, missed (2, 1.5, 1.5) keeps its values, covered (1.2, -2, -2) turns black.  The two
    # singular lightnesses collapse whether covered or not: exactly 1 comes back white, exactly 0 black.
    # The last pixel's lightness equals the coefficient, so coverage is strict ``<``.
    # Snippet used to generate expected:
    #   x = torch.tensor([[1.5, 0.0, 0.0], [2.0, 1.5, 1.5], [1.2, -2.0, -2.0], [1.5, 0.5, 0.5],
    #                     [2.0, -2.0, -2.0], [1.7999999523162842, 0.0, 0.0]]).T.reshape(1, 3, 1, 6)
    #   torch.manual_seed(0); print(K.RandomSnow(snow_coefficient=(0.9, 0.9), brightness=(2.0, 2.0), p=1.0)(x))
    def test_convention_random_snow_clamps_only_covered_lightness(self, device, dtype):
        pixels = torch.tensor(
            [
                [1.5, 0.0, 0.0],
                [2.0, 1.5, 1.5],
                [1.2, -2.0, -2.0],
                [1.5, 0.5, 0.5],
                [2.0, -2.0, -2.0],
                [_SNOW_BOUNDARY_RED, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomSnow(snow_coefficient=(0.9, 0.9), brightness=(2.0, 2.0), p=1.0)
        out = aug(pixels.T.reshape(1, 3, 1, 6).contiguous())[0, :, 0].T
        self.assert_close(out[:3], pixels.new_tensor([[1.0, 1.0, 1.0], [2.0, 1.5, 1.5], [0.0, 0.0, 0.0]]))
        # Lightness exactly at the coefficient is missed, so the pixel survives the round trip unchanged.
        self.assert_close(out[5], pixels[5])
        self.assert_close(out[3], pixels.new_ones(3))
        self.assert_close(out[4], pixels.new_zeros(3))

    # Every achromatic pixel -- black, gray or white -- stays finite through the HLS round trip (#4571).
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_convention_random_snow_achromatic_pixel_is_finite_4571(self, device, dtype, value):
        gray = torch.full((1, 3, 2, 2), value, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSnow(p=1.0)(gray)
        assert bool(torch.isfinite(out).all())
        expected = gray.new_full(gray.shape, value)
        self.assert_close(out, expected)

    # The channel-count precondition, one table: "1 or 3" against "3 only".  The type and kornia's own
    # message are asserted, so an unrelated failure does not satisfy a row.
    # Snippet used to generate expected:
    #   for c in (1, 2, 4): torch.manual_seed(0); factory()(torch.rand(2, c, 32, 32))
    @pytest.mark.parametrize(
        ("name", "accepts_one", "error", "match"),
        [
            ("ColorJiggle", False, ValueError, r"Input size must have a shape of \(\*, 3, H, W\)"),
            ("RandomHue", False, ValueError, r"Input size must have a shape of \(\*, 3, H, W\)"),
            ("RandomSaturation", False, ValueError, r"Input size must have a shape of \(\*, 3, H, W\)"),
            ("RandomJPEG", False, ShapeError, r"Shape mismatch at dimension 0: expected 3, got"),
            ("RandomPlanckianJitter", False, ShapeError, r"Shape mismatch at dimension 0: expected 3, got"),
            ("RandomRGBShift", False, ImageError, r"Not a color tensor"),
            ("RandomSnow", False, BaseError, r"Number of color channels should be 3\."),
            ("RandomRain", True, BaseError, r"Number of color channels should be 1 or 3\."),
            ("RandomSaltAndPepperNoise", True, BaseError, r"Number of color channels should be 1 or 3\."),
        ],
    )
    def test_convention_intensity_channel_count_preconditions(self, device, dtype, name, accepts_one, error, match):
        if name == "RandomJPEG" and not supports_replicate_padding(device, dtype):
            # The 3-channel baseline call needs the codec's replication padding.
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        factories = {
            "ColorJiggle": lambda: K.ColorJiggle(0.0, 0.0, 0.2, 0.0, p=1.0),
            "RandomHue": lambda: K.RandomHue((-0.2, 0.2), p=1.0),
            "RandomSaturation": lambda: K.RandomSaturation((0.7, 1.3), p=1.0),
            "RandomJPEG": lambda: K.RandomJPEG((50.0, 90.0), p=1.0),
            "RandomPlanckianJitter": lambda: K.RandomPlanckianJitter(p=1.0),
            "RandomRGBShift": lambda: K.RandomRGBShift(p=1.0),
            "RandomSnow": lambda: K.RandomSnow(p=1.0),
            "RandomRain": lambda: K.RandomRain(number_of_drops=(2, 3), drop_height=(2, 3), drop_width=(1, 2), p=1.0),
            "RandomSaltAndPepperNoise": lambda: K.RandomSaltAndPepperNoise(p=1.0),
        }
        torch.manual_seed(_FIXTURE_SEED)
        three = torch.rand(2, 3, 32, 32).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert factories[name]()(three).shape == three.shape, f"{name} must accept a 3-channel image"
        for channels in (1, 2, 4):
            torch.manual_seed(_FIXTURE_SEED)
            image = torch.rand(2, channels, 32, 32).to(device=device, dtype=dtype)
            torch.manual_seed(_FORWARD_SEED)
            if channels == 1 and accepts_one:
                assert factories[name]()(image).shape == image.shape
                continue
            with pytest.raises(error, match=match):
                _sync(factories[name]()(image).device)


class TestIlluminationAndNormalizeConventions(BaseTester):
    # The three *Illumination classes add a gradient recorded as ``_params["gradient"]`` and clamp the sum
    # into [0, 1]; ``sign`` is drawn per sample and fixes the direction.  At gain 0.8 on a constant 0.5 the
    # clamp engages on an in-range image.
    # Snippet used to generate expected:
    #   c = torch.full((2, 3, 7, 9), 0.5)
    #   torch.manual_seed(0); a = K.RandomGaussianIllumination(gain=(0.8, 0.8), sign=(1.0, 1.0), p=1.0)
    #   y = a(c); raw = c + a._params["gradient"]; print(torch.equal(y, raw.clamp(0, 1)), torch.equal(y, raw))
    @pytest.mark.parametrize(
        "name", ["RandomGaussianIllumination", "RandomLinearIllumination", "RandomLinearCornerIllumination"]
    )
    def test_convention_illumination_adds_a_signed_gradient_and_clamps_to_unit_range(self, device, dtype, name):
        aug = _illumination(name)
        constant = torch.full((2, 3, 7, 9), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(constant)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        assert aug._params["gradient"].shape == constant.shape
        assert float(aug._params["gradient"].float().abs().max()) > 0.0

        # -1 can only darken, +1 can only brighten; the output is the clamped sum, not the raw one.
        for sign, low, high in ((-1.0, 0.0, 0.5), (1.0, 0.5, 1.0)):
            torch.manual_seed(_FORWARD_SEED)
            clamped_aug = _illumination(name, gain=(0.8, 0.8), sign=(sign, sign))
            clamped = clamped_aug(constant)
            gradient = clamped_aug._params["gradient"]
            if sign > 0:
                assert float(gradient.float().min()) >= 0.0
            else:
                assert float(gradient.float().max()) <= 0.0
            assert float(clamped.min()) >= low - 1e-3 and float(clamped.max()) <= high + 1e-3
            raw = constant + gradient.to(device=device, dtype=dtype)
            self.assert_close(clamped, raw.clamp(0.0, 1.0))
            assert not torch.equal(clamped, raw), "the clamp did not engage, so this leg proves nothing"

        # With the default range one batch holds both directions; same_on_batch=True shares one gradient.
        # B=16 makes an all-same-sign draw unlikely (2 * 2 ** -16).
        batch_size = 16
        batch = torch.full((batch_size, 3, 7, 9), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        per_sample = _illumination(name)
        per_sample(batch)
        directions = [float(per_sample._params["gradient"][b].float().max()) for b in range(batch_size)]
        assert any(value > 0.0 for value in directions) and any(value <= 0.0 for value in directions)
        torch.manual_seed(_FORWARD_SEED)
        shared = _illumination(name, same_on_batch=True)
        shared(batch)
        gradients = shared._params["gradient"]
        assert all(torch.equal(gradients[0], gradients[b]) for b in range(batch_size))
        # The edge, corner or centre where the gradient is strongest is also drawn per sample.
        flat = per_sample._params["gradient"].flatten(1).float().abs()
        assert len(set(flat.argmax(1).tolist())) > 1, "the strongest-gradient location is not drawn per sample"

    # Issue #4811: ``center`` is rounded to a whole pixel, half to even, so ``center=0.5`` peaks one column
    # past the middle of a 7-wide image but on it for a 9-wide one.  A fix that maps ``center`` to the pixel
    # centre flips the 7-wide leg.
    # Snippet used to generate expected:
    #   a = K.RandomGaussianIllumination(gain=(0.5, 0.5), sigma=(0.2, 0.2), center=(0.5, 0.5), sign=(1., 1.), p=1.)
    #   torch.manual_seed(0); a(torch.zeros(1, 1, w, w)); print(int(a._params["gradient"][0, 0].sum(0).argmax()))
    @pytest.mark.parametrize(("width", "peak"), [(7, 4), (9, 4)])
    def test_wart_random_gaussian_illumination_center_rounds_half_to_even_4811(self, device, dtype, width, peak):
        aug = K.RandomGaussianIllumination(gain=(0.5, 0.5), sigma=(0.2, 0.2), center=(0.5, 0.5), sign=(1.0, 1.0), p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        aug(torch.zeros(1, 1, width, width, device=device, dtype=dtype))
        assert int(aug._params["gradient"][0, 0].float().sum(0).argmax()) == peak

    # The three classes round-trip through pickle, deepcopy and torch.save (#4435), and the copy
    # reproduces the original's output under the same seed.
    @pytest.mark.parametrize(
        "name", ["RandomGaussianIllumination", "RandomLinearIllumination", "RandomLinearCornerIllumination"]
    )
    def test_convention_illumination_augmentations_are_picklable(self, device, dtype, name):
        constant = torch.full((2, 3, 7, 9), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        expected = _illumination(name)(constant)
        torch.manual_seed(_FORWARD_SEED)
        unpickled = pickle.loads(pickle.dumps(_illumination(name)))  # noqa: S301
        assert torch.equal(unpickled(constant), expected)
        torch.manual_seed(_FORWARD_SEED)
        cloned = copy.deepcopy(_illumination(name))
        assert torch.equal(cloned(constant), expected)
        buffer = io.BytesIO()
        torch.save(_illumination(name), buffer)
        buffer.seek(0)
        torch.manual_seed(_FORWARD_SEED)
        reloaded = torch.load(buffer, weights_only=False)
        assert torch.equal(reloaded(constant), expected)

    # The illumination is finite at every admitted ``sigma``, including ``0`` and a small absolute width
    # ``sigma * axis`` on an even axis (#4589).  The kernel itself is pinned in tests/filters/test_gaussian.py.
    @pytest.mark.parametrize("size", [3, 4])
    def test_convention_random_gaussian_illumination_is_finite_at_every_sigma(self, device, dtype, size):
        image = torch.full((1, 3, size, size), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert bool(K.RandomGaussianIllumination(p=1.0, sigma=0.0)(image).isfinite().all())
        for sigma in (0.005, 0.01):
            torch.manual_seed(_FORWARD_SEED)
            small = K.RandomGaussianIllumination(p=1.0, sigma=(sigma, sigma), center=(0.5, 0.5))(image)
            assert bool(small.isfinite().all()), (size, sigma)
        # Absolute widths straddling the float32 underflow threshold (about 0.034).
        for axis, small_sigma in ((4, 0.005), (8, 0.005), (8, 0.002), (64, 0.0005), (64, 0.00055)):
            square = torch.full((1, 3, axis, axis), 0.5, device=device, dtype=dtype)
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomGaussianIllumination(p=1.0, sigma=(small_sigma, small_sigma), center=(0.5, 0.5))(square)
            assert bool(out.isfinite().all()), (axis, small_sigma)
        # And with a drawn centre.
        for odd in (1, 3, 5, 7):
            for seed in range(16):
                torch.manual_seed(seed)
                odd_image = torch.full((1, 3, odd, odd), 0.5, device=device, dtype=dtype)
                assert bool(K.RandomGaussianIllumination(p=1.0, sigma=(0.005, 0.005))(odd_image).isfinite().all()), (
                    odd,
                    seed,
                )
        torch.manual_seed(_FORWARD_SEED)
        assert bool(K.RandomGaussianIllumination(p=1.0)(image).isfinite().all())

    # Issue #4807: the classes with their own ``.compile()`` store compiled callables. Pickling stores the
    # uncompiled ones and unpickling compiles them again, so like the linear illumination classes (torch's
    # ``Module.compile``) they pickle, pass through torch.save and still run after the round trip.
    # "compile" in the name is load-bearing: conftest deselects it unless KORNIA_TEST_OPTIMIZER is set, which
    # keeps ``torch.compile``'s process-wide side effects (it disables Distribution argument validation) out
    # of the ordinary CPU legs.
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize(
        ("name", "own_compile"),
        [
            ("RandomGaussianIllumination", True),
            ("RandomGaussianBlur", True),
            ("ColorJitter", True),
            ("RandomLinearIllumination", False),
            ("RandomLinearCornerIllumination", False),
        ],
    )
    @pytest.mark.device_agnostic
    def test_compiled_intensity_augmentation_pickles(self, name, own_compile):
        if name == "RandomGaussianBlur":
            compiled = K.RandomGaussianBlur((3, 3), (1.0, 1.0), p=1.0)
        elif name == "ColorJitter":
            compiled = K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)
        else:
            compiled = _illumination(name)
        validate_args = Distribution._validate_args
        try:
            compiled.compile()
        finally:
            Distribution.set_default_validate_args(validate_args)
        constant = torch.full((2, 3, 7, 9), 0.5)
        torch.manual_seed(_FORWARD_SEED)
        cloned = copy.deepcopy(compiled)
        assert isinstance(cloned, type(compiled))
        assert cloned(constant).shape == constant.shape
        torch.save(compiled, io.BytesIO())
        try:
            restored = pickle.loads(pickle.dumps(compiled))  # noqa: S301
        finally:
            Distribution.set_default_validate_args(validate_args)
        assert isinstance(restored, type(compiled))
        assert restored(constant).shape == constant.shape
        if own_compile:
            # Compiled again on load, with the arguments of the original compile() call.
            assert restored._compile_kwargs == compiled._compile_kwargs

    # The RandomPlasma* classes clamp into [0, 1] and record the fractal as ``_params["plasma"]``, so a
    # replay with ``params=`` reproduces the output bitwise.  The replay runs on a non-constant image,
    # because a constant 0.5 is a fixed point of the plasma contrast transform.
    @pytest.mark.parametrize("name", ["RandomPlasmaBrightness", "RandomPlasmaContrast", "RandomPlasmaShadow"])
    def test_convention_plasma_clamps_and_replays_from_params(self, device, dtype, name):
        constant = torch.full((2, 3, 7, 9), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        constant_out = _plasma(name)(constant)
        assert float(constant_out.min()) >= 0.0
        assert float(constant_out.max()) <= 1.0

        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 32, 32).to(device=device, dtype=dtype)
        aug = _plasma(name)
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        assert "plasma" in aug._params
        # The transform actually moved this fixture, so an identity would not satisfy the replay.
        assert float((out - image).abs().max()) > 0.03
        assert torch.equal(aug(image, params=aug._params), out)

    # With same_on_batch=True the RandomPlasma* classes share their scalar draws and one expanded fractal
    # map (#4570).
    @pytest.mark.parametrize("name", ["RandomPlasmaBrightness", "RandomPlasmaContrast", "RandomPlasmaShadow"])
    def test_convention_plasma_same_on_batch_shares_map_4570(self, device, dtype, name):
        image = torch.full((4, 3, 8, 8), 0.3, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = getattr(K, name)(p=1.0, same_on_batch=True)
        out = aug(image)
        for key, value in aug._params.items():
            if key not in ("plasma", "forward_input_shape"):
                assert all(torch.equal(value[0], value[b]) for b in range(4)), f"{key} is not shared"
        assert torch.equal(aug._params["plasma"][0], aug._params["plasma"][1])
        # Expanded, not repeated: the stored map has batch stride 0.
        assert aug._params["plasma"].stride()[0] == 0
        assert torch.equal(out[0], out[1])

    # A one-pixel spatial axis is accepted.
    @pytest.mark.parametrize("name", ["RandomPlasmaBrightness", "RandomPlasmaContrast", "RandomPlasmaShadow"])
    @pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 1, 8), (1, 3, 8, 1)])
    def test_convention_plasma_accepts_a_single_pixel_axis(self, device, dtype, name, shape):
        image = torch.full(shape, 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert _plasma(name)(image).shape == shape

    # ``mean`` and ``std`` accept a float, a sequence, a 1- or C-tensor and a per-sample (B, C) tensor, and
    # each form rescales the image; a (B, C) statistic is applied to its own sample.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(4, 3, 6, 8)
    #   torch.manual_seed(0); print((K.Normalize(mean=0.5, std=0.25, p=1.0)(x) - x).abs().max())
    @pytest.mark.parametrize("form", ["float", "list3", "tuple3", "tensor1", "tensor3", "tensorB3"])
    @pytest.mark.parametrize(("cls", "moved"), [(K.Normalize, 1.0), (K.Denormalize, 0.25)])
    def test_convention_normalize_accepts_scalar_sequence_and_per_sample_statistics(
        self, device, dtype, cls, moved, form
    ):
        mean, std = _statistics(form, device, dtype)
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = cls(mean=mean, std=std, p=1.0)(image)
        assert out.shape == image.shape
        assert float((out - image).abs().max()) > moved
        if form == "tensorB3":
            # The exact field: the threshold above cannot tell a per-sample application from a row-0 one.
            stats = torch.arange(1, 13, device=device, dtype=dtype).reshape(4, 3) / 10.0
            field = stats[..., None, None]
            torch.manual_seed(_FORWARD_SEED)
            per_sample = cls(mean=stats, std=stats, p=1.0)(image)
            expected = image * field + field if cls is K.Denormalize else (image - field) / field
            self.assert_close(per_sample, expected)

    # Denormalize inverts Normalize built with the same statistics, checked on a forward that moved the
    # tensor, so a no-op cannot satisfy it; integer statistics round-trip too (#4573).
    @pytest.mark.parametrize("form", ["float", "list3", "tensor1", "tensorB3", "int"])
    def test_convention_denormalize_inverts_normalize(self, device, dtype, form):
        mean, std = (0, 4) if form == "int" else _statistics(form, device, dtype)
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        normalized = K.Normalize(mean=mean, std=std, p=1.0)(image)
        assert float((normalized - image).abs().max()) > 0.5
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(K.Denormalize(mean=mean, std=std, p=1.0)(normalized), image)

    # A mean or std whose length is neither 1 nor the channel count is rejected.
    @pytest.mark.parametrize("cls", [K.Normalize, K.Denormalize])
    def test_convention_normalize_rejects_a_statistic_of_the_wrong_length(self, device, dtype, cls):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 6, 8).to(device=device, dtype=dtype)
        mean = torch.tensor([0.4, 0.5], device=device, dtype=dtype)
        std = torch.tensor([0.2, 0.25], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match="do not match"):
            _sync(cls(mean=mean, std=std, p=1.0)(image).device)

    # Normalize and Denormalize hard-code ``same_on_batch=True`` and take no such argument, so ``p`` gates
    # the whole batch together.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(8, 3, 6, 8)
    #   torch.manual_seed(seed); aug = K.Normalize(mean=0.5, std=0.25, p=0.5); aug(x); print(aug._params["batch_prob"])
    @pytest.mark.parametrize("cls", [K.Normalize, K.Denormalize])
    def test_convention_normalize_p_gates_the_whole_batch(self, device, dtype, cls):
        assert "same_on_batch" not in inspect.signature(cls.__init__).parameters
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(8, 3, 6, 8).to(device=device, dtype=dtype)
        applied = 0
        for seed in range(6):
            torch.manual_seed(seed)
            aug = cls(mean=0.5, std=0.25, p=0.5)
            out = aug(image)
            gate = aug._params["batch_prob"].flatten()
            assert gate.shape == (8,)
            unchanged = sum(1 for b in range(8) if torch.equal(out[b], image[b]))
            assert unchanged == int((gate == 0).sum())
            assert len(set(gate.tolist())) == 1
            applied += 8 - unchanged
        # Both outcomes occur over these six seeds.
        assert 0 < applied < 48
        # p=0.0 is the identity.
        torch.manual_seed(_FORWARD_SEED)
        assert torch.equal(cls(mean=0.5, std=0.25, p=0.0)(image), image)

    # The statistics live in ``flags``, not in buffers: ``state_dict()`` is empty and ``.to()`` leaves them
    # alone (the forward casts them to the input).
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("cls", [K.Normalize, K.Denormalize])
    def test_convention_normalize_keeps_no_state(self, cls):
        aug = cls(mean=0.5, std=0.25, p=1.0)
        assert list(aug.state_dict()) == []
        assert [name for name, _ in aug.named_buffers()] == []
        assert sorted(aug.flags) == ["mean", "std"]
        before = aug.flags["mean"].clone()
        moved = aug.to(torch.float64)
        assert moved.flags["mean"].dtype == torch.float32
        assert torch.equal(moved.flags["mean"], before)


def _illumination(name: str, **kwargs):
    """Point-range constructors for the three *Illumination classes."""
    classes = {
        "RandomGaussianIllumination": K.RandomGaussianIllumination,
        "RandomLinearIllumination": K.RandomLinearIllumination,
        "RandomLinearCornerIllumination": K.RandomLinearCornerIllumination,
    }
    kwargs.setdefault("gain", (0.5, 0.5))
    return classes[name](p=1.0, **kwargs)


def _plasma(name: str):
    """Default constructors for the three RandomPlasma* classes."""
    factories = {
        "RandomPlasmaBrightness": lambda: K.RandomPlasmaBrightness(p=1.0),
        "RandomPlasmaContrast": lambda: K.RandomPlasmaContrast(p=1.0),
        "RandomPlasmaShadow": lambda: K.RandomPlasmaShadow(p=1.0),
    }
    return factories[name]()


def _field_augmentation(name: str, p: float = 1.0, keepdim: bool = False):
    """Return a field-producing augmentation, its parameter key, and its field channel count."""
    factories = {
        "RandomGaussianNoise": lambda: (
            K.RandomGaussianNoise(mean=0.0, std=1.0, p=p, keepdim=keepdim),
            "gaussian_noise",
            3,
        ),
        "RandomGaussianIllumination": lambda: (
            K.RandomGaussianIllumination(gain=(0.1, 0.1), p=p, keepdim=keepdim),
            "gradient",
            3,
        ),
        "RandomLinearIllumination": lambda: (
            K.RandomLinearIllumination(gain=(0.1, 0.1), p=p, keepdim=keepdim),
            "gradient",
            3,
        ),
        "RandomLinearCornerIllumination": lambda: (
            K.RandomLinearCornerIllumination(gain=(0.1, 0.1), p=p, keepdim=keepdim),
            "gradient",
            3,
        ),
        "RandomPlasmaBrightness": lambda: (K.RandomPlasmaBrightness(p=p, keepdim=keepdim), "plasma", 3),
        "RandomPlasmaContrast": lambda: (K.RandomPlasmaContrast(p=p, keepdim=keepdim), "plasma", 3),
        "RandomPlasmaShadow": lambda: (K.RandomPlasmaShadow(p=p, keepdim=keepdim), "plasma", 1),
    }
    return factories[name]()


def _statistics(form: str, device, dtype):
    """The six accepted ``mean`` / ``std`` forms."""
    if form == "float":
        return 0.5, 0.25
    if form == "list3":
        return [0.4, 0.5, 0.6], [0.2, 0.25, 0.3]
    if form == "tuple3":
        return (0.4, 0.5, 0.6), (0.2, 0.25, 0.3)
    if form == "tensor1":
        return (
            torch.tensor([0.5], device=device, dtype=dtype),
            torch.tensor([0.25], device=device, dtype=dtype),
        )
    if form == "tensor3":
        return (
            torch.tensor([0.4, 0.5, 0.6], device=device, dtype=dtype),
            torch.tensor([0.2, 0.25, 0.3], device=device, dtype=dtype),
        )
    assert form == "tensorB3"
    # Distinct rows, so a per-sample statistic is distinguishable from the scalar form.
    return (
        torch.tensor(
            [[0.40, 0.50, 0.60], [0.45, 0.55, 0.65], [0.35, 0.45, 0.55], [0.50, 0.60, 0.70]],
            device=device,
            dtype=dtype,
        ),
        torch.tensor(
            [[0.20, 0.25, 0.30], [0.22, 0.27, 0.32], [0.18, 0.23, 0.28], [0.25, 0.30, 0.35]],
            device=device,
            dtype=dtype,
        ),
    )
