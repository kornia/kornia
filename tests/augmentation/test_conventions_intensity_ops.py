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
import io
import itertools
import pickle

import pytest
import torch
from torch.distributions import Distribution

import kornia.augmentation as K
from kornia.constants import BorderType, Resample
from kornia.core.exceptions import BaseError, ImageError, ShapeError
from kornia.filters import box_blur, motion_blur

from testing.base import (
    DYNAMO_UNAVAILABLE_REASON,
    BaseTester,
    dynamo_is_available,
    supports_reflect_padding,
    supports_replicate_padding,
)


@pytest.fixture(autouse=True)
def _restore_global_rng(restore_torch_rng):
    # Every pin below seeds the global RNG so its draw is reproducible.  ``torch.manual_seed`` also
    # reseeds the CUDA and MPS generators, so the root fixture (#4446) snapshots and restores all of
    # them; ``fork_rng(devices=[])`` would restore the CPU generator only and shift the draw of an
    # unseeded later test on an accelerator leg.
    yield


# The seed drawn immediately before each construct-and-forward, as the audit probe did it.
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
    # Row 6c-19: ``kernel_size`` is ``(kH, kW)`` -- the first entry counts rows, the second counts
    # columns -- for RandomBoxBlur and RandomGaussianBlur alike, matching kornia.filters.box_blur
    # and gaussian_blur2d.  The claim is a labelling, so it is also checked under relabelling: the
    # transposed image blurred with the transposed kernel gives the transposed support.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 3, 4] = 1.0
    #   for ks in ((1, 5), (5, 1)):
    #       torch.manual_seed(0); y = K.RandomBoxBlur(ks, p=1.0)(x)
    #       print(ks, (y[0, 0] > 0).nonzero().T.tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(1, 5)` lights rows [3] cols [2, 3, 4, 5, 6] and
    # `(5, 1)` rows [1, 2, 3, 4, 5] col [4], identically for RandomGaussianBlur; with the impulse
    # moved off the centre lines to (2, 3) the same run gives rows [2] cols [1..5] and rows [0..4]
    # col [3], and on the transposed 9x7 image rows [1..5] col [2] and row [3] cols [0..4].
    @pytest.mark.parametrize("name", ["RandomBoxBlur", "RandomGaussianBlur"])
    def test_convention_blur_kernel_size_is_height_then_width(self, device, dtype, name):
        if not supports_reflect_padding(device, dtype):
            pytest.skip("reflection_pad2d is unavailable for this device/dtype")
        factories = {
            "RandomBoxBlur": lambda ks: K.RandomBoxBlur(ks, p=1.0),
            "RandomGaussianBlur": lambda ks: K.RandomGaussianBlur(ks, (1.0, 1.0), p=1.0),
        }
        make = factories[name]

        # The audit's own fixture, hot pixel on the 7x9 centre.
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

    # Row 6c-19, the median half: a rank filter cannot be read off an impulse, so the detector is a
    # one-row bar.  A (1, kW) window slides along that row and keeps it; a (kH, 1) window spans five
    # rows of which four are zero, so the median is zero and the bar is erased.  Checked under
    # relabelling on the transposed bar.
    # Snippet used to generate expected:
    #   bar = torch.zeros(1, 1, 7, 9); bar[0, 0, 3, :] = 1.0
    #   for ks in ((1, 5), (5, 1)):
    #       torch.manual_seed(0); print(ks, K.RandomMedianBlur(ks, p=1.0)(bar)[0, 0].sum(-1).tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(1, 5)` keeps row sums
    # [0, 0, 0, 9, 0, 0, 0] and `(5, 1)` gives [0, 0, 0, 0, 0, 0, 0]; with the bar on row 2 the
    # surviving sum moves to index 2, and on the transposed 9x7 bar the roles of the two kernels swap.
    def test_convention_median_blur_border_median_is_over_zero_padding(self, device, dtype):
        # A constant-ones image: the 3x3 window at a corner holds 5 zeros and 4 ones, so the corner comes
        # back 0 while an edge pixel (3 zeros, 6 ones) stays 1.  Replicate padding would return 1 at both.
        # Snippet used to generate expected:
        #   torch.manual_seed(0); y = K.RandomMedianBlur((3, 3), p=1.0)(torch.ones(1, 1, 4, 4)); print(y[0, 0, 0, :2])
        #   torch.manual_seed(0); print(K.RandomMedianBlur((5, 5), p=1.0)(torch.ones(1, 1, 6, 6))[0, 0, 0])
        # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> `[0., 1.]`; `[0., 0., 1., 1., 0., 0.]`.
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
        # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> `[2, 3] [3, 4]`, `[1, 2, 3, 4] [2, 3, 4, 5]`,
        # `[2, 3, 4] [2, 3, 4, 5]`; each output sums to 1.
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
        # An even entry is constructed and then fails on any image size, as a raw reshape error
        # (`shape '[2, 3, 16, 32, 32]' is invalid`, measured on cpu in all four dtypes and on mps float32).
        torch.manual_seed(_FORWARD_SEED)
        even = K.RandomMedianBlur((4, 4), p=1.0)
        large = torch.rand(2, 3, 32, 32).to(device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="is invalid for input of size"):
            _sync(even(large).device)
        # Below the kernel the zero-padded image is too small for F.conv2d, which raises before the view:
        # the claim is "raises for every image", not "raises the same error for every image".
        small = torch.rand(2, 3, 1, 1).to(device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match="Kernel size can't be greater than actual input size"):
            _sync(even(small).device)

    # Row 6c-20 (issue #4433, closed by #4486, which documents the mapping): RandomBoxBlur's
    # ``normalized`` flag is not a normalization switch -- it is forwarded as box_blur's
    # ``separable`` argument.  Both branches are means, so a constant image survives both and the
    # two outputs differ only by float rounding.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(2, 3, 7, 9)
    #   torch.manual_seed(0); yt = K.RandomBoxBlur((3, 3), normalized=True, p=1.0)(x)
    #   torch.manual_seed(0); yf = K.RandomBoxBlur((3, 3), normalized=False, p=1.0)(x)
    #   print(torch.equal(yt, box_blur(x, (3, 3), "reflect", separable=True)),
    #         torch.equal(yf, box_blur(x, (3, 3), "reflect", separable=False)), (yt - yf).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `True True 1.19209e-07`, and
    # `"normalized" in inspect.signature(box_blur).parameters` is False.
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

    # Row 6c-21: RandomGaussianBlur draws one sigma per sample -- ``_params["sigma"]`` is a (B,)
    # tensor, not one value for the batch and not a per-axis pair -- and the class defaults repeat
    # gaussian_blur2d's own (``separable=True``, ``border_type=BorderType.REFLECT``).
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(4, 3, 7, 9)
    #   aug = K.RandomGaussianBlur((3, 3), (0.5, 1.5), p=1.0)
    #   torch.manual_seed(0); aug(x); print(aug._params["sigma"].shape, aug._params["sigma"].tolist(), aug.flags)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `torch.Size([4])`, `[0.996257, 1.268222, 0.588477,
    # 0.632030]`, `{'kernel_size': (3, 3), 'separable': True, 'border_type': <BorderType.REFLECT: 1>}`.
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

    # Row 6c-21: an even kernel_size is rejected rather than silently rounded, with the message
    # gaussian_blur2d's own kernel check raises.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); K.RandomGaussianBlur((2, 2), (1.0, 1.0), p=1.0)(torch.rand(4, 3, 7, 9))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `kornia.core.exceptions.BaseError: Kernel size must
    # be an odd integer bigger than 0. Gotcha 2 on (2, 2)`.
    def test_convention_random_gaussian_blur_rejects_even_kernel_size(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 7, 9).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(Exception, match="odd integer"):
            _sync(K.RandomGaussianBlur((2, 2), (1.0, 1.0), p=1.0)(image).device)

    # Row 6c-22: RandomMotionBlur's ``angle`` is counter-clockwise as the image is displayed.  The
    # fixture is a 7x9 impulse at (row 2, col 3), off both centre lines, so no literal below is its
    # own transpose: +45 deg carries the blur from (row 3, col 2) up to (row 1, col 4), and -45 deg
    # down and to the right.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 2, 3] = 1.0
    #   for a in (0.0, 45.0, -45.0, 90.0):
    #       torch.manual_seed(0)
    #       print(a, (K.RandomMotionBlur(5, (a, a), (0.0, 0.0), p=1.0)(x)[0, 0].abs() > 1e-6).nonzero().tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0.0 [[2, 1], [2, 2], [2, 3], [2, 4], [2, 5]]`,
    # `45.0 [[1, 4], [2, 3], [3, 2]]`, `-45.0 [[1, 2], [2, 3], [3, 4]]`,
    # `90.0 [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3]]`.
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

    # Row 6c-23: ``direction`` re-weights the kernel along the kernel's own axis, whichever axis
    # ``angle`` selected -- -1 piles the weight at the start of the sweep, +1 at the end, 0 is
    # uniform.  Read on two orientations so the claim is not an artefact of one axis.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 7, 9); x[0, 0, 2, 3] = 1.0
    #   for d in (-1.0, 0.0, 1.0):
    #       torch.manual_seed(0); print(K.RandomMotionBlur(5, (0.0, 0.0), (d, d), p=1.0)(x)[0, 0, 2].tolist())
    #       torch.manual_seed(0); print(K.RandomMotionBlur(5, (90.0, 90.0), (d, d), p=1.0)(x)[0, 0, :, 3].tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> angle 0: `[0, .4, .3, .2, .1, 0, 0, 0, 0]`,
    # `[0, .2, .2, .2, .2, .2, 0, 0, 0]`, `[0, 0, .1, .2, .3, .4, 0, 0, 0]`; angle 90:
    # `[0, .1, .2, .3, .4, 0, 0]`, `[.2, .2, .2, .2, .2, 0, 0]`, `[.4, .3, .2, .1, 0, 0, 0]`.
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

    # Row 6c-24, the rotation half of `direction`: the weights are laid on a horizontal line and then
    # rotated with `resample`, and a "nearest" rotation off the axes drops taps, so the line's length and
    # its end weights change with the angle -- at 45 degrees and direction +1 the five taps become three.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 9, 9); x[0, 0, 4, 4] = 1.0
    #   for angle in (0.0, 45.0):
    #       torch.manual_seed(0); y = K.RandomMotionBlur(5, (angle, angle), (1.0, 1.0), p=1.0)(x)
    #       print(angle, sorted(y[y > 0].tolist()))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `0.0 [0.1, 0.2, 0.3, 0.4]` (the weight-0 end tap is empty)
    # and `45.0 [0.1667, 0.3333, 0.5]`; the impulse response is the kernel, so these are its taps.
    def test_convention_random_motion_blur_nearest_rotation_resamples_the_taps(self, device, dtype):
        image = _impulse(device, dtype, height=9, width=9, row=4, col=4)
        taps = {}
        for angle in (0.0, 45.0):
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomMotionBlur(5, (angle, angle), (1.0, 1.0), p=1.0)(image)
            taps[angle] = out[out.abs() > 1e-3].sort().values
        self.assert_close(taps[0.0], image.new_tensor([0.1, 0.2, 0.3, 0.4]))
        self.assert_close(taps[45.0], image.new_tensor([1.0, 2.0, 3.0]) / 6.0)

    # Row 6c-24: RandomMotionBlur keeps kornia.filters.motion_blur's defaults, so the class does not
    # quietly change the padding or the interpolation of the kernel rotation.
    # Snippet used to generate expected:
    #   aug = K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), p=1.0); print(aug.flags)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `{'border_type': <BorderType.CONSTANT: 0>,
    # 'resample': <Resample.NEAREST: 0>}`.
    @pytest.mark.device_agnostic
    def test_convention_random_motion_blur_defaults_are_constant_and_nearest(self):
        aug = K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), p=1.0)
        assert aug.flags["border_type"] == BorderType.CONSTANT
        assert aug.flags["resample"] == Resample.NEAREST

    # Issue #4599: `kernel_size` is drawn once per SAMPLE and then one entry -- the one at a uniformly
    # drawn `_params["idx"]`, not the first -- is applied to the whole batch; and because the draw is
    # `UniformDistribution(ks[0] // 2, ks[1] // 2)` truncated with `.int()`, the range's upper bound is
    # practically never reached.  `(3, 5)` is therefore a constant 3 -- "practically": the float32 draw
    # rounds onto the bound about once in 2**24 (seed 204 draws one 5 in 50000), so the pin is that the
    # bound is vanishingly rare, not impossible.  RandomRain had the same wart and no longer does
    # (#4567): it draws over `[lo, hi + 1)`, floors, and clamps that rounding escape back onto the closed
    # range -- the remedy this one still needs.
    # Snippet used to generate expected:
    #   import collections
    #   for ks in ((3, 5), (3, 7), (5, 11)):
    #       torch.manual_seed(0)
    #       v = K.RandomMotionBlur(ks, (0., 0.), (0., 0.), p=1.).forward_parameters((20000, 1, 8, 8))
    #       print(ks, sorted(collections.Counter(v["ksize_factor"].tolist())))
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `[3]`, `[3, 5]`, `[5, 7, 9]`; `idx` over 400 seeds at
    # B = 6 lands on every row (`{0: 67, 1: 71, 2: 66, 3: 57, 4: 83, 5: 56}`).
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize(("kernel_size", "drawn"), [((3, 5), [3]), ((3, 7), [3, 5]), ((5, 11), [5, 7, 9])])
    def test_wart_random_motion_blur_upper_bound_is_never_drawn_4599(self, kernel_size, drawn):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomMotionBlur(kernel_size, (0.0, 0.0), (0.0, 0.0), p=1.0)
        factors = aug.forward_parameters((20000, 1, 8, 8))["ksize_factor"]
        assert sorted(set(factors[factors != kernel_size[1]].tolist())) == drawn
        # The requested upper bound is odd and admissible, and it is practically absent: at most a rounding
        # handful of 20000 draws, where a uniform draw over the odd sizes would give thousands.
        assert kernel_size[1] % 2 == 1 and kernel_size[1] not in drawn
        assert int((factors == kernel_size[1]).sum()) < 5

    # Issue #4599, the other half: the per-sample draw is real, and a random index selects which one
    # the batch gets -- so `_params["ksize_factor"]` holds B different values while one kernel is used.
    @pytest.mark.device_agnostic
    def test_wart_random_motion_blur_applies_one_randomly_indexed_kernel_size_4599(self):
        picked = set()
        distinct = 0
        for seed in range(64):
            torch.manual_seed(seed)
            params = K.RandomMotionBlur((3, 21), (0.0, 0.0), (0.0, 0.0), p=1.0).forward_parameters((6, 3, 8, 8))
            assert params["ksize_factor"].shape == (6,)
            assert params["idx"].shape == (1,)
            picked.add(int(params["idx"][0]))
            distinct += len(set(params["ksize_factor"].tolist())) > 1
        # Every row of the batch can be the one applied, so it is not "the first sample's".
        assert picked == set(range(6))
        # And the draw really is per sample: a constant `ksize_factor` would make `idx` immaterial.
        assert distinct > 60
        # The kernel that is applied is the one at `idx`, not the first sample's: `_params` alone cannot
        # see that (`kernel_size_list[0]` in apply_transform left this pin green), so compare the output
        # with the functional at `ksize_factor[idx]` on seeds where that differs from `ksize_factor[0]`.
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(6, 3, 8, 8)
        checked = 0
        for seed in range(8):
            torch.manual_seed(seed)
            aug = K.RandomMotionBlur((3, 21), (-45.0, 45.0), (-1.0, 1.0), p=1.0)
            out = aug(image)
            params = aug._params
            idx = int(params["idx"][0])
            applied, first = int(params["ksize_factor"][idx]), int(params["ksize_factor"][0])
            reference = motion_blur(
                image,
                applied,
                params["angle_factor"],
                params["direction_factor"],
                border_type="constant",
                mode="nearest",
            )
            assert torch.equal(out, reference)
            if applied != first:
                assert not torch.equal(
                    out,
                    motion_blur(
                        image,
                        first,
                        params["angle_factor"],
                        params["direction_factor"],
                        border_type="constant",
                        mode="nearest",
                    ),
                )
                checked += 1
        assert checked >= 3

    # Row 6c-24, the resampling caveat: the kernel's weights are laid on a line and then rotated, so
    # with ``border_type="reflect"`` a "nearest" or "bilinear" rotation keeps them non-negative and the
    # output inside the input's extremes, while "bicubic" gives the kernel negative lobes that overshoot
    # both extremes at an edge.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 15, 15); x[..., 7:] = 1.0
    #   for rs in ("nearest", "bilinear", "bicubic"):
    #       torch.manual_seed(0)
    #       aug = K.RandomMotionBlur(5, (80., 80.), (1., 1.), border_type="reflect", resample=rs, p=1.)
    #       print(rs, aug(x).aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu and mps float32) -> `0 / 1`, `0 / 1`, `-0.0573 / 1.0285`.
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

    # The four filters do not clamp their result.  A 1.1 impulse is nevertheless attenuated below
    # one by each supported kernel, while median blur erases this isolated impulse altogether.  The
    # half-amplitude and negative legs make the range observation a property of the filters rather
    # than an implicit clamp in the augmentation wrapper.
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
            # The linearity legs below are unreachable for median blur, so this branch carried no
            # evidence against a clamp at all: appending `.clamp(0, 1)` to its apply_transform leaves it
            # green.  A negative plateau has a negative median and must come back negative.
            negative = torch.full((1, 1, 7, 7), -0.5, device=device, dtype=dtype)
            torch.manual_seed(_FORWARD_SEED)
            assert float(make()(negative).min()) < -0.4, "the wrapper clamped the median into [0, 1]"
            return

        half_impulse = impulse * 0.5
        self.assert_close(out, make()(half_impulse) * 2.0)
        self.assert_close(make()(-half_impulse), -make()(half_impulse))

    # Issue #4559, fixed: RandomBoxBlur, RandomGaussianBlur and RandomSharpness now raise a kornia
    # `ValueError` naming the class, the kernel and the input shape, where they used to let a raw torch
    # padding error out.  The thresholds are unchanged, and they are what the message states: the two
    # blurs reflect-pad, so an axis must be longer than the kernel radius (`k // 2`, one pixel for the
    # 3x3 default); RandomSharpness convolves without padding, so it needs the full 3x3.
    # RandomMedianBlur and RandomMotionBlur accept the same degenerate image, which is what made the
    # split a defect rather than a package-wide rule, and they are unchanged.
    # Snippet used to generate expected:
    #   for shape in ((2, 3, 1, 8), (2, 3, 2, 2), (2, 3, 3, 3)):
    #       x = torch.rand(*shape)
    #       torch.manual_seed(0); K.RandomBoxBlur(p=1.0)(x)  # and the four other classes
    # executed 2026-09-18 (torch 2.5.0, cpu) -> on (2, 3, 1, 8) the two blurs raise `ValueError:
    # RandomBoxBlur cannot filter an image this small: kernel_size=(3, 3) with border_type='reflect'
    # needs at least 2 pixel(s) along height, but the input is (2, 3, 1, 8)` and RandomSharpness the
    # same shape of message with `border_type='valid'` and 3 pixels; on (2, 3, 2, 2) only
    # RandomSharpness raises; on (2, 3, 3, 3) all five run; RandomMedianBlur and RandomMotionBlur run
    # on every shape.
    def test_convention_blur_and_sharpness_name_the_size_they_need_4559(self, device, dtype):
        # The skip is scoped to the legs that actually reflect-pad.  RandomSharpness fails through
        # F.conv2d, and RandomMedianBlur and RandomMotionBlur run on a 1x8 image with no reflect padding
        # at all, so skipping the whole test where reflection_pad2d is missing -- the torch 2.5.1 float16
        # leg -- would drop three legs of #4559 for a reason that does not apply to them.
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
        # 2-column image (2 > 3 // 2) and raises on a 3-row one (3 <= 7 // 2), in both classes (measured on
        # cpu in all four dtypes and on mps float32).
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

    # Issue #4559, the same family one border type over: ``border_type="circular"`` has a padding
    # failure of its own.  torch's circular pad refuses to wrap more than once, so it raises as soon
    # as the kernel radius (kernel_size // 2) exceeds the axis it pads -- which is a strictly smaller
    # image than the reflect guard needs, and is why the docstrings cannot say the non-reflect border
    # types run on every image.  ``"constant"`` and ``"replicate"`` do run there.  The message is
    # still torch's for RandomMotionBlur, which #4559 leaves alone, while the two padding blurs now
    # raise kornia's own error before reaching the pad -- at the same threshold, the radius.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); one = torch.rand(2, 3, 1, 1); thin = torch.rand(2, 3, 3, 12)
    #   for bt in ("constant", "reflect", "replicate", "circular"):
    #       torch.manual_seed(0); K.RandomMotionBlur(k, (45., 45.), (0., 0.), border_type=bt, p=1.)(one)
    #       torch.manual_seed(0); K.RandomBoxBlur((9, 9), border_type=bt, p=1.)(thin)
    # executed 2026-09-15 (torch 2.14.0, cpu float32/float64/float16/bfloat16 and mps float32, all
    # identical) -> motion `k=5` on 1x1 and both `(9, 9)` blurs on H=3 raise `RuntimeError: Padding
    # value causes wrapping around more than once.`, while motion `k=3` on 1x1 (radius 1, axis 1) and
    # the `"constant"`/`"replicate"` legs return the input shape.  The boundary is the radius, not the
    # kernel: `(9, 9)` circular runs at H=4 and raises at H=3; motion `k=5` runs at 2x2.
    def test_convention_circular_padding_rejects_images_smaller_than_kernel_radius_4559(
        self, device, dtype
    ):
        if not supports_replicate_padding(device, dtype):
            pytest.skip("replication_pad2d is unavailable for this device/dtype")
        torch.manual_seed(_FIXTURE_SEED)
        one = torch.rand(2, 3, 1, 1).to(device=device, dtype=dtype)
        thin = torch.rand(2, 3, 3, 12).to(device=device, dtype=dtype)

        # Motion blur: radius 2 exceeds a 1-pixel axis and raises; radius 1 does not and runs, so
        # the failure is the radius rather than "smaller than the kernel".
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(RuntimeError, match="wrapping around"):
            _sync(K.RandomMotionBlur(5, (45.0, 45.0), (0.0, 0.0), border_type="circular", p=1.0)(one).device)
        torch.manual_seed(_FORWARD_SEED)
        assert K.RandomMotionBlur(3, (45.0, 45.0), (0.0, 0.0), border_type="circular", p=1.0)(one).shape == one.shape

        # The two padding blurs, at a kernel whose radius exceeds the short axis: circular raises
        # where constant and replicate run, which is the pair of legs the #4559 warnings name.
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

    # The constant-border claim is directional, and a positive fixture cannot tell "pulled toward 0" from
    # "pulled below the minimum".  On an all-negative image zero padding pushes the border ABOVE the
    # maximum and nothing ever falls below the minimum, so both fixtures are checked.
    # Snippet used to generate expected:
    #   neg = torch.full((1, 1, 7, 7), -1.0)
    #   torch.manual_seed(0); print(K.RandomBoxBlur((3, 3), border_type="constant", p=1.0)(neg).aminmax())
    # executed 2026-09-16 (torch 2.14.0, cpu) -> box max -0.4444, gaussian max -0.5270, motion max -0.6667,
    # each with min -1.0.
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
        # Toward *zero*, and not toward some other constant: for a kernel whose weights sum to one a
        # border pixel is `s * x + (1 - s) * v`, so `out(+1) + out(-1)` is `2 * (1 - s) * v` and vanishes
        # only at `v = 0`.  Padding with 0.5 instead satisfies every bound above and is caught only here.
        self.assert_close(positive_out + negative_out, torch.zeros_like(positive_out), atol=1e-6, rtol=0)


# Twice float32(0.9): the snow coefficient is drawn in float32 whatever the image dtype, so this is the
# red channel whose HLS lightness equals the drawn coefficient exactly in float16, bfloat16, float32 and
# float64 alike.  It is what makes the coverage comparison's `<` distinguishable from `<=`.
_SNOW_BOUNDARY_RED = 1.7999999523162842


class TestNoiseAndWeatherConventions(BaseTester):
    # Row 6c-25: RandomErasing's ``scale`` is the target area fraction of the erased box -- met exactly
    # by these two fixtures, which need no rounding -- and ``ratio`` is height / width, so ratio 2.0
    # gives a tall box and 0.5 a wide one; the box is the half-open
    # rectangle [ys, ys + h) x [xs, xs + w) named by ``_params``, and ``value`` is the literal fill.
    # Snippet used to generate expected:
    #   x = torch.ones(1, 1, 10, 20)
    #   torch.manual_seed(0); aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(2.0, 2.0), p=1.0); y = aug(x)
    #   print((y != 1).sum(), {k: v.flatten().tolist() for k, v in aug._params.items()})
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `50` of 200 with
    # `{'widths': [5.0], 'heights': [10.0], 'xs': [1.0], 'ys': [0.0], 'values': [0.0]}`, and at
    # ratio 0.5 with value 0.3 `{'widths': [10.0], 'heights': [5.0], 'values': [0.3]}`.
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

    # Row 6c-25, the rounding caveat the class block states with these numbers: the target box for
    # scale 0.25 and ratio 3 on a 10x20 image is about 12.2 x 4.1, which is rounded to 12 x 4 and then
    # clipped to the 10-row image, so 40 of the 50 requested pixels are erased.
    # Snippet used to generate expected:
    #   for seed in range(200):
    #       torch.manual_seed(seed); aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(3.0, 3.0), p=1.0)
    #       y = aug(torch.ones(1, 1, 10, 20)); print((y != 1).sum(), aug._params["heights"], aug._params["widths"])
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `40`, `[10.]`, `[4.]` on every one of the 200 seeds.
    def test_convention_random_erasing_box_is_rounded_and_clipped(self, device, dtype):
        image = torch.ones(1, 1, 10, 20, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomErasing(scale=(0.25, 0.25), ratio=(3.0, 3.0), p=1.0)
        out = aug(image)
        assert (int(aug._params["heights"][0]), int(aug._params["widths"][0])) == (10, 4)
        assert int((out != 1.0).sum()) == 40

    # Row 6c-25 and the 6a anchor's claim that RandomErasing erases masks: routed through
    # AugmentationSequential with a ``mask`` data key the mask is erased in the same box, but it is
    # filled with 0 rather than with the image's ``value``.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   seq = K.AugmentationSequential(K.RandomErasing(scale=(0.25, 0.25), ratio=(2.0, 2.0), value=0.3, p=1.0),
    #                                  data_keys=["input", "mask"])
    #   oi, om = seq(torch.ones(1, 1, 10, 20), torch.ones(1, 1, 10, 20))
    #   print((oi != 1).sum(), (om != 1).sum(), sorted(set(om.flatten().tolist())))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `50 50 [0.0, 1.0]`, with the two erased index sets equal.
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

    # Row 6c-26: RandomGaussianNoise adds a draw of N(mean, std) to the image and never clamps, so a
    # constant 0.5 leaves [0, 1] in both directions; the draw is one value per element and is
    # recorded under ``_params["gaussian_noise"]`` with the input's shape.
    # Snippet used to generate expected:
    #   c = torch.full((2, 3, 7, 9), 0.5)
    #   torch.manual_seed(0); aug = K.RandomGaussianNoise(mean=0.0, std=1.0, p=1.0); y = aug(c)
    #   print(y.aminmax(), aug._params["gaussian_noise"].shape)
    #   torch.manual_seed(0); print(K.RandomGaussianNoise(mean=2.0, std=0.0, p=1.0)(c).unique())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `min=-2.11332 max=4.60149`, `torch.Size([2, 3, 7, 9])`,
    # `[2.5]`.
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

    # Parameter fields use the normalized BCHW shape even when their CHW input is restored by
    # ``keepdim``.  Gaussian noise samples just one such field for ``same_on_batch=True`` and lets
    # broadcasting share it with every selected sample.
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

    # At a partial probability the stored field keeps the original batch length, rather than being
    # compacted to the selected rows.  ``batch_prob`` identifies which rows were transformed; the
    # remaining rows pass through unchanged, and the full parameter state still replays exactly.
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
        # B is 16, not 4: at p=0.5 an all-or-nothing draw has probability 2 * 2 ** -B, which is 12.5% at
        # B=4 -- this node failed on 28 of seeds 0..199 and only _FORWARD_SEED kept it green.  At B=16 it is
        # 3e-5.
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

    # Row 6c-27: ``amount`` is the fraction of *pixels* touched, not of scalars -- a chosen pixel is
    # rewritten in every channel, so the changed mask is identical across channels -- and
    # ``salt_vs_pepper`` splits those pixels, 0.25 meaning a quarter of them become salt.  Both are
    # per-pixel Bernoulli draws, so each realised fraction is asserted within five binomial standard
    # deviations of the requested value: at a fixed band the pin held on seed 0 only by luck (the salt
    # ratio at amount 0.1 left a 0.02 band on 106 of seeds 0..199).  The image is large enough that five
    # deviations is still a tight band (0.0086 for the fraction at amount 0.1).
    # Snippet used to generate expected:
    #   x = torch.full((2, 3, 96, 160), 0.5)
    #   for amount in (0.1, 0.3, 0.6):
    #       torch.manual_seed(0)
    #       y = K.RandomSaltAndPepperNoise(amount=(amount, amount), salt_vs_pepper=(0.25, 0.25), p=1.0)(x)
    #       changed = (y != 0.5)[:, 0]; salt = (y[:, 0] == 1.0).sum(); pepper = (y[:, 0] == 0.0).sum()
    #       print(amount, changed.float().mean(), salt / (salt + pepper))
    # executed 2026-09-15 (torch 2.14.0, cpu float16/bfloat16/float32/float64 and mps float32, identical)
    # -> errors of -0.78 / -1.92 / -1.19 deviations for the fraction and -1.71 / -0.52 / -0.38 for the
    # salt ratio; over seeds 0..99 and the three amounts no draw leaves the five-deviation band.
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

    # Row 6c-27: the two poles are the literals 1.0 and 0.0, written regardless of the input's range,
    # so on a [0, 2] image the "salt" is darker than the untouched pixels.
    # Snippet used to generate expected:
    #   x = torch.full((2, 3, 12, 20), 2.0)
    #   for svp in (1.0, 0.0):
    #       torch.manual_seed(0)
    #       print(svp, K.RandomSaltAndPepperNoise(amount=(0.3, 0.3), salt_vs_pepper=(svp, svp), p=1.0)(x).unique())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `1.0 -> [1., 2.]` and `0.0 -> [0., 2.]`, each with
    # 0.322917 of the pixels changed.
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

    # Row 6c-27: only 1- and 3-channel images are accepted, so a 2-channel tensor is rejected rather
    # than silently treated as RGB.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); K.RandomSaltAndPepperNoise(p=1.0)(torch.rand(2, 2, 7, 9))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `kornia.core.exceptions.BaseError: Number of color
    # channels should be 1 or 3.`
    def test_convention_random_salt_and_pepper_requires_one_or_three_channels(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 2, 7, 9).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(Exception, match="1 or 3"):
            _sync(K.RandomSaltAndPepperNoise(p=1.0)(image).device)

    # Row 6c-28: RandomRain's ``drop_height`` runs down rows and ``drop_width`` along columns -- the
    # names are image axes, not drop-local ones.  The claim is a labelling, so it is checked under
    # relabelling on the transposed 30x12 image with the two arguments swapped.
    # Snippet used to generate expected:
    #   z = torch.zeros(1, 1, 12, 30)
    #   for dh, dw in ((6, 1), (1, 6)):
    #       torch.manual_seed(0)
    #       y = K.RandomRain(number_of_drops=(1, 1), drop_height=(dh, dh), drop_width=(dw, dw), p=1.0)(z)
    #       print(dh, dw, (y[0, 0] > 0).nonzero().tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `6 1 -> [[0, 8], [1, 8], [2, 8], [3, 8], [4, 8],
    # [6, 9]]` (six rows, two columns) and `1 6 -> [[1, 7], [1, 8], [1, 9], [1, 10], [1, 11],
    # [2, 13]]` (two rows, six columns); on the transposed 30x12 image the extents swap.
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
        # Relabelling: on a 30x12 image the same arguments still mean rows and columns, so the
        # extents do not follow the longer image axis.
        wide_rows, wide_cols = extents(torch.zeros(1, 1, 30, 12, device=device, dtype=dtype))
        if drop_height > drop_width:
            assert wide_rows >= 5 and wide_cols <= 2
        else:
            assert wide_cols >= 5 and wide_rows <= 2

    # Row 6c-31: a drop is written as the literal 200/255, not as a function of the image, so on an
    # image outside [0, 1] the "rain" is a dark smudge rather than a bright one.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   y = K.RandomRain(number_of_drops=(5, 5), drop_height=(3, 3), drop_width=(1, 1), p=1.0)(
    #       torch.full((1, 1, 12, 30), 2.0))
    #   print(y.aminmax())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `min=0.784314 max=2` (200/255 = 0.7843137...).
    def test_convention_random_rain_drop_value_is_two_hundred_over_255(self, device, dtype):
        image = torch.full((1, 1, 12, 30), 2.0, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomRain(number_of_drops=(5, 5), drop_height=(3, 3), drop_width=(1, 1), p=1.0)(image)
        self.assert_close(out.min(), out.new_tensor(200 / 255))
        self.assert_close(out.max(), out.new_tensor(2.0))

    # Rows 6c-28 / 6c-29 in the state #4451 left them (it closed #4434): a drop as tall as the image
    # or as wide as the image is rejected with a kornia error, and one pixel smaller runs.  The guard
    # is `<=` under a message that says "less than", so the rejected case is the equal one.  The
    # check lives in the forward, not the constructor -- the drop size is compared against the image
    # it is given -- so the construction happens outside the `raises` block, the converse of the
    # RandomSnow pin below, which pins a constructor-time check.
    # Snippet used to generate expected:
    #   z = torch.zeros(1, 1, 12, 30)
    #   for dh in (11, 12):
    #       aug = K.RandomRain(number_of_drops=(1, 1), drop_height=(dh, dh), drop_width=(1, 1), p=1.0)
    #       torch.manual_seed(0); aug(z)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> construction succeeds in both cases; the forward
    # gives `11 ok`, `12 -> BaseError: Height of drop should be greater than zero and less than image
    # height.`; `drop_width` 29 ok and 30 -> `BaseError: Width of drop should be less than image
    # width.`
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
        with pytest.raises(Exception, match=message):
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

    # Row 6c-29, the painted extent: the drop is sampled along `linspace(0, h, steps=max(h, |w|))`, end point
    # included, so a drop of size h spans h + 1 rows, and one row short of the image already runs from edge
    # to edge.  With both sizes at most 1 there is a single step, which is the start point alone, so the
    # drop is one pixel.
    # Snippet used to generate expected:
    #   x = torch.zeros(1, 1, 6, 10)
    #   for h, w in ((5, 0), (1, 1), (1, 0)):
    #       torch.manual_seed(0)
    #       y = K.RandomRain(number_of_drops=(1, 1), drop_height=(h, h), drop_width=(w, w), p=1.0)(x)
    #       print((h, w), sorted({r for r, _ in (y[0, 0] != 0).nonzero().tolist()}), int((y != 0).sum()))
    # executed 2026-09-15 (torch 2.14.0, cpu float16/bfloat16/float32/float64 and mps float32) ->
    # `[0, 1, 2, 3, 5]`, then a single painted pixel for both `(1, 1)` and `(1, 0)`.
    def test_convention_random_rain_drop_spans_one_row_more_than_its_height(self, device, dtype):
        image = torch.zeros(1, 1, 6, 10, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomRain(number_of_drops=(1, 1), drop_height=(5, 5), drop_width=(0, 0), p=1.0)(image)
        rows, _ = _lit_extent(out[0, 0])
        assert (rows[0], rows[-1]) == (0, 5)
        for width in (1, 0):
            torch.manual_seed(_FORWARD_SEED)
            aug = K.RandomRain(number_of_drops=(1, 1), drop_height=(1, 1), drop_width=(width, width), p=1.0)
            assert int((aug(image) != 0).sum()) == 1

    # The three integer ranges are closed and uniform (the fix for #4567: the sampler covers
    # ``[lo, hi + 1)``, the draw is floored and then clamped to ``hi``, where a truncating cast of a
    # ``[lo, hi)`` draw used to skip the upper bound and fold ``(-1, 0)`` and ``(0, 1)`` onto ``0``).
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   p = K.RandomRain(number_of_drops=(2, 4), drop_height=(5, 20), drop_width=(-5, 5), p=1.0)
    #   p = p.forward_parameters((22000, 1, 64, 64))
    #   for k in ("number_of_drops_factor", "drop_height_factor", "drop_width_factor"):
    #       print(k, sorted(collections.Counter(p[k].flatten().tolist()).items()))
    # executed 2026-09-16 (torch 2.14.0, cpu) -> drops 2..4 (7298, 7349, 7353), heights 5..20 (1301 to
    # 1455) and widths -5..5 (1906 to 2090), every count within 6% of the uniform expectation (7333,
    # 1375 and 2000).  The pin allows 15%, not 10%: the 16-bin height is the tight one, where 22000
    # draws put 10% at 3.8 sigma and 1 of seeds 0..99 exceeds it (max deviation 10.7%), against 5.7
    # sigma and 0 of 100 at 15%.  What the tolerance has to separate is a factor of two, not a few
    # percent -- truncation gave `0` twice its share -- so the wider band loses no mutation.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_integer_ranges_are_closed_and_uniform(self):
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomRain(number_of_drops=(2, 4), drop_height=(5, 20), drop_width=(-5, 5), p=1.0)
        params = aug.forward_parameters((22000, 1, 64, 64))
        for key, low, high in (
            ("number_of_drops_factor", 2, 4),
            ("drop_height_factor", 5, 20),
            ("drop_width_factor", -5, 5),
        ):
            drawn = params[key].flatten()
            assert drawn.dtype == torch.long
            assert (int(drawn.min()), int(drawn.max())) == (low, high), key
            counts = torch.bincount(drawn - low, minlength=high - low + 1)
            expected = drawn.numel() / (high - low + 1)
            assert bool(((counts > 0.85 * expected) & (counts < 1.15 * expected)).all()), (key, counts.tolist())

    # The excluded end point of the half-open sampler range is clamped back, not drawn: for the largest
    # value `torch.rand` can return, `lo + u * (hi + 1 - lo)` rounds up onto `hi + 1` in float32
    # (`5 + (1 - 2 ** -24) * 16` is exactly `21.0`), which `floor` would keep -- about one draw in 2**24,
    # and a `drop_height` of 21 makes the forward raise on a 21-row image where 20 is legal.
    # Snippet used to generate expected: the same three ranges, with `torch.rand` patched to return
    # `1 - 2 ** -24` everywhere.
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `[4, 20, 5]` with the clamp, `[5, 21, 5]` without it.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_largest_draw_stays_inside_the_closed_range(self, monkeypatch):
        largest = 1 - 2**-24
        real_rand = torch.rand
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.full_like(real_rand(*args, **kwargs), largest))
        aug = K.RandomRain(number_of_drops=(2, 4), drop_height=(5, 20), drop_width=(-5, 5), p=1.0)
        params = aug.forward_parameters((4, 1, 64, 64))
        # The patch has to reach the generator's own draws, or the pin below is vacuous: the coordinate
        # sampler runs over [0, 1], so under the patch every coordinate is exactly the patched value.
        assert float(params["coordinates_factor"].min()) == largest
        assert params["number_of_drops_factor"].tolist() == [4, 4, 4, 4]
        assert params["drop_height_factor"].tolist() == [20, 20, 20, 20]
        assert params["drop_width_factor"].tolist() == [5, 5, 5, 5]

    # A reversed range is rejected at construction, where the samplers are built -- on each of the three
    # parameters, not only the one a single example would name.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("name", ["number_of_drops", "drop_height", "drop_width"])
    def test_convention_random_rain_rejects_a_reversed_range(self, name):
        ranges = {"number_of_drops": (1, 1), "drop_height": (1, 1), "drop_width": (1, 1)}
        ranges[name] = (6, 5)
        with pytest.raises(ValueError, match=name):
            K.RandomRain(p=1.0, **ranges)

    # The ranges are integer ranges, so a bound that is not a whole number is rejected at construction
    # too -- including `nan` and `inf`, which an ordering test alone does not catch (they are neither
    # greater nor smaller than the other bound, so `drop_height=(nan, 5)` used to reach the sampler and
    # draw a silent `0`), and a fractional pair, which the clamp draws more unevenly than the truncation
    # it replaces: `(0.5, 2.5)` gave `{0: 25%, 1: 50%, 2: 25%}` and would give `{0: 17%, 1: 33%, 2: 50%}`.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("bounds", [(0.5, 2.5), (1, 2.5), (float("nan"), 5), (5, float("nan")), (1, float("inf"))])
    @pytest.mark.parametrize("name", ["number_of_drops", "drop_height", "drop_width"])
    def test_convention_random_rain_rejects_a_fractional_or_non_finite_range(self, name, bounds):
        ranges = {"number_of_drops": (1, 1), "drop_height": (1, 1), "drop_width": (1, 1)}
        ranges[name] = bounds
        with pytest.raises(ValueError, match=name):
            K.RandomRain(p=1.0, **ranges)
        # Integral floats are the same range written another way, and stay accepted.
        ranges[name] = (1.0, 2.0)
        K.RandomRain(p=1.0, **ranges)

    # The pair itself is validated: a range that is not two values, whatever its length, is rejected
    # rather than silently indexed, and a tensor range is accepted as the old `_range_bound` accepted it.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_validates_the_range_pair(self):
        for bad in ((1, 2, 3), (1,), 5):
            with pytest.raises(ValueError, match="drop_height"):
                K.RandomRain(p=1.0, drop_height=bad)
        torch.manual_seed(_FORWARD_SEED)
        aug = K.RandomRain(p=1.0, number_of_drops=(1, 1), drop_height=torch.tensor([2, 4]), drop_width=(1, 1))
        drawn = aug.forward_parameters((2000, 1, 64, 64))["drop_height_factor"]
        assert sorted(set(drawn.flatten().tolist())) == [2, 3, 4]

    # Issue #4604: a drop starts anywhere that keeps it inside the image, so every row and column is painted,
    # for a single pixel and for drops slanting either way.  Literal seeds, as for the other census pins: the
    # claim is over the union of 300 draws, not one.  Before the fix the single-pixel case painted rows
    # `[0, 1, 2, 3]` and cols `[0, ..., 8]` on `6 x 10`.
    @pytest.mark.parametrize("drop_height, drop_width", [(1, 0), (2, 2), (2, -2)])
    def test_convention_random_rain_paints_every_row_and_column_4604(self, device, dtype, drop_height, drop_width):
        image = torch.zeros(1, 3, 6, 10, device=device, dtype=dtype)
        # `.to` so the coordinate sampler runs in the test's dtype: left at construction defaults it stays
        # float32 even for a float16 image, and the MPS half `rand` that returns exactly 1.0 (#4553) -- the
        # draw the clamp exists for -- is never reached.
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

    # Issue #4604, the painted extent: a single drop paints a bounding box of exactly ``(h, |w|)`` -- the
    # linspace includes both end points once it has two steps, and is the start alone when it has one -- and
    # that box lands inside the image for every legal start.  The union pin above cannot see this: it only
    # records which rows and columns are hit somewhere across 300 draws, so a drop split across opposite
    # edges by a negative-index wrap still covers ``range(10)`` and passes.  Swept rather than pinned to
    # literals, because the claim is a bound over the whole legal domain, not a census of one shape.
    # Snippet used to generate expected:
    #   for H, W in itertools.product(range(2, 8), repeat=2):
    #       for h in range(1, H):
    #           for w in range(-(W - 1), W):
    #               for seed in range(2):
    #                   ... print(box, expected, in_bounds)
    # executed 2026-09-18 (torch 2.14.0, cpu) -> 0 violations.  With ``+ max(-last_dx, 0)`` deleted, seed 1
    # of ``6 x 10, h=2, w=-2`` starts a drop at ``(2, 0)`` and paints ``(2, 0)`` and ``(4, -2)``, the latter
    # wrapping round to ``(4, 8)``: a box of ``(2, 8)`` against the expected ``(2, 2)``.
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

    # Issue #4604, the uniformity half of the bullet: every legal start is equally likely, not merely
    # reachable.  The union pin is blind to the distribution -- folding the row overflow onto the last row
    # keeps the union at ``range(6)`` while badly skewing where drops land -- so this counts start rows the
    # way the #4567 pin above counts drawn sizes, with the same 15% band and for the same reason.  Single
    # pixel drops on a wide image, so each painted cell is one start and collisions are rare (100 drops over
    # 12000 cells, about 0.4 an image), read off the output rather than recomputed from the parameters.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   aug = K.RandomRain(number_of_drops=(100, 100), drop_height=(1, 1), drop_width=(0, 0), p=1.0)
    #   out = aug(torch.zeros(200, 1, 6, 2000))
    #   print(torch.bincount((out[:, 0] != 0).nonzero()[:, 1], minlength=6).tolist())
    # executed 2026-09-18 (torch 2.14.0, cpu) -> every bin within 1% of the uniform expectation.  Scaling
    # the row draw by ``cols`` instead of ``rows`` puts every drop on the last row.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_start_row_is_uniform_4604(self):
        aug = K.RandomRain(number_of_drops=(100, 100), drop_height=(1, 1), drop_width=(0, 0), p=1.0)
        torch.manual_seed(_FORWARD_SEED)
        lit = (aug(torch.zeros(200, 1, 6, 2000))[:, 0] != 0).nonzero()
        counts = torch.bincount(lit[:, 1], minlength=6)
        expected = int(counts.sum()) / 6
        assert bool(((counts > 0.85 * expected) & (counts < 1.15 * expected)).all()), counts.tolist()

    # Issue #4604, the clamp on the start draw: the coordinate sampler runs over ``[0, 1]`` and the MPS half
    # ``rand`` really does return an exact ``1.0`` (#4553), which would put the start one past the last
    # admissible one and index out of the image.  Reproduced here by patching the draw rather than left to
    # the MPS leg, so the pin fires on every device; same idiom as the closed-range pin above.  Main was
    # safe against this only by accident, its multiplier being one smaller than the true count of starts.
    # Snippet used to generate expected: the config below with `torch.rand` patched to return `1.0`.
    # executed 2026-09-18 (torch 2.14.0, cpu) -> the drop paints `(3, 9)` and `(5, 7)`, the last row and the
    # last column.  Clamping to `rows` instead of `rows - 1` raises `index 6 is out of bounds`, and `cols`
    # instead of `cols - 1` raises `index 10 is out of bounds`.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_start_draw_of_one_stays_inside_4604(self, monkeypatch):
        real_rand = torch.rand
        monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.ones_like(real_rand(*a, **kw)))
        aug = K.RandomRain(number_of_drops=(1, 1), drop_height=(2, 2), drop_width=(-2, -2), p=1.0)
        # The patch has to reach the generator's own draws, or the pin is vacuous.
        assert float(aug.forward_parameters((1, 1, 6, 10))["coordinates_factor"].min()) == 1.0
        lit = (aug(torch.zeros(1, 1, 6, 10))[0, 0] != 0).nonzero().tolist()
        assert sorted(lit) == [[3, 9], [5, 7]]

    # Row 6c-28 in the state #4453 left it (it closed #4448): with ``same_on_batch=True`` every
    # sample of the batch gets the same number of drops, the same drop size and the same coordinates;
    # with the default False each sample draws its own count.
    # Snippet used to generate expected:
    #   for sob in (True, False):
    #       torch.manual_seed(0)
    #       aug = K.RandomRain(number_of_drops=(1, 50), drop_height=(1, 3), drop_width=(1, 3), p=1.0,
    #                          same_on_batch=sob)
    #       aug(torch.zeros(4, 1, 12, 30)); print(sob, aug._params["number_of_drops_factor"].tolist())
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `True [25, 25, 25, 25]` and `False [25, 39, 5, 7]`.
    def test_convention_random_rain_same_on_batch_draws_one_drop_count(self, device, dtype):
        # B is 16, not 4: `drop_height=(1, 3)` and `drop_width=(1, 3)` each draw from {1, 2, 3}, so
        # "the unshared draw differs across the batch" fails when one of the two factors comes out
        # constant, a `2 * 3 ** -(B - 1)` event -- 16 of seeds 0..199 coincide at B=4, 1 at B=8 and none
        # at B=16.
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
        # The sentence claims three things, not one: the drop SIZES and the COORDINATES are shared too.
        # A generator that passed `same_on_batch=False` to the size or coordinate sampler would leave the
        # drop-count assertions above untouched.
        for key in ("drop_height_factor", "drop_width_factor"):
            assert len(set(shared[True][key].flatten().tolist())) == 1, key
            assert len(set(shared[False][key].flatten().tolist())) > 1, key
        coordinates = shared[True]["coordinates_factor"]
        assert all(torch.equal(coordinates[0], coordinates[row]) for row in range(1, batch_size))
        assert not torch.equal(shared[False]["coordinates_factor"][0], shared[False]["coordinates_factor"][1])

    # Row 6c-32: RandomSnow is an RGB-only operator -- it rejects any other channel count -- and its
    # ``snow_coefficient`` is validated against [0, 1] at construction rather than at the forward.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); K.RandomSnow(p=1.0)(torch.rand(2, 1, 7, 9))
    #   K.RandomSnow(snow_coefficient=(1.5, 1.5), p=1.0)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `BaseError: Number of color channels should be 3.`
    # and `BaseError: Snow coefficient values must be between 0 and 1.` (the same for -0.5).
    def test_convention_random_snow_requires_rgb_and_a_unit_coefficient(self, device, dtype):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 1, 7, 9).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(Exception, match="should be 3"):
            _sync(K.RandomSnow(p=1.0)(image).device)
        for coefficient in (1.5, -0.5):
            with pytest.raises(Exception, match="between 0 and 1"):
                K.RandomSnow(snow_coefficient=(coefficient, coefficient), p=1.0)

    # Row 6c-32: ``snow_coefficient`` and ``brightness`` are drawn once per sample, and
    # ``same_on_batch=True`` collapses both to a single value for the batch.
    # Snippet used to generate expected:
    #   torch.manual_seed(0)
    #   aug = K.RandomSnow(snow_coefficient=(0.2, 0.8), brightness=(1.5, 3.0), p=1.0)
    #   aug(torch.rand(4, 3, 7, 9)); print({k: v.flatten().tolist() for k, v in aug._params.items()})
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `snow_coefficient [0.370124, 0.726453, 0.375167,
    # 0.291604]`, `brightness [2.365495, 2.699539, 1.573806, 2.927976]`; with same_on_batch=True both
    # lists are four copies of one value.
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
            # Reach, not only containment: a draw shrunk to the lower half of the range satisfies the above.
            width = high - low
            assert float(many[key].min()) < low + 0.1 * width and float(many[key].max()) > high - 0.1 * width
        torch.manual_seed(_FORWARD_SEED)
        shared = K.RandomSnow(snow_coefficient=(0.2, 0.8), brightness=(1.5, 3.0), p=1.0, same_on_batch=True)
        shared(image)
        for key in ("snow_coefficient", "brightness"):
            assert len(set(shared._params[key].flatten().tolist())) == 1

    # Row 6c-32, the clamp: only a covered pixel -- lightness below the drawn coefficient -- has its
    # lightness scaled and clamped.  At coefficient 0.9 and brightness 2, (1.5, 0, 0) has lightness 0.75, is
    # covered and turns white; (2, 1.5, 1.5) has lightness 1.75, is missed and keeps its values; and
    # (1.2, -2, -2) has lightness -0.4 and turns black.  A missed pixel still makes the unclamped HLS round
    # trip, which is not the identity at either singular lightness: (1.5, 0.5, 0.5) has lightness exactly 1,
    # where `2 - max - min` vanishes, and comes back white; (2, -2, -2) has lightness exactly 0, where
    # `max + min` vanishes, and comes back black although it is neither all-negative nor negative in
    # lightness.  The guarded denominators make both endpoints finite in every supported dtype.
    # The coefficient is 0.9, not 1.0, and (1.8, 0, 0) is the last row: coverage is `lightness < coefficient`
    # at random_snow.py:118, so the only pixel that can tell `<` from `<=` is one whose lightness equals the
    # coefficient exactly.  At coefficient 1.0 that pixel was (1.5, 0.5, 0.5) -- but it is also the lightness-1
    # singularity, which comes back white either way, so flipping the comparison left every pin in this file
    # green.  The last row is not singular and sits exactly on the boundary: its red channel is
    # `2 * float32(0.9)`, not 1.8, because the coefficient is drawn in float32 whatever the image dtype --
    # with a plain 1.8 a float64 image computes a lightness a hair above the float32 coefficient and the
    # mutant survives on that leg.  Missed it keeps its values; covered it would be scaled by 2 and clamped
    # to white.
    # Snippet used to generate expected:
    #   x = torch.tensor([[1.5, 0.0, 0.0], [2.0, 1.5, 1.5], [1.2, -2.0, -2.0], [1.5, 0.5, 0.5],
    #                     [2.0, -2.0, -2.0], [1.7999999523162842, 0.0, 0.0]]).T.reshape(1, 3, 1, 6)
    #   torch.manual_seed(0); print(K.RandomSnow(snow_coefficient=(0.9, 0.9), brightness=(2.0, 2.0), p=1.0)(x))
    # executed 2026-09-16 (torch 2.9.1, cpu float32/float64/float16/bfloat16) -> the pixels
    # `(1, 1, 1)`, `(2, 1.5, 1.5)`, `(0, 0, 0)`, `(1, 1, 1)`, `(0, 0, 0)` and the boundary pixel unchanged;
    # rows 3 and 4 are finite white and black in every supported dtype.
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

    # The singular region at lightness 1 has the width of rgb_to_hls's eps = 1e-8: a lightness 2e-7 off
    # comes back close to its input in float32 and float64 (half precision rounds it onto the point), while
    # 1 + 1e-8 and 1 + 5e-9, which only float64 resolves, come back as (2, 0, 0) and as values of order 1e8.
    # Snippet used to generate expected:
    #   for L in (1 + 2e-7, 1 + 1e-8, 1 + 5e-9):
    #       x = torch.tensor([2 * L - 0.5, 0.5, 0.5], dtype=torch.float64).reshape(1, 3, 1, 1)
    #       torch.manual_seed(0); print(L, K.RandomSnow(p=1.0)(x).flatten().tolist())
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `[1.5263, 0.4737, 0.4737]` (same at 1e-7), `[2.0, -0.0, -0.0]`,
    # `[-82271062.5, 82271064.5, 82271064.5]`.
    def test_convention_random_snow_singularity_has_eps_width(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half precision: 1.5 + 4e-7 rounds onto the singular point")

        def run(lightness: float) -> torch.Tensor:
            pixel = torch.tensor([2 * lightness - 0.5, 0.5, 0.5], dtype=torch.float64).reshape(1, 3, 1, 1)
            pixel = pixel.to(device=device, dtype=dtype)
            torch.manual_seed(_FORWARD_SEED)
            return K.RandomSnow(snow_coefficient=(0.9, 0.9), brightness=(2.0, 2.0), p=1.0)(pixel).flatten()

        near = run(1 + 2e-7)
        assert float(near[0]) > 1.4 and float((near - near.new_tensor([1.5, 0.5, 0.5])).abs().max()) < 0.1
        if dtype == torch.float64:
            self.assert_close(run(1 + 1e-8), near.new_tensor([2.0, 0.0, 0.0]))
            assert float(run(1 + 5e-9).abs().min()) > 1e6

    # Every achromatic pixel -- black, gray or white -- stays finite through the HLS round trip.
    # Snippet used to generate expected:
    #   for dt in (torch.float16, torch.bfloat16, torch.float32):
    #       torch.manual_seed(0); y = K.RandomSnow(p=1.0)(torch.full((1, 3, 2, 2), 0.5, dtype=dt))
    #       print(dt, torch.isfinite(y).all())
    # current CPU run -> True for float16, bfloat16 and float32, and the same for constants 0.0 and 1.0.
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_convention_random_snow_achromatic_pixel_is_finite_4571(self, device, dtype, value):
        gray = torch.full((1, 3, 2, 2), value, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.RandomSnow(p=1.0)(gray)
        assert bool(torch.isfinite(out).all())
        expected = gray.new_full(gray.shape, value)
        self.assert_close(out, expected)

    # The channel-count precondition is a class-by-class contract, so it is pinned as one table rather
    # than per class: five of these were documented before this batch and five were not, and the split
    # between "1 or 3" and "3 only" is the part a reader gets wrong.
    # Snippet used to generate expected:
    #   for c in (1, 2, 4): torch.manual_seed(0); factory()(torch.rand(2, c, 32, 32))
    # executed 2026-09-16 (torch 2.14.0, cpu) -> RandomRain and RandomSaltAndPepperNoise accept C=1 and
    # reject C=2/C=4 with `Number of color channels should be 1 or 3.`; ColorJiggle, RandomHue,
    # RandomSaturation, RandomJPEG, RandomPlanckianJitter, RandomRGBShift and RandomSnow reject all three.
    # The rejection TYPE and message are part of the contract: with a bare ``pytest.raises(Exception)``
    # a class that started failing for an unrelated reason would still satisfy this table.  The five
    # message families below were executed on 2026-09-16 (torch 2.14.0, cpu) at C in (1, 2, 4).
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
            # The 3-channel baseline call below needs the codec's replication padding, which torch 2.5.1
            # has no `Half` kernel for ("replication_pad2d" not implemented for 'Half').  The rejection
            # this test pins happens before that, but the baseline would fail for an unrelated reason.
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

    # The degenerate signed range `(-1, 1)` is where the old truncation toward zero was total: it mapped
    # the whole interval onto `0`, except an exact `-1.0` draw (#4567).  A closed uniform draw gives the
    # three values equal weight, and `0` stops carrying its neighbours' weight one step wider too.
    # Snippet used to generate expected:
    #   for bounds in ((-1, 1), (-2, 2)):
    #       g = RainGenerator(number_of_drops=(5, 6), drop_height=(2, 3), drop_width=bounds)
    #       torch.manual_seed(0)
    #       print(bounds, sorted(Counter(g((20000, 3, 64, 64))["drop_width_factor"].flatten().tolist()).items()))
    # executed 2026-09-16 (torch 2.14.0, cpu) -> `{-1: 6688, 0: 6776, 1: 6536}` and, for `(-2, 2)`,
    # `{-2: 3941, -1: 4083, 0: 4075, 1: 3976, 2: 3925}`.
    @pytest.mark.device_agnostic
    def test_convention_random_rain_symmetric_signed_width_range_is_uniform(self):
        # `(-5, -1)` and `(-3, 0)` come from the wart pin this replaces: on a range that never crosses
        # zero, truncation toward zero drops the LOWER bound instead of the upper, so they are the cases
        # where `floor` and the old cast differ and a half-fixed draw would still show.
        for low, high in ((-1, 1), (-2, 2), (-5, -1), (-3, 0)):
            torch.manual_seed(_FORWARD_SEED)
            aug = K.RandomRain(number_of_drops=(5, 6), drop_height=(2, 3), drop_width=(low, high), p=1.0)
            drawn = aug.forward_parameters((20000, 1, 64, 64))["drop_width_factor"].flatten()
            assert sorted(set(drawn.tolist())) == list(range(low, high + 1))
            counts = torch.bincount(drawn - low, minlength=high - low + 1)
            expected = drawn.numel() / (high - low + 1)
            assert bool(((counts > 0.9 * expected) & (counts < 1.1 * expected)).all()), ((low, high), counts.tolist())


class TestIlluminationAndNormalizeConventions(BaseTester):
    # Row 6c-42: all three *Illumination classes add a gradient whose direction is the drawn ``sign``
    # -- one draw per sample, from (-1.0, 1.0) by default -- and clamp the sum into [0, 1]
    # (`_apply_gaussian_illumination`, and `RandomLinearIllumination.apply_transform` /
    # `RandomLinearCornerIllumination.apply_transform`).  The gradient is recorded under
    # ``_params["gradient"]`` with the input's shape.  The clamp is load-bearing on an *in-range*
    # image as soon as ``gain`` exceeds the headroom: at gain 0.8 on a constant 0.5 the unclamped sum
    # reaches 1.3 or -0.3, so deleting ``.clamp_(0, 1)`` fails this pin.
    # Snippet used to generate expected:
    #   c = torch.full((2, 3, 7, 9), 0.5)
    #   torch.manual_seed(0); aug = K.RandomGaussianIllumination(gain=(0.5, 0.5), p=1.0); y = aug(c)
    #   print(y.aminmax(), aug._params["gradient"].aminmax())
    #   for sign in ((1.0, 1.0), (-1.0, -1.0)):
    #       torch.manual_seed(0); a = K.RandomGaussianIllumination(gain=(0.8, 0.8), sign=sign, p=1.0)
    #       y = a(c); raw = c + a._params["gradient"]
    #       print(sign, y.aminmax(), raw.aminmax(), torch.equal(y, raw.clamp(0, 1)), torch.equal(y, raw))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> default gain 0.5: `min=1.93119e-05 max=0.5`
    # (Gaussian), `0/0.5` (Linear), `2.38419e-07/0.5` (LinearCorner), gradient in [-0.5, -0] and
    # `gradient` of shape (2, 3, 7, 9); at gain 0.8, `sign=(1, 1)` -> out `0.5/1` against an unclamped
    # `0.5/1.3`, `sign=(-1, -1)` -> out `0/0.5` against an unclamped `-0.3/0.5`, with
    # `out == (c + gradient).clamp(0, 1)` bitwise and `out != c + gradient` in both directions.
    # NOTE: the negative direction of the default-``sign`` leg is the draw at this seed and batch
    # size, not a contract -- at B=1 and B=5 the same seed draws sign +1 and the image brightens --
    # so the darkening claim is asserted only where ``sign`` is a point range.
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

        # `sign` fixes the direction: -1 can only darken, +1 can only brighten.  At gain 0.8 the sum
        # leaves [0, 1] on this in-range image, so the clamp is what brings it back -- and the output
        # is the clamped sum bitwise, not the raw one.
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

        # One parameter away (B=2 -> B=4, and same_on_batch): with the default range the sign is drawn
        # per sample, so one batch can hold both directions; same_on_batch=True collapses it to one.
        # Snippet: torch.manual_seed(0); aug(torch.full((4, 3, 7, 9), 0.5));
        #   [float(aug._params["gradient"][b].max()) for b in range(4)]
        # executed 2026-09-15 (torch 2.14.0, cpu) -> `[-0.0, 0.5, -0.0, 0.5]` for RandomLinearIllumination
        # and `[0.5, 0.5, 0.5, 0.5]` with same_on_batch=True.
        # B is 16, not 4: "one batch holds both directions" fails when every sample draws the same sign,
        # which is 2 * 2 ** -B -- 12.5% at B=4, and this leg failed on 29 of seeds 0..199.  At B=16 it is 3e-5.
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
        # The blocks also say the EDGE (or corner) the gradient is strongest at is drawn per sample.
        # Nothing above reads it: an implementation that collapsed all four directions to one would
        # still pass every assertion so far, because the sign and the magnitude would be unchanged.
        # The brightest cell of each sample's gradient names its direction, so more than one distinct
        # location across a batch of 16 is the claim.  `RandomGaussianIllumination` draws a centre
        # rather than an edge, and it is covered by the same reading.
        # Snippet used to generate expected:
        #   torch.manual_seed(0); a = _illumination(name); a(torch.full((16, 3, 7, 9), 0.5))
        #   g = a._params["gradient"].flatten(1).abs(); print(len(set(g.argmax(1).tolist())))
        # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> at least 2 for each of the three.
        flat = per_sample._params["gradient"].flatten(1).float().abs()
        assert len(set(flat.argmax(1).tolist())) > 1, "the strongest-gradient location is not drawn per sample"

    # The all-negative collapse documented for these augmentations is conditional on the sampled
    # gradient.  With a fixed positive sign and enough gain, a near-zero negative image retains the
    # positive portion of that gradient after the unit-range clamp.
    @pytest.mark.parametrize(
        "name", ["RandomGaussianIllumination", "RandomLinearIllumination", "RandomLinearCornerIllumination"]
    )
    def test_convention_positive_illumination_can_lift_an_all_negative_image(self, device, dtype, name):
        image = torch.full((2, 3, 7, 9), -0.01, device=device, dtype=dtype)
        aug = _illumination(name, gain=(0.1, 0.1), sign=(1.0, 1.0))
        torch.manual_seed(_FORWARD_SEED)
        out = aug(image)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        assert float(out.max()) > 0.0

    # Row 6c-41 in the state #4457 left it (it closed #4435): RandomGaussianIllumination used to
    # define its transform as a closure in __init__, which neither pickle nor torch.save could
    # serialize.  All three classes now round-trip through pickle and deepcopy, and the copy
    # reproduces the original's output under the same seed.
    # Snippet used to generate expected:
    #   c = torch.full((2, 3, 7, 9), 0.5)
    #   torch.manual_seed(0); a = K.RandomGaussianIllumination(gain=(0.5, 0.5), p=1.0); y = a(c)
    #   torch.manual_seed(0); b = pickle.loads(pickle.dumps(K.RandomGaussianIllumination(gain=(0.5, 0.5), p=1.0)))
    #   print(torch.equal(y, b(c)))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `True` for pickle, deepcopy and torch.save on all three.
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
        # torch.save is what #4435 reported: it pickles the module, so it failed for the same reason.
        # `nbytes > 0` would hold for any object, so the round trip is reloaded and run instead.
        buffer = io.BytesIO()
        torch.save(_illumination(name), buffer)
        buffer.seek(0)
        torch.manual_seed(_FORWARD_SEED)
        reloaded = torch.load(buffer, weights_only=False)
        assert torch.equal(reloaded(constant), expected)

    # Issue #4589: the constructor's own check admits `0 <= sigma <= 1`, but filters.kernels.gaussian
    # normalizes by `gauss.sum()`, which underflows to zero at sigma 0, so the kernel is 0 / 0 and the
    # whole output is NaN.  An even-length axis carries a half-pixel offset, so no sample sits at the mean
    # and a small non-zero sigma does it too -- with center 0.5, 4x4 is NaN at sigma 0.005 where 3x3 is not.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); print(K.RandomGaussianIllumination(p=1.0, sigma=0.0)(torch.rand(1, 3, 4, 4)))
    # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> every element NaN; the default
    # sigma=(0.2, 1.0) is finite over 200 seeds at sizes 1..4.  The even/odd leg needs `center` pinned:
    # with the drawn centre the odd size is NaN on some draws too, and with centre 0.5 a 4x4 image is NaN
    # at sigma 0.005 on all 30 seeds tried while 3x3 and 5x5 are finite on all of them.
    @pytest.mark.parametrize("size", [3, 4])
    def test_wart_random_gaussian_illumination_zero_sigma_is_nan_4589(self, device, dtype, size):
        image = torch.full((1, 3, size, size), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert bool(K.RandomGaussianIllumination(p=1.0, sigma=0.0)(image).isnan().all())
        # The even axis loses the on-grid sample to the half-pixel offset, so it is NaN at a small
        # non-zero sigma that the odd one survives.
        for sigma in (0.005, 0.01):
            torch.manual_seed(_FORWARD_SEED)
            small = K.RandomGaussianIllumination(p=1.0, sigma=(sigma, sigma), center=(0.5, 0.5))(image)
            if sigma == 0.005 and size % 2 == 0:
                assert bool(small.isnan().all())
            else:
                assert bool(small.isfinite().all())
        # Evenness is not the variable: `sigma` is relative, so it is `sigma * axis` -- the kernel's
        # absolute width -- that underflows, at about 0.034 in the float32 the parameters are drawn in.
        # An `8 x 8` image is finite at the sigma that kills `4 x 4`, and a `64 x 64` one dies at a
        # proportionally smaller sigma.  The even/odd leg above runs only at sizes 3 and 4 and cannot
        # see either half.
        # Snippet used to generate expected:
        #   for n, sg in ((4, 0.005), (8, 0.005), (8, 0.002), (64, 0.0005), (64, 0.00055)):
        #       torch.manual_seed(0)
        #       K.RandomGaussianIllumination(p=1.0, sigma=(sg, sg), center=(0.5, 0.5))(
        #           torch.full((1, 3, n, n), 0.5))
        # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes -- the parameters are float32 in
        # every one of them) -> NaN, finite, NaN, NaN, finite.  `n * sigma` is 0.020, 0.040, 0.016,
        # 0.032 and 0.035 respectively, so the cut is on the product and not on `n`.
        for axis, small_sigma, nan_expected in ((4, 0.005, True), (8, 0.005, False), (8, 0.002, True)):
            square = torch.full((1, 3, axis, axis), 0.5, device=device, dtype=dtype)
            torch.manual_seed(_FORWARD_SEED)
            out = K.RandomGaussianIllumination(p=1.0, sigma=(small_sigma, small_sigma), center=(0.5, 0.5))(square)
            assert bool(out.isnan().all()) is nan_expected, (axis, small_sigma)
        # The parameter dtype, not the image dtype, sets the threshold: float64 parameters move it.
        # MPS has no float64, so this leg is CPU/CUDA only.
        if device.type != "mps":
            wide = torch.full((1, 3, 4, 4), 0.5, device=device, dtype=dtype)
            for param_sigma, nan_expected in ((0.005, False), (0.003, True)):
                aug = K.RandomGaussianIllumination(p=1.0, sigma=(param_sigma, param_sigma), center=(0.5, 0.5))
                aug.set_rng_device_and_dtype(device, torch.float64)
                torch.manual_seed(_FORWARD_SEED)
                assert bool(aug(wide).isnan().all()) is nan_expected, param_sigma
        # And with the DRAWN centre an odd axis is safe from 5 up, where 1 and 3 are not.
        for odd, safe in ((3, False), (5, True), (7, True)):
            hits = 0
            for seed in range(16):
                torch.manual_seed(seed)
                odd_image = torch.full((1, 3, odd, odd), 0.5, device=device, dtype=dtype)
                hits += bool(K.RandomGaussianIllumination(p=1.0, sigma=(0.005, 0.005))(odd_image).isnan().any())
            assert (hits == 0) is safe, (odd, hits)
        # The documented default is unaffected, so the wart is the admitted extreme and not the class.
        torch.manual_seed(_FORWARD_SEED)
        assert bool(K.RandomGaussianIllumination(p=1.0)(image).isfinite().all())

    # `.compile()` on RandomGaussianIllumination swaps in a compiled transform that neither pickle nor
    # torch.save can serialize, while deepcopy still works; the linear classes keep pickling.  Measured
    # on cpu: `PicklingError: Can't pickle <function _apply_gaussian_illumination ...>` for both, and a
    # deep copy of type RandomGaussianIllumination.
    #
    # This leg is separate from the pickle pin above because calling ``torch.compile`` is not free even
    # when nothing is run through it, so it must not ride along on an ordinary test:
    #   * its first statement raises ``RuntimeError: Dynamo is not supported on Python 3.13+`` on
    #     torch<2.6, i.e. the call fails before any compilation is attempted; and
    #   * where it succeeds, Dynamo's one-time init calls
    #     ``torch.distributions.Distribution.set_default_validate_args(False)`` process-wide and never
    #     restores it, stripping argument validation from every later test in the same process.  The
    #     cpu-float16 known-failure manifest records 108
    #     ``TestRandomCutMixGen::test_valid_param_combinations`` nodes that fail only because
    #     ``beta=1e-15`` underflows to 0 in float16 and ``Beta(0, 0)`` rejects it; unvalidated they
    #     XPASS and the strict manifest gate goes red.
    #
    # ``compile`` in the test name is load-bearing, not decorative: ``pytest_collection_modifyitems``
    # drops any test whose name contains "compile" or "dynamo" unless ``KORNIA_TEST_OPTIMIZER`` is set,
    # which is what confines both hazards to the dynamo job.  Renaming it without that substring would
    # put ``torch.compile`` back into every CPU leg.  The skipif and the validate-args restore cover the
    # dynamo job itself, where conftest's session warm-up is skipped under ``KORNIA_TEST_IN_SUBPROCESS``.
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    @pytest.mark.parametrize(
        "name", ["RandomGaussianIllumination", "RandomLinearIllumination", "RandomLinearCornerIllumination"]
    )
    @pytest.mark.device_agnostic
    def test_convention_compiled_illumination_pickling(self, name):
        compiled = _illumination(name)
        validate_args = Distribution._validate_args
        try:
            compiled.compile()
        finally:
            Distribution.set_default_validate_args(validate_args)
        # `isinstance(deepcopy(x), type(x))` holds unless __deepcopy__ is overridden, so the copy is run:
        # deepcopy survives .compile() where pickle does not, and the copy still transforms.
        constant = torch.full((2, 3, 7, 9), 0.5)
        torch.manual_seed(_FORWARD_SEED)
        cloned = copy.deepcopy(compiled)
        assert isinstance(cloned, type(compiled))
        assert cloned(constant).shape == constant.shape
        if name == "RandomGaussianIllumination":
            with pytest.raises(pickle.PicklingError):
                pickle.dumps(compiled)
            with pytest.raises(pickle.PicklingError):
                torch.save(compiled, io.BytesIO())
        else:
            assert len(pickle.dumps(compiled)) > 0

    # Row 6c-43: the three RandomPlasma* classes clamp into [0, 1], and -- since #4462 closed #4445
    # by recording the sampled fractal under ``_params["plasma"]`` -- replaying a forward with the
    # recorded parameters reproduces the first output bitwise.
    #
    # The replay leg runs on a *non-constant* image on purpose.  A constant 0.5 is a fixed point of
    # the plasma contrast transform, so on that fixture RandomPlasmaContrast returns 0.5 whether or
    # not `params` is honoured and the replay assertion would hold with #4462 reverted.  The clamp
    # leg keeps the constant fixture, because that is where audit row 6c-43's literals were read.
    # Snippet used to generate expected:
    #   c = torch.full((2, 3, 7, 9), 0.5); torch.manual_seed(1234); x = torch.rand(2, 3, 7, 9)
    #   torch.manual_seed(0); aug = K.RandomPlasmaBrightness(p=1.0); print(aug(c).aminmax())
    #   torch.manual_seed(0); aug = K.RandomPlasmaBrightness(p=1.0); y = aug(x)
    #   print(list(aug._params), torch.equal(y, aug(x, params=aug._params)), (y - x).abs().max())
    #   torch.manual_seed(0); b = K.RandomPlasmaBrightness(p=1.0); y2 = b(x)
    #   print("params ignored still equal:", torch.equal(b(x), y2))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> constant fixture `min=0.370964 max=0.589035`
    # (Brightness), `0.5/0.5` (Contrast), `0/0.5` (Shadow); random fixture `0/0.98835`, `0/1`,
    # `0/0.994917` with max|out - in| of 0.129036 / 0.327044 / 0.911523, `plasma` present in every key
    # list, replay `True` for all three, and "params ignored still equal" `False` for all three.
    @pytest.mark.parametrize("name", ["RandomPlasmaBrightness", "RandomPlasmaContrast", "RandomPlasmaShadow"])
    def test_convention_plasma_clamps_and_replays_from_params(self, device, dtype, name):
        # Clamp leg: the audit's own constant fixture and its literals.
        constant = torch.full((2, 3, 7, 9), 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        constant_out = _plasma(name)(constant)
        assert float(constant_out.min()) >= 0.0
        assert float(constant_out.max()) <= 1.0

        # Replay leg: a seeded non-constant image, which no plasma transform leaves untouched.  The image
        # is 4x3x32x32, not 2x3x7x9: on the small one the largest deviation fell below the 0.05 floor on 3
        # of seeds 0..199 (RandomPlasmaShadow twice, RandomPlasmaBrightness once).  Over 200 seeds at this
        # size the smallest deviation observed was 0.0601, against a floor of 0.03.
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

    # Issue #4570: with same_on_batch=True the three RandomPlasma* classes share their scalar draws and
    # expanded fractal map view, so identical inputs receive identical transformations.  #3624 reported this
    # together with RandomGaussianNoise and RandomChannelShuffle, and #3723 fixed only those two.
    # Snippet used to generate expected:
    #   torch.manual_seed(0); aug = K.RandomPlasmaBrightness(p=1.0, same_on_batch=True)
    #   y = aug(torch.full((4, 3, 8, 8), 0.3)); print(torch.equal(aug._params["plasma"][0], aug._params["plasma"][1]))
    #   print(torch.equal(y[0], y[1]))
    # executed 2026-09-16 (torch 2.9.1+cpu, cpu) -> `True` for maps and outputs for all three classes.
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
        # The map is expanded, not repeated: a batch stride of 0 is the storage promise
        # `IntensityAugmentationBase2D`'s Convention block makes, and `repeat` would satisfy the
        # equality above while quietly allocating (and un-aliasing) one map per sample.
        assert aug._params["plasma"].stride()[0] == 0
        assert torch.equal(out[0], out[1])

    # Row 6c-43 in its new state: the `math domain error` the audit saw on a one-pixel axis is gone,
    # so a 1x1 and a 1x8 image now run and keep their shape.
    # Snippet used to generate expected:
    #   for shape in ((1, 3, 1, 1), (1, 3, 1, 8), (1, 3, 8, 1)):
    #       torch.manual_seed(0); print(shape, K.RandomPlasmaBrightness(p=1.0)(torch.full(shape, 0.5)).shape)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> each shape comes back unchanged for all three classes.
    @pytest.mark.parametrize("name", ["RandomPlasmaBrightness", "RandomPlasmaContrast", "RandomPlasmaShadow"])
    @pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 1, 8), (1, 3, 8, 1)])
    def test_convention_plasma_accepts_a_single_pixel_axis(self, device, dtype, name, shape):
        image = torch.full(shape, 0.5, device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        assert _plasma(name)(image).shape == shape

    # Row 6c-16: ``mean`` and ``std`` accept a float, a 3-list, a 3-tuple, a 1-tensor, a 3-tensor and
    # a per-sample (B, C) tensor, and every one of them actually rescales the image.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(4, 3, 6, 8)
    #   for m, s in ((0.5, 0.25), ([0.4, 0.5, 0.6], [0.2, 0.25, 0.3]), ...):
    #       torch.manual_seed(0); print((K.Normalize(mean=m, std=s, p=1.0)(x) - x).abs().max())
    #       torch.manual_seed(0); print((K.Denormalize(mean=m, std=s, p=1.0)(x) - x).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> Normalize moves the tensor by 1.99809 (scalar and
    # 1-element forms) or 1.99851 (3-element forms); Denormalize, which multiplies by std instead of
    # dividing, moves it by 0.499522 and 0.599554 respectively.
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
            # Even with distinct rows the threshold above cannot tell a per-sample application from a
            # row-0 one, so assert the exact field: replacing `mean` with `mean[0]` inside
            # kornia.enhance.normalize (or denormalize) leaves every other assertion here green.
            stats = torch.arange(1, 13, device=device, dtype=dtype).reshape(4, 3) / 10.0
            field = stats[..., None, None]
            torch.manual_seed(_FORWARD_SEED)
            per_sample = cls(mean=stats, std=stats, p=1.0)(image)
            expected = image * field + field if cls is K.Denormalize else (image - field) / field
            self.assert_close(per_sample, expected)

    # Row 6c-16: Denormalize is Normalize's inverse, and the round trip is checked against a forward
    # that actually moved the tensor (max abs change 1.99809 on this fixture) so the identity cannot
    # be satisfied by a no-op.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(4, 3, 6, 8)
    #   y = K.Normalize(mean=0.5, std=0.25, p=1.0)(x); z = K.Denormalize(mean=0.5, std=0.25, p=1.0)(y)
    #   print((y - x).abs().max(), (z - x).abs().max())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `1.99809` and `0` for the scalar form, `1.99851` and
    # `5.96046e-08` for the 3-element forms.
    @pytest.mark.parametrize("form", ["float", "list3", "tensor1", "tensorB3"])
    def test_convention_denormalize_inverts_normalize(self, device, dtype, form):
        mean, std = _statistics(form, device, dtype)
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 6, 8).to(device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        normalized = K.Normalize(mean=mean, std=std, p=1.0)(image)
        assert float((normalized - image).abs().max()) > 1.0
        torch.manual_seed(_FORWARD_SEED)
        self.assert_close(K.Denormalize(mean=mean, std=std, p=1.0)(normalized), image)

    # Issue #4577: normalization preserves image values across contiguous and strided layouts.
    @pytest.mark.parametrize("name", ["Normalize", "RandomAutoContrast"])
    def test_convention_non_contiguous_input_matches_contiguous_4577(self, device, dtype, name):
        factories = {
            "Normalize": lambda: K.Normalize(
                mean=torch.tensor([0.5], device=device, dtype=dtype),
                std=torch.tensor([0.5], device=device, dtype=dtype),
                p=1.0,
            ),
            "RandomAutoContrast": lambda: K.RandomAutoContrast(p=1.0),
        }
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 8, 8).to(device=device, dtype=dtype)
        view = image.transpose(2, 3)
        assert not view.is_contiguous()
        torch.manual_seed(_FORWARD_SEED)
        actual = factories[name]()(view)
        torch.manual_seed(_FORWARD_SEED)
        expected = factories[name]()(view.contiguous())
        assert actual.shape == view.shape
        self.assert_close(actual, expected)

    # Row 6c-16: a ``(B, C)`` statistic is applied per SAMPLE.  The pins above use one fixture for all
    # six forms, and their thresholds cannot tell a per-sample application from a row-0 or batch-mean
    # one, so this row gives two identical samples different statistics and reads the two results.
    # Snippet used to generate expected:
    #   x = torch.full((2, 1, 2, 2), 0.5)
    #   m = torch.tensor([[0.0], [0.5]]); s = torch.tensor([[1.0], [0.25]])
    #   torch.manual_seed(0); print(K.Normalize(mean=m, std=s, p=1.0)(x))
    # executed 2026-09-16 (torch 2.14.0, cpu, all four dtypes) -> sample 0 comes back `0.5`
    # throughout and sample 1 comes back `0.0` throughout.
    def test_convention_normalize_applies_a_per_sample_statistic_to_its_own_sample(self, device, dtype):
        image = torch.full((2, 1, 2, 2), 0.5, device=device, dtype=dtype)
        mean = torch.tensor([[0.0], [0.5]], device=device, dtype=dtype)
        std = torch.tensor([[1.0], [0.25]], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        out = K.Normalize(mean=mean, std=std, p=1.0)(image)
        self.assert_close(out[0], torch.full_like(out[0], 0.5))
        self.assert_close(out[1], torch.zeros_like(out[1]))

    # Row 6c-16: a mean or std whose length is neither 1 nor the channel count is rejected.
    # Snippet used to generate expected: the last line of the snippet above; executed 2026-09-15
    # (torch 2.14.0, cpu) -> `ValueError: mean length and number of channels do not match.`
    @pytest.mark.parametrize("cls", [K.Normalize, K.Denormalize])
    def test_convention_normalize_rejects_a_statistic_of_the_wrong_length(self, device, dtype, cls):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(4, 3, 6, 8).to(device=device, dtype=dtype)
        mean = torch.tensor([0.4, 0.5], device=device, dtype=dtype)
        std = torch.tensor([0.2, 0.25], device=device, dtype=dtype)
        torch.manual_seed(_FORWARD_SEED)
        with pytest.raises(ValueError, match="do not match"):
            _sync(cls(mean=mean, std=std, p=1.0)(image).device)

    # Issue #4573: Normalize and Denormalize both accept integer scalar statistics, so the pair
    # built with ``mean=0, std=255`` round-trips.
    # Snippet used to generate expected:
    #   x = torch.rand(2, 3, 4, 4); y = K.Normalize(mean=0, std=255, p=1.0)(x)
    #   z = K.Denormalize(mean=0, std=255, p=1.0)(y)
    #   torch.testing.assert_close(z, x)
    # executed 2026-09-16 (torch 2.5.1, cpu).
    @pytest.mark.device_agnostic
    def test_convention_denormalize_accepts_the_int_statistics_normalize_accepts_4573(self):
        torch.manual_seed(_FIXTURE_SEED)
        image = torch.rand(2, 3, 4, 4)
        normalized = K.Normalize(mean=0, std=255, p=1.0)(image)
        self.assert_close(normalized, image / 255)
        self.assert_close(K.Denormalize(mean=0, std=255, p=1.0)(normalized), image)
        self.assert_close(K.Denormalize(mean=0.0, std=255.0, p=1.0)(normalized), image)

    # Rows 6c-17 and 6c-18: Normalize and Denormalize hard-code ``same_on_batch=True`` (normalize.py
    # and denormalize.py, `super().__init__(p=p, same_on_batch=True, keepdim=keepdim)`), so the
    # constructor has no ``same_on_batch`` argument at all and ``p`` gates the whole batch together
    # rather than sample by sample: ``batch_prob`` is all ones or all zeros, and the number of
    # unchanged rows is exactly ``(batch_prob == 0).sum()``.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(8, 3, 6, 8)
    #   for seed in range(6):
    #       torch.manual_seed(seed); aug = K.Normalize(mean=0.5, std=0.25, p=0.5); y = aug(x)
    #       print(seed, sum(torch.equal(y[b], x[b]) for b in range(8)), aug._params["batch_prob"].tolist())
    # executed 2026-09-15 (torch 2.14.0, cpu) -> seeds 0 and 3 change all eight rows, seeds 1, 2, 4
    # and 5 change none, and no seed in range(20) produced a mixed batch_prob for either class;
    # `K.Normalize(mean=0.5, std=0.25, p=1.0, same_on_batch=True)` raises `TypeError:
    # Normalize.__init__() got an unexpected keyword argument 'same_on_batch'`.
    @pytest.mark.parametrize("cls", [K.Normalize, K.Denormalize])
    def test_convention_normalize_p_gates_the_whole_batch(self, device, dtype, cls):
        with pytest.raises(TypeError, match="same_on_batch"):
            cls(mean=0.5, std=0.25, p=1.0, same_on_batch=True)
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
            # The gate is shared by the batch, which is the visible consequence of same_on_batch=True.
            assert len(set(gate.tolist())) == 1
            applied += 8 - unchanged
        # Both outcomes occur over these six seeds, so the equality above is not read off a
        # degenerate all-on or all-off run.
        assert 0 < applied < 48
        # p=0.0 is the identity.
        torch.manual_seed(_FORWARD_SEED)
        assert torch.equal(cls(mean=0.5, std=0.25, p=0.0)(image), image)

    # Row 6c-16: both classes keep their statistics in ``flags`` rather than as buffers, so
    # ``state_dict()`` is empty and a ``Module.to(...)`` does not reach them -- the statistics are
    # not carried by the module's own dtype/device machinery and do not survive a checkpoint round
    # trip.  (The 6a pin test_convention_augmentations_are_stateless_modules covers Normalize as one
    # of five stateless representatives; this one adds Denormalize and names where the statistics
    # actually live.)
    # Snippet used to generate expected:
    #   aug = K.Normalize(mean=0.5, std=0.25, p=1.0)
    #   print(list(aug.state_dict()), [n for n, _ in aug.named_buffers()], list(aug.flags))
    #   before = aug.flags["mean"].clone(); aug = aug.to(torch.float64)
    #   print(aug.flags["mean"].dtype, torch.equal(aug.flags["mean"], before))
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `[] [] ['mean', 'std']` and, after
    # `.to(torch.float64)`, `torch.float32 True` -- unchanged -- for both classes.
    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("cls", [K.Normalize, K.Denormalize])
    def test_convention_normalize_keeps_no_state(self, cls):
        aug = cls(mean=0.5, std=0.25, p=1.0)
        assert list(aug.state_dict()) == []
        assert [name for name, _ in aug.named_buffers()] == []
        assert sorted(aug.flags) == ["mean", "std"]
        # `.to()` walks parameters and buffers; the statistics are in neither, so it leaves them be.
        before = aug.flags["mean"].clone()
        moved = aug.to(torch.float64)
        assert moved.flags["mean"].dtype == torch.float32
        assert torch.equal(moved.flags["mean"], before)

    # Row 6c-46: an unbatched (C, H, W) input is promoted to (1, C, H, W), and ``keepdim=True``
    # returns the unbatched shape again.  Checked on four classes of this half -- one filter, one
    # noise, one weather and one normalization -- so the claim is the package convention rather than
    # one implementation.  Every class of this half accepts ``keepdim``.
    # Snippet used to generate expected:
    #   torch.manual_seed(1234); x = torch.rand(3, 6, 8)
    #   torch.manual_seed(0); print(ctor(keepdim=False)(x).shape, ctor(keepdim=True)(x).shape)
    # executed 2026-09-15 (torch 2.14.0, cpu) -> `(1, 3, 6, 8)` and `(3, 6, 8)` for Normalize,
    # RandomSaltAndPepperNoise, RandomGaussianIllumination and RandomRain.
    @pytest.mark.parametrize(
        "name", ["Normalize", "RandomSaltAndPepperNoise", "RandomGaussianIllumination", "RandomRain"]
    )
    def test_convention_intensity_ops_promote_chw_to_bchw(self, device, dtype, name):
        factories = {
            "Normalize": lambda keepdim: K.Normalize(mean=0.5, std=0.25, p=1.0, keepdim=keepdim),
            "RandomSaltAndPepperNoise": lambda keepdim: K.RandomSaltAndPepperNoise(
                amount=(0.3, 0.3), p=1.0, keepdim=keepdim
            ),
            "RandomGaussianIllumination": lambda keepdim: K.RandomGaussianIllumination(
                gain=(0.5, 0.5), p=1.0, keepdim=keepdim
            ),
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


def _illumination(name: str, **kwargs):
    """Point-range constructors for the three *Illumination classes, as the audit probe used them."""
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
    """The six ``mean`` / ``std`` forms row 6c-16 records as accepted."""
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
    # Distinct rows: a per-sample statistic whose rows are all equal is indistinguishable from the
    # scalar form, so an implementation that read row 0 or the batch mean would pass unnoticed.
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
