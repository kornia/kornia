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

import itertools

import pytest
import torch

import kornia

from testing.base import BaseTester


def _reference_nms_mask(x: torch.Tensor, kernel_size: tuple[int, ...]) -> torch.Tensor:
    """Strict local maxima, written out from the definition rather than from the implementation.

    A position is a maximum when its whole window lies inside the input and its value is strictly
    greater than every *other* value in that window. A position whose window would run off an edge is
    not a maximum: the comparisons that would decide it have not been made. This is the rule the
    ``(3, 3)``/``(5, 5)``/``(7, 7)`` fast paths have always applied, and since #4239 the general path
    applies it too.

    Built by shifting whole views and reducing with :func:`torch.maximum`, so it shares no structure
    with the pooled implementation it checks.
    """
    ndim = len(kernel_size)
    sizes = x.shape[-ndim:]
    before = [(k - 1) // 2 for k in kernel_size]
    after = [k - b - 1 for k, b in zip(kernel_size, before)]
    centre_at = tuple(slice(b, s - a) for b, a, s in zip(before, after, sizes))
    centre = x[(..., *centre_at)]
    neighbours = torch.full_like(centre, float("-inf"))
    for offset in itertools.product(*(range(k) for k in kernel_size)):
        if list(offset) == before:
            continue  # the centre itself is not one of its own neighbours
        shifted = tuple(slice(o, o + s - k + 1) for o, s, k in zip(offset, sizes, kernel_size))
        neighbours = torch.maximum(neighbours, x[(..., *shifted)])
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[(..., *centre_at)] = centre > neighbours
    return mask


class TestNMS2d(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_shape_batch(self, device):
        inp = torch.ones(4, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_shape_5x5(self, device):
        inp = torch.ones(1, 2, 10, 10, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((5, 5)).to(device)
        assert nms(inp).shape == inp.shape

    def test_shape_7x7(self, device):
        inp = torch.ones(1, 2, 14, 14, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((7, 7)).to(device)
        assert nms(inp).shape == inp.shape

    def test_nms_5x5_single_peak(self, device):
        # A single isolated peak should be preserved; everything else zeroed.
        inp = torch.zeros(1, 1, 15, 15, device=device)
        inp[0, 0, 7, 7] = 1.0
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((5, 5)).to(device)
        out = nms(inp)
        assert out[0, 0, 7, 7].item() == pytest.approx(1.0)
        assert out.sum().item() == pytest.approx(1.0)

    def test_nms_5x5_suppress_close_neighbor(self, device):
        # 5x5 kernel has radius 2 (checks ±2 pixels).  Two peaks separated by exactly 2 pixels
        # are inside each other's window; only the higher one survives.
        inp = torch.zeros(1, 1, 20, 20, device=device)
        inp[0, 0, 8, 8] = 2.0
        inp[0, 0, 8, 10] = 1.0  # distance 2 — inside the 5x5 neighbourhood of (8,8)
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((5, 5)).to(device)
        out = nms(inp)
        assert out[0, 0, 8, 8].item() == pytest.approx(2.0)
        assert out[0, 0, 8, 10].item() == pytest.approx(0.0)

    def test_nms_5x5_keep_far_peaks(self, device):
        # Two peaks separated by 5 pixels (outside a 5x5 ±2 window): both survive.
        inp = torch.zeros(1, 1, 20, 20, device=device)
        inp[0, 0, 4, 4] = 2.0
        inp[0, 0, 4, 9] = 1.0  # distance 5 — outside the 5x5 neighbourhood
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((5, 5)).to(device)
        out = nms(inp)
        assert out[0, 0, 4, 4].item() == pytest.approx(2.0)
        assert out[0, 0, 4, 9].item() == pytest.approx(1.0)

    def test_nms_5x5_matches_3x3_on_well_separated_peaks(self, device):
        # When peaks are far apart, 3x3 and 5x5 NMS should agree.
        inp = torch.zeros(1, 1, 30, 30, device=device)
        inp[0, 0, 5, 5] = 3.0
        inp[0, 0, 20, 20] = 2.0
        nms3 = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        nms5 = kornia.geometry.subpix.NonMaximaSuppression2d((5, 5)).to(device)
        out3 = nms3(inp)
        out5 = nms5(inp)
        # Both NMS variants should detect the same two peaks (peaks are far from each other).
        assert (out3 > 0).equal(out5 > 0)

    def test_nms_7x7_single_peak(self, device):
        # A single isolated peak should be preserved.
        inp = torch.zeros(1, 1, 20, 20, device=device)
        inp[0, 0, 10, 10] = 1.0
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((7, 7)).to(device)
        out = nms(inp)
        assert out[0, 0, 10, 10].item() == pytest.approx(1.0)
        assert out.sum().item() == pytest.approx(1.0)

    def test_nms_7x7_suppress_close_neighbor(self, device):
        # 7x7 kernel has radius 3 (checks ±3 pixels).  Two peaks separated by exactly 3 pixels
        # are inside each other's window; only the higher one survives.
        inp = torch.zeros(1, 1, 25, 25, device=device)
        inp[0, 0, 10, 10] = 2.0
        inp[0, 0, 10, 13] = 1.0  # distance 3 — inside the 7x7 neighbourhood
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((7, 7)).to(device)
        out = nms(inp)
        assert out[0, 0, 10, 10].item() == pytest.approx(2.0)
        assert out[0, 0, 10, 13].item() == pytest.approx(0.0)

    def test_nms_7x7_keep_far_peaks(self, device):
        # Two peaks separated by 7 pixels (outside 7x7 ±3 window): both survive.
        inp = torch.zeros(1, 1, 30, 30, device=device)
        inp[0, 0, 5, 5] = 2.0
        inp[0, 0, 5, 12] = 1.0  # distance 7
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((7, 7)).to(device)
        out = nms(inp)
        assert out[0, 0, 5, 5].item() == pytest.approx(2.0)
        assert out[0, 0, 5, 12].item() == pytest.approx(1.0)

    def test_gradcheck_5x5(self, device):
        img = torch.rand(1, 2, 7, 7, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms2d, (img, (5, 5)), nondet_tol=1e-4)

    def test_gradcheck_7x7(self, device):
        img = torch.rand(1, 2, 9, 9, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms2d, (img, (7, 7)), nondet_tol=1e-4)

    def test_nms(self, device):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.7, 1.1, 0.0, 1.0, 2.0, 0.0],
                        [0.0, 0.8, 1.0, 0.0, 1.0, 1.0, 0.0],
                    ]
                ]
            ],
            device=device,
        ).float()

        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0, 0, 0.0, 0, 0.0, 0.0],
                        [0.0, 0, 1.1, 0.0, 0.0, 2.0, 0.0],
                        [0.0, 0, 0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
        ).float()
        nms = kornia.geometry.subpix.NonMaximaSuppression2d((3, 3)).to(device)
        scores = nms(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms2d, (img, (3, 3)), nondet_tol=1e-4)

    # ------------------------------------------------------------------ #4239, #4240
    @pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5), (7, 7), (9, 9), (15, 15), (4, 4)])
    def test_matches_the_reference_rule(self, device, dtype, kernel_size):
        # The three hand-written fast paths and the general path have to answer the same question.
        # Parametrizing one reference over both is the pin: (3, 3), (5, 5) and (7, 7) take the
        # explicit branches, the rest take the general one.
        torch.manual_seed(0)
        inp = (torch.rand(2, 3, 23, 27, device=device) * 4).to(dtype)
        self.assert_close(
            kornia.geometry.subpix.nms2d(inp, kernel_size, mask_only=True).to(dtype),
            _reference_nms_mask(inp, kernel_size).to(dtype),
        )

    @pytest.mark.parametrize("kernel_size", [(5, 5), (9, 9)])
    def test_a_plateau_holds_no_strict_maximum(self, device, dtype, kernel_size):
        # The suppression is strict: tied neighbours kill each other. A max-pool written as
        # `x == max_pool(x)` would keep the whole plateau instead.
        inp = torch.zeros(1, 1, 31, 31, device=device, dtype=dtype)
        inp[0, 0, 10:13, 10:13] = 1.0  # nine equal, mutually adjacent maxima
        inp[0, 0, 22, 22] = 0.5  # one strictly larger than everything around it
        mask = kornia.geometry.subpix.nms2d(inp, kernel_size, mask_only=True)
        assert not bool(mask[0, 0, 10:13, 10:13].any())
        assert bool(mask[0, 0, 22, 22])
        assert int(mask.sum()) == 1

    @pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5), (9, 9), (15, 15)])
    def test_the_border_strip_is_never_a_maximum(self, device, dtype, kernel_size):
        # A peak whose window does not fit inside the image has not been compared against the
        # neighbours that would decide it, so it is not reported (#4239). The strip is
        # `(k - 1) // 2` wide, the same one the explicit paths have always rejected.
        radius = (kernel_size[0] - 1) // 2
        inside = torch.zeros(1, 1, 41, 41, device=device, dtype=dtype)
        inside[0, 0, radius, radius] = 1.0
        assert bool(kornia.geometry.subpix.nms2d(inside, kernel_size, mask_only=True)[0, 0, radius, radius])
        outside = torch.zeros(1, 1, 41, 41, device=device, dtype=dtype)
        outside[0, 0, radius - 1, radius] = 1.0
        assert not bool(kornia.geometry.subpix.nms2d(outside, kernel_size, mask_only=True).any())

    @pytest.mark.parametrize("kernel_size", [(11, 5), (5, 11), (13, 9)])
    def test_non_square_kernel_size(self, device, dtype, kernel_size):
        # A non-square window used to raise out of an internal `view`, because the conv kernel was
        # built with its two extents swapped and the padding was applied to the wrong pair of
        # edges (#4240).
        torch.manual_seed(0)
        inp = (torch.rand(1, 2, 29, 31, device=device) * 4).to(dtype)
        self.assert_close(
            kornia.geometry.subpix.nms2d(inp, kernel_size, mask_only=True).to(dtype),
            _reference_nms_mask(inp, kernel_size).to(dtype),
        )

    def test_unit_window_suppresses_nothing(self, device, dtype):
        # A 1x1 window has no neighbours, so every position is vacuously a strict maximum and the
        # input comes back untouched. It used to come back thresholded at zero, an artifact of the
        # zeroed centre tap summing to 0.0 rather than anything NMS means.
        inp = torch.tensor([[[[-1.0, 0.0, 2.0], [0.5, -0.5, 0.0]]]], device=device, dtype=dtype)
        assert bool(kornia.geometry.subpix.nms2d(inp, (1, 1), mask_only=True).all())
        self.assert_close(kornia.geometry.subpix.nms2d(inp, (1, 1)), inp)

    def test_window_larger_than_the_input_finds_no_maxima(self, device, dtype):
        inp = torch.rand(1, 1, 5, 5, device=device, dtype=dtype)
        assert int(kornia.geometry.subpix.nms2d(inp, (9, 9), mask_only=True).sum()) == 0

    def test_module_and_functional_agree(self, device, dtype):
        torch.manual_seed(0)
        inp = (torch.rand(1, 2, 21, 23, device=device) * 4).to(dtype)
        module = kornia.geometry.subpix.NonMaximaSuppression2d((9, 9)).to(device)
        self.assert_close(module(inp), kornia.geometry.subpix.nms2d(inp, (9, 9)))

    def test_gradcheck_general_path(self, device):
        img = torch.rand(1, 2, 13, 13, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms2d, (img, (9, 9)), nondet_tol=1e-4)


class TestNMS3d(BaseTester):
    def test_shape(self, device):
        inp = torch.ones(1, 1, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d((3, 3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_shape_batch(self, device):
        inp = torch.ones(4, 1, 3, 4, 4, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d((3, 3, 3)).to(device)
        assert nms(inp).shape == inp.shape

    def test_nms(self, device):
        inp = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 2.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ]
        ).to(device)

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 2.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ]
            ]
        ).to(device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d((3, 3, 3)).to(device)
        scores = nms(inp)
        self.assert_close(scores, expected, atol=1e-4, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, depth, height, width = 1, 1, 4, 5, 4
        img = torch.rand(batch_size, channels, depth, height, width, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms3d, (img, (3, 3, 3)), nondet_tol=1e-4)

    # ------------------------------------------------------------------------------- #4241
    @pytest.mark.parametrize("kernel_size", [(3, 3, 3), (5, 5, 5), (3, 5, 7), (5, 3, 3)])
    def test_matches_the_reference_rule(self, device, dtype, kernel_size):
        # Every kernel size other than (3, 3, 3) used to raise: `_compute_zero_padding3d` defined a
        # `(k - 1) // 2` helper and then returned the full kernel sizes instead, so the padded
        # volume did not match the kernel and the `view` after the convolution failed (#4241).
        torch.manual_seed(0)
        inp = (torch.rand(2, 2, 11, 13, 12, device=device) * 4).to(dtype)
        self.assert_close(
            kornia.geometry.subpix.nms3d(inp, kernel_size, mask_only=True).to(dtype),
            _reference_nms_mask(inp, kernel_size).to(dtype),
        )

    @pytest.mark.parametrize("kernel_size", [(5, 5, 5), (3, 5, 7)])
    def test_shape_general_kernel(self, device, kernel_size):
        inp = torch.ones(1, 2, 9, 12, 12, device=device)
        nms = kornia.geometry.subpix.NonMaximaSuppression3d(kernel_size).to(device)
        assert nms(inp).shape == inp.shape

    def test_a_plateau_holds_no_strict_maximum(self, device, dtype):
        inp = torch.zeros(1, 1, 11, 21, 21, device=device, dtype=dtype)
        inp[0, 0, 4:6, 8:10, 8:10] = 1.0
        inp[0, 0, 5, 15, 15] = 0.5
        mask = kornia.geometry.subpix.nms3d(inp, (5, 5, 5), mask_only=True)
        assert not bool(mask[0, 0, 4:6, 8:10, 8:10].any())
        assert int(mask.sum()) == 1

    def test_window_larger_than_the_input_finds_no_maxima(self, device, dtype):
        inp = torch.rand(1, 1, 2, 4, 4, device=device, dtype=dtype)
        assert int(kornia.geometry.subpix.nms3d(inp, (5, 5, 5), mask_only=True).sum()) == 0

    def test_gradcheck_general_path(self, device):
        img = torch.rand(1, 1, 7, 9, 9, device=device, dtype=torch.float64)
        self.gradcheck(kornia.geometry.subpix.nms3d, (img, (5, 5, 5)), nondet_tol=1e-4)


class TestNMS3dMinMax(BaseTester):
    def test_shapes(self, device):
        inp = torch.randn(1, 1, 5, 10, 10, device=device)
        max_mask, min_mask = kornia.geometry.subpix.nms3d_minmax(inp)
        assert max_mask.shape == inp.shape
        assert min_mask.shape == inp.shape
        assert max_mask.dtype == torch.bool
        assert min_mask.dtype == torch.bool

    def test_consistent_with_nms3d(self, device):
        """nms3d_minmax must match nms3d(x) and nms3d(-x) exactly."""
        inp = torch.randn(2, 3, 7, 12, 12, device=device)
        max_mask, min_mask = kornia.geometry.subpix.nms3d_minmax(inp)
        max_ref = kornia.geometry.subpix.nms3d(inp, (3, 3, 3), mask_only=True)
        min_ref = kornia.geometry.subpix.nms3d(-inp, (3, 3, 3), mask_only=True)
        assert max_mask.equal(max_ref), "max mask mismatch"
        assert min_mask.equal(min_ref), "min mask mismatch"

    def test_no_overlap(self, device):
        """A voxel cannot be both a strict local maximum and minimum."""
        inp = torch.randn(1, 1, 5, 10, 10, device=device)
        max_mask, min_mask = kornia.geometry.subpix.nms3d_minmax(inp)
        assert not (max_mask & min_mask).any()

    def test_gradcheck(self, device):
        # nms3d_minmax is not differentiable (bool masks), so we just check it runs.
        inp = torch.randn(1, 1, 5, 7, 7, device=device)
        kornia.geometry.subpix.nms3d_minmax(inp)
