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

import pytest
import torch

import kornia

from testing.base import BaseTester


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
