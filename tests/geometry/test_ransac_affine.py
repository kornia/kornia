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

import pytest
import torch

from kornia.geometry import RANSAC, transform_points
from kornia.geometry.homography import oneway_transfer_error
from kornia.geometry.ransac import _affine_transfer_error

from testing.base import BaseTester


class TestRANSACAffine(BaseTester):
    @pytest.mark.parametrize("score_type", ["ransac", "msac"])
    @pytest.mark.parametrize("prosac", [False, True])
    @pytest.mark.parametrize("lo_sample_size", [None, 8])
    def test_affine_with_outliers(self, device, dtype, score_type, prosac, lo_sample_size):
        grid = torch.arange(-3, 5, device=device, dtype=dtype)
        source = torch.cartesian_prod(grid, grid)
        expected = torch.tensor([[1.0, 0.25, 2.0], [-0.5, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        target = transform_points(expected[None], source[None])[0]
        target[-16:] += 20.0
        estimator = RANSAC(
            "affine",
            inl_th=0.125,
            batch_size=32,
            max_samples=256,
            max_lo_iters=3,
            score_type=score_type,
            prosac_sampling=prosac,
            lo_sample_size=lo_sample_size,
            seed=4,
        )
        model, inliers = estimator(source, target)
        self.assert_close(model, expected)
        self.assert_close(inliers, torch.arange(len(source), device=device) < 48)
        self.assert_close(model[2], expected[2], atol=0, rtol=0)
        classified = oneway_transfer_error(source[None], target[None], model[None])[0] <= estimator.inl_th**2
        self.assert_close(inliers, classified)
        assert model.device == device
        assert model.dtype == dtype
        assert inliers.dtype == torch.bool

    @pytest.mark.parametrize("case", ["collinear", "duplicate", "minimal_only"])
    def test_no_supported_model(self, device, dtype, case):
        source = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], device=device, dtype=dtype)
        if case == "collinear":
            source[:, 1] = 0
        elif case == "duplicate":
            source[:] = 1
        else:
            source = source[:3]
        estimator = RANSAC("affine", batch_size=8, max_samples=16, seed=0)
        model, inliers = estimator(source, source + 1)
        self.assert_close(model, torch.zeros(3, 3, device=device, dtype=dtype), atol=0, rtol=0)
        assert inliers.shape == (len(source),)
        assert not inliers.any()

    def test_threshold_is_target_distance(self, device, dtype):
        source = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 1.0]], device=device, dtype=dtype)
        target = source.clone()
        target[-2:, 0] += torch.tensor([0.375, 0.75], device=device, dtype=dtype)
        estimator = RANSAC("affine", inl_th=0.5)
        _, inliers, _, count = estimator.verify(source, target, torch.eye(3, device=device, dtype=dtype)[None], 0.5**2)
        self.assert_close(inliers, torch.tensor([True, True, True, True, False], device=device))
        assert count == 4

    def test_transfer_error_matches_oneway(self, device, dtype):
        generator = torch.Generator().manual_seed(11)
        source = (1000 * torch.rand(1, 50, 2, generator=generator)).to(device=device, dtype=dtype)
        target = (1000 * torch.rand(1, 50, 2, generator=generator)).to(device=device, dtype=dtype)
        models = torch.eye(3).repeat(6, 1, 1)
        models[:, :2] += 0.1 * torch.randn(6, 2, 3, generator=generator) + torch.tensor([0.0, 0.0, 20.0])
        models = models.to(device=device, dtype=dtype)
        actual = _affine_transfer_error(source, target, models)
        assert actual.shape == (6, 50)
        assert actual.dtype == dtype
        self.assert_close(actual, oneway_transfer_error(source, target, models))

    @pytest.mark.parametrize("kwargs", [{"local_optimization": "lm"}, {"compile": True}, {"degensac": True}])
    def test_unsupported_refinement(self, kwargs):
        with pytest.raises(ValueError):
            RANSAC("affine", **kwargs)

    def test_auto_batch_size(self):
        # Three-point samples: the epipolar models' CPU batches, the homographies' accelerator batches.
        estimator = RANSAC("affine")
        assert estimator.resolve_batch_size(2000, torch.device("cpu")) == 128
        assert estimator.resolve_batch_size(500, torch.device("cpu")) == 262
        assert estimator.resolve_batch_size(2000, torch.device("cuda")) == 8192
        assert estimator.resolve_batch_size(50000, torch.device("cuda")) == 2684

    def test_too_few_correspondences(self, device, dtype):
        points = torch.zeros(2, 2, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="at least"):
            RANSAC("affine")(points, points)

    def test_seeded_reproducibility(self, device, dtype):
        generator = torch.Generator().manual_seed(21)
        source = torch.rand(40, 2, generator=generator).to(device=device, dtype=dtype)
        target = source + 2
        target[-10:] -= 20
        estimator = RANSAC("affine", inl_th=0.1, max_samples=32, batch_size=16, seed=7, max_lo_iters=0)
        first = estimator(source, target)
        second = estimator(source, target)
        for actual, expected in zip(first, second):
            self.assert_close(actual, expected, atol=0, rtol=0)
