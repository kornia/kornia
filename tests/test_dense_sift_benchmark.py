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

"""Geometric-quality checks for the shared-pyramid SIFT benchmark."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("PIL")

from benchmarks.feature.dense_sift import homography_quality
from kornia.geometry import transform_points


def test_ransac_recovers_homography_with_outliers():
    generator = torch.Generator().manual_seed(12)
    source = torch.rand(60, 2, generator=generator) * 100
    truth = torch.tensor([[1.1, 0.05, 3.0], [-0.03, 0.9, 8.0], [0.0002, -0.0001, 1.0]])
    target = transform_points(truth[None], source[None])[0]
    target[-10:] = torch.rand(10, 2, generator=generator) * 100
    result = homography_quality(source, target, truth, 100, 100)
    assert result["ransac_status"] == "estimated"
    assert result["ransac_inliers"] >= 50
    assert result["ransac_corner_error_px"] < 0.01


def test_insufficient_matches_has_no_fake_zero_error():
    result = homography_quality(torch.zeros(3, 2), torch.zeros(3, 2), torch.eye(3), 100, 100)
    assert result["ransac_status"] == "insufficient_matches"
    assert result["ransac_inliers"] == 0
    assert result["ransac_corner_error_px"] is None


def test_corner_metric_uses_non_square_pixel_extents_and_mean_l1(monkeypatch):
    import benchmarks.feature.dense_sift as benchmark

    # On a 21-wide, 11-high image, corner L1 errors are 0, 2, 4, 2.
    estimate = torch.diag(torch.tensor([1.1, 1.2, 1.0]))
    monkeypatch.setattr(
        benchmark, "RANSAC", lambda *args, **kwargs: lambda source, target: (estimate, torch.ones(4, dtype=torch.bool))
    )
    result = homography_quality(torch.zeros(4, 2), torch.zeros(4, 2), torch.eye(3), 11, 21)
    assert result["ransac_corner_error_px"] == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("inliers,status", [(0, "failed_estimate"), (4, "nonfinite_projection")])
def test_failed_geometry_retains_null_error(monkeypatch, inliers, status):
    import benchmarks.feature.dense_sift as benchmark

    # The denominator is x: the two left-hand corners project to infinity.
    estimate = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    monkeypatch.setattr(
        benchmark,
        "RANSAC",
        lambda *args, **kwargs: lambda source, target: (estimate, torch.arange(4) < inliers),
    )
    result = homography_quality(torch.zeros(4, 2), torch.zeros(4, 2), torch.eye(3), 11, 21)
    assert result["ransac_status"] == status
    assert result["ransac_corner_error_px"] is None
