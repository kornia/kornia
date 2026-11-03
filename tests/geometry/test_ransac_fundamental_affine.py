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

from typing import Tuple

import pytest
import torch

from kornia.geometry import RANSAC
from kornia.geometry.epipolar import sampson_epipolar_distance

from testing.base import BaseTester
from testing.two_view import affine_two_view_scene

# Mismatches 29 to 156 px (geometric distance) away from the fixture's affine epipolar geometry.
_OUTLIERS1 = [
    [220.6, 256.8], [222.6, 304.0], [376.3, 227.3], [268.2, 355.7],
    [200.4, 433.6], [414.0, 340.4], [207.3, 369.7], [358.5, 423.6],
]  # fmt: skip
_OUTLIERS2 = [
    [334.3, 283.9], [260.2, 186.7], [240.0, 268.9], [310.1, 217.3],
    [218.6, 222.2], [258.4, 327.9], [236.1, 170.9], [197.7, 294.9],
]  # fmt: skip


def _with_outliers(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The twelve exact correspondences of the affine fixture followed by eight mismatches, and the true F."""
    scene = affine_two_view_scene(device, dtype)
    x1 = torch.cat([scene["x1"][0], torch.tensor(_OUTLIERS1, device=device, dtype=dtype)])
    x2 = torch.cat([scene["x2"][0], torch.tensor(_OUTLIERS2, device=device, dtype=dtype)])
    return x1, x2, scene["F"][0]


class TestRANSACFundamentalAffine(BaseTester):
    @pytest.mark.parametrize("score_type", ["ransac", "msac"])
    @pytest.mark.parametrize("lo_sample_size", [None, 8])
    def test_inliers_and_model_with_outliers(self, device, dtype, score_type, lo_sample_size):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half precision rounds the fixture's pixel coordinates to steps of 0.125 to 0.25")
        x1, x2, expected = _with_outliers(device, dtype)
        estimator = RANSAC(
            "fundamental_affine",
            inl_th=1.0,
            batch_size=32,
            max_samples=256,
            score_type=score_type,
            lo_sample_size=lo_sample_size,
            seed=0,
        )
        model, inliers = estimator(x1, x2)
        assert model.dtype == dtype
        assert model.device == device
        assert torch.equal(inliers, torch.arange(20, device=device) < 12)
        assert not model[:2, :2].any()
        tol = {"rtol": 1e-7, "atol": 1e-9} if dtype == torch.float64 else {"rtol": 5e-5, "atol": 1e-7}
        self.assert_close(model, expected, **tol)

    def test_threshold_is_the_geometric_distance(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half precision rounds the fixture's pixel coordinates to steps of 0.125 to 0.25")
        x1, x2, _ = _with_outliers(device, dtype)
        # Moves of x2 along the epipolar lines' normal (a, b) by which the correspondence is 0.75 px and 1.5 px from
        # the nearest pair of corresponding points (the four-dimensional distance, which the Sampson distance of an
        # affine F equals): r = a dx + b dy = d * |(a, b, c, d)|.
        x1, x2 = torch.cat([x1, x1[:2]]), torch.cat([x2, x2[:2]])
        x2[-2:] += torch.tensor([[0.532378098326968, -0.9264217301595274], [1.064756196653936, -1.852843460319055]],
                                device=device, dtype=dtype)  # fmt: skip
        model, inliers = RANSAC("fundamental_affine", inl_th=1.0, max_samples=256, seed=0)(x1, x2)
        assert inliers[-2]
        assert not inliers[-1]
        classified = sampson_epipolar_distance(x1[None], x2[None], model[None])[0] <= 1.0
        assert torch.equal(inliers, classified)

    def test_minimal_sample_is_four(self, device, dtype):
        x1, x2, _ = _with_outliers(device, dtype)
        with pytest.raises(ValueError, match=r"at least \[4, 2\]"):
            RANSAC("fundamental_affine").validate_inputs(x1[:3], x2[:3])
        # Four correspondences fit a model exactly, which is no evidence of a consensus.
        model, inliers = RANSAC("fundamental_affine", max_samples=16, seed=0)(x1[:4], x2[:4])
        assert not model.any()
        assert not inliers.any()

    @pytest.mark.parametrize("kwargs", [{"local_optimization": "lm"}, {"degensac": True}, {"compile": True}])
    def test_unsupported_options_raise(self, kwargs):
        with pytest.raises(ValueError):
            RANSAC("fundamental_affine", **kwargs)
