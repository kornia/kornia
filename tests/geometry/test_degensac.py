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

import math

import pytest
import torch

from kornia.geometry._degensac import (
    _h_degenerate_sample,
    _homographies_from_fundamental,
    _left_epipole,
    _plane_parallax_fundamentals,
)
from kornia.geometry.epipolar.fundamental import _epipolar_design_rows, _seven_point_candidates
from kornia.geometry.epipolar.numeric import cross_product_matrix

# The kernels run on the host in float64 by design (RANSAC moves one sample there), so these tests do not take the
# device and dtype fixtures; RANSAC's own tests cover the device paths.
F64 = torch.float64


def _two_view_geometry():
    """A calibrated pair, the plane z - 0.2 x = 6 and its homography, and F; all in pixels, float64."""
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=F64)
    a, b = math.radians(8.0), math.radians(3.0)
    Ry = torch.tensor([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]], dtype=F64)
    Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]], dtype=F64)
    R, t = Ry @ Rx, torch.tensor([-1.0, 0.1, 0.15], dtype=F64)
    normal, distance = torch.tensor([-0.2, 0.0, 1.0], dtype=F64), 6.0
    K_inv = torch.linalg.inv(K)
    H = K @ (R + torch.outer(t, normal) / distance) @ K_inv
    F = K_inv.T @ cross_product_matrix(t) @ R @ K_inv
    return K, R, t, H / H[2, 2], F / F.norm()


def _points(K, R, t, count, planar, generator):
    if planar:
        xy = (torch.rand(count, 2, generator=generator, dtype=F64) - 0.5) * torch.tensor([4.0, 3.0], dtype=F64)
        X = torch.cat([xy, 6.0 + 0.2 * xy[:, :1]], 1)
    else:
        X = torch.cat(
            [
                (torch.rand(count, 2, generator=generator, dtype=F64) - 0.5) * 2,
                3 + 5 * torch.rand(count, 1, generator=generator, dtype=F64),
            ],
            1,
        )
    x1, x2 = X @ K.T, (X @ R.T + t) @ K.T
    return x1 / x1[:, 2:], x2 / x2[:, 2:]


def _same_up_to_scale(A, B):
    A, B = A / A.norm(), B / B.norm()
    return float(torch.minimum((A - B).abs().max(), (A + B).abs().max()))


class TestLeftEpipole:
    def test_null_vector_of_transpose(self):
        _, _, _, _, F = _two_view_geometry()
        e = _left_epipole(F[None])[0]
        assert float((F.T @ e).abs().max()) < 1e-12
        assert abs(float(e.norm()) - 1.0) < 1e-12


class TestHomographiesFromFundamental:
    def test_recovers_plane_homography(self):
        K, R, t, H, F = _two_view_geometry()
        x1, x2 = _points(K, R, t, 3, True, torch.Generator().manual_seed(0))
        estimate = _homographies_from_fundamental(F, x1[None], x2[None])[0]
        assert _same_up_to_scale(estimate, H) < 1e-10
        compatibility = estimate.T @ F  # skew-symmetric exactly when H is compatible with F
        assert float((compatibility + compatibility.T).abs().max()) < 1e-10 * float(compatibility.abs().max())

    def test_collinear_triplet_is_nan(self):
        _, _, _, _, F = _two_view_geometry()
        x1 = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]]], dtype=F64)
        assert torch.isnan(_homographies_from_fundamental(F, x1, x1)).any()


class TestHDegenerateSample:
    THRESHOLD = 3.0  # 3 t with t = 1 px^2; the sample is noise-free

    def test_five_plane_pairs_flag_the_compatible_root(self):
        # Five H-related pairs and two others: only one seven-point root is guaranteed compatible with H (paper,
        # section 2.3), so only that one must flag.
        K, R, t, H, _ = _two_view_geometry()
        generator = torch.Generator().manual_seed(1)
        p1, p2 = _points(K, R, t, 5, True, generator)
        o1, o2 = _points(K, R, t, 2, False, generator)
        x1, x2 = torch.cat([p1, o1]), torch.cat([p2, o2])
        candidates, valid = _seven_point_candidates(_epipolar_design_rows(x1[None], x2[None]))
        compatible = []
        for root in range(3):
            if not bool(valid[0, root]):
                continue
            product = H.T @ candidates[0, root]
            skew = float((product + product.T).abs().max() / product.abs().max())
            compatible.append((skew, root))
        _, root = min(compatible)
        assert _h_degenerate_sample(candidates[0, root], x1, x2, self.THRESHOLD) is not None

    @pytest.mark.parametrize("on_plane", [6, 7])
    def test_six_and_seven_plane_pairs_flag_a_compatible_model(self, on_plane):
        # Exact 6- and 7-of-7 data make the seven-point cubic vanish identically (paper, sections 2.1-2.2), so the
        # model is an explicit member of the compatible family F = [e']_x H rather than a solver root.
        K, R, t, H, _ = _two_view_geometry()
        generator = torch.Generator().manual_seed(2)
        p1, p2 = _points(K, R, t, on_plane, True, generator)
        o1, o2 = _points(K, R, t, 7 - on_plane, False, generator)
        x1, x2 = torch.cat([p1, o1]), torch.cat([p2, o2])
        F = cross_product_matrix(K @ t) @ H
        homography = _h_degenerate_sample(F / F.norm(), x1, x2, self.THRESHOLD)
        assert homography is not None
        assert _same_up_to_scale(homography, H) < 1e-6

    def test_general_position_does_not_flag(self):
        K, R, t, _, _ = _two_view_geometry()
        x1, x2 = _points(K, R, t, 7, False, torch.Generator().manual_seed(3))
        candidates, valid = _seven_point_candidates(_epipolar_design_rows(x1[None], x2[None]))
        for root in range(3):
            if bool(valid[0, root]):
                assert _h_degenerate_sample(candidates[0, root], x1, x2, self.THRESHOLD) is None


class TestPlaneParallax:
    def test_two_off_plane_pairs_give_the_fundamental_matrix(self):
        K, R, t, H, F = _two_view_geometry()
        o1, o2 = _points(K, R, t, 4, False, torch.Generator().manual_seed(4))
        models = _plane_parallax_fundamentals(H, o1, o2, torch.tensor([0, 2]), torch.tensor([1, 3]))
        for model in models:
            assert _same_up_to_scale(model, F) < 1e-10
            assert abs(float(model.norm()) - 1.0) < 1e-12

    def test_parallel_lines_give_nan(self):
        _, _, _, H, _ = _two_view_geometry()
        x1 = torch.tensor([[100.0, 100.0, 1.0]], dtype=F64)
        x2 = torch.tensor([[150.0, 120.0, 1.0]], dtype=F64)
        models = _plane_parallax_fundamentals(H, x1, x2, torch.tensor([0]), torch.tensor([0]))
        assert torch.isnan(models).all()
