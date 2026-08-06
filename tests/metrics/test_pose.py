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

import kornia

from testing.base import BaseTester

# 90-degree rotation about +z.
ROT_Z_90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


class TestAngleErrorMat(BaseTester):
    def test_identity_is_zero(self, device, dtype):
        eye = torch.eye(3, device=device, dtype=dtype)
        expected = torch.tensor(0.0, device=device, dtype=dtype)
        self.assert_close(kornia.metrics.angle_error_mat(eye, eye), expected)

    def test_known_angle(self, device, dtype):
        eye = torch.eye(3, device=device, dtype=dtype)
        rot = torch.tensor(ROT_Z_90, device=device, dtype=dtype)
        expected = torch.tensor(90.0, device=device, dtype=dtype)
        self.assert_close(kornia.metrics.angle_error_mat(eye, rot), expected)

    def test_batched(self, device, dtype):
        eye = torch.eye(3, device=device, dtype=dtype)
        rot = torch.tensor(ROT_Z_90, device=device, dtype=dtype)
        batch1 = torch.stack([eye, eye])
        batch2 = torch.stack([eye, rot])
        expected = torch.tensor([0.0, 90.0], device=device, dtype=dtype)
        out = kornia.metrics.angle_error_mat(batch1, batch2)
        assert out.shape == (2,)
        self.assert_close(out, expected)

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import ShapeError

        with pytest.raises(ShapeError):
            kornia.metrics.angle_error_mat(
                torch.eye(3, device=device, dtype=dtype), torch.ones(2, device=device, dtype=dtype)
            )


class TestAngleErrorVec(BaseTester):
    def test_aligned_orthogonal_opposite(self, device, dtype):
        x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        self.assert_close(kornia.metrics.angle_error_vec(x, x), torch.tensor(0.0, device=device, dtype=dtype))
        self.assert_close(kornia.metrics.angle_error_vec(x, y), torch.tensor(90.0, device=device, dtype=dtype))
        self.assert_close(kornia.metrics.angle_error_vec(x, -x), torch.tensor(180.0, device=device, dtype=dtype))

    def test_batched(self, device, dtype):
        a = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device, dtype=dtype)
        b = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 90.0], device=device, dtype=dtype)
        out = kornia.metrics.angle_error_vec(a, b)
        assert out.shape == (2,)
        self.assert_close(out, expected)

    def test_zero_vector_is_nan(self, device, dtype):
        # Undefined angle, so NaN rather than an exception.
        zero = torch.zeros(3, device=device, dtype=dtype)
        unit = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        assert torch.isnan(kornia.metrics.angle_error_vec(zero, unit))

    def test_gradcheck(self, device):
        v1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float64)
        v2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
        self.gradcheck(kornia.metrics.angle_error_vec, (v1, v2), requires_grad=(True, False))


class TestTranslationAte(BaseTester):
    def test_single_returns_batch_shape(self, device, dtype):
        # ||[0,0,0] - [3,4,0]|| = 5 (3-4-5 triangle).
        t = torch.zeros(3, device=device, dtype=dtype)
        t_gt = torch.tensor([3.0, 4.0, 0.0], device=device, dtype=dtype)
        out = kornia.metrics.translation_ate(t, t_gt)
        assert out.shape == (1,)
        self.assert_close(out, torch.tensor([5.0], device=device, dtype=dtype))

    def test_batched(self, device, dtype):
        t = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        t_gt = torch.tensor([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        out = kornia.metrics.translation_ate(t, t_gt)
        assert out.shape == (2,)
        self.assert_close(out, torch.tensor([5.0, 0.0], device=device, dtype=dtype))

    def test_mismatched_shapes_raise(self, device, dtype):
        # This used to broadcast to (1, B) instead of raising.
        t = torch.zeros(3, device=device, dtype=dtype)
        t_gt = torch.ones(4, 3, device=device, dtype=dtype)
        with pytest.raises(Exception):
            kornia.metrics.translation_ate(t, t_gt)


class TestPoseErrors(BaseTester):
    @staticmethod
    def _pose(t, device, dtype):
        eye = torch.eye(3, device=device, dtype=dtype)
        t = torch.tensor(t, device=device, dtype=dtype).reshape(3, 1)
        return torch.cat([eye, t], dim=1)

    def test_keys_and_values(self, device, dtype):
        # R=I both; t=[1,0,0] vs t_gt=[0,1,0] -> R_err=0, t_err=90, max=90.
        P = self._pose([1.0, 0.0, 0.0], device, dtype)
        P_gt = self._pose([0.0, 1.0, 0.0], device, dtype)
        out = kornia.metrics.pose_errors(P, P_gt)
        assert set(out) == {"R_err", "t_err", "max_err"}
        assert out["R_err"].shape == (1,)
        self.assert_close(out["R_err"], torch.tensor([0.0], device=device, dtype=dtype))
        self.assert_close(out["t_err"], torch.tensor([90.0], device=device, dtype=dtype))
        self.assert_close(out["max_err"], torch.tensor([90.0], device=device, dtype=dtype))

    def test_translation_sign_fold(self, device, dtype):
        # t=[1,0,0] vs t_gt=[-1,0,0]: 180 deg.
        P = self._pose([1.0, 0.0, 0.0], device, dtype)
        P_gt = self._pose([-1.0, 0.0, 0.0], device, dtype)
        # Folded (default) -> min(180, 0) = 0.
        out = kornia.metrics.pose_errors(P, P_gt)
        self.assert_close(out["t_err"], torch.tensor([0.0], device=device, dtype=dtype))
        # Unfolded -> the raw 180 deg.
        out = kornia.metrics.pose_errors(P, P_gt, fold_translation=False)
        self.assert_close(out["t_err"], torch.tensor([180.0], device=device, dtype=dtype))

    def test_exception(self, device, dtype):
        with pytest.raises(Exception):
            kornia.metrics.pose_errors(
                torch.eye(3, device=device, dtype=dtype), torch.eye(4, device=device, dtype=dtype)
            )


class TestAucFromErrors(BaseTester):
    def test_perfect_recall_is_100(self, device, dtype):
        # A single error of 0 sits below any threshold -> full area -> AUC 100.
        aucs = kornia.metrics.auc_from_errors(torch.zeros(1, device=device, dtype=dtype), thresholds=4.0)
        assert math.isclose(aucs[4.0], 100.0, abs_tol=1e-3)

    def test_known_value(self, device, dtype):
        # errors=[2], thr=4. Augmented errors=[0,2], recall=[0,1]; step up to thr=4:
        #   trapezoid(recall=[0,1,1], x=[0,2,4]) = 1 (0..2) + 2 (2..4) = 3; 3/4*100 = 75.
        errors = torch.tensor([2.0], device=device, dtype=dtype)
        aucs = kornia.metrics.auc_from_errors(errors, thresholds=(2.0, 4.0))
        assert set(aucs) == {2.0, 4.0}
        # thr=2 sits exactly at the error -> no area below -> 0.
        assert math.isclose(aucs[2.0], 0.0, abs_tol=1e-3)
        assert math.isclose(aucs[4.0], 75.0, abs_tol=1e-3)

    def test_integer_errors_do_not_truncate_threshold(self, device):
        # On an integer tensor, thr=2.5 used to truncate to 2 and score 0 instead of 60.
        int_errors = torch.tensor([2], device=device, dtype=torch.int64)
        float_errors = torch.tensor([2.0], device=device)
        assert math.isclose(
            kornia.metrics.auc_from_errors(int_errors, thresholds=2.5)[2.5],
            kornia.metrics.auc_from_errors(float_errors, thresholds=2.5)[2.5],
            abs_tol=1e-3,
        )

    def test_errors_above_threshold_and_empty(self, device, dtype):
        above = torch.tensor([50.0, 60.0], device=device, dtype=dtype)
        assert math.isclose(kornia.metrics.auc_from_errors(above, thresholds=10.0)[10.0], 0.0, abs_tol=1e-3)
        empty = torch.empty(0, device=device, dtype=dtype)
        assert math.isclose(kornia.metrics.auc_from_errors(empty, thresholds=5.0)[5.0], 0.0, abs_tol=1e-3)

    @pytest.mark.parametrize("thr", [0.0, -3.0])
    def test_non_positive_threshold_raises(self, device, dtype, thr):
        # These used to fall off the front of the curve and return nan or -0.0.
        errors = torch.tensor([1.0], device=device, dtype=dtype)
        with pytest.raises(Exception):
            kornia.metrics.auc_from_errors(errors, thresholds=thr)
