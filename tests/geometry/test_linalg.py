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
import kornia.geometry.linalg as kgl
from kornia.core._compat import torch_version_lt

from testing.base import BaseTester
from testing.geometry.create import create_random_homography
from testing.geometry.linalg import euler_angles_to_rotation_matrix, identity_matrix


def _rigid_transforms(batch_size, device, dtype):
    """Distinct, non-identity rigid transforms, so a swapped or dropped operand changes the result."""
    angles = torch.linspace(0.1, 0.9, batch_size, device=device, dtype=dtype)
    # the helper returns homogeneous (4, 4) rotations
    trans = euler_angles_to_rotation_matrix(angles, 2 * angles, -angles).reshape(batch_size, 4, 4).clone()
    trans[:, :3, 3] = torch.stack([angles, 1 - angles, 2 * angles], dim=-1)
    return trans


class TestTransformPoints(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    @pytest.mark.parametrize("num_dims", [2, 3])
    def test_transform_points(self, batch_size, num_points, num_dims, device, dtype):
        points_src = (
            torch.arange(batch_size * num_points * num_dims, device=device)
            .remainder(5)
            .to(dtype)
            .reshape(batch_size, num_points, num_dims)
            - 2
        ) / 2
        scale = torch.arange(2, num_dims + 2, device=device).to(dtype)
        translation = torch.arange(-1, num_dims - 1, device=device).to(dtype)
        dst_homo_src = torch.eye(num_dims + 1, device=device, dtype=dtype).expand(batch_size, -1, -1).clone()
        dst_homo_src[:, :num_dims, :num_dims] = torch.diag(scale)
        dst_homo_src[:, :num_dims, -1] = translation
        perspective = torch.arange(1, batch_size + 1, device=device).to(dtype) * 0.1
        dst_homo_src[:, -1, 0] = perspective
        denominator = 1 + points_src[..., 0] * perspective[:, None]
        expected = (points_src * scale + translation) / denominator.unsqueeze(-1)

        actual = kgl.transform_points(dst_homo_src, points_src)

        self.assert_close(actual, expected)

    @pytest.mark.parametrize("num_dims", [2, 3])
    def test_transform_points_empty(self, num_dims, device, dtype):
        # No points to transform (e.g. an image chip with no annotations) must return an empty
        # tensor of the same shape rather than crashing on the internal reshape.
        points = torch.zeros(2, 0, num_dims, device=device, dtype=dtype)
        trans = torch.eye(num_dims + 1, device=device, dtype=dtype).expand(2, -1, -1)
        out = kgl.transform_points(trans, points)
        assert out.shape == points.shape
        self.assert_close(out, points)

    @pytest.mark.parametrize("num_dims", [2, 3])
    @pytest.mark.parametrize("points_shape", [(0, 1), (0, 5)])
    def test_transform_points_empty_batch_with_points(self, num_dims, points_shape, device, dtype):
        # An empty transform batch with a non-empty point axis divided 0 // 0 while expanding the
        # transforms (kornia#4466); it must return the same empty shape as a B=1 transform does.
        points = torch.zeros(*points_shape, num_dims, device=device, dtype=dtype)
        empty_trans = torch.eye(num_dims + 1, device=device, dtype=dtype).expand(0, -1, -1)
        single_trans = torch.eye(num_dims + 1, device=device, dtype=dtype)[None]

        out = kgl.transform_points(empty_trans, points)

        assert out.shape == points.shape
        assert out.dtype == dtype
        assert out.shape == kgl.transform_points(single_trans, points).shape

    def test_gradcheck(self, device):
        # generate input data
        batch_size, num_points, num_dims = 2, 3, 2
        eye_size = num_dims + 1
        points_src = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        dst_homo_src = create_random_homography(points_src, eye_size)
        # evaluate function gradient
        self.gradcheck(kornia.geometry.transform_points, (dst_homo_src, points_src))

    def test_dynamo(self, device, dtype, torch_optimizer):
        points = torch.ones(1, 2, 2, device=device, dtype=dtype)
        transform = kornia.core.ops.eye_like(3, points)
        op = kornia.geometry.transform_points
        op_script = torch_optimizer(op)
        actual = op_script(transform, points)
        expected = op(transform, points)
        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("trans_dtype", [torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("points_dtype", [torch.float16, torch.float32, torch.float64])
    def test_mixed_dtypes(self, device, trans_dtype, points_dtype):
        # Regression test for https://github.com/kornia/kornia/issues/3705
        if device.type == "mps" and torch.float64 in (trans_dtype, points_dtype):
            pytest.skip("MPS does not support float64")
        if trans_dtype == torch.float16 and device.type == "cpu" and torch_version_lt(2, 2, 0):
            # transform_points harmonises at the bmm site in the transform's dtype, and CPU
            # float16 bmm ("bmm" not implemented for 'Half') only landed in PyTorch 2.2.
            pytest.skip("CPU float16 bmm requires PyTorch 2.2")
        points_src = torch.rand(2, 3, 2, device=device, dtype=points_dtype)
        trans = kornia.core.ops.eye_like(3, points_src).to(trans_dtype)
        out = kgl.transform_points(trans, points_src)
        assert out.dtype == points_dtype
        self.assert_close(out.to(torch.float32), points_src.to(torch.float32), atol=1e-2, rtol=1e-2)


class TestComposeTransforms(BaseTester):
    def test_smoke(self, device, dtype):
        batch_size = 2
        trans_01 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)

        to_check_1 = kornia.geometry.compose_transformations(trans_01, trans_12)
        to_check_2 = kornia.geometry.compose_transformations(trans_01[0], trans_12[0])

        assert to_check_1.shape == (batch_size, 4, 4)
        assert to_check_2.shape == (4, 4)

    def test_exception(self, device, dtype):
        to_check_1 = torch.rand((7, 4, 4, 3), device=device, dtype=dtype)
        to_check_2 = torch.rand((5, 10, 10), device=device, dtype=dtype)
        to_check_3 = torch.rand((6, 4, 4), device=device, dtype=dtype)
        to_check_4 = torch.rand((4, 4), device=device, dtype=dtype)
        to_check_5 = torch.rand((3, 3), device=device, dtype=dtype)

        # Testing if exception is thrown when both inputs have shape (3, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_5, to_check_5)

        # Testing if exception is thrown when both inputs have shape (5, 10, 10)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_2, to_check_2)

        # Testing if exception is thrown when one input has shape (6, 4, 4)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_3, to_check_4)

        # Testing if exception is thrown when one input has shape (7, 4, 4, 3)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.compose_transformations(to_check_1, to_check_4)

    def test_translation_4x4(self, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_12 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kgl.compose_transformations(trans_01, trans_12)
        self.assert_close(trans_02, trans_12, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_12[..., :3, -1] += offset  # add offset to translation vector

        trans_02 = kgl.compose_transformations(trans_01, trans_12)
        self.assert_close(trans_02, trans_12, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device):
        trans_01 = identity_matrix(batch_size, device=device, dtype=torch.float64)
        trans_12 = identity_matrix(batch_size, device=device, dtype=torch.float64)

        self.gradcheck(kgl.compose_transformations, (trans_01, trans_12))

    def test_broadcast(self, device, dtype):
        # Broadcasting batch size 1 against B in both directions (#4933)
        A = _rigid_transforms(1, device=device, dtype=dtype)
        B = _rigid_transforms(3, device=device, dtype=dtype)
        out_ab = kgl.compose_transformations(A, B)
        out_ba = kgl.compose_transformations(B, A)
        assert out_ab.shape == (3, 4, 4)
        assert out_ba.shape == (3, 4, 4)
        self.assert_close(out_ab, A @ B)
        self.assert_close(out_ba, B @ A)

    def test_mismatched_batch_exception(self, device, dtype):
        A = identity_matrix(batch_size=2, device=device, dtype=dtype)
        B = identity_matrix(batch_size=3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Incompatible batch shapes"):
            kgl.compose_transformations(A, B)


class TestInverseTransformation(BaseTester):
    def test_smoke(self, device, dtype):
        batch_size = 2
        trans_01 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)

        to_check_1 = kornia.geometry.inverse_transformation(trans_01)
        to_check_2 = kornia.geometry.inverse_transformation(trans_01[0])

        assert to_check_1.shape == (batch_size, 4, 4)
        assert to_check_2.shape == (4, 4)

    def test_exception(self, device, dtype):
        to_check_1 = torch.rand((7, 4, 4, 3), device=device, dtype=dtype)
        to_check_2 = torch.rand((5, 10, 10), device=device, dtype=dtype)
        to_check_3 = torch.rand((3, 3), device=device, dtype=dtype)

        # Testing if exception is thrown when the input has shape (7, 4, 4, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.inverse_transformation(to_check_1)

        # Testing if exception is thrown when the input has shape (5, 10, 10)
        with pytest.raises(ValueError):
            _ = kornia.geometry.inverse_transformation(to_check_2)

        # Testing if exception is thrown when the input has shape (3, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.inverse_transformation(to_check_3)

    def test_translation_4x4(self, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kgl.inverse_transformation(trans_01)
        trans_01_hat = kgl.inverse_transformation(trans_10)
        self.assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01[..., :3, -1] += offset  # add offset to translation vector

        trans_10 = kgl.inverse_transformation(trans_01)
        trans_01_hat = kgl.inverse_transformation(trans_10)
        self.assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10
        x, y, z = 0, 0, kornia.pi
        ones = torch.ones(batch_size, device=device, dtype=dtype)
        rmat_01 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_01[..., :3, -1] += offset  # add offset to translation vector
        trans_01[..., :3, :3] = rmat_01[..., :3, :3]

        trans_10 = kgl.inverse_transformation(trans_01)
        trans_01_hat = kgl.inverse_transformation(trans_10)
        self.assert_close(trans_01, trans_01_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device):
        trans_01 = identity_matrix(batch_size, device=device, dtype=torch.float64)
        self.gradcheck(kgl.inverse_transformation, (trans_01,))


class TestRelativeTransformation(BaseTester):
    def test_smoke(self, device, dtype):
        batch_size = 2
        trans_01 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)
        trans_02 = identity_matrix(batch_size=batch_size, device=device, dtype=dtype)

        to_check_1 = kornia.geometry.relative_transformation(trans_01, trans_02)
        to_check_2 = kornia.geometry.relative_transformation(trans_01[0], trans_02[0])

        assert to_check_1.shape == (batch_size, 4, 4)
        assert to_check_2.shape == (4, 4)

    def test_exception(self, device, dtype):
        to_check_1 = torch.rand((7, 4, 4, 3), device=device, dtype=dtype)
        to_check_2 = torch.rand((5, 10, 10), device=device, dtype=dtype)
        to_check_3 = torch.rand((6, 4, 4), device=device, dtype=dtype)
        to_check_4 = torch.rand((4, 4), device=device, dtype=dtype)
        to_check_5 = torch.rand((3, 3), device=device, dtype=dtype)

        # Testing if exception is thrown when both inputs have shape (3, 3)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_5, to_check_5)

        # Testing if exception is thrown when both inputs have shape (5, 10, 10)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_2, to_check_2)

        # Testing if exception is thrown when one input has shape (6, 4, 4)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_3, to_check_4)

        # Testing if exception is thrown when one input has shape (7, 4, 4, 3)
        # whereas the other input has shape (4, 4)
        with pytest.raises(ValueError):
            _ = kornia.geometry.relative_transformation(to_check_1, to_check_4)

    def test_translation_4x4(self, device, dtype):
        offset = 10.0
        trans_01 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_02 = identity_matrix(batch_size=1, device=device, dtype=dtype)[0]
        trans_02[..., :3, -1] += offset  # add offset to translation vector

        trans_12 = kgl.relative_transformation(trans_01, trans_02)
        trans_02_hat = kgl.compose_transformations(trans_01, trans_12)
        self.assert_close(trans_02_hat, trans_02, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_rotation_translation_Bx4x4(self, batch_size, device, dtype):
        offset = 10.0
        x, y, z = 0.0, 0.0, kornia.pi
        ones = torch.ones(batch_size, device=device, dtype=dtype)
        rmat_02 = euler_angles_to_rotation_matrix(x * ones, y * ones, z * ones)

        trans_01 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_02 = identity_matrix(batch_size, device=device, dtype=dtype)
        trans_02[..., :3, -1] += offset  # add offset to translation vector
        trans_02[..., :3, :3] = rmat_02[..., :3, :3]

        trans_12 = kgl.relative_transformation(trans_01, trans_02)
        trans_02_hat = kgl.compose_transformations(trans_01, trans_12)
        self.assert_close(trans_02_hat, trans_02, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_gradcheck(self, batch_size, device):
        trans_01 = identity_matrix(batch_size, device=device, dtype=torch.float64)
        trans_02 = identity_matrix(batch_size, device=device, dtype=torch.float64)

        self.gradcheck(kgl.relative_transformation, (trans_01, trans_02))

    def test_broadcast(self, device, dtype):
        # Broadcasting batch size 1 against B in both directions (#4933)
        A = _rigid_transforms(1, device=device, dtype=dtype)
        B = _rigid_transforms(3, device=device, dtype=dtype)
        out_ab = kgl.relative_transformation(A, B)
        out_ba = kgl.relative_transformation(B, A)
        assert out_ab.shape == (3, 4, 4)
        assert out_ba.shape == (3, 4, 4)
        self.assert_close(out_ab, kgl.inverse_transformation(A) @ B)
        self.assert_close(out_ba, kgl.inverse_transformation(B) @ A)

    def test_mismatched_batch_exception(self, device, dtype):
        A = identity_matrix(batch_size=2, device=device, dtype=dtype)
        B = identity_matrix(batch_size=3, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Incompatible batch shapes"):
            kgl.relative_transformation(A, B)


class TestPointsLinesDistances(BaseTester):
    def test_smoke(self, device, dtype):
        pts = torch.rand(1, 1, 2, device=device, dtype=dtype)
        lines = torch.rand(1, 1, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (1, 1)

        # homogeneous
        pts = torch.rand(1, 1, 3, device=device, dtype=dtype)
        lines = torch.rand(1, 1, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (1, 1)

    @pytest.mark.parametrize(
        "batch_size, sample_size", [(1, 1), (2, 1), (4, 1), (7, 1), (1, 3), (2, 3), (4, 3), (7, 3)]
    )
    def test_shape(self, batch_size, sample_size, device, dtype):
        B, N = batch_size, sample_size
        pts = torch.rand(B, N, 2, device=device, dtype=dtype)
        lines = torch.rand(B, N, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (B, N)

    @pytest.mark.parametrize(
        "batch_size, extra_dim_size", [(1, 1), (2, 1), (4, 1), (7, 1), (1, 3), (2, 3), (4, 3), (7, 3)]
    )
    def test_shapes(self, batch_size, extra_dim_size, device, dtype):
        B, T, N = batch_size, extra_dim_size, 3
        pts = torch.rand(B, T, N, 2, device=device, dtype=dtype)
        lines = torch.rand(B, T, N, 3, device=device, dtype=dtype)
        distances = kgl.point_line_distance(pts, lines)
        assert distances.shape == (B, T, N)

    def test_functional(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        pts = torch.tensor([1.0, 0], device=device, dtype=torch.float64).view(1, 1, 2).tile(1, 6, 1)
        lines = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=torch.float64,
        ).view(1, 6, 3)
        distances = kgl.point_line_distance(pts, lines)
        distances_expected = torch.tensor(
            [
                0.0,
                1.0,
                1.0,
                2.0,
                torch.sqrt(torch.tensor(2, dtype=torch.float64)) / 2,
                torch.sqrt(torch.tensor(2, dtype=torch.float64)),
            ],
            device=device,
        ).view(1, 6)
        self.assert_close(distances, distances_expected, rtol=1e-6, atol=1e-6)

    def test_gradcheck(self, device):
        pts = torch.rand(2, 3, 2, device=device, requires_grad=True, dtype=torch.float64)
        lines = torch.rand(2, 3, 3, device=device, requires_grad=True, dtype=torch.float64)
        self.gradcheck(kgl.point_line_distance, (pts, lines))
        # homogeneous points with weights away from 0 and 1
        pts = (torch.rand(2, 3, 3, device=device, dtype=torch.float64) + 0.5).requires_grad_(True)
        self.gradcheck(kgl.point_line_distance, (pts, lines))

    def test_homogeneous_weight_4935(self, device, dtype):
        # #4935: the third coordinate of a (*, N, 3) point was ignored, so the point (1.5, -0.7) was 2.68 from
        # 3x + 4y + 10 = 0 when written as (3, -1.4, 2) and 1.66 as (-1.5, 0.7, -1), instead of 2.34. The homogeneous
        # distance is |ax + by + cw| / (|w| |(a, b)|), and a point at infinity (w = 0) is at distance inf.
        line = torch.tensor([[[3.0, 4.0, 10.0]]], device=device, dtype=dtype)
        point = torch.tensor([[[1.5, -0.7]]], device=device, dtype=dtype)
        expected = torch.tensor([[2.34]], device=device, dtype=dtype)
        self.assert_close(kgl.point_line_distance(point, line), expected)
        for w in (1.0, 2.0, -1.0, 0.25):
            homogeneous = torch.cat((w * point, torch.full_like(point[..., :1], w)), dim=-1)
            self.assert_close(kgl.point_line_distance(homogeneous, line), expected)
        at_infinity = torch.tensor([[[1.5, -0.7, 0.0]]], device=device, dtype=dtype, requires_grad=True)
        distance = kgl.point_line_distance(at_infinity, line)
        assert bool(distance.isposinf().all()), distance
        distance.sum().backward()
        assert bool(torch.isfinite(at_infinity.grad).all()), at_infinity.grad


class TestEuclideanDistance(BaseTester):
    def test_smoke(self, device, dtype):
        pt1 = torch.tensor([0, 0, 0], device=device, dtype=dtype)
        pt2 = torch.tensor([1, 0, 0], device=device, dtype=dtype)
        dst = kgl.euclidean_distance(pt1, pt2)
        self.assert_close(dst, torch.tensor(1.0, device=device, dtype=dtype))

    def test_coincident_points_are_zero(self, device, dtype):
        # Exactly 0 with a zero gradient, as torch.linalg.norm gives. Compared exactly: the former sqrt(eps)
        # floor of 1e-3 is inside the default float16 and bfloat16 tolerances.
        pt1 = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
        pt2 = torch.zeros(3, device=device, dtype=dtype)
        dst = kgl.euclidean_distance(pt1, pt2)
        assert dst.item() == 0.0
        (grad,) = torch.autograd.grad(dst.sum(), pt1)
        assert (grad == 0).all(), grad

    @pytest.mark.parametrize("dist", [1e-4, 1e-3, 1e-2, 1.0])
    def test_distance_is_exact(self, device, dtype, dist):
        if dist**2 < torch.finfo(dtype).tiny:
            pytest.skip(f"the squared distance {dist**2:g} is subnormal in {dtype}")
        pt1 = torch.zeros(3, device=device, dtype=dtype)
        pt2 = torch.tensor([dist, 0.0, 0.0], device=device, dtype=dtype)
        # relative tolerance only: the default half tolerances would absorb the former bias of up to 1e-3
        rtol = 2 * torch.finfo(dtype).eps
        self.assert_close(kgl.euclidean_distance(pt1, pt2), pt2[0], rtol=rtol, atol=0.0)

    @pytest.mark.parametrize("shape", [(2,), (3,), (1, 2), (2, 3)])
    def test_cardinality(self, device, dtype, shape):
        pt1 = torch.rand(shape, device=device, dtype=dtype)
        pt2 = torch.rand(shape, device=device, dtype=dtype)
        dst = kgl.euclidean_distance(pt1, pt2)
        assert len(dst.shape) == len(shape) - 1

    def test_exception(self, device, dtype):
        pt1 = torch.tensor([0, 0, 0], device=device, dtype=dtype)
        pt2 = torch.rand(1, 2, device=device, dtype=dtype)
        with pytest.raises(Exception):
            kgl.euclidean_distance(pt1, pt2)

    def test_gradcheck(self, device):
        pt1 = torch.rand(2, 3, device=device, dtype=torch.float64, requires_grad=True)
        pt2 = torch.rand(2, 3, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(kgl.euclidean_distance, (pt1, pt2))

    def test_dynamo(self, device, dtype, torch_optimizer):
        pt1 = torch.rand(2, 3, device=device, dtype=dtype)
        pt2 = torch.rand(2, 3, device=device, dtype=dtype)
        op = kgl.euclidean_distance
        op_optimized = torch_optimizer(op)
        self.assert_close(op(pt1, pt2), op_optimized(pt1, pt2))

    def test_module(self, device, dtype):
        pass


# Two rigid, non-commuting transforms (T01 T02 != T02 T01 by 0.854), neither the identity nor a pure translation.
# Generated in float64:
#   from kornia.geometry.liegroup import Se3
#   T01 = Se3.exp(torch.tensor([1.0, -2.0, 3.0, 0.4, 0.2, -0.3], dtype=torch.float64)).matrix()
#   T02 = Se3.exp(torch.tensor([0.3, 0.1, -0.2, -0.1, 0.3, 0.05], dtype=torch.float64)).matrix()
_T01 = [
    [0.9365557269934556, 0.32475143364814213, 0.13190859175670216, 0.8932266973410873],
    [-0.24666617456316434, 0.8779917826797222, -0.4102270442977377, -2.6663426561315666],
    [-0.24903648038416887, 0.35166309998400436, 0.9023934261437778, 2.4134071590337385],
    [0.0, 0.0, 0.0, 1.0],
]
_T02 = [
    [0.9541437047897824, -0.06402251222932676, 0.2924224829555252, 0.26284367361699246],
    [0.034277888309185565, 0.9938032033499706, 0.10573655651854763, 0.09532423813367488],
    [-0.2973799202755487, -0.09086424455847703, 0.9504256267997647, -0.24625808156806428],
    [0.0, 0.0, 0.0, 1.0],
]


class TestLinalgConventions(BaseTester):
    @staticmethod
    def _fixture(device, dtype):
        t01 = torch.tensor(_T01, device=device, dtype=dtype)
        t02 = torch.tensor(_T02, device=device, dtype=dtype)
        # precondition: the pair does not commute, so an operand swap changes every result below, and neither
        # transform is a pure translation
        assert (t01 @ t02 - t02 @ t01).abs().max() > 0.5
        eye = torch.eye(3, device=device, dtype=dtype)
        assert (t01[:3, :3] - eye).abs().max() > 0.1 and (t02[:3, :3] - eye).abs().max() > 0.1
        return t01, t02

    def test_convention_relative_transformation_is_inverse_first_times_second(self, device, dtype):
        t01, t02 = self._fixture(device, dtype)
        # trans_12 = trans_01^-1 @ trans_02, generated in float64 as torch.linalg.inv(T01) @ T02
        expected = torch.tensor(
            [
                [0.9592120041966407, -0.2824697732662307, 0.01109766624060022, -0.6092449687748074],
                [0.23537769566281422, 0.8198060415906909, 0.5220300705475368, 1.2846969255319278],
                [-0.15655564949731832, -0.49812536711386907, 0.8528548805325243, -3.6161278131582963],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        trans_12 = kgl.relative_transformation(t01, t02)
        self.assert_close(trans_12, expected)
        # and composing it back onto trans_01 recovers trans_02
        self.assert_close(kgl.compose_transformations(t01, trans_12), t02)

    def test_convention_compose_transformations_order(self, device, dtype):
        t01, t02 = self._fixture(device, dtype)
        # compose(a, b) = a @ b, generated in float64 as T01 @ T02
        expected = torch.tensor(
            [
                [0.8655135779661945, 0.25079259002638926, 0.4335773554326974, 1.1378675714180575],
                [-0.08326598765280734, 0.9256182047971615, -0.3691852031816589, -2.5464616769031294],
                [-0.4939160066816611, 0.28343255941393825, 0.8220176169691901, 2.1592498388088193],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(kgl.compose_transformations(t01, t02), expected)

    def test_convention_transform_points_maps_frame1_to_frame0(self, device, dtype):
        t01, _ = self._fixture(device, dtype)
        points_1 = torch.tensor([[[1.0, 2.0, 3.0]]], device=device, dtype=dtype)
        # points_0 = R01 p1 + t01, generated in float64 as (T01 @ [1, 2, 3, 1])[:3]; the inverse direction,
        # inv(T01) @ p1, is [-1.197, 4.338, -1.371]
        expected = torch.tensor(
            [[[2.8750110669009334, -2.3877063982284996, 5.5748771570489115]]], device=device, dtype=dtype
        )
        self.assert_close(kgl.transform_points(t01[None], points_1), expected)

    def test_convention_transform_points_projective_division(self, device, dtype):
        # A homography with a non-trivial last row: w = 0.1 * 2 + 2 = 2.2, and the result is (x, y) / w.
        trans = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 2.0]]], device=device, dtype=dtype)
        points = torch.tensor([[[2.0, 3.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[2.0 / 2.2, 3.0 / 2.2]]], device=device, dtype=dtype)
        self.assert_close(kgl.transform_points(trans, points), expected)

    def test_convention_inverse_transformation_assumes_rigid(self, device, dtype):
        t01, _ = self._fixture(device, dtype)
        # A non-rigid matrix: the rotation block of T01 scaled by 2.
        m = t01.clone()
        m[:3, :3] = 2.0 * m[:3, :3]
        eye3 = torch.eye(3, device=device, dtype=dtype)
        # precondition: m is not rigid (M^T M = 4 I)
        assert (m[:3, :3].T @ m[:3, :3] - eye3).abs().max() > 1.0
        inv = kgl.inverse_transformation(m)
        # The rigid formula [M^T, -M^T t; 0, 1] is applied without validation: the rotation block is the transpose
        # to the bit, and the result is not the inverse of m (inv @ m = [4 I, ...], residual 3).
        assert torch.equal(inv[:3, :3], m[:3, :3].T)
        # generated in float64 as -(2 R01)^T t01
        expected_t = torch.tensor(
            [-1.7864533946821743, 2.4044880965764657, -6.778945795191922], device=device, dtype=dtype
        )
        self.assert_close(inv[:3, 3], expected_t)
        assert (inv @ m - torch.eye(4, device=device, dtype=dtype)).abs().max() > 1.0

    def test_convention_transform_helpers_read_the_top_three_rows(self, device, dtype):
        t01, t02 = self._fixture(device, dtype)
        # compose, relative and inverse read only the top three rows of each input and write [0, 0, 0, 1] as the
        # last row of the result, so a projective last row is used as if it were [0, 0, 0, 1].
        row = torch.tensor([0.1, -0.2, 0.05, 1.3], device=device, dtype=dtype)
        p02, p01 = t02.clone(), t01.clone()
        p02[3], p01[3] = row, row
        # precondition: the projective row changes the full product
        assert (t01 @ p02 - t01 @ t02).abs().max() > 0.5
        self.assert_close(kgl.compose_transformations(t01, p02), t01 @ t02)
        self.assert_close(kgl.relative_transformation(t01, p02), kgl.inverse_transformation(t01) @ t02)
        self.assert_close(kgl.inverse_transformation(p01), kgl.inverse_transformation(t01))
        # The first argument of relative_transformation is inverted by transposition, as in inverse_transformation.
        m = t01.clone()
        m[:3, :3] = 2.0 * m[:3, :3]
        self.assert_close(kgl.relative_transformation(m, t02), kgl.inverse_transformation(m) @ t02)

    def test_convention_relative_transformation_vs_relative_camera_motion(self, device, dtype):
        # Read as world-to-camera extrinsics E1 = T01, E2 = T02, relative_camera_motion returns E2 E1^-1, which is
        # relative_transformation(E2^-1, E1^-1) and not relative_transformation(E1, E2).
        e1, e2 = self._fixture(device, dtype)
        # generated in float64 as (T02 @ torch.linalg.inv(T01))[:3]
        expected = torch.tensor(
            [
                [0.9113903863880556, -0.4115258281569324, 0.0037491811348190907, -1.6575517215108124],
                [0.36878972792479336, 0.8207198555059045, 0.4363634441099808, 0.9011091069599761],
                [-0.18265185511400522, -0.39631478844236734, 0.8997626844258959, -3.311313297965872],
            ],
            device=device,
            dtype=dtype,
        )
        rot, t = kornia.geometry.epipolar.relative_camera_motion(
            e1[None, :3, :3], e1[None, :3, 3:], e2[None, :3, :3], e2[None, :3, 3:]
        )
        self.assert_close(torch.cat([rot, t], -1)[0], expected)
        inv_e1, inv_e2 = kgl.inverse_transformation(e1), kgl.inverse_transformation(e2)
        self.assert_close(kgl.relative_transformation(inv_e2, inv_e1)[:3], expected)
        # control: the same-order call is off by more than 1 on this pair
        assert (kgl.relative_transformation(e1, e2)[:3] - expected).abs().max() > 0.5

    def test_wart_point_line_distance_eps_bias_4881(self, device, dtype):
        # https://github.com/kornia/kornia/issues/4881: eps is added to the line norm,
        # |ax + by + c| / (||(a, b)|| + eps), so the distance depends on the scale of the line. The point (1.5, -0.7)
        # is 2.34 from 3x + 4y + 10 = 0 at every scale (11.7 / 5); at scale 1e-9 the norm is 5e-9 and the result is
        # 2.34 * 5 / 6 = 1.95 (-16.7 %).
        point = torch.tensor([[1.5, -0.7]], device=device, dtype=dtype)
        line = torch.tensor([[3.0, 4.0, 10.0]], device=device, dtype=dtype)
        exact = torch.tensor([2.34], device=device, dtype=dtype)
        self.assert_close(kgl.point_line_distance(point, line), exact)
        degenerate = kgl.point_line_distance(point, torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype))
        if dtype == torch.float16:
            # The default eps=1e-9 rounds to zero in float16 (and a 1e-9-scaled line underflows), so the degenerate
            # line gives inf.
            assert torch.isinf(degenerate).all()
            return
        scaled = kgl.point_line_distance(point, 1e-9 * line)
        assert ((scaled - exact) / exact).max() < -0.1
        # A degenerate line (0, 0, 1) returns |c| / eps = 1e9 instead of flagging the singular case.
        assert torch.isfinite(degenerate).all() and (degenerate > 1e8).all()

    def test_wart_point_line_distance_ignores_w_4935(self, device, dtype):
        # https://github.com/kornia/kornia/issues/4935: the docstring says "possibly homogeneous" (*, N, 3) points, but
        # the third coordinate is never read. The point (1.5, -0.7) is 2.34 from 3x + 4y + 10 = 0; written with w = 1
        # it gives 2.34, with w = 2 as (3, -1.4, 2) it gives |9 - 5.6 + 10| / 5 = 2.68, and with w = -1 as
        # (-1.5, 0.7, -1) it gives |-4.5 + 2.8 + 10| / 5 = 1.66.
        line = torch.tensor([[3.0, 4.0, 10.0]], device=device, dtype=dtype).expand(3, 3)
        points = torch.tensor([[1.5, -0.7, 1.0], [3.0, -1.4, 2.0], [-1.5, 0.7, -1.0]], device=device, dtype=dtype)
        # precondition: the three rows are the same Euclidean point
        self.assert_close(points[:, :2] / points[:, 2:], points[:1, :2].expand(3, 2))
        expected = torch.tensor([2.34, 2.68, 1.66], device=device, dtype=dtype)
        self.assert_close(kgl.point_line_distance(points, line), expected)
