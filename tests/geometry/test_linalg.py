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

from testing.base import BaseTester
from testing.geometry.create import create_random_homography
from testing.geometry.linalg import euler_angles_to_rotation_matrix, identity_matrix


class TestTransformPoints(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("num_points", [2, 3, 5])
    @pytest.mark.parametrize("num_dims", [2, 3])
    def test_transform_points(self, batch_size, num_points, num_dims, device, dtype):
        # generate input data
        eye_size = num_dims + 1
        points_src = torch.rand(batch_size, num_points, num_dims, device=device, dtype=dtype)

        dst_homo_src = create_random_homography(points_src, eye_size)
        dst_homo_src = dst_homo_src.to(device)

        # transform the points from dst to ref
        points_dst = kgl.transform_points(dst_homo_src, points_src)

        # transform the points from ref to dst
        src_homo_dst = torch.inverse(dst_homo_src)
        points_dst_to_src = kgl.transform_points(src_homo_dst, points_dst)

        # projected should be equal as initial
        self.assert_close(points_src, points_dst_to_src, atol=1e-4, rtol=1e-4)

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


class TestEuclideanDistance(BaseTester):
    def test_smoke(self, device, dtype):
        pt1 = torch.tensor([0, 0, 0], device=device, dtype=dtype)
        pt2 = torch.tensor([1, 0, 0], device=device, dtype=dtype)
        dst = kgl.euclidean_distance(pt1, pt2)
        self.assert_close(dst, torch.tensor(1.0, device=device, dtype=dtype))

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
