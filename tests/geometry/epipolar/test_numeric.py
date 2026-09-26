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
import kornia.geometry.epipolar as epi

from testing.base import BaseTester
from testing.two_view import two_view_scene


class TestSkewSymmetric(BaseTester):
    def test_smoke(self, device, dtype):
        vec = torch.rand(1, 3, device=device, dtype=dtype)
        cross_product_matrix = epi.cross_product_matrix(vec)
        assert cross_product_matrix.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B = batch_size
        vec = torch.rand(B, 3, device=device, dtype=dtype)
        cross_product_matrix = epi.cross_product_matrix(vec)
        assert cross_product_matrix.shape == (B, 3, 3)

    @pytest.mark.parametrize("shapes", [(1, 1), (1, 5), (2, 1), (2, 5), (4, 1), (4, 5)])
    def test_shapes(self, device, dtype, shapes):
        input_shape = (*shapes, 3)
        output_shape = (*shapes, 3, 3)
        t = torch.rand(*input_shape, device=device, dtype=dtype)
        cross_product_matrix = epi.cross_product_matrix(t)
        assert cross_product_matrix.shape == output_shape

    @pytest.mark.parametrize("shapes", [(1, 1), (1, 5), (2, 1), (2, 5), (4, 1), (4, 5)])
    def test_funcional_shapes(self, device, dtype, shapes):
        input_shape = (*shapes, 3)
        t = torch.rand(*input_shape, device=device, dtype=dtype)

        # Feed batches
        cross_product_matrices = []
        for i in range(t.shape[1]):
            cross_product_matrices.append(epi.cross_product_matrix(t[:, i, ...]))
        cross_product_matrix_parts = torch.stack(cross_product_matrices, dim=1)

        # Feed one-shot
        cross_product_matrix_whole = epi.cross_product_matrix(t)

        self.assert_close(cross_product_matrix_parts, cross_product_matrix_whole)

    def test_mean_std(self, device, dtype):
        vec = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        cross_product_matrix = epi.cross_product_matrix(vec)
        self.assert_close(cross_product_matrix[..., 0, 1], -cross_product_matrix[..., 1, 0])
        self.assert_close(cross_product_matrix[..., 0, 2], -cross_product_matrix[..., 2, 0])
        self.assert_close(cross_product_matrix[..., 1, 2], -cross_product_matrix[..., 2, 1])

    def test_gradcheck(self, device):
        vec = torch.ones(2, 3, device=device, requires_grad=True, dtype=torch.float64)
        assert self.gradcheck(epi.cross_product_matrix, (vec,), raise_exception=True, fast_mode=True)


class TestEyeLike:
    def test_smoke(self, device, dtype):
        image = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        identity = kornia.core.ops.eye_like(3, image)
        assert identity.shape == (1, 3, 3)
        assert identity.device == image.device
        assert identity.dtype == image.dtype

    @pytest.mark.parametrize("batch_size, eye_size", [(1, 2), (2, 3), (3, 3), (2, 4)])
    def test_shape(self, batch_size, eye_size, device, dtype):
        B, N = batch_size, eye_size
        image = torch.rand(B, 3, 4, 4, device=device, dtype=dtype)
        identity = kornia.core.ops.eye_like(N, image)
        assert identity.shape == (B, N, N)
        assert identity.device == image.device
        assert identity.dtype == image.dtype


class TestVecLike:
    def test_smoke(self, device, dtype):
        image = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        vec = kornia.core.ops.vec_like(3, image)
        assert vec.shape == (1, 3, 1)
        assert vec.device == image.device
        assert vec.dtype == image.dtype

    @pytest.mark.parametrize("batch_size, eye_size", [(1, 2), (2, 3), (3, 3), (2, 4)])
    def test_shape(self, batch_size, eye_size, device, dtype):
        B, N = batch_size, eye_size
        image = torch.rand(B, 3, 4, 4, device=device, dtype=dtype)
        vec = kornia.core.ops.vec_like(N, image)
        assert vec.shape == (B, N, 1)
        assert vec.device == image.device
        assert vec.dtype == image.dtype


class TestConventionCrossProductMatrix(BaseTester):
    def test_convention_cross_product_matrix_equals_vector_to_skew_symmetric_matrix(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        v = two_view["X"][0]  # (12, 3), no two entries equal
        M = epi.cross_product_matrix(v)
        # The same matrix as kornia.geometry.conversions.vector_to_skew_symmetric_matrix, to the bit.
        assert torch.equal(M, kornia.geometry.conversions.vector_to_skew_symmetric_matrix(v))
        # Skew-symmetric, with [v]x w = v x w (the transposed matrix would give w x v).
        assert torch.equal(M.transpose(-2, -1), -M)
        w = two_view["t"][0, :, 0].expand_as(v)
        self.assert_close((M @ w[..., None])[..., 0], torch.linalg.cross(v, w, dim=-1), low_tolerance=True)
        # Any number of leading dims.
        assert epi.cross_product_matrix(two_view["X"].reshape(2, 2, 3, 3)).shape == (2, 2, 3, 3, 3)
