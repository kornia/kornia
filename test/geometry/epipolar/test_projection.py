import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.geometry.epipolar as epi


class TestIntrinsicsLike:
    def test_smoke(self, device, dtype):
        image = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        focal = torch.rand(1, device=device, dtype=dtype)
        camera_matrix = epi.intrinsics_like(focal, image)
        assert camera_matrix.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 9])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        focal: float = 100.0
        image = torch.rand(B, 3, 4, 4, device=device, dtype=dtype)
        camera_matrix = epi.intrinsics_like(focal, image)
        assert camera_matrix.shape == (B, 3, 3)
        assert camera_matrix.device == image.device
        assert camera_matrix.dtype == image.dtype


class TestScaleIntrinsics:
    def test_smoke_float(self, device, dtype):
        scale_factor: float = 1.0
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        camera_matrix_scale = epi.scale_intrinsics(camera_matrix, scale_factor)
        assert camera_matrix_scale.shape == (1, 3, 3)

    def test_smoke_tensor(self, device, dtype):
        scale_factor = torch.tensor(1.0)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        camera_matrix_scale = epi.scale_intrinsics(camera_matrix, scale_factor)
        assert camera_matrix_scale.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 9])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        scale_factor = torch.rand(B, device=device, dtype=dtype)
        camera_matrix = torch.rand(B, 3, 3, device=device, dtype=dtype)
        camera_matrix_scale = epi.scale_intrinsics(camera_matrix, scale_factor)
        assert camera_matrix_scale.shape == (B, 3, 3)

    def test_scale_double(self, device, dtype):
        scale_factor = torch.tensor(0.5)
        camera_matrix = torch.tensor([[
            [100., 0., 50.],
            [0., 100., 50.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        camera_matrix_expected = torch.tensor([[
            [50., 0., 25.],
            [0., 50., 25.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        camera_matrix_scale = epi.scale_intrinsics(camera_matrix, scale_factor)
        assert_allclose(camera_matrix_scale, camera_matrix_expected)

    def test_gradcheck(self, device):
        scale_factor = torch.ones(1, device=device, dtype=torch.float64, requires_grad=True)
        camera_matrix = torch.ones(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.scale_intrinsics,
                         (camera_matrix, scale_factor,), raise_exception=True)


class TestProjectionFromKRt:
    def test_smoke(self, device, dtype):
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        R = torch.rand(1, 3, 3, device=device, dtype=dtype)
        t = torch.rand(1, 3, 1, device=device, dtype=dtype)
        P = epi.projection_from_KRt(K, R, t)
        assert P.shape == (1, 3, 4)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        R = torch.rand(B, 3, 3, device=device, dtype=dtype)
        t = torch.rand(B, 3, 1, device=device, dtype=dtype)
        P = epi.projection_from_KRt(K, R, t)
        assert P.shape == (B, 3, 4)

    def test_simple(self, device, dtype):
        K = torch.tensor([[
            [10., 0., 30.],
            [0., 20., 40.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        R = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        t = torch.tensor([
            [[1.], [2.], [3.]],
        ], device=device, dtype=dtype)

        P_expected = torch.tensor([[
            [10., 0., 30., 100.],
            [0., 20., 40., 160.],
            [0., 0., 1., 3.],
        ]], device=device, dtype=dtype)

        P_estimated = epi.projection_from_KRt(K, R, t)
        assert_allclose(P_estimated, P_expected)

    def test_gradcheck(self, device):
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.projection_from_KRt,
                         (K, R, t,), raise_exception=True)


class TestProjectionsFromFundamental:
    def test_smoke(self, device, dtype):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        P = epi.projections_from_fundamental(F_mat)
        assert P.shape == (1, 3, 4, 2)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        F_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        P = epi.projections_from_fundamental(F_mat)
        assert P.shape == (B, 3, 4, 2)

    def test_gradcheck(self, device):
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(epi.projections_from_fundamental,
                         (F_mat,), raise_exception=True)
