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

    def test_krt_from_projection(self, device, dtype):
        P = torch.tensor([[
            [10., 0., 30., 100.],
            [0., 20., 40., 160.],
            [0., 0., 1., 3.],
        ]], device=device, dtype=dtype)

        K_expected = torch.tensor([[
            [10., 0., 30.],
            [0., 20., 40.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        R_expected = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]], device=device, dtype=dtype)

        t_expected = torch.tensor([
            [[1.], [2.], [3.]],
        ], device=device, dtype=dtype)

        K_estimated, R_estimated, t_estimated = epi.KRt_from_projection(P)
        assert_allclose(K_expected, K_estimated)
        assert_allclose(R_expected, R_estimated)
        assert_allclose(t_expected, t_estimated)



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


class TestKRtFromProjection:
    def test_smoke(self, device, dtype):
        P = torch.randn(1, 3, 4, device=device, dtype=dtype)
        K, R, t = epi.KRt_from_projection(P)
        assert K.shape == (1, 3, 3)
        assert R.shape == (1, 3, 3)
        assert t.shape == (1, 3, 1)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        P = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K, R, t = epi.KRt_from_projection(P)
        assert K.shape == (B, 3, 3)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3, 1)

    def test_simple(self, device, dtype):
        P = torch.tensor([[
            [232., 71., 267., 352.]
            [280., 150., 140., 198.]
            [158., 468., 237., 371.]
        ]], device=device, dtype=dtype)

        K_expected = torch.tensor([[
            [154.8365, 217.0313, 243.0576],
            [0.0, 218.8586, 269.4455],
            [0.0, 0.0, 547.8659]
        ]], device=device, dtype=dtype)

        R_expected = torch.tensor([[
            [-0.2499, -0.3690, 0.8952],
            [0.9243, -0.3663, 0.1071],
            [0.2884, 0.8542, 0.4326]
        ]], device=device, dtype=dtype)

        t_expected = torch.tensor([
            [[0.0167], [-0.1426], [-1.2950]],
        ], device=device, dtype=dtype)

        K_estimated, R_estimated, t_estimated = epi.KRt_from_projection(P)
        assert_allclose(K_estimated, K_expected)
        assert_allclose(R_estimated, R_expected)
        assert_allclose(t_estimated, t_expected)

    def test_projection_from_krt(self, device, dtype):
        K = torch.tensor([[
            [154.8365, 217.0313, 243.0576],
            [0.0, 218.8586, 269.4455],
            [0.0, 0.0, 547.8659]
        ]], device=device, dtype=dtype)

        R = torch.tensor([[
            [-0.2499, -0.3690, 0.8952],
            [0.9243, -0.3663, 0.1071],
            [0.2884, 0.8542, 0.4326]
        ]], device=device, dtype=dtype)

        t = torch.tensor([
            [[0.0167], [-0.1426], [-1.2950]],
        ], device=device, dtype=dtype)

        P_expected = torch.tensor([[
            [232., 71., 267., 352.]
            [280., 150., 140., 198.]
            [158., 468., 237., 371.]
        ]], device=device, dtype=dtype)

        P_estimated = epi.projection_from_KRt(K, R, t)
        assert_allclose(P_expected, P_expected)

    def test_gradcheck(self, device):
        P_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(epi.KRt_from_projection,
                         (P_mat,), raise_exception=True)
