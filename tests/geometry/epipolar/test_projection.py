import pytest
import torch
from torch.autograd import gradcheck

import kornia.geometry.epipolar as epi
from kornia.testing import assert_close


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
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        camera_matrix_expected = torch.tensor(
            [[[50.0, 0.0, 25.0], [0.0, 50.0, 25.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        camera_matrix_scale = epi.scale_intrinsics(camera_matrix, scale_factor)
        assert_close(camera_matrix_scale, camera_matrix_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        scale_factor = torch.ones(1, device=device, dtype=torch.float64, requires_grad=True)
        camera_matrix = torch.ones(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.scale_intrinsics, (camera_matrix, scale_factor), raise_exception=True, fast_mode=True)


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
        K = torch.tensor([[[10.0, 0.0, 30.0], [0.0, 20.0, 40.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        R = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        t = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        P_expected = torch.tensor(
            [[[10.0, 0.0, 30.0, 100.0], [0.0, 20.0, 40.0, 160.0], [0.0, 0.0, 1.0, 3.0]]], device=device, dtype=dtype
        )

        P_estimated = epi.projection_from_KRt(K, R, t)
        assert_close(P_estimated, P_expected, atol=1e-4, rtol=1e-4)

    def test_krt_from_projection(self, device, dtype):
        P = torch.tensor(
            [[[10.0, 0.0, 30.0, 100.0], [0.0, 20.0, 40.0, 160.0], [0.0, 0.0, 1.0, 3.0]]], device=device, dtype=dtype
        )

        K_expected = torch.tensor([[[10.0, 0.0, 30.0], [0.0, 20.0, 40.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        R_expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        t_expected = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        K_estimated, R_estimated, t_estimated = epi.KRt_from_projection(P)
        assert_close(K_estimated, K_expected, atol=1e-4, rtol=1e-4)
        assert_close(R_estimated, R_expected, atol=1e-4, rtol=1e-4)
        assert_close(t_estimated, t_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.projection_from_KRt, (K, R, t), raise_exception=True, fast_mode=True)


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
        assert gradcheck(epi.projections_from_fundamental, (F_mat,), raise_exception=True, fast_mode=True)


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
        P = torch.rand(B, 3, 4, device=device, dtype=dtype)
        K, R, t = epi.KRt_from_projection(P)

        assert K.shape == (B, 3, 3)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3, 1)

    def test_simple(self, device, dtype):
        P = torch.tensor(
            [[[308.0, 139.0, 231.0, 84.0], [481.0, 161.0, 358.0, 341.0], [384.0, 387.0, 459.0, 102.0]]],
            device=device,
            dtype=dtype,
        )

        K_expected = torch.tensor(
            [[[17.006138, 122.441254, 390.211426], [0.0, 228.743622, 577.167480], [0.0, 0.0, 712.675232]]],
            device=device,
            dtype=dtype,
        )

        R_expected = torch.tensor(
            [[[0.396559, 0.511023, -0.762625], [0.743249, -0.666318, -0.060006], [0.538815, 0.543024, 0.644052]]],
            device=device,
            dtype=dtype,
        )

        t_expected = torch.tensor([[[-6.477699], [1.129624], [0.143123]]], device=device, dtype=dtype)

        K_estimated, R_estimated, t_estimated = epi.KRt_from_projection(P)
        assert_close(K_estimated, K_expected, atol=1e-4, rtol=1e-4)
        assert_close(R_estimated, R_expected, atol=1e-4, rtol=1e-4)
        assert_close(t_estimated, t_expected, atol=1e-4, rtol=1e-4)

    def test_projection_from_krt(self, device, dtype):
        K = torch.tensor(
            [[[17.006138, 122.441254, 390.211426], [0.0, 228.743622, 577.167480], [0.0, 0.0, 712.675232]]],
            device=device,
            dtype=dtype,
        )

        R = torch.tensor(
            [[[0.396559, 0.511023, -0.762625], [0.743249, -0.666318, -0.060006], [0.538815, 0.543024, 0.644052]]],
            device=device,
            dtype=dtype,
        )

        t = torch.tensor([[[-6.477699], [1.129624], [0.143123]]], device=device, dtype=dtype)

        P_expected = torch.tensor(
            [[[308.0, 139.0, 231.0, 84.0], [481.0, 161.0, 358.0, 341.0], [384.0, 387.0, 459.0, 102.0]]],
            device=device,
            dtype=dtype,
        )

        P_estimated = epi.projection_from_KRt(K, R, t)
        assert_close(P_estimated, P_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        P_mat = torch.rand(1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(epi.KRt_from_projection, (P_mat,), raise_exception=True, fast_mode=True)
