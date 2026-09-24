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
from torch.autograd import gradcheck

import kornia.geometry.epipolar as epi
from kornia.geometry.camera import PinholeCamera

from testing.base import BaseTester
from testing.two_view import two_view_scene


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


class TestScaleIntrinsics(BaseTester):
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
        self.assert_close(camera_matrix_scale, camera_matrix_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        scale_factor = torch.ones(1, device=device, dtype=torch.float64, requires_grad=True)
        camera_matrix = torch.ones(1, 3, 3, device=device, dtype=torch.float64)
        assert self.gradcheck(epi.scale_intrinsics, (camera_matrix, scale_factor), raise_exception=True, fast_mode=True)


class TestProjectionFromKRt(BaseTester):
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
        self.assert_close(P_estimated, P_expected, atol=1e-4, rtol=1e-4)

    def test_krt_from_projection(self, device, dtype):
        P = torch.tensor(
            [[[10.0, 0.0, 30.0, 100.0], [0.0, 20.0, 40.0, 160.0], [0.0, 0.0, 1.0, 3.0]]], device=device, dtype=dtype
        )

        K_expected = torch.tensor([[[10.0, 0.0, 30.0], [0.0, 20.0, 40.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        R_expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        t_expected = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, dtype=dtype)

        K_estimated, R_estimated, t_estimated = epi.KRt_from_projection(P)
        self.assert_close(K_estimated, K_expected, atol=1e-4, rtol=1e-4)
        self.assert_close(R_estimated, R_expected, atol=1e-4, rtol=1e-4)
        self.assert_close(t_estimated, t_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        R = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        t = torch.rand(1, 3, 1, device=device, dtype=torch.float64)
        assert gradcheck(epi.projection_from_KRt, (K, R, t), raise_exception=True, fast_mode=True)


class TestProjectionsFromFundamental(BaseTester):
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
        assert self.gradcheck(epi.projections_from_fundamental, (F_mat,), raise_exception=True, fast_mode=True)


class TestKRtFromProjection(BaseTester):
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
        self.assert_close(K_estimated, K_expected, atol=1e-4, rtol=1e-4)
        self.assert_close(R_estimated, R_expected, atol=1e-4, rtol=1e-4)
        self.assert_close(t_estimated, t_expected, atol=1e-4, rtol=1e-4)

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
        self.assert_close(P_estimated, P_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        P_mat = torch.rand(1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        assert self.gradcheck(epi.KRt_from_projection, (P_mat,), raise_exception=True, fast_mode=True)


_NO_HALF_QR = "KRt_from_projection does not upcast half input, and torch.linalg.qr has no float16/bfloat16 kernel"


def _skip_half(dtype: torch.dtype, reason: str) -> None:
    if dtype in (torch.float16, torch.bfloat16):
        pytest.skip(reason)


def _dehom(x: torch.Tensor) -> torch.Tensor:
    return x[..., :2] / x[..., 2:]


def _project(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Pixels of the (B, N, 3) points X through the (B, 3, 4) camera P."""
    return _dehom(torch.cat([X, torch.ones_like(X[..., :1])], -1) @ P.transpose(-2, -1))


def _det3(M: torch.Tensor) -> torch.Tensor:
    """Determinant of a (..., 3, 3) matrix from its rows, without an LU kernel."""
    return (torch.linalg.cross(M[..., 0, :], M[..., 1, :]) * M[..., 2, :]).sum(-1)


def _unit(M: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) matrix scaled to unit Frobenius norm."""
    return M / M.flatten(-2).norm(dim=-1)[..., None, None]


class TestConventionProjection(BaseTester):
    def test_convention_projection_from_KRt_is_K_R_t(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        K1, K2, R, t, X = two_view["K1"], two_view["K2"], two_view["R"], two_view["t"], two_view["X"]
        P = epi.projection_from_KRt(K2, R, t)
        assert P.shape == (1, 3, 4)
        self.assert_close(P, K2 @ torch.cat([R, t], -1))
        # (R, t) is world-to-camera: a world point X projects like K (R X + t). The camera-to-world reading
        # K R^T (X - t) is the control; on this fixture it is more than 100 px away.
        atol = {torch.float16: 1.0, torch.bfloat16: 4.0, torch.float32: 1e-3}.get(dtype, 1e-8)
        world_to_camera = _dehom((X @ R.transpose(-2, -1) + t.transpose(-2, -1)) @ K2.transpose(-2, -1))
        camera_to_world = _dehom(((X - t.transpose(-2, -1)) @ R) @ K2.transpose(-2, -1))
        self.assert_close(_project(P, X), world_to_camera, rtol=0.0, atol=atol)
        assert (_project(P, X) - camera_to_world).abs().max() > 100.0
        # Batched inputs give one camera per batch element.
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        Pb = epi.projection_from_KRt(torch.cat([K1, K2]), torch.cat([eye, R]), torch.cat([torch.zeros_like(t), t]))
        assert Pb.shape == (2, 3, 4)
        self.assert_close(Pb[:1], torch.cat([K1, torch.zeros_like(t)], -1))
        self.assert_close(Pb[1:], P)
        # K, R and t must have the same number of dimensions.
        with pytest.raises(AssertionError):
            epi.projection_from_KRt(K2[0], R, t)

    def test_convention_krt_from_projection_returns_extrinsic_t(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        # P must have exactly one batch dimension.
        for P_bad in (two_view["P2"][0], two_view["P2"][None]):
            with pytest.raises(Exception):
                epi.KRt_from_projection(P_bad)
        _skip_half(dtype, _NO_HALF_QR)
        K_true, R_true, t_true = two_view["K2"], two_view["R"], two_view["t"]
        K, R, t = epi.KRt_from_projection(two_view["P2"])
        assert K.shape == (1, 3, 3) and R.shape == (1, 3, 3) and t.shape == (1, 3, 1)
        # K is upper triangular with a positive diagonal; R is a rotation.
        assert torch.tril(K, diagonal=-1).abs().max() <= 1e-6 * K.abs().max()
        assert (K.diagonal(dim1=-2, dim2=-1) > 0).all()
        # K's entries are in pixels, so its zero skew carries float32 roundoff at that scale.
        self.assert_close(K, K_true, rtol=1e-4, atol=1e-3)
        self.assert_close(R, R_true)
        self.assert_close(R @ R.transpose(-2, -1), torch.eye(3, device=device, dtype=dtype)[None])
        self.assert_close(_det3(R), torch.ones(1, device=device, dtype=dtype))
        # t is the extrinsic translation of P = K [R | t], not the camera centre -R^T t (OpenCV's
        # decomposeProjectionMatrix returns the centre); the two are about 1.0 apart on this fixture.
        self.assert_close(t, t_true)
        assert (t - (-R_true.transpose(-2, -1) @ t_true)).abs().max() > 0.5
        # K is not normalised to K[2, 2] = 1: it carries the scale of P, while R and t do not.
        K2x, R2x, t2x = epi.KRt_from_projection(2.0 * two_view["P2"])
        self.assert_close(K2x[:, 2, 2], torch.full_like(K2x[:, 2, 2], 2.0))
        self.assert_close(K2x / K2x[..., 2:, 2:], K_true, rtol=1e-4, atol=1e-3)
        self.assert_close(R2x, R_true)
        self.assert_close(t2x, t_true)

    def test_convention_projections_from_fundamental_layout(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        # F is the essential matrix of the fixture, the fundamental matrix of normalised image coordinates, so
        # its entries stay inside the half-precision range; it satisfies x2^T F x1 = 0 like find_fundamental's F.
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        F = epi.essential_from_Rt(eye, torch.zeros_like(two_view["t"]), two_view["R"], two_view["t"])
        PP = epi.projections_from_fundamental(F)
        # The pair is stacked on the last dimension, first image first: [..., 0] = [I | 0].
        assert PP.shape == (1, 3, 4, 2)
        self.assert_close(PP[..., 0], torch.eye(3, 4, device=device, dtype=dtype)[None])
        # [..., 1] = [[e2]_x F | e2] with e2 the epipole in the second image, the left null vector of F.
        # The right null vector (the epipole in the first image) is the control.
        e2 = PP[:, :, 3, 1]
        assert (e2[:, None] @ F).abs().max() < 1e-2 * (F @ e2[..., None]).abs().max()
        self.assert_close(PP[:, :, :3, 1], epi.cross_product_matrix(e2) @ F)
        # The pair reproduces F up to scale in the (first, second) order; the reversed order gives F^T.
        F_pair = epi.fundamental_from_projections(PP[..., 0], PP[..., 1])
        F_reversed = epi.fundamental_from_projections(PP[..., 1], PP[..., 0])
        Ft = F.transpose(-2, -1)
        self.assert_close(_unit(F_pair) * torch.sign((F_pair * F).sum()), _unit(F))
        self.assert_close(_unit(F_reversed) * torch.sign((F_reversed * Ft).sum()), _unit(Ft))
        assert (_unit(F_reversed) * torch.sign((F_reversed * F).sum()) - _unit(F)).abs().max() > 0.1
        # F_mat must have exactly one batch dimension.
        for F_bad in (F[0], F[None]):
            with pytest.raises(Exception):
                epi.projections_from_fundamental(F_bad)

    def test_convention_depth_from_point_is_camera_z(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        R, t, X = two_view["R"], two_view["t"], two_view["X"]
        depth = epi.depth_from_point(R, t, X)
        # The z coordinate of R X + t, one value per point.
        assert depth.shape == (1, 12)
        self.assert_close(depth, (X @ R.transpose(-2, -1) + t.transpose(-2, -1))[..., 2])
        # Controls: the camera-to-world reading R^T (X - t) is about 0.58 away, and negating t moves every depth
        # by 2 * t_z = 0.04.
        assert (depth - ((X - t.transpose(-2, -1)) @ R)[..., 2]).abs().max() > 0.25
        assert (epi.depth_from_point(R, -t, X) - depth).abs().min() > 0.02
        # The sign is not checked: the fixture points are in front of the camera, and their negatives, behind it,
        # get a negative depth instead of an error.
        assert (depth > 0).all()
        assert (epi.depth_from_point(R, t, -X) < 0).all()

    def test_convention_scale_intrinsics_matches_pinhole_scale(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        K = torch.cat([two_view["K1"], two_view["K2"]])
        K_before = K.clone()
        scale = torch.tensor([0.5, 3.0], device=device, dtype=dtype)
        out = epi.scale_intrinsics(K, scale)
        # A (B,) scale factor scales batch element b by scale[b]; the focal lengths show the pairing.
        expected_f = torch.tensor([[400.0, 380.0], [2100.0, 2220.0]], device=device, dtype=dtype)
        self.assert_close(out[:, [0, 1], [0, 1]], expected_f)
        # It is the rule of PinholeCamera.scale, bit for bit, principal point included (its value is the #4263 wart
        # pinned below).
        bottom = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype).expand(2, 1, 4)
        K44 = torch.cat([torch.cat([K, torch.zeros_like(K[..., :1])], -1), bottom], -2)
        E44 = torch.eye(4, device=device, dtype=dtype)[None].expand(2, 4, 4)
        size = torch.tensor([480.0, 480.0], device=device, dtype=dtype)
        pinhole = PinholeCamera(K44, E44, size, size.clone()).scale(scale)
        assert torch.equal(out, pinhole.intrinsics[:, :3, :3])
        # A new tensor is returned; the input is not modified.
        assert torch.equal(K, K_before)

    def test_convention_intrinsics_like_follows_input(self, device, dtype):
        image = torch.zeros(2, 3, 4, 6, device=device, dtype=dtype)  # H = 4, W = 6
        K = epi.intrinsics_like(500.0, image)
        assert K.shape == (2, 3, 3)
        assert K.dtype == dtype and K.device == image.device
        # fx = fy = focal, no skew, K[2] = [0, 0, 1].
        self.assert_close(K[:, [0, 1], [0, 1]], torch.full((2, 2), 500.0, device=device, dtype=dtype))
        zeros = torch.zeros(2, device=device, dtype=dtype)
        self.assert_close(K[:, 0, 1], zeros)
        self.assert_close(K[:, 1, 0], zeros)
        self.assert_close(K[:, 2], torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).expand(2, 3))
        # cx follows the width and cy the height: cx - cy = (W - H) / 2 whichever pixel-centre rule #4263 settles
        # on; the transposed image is the control.
        self.assert_close(K[:, 0, 2] - K[:, 1, 2], torch.full_like(zeros, 1.0))
        K_t = epi.intrinsics_like(500.0, image.transpose(-2, -1))
        self.assert_close(K_t[:, 0, 2] - K_t[:, 1, 2], torch.full_like(zeros, -1.0))
        # An integer image raises instead of returning an integer K.
        with pytest.raises(Exception):
            epi.intrinsics_like(500.0, torch.zeros(1, 3, 4, 6, device=device, dtype=torch.uint8))

    def test_wart_krt_from_projection_negative_P_reflection_4864(self, device, dtype):
        two_view = two_view_scene(device, dtype)
        _skip_half(dtype, _NO_HALF_QR)
        # #4864: -P is the same camera as P, but the result keeps K's diagonal positive and returns the reflection
        # -R (det -1) with -t. Once fixed, R is a proper rotation and the sign moves into K.
        P, R_true, t_true = two_view["P2"], two_view["R"], two_view["t"]
        _, R_pos, _ = epi.KRt_from_projection(P)
        self.assert_close(_det3(R_pos), torch.ones(1, device=device, dtype=dtype))
        K, R, t = epi.KRt_from_projection(-P)
        assert (K.diagonal(dim1=-2, dim2=-1) > 0).all()
        self.assert_close(_det3(R), -torch.ones(1, device=device, dtype=dtype))
        self.assert_close(R, -R_true)
        self.assert_close(t, -t_true)
        # Same issue: eps is added to K's raw diagonal before its sign is taken, so 1e-7 * P (a positive scale)
        # keeps a negative K[2, 2] and returns a reflection.
        # The matching (last) row of R is negated; the other two are R_true's.
        K_small, R_small, _ = epi.KRt_from_projection(1e-7 * P)
        assert K_small[0, 2, 2] < 0
        self.assert_close(R_small[:, :2], R_true[:, :2])
        self.assert_close(R_small[:, 2], -R_true[:, 2])

    def test_wart_scale_intrinsics_principal_point_rule_4263(self, device, dtype):
        # #4263: the same rule as PinholeCamera.scale, pinned there by
        # test_wart_scale_rescales_the_principal_point_by_the_half_pixel_rule_4263 on the same numbers: cx' = s * cx
        # (2.0 for cx = 4, s = 0.5), the half-pixel rule, while kornia's integer pixel centres give
        # cx' = s * cx + (s - 1) / 2 = 1.75. The skew K[0, 1] is not scaled (a full rescale gives 1.5).
        K = torch.tensor([[[100.0, 3.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        out = epi.scale_intrinsics(K, torch.tensor([0.5], device=device, dtype=dtype))
        self.assert_close(out[:, 0, 2], torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(out[:, 1, 2], torch.tensor([1.5], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(out[:, 0, 0], torch.tensor([50.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(out[:, 0, 1], torch.tensor([3.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_wart_intrinsics_like_principal_point_half_pixel_4263(self, device, dtype):
        # #4263 (comment 5820150381 adds this construction site): the principal point is (W / 2, H / 2), (3.0, 2.0)
        # for H = 4, W = 6, the half-pixel centre, while kornia's integer pixel centres put it at
        # ((W - 1) / 2, (H - 1) / 2) = (2.5, 1.5).
        K = epi.intrinsics_like(500.0, torch.zeros(1, 3, 4, 6, device=device, dtype=dtype))
        self.assert_close(K[:, 0, 2], torch.tensor([3.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(K[:, 1, 2], torch.tensor([2.0], device=device, dtype=dtype), atol=0.0, rtol=0.0)
