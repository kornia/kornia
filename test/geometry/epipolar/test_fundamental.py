import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kornia.geometry.epipolar as epi

import test_common as utils


class TestNormalizePoints:
    def test_smoke(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        output = epi.normalize_points(points)
        assert len(output) == 2
        assert output[0].shape == (1, 1, 2)
        assert output[1].shape == (1, 3, 3)

    @pytest.mark.parametrize(
        "batch_size, num_points", [(1, 2), (2, 3), (3, 2)],
    )
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points = torch.rand(B, N, 2, device=device, dtype=dtype)
        output = epi.normalize_points(points)
        assert output[0].shape == (B, N, 2)
        assert output[1].shape == (B, 3, 3)

    def test_mean_std(self, device, dtype):
        points = torch.tensor([[
            [0., 0.], [0., 2.], [1., 1.], [1., 3.]
        ]], device=device, dtype=dtype)

        points_norm, trans = epi.normalize_points(points)
        points_std, points_mean = torch.std_mean(points_norm, dim=1)

        assert_allclose(points_mean, torch.zeros_like(points_mean))
        assert (points_std < 2.).all()

    def test_gradcheck(self, device):
        points = torch.rand(2, 3, 2,
                            device=device, requires_grad=True, dtype=torch.float64)
        assert gradcheck(epi.normalize_points, (points,), raise_exception=True)


class TestNormalizeTransformation:
    def test_smoke(self, device, dtype):
        trans = torch.rand(2, 2, device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert trans_norm.shape == (2, 2)

    @pytest.mark.parametrize(
        "batch_size, rows, cols", [(1, 2, 2), (2, 3, 3), (3, 4, 4), (2, 1, 2)],
    )
    def test_shape(self, batch_size, rows, cols, device, dtype):
        B, N, M = batch_size, rows, cols
        trans = torch.rand(B, N, M, device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert trans_norm.shape == (B, N, M)

    def test_check_last_val(self, device, dtype):
        trans = torch.tensor([[
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.0, 0.5]
        ]], device=device, dtype=dtype)

        trans_expected = torch.tensor([[
            [0.0, 0.0, 2.0],
            [0.0, 4.0, 0.0],
            [1.0, 0.0, 1.0]
        ]], device=device, dtype=dtype)

        trans_norm = epi.normalize_transformation(trans)
        assert_allclose(trans_norm, trans_expected)

    def test_check_corner_case(self, device, dtype):
        trans = torch.tensor([[
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.0, 0.0]
        ]], device=device, dtype=dtype)

        trans_expected = trans.clone()

        trans_norm = epi.normalize_transformation(trans)
        assert_allclose(trans_norm, trans_expected)

    def test_gradcheck(self, device):
        trans = torch.rand(2, 3, 3,
                           device=device, requires_grad=True, dtype=torch.float64)
        assert gradcheck(epi.normalize_transformation, (trans,), raise_exception=True)


class TestFindFundamental:
    def test_smoke(self, device, dtype):
        points1 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 1, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 1, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize(
        "batch_size, num_points", [(1, 2), (2, 3), (3, 2)],
    )
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (B, 3, 3)

    def test_opencv(self, device, dtype):
        points1 = torch.tensor([[
            [0.8569, 0.5982],
            [0.0059, 0.9649],
            [0.1968, 0.8846],
            [0.6084, 0.3467],
            [0.9633, 0.5274],
            [0.8941, 0.8939],
            [0.0863, 0.5133],
            [0.2645, 0.8882],
            [0.2411, 0.3045],
            [0.8199, 0.4107]]], device=device, dtype=dtype)

        points2 = torch.tensor([[
            [0.0928, 0.3013],
            [0.0989, 0.9649],
            [0.0341, 0.4827],
            [0.8294, 0.4469],
            [0.2230, 0.2998],
            [0.1722, 0.8182],
            [0.5264, 0.8869],
            [0.8908, 0.1233],
            [0.2338, 0.7663],
            [0.4466, 0.5696]]], device=device, dtype=dtype)

        weights = torch.ones(1, 10, device=device, dtype=dtype)

        # generated with OpenCV using above points
        # import cv2
        # Fm_expected, _ = cv2.findFundamentalMat(
        #   points1.detach().numpy().reshape(-1, 1, 2),
        #   points2.detach().numpy().reshape(-1, 1, 2), cv2.FM_8POINT)

        Fm_expected = torch.tensor([[
            [-0.47408533, 0.22033807, -0.00346677],
            [0.54935973, 1.31080955, -1.25028275],
            [-0.36690215, -1.08143769, 1.]]], device=device, dtype=dtype)

        F_mat = epi.find_fundamental(points1, points2, weights)
        assert_allclose(F_mat, Fm_expected, rtol=1e-4, atol=1e-4)

    def test_synthetic_sampson(self, device, dtype):

        scene: Dict[str, torch.Tensor] = utils.generate_two_view_random_scene(device, dtype)

        x1 = scene['x1']
        x2 = scene['x2']

        weights = torch.ones_like(x1)[..., 0]
        F_est = epi.find_fundamental(x1, x2, weights)

        error = epi.sampson_epipolar_distance(x1, x2, F_est)
        assert_allclose(error, torch.tensor(0., device=device, dtype=dtype), atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points1 = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(1, 10, 2, device=device, dtype=torch.float64)
        weights = torch.ones(1, 10, device=device, dtype=torch.float64)
        assert gradcheck(epi.find_fundamental,
                         (points1, points2, weights,), raise_exception=True)


class TestComputeCorrespondEpilimes:
    def test_smoke(self, device, dtype):
        point = torch.rand(1, 1, 2, device=device, dtype=dtype)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        lines = epi.compute_correspond_epilines(point, F_mat)
        assert lines.shape == (1, 1, 3)

    @pytest.mark.parametrize(
        "batch_size, num_points", [(1, 2), (2, 3), (3, 2)],
    )
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        point = torch.rand(B, N, 2, device=device, dtype=dtype)
        F_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        lines = epi.compute_correspond_epilines(point, F_mat)
        assert lines.shape == (B, N, 3)

    def test_opencv(self, device, dtype):
        point = torch.rand(1, 2, 2, device=device, dtype=dtype)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)

        point = torch.tensor([[
            [0.9794, 0.7994],
            [0.8163, 0.8500],
        ]], device=device, dtype=dtype)

        F_mat = torch.tensor([[
            [0.1185, 0.4438, 0.9869],
            [0.5670, 0.9447, 0.4100],
            [0.1546, 0.2554, 0.4485],
        ]], device=device, dtype=dtype)

        # generated with OpenCV using above points
        # import cv2
        # lines_expected = cv2.computeCorrespondEpilines(
        #    point.detach().numpy().reshape(-1, 1, 2), 0,
        #    F_mat.detach().numpy()[0]).transpose(1, 0, 2)

        lines_expected = torch.tensor([[
            [0.64643687, 0.7629675, 0.35658622],
            [0.65710586, 0.7537983, 0.35616538],
        ]], device=device, dtype=dtype)

        lines_est = epi.compute_correspond_epilines(point, F_mat)
        assert_allclose(lines_est, lines_expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        point = torch.rand(1, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.compute_correspond_epilines,
                         (point, F_mat,), raise_exception=True)


class TestFundamentlFromEssential:
    def test_smoke(self, device, dtype):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        E_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (B, 3, 3)

    def test_shape_large(self, device, dtype):
        E_mat = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)  # check broadcasting
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (1, 2, 3, 3)

    def test_from_to_essential(self, device, dtype):
        scene = utils.generate_two_view_random_scene(device, dtype)

        F_mat = scene['F']
        E_mat = epi.essential_from_fundamental(F_mat, scene['K1'], scene['K2'])
        F_hat = epi.fundamental_from_essential(E_mat, scene['K1'], scene['K2'])

        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        assert_allclose(F_mat_norm, F_hat_norm)

    def test_gradcheck(self, device):
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.fundamental_from_essential,
                         (E_mat, K1, K2,), raise_exception=True)


class TestFundamentalFromProjections:
    def test_smoke(self, device, dtype):
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        B: int = batch_size
        P1 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (B, 3, 3)

    def test_shape_large(self, device, dtype):
        P1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (1, 2, 3, 3)

    def test_from_to_projections(self, device, dtype):
        P1 = torch.tensor([[
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
        ]], device=device, dtype=dtype)

        P2 = torch.tensor([[
            [1., 1., 1., 3.],
            [0., 2., 0., 3.],
            [0., 1., 1., 0.],
        ]], device=device, dtype=dtype)

        F_mat = epi.fundamental_from_projections(P1, P2)
        P_mat = epi.projections_from_fundamental(F_mat)
        F_hat = epi.fundamental_from_projections(P_mat[..., 0], P_mat[..., 1])

        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        assert_allclose(F_mat_norm, F_hat_norm)

    def test_gradcheck(self, device):
        P1 = torch.rand(1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        P2 = torch.rand(1, 3, 4, device=device, dtype=torch.float64)
        assert gradcheck(epi.fundamental_from_projections,
                         (P1, P2,), raise_exception=True)
