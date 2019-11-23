import pytest

import kornia
from kornia.testing import tensor_to_gradcheck_var
from test.common import device

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestProjectPoints:
    def test_smoke(self, device):
        point_3d = torch.zeros(1, 3).to(device)
        camera_matrix = torch.eye(3).expand(1, -1, -1).to(device)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (1, 2)

    def test_smoke_batch(self, device):
        point_3d = torch.zeros(2, 3).to(device)
        camera_matrix = torch.eye(3).expand(2, -1, -1).to(device)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 2)

    def test_smoke_batch_multi(self, device):
        point_3d = torch.zeros(2, 4, 3).to(device)
        camera_matrix = torch.eye(3).expand(2, 4, -1, -1).to(device)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 4, 2)

    def test_project_and_unproject(self, device):
        point_3d = torch.tensor([[10., 2., 30.]]).to(device)
        depth = point_3d[..., -1:]
        camera_matrix = torch.tensor([[
            [2746., 0., 991.],
            [0., 2748., 619.],
            [0., 0., 1.],
        ]]).to(device)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        point_3d_hat = kornia.unproject_points(point_2d, depth, camera_matrix)
        assert_allclose(point_3d, point_3d_hat)

    def test_gradcheck(self, device):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3).to(device)
        camera_matrix = torch.eye(3).expand(1, -1, -1).to(device)

        # evaluate function gradient
        points_3d = tensor_to_gradcheck_var(points_3d)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.project_points,
                         (points_3d, camera_matrix),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(points_3d, camera_matrix):
            return kornia.project_points(points_3d, camera_matrix)

        points_3d = torch.zeros(1, 3).to(device)
        camera_matrix = torch.eye(3).expand(1, -1, -1).to(device)
        actual = op_script(points_3d, camera_matrix)
        expected = kornia.project_points(points_3d, camera_matrix)

        assert_allclose(actual, expected)


class TestUnprojectPoints:
    def test_smoke(self, device):
        points_2d = torch.zeros(1, 2).to(device)
        depth = torch.ones(1, 1).to(device)
        camera_matrix = torch.eye(3).expand(1, -1, -1).to(device)
        point_3d = kornia.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (1, 3)

    def test_smoke_batch(self, device):
        points_2d = torch.zeros(2, 2).to(device)
        depth = torch.ones(2, 1).to(device)
        camera_matrix = torch.eye(3).expand(2, -1, -1).to(device)
        point_3d = kornia.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (2, 3)

    def test_smoke_multi_batch(self, device):
        points_2d = torch.zeros(2, 3, 2).to(device)
        depth = torch.ones(2, 3, 1).to(device)
        camera_matrix = torch.eye(3).expand(2, 3, -1, -1).to(device)
        point_3d = kornia.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (2, 3, 3)

    def test_unproject_center(self, device):
        point_2d = torch.tensor([[0., 0.]]).to(device)
        depth = torch.tensor([[2.]]).to(device)
        camera_matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]).to(device)
        expected = torch.tensor([[0., 0., 2.]]).to(device)
        actual = kornia.unproject_points(point_2d, depth, camera_matrix)
        assert_allclose(actual, expected)

    def test_unproject_center_normalize(self, device):
        point_2d = torch.tensor([[0., 0.]]).to(device)
        depth = torch.tensor([[2.]]).to(device)
        camera_matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]).to(device)
        expected = torch.tensor([[0., 0., 2.]]).to(device)
        actual = kornia.unproject_points(point_2d, depth, camera_matrix, True)
        assert_allclose(actual, expected)

    def test_unproject_and_project(self, device):
        point_2d = torch.tensor([[0., 0.]]).to(device)
        depth = torch.tensor([[2.]]).to(device)
        camera_matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]).to(device)
        point_3d = kornia.unproject_points(point_2d, depth, camera_matrix)
        point_2d_hat = kornia.project_points(point_3d, camera_matrix)
        assert_allclose(point_2d, point_2d_hat)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2).to(device)
        depth = torch.ones(1, 1).to(device)
        camera_matrix = torch.eye(3).expand(1, -1, -1).to(device)

        # evaluate function gradient
        points_2d = tensor_to_gradcheck_var(points_2d)
        depth = tensor_to_gradcheck_var(depth)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.unproject_points,
                         (points_2d, depth, camera_matrix),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):
        @torch.jit.script
        def op_script(points_2d, depth, camera_matrix):
            return kornia.unproject_points(
                points_2d, depth, camera_matrix, False)

        points_2d = torch.zeros(1, 2).to(device)
        depth = torch.ones(1, 1).to(device)
        camera_matrix = torch.eye(3).expand(1, -1, -1).to(device)
        actual = op_script(points_2d, depth, camera_matrix)
        expected = kornia.unproject_points(points_2d, depth, camera_matrix)

        assert_allclose(actual, expected)
