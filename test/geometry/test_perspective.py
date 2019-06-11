import pytest

import kornia
from kornia.testing import tensor_to_gradcheck_var

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestProjectPoints:
    def test_smoke(self):
        point_3d = torch.zeros(1, 3)
        camera_matrix = torch.eye(3).expand(1, -1, -1)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (1, 2)

    def test_smoke_batch(self):
        point_3d = torch.zeros(2, 3)
        camera_matrix = torch.eye(3).expand(2, -1, -1)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 2)

    def test_smoke_batch_multi(self):
        point_3d = torch.zeros(2, 4, 3)
        camera_matrix = torch.eye(3).expand(2, 4, -1, -1)
        point_2d = kornia.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 4, 2)

    def test_project_and_unproject(self):
        point_3d = torch.tensor([[10., 2., 30.]])
        depth = point_3d[..., -1:]
        camera_matrix = torch.tensor([[
            [2746., 0., 991.],
            [0., 2748., 619.],
            [0., 0., 1.],
        ]])
        point_2d = kornia.project_points(point_3d, camera_matrix)
        point_3d_hat = kornia.unproject_points(point_2d, depth, camera_matrix)
        assert_allclose(point_3d, point_3d_hat)

    def test_gradcheck(self):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3)
        camera_matrix = torch.eye(3).expand(1, -1, -1)

        # evaluate function gradient
        points_3d = tensor_to_gradcheck_var(points_3d)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.project_points,
                         (points_3d, camera_matrix),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(points_3d, camera_matrix):
            return kornia.project_points(points_3d, camera_matrix)

        points_3d = torch.zeros(1, 3)
        camera_matrix = torch.eye(3).expand(1, -1, -1)
        actual = op_script(points_3d, camera_matrix)
        expected = kornia.project_points(points_3d, camera_matrix)

        assert_allclose(actual, expected)


class TestUnprojectPoints:
    def test_smoke(self):
        points_2d = torch.zeros(1, 2)
        depth = torch.ones(1, 1)
        camera_matrix = torch.eye(3).expand(1, -1, -1)
        point_3d = kornia.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (1, 3)

    def test_smoke_batch(self):
        points_2d = torch.zeros(2, 2)
        depth = torch.ones(2, 1)
        camera_matrix = torch.eye(3).expand(2, -1, -1)
        point_3d = kornia.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (2, 3)

    def test_smoke_multi_batch(self):
        points_2d = torch.zeros(2, 3, 2)
        depth = torch.ones(2, 3, 1)
        camera_matrix = torch.eye(3).expand(2, 3, -1, -1)
        point_3d = kornia.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (2, 3, 3)

    def test_unproject_center(self):
        point_2d = torch.tensor([[0., 0.]])
        depth = torch.tensor([[2.]])
        camera_matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        expected = torch.tensor([[0., 0., 2.]])
        actual = kornia.unproject_points(point_2d, depth, camera_matrix)
        assert_allclose(actual, expected)

    def test_unproject_center_normalize(self):
        point_2d = torch.tensor([[0., 0.]])
        depth = torch.tensor([[2.]])
        camera_matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        expected = torch.tensor([[0., 0., 2.]])
        actual = kornia.unproject_points(point_2d, depth, camera_matrix, True)
        assert_allclose(actual, expected)

    def test_unproject_and_project(self):
        point_2d = torch.tensor([[0., 0.]])
        depth = torch.tensor([[2.]])
        camera_matrix = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        point_3d = kornia.unproject_points(point_2d, depth, camera_matrix)
        point_2d_hat = kornia.project_points(point_3d, camera_matrix)
        assert_allclose(point_2d, point_2d_hat)

    def test_gradcheck(self):
        points_2d = torch.zeros(1, 2)
        depth = torch.ones(1, 1)
        camera_matrix = torch.eye(3).expand(1, -1, -1)

        # evaluate function gradient
        points_2d = tensor_to_gradcheck_var(points_2d)
        depth = tensor_to_gradcheck_var(depth)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.unproject_points,
                         (points_2d, depth, camera_matrix),
                         raise_exception=True)

    def test_jit(self):
        @torch.jit.script
        def op_script(points_2d, depth, camera_matrix):
            return kornia.unproject_points(points_2d, depth, camera_matrix, False)

        points_2d = torch.zeros(1, 2)
        depth = torch.ones(1, 1)
        camera_matrix = torch.eye(3).expand(1, -1, -1)
        actual = op_script(points_2d, depth, camera_matrix)
        expected = kornia.unproject_points(points_2d, depth, camera_matrix)

        assert_allclose(actual, expected)
