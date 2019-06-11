import pytest

import kornia
from kornia.testing import tensor_to_gradcheck_var

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


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
