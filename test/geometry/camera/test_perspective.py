import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import assert_close, tensor_to_gradcheck_var


class TestProjectPoints:
    def test_smoke(self, device, dtype):
        point_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (1, 2)

    def test_smoke_batch(self, device, dtype):
        point_3d = torch.zeros(2, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 2)

    def test_smoke_batch_multi(self, device, dtype):
        point_3d = torch.zeros(2, 4, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, 4, -1, -1)
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert point_2d.shape == (2, 4, 2)

    def test_project_and_unproject(self, device, dtype):
        point_3d = torch.tensor([[10.0, 2.0, 30.0]], device=device, dtype=dtype)
        depth = point_3d[..., -1:]
        camera_matrix = torch.tensor(
            [[[2746.0, 0.0, 991.0], [0.0, 2748.0, 619.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        point_2d = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        point_3d_hat = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)

        # evaluate function gradient
        points_3d = tensor_to_gradcheck_var(points_3d)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.geometry.camera.project_points, (points_3d, camera_matrix), raise_exception=True)

    def test_jit(self, device, dtype):
        points_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.camera.project_points
        op_jit = torch.jit.script(op)
        assert_close(op(points_3d, camera_matrix), op_jit(points_3d, camera_matrix))


class TestUnprojectPoints:
    def test_smoke(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        depth = torch.ones(1, 1, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        point_3d = kornia.geometry.camera.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (1, 3)

    def test_smoke_batch(self, device, dtype):
        points_2d = torch.zeros(2, 2, device=device, dtype=dtype)
        depth = torch.ones(2, 1, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1)
        point_3d = kornia.geometry.camera.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (2, 3)

    def test_smoke_multi_batch(self, device, dtype):
        points_2d = torch.zeros(2, 3, 2, device=device, dtype=dtype)
        depth = torch.ones(2, 3, 1, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, 3, -1, -1)
        point_3d = kornia.geometry.camera.unproject_points(points_2d, depth, camera_matrix)
        assert point_3d.shape == (2, 3, 3)

    def test_unproject_center(self, device, dtype):
        point_2d = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        depth = torch.tensor([[2.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)
        actual = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_unproject_center_normalize(self, device, dtype):
        point_2d = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        depth = torch.tensor([[2.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)
        actual = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix, True)
        assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_unproject_and_project(self, device, dtype):
        point_2d = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        depth = torch.tensor([[2.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        point_3d = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        point_2d_hat = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        assert_close(point_2d, point_2d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        depth = torch.ones(1, 1, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)

        # evaluate function gradient
        points_2d = tensor_to_gradcheck_var(points_2d)
        depth = tensor_to_gradcheck_var(depth)
        camera_matrix = tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(
            kornia.geometry.camera.unproject_points, (points_2d, depth, camera_matrix), raise_exception=True
        )

    def test_jit(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        depth = torch.ones(1, 1, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        args = (points_2d, depth, camera_matrix)
        op = kornia.geometry.camera.unproject_points
        op_jit = torch.jit.script(op)
        assert_close(op(*args), op_jit(*args))
