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

import torch

import kornia

from testing.base import BaseTester


class TestProjectPoints(BaseTester):
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
        self.assert_close(point_3d, point_3d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # TODO: point [0, 0, 0] crashes
        points_3d = torch.ones(1, 3, device=device)
        camera_matrix = torch.eye(3, device=device).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.camera.project_points, (points_3d, camera_matrix))

    def test_jit(self, device, dtype):
        points_3d = torch.zeros(1, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        op = kornia.geometry.camera.project_points
        op_jit = torch.jit.script(op)
        self.assert_close(op(points_3d, camera_matrix), op_jit(points_3d, camera_matrix))

    def test_wart_project_points_skips_the_divide_at_z_zero_4267(self, device, dtype):
        # Wart pin for kornia#4267 (audit labels 5a-pp-13, 5a-pp-15, Y4-05): convert_points_from_homogeneous masks
        # |z| <= 1e-8 to a divisor of 1 and project_points applies K AFTER that divide, so a point on the camera
        # plane projects to fx*x + cx = 104, fy*y + cy = 203 instead of raising or returning inf. Four other
        # entry points answer differently at the same input: PinholeCamera.project gives [[100, 200]],
        # project_points_z1 and Z1Projection.project give inf, cam2pixel gives a finite 1e14.
        # A point BEHIND the camera is projected just as silently: z = -4 gives [[-21, -47]].
        # Snippet used to generate expected: project_points([[1., 2., 0.]], K3) executed 2026-09-05 (torch 2.14.0,
        # cpu and mps, every dtype). Pins the CURRENT value; NOT a contract; delete when #4267 is repaired.
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        singular = torch.tensor([[1.0, 2.0, 0.0]], device=device, dtype=dtype)
        out = kornia.geometry.camera.project_points(singular, camera_matrix)
        self.assert_close(out, torch.tensor([[104.0, 203.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        behind = kornia.geometry.camera.project_points(
            torch.tensor([[1.0, 2.0, -4.0]], device=device, dtype=dtype), camera_matrix
        )
        self.assert_close(behind, torch.tensor([[-21.0, -47.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)

    def test_convention_integer_pixel_centres_put_the_principal_point_at_w_minus_one_half(self, device, dtype):
        # Convention pin for the anchor block on PinholeCamera (audit labels 5a-al-04, 5a-pp-02): pixel
        # coordinates are (u, v) = (column, row) with INTEGER pixel centres -- create_meshgrid enumerates
        # [0, 0] for the first pixel and [W - 1, H - 1] for the last -- so a centred image has its principal
        # point at cx = (W - 1) / 2, cy = (H - 1) / 2, and NOT at (W / 2, H / 2), the half-pixel/COLMAP value.
        # The pin uses H = 2 != W = 3 so a transposed reading of the convention changes every literal, and
        # checks the definition of "centred" directly: under cx = 1, cy = 0.5 the first and the last pixel
        # unproject to exact negatives of each other, and the principal point itself is where the optical axis
        # (0, 0, 1) lands. A half-pixel cx = 1.5, cy = 1.0 would put neither of those where they are here.
        # Snippet used to generate expected: create_meshgrid(2, 3, normalized_coordinates=False) and the three
        # calls below executed 2026-09-05 (torch 2.14.0, cpu and mps, every dtype)
        # -> grid [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], unproject [[-1, -0.5, 1]] and [[1, 0.5, 1]],
        # project [[1.0, 0.5]]. Every literal is a dyadic rational, so the comparisons are exact.
        grid = kornia.geometry.create_meshgrid(2, 3, normalized_coordinates=False, device=device, dtype=dtype)
        assert grid.shape == (1, 2, 3, 2)
        flat = grid.reshape(-1, 2)
        first, last = flat[0][None], flat[-1][None]
        self.assert_close(first, torch.tensor([[0.0, 0.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        self.assert_close(last, torch.tensor([[2.0, 1.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0)
        # cx = (3 - 1) / 2 = 1, cy = (2 - 1) / 2 = 0.5, fx = fy = 1.
        camera_matrix = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        depth = torch.tensor([[1.0]], device=device, dtype=dtype)
        unprojected_first = kornia.geometry.camera.unproject_points(first, depth, camera_matrix)
        unprojected_last = kornia.geometry.camera.unproject_points(last, depth, camera_matrix)
        self.assert_close(
            unprojected_first, torch.tensor([[-1.0, -0.5, 1.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        self.assert_close(
            unprojected_last, torch.tensor([[1.0, 0.5, 1.0]], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        # The two are exact negatives in x and y, which is what "centred" means under integer pixel centres.
        self.assert_close(unprojected_first[..., :2], -unprojected_last[..., :2], atol=0.0, rtol=0.0)
        # The optical axis lands exactly on the principal point.
        on_axis = kornia.geometry.camera.project_points(
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype), camera_matrix
        )
        self.assert_close(on_axis, torch.tensor([[1.0, 0.5]], device=device, dtype=dtype), atol=0.0, rtol=0.0)


class TestUnprojectPoints(BaseTester):
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
        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_unproject_center_normalize(self, device, dtype):
        point_2d = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        depth = torch.tensor([[2.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)
        actual = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix, True)
        self.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_unproject_and_project(self, device, dtype):
        point_2d = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
        depth = torch.tensor([[2.0]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        point_3d = kornia.geometry.camera.unproject_points(point_2d, depth, camera_matrix)
        point_2d_hat = kornia.geometry.camera.project_points(point_3d, camera_matrix)
        self.assert_close(point_2d, point_2d_hat, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        points_2d = torch.zeros(1, 2, device=device, dtype=torch.float64)
        depth = torch.ones(1, 1, device=device, dtype=torch.float64)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).expand(1, -1, -1)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.camera.unproject_points, (points_2d, depth, camera_matrix))

    def test_jit(self, device, dtype):
        points_2d = torch.zeros(1, 2, device=device, dtype=dtype)
        depth = torch.ones(1, 1, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(1, -1, -1)
        args = (points_2d, depth, camera_matrix)
        op = kornia.geometry.camera.unproject_points
        op_jit = torch.jit.script(op)
        self.assert_close(op(*args), op_jit(*args))

    def test_convention_normalize_makes_depth_the_ray_length(self, device, dtype):
        # Convention pin (audit labels 5a-up-01, 5a-up-02): ``depth`` is the CAMERA-frame z by default, so pixel
        # (29, 53) at depth 2 unprojects to (0.5, 1, 2); with ``normalize=True`` the same depth is the length of
        # the ray instead, so the result has norm 2 and its z component is strictly below 2. The pixel is off the
        # principal point (cx = 4 != cy = 3) so the two readings differ; a centred pixel would not discriminate.
        # Snippet used to generate expected: hand arithmetic ((29-4)/100*2, (53-3)/100*2, 2), re-executed
        # 2026-09-05 (torch 2.14.0, cpu and mps) -> [[0.5, 1.0, 2.0]] and [[0.43643576, 0.87287152, 1.74574304]].
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )
        points_2d = torch.tensor([[29.0, 53.0]], device=device, dtype=dtype)
        depth = torch.tensor([[2.0]], device=device, dtype=dtype)
        z_depth = kornia.geometry.camera.unproject_points(points_2d, depth, camera_matrix)
        ray_depth = kornia.geometry.camera.unproject_points(points_2d, depth, camera_matrix, normalize=True)
        self.assert_close(z_depth, torch.tensor([[0.5, 1.0, 2.0]], device=device, dtype=dtype))
        self.assert_close(ray_depth.norm(dim=-1), torch.tensor([2.0], device=device, dtype=dtype))
        assert bool(ray_depth[0, 2] < 2.0)
