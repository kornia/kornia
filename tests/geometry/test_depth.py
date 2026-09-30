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

import kornia
from kornia.core.exceptions import BaseError, ShapeError
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.depth import (
    DepthWarper,
    depth_from_disparity,
    depth_from_plane_equation,
    depth_to_3d,
    depth_to_3d_v2,
    depth_to_normals,
    depth_warp,
    unproject_meshgrid,
    warp_frame_depth,
)

from testing.base import BaseTester


def _k_asymmetric(device, dtype, fy=100.0):
    """Return (1, 3, 3) intrinsics with fx = 100, cx = 4, cy = 3 and a caller-chosen fy.

    cx != cy, so a row/column swap moves x and y by different amounts; fy is a parameter so a pin can vary one
    intrinsic at a time instead of proving its claim only on a symmetric camera.
    """
    return torch.tensor([[[100.0, 0.0, 4.0], [0.0, fy, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)


def _k44_warp(device, dtype):
    """Return the (1, 4, 4) PinholeCamera intrinsics matching ``_k53_warp``: fx = fy = 1, cx = 2, cy = 1.5."""
    return torch.tensor(
        [[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 1.5, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )


def _k53_warp(device, dtype):
    """Return the (1, 3, 3) intrinsics used by the warp pins: fx = fy = 1, cx = 2, cy = 1.5.

    Unit focal lengths make a one-unit x translation move the sampling point by exactly one pixel at depth 1,
    which is what turns the warp-direction claim into an integer-valued pin.
    """
    return torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)


def _eye4(device, dtype):
    return torch.eye(4, device=device, dtype=dtype)[None]


def _tx_plus_one(device, dtype):
    """Return the (1, 4, 4) rigid transform with R = I and t = (1, 0, 0)."""
    return torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    )


class TestDepthTo3d(BaseTester):
    def test_smoke(self, device, dtype):
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes(self, batch_size, device, dtype):
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(batch_size, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
    def test_shapes_broadcast(self, batch_size, device, dtype):
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    def test_depth_to_3d_v2(self, device, dtype):
        depth = torch.rand(5, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(5, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)

        # TODO: implement me with batch
        # Permute the depth tensor to match the expected input shape for depth_to_3d_v2.
        depth = torch.permute(depth, (1, 0, 2, 3))
        points3d_v2 = kornia.geometry.depth.depth_to_3d_v2(depth[0], camera_matrix)
        # Align the output format of depth_to_3d with depth_to_3d_v2 by reordering dimensions.
        self.assert_close(points3d.permute(0, 2, 3, 1), points3d_v2)

    def test_depth_to_3d_v2_cached_grid(self, device, dtype):
        # Passing a pre-computed xyz_grid must not raise "Boolean value of Tensor
        # is ambiguous" (regression for the `xyz_grid or ...` truthiness bug).
        depth = torch.rand(2, 3, 4, device=device, dtype=dtype).add_(0.1)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(2, -1, -1).contiguous()
        grid = kornia.geometry.unproject_meshgrid(3, 4, camera_matrix, device=device, dtype=dtype)
        out_cached = kornia.geometry.depth.depth_to_3d_v2(depth, camera_matrix, xyz_grid=grid)
        out_uncached = kornia.geometry.depth.depth_to_3d_v2(depth, camera_matrix)
        self.assert_close(out_cached, out_uncached)

    @pytest.mark.parametrize("normalize_points", [False, True])
    def test_jit(self, normalize_points, device, dtype):
        depth = torch.rand(2, 1, 3, 4, device=device, dtype=dtype).add_(1)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        expected = kornia.geometry.depth.depth_to_3d(depth, camera_matrix, normalize_points).permute(0, 2, 3, 1)

        grid_jit = torch.jit.script(kornia.geometry.unproject_meshgrid)
        grid = grid_jit(3, 4, camera_matrix, normalize_points, device, dtype)
        self.assert_close(grid * depth[:, 0, ..., None], expected)

        depth_to_3d_jit = torch.jit.script(kornia.geometry.depth.depth_to_3d_v2)
        self.assert_close(depth_to_3d_jit(depth[:, 0], camera_matrix, normalize_points), expected)
        self.assert_close(depth_to_3d_jit(depth[:, 0], camera_matrix, normalize_points, grid), expected)

    def test_unproject_meshgrid(self, device, dtype):
        # TODO: implement me with batch
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        grid = kornia.geometry.unproject_meshgrid(3, 4, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (2, 3, 4, 3)
        # test for now that the grid is correct and have homogeneous coords
        self.assert_close(grid[..., 2], torch.ones_like(grid[..., 2]))

    @pytest.mark.parametrize("shape", [(3, 3), (2, 1, 3, 3), (2, 2, 3, 3), (2, 3, 3, 3)])
    def test_unproject_meshgrid_invalid_camera_rank(self, shape, device, dtype):
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(shape)
        # An extra camera axis of size W must not broadcast across pixel columns.
        with pytest.raises(ShapeError) as exc_info:
            kornia.geometry.unproject_meshgrid(2, 3, camera_matrix, device=device, dtype=dtype)
        assert f"Actual shape: {list(shape)}" in str(exc_info.value)

    @pytest.mark.parametrize(("height", "width"), [(1, 1), (1, 3), (3, 1), (2, 5)])
    def test_unproject_meshgrid_degenerate_sizes(self, height, width, device, dtype):
        # a single-column (W = 1) or single-row (H = 1) grid keeps its own axis
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        grid = kornia.geometry.unproject_meshgrid(height, width, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (2, height, width, 3)

    @pytest.mark.parametrize(("height", "width"), [(1, 1), (1, 3), (3, 1), (2, 5)])
    def test_depth_to_3d_v2_degenerate_sizes(self, height, width, device, dtype):
        depth = torch.rand(2, 1, height, width, device=device, dtype=dtype)
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 4.0], [0.0, 50.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        ).repeat(2, 1, 1)

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        points3d_v2 = kornia.geometry.depth.depth_to_3d_v2(depth[:, 0], camera_matrix)

        assert points3d_v2.shape == (2, height, width, 3)
        self.assert_close(points3d.permute(0, 2, 3, 1), points3d_v2)

    def test_unproject_denormalized(self, device, dtype):
        # this is for default normalize_points=False
        depth = 2 * torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        points3d_expected = torch.tensor(
            [
                [
                    [[0.0, 2.0, 4.0], [0.0, 2.0, 4.0], [0.0, 2.0, 4.0], [0.0, 2.0, 4.0]],
                    [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)  # default is normalize_points=False
        self.assert_close(points3d, points3d_expected, atol=1e-4, rtol=1e-4)

    def test_unproject_normalized(self, device, dtype):
        # this is for normalize_points=True
        depth = 2 * torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        points3d_expected = torch.tensor(
            [
                [
                    [
                        [0.0000, 1.4142, 1.7889],
                        [0.0000, 1.1547, 1.6330],
                        [0.0000, 0.8165, 1.3333],
                        [0.0000, 0.6030, 1.0690],
                    ],
                    [
                        [0.0000, 0.0000, 0.0000],
                        [1.4142, 1.1547, 0.8165],
                        [1.7889, 1.6330, 1.3333],
                        [1.8974, 1.8091, 1.6036],
                    ],
                    [
                        [2.0000, 1.4142, 0.8944],
                        [1.4142, 1.1547, 0.8165],
                        [0.8944, 0.8165, 0.6667],
                        [0.6325, 0.6030, 0.5345],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix, normalize_points=True)
        self.assert_close(points3d, points3d_expected, atol=1e-4, rtol=1e-4)

    def test_unproject_and_project(self, device, dtype):
        depth = 2 * torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        points2d = kornia.geometry.camera.project_points(points3d.permute(0, 2, 3, 1), camera_matrix[:, None, None])
        points2d_expected = kornia.geometry.create_meshgrid(4, 3, False, device=device).to(dtype=dtype)
        self.assert_close(points2d, points2d_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # generate input data
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=torch.float64)

        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.depth_to_3d, (depth, camera_matrix))

    def test_convention_pixel_origin_is_the_integer_centre(self, device, dtype):
        # Pixel (0, 0) unprojects as ((0 - cx) d / fx, (0 - cy) d / fy, d): the first pixel centre is at the integer
        # coordinate 0, and ``depth`` is the camera-frame z. A half-pixel (COLMAP) grid would give (-0.07, -0.05, 2).
        # cx != cy and H != W, so a transposed grid also changes the literal.
        # Snippet used to generate expected: depth_to_3d(full((1, 1, 2, 3), 2.0), K)[0, :, 0, 0]
        camera_matrix = _k_asymmetric(device, dtype)
        depth = torch.full((1, 1, 2, 3), 2.0, device=device, dtype=dtype)
        expected = torch.tensor([-0.08, -0.06, 2.0], device=device, dtype=dtype)
        self.assert_close(depth_to_3d(depth, camera_matrix)[0, :, 0, 0], expected)
        self.assert_close(depth_to_3d_v2(depth[:, 0], camera_matrix)[0, 0, 0], expected)

    def test_convention_integer_depth_follows_intrinsics_dtype(self, device, dtype):
        # Integer sensor depth follows the floating calibration dtype for both layouts.
        # At pixel (0, 0): ((0 - 4) * 2 / 100, (0 - 3) * 2 / 100, 2).
        camera_matrix = _k_asymmetric(device, dtype)
        for int_dtype in (torch.int64, torch.int32):
            depth = torch.full((1, 1, 2, 3), 2, device=device, dtype=int_dtype)
            points = depth_to_3d(depth, camera_matrix)
            points_v2 = depth_to_3d_v2(depth[:, 0], camera_matrix)
            assert points.dtype == dtype
            assert points_v2.dtype == dtype
            expected = torch.tensor([-0.08, -0.06, 2.0], device=device, dtype=dtype)
            self.assert_close(points[0, :, 0, 0], expected)
            self.assert_close(points_v2[0, 0, 0], expected)

    def test_convention_depth_to_3d_and_v2_agree_up_to_layout(self, device, dtype):
        # The two functions compute the same points in (B, 3, H, W) and (B, H, W, 3) layouts, equal after the
        # permutation; the flattened buffers differ, so a comparison that skipped the permute would fail.
        camera_matrix = _k_asymmetric(device, dtype, fy=50.0)
        depth = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]]]],
            device=device,
            dtype=dtype,
        )
        v1 = depth_to_3d(depth, camera_matrix)
        v2 = depth_to_3d_v2(depth[:, 0], camera_matrix)
        assert v1.shape == (1, 3, 3, 5)
        assert v2.shape == (1, 3, 5, 3)
        assert torch.equal(v1.permute(0, 2, 3, 1), v2)
        assert not torch.equal(v1.reshape(-1), v2.reshape(-1))

    def test_convention_depth_to_3d_v2_keeps_the_w_one_axis_4278(self, device, dtype):
        # kornia#4278: a single column keeps its width axis and agrees across layouts; warp_frame_depth keeps it too.
        camera_matrix = _k_asymmetric(device, dtype)
        depth = torch.full((1, 1, 3, 1), 2.0, device=device, dtype=dtype)
        v2 = depth_to_3d_v2(depth[:, 0], camera_matrix)
        assert v2.shape == (1, 3, 1, 3)
        assert torch.equal(depth_to_3d(depth, camera_matrix).permute(0, 2, 3, 1), v2)
        assert warp_frame_depth(depth, depth, _eye4(device, dtype), camera_matrix).shape == (1, 1, 3, 1)

    def test_convention_xyz_grid_bypasses_the_camera_matrix(self, device, dtype):
        # With ``xyz_grid`` given, depth_to_3d_v2 reads ``camera_matrix`` only through its (*, 3, 3) guard, so the
        # bare (3, 3) that the no-grid call rejects (#4271) is accepted and gives the batched call's result.
        camera_matrix = _k_asymmetric(device, dtype)
        depth = torch.ones(1, 3, 5, device=device, dtype=dtype)
        grid = unproject_meshgrid(3, 5, camera_matrix, device=device, dtype=dtype)
        bypassed = depth_to_3d_v2(depth, camera_matrix[0], xyz_grid=grid)
        assert bypassed.shape == (1, 3, 5, 3)
        assert torch.equal(bypassed, depth_to_3d_v2(depth, camera_matrix))
        with pytest.raises(ShapeError):
            depth_to_3d_v2(depth, camera_matrix[0])


class TestUnprojectMeshgrid(BaseTester):
    def test_convention_returns_b_h_w_3_and_equals_depth_to_3d_v2_at_depth_one(self, device, dtype):
        # unproject_meshgrid is depth_to_3d_v2's cache: the per-pixel ray at depth 1 in (*, H, W, 3), so multiplying
        # it by a depth map reproduces depth_to_3d_v2. The fy = 50 arm moves only the y component of the pixel-(1, 0)
        # ray, which fixes fy as the divisor of the ROW index.
        # Snippet used to generate expected: unproject_meshgrid(2, 3, K, device=..., dtype=...)
        camera_matrix = _k_asymmetric(device, dtype)
        grid = unproject_meshgrid(2, 3, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (1, 2, 3, 3)
        self.assert_close(grid[0, 0, 0], torch.tensor([-0.04, -0.03, 1.0], device=device, dtype=dtype))
        self.assert_close(grid[0, 1, 2], torch.tensor([-0.02, -0.02, 1.0], device=device, dtype=dtype))
        assert torch.equal(grid, depth_to_3d_v2(torch.ones(1, 2, 3, device=device, dtype=dtype), camera_matrix))
        asymmetric = unproject_meshgrid(2, 3, _k_asymmetric(device, dtype, fy=50.0), device=device, dtype=dtype)
        self.assert_close(asymmetric[0, 0, 0], torch.tensor([-0.04, -0.06, 1.0], device=device, dtype=dtype))
        self.assert_close(asymmetric[0, 1, 0], torch.tensor([-0.04, -0.04, 1.0], device=device, dtype=dtype))


class TestDepthToNormals(BaseTester):
    @pytest.mark.parametrize(("height", "width"), [(1, 3), (3, 1), (1, 1), (0, 3), (3, 0), (0, 0)])
    @pytest.mark.parametrize("normalize_points", [False, True])
    def test_convention_normals_require_two_spatial_axes_4398(self, height, width, normalize_points, device, dtype):
        # A surface normal needs two tangent directions, including when depth represents ray length.
        depth = torch.full((1, 1, height, width), 2.0, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype)[None]

        with pytest.raises(ShapeError, match="H >= 2 and W >= 2") as exc_info:
            depth_to_normals(depth, camera_matrix, normalize_points)

        assert exc_info.value.actual_shape == list(depth.shape)
        assert f"H={height}, W={width}" in str(exc_info.value)

    @pytest.mark.parametrize(("height", "width"), [(2, 2), (2, 3), (3, 2)])
    @pytest.mark.parametrize("normalize_points", [False, True])
    @pytest.mark.parametrize("camera_batch_size", [1, 2])
    def test_minimum_spatial_size_plane(self, height, width, normalize_points, camera_batch_size, device, dtype):
        camera_matrix = torch.tensor(
            [[[1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        ).expand(camera_batch_size, -1, -1)
        depth = torch.tensor([2.0, 4.0], device=device, dtype=dtype).view(2, 1, 1, 1).expand(2, 1, height, width)
        if normalize_points:
            # Ray lengths for the same fronto-parallel planes z=2 and z=4, rather than constant ray lengths.
            x = torch.arange(width, device=device, dtype=dtype) - 1.0
            y = (torch.arange(height, device=device, dtype=dtype) - 1.0) / 2.0
            depth = depth * (x[None, :] ** 2 + y[:, None] ** 2 + 1.0).sqrt()

        normals = depth_to_normals(depth, camera_matrix, normalize_points)

        assert normals.shape == (2, 3, height, width)
        assert normals.device == depth.device
        assert normals.dtype == dtype
        assert torch.isfinite(normals).all()
        expected = torch.zeros_like(normals)
        expected[:, 2] = 1.0
        self.assert_close(normals, expected)

    @pytest.mark.parametrize("normalize_points", [False, True])
    def test_dynamo_minimum_spatial_size(self, normalize_points, device, dtype, torch_optimizer):
        depth = torch.full((1, 1, 2, 3), 2.0, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype)[None]
        expected = depth_to_normals(depth, camera_matrix, normalize_points)

        compiled = torch_optimizer(depth_to_normals, fullgraph=True)
        self.assert_close(compiled(depth, camera_matrix, normalize_points), expected)

    def test_smoke(self, device, dtype):
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes(self, batch_size, device, dtype):
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(batch_size, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes_broadcast(self, batch_size, device, dtype):
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    def test_simple(self, device, dtype):
        # this is for default normalize_points=False
        depth = 2 * torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        normals_expected = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        normals = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)  # default is normalize_points=False
        self.assert_close(normals, normals_expected, rtol=1e-3, atol=1e-3)

    def test_simple_normalized(self, device, dtype):
        # this is for default normalize_points=False
        depth = 2 * torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)

        normals_expected = torch.tensor(
            [
                [
                    [
                        [0.3432, 0.4861, 0.7628],
                        [0.2873, 0.4260, 0.6672],
                        [0.2284, 0.3683, 0.5596],
                        [0.1695, 0.2980, 0.4496],
                    ],
                    [
                        [0.3432, 0.2873, 0.2363],
                        [0.4861, 0.4260, 0.3785],
                        [0.8079, 0.7261, 0.6529],
                        [0.8948, 0.8237, 0.7543],
                    ],
                    [
                        [0.8743, 0.8253, 0.6019],
                        [0.8253, 0.7981, 0.6415],
                        [0.5432, 0.5807, 0.5105],
                        [0.4129, 0.4824, 0.4784],
                    ],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        normals = kornia.geometry.depth.depth_to_normals(depth, camera_matrix, normalize_points=True)
        self.assert_close(normals, normals_expected, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        # generate input data
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=torch.float64)

        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.depth_to_normals, (depth, camera_matrix))

    def test_convention_normals_point_away_from_the_camera_and_x_tracks_the_column(self, device, dtype):
        # depth_to_normals takes dx x dy of the unprojected point cloud, so a fronto-parallel plane gets (0, 0, 1):
        # +z points AWAY from the camera. A depth growing with the COLUMN tilts the normal towards -x, one growing
        # with the ROW towards -y. The ramp magnitudes are dtype-dependent, so only their signs are asserted.
        # Snippet used to generate expected: depth_to_normals(depth, K)[0, :, 1, 1] on the 3 x 4 map
        camera_matrix = _k_asymmetric(device, dtype, fy=50.0)
        flat = depth_to_normals(torch.full((1, 1, 3, 4), 2.0, device=device, dtype=dtype), camera_matrix)
        assert flat.shape == (1, 3, 3, 4)
        self.assert_close(
            flat[0, :, 1, 1], torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype), atol=0.0, rtol=0.0
        )
        columns = torch.arange(4.0, device=device, dtype=dtype) + 1.0
        by_column = depth_to_normals(columns.view(1, 1, 1, 4).expand(1, 1, 3, 4).contiguous(), camera_matrix)[
            0, :, 1, 1
        ]
        assert by_column[0].item() < -0.9
        assert by_column[1].abs().item() < 0.01
        assert by_column[2].item() < -0.001
        rows = torch.arange(3.0, device=device, dtype=dtype) + 1.0
        by_row = depth_to_normals(rows.view(1, 1, 3, 1).expand(1, 1, 3, 4).contiguous(), camera_matrix)[0, :, 1, 1]
        assert by_row[1].item() < -0.9
        assert by_row[0].abs().item() < 0.01


class TestWarpFrameDepth(BaseTester):
    @pytest.mark.parametrize("normalize_points", [False, True])
    def test_jit(self, normalize_points, device, dtype):
        image = torch.rand(2, 3, 3, 4, device=device, dtype=dtype)
        depth = torch.rand(2, 1, 3, 4, device=device, dtype=dtype).add_(1)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        transform = torch.eye(4, device=device, dtype=dtype).repeat(2, 1, 1)
        op = kornia.geometry.depth.warp_frame_depth
        op_jit = torch.jit.script(op)
        expected = op(image, depth, transform, camera_matrix, normalize_points)
        self.assert_close(op_jit(image, depth, transform, camera_matrix, normalize_points), expected)

    def test_smoke(self, device, dtype):
        image_src = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        depth_dst = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        src_trans_dst = torch.rand(1, 4, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)

        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    @pytest.mark.parametrize("num_features", [1, 3, 5])
    def test_shape(self, batch_size, num_features, device, dtype):
        image_src = torch.rand(batch_size, num_features, 3, 4, device=device, dtype=dtype)
        depth_dst = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        src_trans_dst = torch.rand(batch_size, 4, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(batch_size, 3, 3, device=device, dtype=dtype)

        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (batch_size, num_features, 3, 4)

    @pytest.mark.parametrize(("height", "width"), [(1, 1), (1, 5), (4, 1)])
    def test_shape_degenerate_sizes(self, height, width, device, dtype):
        image_src = torch.rand(1, 2, height, width, device=device, dtype=dtype)
        depth_dst = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        src_trans_dst = torch.eye(4, device=device, dtype=dtype)[None]
        camera_matrix = torch.tensor(
            [[[100.0, 0.0, 4.0], [0.0, 50.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (1, 2, height, width)

    @pytest.mark.parametrize("batch_size", [0, 1])
    @pytest.mark.parametrize("depth_hw", [(3, 4), (3, 8), (6, 4)])
    def test_exception_image_depth_size_mismatch(self, batch_size, depth_hw, device, dtype):
        # kornia#4800: a depth map of another size used to resample the whole image onto its own grid. A mismatch
        # in the height alone or the width alone is rejected too.
        image_src = torch.rand(batch_size, 1, 6, 8, device=device, dtype=dtype)
        depth_dst = torch.ones(batch_size, 1, *depth_hw, device=device, dtype=dtype)
        src_trans_dst = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(batch_size, 1, 1)

        with pytest.raises(BaseError, match="same height and width"):
            warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)

    def test_empty_batch_4281(self, device, dtype):
        # Regression for kornia#4281: output geometry comes from the destination depth map.
        image_src = torch.zeros(0, 3, 4, 5, device=device, dtype=dtype, requires_grad=True)
        depth_dst = torch.zeros(0, 1, 4, 5, device=device, dtype=dtype)
        src_trans_dst = torch.zeros(0, 4, 4, device=device, dtype=dtype)
        camera_matrix = torch.zeros(0, 3, 3, device=device, dtype=dtype)

        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)

        assert image_dst.shape == (0, 3, 4, 5)
        assert image_dst.dtype == dtype
        assert image_dst.device == device
        assert image_dst.requires_grad
        image_dst.sum().backward()
        assert image_src.grad is not None

    def test_translation(self, device, dtype):
        # this is for normalize_points=False
        image_src = torch.tensor(
            [[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]], device=device, dtype=dtype
        )

        depth_dst = torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        src_trans_dst = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        h, w = image_src.shape[-2:]
        camera_matrix = torch.tensor(
            [[[1.0, 0.0, w / 2], [0.0, 1.0, h / 2], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        image_dst_expected = torch.tensor(
            [[[[2.0, 3.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 0.0]]]], device=device, dtype=dtype
        )

        image_dst = kornia.geometry.depth.warp_frame_depth(
            image_src, depth_dst, src_trans_dst, camera_matrix
        )  # default is normalize_points=False
        self.assert_close(image_dst, image_dst_expected, rtol=1e-3, atol=1e-3)

    def test_convention_subpixel_border_blends_with_zero_padding(self, device, dtype):
        # The baked padding_mode='zeros' extends the image with zeros: a quarter-pixel translation samples the last
        # column at x = 2.25 and blends the in-bounds 1.0 (weight 0.75) with the padded zero (weight 0.25).
        # Snippet used to generate expected: warp_frame_depth(ones(1, 1, 2, 3), ones, T, eye(3)), T[0, 0, 3] = 0.25
        image = torch.ones(1, 1, 2, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype)[None]
        transform = torch.eye(4, device=device, dtype=dtype)[None]
        transform[0, 0, 3] = 0.25
        result = warp_frame_depth(image, torch.ones_like(image), transform, camera_matrix)
        expected = torch.tensor([[[[1.0, 1.0, 0.75], [1.0, 1.0, 0.75]]]], device=device, dtype=dtype)
        self.assert_close(result, expected)

    def test_translation_normalized(self, device, dtype):
        # this is for normalize_points=True
        image_src = torch.tensor(
            [[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]], device=device, dtype=dtype
        )

        depth_dst = torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        src_trans_dst = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )

        h, w = image_src.shape[-2:]
        camera_matrix = torch.tensor(
            [[[1.0, 0.0, w / 2], [0.0, 1.0, h / 2], [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        image_dst_expected = torch.tensor(
            [
                [
                    [
                        [0.9223, 0.0000, 0.0000],
                        [2.8153, 1.5000, 0.0000],
                        [2.8028, 2.6459, 0.0000],
                        [2.8153, 1.5000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        image_dst = kornia.geometry.depth.warp_frame_depth(
            image_src, depth_dst, src_trans_dst, camera_matrix, normalize_points=True
        )
        self.assert_close(image_dst, image_dst_expected, rtol=1e-3, atol=1e-3)

    def test_gradcheck(self, device):
        dtype = torch.float64
        image_src = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)

        depth_dst = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)

        src_trans_dst = torch.rand(1, 4, 4, device=device, dtype=dtype)

        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.warp_frame_depth, (image_src, depth_dst, src_trans_dst, camera_matrix))

    def test_convention_src_trans_dst_moves_the_sampling_point(self, device, dtype):
        # warp_frame_depth reads the depth in the DESTINATION frame and the image in the SOURCE frame, and
        # ``src_trans_dst`` maps destination points into the source frame: with fx = fy = 1 and depth 1, a +1 x
        # translation samples one pixel to the RIGHT, out[u] = image_src[u + 1], and the last column reads the zero
        # padding. The identity transform is the contrast.
        # Snippet used to generate expected: warp_frame_depth(arange(20).view(1, 1, 4, 5), ones, T, K)[0, 0, 0]
        camera_matrix = _k53_warp(device, dtype)
        image_src = torch.arange(20.0, device=device, dtype=dtype).view(1, 1, 4, 5)
        depth_dst = torch.ones(1, 1, 4, 5, device=device, dtype=dtype)
        warped = warp_frame_depth(image_src, depth_dst, _tx_plus_one(device, dtype), camera_matrix)
        self.assert_close(warped[0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0], device=device, dtype=dtype))
        assert (warped - image_src).abs().max().item() > 0.5
        assert torch.equal(warp_frame_depth(image_src, depth_dst, _eye4(device, dtype), camera_matrix), image_src)


class TestDepthWarperConventions(BaseTester):
    """Convention and wart pins for :class:`~kornia.geometry.depth.DepthWarper` and ``depth_warp``."""

    @staticmethod
    def _warper(device, dtype, dst_extrinsics, src_extrinsics):
        """Build a DepthWarper for a 4x5 image and run compute_projection_matrix on the source camera."""
        intrinsics = _k44_warp(device, dtype)
        height = torch.tensor([4], device=device)
        width = torch.tensor([5], device=device)
        warper = DepthWarper(PinholeCamera(intrinsics.clone(), dst_extrinsics.clone(), height, width), 4, 5)
        warper.compute_projection_matrix(PinholeCamera(intrinsics.clone(), src_extrinsics.clone(), height, width))
        return warper

    @staticmethod
    def _random_warp_inputs(device, dtype, batch, seed, rotate):
        """Generate a reproducible random camera pair.

        Everything is drawn on cpu from an explicitly seeded generator and moved to ``device`` afterwards, so
        the fixture is the same tensor on every device and dtype.
        """
        generator = torch.Generator().manual_seed(seed)
        height, width = 4, 5
        fx = torch.rand(batch, generator=generator) * 2 + 1.0
        fy = torch.rand(batch, generator=generator) * 2 + 1.0
        cx = torch.rand(batch, generator=generator) * 2 + 1.0
        cy = torch.rand(batch, generator=generator) * 2 + 1.0
        k3 = torch.zeros(batch, 3, 3)
        k3[:, 0, 0], k3[:, 1, 1], k3[:, 0, 2], k3[:, 1, 2], k3[:, 2, 2] = fx, fy, cx, cy, 1.0
        k4 = torch.eye(4)[None].repeat(batch, 1, 1).contiguous()
        k4[:, :3, :3] = k3
        axis_angle = torch.rand(batch, 3, generator=generator) * 0.2 if rotate else torch.zeros(batch, 3)
        transform = torch.eye(4)[None].repeat(batch, 1, 1).contiguous()
        transform[:, :3, :3] = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)
        transform[:, :3, 3] = torch.rand(batch, 3, generator=generator) * 0.4 - 0.2
        image = torch.rand(batch, 2, height, width, generator=generator)
        depth = torch.rand(batch, 1, height, width, generator=generator) + 1.0
        return (
            image.to(device, dtype),
            depth.to(device, dtype),
            transform.to(device, dtype),
            k3.to(device, dtype),
            k4.to(device, dtype),
            torch.tensor([height] * batch, device=device),
            torch.tensor([width] * batch, device=device),
        )

    def test_convention_projection_matrix_is_k_dst_times_dst_trans_src(self, device, dtype):
        # compute_projection_matrix stores ``K_dst @ (E_dst @ inv(E_src))``: K_dst and E_dst from the constructor's
        # camera, E_src from the argument. E_src is a rotation plus a translation, so the transposed reading
        # (``R^T``) differs. The identity-source arm uses the rotation as the destination because a pure x
        # translation commutes with this K. The inverse is a literal so the pin also runs on mps and in half.
        intrinsics = _k44_warp(device, dtype)
        dst_extrinsics = _tx_plus_one(device, dtype)
        src_extrinsics = torch.tensor(
            [[[0.0, -1.0, 0.0, 0.5], [1.0, 0.0, 0.0, -0.25], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        src_inverse = torch.tensor(
            [[[0.0, 1.0, 0.0, 0.25], [-1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, -2.0], [0.0, 0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        assert torch.equal(src_extrinsics @ src_inverse, _eye4(device, dtype))
        expected = intrinsics @ (dst_extrinsics @ src_inverse)
        warper = self._warper(device, dtype, dst_extrinsics, src_extrinsics)
        assert torch.equal(warper._dst_proj_src, expected)
        transposed_reading = intrinsics @ (dst_extrinsics @ src_extrinsics.transpose(-1, -2))
        assert (transposed_reading - expected).abs().max().item() > 0.5
        identity_source = self._warper(device, dtype, src_extrinsics, _eye4(device, dtype))
        assert torch.equal(identity_source._dst_proj_src, intrinsics @ src_extrinsics)
        assert (src_extrinsics @ intrinsics - intrinsics @ src_extrinsics).abs().max().item() > 0.5

    def test_convention_forward_returns_b_c_h_w_and_align_corners_defaults_to_true(self, device, dtype):
        # forward takes the depth in the reference frame and the patch in the destination frame and returns
        # (B, C, H, W) for any C; ``align_corners`` defaults to True (warp_frame_depth has no such parameter).
        # The identity pair returns the patch, and a +1 x translation moves it, so the shape is backed by values.
        identity = _eye4(device, dtype)
        depth_src = torch.ones(1, 1, 4, 5, device=device, dtype=dtype)
        warper = self._warper(device, dtype, identity, identity)
        assert warper.align_corners is True
        assert (
            DepthWarper(
                PinholeCamera(
                    _k44_warp(device, dtype),
                    identity,
                    torch.tensor([4], device=device),
                    torch.tensor([5], device=device),
                ),
                4,
                5,
            ).align_corners
            is True
        )
        moving = self._warper(device, dtype, _tx_plus_one(device, dtype), identity)
        for channels in (2, 5):
            patch = torch.arange(float(20 * channels), device=device, dtype=dtype).view(1, channels, 4, 5)
            warped = warper(depth_src, patch)
            assert warped.shape == (1, channels, 4, 5)
            self.assert_close(warped, patch)
            shifted = moving(depth_src, patch)
            assert shifted.shape == (1, channels, 4, 5)
            assert (shifted - patch).abs().max().item() > 0.5

    def test_convention_warp_grid_requires_compute_projection_matrix_first(self, device, dtype):
        # DepthWarper is a two-step API: before compute_projection_matrix, warp_grid and forward raise ValueError
        # and compute_subpixel_step raises RuntimeError. Afterwards warp_grid returns the NORMALIZED (B, H, W, 2)
        # grid, with pixel (0, 0) at (-1, -1) and pixel (H - 1, W - 1) at (1, 1).
        identity = _eye4(device, dtype)
        depth_src = torch.ones(1, 1, 4, 5, device=device, dtype=dtype)
        bare = DepthWarper(
            PinholeCamera(
                _k44_warp(device, dtype), identity, torch.tensor([4], device=device), torch.tensor([5], device=device)
            ),
            4,
            5,
        )
        with pytest.raises(ValueError, match="Please, call compute_projection_matrix"):
            bare.warp_grid(depth_src)
        with pytest.raises(ValueError, match="Please, call compute_projection_matrix"):
            bare(depth_src, torch.zeros(1, 2, 4, 5, device=device, dtype=dtype))
        with pytest.raises(RuntimeError, match="but got None Type from the projection matrix"):
            bare.compute_subpixel_step()
        grid = self._warper(device, dtype, identity, identity).warp_grid(depth_src)
        assert grid.shape == (1, 4, 5, 2)
        self.assert_close(grid[0, 0, 0], torch.tensor([-1.0, -1.0], device=device, dtype=dtype))
        self.assert_close(grid[0, -1, -1], torch.tensor([1.0, 1.0], device=device, dtype=dtype))

    def test_convention_depth_warp_is_byte_identical_to_depth_warper(self, device, dtype):
        # depth_warp builds a DepthWarper, calls compute_projection_matrix and forwards, so the results are equal
        # bit for bit, on the integer ramp and on the seeded random camera pair; the warp does move the image.
        intrinsics = _k44_warp(device, dtype)
        dst_extrinsics = _tx_plus_one(device, dtype)
        identity = _eye4(device, dtype)
        height = torch.tensor([4], device=device)
        width = torch.tensor([5], device=device)
        image = torch.arange(20.0, device=device, dtype=dtype).view(1, 1, 4, 5)
        depth_src = torch.ones(1, 1, 4, 5, device=device, dtype=dtype)
        functional = depth_warp(
            PinholeCamera(intrinsics.clone(), dst_extrinsics.clone(), height, width),
            PinholeCamera(intrinsics.clone(), identity.clone(), height, width),
            depth_src,
            image,
            4,
            5,
        )
        assert torch.equal(functional, self._warper(device, dtype, dst_extrinsics, identity)(depth_src, image))
        assert (functional - image).abs().max().item() > 0.5
        img, dep, transform, _, k4, batch_h, batch_w = self._random_warp_inputs(device, dtype, 2, 11, True)
        batch_identity = torch.eye(4, device=device, dtype=dtype)[None].repeat(2, 1, 1).contiguous()
        warper = DepthWarper(PinholeCamera(k4.clone(), transform.clone(), batch_h, batch_w), 4, 5)
        warper.compute_projection_matrix(PinholeCamera(k4.clone(), batch_identity.clone(), batch_h, batch_w))
        by_class = warper(dep, img)
        by_function = depth_warp(
            PinholeCamera(k4.clone(), transform.clone(), batch_h, batch_w),
            PinholeCamera(k4.clone(), batch_identity.clone(), batch_h, batch_w),
            dep,
            img,
            4,
            5,
        )
        assert torch.equal(by_class, by_function)
        assert (by_class - img).abs().max().item() > 0.1

    def test_convention_warp_frame_depth_matches_depth_warper(self, device, dtype):
        # The two APIs compute the same warp through different arithmetic (depth_to_3d_v2 + project_points versus
        # pixel2cam + cam2pixel) and agree to the dtype tolerance on the seeded random camera pairs.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("half precision: the two paths diverge beyond the half-precision tolerances")
        for seed, rotate in ((11, True), (7, False)):
            img, dep, transform, k3, k4, batch_h, batch_w = self._random_warp_inputs(device, dtype, 2, seed, rotate)
            batch_identity = torch.eye(4, device=device, dtype=dtype)[None].repeat(2, 1, 1).contiguous()
            warper = DepthWarper(PinholeCamera(k4.clone(), transform.clone(), batch_h, batch_w), 4, 5)
            warper.compute_projection_matrix(PinholeCamera(k4.clone(), batch_identity.clone(), batch_h, batch_w))
            by_class = warper(dep, img)
            by_function = warp_frame_depth(img, dep, transform, k3)
            self.assert_close(by_function, by_class)
            assert (by_class - img).abs().max().item() > 0.1

    def test_wart_warp_frame_depth_and_depth_warper_name_the_depth_frame_oppositely_4273(self, device, dtype):
        # kornia#4273: the two APIs produce the same warp from OPPOSITE argument names:
        #   warp_frame_depth(image_src=IMG, depth_dst=D, src_trans_dst=T)
        #   DepthWarper(pinhole_dst=cam(T)).compute_projection_matrix(pinhole_src=cam(I))(depth_src=D, patch_dst=IMG)
        # Mapping 'dst' to 'dst' across the two gives the INVERSE warp (the swapped pair). The arguments are
        # keywords on purpose: the #4273 fix is a rename, so this pin fails when it lands.
        image = torch.arange(20.0, device=device, dtype=dtype).view(1, 1, 4, 5)
        depth = torch.ones(1, 1, 4, 5, device=device, dtype=dtype)
        identity = _eye4(device, dtype)
        tx_plus_one = _tx_plus_one(device, dtype)
        intrinsics = _k44_warp(device, dtype)
        height = torch.tensor([4], device=device)
        width = torch.tensor([5], device=device)
        shifted_right = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0], device=device, dtype=dtype)
        by_function = warp_frame_depth(
            image_src=image,
            depth_dst=depth,
            src_trans_dst=tx_plus_one,
            camera_matrix=_k53_warp(device, dtype),
        )[0, 0, 0]
        moving = DepthWarper(
            pinhole_dst=PinholeCamera(intrinsics.clone(), tx_plus_one.clone(), height, width), height=4, width=5
        )
        moving.compute_projection_matrix(pinhole_src=PinholeCamera(intrinsics.clone(), identity.clone(), height, width))
        by_class = moving(depth_src=depth, patch_dst=image)[0, 0, 0]
        self.assert_close(by_function, shifted_right)
        self.assert_close(by_class, shifted_right)
        naive = DepthWarper(
            pinhole_dst=PinholeCamera(intrinsics.clone(), identity.clone(), height, width), height=4, width=5
        )
        naive.compute_projection_matrix(
            pinhole_src=PinholeCamera(intrinsics.clone(), tx_plus_one.clone(), height, width)
        )
        swapped = naive(depth_src=depth, patch_dst=image)[0, 0, 0]
        self.assert_close(swapped, torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0], device=device, dtype=dtype))
        assert (by_class - swapped).abs().max().item() > 0.5

    def test_wart_warp_frame_depth_and_depth_warper_split_at_zero_transformed_depth_4267(self, device, dtype):
        # kornia#4267: the two warps guard z = 0 differently. With K = I, unit depth and a (+1, 0, -1) move, every
        # pixel lands at z = 0. warp_frame_depth skips the divide (abs(z) <= 1e-8) and returns the image shifted by
        # one column; DepthWarper divides by z + 1e-12, sending the pixel to a coordinate of order 1e12. At z = 0.5
        # the two agree. The pin never samples through the 1e12 grid: torch 2.5.1's aarch64 CPU grid_sample
        # segfaults on such coordinates (pytorch/pytorch#24823). Delete when #4267 settles one z = 0 policy.
        image = torch.arange(1.0, 7.0, device=device, dtype=dtype).view(1, 1, 2, 3)
        depth = torch.ones(1, 1, 2, 3, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype)[None]
        intrinsics = _eye4(device, dtype)
        height = torch.tensor([2], device=device)
        width = torch.tensor([3], device=device)

        def warper_for(transform):
            warper = DepthWarper(PinholeCamera(intrinsics.clone(), transform.clone(), height, width), 2, 3)
            warper.compute_projection_matrix(PinholeCamera(intrinsics.clone(), _eye4(device, dtype), height, width))
            return warper

        to_zero_depth = _eye4(device, dtype)
        to_zero_depth[0, 0, 3] = 1.0
        to_zero_depth[0, 2, 3] = -1.0
        by_function = warp_frame_depth(image, depth, to_zero_depth, camera_matrix)
        self.assert_close(by_function, torch.tensor([[[[2.0, 3.0, 0.0], [5.0, 6.0, 0.0]]]], device=device, dtype=dtype))
        singular = warper_for(to_zero_depth)
        # compared in float32: a float16 1e9 is itself inf, and inf > inf is False
        assert (singular.warp_grid(depth)[..., 0].abs().to(torch.float32) > 1.0e9).all()
        if dtype in (torch.float32, torch.float64):
            half_way = to_zero_depth.clone()
            half_way[0, 2, 3] = -0.5
            self.assert_close(
                warper_for(half_way)(depth, image), warp_frame_depth(image, depth, half_way, camera_matrix)
            )


class TestDepthFromDisparity(BaseTester):
    @pytest.mark.parametrize("baseline_kind", ["int", "float", "scalar_tensor", "tensor"])
    @pytest.mark.parametrize("focal_kind", ["int", "float", "scalar_tensor", "tensor"])
    @pytest.mark.parametrize("batched", [False, True])
    def test_scalar_camera_parameters_4272(self, baseline_kind, focal_kind, batched, device, dtype):
        def parameter(value, kind):
            if kind == "int":
                return int(value)
            if kind == "float":
                return float(value)
            return torch.tensor(value if kind == "scalar_tensor" else [value], device=device, dtype=dtype)

        disparity = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], device=device, dtype=dtype)
        expected = torch.tensor([[60.0, 30.0, 15.0], [7.5, 3.75, 1.875]], device=device, dtype=dtype)
        if batched:
            disparity = disparity.expand(2, 1, 2, 3)
            expected = expected.expand(2, 1, 2, 3)
        depth = kornia.geometry.depth.depth_from_disparity(
            disparity, parameter(2.0, baseline_kind), parameter(30.0, focal_kind)
        )
        assert depth.shape == disparity.shape
        assert depth.dtype == dtype
        assert depth.device == disparity.device
        self.assert_close(depth, expected)

    @pytest.mark.parametrize("parameter_name", ["baseline", "focal"])
    def test_scalar_camera_parameters_reject_invalid(self, parameter_name, device, dtype):
        disparity = torch.ones(2, 1, 3, 4, device=device, dtype=dtype)
        parameters = {"baseline": 1.0, "focal": 1.0}
        # (2,) would be a valid per-batch-element parameter for this batch-2 disparity, so the mismatched
        # sizes are 0 and 3; (1, 1) has the wrong rank for either form.
        for shape in [(0,), (3,), (1, 1)]:
            parameters[parameter_name] = torch.ones(shape, device=device, dtype=dtype)
            with pytest.raises(ShapeError):
                kornia.geometry.depth.depth_from_disparity(disparity, **parameters)
        for value in ["1", 1j, True]:
            parameters[parameter_name] = value
            with pytest.raises(BaseError, match=f"Input {parameter_name}"):
                kornia.geometry.depth.depth_from_disparity(disparity, **parameters)

    def test_smoke(self, device, dtype):
        disparity = 2 * torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype
        )

        baseline = torch.tensor([1.0], device=device, dtype=dtype)
        focal = torch.tensor([1.0], device=device, dtype=dtype)

        depth_expected = torch.tensor(
            [
                [
                    [
                        [0.5000, 0.5000, 0.5000],
                        [0.5000, 0.5000, 0.5000],
                        [0.5000, 0.5000, 0.5000],
                        [0.5000, 0.5000, 0.5000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        depth = kornia.geometry.depth.depth_from_disparity(disparity, baseline, focal)
        self.assert_close(depth, depth_expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_cardinality(self, batch_size, device, dtype):
        disparity = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        baseline = torch.rand(1, device=device, dtype=dtype)
        focal = torch.rand(1, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_from_disparity(disparity, baseline, focal)
        assert points3d.shape == (batch_size, 1, 3, 4)

    @pytest.mark.parametrize("shape", [(1, 1, 3, 4), (4, 1, 3, 4), (4, 3, 4), (1, 3, 4), (3, 4)])
    def test_shapes(self, shape, device, dtype):
        disparity = torch.randn(shape, device=device, dtype=dtype)
        baseline = torch.rand(1, device=device, dtype=dtype)
        focal = torch.rand(1, device=device, dtype=dtype)

        points3d = kornia.geometry.depth.depth_from_disparity(disparity, baseline, focal)
        assert points3d.shape == shape

    @pytest.mark.parametrize("parameter_shape", [(), (1,), (2,)])
    def test_gradcheck(self, device, parameter_shape):
        # generate input data; a batch of 2 so the (2,) case is a per-batch-element parameter
        disparity = torch.rand(2, 1, 3, 4, device=device, dtype=torch.float64)

        baseline = torch.rand(parameter_shape, device=device, dtype=torch.float64)

        focal = torch.rand(parameter_shape, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.depth_from_disparity, (disparity, baseline, focal))

    def test_wart_zero_disparity_gives_a_finite_depth_4272(self, device, dtype):
        # kornia#4272: depth_from_disparity computes ``baseline * focal / (disparity + 1e-8)``, so a zero
        # disparity (the usual 'no match' fill) returns 5e9 for baseline 0.5 and focal 100, a function of the
        # epsilon rather than inf. In float16 the 1e-8 rounds to 0 and the result is inf. Disparity 2 gives 25.
        # Delete when #4272 is fixed.
        disparity = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)
        depth = depth_from_disparity(disparity, 0.5, 100.0)
        if dtype == torch.float16:
            assert torch.isinf(depth).all()
            return
        assert torch.isfinite(depth).all()
        self.assert_close(depth, torch.full((1, 1, 1, 1), 5.0e9, device=device, dtype=dtype))
        self.assert_close(
            depth_from_disparity(torch.full((1, 1, 1, 1), 2.0, device=device, dtype=dtype), 0.5, 100.0),
            torch.full((1, 1, 1, 1), 25.0, device=device, dtype=dtype),
        )

    @pytest.mark.parametrize("disparity_shape", [(2, 1, 2, 3), (2, 2, 3)])
    @pytest.mark.parametrize("per_batch", ["baseline", "focal", "both"])
    def test_convention_per_batch_camera_parameters_4272(self, disparity_shape, per_batch, device, dtype):
        # kornia#4272: a (B,) baseline or focal pairs element b with disparity[b]. The expected result is built one
        # sample at a time with (1,) parameters, and the two samples use different camera values, so a swapped or
        # shared pairing fails; (B, 1, H, W) and (B, H, W) disparities are covered.
        disparity = torch.arange(1.0, 13.0, device=device, dtype=dtype).reshape(disparity_shape)
        baselines = torch.tensor([0.5, 2.0], device=device, dtype=dtype)
        focals = torch.tensor([100.0, 30.0], device=device, dtype=dtype)
        shared_baseline = torch.tensor([0.5], device=device, dtype=dtype)
        shared_focal = torch.tensor([100.0], device=device, dtype=dtype)

        baseline = baselines if per_batch in ("baseline", "both") else shared_baseline
        focal = focals if per_batch in ("focal", "both") else shared_focal
        depth = depth_from_disparity(disparity, baseline, focal)

        per_sample = torch.cat(
            [
                depth_from_disparity(
                    disparity[b : b + 1],
                    baselines[b : b + 1] if per_batch in ("baseline", "both") else shared_baseline,
                    focals[b : b + 1] if per_batch in ("focal", "both") else shared_focal,
                )
                for b in range(2)
            ]
        )
        assert depth.shape == disparity.shape
        self.assert_close(depth, per_sample, atol=0.0, rtol=0.0)
        swapped = depth_from_disparity(disparity, baseline.flip(0), focal.flip(0))
        assert not torch.allclose(depth, swapped)

    def test_convention_per_batch_camera_parameters_must_match_the_batch_4272(self, device, dtype):
        # A (B,) parameter is only meaningful against a batch of the same size. A mismatched size raises, and so
        # does a (B,) parameter for a disparity with no batch axis, where it would otherwise broadcast against
        # the width instead. The size-1 and scalar forms are unaffected, and an empty batch stays empty.
        disparity = torch.ones(2, 1, 2, 3, device=device, dtype=dtype)
        three = torch.ones(3, device=device, dtype=dtype)
        with pytest.raises(ShapeError, match="expected 2, got 3"):
            depth_from_disparity(disparity, three, 100.0)
        with pytest.raises(ShapeError, match="expected 2, got 3"):
            depth_from_disparity(disparity, 0.5, three)
        with pytest.raises(ShapeError, match="expected 1, got 3"):
            depth_from_disparity(torch.ones(2, 3, device=device, dtype=dtype), three, 100.0)

        assert depth_from_disparity(disparity, 1.0, 100.0).shape == (2, 1, 2, 3)
        assert depth_from_disparity(disparity, torch.tensor([0.5], device=device, dtype=dtype), 100.0).shape == (
            2,
            1,
            2,
            3,
        )
        empty = torch.ones(0, 1, 2, 3, device=device, dtype=dtype)
        nothing = torch.ones(0, device=device, dtype=dtype)
        assert depth_from_disparity(empty, nothing, nothing).shape == (0, 1, 2, 3)


class TestDepthFromPlaneEquation(BaseTester):
    def test_smoke(self, device, dtype):
        B = 2
        N = 10
        plane_normals = torch.randn(B, 3, device=device, dtype=dtype)
        plane_offsets = torch.randn(B, 1, device=device, dtype=dtype)
        points_uv = torch.randn(B, N, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

        depth = kornia.geometry.depth.depth_from_plane_equation(plane_normals, plane_offsets, points_uv, camera_matrix)
        assert depth.shape == (B, N), f"Expected depth shape to be ({B}, {N}), but got {depth.shape}"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shapes(self, batch_size, device, dtype):
        B = batch_size
        N = 10
        plane_normals = torch.randn(B, 3, device=device, dtype=dtype)
        plane_offsets = torch.randn(B, 1, device=device, dtype=dtype)
        points_uv = torch.randn(B, N, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

        depth = kornia.geometry.depth.depth_from_plane_equation(plane_normals, plane_offsets, points_uv, camera_matrix)
        assert depth.shape == (B, N), f"Expected depth shape to be ({B}, {N}), but got {depth.shape}"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shapes_broadcast(self, batch_size, device, dtype):
        B = batch_size
        N = 10
        plane_normals = torch.randn(1, 3, device=device, dtype=dtype)  # Broadcasting plane normals
        plane_offsets = torch.randn(1, 1, device=device, dtype=dtype)
        points_uv = torch.randn(B, N, 2, device=device, dtype=dtype)
        camera_matrix = torch.eye(3, device=device, dtype=dtype)

        depth = kornia.geometry.depth.depth_from_plane_equation(
            plane_normals.expand(B, -1), plane_offsets.expand(B, -1), points_uv, camera_matrix.expand(B, -1, -1)
        )
        assert depth.shape == (B, N), f"Expected depth shape to be ({B}, {N}), but got {depth.shape}"

    def test_simple(self, device, dtype):
        """Test the function with a simple plane equation to verify numerical correctness.

        Plane equation: z = 2 (plane normal [0, 0, 1], offset 2)
        Expected depth for any point is 2.
        """
        # Define plane parameters
        plane_normals = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)  # Shape: (B, 3)
        plane_offsets = torch.tensor([[2.0]], device=device, dtype=dtype)

        # Define pixel coordinates
        points_uv = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )  # Shape: (B, N, 2)

        # Camera intrinsic matrix (identity)
        camera_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # Shape: (B, 3, 3)

        # Expected depth values
        depth_expected = torch.tensor([[2.0, 2.0, 2.0, 2.0]], device=device, dtype=dtype)  # Shape: (B, N)

        # Compute depth
        depth = kornia.geometry.depth.depth_from_plane_equation(plane_normals, plane_offsets, points_uv, camera_matrix)

        # Assert that the computed depth matches the expected depth
        self.assert_close(depth, depth_expected, rtol=1e-6, atol=1e-6)

    def test_grazing_ray_is_finite(self, device, dtype):
        """A ray exactly parallel to the plane must take the epsilon guard.

        The guard was `eps * sign(denom)`, and `sign` is zero at zero, so at the
        exact singularity the epsilon was multiplied away and the depth came
        back as inf. A grazing ray is not exotic: it is every pixel on the
        horizon of a ground plane.
        """
        camera_matrix = torch.tensor(
            [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )[None]
        # The principal point's ray is (0, 0, 1); this normal is perpendicular
        # to it, so the ray-plane dot product is exactly zero.
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype)
        plane_offsets = torch.tensor([[2.0]], device=device, dtype=dtype)
        points_uv = torch.tensor([[[4.0, 3.0]]], device=device, dtype=dtype)

        # In float16 the default eps is floored at 2**-24, and 2 / 2**-24 is
        # outside float16's finite range. Ask for an epsilon large enough that
        # the guarded depth fits the dtype.
        eps = max(1e-8, float(torch.finfo(dtype).eps))
        depth = kornia.geometry.depth.depth_from_plane_equation(
            plane_normals, plane_offsets, points_uv, camera_matrix, eps=eps
        )
        assert torch.isfinite(depth).all(), f"grazing ray returned {depth.tolist()}"

    def test_grazing_ray_default_eps_float16(self, device):
        # kornia#4803: the default eps=1e-8 rounds to zero in float16, so the guard replaced a zero
        # denominator with zero and returned inf. It is floored at float16's smallest subnormal instead.
        dtype = torch.float16
        camera_matrix = torch.tensor(
            [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )[None]
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype)
        plane_offsets = torch.tensor([[1e-4]], device=device, dtype=dtype)
        points_uv = torch.tensor([[[4.0, 3.0]]], device=device, dtype=dtype)
        depth = kornia.geometry.depth.depth_from_plane_equation(plane_normals, plane_offsets, points_uv, camera_matrix)
        expected = plane_offsets / torch.full_like(plane_offsets, 5.960464477539063e-08)
        assert torch.isfinite(depth).all(), f"grazing ray returned {depth.tolist()}"
        self.assert_close(depth, expected)

    def test_grazing_ray_float16_floor_limits(self, device):
        # kornia#4803: the float16 floor applies to a positive eps only; eps = 0 keeps the unguarded division. A plane
        # through the camera centre gives 0 / 2**-24 = 0 (it was 0 / 0 = nan), and an offset above 65504 * 2**-24
        # still overflows float16.
        dtype = torch.float16
        camera_matrix = torch.tensor(
            [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )[None]
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype)
        points_uv = torch.tensor([[[4.0, 3.0]]], device=device, dtype=dtype)

        def depth(offset: float, **kwargs: float) -> torch.Tensor:
            offsets = torch.tensor([[offset]], device=device, dtype=dtype)
            return depth_from_plane_equation(plane_normals, offsets, points_uv, camera_matrix, **kwargs)

        assert depth(1e-4, eps=0.0).isposinf().all()
        assert depth(1e-4, eps=-1e-3).isposinf().all()
        self.assert_close(depth(0.0), torch.zeros(1, 1, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        assert depth(1e-2).isposinf().all()

    def test_small_denominators_keep_their_sign(self, device, dtype):
        """The guard already handled small non-zero denominators; keep that.

        Two rays whose dot products differ only in sign must come back with
        depths of the same magnitude and opposite signs, rather than both
        collapsing onto one branch.
        """
        camera_matrix = torch.eye(3, device=device, dtype=dtype)[None].repeat(2, 1, 1)
        # As above: 1e-8 and eps/4 both round to zero in float16, which would
        # turn this into the grazing-ray case and lose the sign under test.
        eps = max(1e-8, float(torch.finfo(dtype).eps))
        half = eps / 4
        # Ray (0, 0, 1) for both; the normal's z carries the whole dot product.
        plane_normals = torch.tensor([[0.0, 0.0, half], [0.0, 0.0, -half]], device=device, dtype=dtype)
        plane_offsets = torch.tensor([[2.0], [2.0]], device=device, dtype=dtype)
        points_uv = torch.zeros(2, 1, 2, device=device, dtype=dtype)

        depth = kornia.geometry.depth.depth_from_plane_equation(
            plane_normals, plane_offsets, points_uv, camera_matrix, eps=eps
        )
        assert torch.isfinite(depth).all()
        self.assert_close(depth[0], -depth[1])

    def test_gradcheck(self, device):
        B = 2
        N = 5
        plane_normals = torch.rand(B, 3, device=device, dtype=torch.float64, requires_grad=True)
        plane_offsets = torch.rand(B, 1, device=device, dtype=torch.float64, requires_grad=True)
        points_uv = torch.rand(B, N, 2, device=device, dtype=torch.float64, requires_grad=True)
        camera_matrix = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0).repeat(B, 1, 1)
        camera_matrix.requires_grad_()

        # Perform gradient check
        self.gradcheck(
            kornia.geometry.depth.depth_from_plane_equation,
            (plane_normals, plane_offsets, points_uv, camera_matrix),
            eps=1e-6,
            atol=1e-4,
        )

    def test_convention_plane_z_equals_two_gives_depth_two(self, device, dtype):
        # The plane is in Hessian form ``n . X = d`` in the CAMERA frame, and the pixels are pixel coordinates the
        # function normalizes itself: n = (0, 0, 1), d = 2 is the plane z = 2, so every pixel has depth 2, and the
        # output is (B, N). The tilted normal makes two pixels disagree, so a reading where the offset alone sets
        # the depth fails.
        # Snippet used to generate expected: depth_from_plane_equation(n, offsets, pixels, K)
        camera_matrix = _k_asymmetric(device, dtype)
        offsets = torch.tensor([[2.0]], device=device, dtype=dtype)
        pixels = torch.tensor([[[4.0, 3.0], [29.0, 53.0]]], device=device, dtype=dtype)
        depth = depth_from_plane_equation(
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype), offsets, pixels, camera_matrix
        )
        assert depth.shape == (1, 2)
        self.assert_close(depth, torch.full((1, 2), 2.0, device=device, dtype=dtype), atol=0.0, rtol=0.0)
        tilted = depth_from_plane_equation(
            torch.tensor([[0.0, 0.4472135954999579, 0.8944271909999159]], device=device, dtype=dtype),
            offsets,
            torch.tensor([[[4.0, 3.0], [4.0, 103.0]]], device=device, dtype=dtype),
            camera_matrix,
        )
        self.assert_close(tilted, torch.tensor([[2.2360679774997896, 1.4907119849998598]], device=device, dtype=dtype))

    def test_convention_depth_from_plane_equation_clamps_a_tiny_denominator(self, device, dtype):
        # A ray-plane dot product inside (-eps, eps) is clamped to +/- eps with its sign kept, so a nearly grazing
        # ray returns a large SIGNED finite depth (2 / 1e-8 = 2e8); eps is passed explicitly, and eps = 1e-6 gives
        # 2e6. An exact zero is covered by the #4280 pin below.
        # Snippet used to generate expected: depth_from_plane_equation([[0, 1, 2.384185791015625e-09]], [[2.0]],
        # [[[4.0, 3.0]]], K, eps=1e-8) and the negated normal
        if dtype == torch.float16:
            pytest.skip("float16: eps is floored at 2**-24 and 2 / 2**-24 is past the float16 range: the depth is inf")
        camera_matrix = _k_asymmetric(device, dtype)
        offsets = torch.tensor([[2.0]], device=device, dtype=dtype)
        principal_point = torch.tensor([[[4.0, 3.0]]], device=device, dtype=dtype)
        normal = torch.tensor([[0.0, 1.0, 2.384185791015625e-09]], device=device, dtype=dtype)
        positive = depth_from_plane_equation(normal, offsets, principal_point, camera_matrix, eps=1e-8)
        negative = depth_from_plane_equation(-normal, offsets, principal_point, camera_matrix, eps=1e-8)
        assert torch.isfinite(positive).all()
        assert torch.isfinite(negative).all()
        self.assert_close(positive, torch.full((1, 1), 2.0e8, device=device, dtype=dtype))
        self.assert_close(negative, torch.full((1, 1), -2.0e8, device=device, dtype=dtype))
        wider = depth_from_plane_equation(normal, offsets, principal_point, camera_matrix, eps=1e-6)
        self.assert_close(wider, torch.full((1, 1), 2.0e6, device=device, dtype=dtype))

    def test_convention_depth_from_plane_equation_clamps_the_singularity_4280(self, device, dtype):
        # kornia#4280: an exactly zero denominator is replaced by POSITIVE eps, so the grazing ray returns +2e8 for
        # either sign of the normal.
        if dtype == torch.float16:
            pytest.skip("float16: 2e8 overflows the float16 range")
        camera_matrix = _k_asymmetric(device, dtype)
        for sign in (1.0, -1.0):
            grazing = depth_from_plane_equation(
                torch.tensor([[0.0, sign, 0.0]], device=device, dtype=dtype),
                torch.tensor([[2.0]], device=device, dtype=dtype),
                torch.tensor([[[4.0, 3.0]]], device=device, dtype=dtype),
                camera_matrix,
            )
            assert torch.isfinite(grazing).all()
            self.assert_close(grazing, torch.full((1, 1), 2.0e8, device=device, dtype=dtype))
