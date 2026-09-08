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

    def test_unproject_meshgrid(self, device, dtype):
        # TODO: implement me with batch
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        grid = kornia.geometry.unproject_meshgrid(3, 4, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (2, 3, 4, 3)
        # test for now that the grid is correct and have homogeneous coords
        self.assert_close(grid[..., 2], torch.ones_like(grid[..., 2]))

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
        # Convention pin: pixel (0, 0) unprojects as
        # ((0 - cx) d / fx, (0 - cy) d / fy, d). kornia's pixel grid puts the centre of the first pixel at the
        # INTEGER coordinate 0 (the OpenCV convention that create_meshgrid already pins in
        # tests/geometry/test_conversions.py), and ``depth`` here is the camera-frame z, not a ray length.
        # A half-pixel (COLMAP) grid, whose first pixel centre is 0.5, would give
        # ((0.5 - 4) * 2 / 100, (0.5 - 3) * 2 / 100, 2) = (-0.07, -0.05, 2) instead -- a 0.01 gap in x and y,
        # far outside every dtype tolerance used here. cx = 4 != cy = 3 and H = 2 != W = 3, so a transposed
        # reading of the grid also changes the literal. depth_to_3d_v2 answers the same question in the
        # (B, H, W, 3) layout and gives the same triple.
        # Snippet used to generate expected: depth_to_3d(full((1, 1, 2, 3), 2.0), K)[0, :, 0, 0] executed
        # 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> cpu float32 and float64
        # [-0.07999999821186066, -0.05999999865889549, 2.0], cpu float16 [-0.08001708984375, -0.05999755859375,
        # 2.0], cpu bfloat16 [-0.080078125, -0.06005859375, 2.0]; mps float32 and float16 reproduce their cpu
        # cells exactly.
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
        # Convention pin: the two functions compute the same points in two
        # layouts -- (B, 3, H, W) for depth_to_3d, (B, H, W, 3) for depth_to_3d_v2 -- and are byte-identical
        # after the permutation, under an asymmetric camera (fx = 100 != fy = 50, cx = 4 != cy = 3), H = 3 != W
        # = 5 and a depth that varies at every pixel. The permutation is load-bearing rather than decorative:
        # the two flattened buffers are NOT equal, so a comparison that skipped the permute would fail here.
        # The W = 1 regression (kornia#4278) is pinned below too.
        # Snippet used to generate expected: torch.equal(depth_to_3d(d, K).permute(0, 2, 3, 1),
        # depth_to_3d_v2(d[:, 0], K)) and torch.equal(v1.reshape(-1), v2.reshape(-1)) on the depth ramp 1..15
        # executed 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> True and False respectively, on cpu for
        # float32, float64, float16 and bfloat16 and on mps for float32 and float16.
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
        # Regression for #4278: a single column retains its width axis and agrees across layouts.
        camera_matrix = _k_asymmetric(device, dtype)
        depth = torch.full((1, 1, 3, 1), 2.0, device=device, dtype=dtype)
        v2 = depth_to_3d_v2(depth[:, 0], camera_matrix)
        assert v2.shape == (1, 3, 1, 3)
        assert torch.equal(depth_to_3d(depth, camera_matrix).permute(0, 2, 3, 1), v2)


class TestUnprojectMeshgrid(BaseTester):
    def test_convention_returns_b_h_w_3_and_equals_depth_to_3d_v2_at_depth_one(self, device, dtype):
        # Convention pin: unproject_meshgrid is depth_to_3d_v2's
        # cache: it returns the per-pixel ray through each pixel at depth 1 in the (*, H, W, 3) layout, so
        # multiplying it by a depth map reproduces depth_to_3d_v2 exactly. The grid is not constant -- pixel
        # (0, 0) and pixel (1, 2) differ -- so the equality is not the trivial one.
        # The asymmetric arm varies ONE intrinsic: with fy = 50 instead of 100 only the y component of the
        # pixel-(1, 0) ray changes, which is what fixes fy as the divisor of the ROW index.
        # Snippet used to generate expected: unproject_meshgrid(2, 3, K, device=..., dtype=...) and
        # torch.equal(that, depth_to_3d_v2(ones(1, 2, 3), K)) executed 2026-09-06 at commit 1a96bfd1
        # (torch 2.14.0) -> shape (1, 2, 3, 3), pixel (0, 0) [-0.03999999910593033, -0.029999999329447746, 1.0]
        # and True, on cpu for float32, float64, float16 and bfloat16 and on mps for float32 and float16; with
        # fy = 50, pixel (0, 0) [-0.04, -0.06, 1.0] and pixel (1, 0) [-0.04, -0.04, 1.0].
        camera_matrix = _k_asymmetric(device, dtype)
        grid = unproject_meshgrid(2, 3, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (1, 2, 3, 3)
        self.assert_close(grid[0, 0, 0], torch.tensor([-0.04, -0.03, 1.0], device=device, dtype=dtype))
        self.assert_close(grid[0, 1, 2], torch.tensor([-0.02, -0.02, 1.0], device=device, dtype=dtype))
        assert torch.equal(grid, depth_to_3d_v2(torch.ones(1, 2, 3, device=device, dtype=dtype), camera_matrix))
        asymmetric = unproject_meshgrid(2, 3, _k_asymmetric(device, dtype, fy=50.0), device=device, dtype=dtype)
        self.assert_close(asymmetric[0, 0, 0], torch.tensor([-0.04, -0.06, 1.0], device=device, dtype=dtype))
        self.assert_close(asymmetric[0, 1, 0], torch.tensor([-0.04, -0.04, 1.0], device=device, dtype=dtype))

    def test_wart_unproject_meshgrid_rejects_unbatched_intrinsics_4271(self, device, dtype):
        # Wart pin for kornia#4271: the guard is written ["*", "3", "3"],
        # which admits a bare (3, 3) -- but the body then does ``camera_matrix[:, None, None]``, which on a 2-D
        # tensor produces a (3, 1, 1, 3) and trips a LATER check whose message describes that shape rather than
        # the one the caller passed. Singleton extra axes pass through, while non-singleton ones can
        # either fail or accidentally broadcast against the width. They are not a supported batch layout.
        # Snippet used to generate expected: unproject_meshgrid(2, 3, K, device=..., dtype=...) at four K
        # shapes, executed 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> (3, 3): ShapeError("Shape mismatch at
        # dimension 0: expected 3, got 1. ... Actual shape: [3, 1, 1, 3]"); (1, 3, 3): (1, 2, 3, 3);
        # (2, 1, 3, 3): (2, 1, 2, 3, 3); (2, 2, 3, 3): RuntimeError("The size of tensor a (3) must match the
        # size of tensor b (2) at non-singleton dimension 3"). All four on cpu for float32, float64, float16
        # and bfloat16 and on mps for float32 and float16.
        # Pins the CURRENT behavior; NOT a contract; delete when #4271 is repaired.
        camera_matrix = _k_asymmetric(device, dtype)
        with pytest.raises(ShapeError) as errinfo:
            unproject_meshgrid(2, 3, camera_matrix[0], device=device, dtype=dtype)
        assert "[3, 1, 1, 3]" in str(errinfo.value)
        assert unproject_meshgrid(2, 3, camera_matrix, device=device, dtype=dtype).shape == (1, 2, 3, 3)
        singleton = camera_matrix.expand(2, 3, 3).unsqueeze(1).contiguous()
        assert singleton.shape == (2, 1, 3, 3)
        assert unproject_meshgrid(2, 3, singleton, device=device, dtype=dtype).shape == (2, 1, 2, 3, 3)
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            unproject_meshgrid(2, 3, singleton.expand(2, 2, 3, 3).contiguous(), device=device, dtype=dtype)

    def test_wart_unproject_meshgrid_extra_camera_axis_broadcasts_over_columns_4271(self, device, dtype):
        camera_matrix = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3, 3).clone()
        camera_matrix[:, :, 0, 2] = torch.tensor([0.0, 2.0, 4.0], device=device, dtype=dtype)
        grid = unproject_meshgrid(2, 3, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (2, 1, 2, 3, 3)
        # Each column uses a different cx: u - cx = [0, -1, -2].
        expected = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-2.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-2.0, 1.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(grid, expected.expand(2, 1, 2, 3, 3))

    @pytest.mark.xfail(
        strict=True, reason="kornia#4271: the guard reports a [3, 1, 1, 3] shape the caller never passed"
    )
    def test_convention_unproject_meshgrid_error_names_the_shape_the_caller_passed_4271(self, device, dtype):
        # Intended contract, asserted as a strict xfail so the repair makes it XPASS and forces this mark out:
        # whatever unproject_meshgrid decides about (3, 3), the error a caller sees has to name the shape the
        # caller actually passed. Today it names a (3, 1, 1, 3) built three lines into the body.
        # Settled by #4271's Expected section, which calls exactly this -- "the guard should reject a bare
        # (3, 3) at the guard rather than three lines later with a message about a (3, 1, 1, 3) shape the caller
        # never passed" -- a focused fix, welcome as a PR. #4271 deliberately does NOT settle whether (3, 3) is
        # then accepted, so this pin asserts only the message, not the acceptance.
        camera_matrix = _k_asymmetric(device, dtype)
        with pytest.raises(ShapeError) as errinfo:
            unproject_meshgrid(2, 3, camera_matrix[0], device=device, dtype=dtype)
        assert "[3, 3]" in str(errinfo.value)
        assert "[3, 1, 1, 3]" not in str(errinfo.value)


class TestDepthToNormals(BaseTester):
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

    def test_convention_normals_face_the_camera_and_x_tracks_the_column(self, device, dtype):
        # Convention pin: depth_to_normals takes
        # the cross product of the spatial gradients of the unprojected point cloud in the order dx x dy, so a
        # fronto-parallel plane gets the unit normal (0, 0, 1) -- +z points AWAY from the camera, along the
        # viewing direction, not back towards it. The two ramps fix which axis is which: a depth that grows with
        # the COLUMN tilts the normal towards -x, a depth that grows with the ROW tilts it towards -y. The
        # magnitudes are dtype-dependent (float32 -0.99995, float16 -1.0), so this pin asserts the sign pattern
        # with separators far from every measured value, and only the flat plane as an exact triple.
        # The map is 3 x 4 with fx = 100 != fy = 50, so H != W and the two focal lengths differ: a transposed
        # depth map does not even have the right shape here, and a reading that swapped fx and fy would move the
        # ramp normals.
        # Snippet used to generate expected: depth_to_normals(depth, K)[0, :, 1, 1] on the 3 x 4 map executed
        # 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> flat plane exactly [0.0, 0.0, 1.0] (cpu
        # float32/float16/bfloat16) and [0.0, -0.0, 1.0] (cpu float64, mps float32); column ramp
        # [-0.9999499917030334, 0.0, -0.009999499656260014] on cpu float32, [-1.0, 0.0, -0.0099945068359375] on
        # cpu float16, [-1.0, 0.0, -0.010009765625] on cpu bfloat16 and [-0.9999499917030334, 3.49e-08,
        # -0.009999497793614864] on mps float32; row ramp [0.0, -1.0, 0.0] on cpu and
        # [0.0, -1.0, -4.656612873077393e-10] on mps float32.
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
        # Convention pin: warp_frame_depth reads the depth in
        # the DESTINATION frame and the image in the SOURCE frame, and ``src_trans_dst`` maps destination-frame
        # points into the source frame. With fx = fy = 1 and depth 1, a +1 translation in x therefore samples
        # image_src one pixel to the RIGHT of each destination pixel: out[u] = image_src[u + 1], and the last
        # column samples outside the image, which the baked ``padding_mode="zeros"`` fills with 0.
        # An identity transform is frame-invariant and proves nothing about the direction, so it is used only as
        # the contrast: it returns the input byte for byte, while the +1 transform moves it by up to 19.0.
        # Snippet used to generate expected: warp_frame_depth(arange(20).view(1, 1, 4, 5), ones(1, 1, 4, 5), T,
        # K)[0, 0, 0] with T = eye(4) except T[0, 0, 3] = 1, executed 2026-09-06 at commit 1a96bfd1
        # (torch 2.14.0) -> [1.0, 2.0, 3.0, 4.0, 0.0] on cpu for float32, float64, float16 and bfloat16 and on
        # mps for float32 and float16; the identity call is torch.equal to the input in every one of those
        # cells, and the +1 call deviates from the input by 19.0.
        camera_matrix = _k53_warp(device, dtype)
        image_src = torch.arange(20.0, device=device, dtype=dtype).view(1, 1, 4, 5)
        depth_dst = torch.ones(1, 1, 4, 5, device=device, dtype=dtype)
        warped = warp_frame_depth(image_src, depth_dst, _tx_plus_one(device, dtype), camera_matrix)
        self.assert_close(warped[0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0], device=device, dtype=dtype))
        assert (warped - image_src).abs().max().item() > 0.5
        assert torch.equal(warp_frame_depth(image_src, depth_dst, _eye4(device, dtype), camera_matrix), image_src)

    def test_wart_warp_frame_depth_rejects_an_empty_batch_4281(self, device, dtype):
        # Wart pin for kornia#4281: an empty batch raises a bare
        # ZeroDivisionError from normalize_pixel_coordinates rather than returning an empty result, although
        # every shape guard on the way in accepts B = 0 and depth_to_3d -- the same computation in the other
        # layout -- handles it. kornia's degenerate-shape convention is empty in, empty out.
        # Snippet used to generate expected: warp_frame_depth(zeros(0, 2, 4, 5), ones(0, 1, 4, 5),
        # zeros(0, 4, 4), zeros(0, 3, 3)) executed 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) ->
        # ZeroDivisionError("integer division or modulo by zero") on cpu for float32, float64, float16 and
        # bfloat16 and on mps for float32 and float16; depth_to_3d(zeros(0, 1, 4, 5), zeros(0, 3, 3)) returns
        # shape (0, 3, 4, 5) in the same cells.
        # Pins the CURRENT behavior; NOT a contract; delete when #4281 is repaired.
        with pytest.raises(ZeroDivisionError, match="integer division or modulo by zero"):
            warp_frame_depth(
                torch.zeros(0, 2, 4, 5, device=device, dtype=dtype),
                torch.ones(0, 1, 4, 5, device=device, dtype=dtype),
                torch.zeros(0, 4, 4, device=device, dtype=dtype),
                torch.zeros(0, 3, 3, device=device, dtype=dtype),
            )
        empty = depth_to_3d(
            torch.zeros(0, 1, 4, 5, device=device, dtype=dtype), torch.zeros(0, 3, 3, device=device, dtype=dtype)
        )
        assert empty.shape == (0, 3, 4, 5)


class TestDepthWarper(BaseTester):
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
        # Convention pin: the class docstring's
        # ``P_src^{dst} = K_dst * T_src^{dst}`` is exactly what compute_projection_matrix stores --
        # ``_dst_proj_src == K_dst @ (E_dst @ inv(E_src))``, byte for byte, with K_dst and E_dst taken from the
        # DESTINATION camera passed to the constructor and E_src from the camera passed to
        # compute_projection_matrix. The extrinsics are inverted, not transposed: E_src here is a 90-degree
        # rotation about z with a translation, so the transposed reading (which ignores the translation and is
        # what a "R^T" shortcut would give) deviates from the stored matrix by 3.75.
        # The source inverse is written out as a literal rather than computed with torch.linalg.inv, so the pin
        # runs unchanged on mps and in half precision.
        # Snippet used to generate expected: torch.equal(warper._dst_proj_src, K44 @ (E_dst @ E_src_inv)) and
        # (K44 @ (E_dst @ E_src.transpose(-1, -2)) - K44 @ (E_dst @ E_src_inv)).abs().max() executed 2026-09-06
        # at commit 1a96bfd1 (torch 2.14.0) -> True and 3.75 on cpu for float32, float64, float16 and bfloat16 and
        # on mps for float32 and float16; the identity-source arm is True in the same cells.
        # The identity-source arm uses the ROTATION as the destination camera on purpose: a pure x translation
        # commutes with this K (K @ E - E @ K measures exactly 0.0), so it could not tell K @ E_dst from
        # E_dst @ K; with the rotation the two orders differ by 4.0.
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
        # Convention pin: forward takes the depth in
        # the reference frame and the patch in the destination frame and returns a (B, C, H, W) tensor with the
        # patch's channel count, for any C. ``align_corners`` defaults to True and is the only one of the three
        # grid_sample knobs DepthWarper exposes that warp_frame_depth bakes in -- warp_frame_depth has no such
        # parameter at all , which this pin records rather than repairs.
        # The shape claim is backed by a value: an identity camera pair returns the patch byte for byte in
        # float32, so the assertion cannot be satisfied by an all-zero padded result.
        # Snippet used to generate expected: DepthWarper(...).compute_projection_matrix(...)(ones(1, 1, 4, 5),
        # patch) for patch = arange(40).view(1, 2, 4, 5) and arange(100).view(1, 5, 4, 5), executed 2026-09-06
        # at commit 1a96bfd1 (torch 2.14.0) -> shapes (1, 2, 4, 5) and (1, 5, 4, 5) in every cell, and
        # torch.equal to the patch on cpu for float32, float16 and bfloat16 and on mps for float32 and float16;
        # in float64 the round trip is not bit-exact (the grid closes to 1e-12), so the value arm uses
        # assert_close rather than torch.equal. The identity pair alone would be frame-invariant, so a second
        # arm runs the same two channel counts through a +1 x translation and checks the output actually moved
        # (measured deviation from the patch: 39.0 for C = 2 and 99.0 for C = 5, cpu float32).
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
        # Convention pin: DepthWarper is a two-step API. Until
        # compute_projection_matrix has been called there is no relative pose, and both warp_grid and forward
        # raise ValueError("Please, call compute_projection_matrix."), while compute_subpixel_step raises
        # RuntimeError from the None projection matrix instead. Afterwards warp_grid returns the
        # grid_sample-ready NORMALIZED coordinates, (B, H, W, 2), in which pixel (0, 0) is (-1, -1) and pixel
        # (H - 1, W - 1) is (1, 1) -- not the integer pixel grid that DepthWarper.grid itself holds.
        # Snippet used to generate expected: DepthWarper(cam, 4, 5).warp_grid(ones(1, 1, 4, 5)) and the same
        # instance called as a module, executed 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> ValueError with
        # that message; compute_subpixel_step on the same bare instance -> RuntimeError("Expected torch.Tensor,
        # but got None Type from the projection matrix"); after compute_projection_matrix, shape (1, 4, 5, 2)
        # with first pixel [-1.0, -1.0] and last pixel [1.0, 1.0]. All hold on cpu for float32, float64,
        # float16 and bfloat16 and on mps for float32 and float16.
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
        # Convention pin: depth_warp is a functional wrapper that constructs a
        # DepthWarper, calls compute_projection_matrix and forwards -- the results are equal bit for bit, on the
        # integer ramp and on the seeded random camera pair (random K, random rotation, B = 2). That is
        # what makes depth_warp a wrapper rather than a second implementation, unlike warp_frame_depth (pinned
        # below). The warp is a real one: it moves the image by 0.88 on the random pair.
        # Snippet used to generate expected: torch.equal(depth_warp(...), warper(...)) executed 2026-09-06 on
        # commit 1a96bfd1 (torch 2.14.0) -> True on cpu for float32, float64, float16 and bfloat16 and on mps for
        # float32 and float16, for both fixtures; the max deviation between the two is 0.0 in every cell.
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

    def test_convention_warp_frame_depth_matches_depth_warper_without_being_bitwise_equal(self, device, dtype):
        # Convention pin: the two APIs compute the same warp
        # but not the same arithmetic -- warp_frame_depth unprojects through depth_to_3d_v2 and divides by z in
        # project_points, while DepthWarper unprojects through pixel2cam with the cached intrinsics_inverse and
        # divides in cam2pixel. They agree to the dtype tolerance and are NEVER bit-identical, on the
        # seeded random camera pair with and without rotation. Collapsing the two implementations changes bits.
        # The residual is a measured fact, not a bound, so no figure is asserted here.
        # Snippet used to generate expected: torch.equal / (a - b).abs().max() on _random_warp_inputs(2, 11,
        # True) and (2, 7, False), executed 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> equal False in every
        # cell; residual cpu float32 4.02e-07 and 5.36e-07, cpu float64 2.90e-12 and 2.25e-12, mps float32
        # 2.98e-07 and 4.77e-07.
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(
                "half precision: the two paths diverge by 4.88e-03 (float16, seed 7) and 1.95e-02 (bfloat16, "
                "seed 11), outside the 2e-03 / 1.6e-02 assert_close tolerances; the claim is float32/float64"
            )
        for seed, rotate in ((11, True), (7, False)):
            img, dep, transform, k3, k4, batch_h, batch_w = self._random_warp_inputs(device, dtype, 2, seed, rotate)
            batch_identity = torch.eye(4, device=device, dtype=dtype)[None].repeat(2, 1, 1).contiguous()
            warper = DepthWarper(PinholeCamera(k4.clone(), transform.clone(), batch_h, batch_w), 4, 5)
            warper.compute_projection_matrix(PinholeCamera(k4.clone(), batch_identity.clone(), batch_h, batch_w))
            by_class = warper(dep, img)
            by_function = warp_frame_depth(img, dep, transform, k3)
            self.assert_close(by_function, by_class)
            assert not torch.equal(by_function, by_class)
            assert (by_class - img).abs().max().item() > 0.1

    def test_wart_warp_frame_depth_and_depth_warper_name_the_depth_frame_oppositely_4273(self, device, dtype):
        # Wart pin for kornia#4273: the two APIs produce the same
        # warp from OPPOSITE argument names, and this pin spells every argument as a KEYWORD so the clash is in
        # the source text and not only in the prose. The identical warp is written
        #   warp_frame_depth(image_src=IMG, depth_dst=D, src_trans_dst=T)          -- the depth is "dst"
        #   DepthWarper(pinhole_dst=camera(T)).compute_projection_matrix(pinhole_src=camera(I))(
        #       depth_src=D, patch_dst=IMG)                                        -- the same depth is "src",
        # and the image the other one calls "src" is "patch_dst" here. A reader who maps "dst" to "dst" across
        # the two APIs gets the INVERSE warp, out[u] = image[u - 1], which is the swapped pair below.
        # The keywords are load-bearing: #4273's Expected section is a parameter RENAME
        # (depth_src -> depth_dst, patch_dst -> image_src, and the matching pinhole_dst / pinhole_src), so a
        # positional version of this pin would keep passing through the repair and never force its own deletion.
        # Snippet used to generate expected: the three row-0 calls executed 2026-09-06 at commit 1a96bfd1
        # (torch 2.14.0) -> warp_frame_depth [1.0, 2.0, 3.0, 4.0, 0.0]; DepthWarper(pinhole_dst = +1 tx) with an
        # identity pinhole_src [1.0, 2.0, 3.0, 4.0, 0.0] (float64: the last entry is 2.00e-11, hence assert_close
        # and not torch.equal); DepthWarper(pinhole_dst = identity) with a +1 tx pinhole_src
        # [0.0, 0.0, 1.0, 2.0, 3.0]. All three hold on cpu for float32, float64, float16 and bfloat16 and on mps
        # for float32 and float16.
        # Pins the CURRENT naming; NOT a contract; delete when #4273 aligns the two argument names.
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


class TestDepthFromDisparity(BaseTester):
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

    def test_gradcheck(self, device):
        # generate input data
        disparity = torch.rand(1, 1, 3, 4, device=device, dtype=torch.float64)

        baseline = torch.rand(1, device=device, dtype=torch.float64)

        focal = torch.rand(1, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.depth_from_disparity, (disparity, baseline, focal))

    def test_wart_zero_disparity_gives_a_finite_depth_4272(self, device, dtype):
        # Wart pin for kornia#4272: depth_from_disparity computes
        # ``baseline * focal / (disparity + 1e-8)``, so the epsilon enters the arithmetic instead of guarding a
        # branch. A zero disparity -- the standard "no match here" fill value of every stereo matcher -- returns
        # 5e9 for baseline 0.5 and focal 100 rather than inf, and that number is a function of the epsilon, not
        # of the camera: it scales with baseline * focal and is nothing a caller can threshold against.
        # The working case is asserted beside it so the pin is not just about the singular input: disparity 2
        # gives 0.5 * 100 / 2 = 25.
        # Snippet used to generate expected: depth_from_disparity(zeros(1, 1, 1, 1), 0.5, 100.0).item() executed
        # 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> 5000000000.0 on cpu float32 and float64 and on mps
        # float32, 4999610368.0 on cpu bfloat16, and inf on cpu and mps float16 (which is why float16 is
        # skipped: 5e9 is past the float16 range, so the finite-depth claim cannot be stated there at all).
        # Pins the CURRENT value; NOT a contract; delete when #4272 is repaired.
        if dtype == torch.float16:
            pytest.skip("float16: 0.5 * 100 / 1e-8 = 5e9 overflows the float16 range and the result is inf")
        disparity = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype)
        depth = depth_from_disparity(disparity, 0.5, 100.0)
        assert torch.isfinite(depth).all()
        self.assert_close(depth, torch.full((1, 1, 1, 1), 5.0e9, device=device, dtype=dtype))
        self.assert_close(
            depth_from_disparity(torch.full((1, 1, 1, 1), 2.0, device=device, dtype=dtype), 0.5, 100.0),
            torch.full((1, 1, 1, 1), 25.0, device=device, dtype=dtype),
        )

    def test_wart_depth_from_disparity_rejects_a_batched_baseline_4272(self, device, dtype):
        # Wart pin for kornia#4272: the
        # docstring says baseline and focal are "float/tensor", but the guard is KORNIA_CHECK_SHAPE(..., ["1"]),
        # so a tensor argument must have exactly shape (1,). A 0-dim tensor -- what ``torch.tensor(0.5)`` or any
        # reduction produces -- is rejected, and so is a per-batch-element (2,) baseline, even though the
        # disparity itself is batched. A python ``int`` is rejected outright by the type check, for either
        # argument -- so ``depth_from_disparity(d, 1, 100.0)`` fails while ``(d, 1.0, 100.0)`` works.
        # Snippet used to generate expected: depth_from_disparity(ones(1, 1, 2, 3), baseline, focal) executed
        # 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> ShapeError("Shape dimension mismatch: expected 1
        # dimensions, got 0.") for the 0-dim baseline and ShapeError("Shape mismatch at dimension 0: expected 1,
        # got 2.") for the (2,) baseline and the (2,) focal, on cpu for float32, float64, float16 and bfloat16
        # and on mps for float32 and float16; a python int raises BaseError("Input baseline should be either a
        # float or torch.Tensor. Got <class 'int'>") and the same sentence for ``focal`` in those cells, and
        # the (1,) form returns shape (1, 1, 2, 3).
        # Pins the CURRENT behavior; NOT a contract; delete when #4272 is repaired.
        disparity = torch.ones(1, 1, 2, 3, device=device, dtype=dtype)
        focal = torch.tensor([100.0], device=device, dtype=dtype)
        with pytest.raises(ShapeError, match="expected 1 dimensions, got 0"):
            depth_from_disparity(disparity, torch.tensor(0.5, device=device, dtype=dtype), focal)
        with pytest.raises(ShapeError, match="expected 1, got 2"):
            depth_from_disparity(disparity, torch.tensor([0.5, 0.5], device=device, dtype=dtype), focal)
        with pytest.raises(ShapeError, match="expected 1, got 2"):
            depth_from_disparity(
                disparity,
                torch.tensor([0.5], device=device, dtype=dtype),
                torch.tensor([100.0, 100.0], device=device, dtype=dtype),
            )
        with pytest.raises(BaseError, match=r"Input baseline should be either a float or torch\.Tensor"):
            depth_from_disparity(disparity, 1, 100.0)
        with pytest.raises(BaseError, match=r"Input focal should be either a float or torch\.Tensor"):
            depth_from_disparity(disparity, 0.5, 100)
        assert depth_from_disparity(disparity, 1.0, 100.0).shape == (1, 1, 2, 3)
        assert depth_from_disparity(disparity, torch.tensor([0.5], device=device, dtype=dtype), focal).shape == (
            1,
            1,
            2,
            3,
        )


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

        # The default eps=1e-8 is below float16 resolution: it rounds to zero,
        # so `denom_abs < eps` can never hold, and 2/1e-8 is outside float16's
        # finite range anyway. Ask for an epsilon this dtype can represent.
        eps = max(1e-8, float(torch.finfo(dtype).eps))
        depth = kornia.geometry.depth.depth_from_plane_equation(
            plane_normals, plane_offsets, points_uv, camera_matrix, eps=eps
        )
        assert torch.isfinite(depth).all(), f"grazing ray returned {depth.tolist()}"

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
        # Convention pin: the plane is given in the HESSIAN form
        # ``n . X = d`` with the normal and the offset in the CAMERA frame, and the pixels are pixel
        # coordinates that the function normalizes with the intrinsics itself. The fronto-parallel plane
        # n = (0, 0, 1), d = 2 is the plane z = 2, so every pixel -- the principal point and a far off-axis one
        # alike -- has depth exactly 2, and the output is (B, N), one depth per pixel, not a map.
        # The tilted arm is the one-parameter-away check: rotating the normal to (0, 0.5, 1) / |.| makes the two
        # pixels disagree (2.236 vs 1.491), so a reading in which the offset alone sets the depth fails.
        # Snippet used to generate expected: depth_from_plane_equation(n, offsets, pixels, K) executed
        # 2026-09-06 at commit 1a96bfd1 (torch 2.14.0) -> [[2.0, 2.0]] and torch.equal to full(2.0) on cpu for
        # float32, float64, float16 and bfloat16 and on mps for float32 and float16; the tilted plane gives
        # [2.2360680103302, 1.49071204662323] on cpu float32, [2.23606797749979, 1.4907119849998598] on cpu
        # float64, [2.236328125, 1.490234375] on cpu float16 and [2.234375, 1.4921875] on cpu bfloat16.
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
        # Convention pin: the ray-plane dot product is clamped, in a
        # masked branch, to +/- eps when it falls inside (-eps, eps), so a nearly-grazing ray returns a large
        # SIGNED finite depth (2 / 1e-8 = 2e8) instead of overflowing -- and the sign of the denominator is
        # preserved, so the two arms differ by their sign rather than only by magnitude. Nothing outside the
        # mask is touched, which is why the fronto-parallel pin above is exact. The mask
        # also clamps exactly zero (kornia#4280), pinned below.
        # ``eps`` is passed explicitly, equal to the function's own default, so the +/- 2e8 literal is a
        # statement about the CLAMP and not about a default that could move: at eps = 1e-6 the same input
        # returns +/- 2000000.0 instead (executed below).
        # Snippet used to generate expected: depth_from_plane_equation([[0, 1, 2.384185791015625e-09]], [[2.0]],
        # [[[4.0, 3.0]]], K, eps=1e-8) and the negated normal, executed 2026-09-06 at commit 1a96bfd1
        # (torch 2.14.0) -> 200000000.0 and -200000000.0 on cpu float32 and float64 and on mps float32,
        # 200278016.0 and -200278016.0 on cpu bfloat16; with eps=1e-6, 2000000.0 / -2000000.0 (cpu float32,
        # float64, mps float32) and 2007040.0 / -2007040.0 (cpu bfloat16). float16 is skipped:
        # 2.384185791015625e-09 underflows to 0 there, so the probe input is no longer inside the mask and the
        # call returns inf on both signs at either eps.
        if dtype == torch.float16:
            pytest.skip(
                "float16: the 2.384185791015625e-09 normal component underflows to 0, so the input is "
                "no longer a tiny-but-non-zero denominator and both signs return inf"
            )
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
        # The zero-safe clamp fixed in #4280 uses positive eps at an exactly zero denominator.
        if dtype == torch.float16:
            pytest.skip("float16: 2 / 1e-8 = 2e8 overflows the float16 range, so the repaired value is inf too")
        camera_matrix = _k_asymmetric(device, dtype)
        grazing = depth_from_plane_equation(
            torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype),
            torch.tensor([[2.0]], device=device, dtype=dtype),
            torch.tensor([[[4.0, 3.0]]], device=device, dtype=dtype),
            camera_matrix,
        )
        assert torch.isfinite(grazing).all()
        self.assert_close(grazing.abs(), torch.full((1, 1), 2.0e8, device=device, dtype=dtype))
