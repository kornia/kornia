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

from testing.base import BaseTester


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

    def test_unproject_meshgrid(self, device, dtype):
        # TODO: implement me with batch
        camera_matrix = torch.eye(3, device=device, dtype=dtype).repeat(2, 1, 1)
        grid = kornia.geometry.unproject_meshgrid(3, 4, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (2, 3, 4, 3)
        # test for now that the grid is correct and have homogeneous coords
        self.assert_close(grid[..., 2], torch.ones_like(grid[..., 2]))

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
        points2d_expected = kornia.utils.create_meshgrid(4, 3, False, device=device).to(dtype=dtype)
        self.assert_close(points2d, points2d_expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        # generate input data
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=torch.float64)

        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=torch.float64)

        # evaluate function gradient
        self.gradcheck(kornia.geometry.depth.depth_to_3d, (depth, camera_matrix))


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
