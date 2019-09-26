import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import TEST_DEVICES

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestDepthTo3d:
    def test_smoke(self):
        depth = torch.rand(1, 1, 3, 4)
        camera_matrix = torch.rand(1, 3, 3)

        points3d = kornia.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes(self, batch_size):
        depth = torch.rand(batch_size, 1, 3, 4)
        camera_matrix = torch.rand(batch_size, 3, 3)

        points3d = kornia.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
    def test_shapes_broadcast(self, batch_size):
        depth = torch.rand(batch_size, 1, 3, 4)
        camera_matrix = torch.rand(1, 3, 3)

        points3d = kornia.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    def test_unproject(self):
        depth = 2 * torch.tensor([[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]]])

        camera_matrix = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]])

        points3d_expected = torch.tensor([[[
            [0.0000, 1.4142, 1.7889],
            [0.0000, 1.1547, 1.6330],
            [0.0000, 0.8165, 1.3333],
            [0.0000, 0.6030, 1.0690],
        ], [
            [0.0000, 0.0000, 0.0000],
            [1.4142, 1.1547, 0.8165],
            [1.7889, 1.6330, 1.3333],
            [1.8974, 1.8091, 1.6036],
        ], [
            [2.0000, 1.4142, 0.8944],
            [1.4142, 1.1547, 0.8165],
            [0.8944, 0.8165, 0.6667],
            [0.6325, 0.6030, 0.5345],
        ]]])

        points3d = kornia.depth_to_3d(depth, camera_matrix)
        assert_allclose(points3d, points3d_expected)

    def test_unproject_and_project(self):
        depth = 2 * torch.tensor([[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]]])

        camera_matrix = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]])

        points3d = kornia.depth_to_3d(depth, camera_matrix)
        points2d = kornia.project_points(
            points3d.permute(0, 2, 3, 1),
            camera_matrix[:, None, None]
        )
        points2d_expected = kornia.create_meshgrid(4, 3, False)
        assert_allclose(points2d, points2d_expected)

    def test_gradcheck(self):
        # generate input data
        depth = torch.rand(1, 1, 3, 4)
        depth = utils.tensor_to_gradcheck_var(depth)  # to var

        camera_matrix = torch.rand(1, 3, 3)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)  # to var

        # evaluate function gradient
        assert gradcheck(kornia.depth_to_3d, (depth, camera_matrix,),
                         raise_exception=True)


class TestDepthToNormals:
    def test_smoke(self):
        depth = torch.rand(1, 1, 3, 4)
        camera_matrix = torch.rand(1, 3, 3)

        points3d = kornia.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes(self, batch_size):
        depth = torch.rand(batch_size, 1, 3, 4)
        camera_matrix = torch.rand(batch_size, 3, 3)

        points3d = kornia.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes_broadcast(self, batch_size):
        depth = torch.rand(batch_size, 1, 3, 4)
        camera_matrix = torch.rand(1, 3, 3)

        points3d = kornia.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    def test_simple(self):
        depth = 2 * torch.tensor([[[
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
        ]]])

        camera_matrix = torch.tensor([[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]])

        normals_expected = torch.tensor([[[
            [0.3432, 0.4861, 0.7628],
            [0.2873, 0.4260, 0.6672],
            [0.2284, 0.3683, 0.5596],
            [0.1695, 0.2980, 0.4496],
        ], [
            [0.3432, 0.2873, 0.2363],
            [0.4861, 0.4260, 0.3785],
            [0.8079, 0.7261, 0.6529],
            [0.8948, 0.8237, 0.7543],
        ], [
            [0.8743, 0.8253, 0.6019],
            [0.8253, 0.7981, 0.6415],
            [0.5432, 0.5807, 0.5105],
            [0.4129, 0.4824, 0.4784],
        ]]])

        normals = kornia.depth_to_normals(depth, camera_matrix)
        assert_allclose(normals, normals_expected, 1e-3, 1e-3)

    def test_gradcheck(self):
        # generate input data
        depth = torch.rand(1, 1, 3, 4)
        depth = utils.tensor_to_gradcheck_var(depth)  # to var

        camera_matrix = torch.rand(1, 3, 3)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)  # to var

        # evaluate function gradient
        assert gradcheck(kornia.depth_to_normals, (depth, camera_matrix,),
                         raise_exception=True)


class TestWarpFrameDepth:
    def test_smoke(self):
        image_src = torch.rand(1, 3, 3, 4)
        depth_dst = torch.rand(1, 1, 3, 4)
        src_trans_dst = torch.rand(1, 4, 4)
        camera_matrix = torch.rand(1, 3, 3)

        image_dst = kornia.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    @pytest.mark.parametrize("num_features", [1, 3, 5])
    def test_shape(self, batch_size, num_features):
        image_src = torch.rand(batch_size, num_features, 3, 4)
        depth_dst = torch.rand(batch_size, 1, 3, 4)
        src_trans_dst = torch.rand(batch_size, 4, 4)
        camera_matrix = torch.rand(batch_size, 3, 3)

        image_dst = kornia.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (batch_size, num_features, 3, 4)

    def test_gradcheck(self):
        image_src = torch.rand(1, 3, 3, 4)
        image_src = utils.tensor_to_gradcheck_var(image_src)  # to var

        depth_dst = torch.rand(1, 1, 3, 4)
        depth_dst = utils.tensor_to_gradcheck_var(depth_dst)  # to var

        src_trans_dst = torch.rand(1, 4, 4)
        src_trans_dst = utils.tensor_to_gradcheck_var(src_trans_dst)  # to var

        camera_matrix = torch.rand(1, 3, 3)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)  # to var

        # evaluate function gradient
        assert gradcheck(kornia.warp_frame_depth, (image_src, depth_dst, src_trans_dst, camera_matrix,),
                         raise_exception=True)
