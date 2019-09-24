import pytest

import kornia
import kornia.testing as utils  # test utils
from test.common import TEST_DEVICES

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose


class TestDepthTo3D:
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
