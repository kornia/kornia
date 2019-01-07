import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils
from common import TEST_DEVICES


class Tester:

    num_tests = 10
    threshold = 0.05

    @pytest.mark.parametrize("device_type", TEST_DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_homography_warper(self, batch_size, device_type):
        # generate input data
        height, width = 128, 64
        eye_size = 3  # identity 3x3
        device = torch.device(device_type)

        # create checkerboard
        board = utils.create_checkerboard(height, width, 4)
        patch_src = torch.from_numpy(board).view(
            1, 1, height, width).expand(batch_size, 1, height, width)
        patch_src = patch_src.to(device)

        # create base homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size).to(device)

        # instantiate warper
        warper = tgm.HomographyWarper(height, width)

        for i in range(self.num_tests):
            # generate homography noise
            homo_delta = torch.zeros_like(dst_homo_src)
            homo_delta[:, -1, -1] = 0.0

            dst_homo_src_i = dst_homo_src + homo_delta

            # transform the points from dst to ref
            patch_dst = warper(patch_src, dst_homo_src_i)
            patch_dst_to_src = warper(patch_dst, torch.inverse(dst_homo_src_i))

            # projected should be equal as initial
            error = utils.compute_patch_error(
                patch_dst, patch_dst_to_src, height, width)

            assert error.item() < self.threshold

            # check functional api
            patch_dst_to_src_functional = tgm.homography_warp(
                patch_dst, torch.inverse(dst_homo_src_i), (height, width))

            assert utils.check_equal_torch(
                patch_dst_to_src, patch_dst_to_src_functional)

    @pytest.mark.parametrize("device_type", TEST_DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("height", [7, 8])
    @pytest.mark.parametrize("width", [5, 16])
    def test_homography_warper_gradcheck(
        self, batch_size, channels, height, width, device_type):
        # generate input data
        device = torch.device(device_type)
        eye_size = 3  # identity 3x3

        # create checkerboard
        patch_src = torch.rand(batch_size, channels, height, width).to(device)
        patch_src = utils.tensor_to_gradcheck_var(patch_src)  # to var

        # create base homography
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size)
        dst_homo_src = utils.tensor_to_gradcheck_var(
            dst_homo_src, requires_grad=False)  # to var

        # instantiate warper
        warper = tgm.HomographyWarper(height, width)

        patch_dst = warper(patch_src, dst_homo_src)
        assert patch_dst.shape == (batch_size, channels, height, width)

        # evaluate function gradient
        assert gradcheck(warper, (patch_src, dst_homo_src,),
                         raise_exception=True)
