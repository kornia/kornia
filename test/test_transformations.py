import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utilities
from common import TEST_DEVICES


class TestTransformPose:
    def test_identity(self):
        pose_1 = torch.eye(4).unsqueeze(0)
        pose_2 = torch.eye(4).unsqueeze(0)
        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21, torch.eye(4).unsqueeze(0))

    def test_translation(self):
        offset = 10.
        pose_1 = torch.eye(4).unsqueeze(0)
        pose_2 = torch.eye(4).unsqueeze(0)
        pose_2[..., :3, -1:] += offset  # add translation

        # compute relative pose
        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21[..., :3, -1:], offset)

    def test_rotation(self):
        pose_1 = torch.eye(4).unsqueeze(0)
        pose_2 = torch.zeros(1, 4, 4)  # Rz (90deg)
        pose_2[..., 0, 1] = -1.0
        pose_2[..., 1, 0] = 1.0
        pose_2[..., 2, 2] = 1.0
        pose_2[..., 3, 3] = 1.0

        # compute relative pose
        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21, pose_2)

    @pytest.mark.parametrize("device_type", TEST_DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_integration(self, batch_size, device_type):
        pose_1 = torch.eye(4).repeat(batch_size, 1, 1)
        pose_1 = pose_1.to(torch.device(device_type))

        pose_2 = torch.eye(4).repeat(batch_size, 1, 1)
        pose_2[..., :3, :3] = torch.rand(batch_size, 3, 3)
        pose_2[..., :3, -1:] = torch.rand(batch_size, 3, 1)
        pose_2 = pose_2.to(torch.device(device_type))

        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(
            torch.matmul(pose_21, pose_1), pose_2)

    @pytest.mark.skip("Converting a tensor to a Python boolean ...")
    def test_jit(self):
        pose_1 = torch.eye(4).unsqueeze(0)
        pose_2 = torch.eye(4).unsqueeze(0)

        pose_21 = tgm.relative_pose(pose_1, pose_2)
        pose_21_jit = torch.jit.trace(
            tgm.relative_pose, (pose_1, pose_2,))(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21, pose_21_jit)

    def test_gradcheck(self):
        pose_1 = torch.eye(4).unsqueeze(0)
        pose_2 = torch.eye(4).unsqueeze(0)

        pose_1 = utils.tensor_to_gradcheck_var(pose_1)  # to var
        pose_2 = utils.tensor_to_gradcheck_var(pose_2)  # to var
        assert gradcheck(tgm.relative_pose, (pose_1, pose_2,),
                         raise_exception=True)
