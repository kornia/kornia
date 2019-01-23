import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utilities
from common import TEST_DEVICES


class TestTransformPose:

    def _generate_identity_matrix(self, batch_size, device_type):
        eye = torch.eye(4).repeat(batch_size, 1, 1)  # Nx4x4
        return eye.to(torch.device(device_type))

    def _test_identity(self):
        pose_1 = self.pose_1.clone()
        pose_2 = self.pose_2.clone()
        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21, torch.eye(4).unsqueeze(0))

    def _test_translation(self):
        offset = 10.
        pose_1 = self.pose_1.clone()
        pose_2 = self.pose_2.clone()
        pose_2[..., :3, -1:] += offset  # add translation

        # compute relative pose
        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21[..., :3, -1:], offset)

    def _test_rotation(self):
        pose_1 = self.pose_1.clone()
        pose_2 = torch.zeros_like(pose_1)  # Rz (90deg)
        pose_2[..., 0, 1] = -1.0
        pose_2[..., 1, 0] = 1.0
        pose_2[..., 2, 2] = 1.0
        pose_2[..., 3, 3] = 1.0

        # compute relative pose
        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21, pose_2)

    def _test_integration(self):
        pose_1 = self.pose_1.clone()
        pose_2 = self.pose_2.clone()

        # apply random rotations and translations
        batch_size, device = pose_2.shape[0], pose_2.device
        pose_2[..., :3, :3] = torch.rand(batch_size, 3, 3, device=device)
        pose_2[..., :3, -1:] = torch.rand(batch_size, 3, 1, device=device)

        pose_21 = tgm.relative_pose(pose_1, pose_2)
        assert utils.check_equal_torch(
            torch.matmul(pose_21, pose_1), pose_2)

    @pytest.mark.skip("Converting a tensor to a Python boolean ...")
    def test_jit(self):
        pose_1 = self.pose_1.clone()
        pose_2 = self.pose_2.clone()

        pose_21 = tgm.relative_pose(pose_1, pose_2)
        pose_21_jit = torch.jit.trace(
            tgm.relative_pose, (pose_1, pose_2,))(pose_1, pose_2)
        assert utils.check_equal_torch(pose_21, pose_21_jit)

    def _test_gradcheck(self):
        pose_1 = self.pose_1.clone()
        pose_2 = self.pose_2.clone()

        pose_1 = utils.tensor_to_gradcheck_var(pose_1)  # to var
        pose_2 = utils.tensor_to_gradcheck_var(pose_2)  # to var
        assert gradcheck(tgm.relative_pose, (pose_1, pose_2,),
                         raise_exception=True)

    @pytest.mark.parametrize("device_type", TEST_DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_run_all(self, batch_size, device_type):
        # generate identity matrices
        self.pose_1 = self._generate_identity_matrix(
            batch_size, device_type)
        self.pose_2 = self.pose_1.clone()

        # run tests
        self._test_identity()
        self._test_translation()
        self._test_rotation()
        self._test_integration()
        self._test_gradcheck()
