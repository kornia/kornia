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

    def test_gradcheck(self):
        # generate input data
        depth = torch.rand(1, 1, 3, 4)
        depth = utils.tensor_to_gradcheck_var(depth)  # to var

        camera_matrix = torch.rand(1, 3, 3)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)  # to var

        # evaluate function gradient
        assert gradcheck(kornia.depth_to_3d, (depth, camera_matrix,),
                         raise_exception=True)
