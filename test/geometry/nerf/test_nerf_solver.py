from test.geometry.nerf.test_data_utils import create_random_images_for_cameras
from test.geometry.nerf.test_rays import create_four_cameras

import torch

from kornia.geometry.nerf.nerf_solver import NerfSolver


class TestNerfSolver:
    def test_parameter_change_after_one_epoch(self, device, dtype):
        nerf_obj = NerfSolver()
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.init_training(cameras, 1.0, 3.0, imgs, 2, 10)

        params_before_update = [torch.clone(param) for param in nerf_obj.nerf_model.parameters()]

        nerf_obj.train_one_epoch()

        params_after_update = [torch.clone(param) for param in nerf_obj.nerf_model.parameters()]

        assert all(
            not torch.equal(param_before_update, param_after_update)
            for param_before_update, param_after_update in zip(params_before_update, params_after_update)
        )
