from test.geometry.nerf.test_data_utils import create_random_images_for_cameras, create_red_images_for_cameras
from test.geometry.nerf.test_rays import create_four_cameras

import torch

from kornia.geometry.camera import PinholeCamera
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

    def test_only_red(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        height = cameras.height[0].unsqueeze(0)
        width = cameras.width[0].unsqueeze(0)
        intrinsics_1st = cameras.intrinsics[0].unsqueeze(0)
        extrinsics_1st = cameras.extrinsics[0].unsqueeze(0)
        camera = PinholeCamera(intrinsics_1st, extrinsics_1st, height, width)
        img = create_red_images_for_cameras(camera)

        nerf_obj = NerfSolver()
        nerf_obj.init_training(camera, 1.0, 3.0, img, 2, 10)
        nerf_obj.train_one_epoch()
