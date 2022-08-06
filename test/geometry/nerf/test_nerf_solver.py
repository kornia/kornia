from test.geometry.nerf.test_data_utils import create_random_images_for_cameras
from test.geometry.nerf.test_rays import create_four_cameras

from kornia.geometry.nerf.nerf_solver import NerfSolver

# import torch


class TestNerfSolver:
    def test_train_one_epoch(self, device, dtype):
        nerf_obj = NerfSolver()
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.init_training(cameras, 1.0, 3.0, imgs, 4, 10)
        nerf_obj.train_one_epoch()
