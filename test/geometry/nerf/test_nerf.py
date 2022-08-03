from test.geometry.nerf.test_data_utils import create_random_images_for_cameras
from test.geometry.nerf.test_rays import create_four_cameras

from kornia.geometry.nerf.nerf_solver import NerfSolver

# import torch


class TestNerf:
    def test_train_one_epoch(self, device, dtype):
        nerf_obj = NerfSolver()
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.init_training(imgs)
        nerf_obj.train_one_epoch()
