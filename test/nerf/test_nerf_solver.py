import pytest
import torch

from kornia.nerf.nerf_solver import NerfSolver
from kornia.testing import assert_close
from test.nerf.test_data_utils import create_random_images_for_cameras, create_red_images_for_cameras
from test.nerf.test_rays import create_four_cameras, create_one_camera


class TestNerfSolver:
    @pytest.mark.slow
    def test_parameter_change_after_one_epoch(self, device, dtype):
        nerf_obj = NerfSolver(device, dtype)
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.init_training(cameras, 1.0, 3.0, True, imgs, num_img_rays=45, batch_size=1, num_ray_points=10)

        params_before_update = [torch.clone(param).detach() for param in nerf_obj.nerf_model.parameters()]

        nerf_obj.run(num_epochs=5)

        params_after_update = [torch.clone(param).detach() for param in nerf_obj.nerf_model.parameters()]

        assert all(
            not torch.equal(param_before_update, param_after_update)
            for param_before_update, param_after_update in zip(params_before_update, params_after_update)
        )

    @pytest.mark.slow
    def test_only_red_uniform_sampling(self, device, dtype):
        torch.manual_seed(1)  # For reproducibility of random processes
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera, device)

        nerf_obj = NerfSolver(device, dtype)
        nerf_obj.init_training(camera, 1.0, 3.0, False, img, None, 2, 10)
        nerf_obj.run(num_epochs=10)

        img_rendered = nerf_obj.render_views(camera)[0].permute(2, 0, 1)

        assert_close(img_rendered.to(device, dtype) / 255.0, img[0].to(device, dtype) / 255.0)

    def test_single_ray(self, device, dtype):
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera, device)

        nerf_obj = NerfSolver(device=device, dtype=dtype)
        nerf_obj.init_training(camera, 1.0, 3.0, True, img, 1, 2, 10)
        nerf_obj.run(num_epochs=20)

    def test_only_red(self, device, dtype):
        torch.manual_seed(0)  # For reproducibility of random processes

        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera, device)

        nerf_obj = NerfSolver(device=device, dtype=dtype)
        num_img_rays = 15
        nerf_obj.init_training(camera, 1.0, 3.0, False, img, num_img_rays, batch_size=5, num_ray_points=10, lr=1e-2)
        nerf_obj.run(num_epochs=10)

        img_rendered = nerf_obj.render_views(camera)[0].permute(2, 0, 1)

        assert_close(img_rendered.to(device, dtype) / 255.0, img[0].to(device, dtype) / 255.0)
