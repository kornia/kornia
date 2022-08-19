from test.geometry.nerf.test_data_utils import create_random_images_for_cameras, create_red_images_for_cameras
from test.geometry.nerf.test_rays import create_four_cameras, create_one_camera

import torch

from kornia.geometry.nerf.nerf_solver import NerfSolver


class TestNerfSolver:
    def test_parameter_change_after_one_epoch(self, device, dtype):
        nerf_obj = NerfSolver()
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.init_training(cameras, 1.0, 3.0, imgs, 11, 2, 10)

        params_before_update = [torch.clone(param) for param in nerf_obj.nerf_model.parameters()]

        nerf_obj.run()

        params_after_update = [torch.clone(param) for param in nerf_obj.nerf_model.parameters()]

        assert all(
            not torch.equal(param_before_update, param_after_update)
            for param_before_update, param_after_update in zip(params_before_update, params_after_update)
        )

    def test_only_red_uniform_sampling(self, device, dtype):
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera)

        nerf_obj = NerfSolver()
        nerf_obj.init_training(camera, 1.0, 3.0, img, None, 2, 10)
        nerf_obj.run(num_epochs=5)

        img_rendered = nerf_obj.render_views(camera)[0]

        assert torch.all(torch.isclose(img[0] / 255.0, img_rendered / 255.0)).item()

    def test_single_ray(self, device, dtype):
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera)

        nerf_obj = NerfSolver()
        nerf_obj.init_training(camera, 1.0, 3.0, img, 1, 2, 10)
        nerf_obj.run(num_epochs=20)

    def test_only_red(self, device, dtype):
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera)

        nerf_obj = NerfSolver()
        num_img_rays = torch.tensor([15])
        nerf_obj.init_training(camera, 1.0, 3.0, img, num_img_rays, 2, 10)
        nerf_obj.run(num_epochs=5)

        img_rendered = nerf_obj.render_views(camera)[0]

        assert torch.all(torch.isclose(img[0] / 255.0, img_rendered / 255.0)).item()
