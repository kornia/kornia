# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

from kornia.core import Tensor
from kornia.geometry.camera import PinholeCamera
from kornia.nerf.nerf_model import NerfModelRenderer
from kornia.nerf.nerf_solver import NerfSolver

from testing.base import assert_close
from testing.nerf import (
    create_four_cameras,
    create_one_camera,
    create_random_images_for_cameras,
    create_red_images_for_cameras,
)


class TestNerfSolver:
    @pytest.mark.slow
    def test_parameter_change_after_one_epoch(self, device, dtype):
        nerf_obj = NerfSolver(device, dtype)
        cameras: PinholeCamera = create_four_cameras(device, dtype)
        imgs: list[Tensor] = create_random_images_for_cameras(cameras)
        nerf_obj.setup_solver(cameras, 1.0, 3.0, True, imgs, num_img_rays=45, batch_size=1, num_ray_points=10)

        params_before_update = [torch.clone(param).detach() for param in nerf_obj.nerf_model.parameters()]

        nerf_obj.run(num_epochs=5)

        params_after_update = [torch.clone(param).detach() for param in nerf_obj.nerf_model.parameters()]

        assert all(
            not torch.equal(param_before_update, param_after_update)
            for param_before_update, param_after_update in zip(params_before_update, params_after_update)
        )

    def test_only_red_uniform_sampling(self, device, dtype):
        torch.manual_seed(1)  # For reproducibility of random processes
        camera: PinholeCamera = create_one_camera(5, 9, device, dtype)
        img: list[Tensor] = create_red_images_for_cameras(camera, device)

        # train the model
        nerf_obj = NerfSolver(device, dtype)
        nerf_obj.setup_solver(camera, 1.0, 3.0, False, img, 15, 2, 10)
        nerf_obj.run(num_epochs=10)

        # render the actual image
        renderer = NerfModelRenderer(nerf_obj.nerf_model, (5, 9), device, dtype)
        img_rendered = renderer.render_view(camera).permute(2, 0, 1)  # CxHxW

        assert_close(img_rendered.to(device, dtype), img[0].to(device, dtype) / 255.0)

    def test_single_ray(self, device, dtype):
        camera: PinholeCamera = create_one_camera(5, 9, device, dtype)
        img: list[Tensor] = create_red_images_for_cameras(camera, device)

        nerf_obj = NerfSolver(device=device, dtype=dtype)
        nerf_obj.setup_solver(camera, 1.0, 3.0, True, img, 1, 2, 10)
        nerf_obj.run(num_epochs=20)

    def test_only_red(self, device, dtype):
        torch.manual_seed(0)  # For reproducibility of random processes

        camera: PinholeCamera = create_one_camera(5, 9, device, dtype)
        img: list[Tensor] = create_red_images_for_cameras(camera, device)

        # train the model
        nerf_obj = NerfSolver(device=device, dtype=dtype)
        nerf_obj.setup_solver(camera, 1.0, 3.0, False, img, num_img_rays=15, batch_size=5, num_ray_points=10, lr=1e-2)
        nerf_obj.run(num_epochs=10)

        # render the actual image
        renderer = NerfModelRenderer(nerf_obj.nerf_model, (5, 9), device, dtype)
        img_rendered = renderer.render_view(camera).permute(2, 0, 1)  # CxHxW

        assert_close(img_rendered.to(device, dtype), img[0].to(device, dtype) / 255.0)
