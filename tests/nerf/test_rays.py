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


from kornia.nerf.samplers import (
    RandomGridRaySampler,
    RandomRaySampler,
    UniformRaySampler,
    calc_ray_t_vals,
    cameras_for_ids,
    sample_lengths,
    sample_ray_points,
)

from kornia.geometry.camera.utils import create_camera_dimensions, create_four_cameras

from testing.base import assert_close


class TestRaySampler_2DPoints:
    def test_dimensions_random_sampler(self, device, dtype):
        heights, widths, num_img_rays = create_camera_dimensions(device, dtype)
        sampler = RandomRaySampler(1, 1, ndc=False, device=device, dtype=dtype)
        points_2d_camera = sampler.sample_points_2d(heights, widths, num_img_rays)
        assert len(points_2d_camera) == 2
        assert points_2d_camera[10].points_2d.shape == (3, 10, 2)
        assert points_2d_camera[15].points_2d.shape == (2, 15, 2)

    def test_dimensions_uniform_sampler(self, device, dtype):
        heights, widths, _ = create_camera_dimensions(device, dtype)
        sampler = UniformRaySampler(1, 1, ndc=False, device=device, dtype=dtype)
        points_2d_camera = sampler.sample_points_2d(heights, widths)
        assert len(points_2d_camera) == 2
        assert points_2d_camera[60000].points_2d.shape == (3, 60000, 2)
        assert points_2d_camera[40000].points_2d.shape == (2, 40000, 2)

    def test_dimensions_radom_grid_sampler(self, device, dtype):
        heights, widths, num_img_rays = create_camera_dimensions(device, dtype)
        sampler = RandomGridRaySampler(1, 1, True, device=device, dtype=dtype)
        points_2d_camera = sampler.sample_points_2d(heights, widths, num_img_rays)
        assert len(points_2d_camera) == 1
        assert points_2d_camera[9].points_2d.shape == (5, 9, 2)


class TestRaySampler_3DPoints:
    def test_dimensions_uniform_sampler(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(1, 1, False, device=device, dtype=dtype)
        points_2d_four_cameras = uniform_sampler_four_cameras.sample_points_2d(cameras.height, cameras.width)
        cameras_28 = cameras_for_ids(cameras, points_2d_four_cameras[28].camera_ids)
        points_3d_28 = cameras_28.unproject(points_2d_four_cameras[28].points_2d, 1)
        cameras_40 = cameras_for_ids(cameras, points_2d_four_cameras[45].camera_ids)
        points_3d_40 = cameras_40.unproject(points_2d_four_cameras[45].points_2d, 1)
        assert points_3d_28.shape == (3, 28, 3)
        assert points_3d_40.shape == (1, 45, 3)

    def test_dimensions_ray_params(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(1, 2, False, device=device, dtype=dtype)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        lengths = sample_lengths(
            uniform_sampler_four_cameras.origins.shape[0], 10, device=device, dtype=dtype, irregular=True
        )
        assert uniform_sampler_four_cameras.origins.shape == (3 * 28 + 45, 3)
        assert uniform_sampler_four_cameras.directions.shape == (3 * 28 + 45, 3)
        assert uniform_sampler_four_cameras.camera_ids.shape == (3 * 28 + 45,)
        assert uniform_sampler_four_cameras.points_2d.shape == (3 * 28 + 45, 2)
        assert lengths.shape == (3 * 28 + 45, 10)

    def test_dimensions_sample_ray_points(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(1, 2, False, device, dtype=dtype)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        lengths = sample_lengths(
            uniform_sampler_four_cameras.origins.shape[0], 10, device=device, dtype=dtype, irregular=True
        )
        points_3d = sample_ray_points(
            uniform_sampler_four_cameras.origins, uniform_sampler_four_cameras.directions, lengths
        )
        assert points_3d.shape == (3 * 28 + 45, 10, 3)

    def test_t_vals(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(2, 3.0, False, device=device, dtype=dtype)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        lengths = sample_lengths(
            uniform_sampler_four_cameras.origins.shape[0], 10, device=device, dtype=dtype, irregular=False
        )
        points_3d = sample_ray_points(
            uniform_sampler_four_cameras.origins, uniform_sampler_four_cameras.directions, lengths
        )
        t_vals = calc_ray_t_vals(points_3d)
        assert t_vals.shape == (3 * 28 + 45, 10)
        assert_close(t_vals[22, -1].item(), 1.0)  # Testing middle ray

    def test_dimensions_ray_params_in_ndc(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        uniform_sampler_four_cameras = UniformRaySampler(2, 3.5, False, device=device, dtype=dtype)
        uniform_sampler_four_cameras.calc_ray_params(cameras)
        origins_ndc, directions_ndc = uniform_sampler_four_cameras.transform_ray_params_world_to_ndc(cameras)
        assert origins_ndc.shape == (3 * 28 + 45, 3)
        assert directions_ndc.shape == (3 * 28 + 45, 3)
