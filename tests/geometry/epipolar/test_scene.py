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

import kornia.geometry.epipolar as epi

from testing.base import BaseTester

_SCENE_SHAPES = {
    "K": (1, 3, 3),
    "R": (3, 3, 3),
    "t": (3, 3, 1),
    "P": (3, 3, 4),
    "points3d": (1, 7, 3),
    "points2d": (3, 7, 2),
}


class TestConventionScene(BaseTester):
    def test_convention_random_intrinsics_global_generator(self, device, dtype):
        # fx, fy, cx, cy are four consecutive draws of U(low, high) from the global generator, in the dtype and on
        # the device of the bounds.
        low = torch.tensor(10.0, device=device, dtype=dtype)
        high = torch.tensor(20.0, device=device, dtype=dtype)
        torch.manual_seed(0)
        K = epi.random_intrinsics(low, high)
        K_next = epi.random_intrinsics(low, high)
        torch.manual_seed(0)
        K_again = epi.random_intrinsics(low, high)
        torch.manual_seed(0)
        draws = torch.distributions.Uniform(low, high).sample((4,))
        assert K.shape == (1, 3, 3) and K.dtype == dtype and K.device == low.device
        assert torch.equal(K, K_again)
        assert not torch.equal(K, K_next)
        self.assert_close(K[0, [0, 1, 0, 1], [0, 1, 2, 2]], draws)
        self.assert_close(K[0, 2], torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype))

    def test_convention_generate_scene_global_generator(self, device, dtype):
        if device.type != "cpu":
            pytest.skip("generate_scene takes no device argument and draws on torch's default device")
        # The same global generator as random_intrinsics, tensors in the default dtype.
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            torch.manual_seed(0)
            scene = epi.generate_scene(3, 7)
            scene_next = epi.generate_scene(3, 7)
            torch.manual_seed(0)
            scene_again = epi.generate_scene(3, 7)
        finally:
            torch.set_default_dtype(default_dtype)
        assert {k: tuple(v.shape) for k, v in scene.items()} == _SCENE_SHAPES
        assert all(v.dtype == dtype and v.device.type == "cpu" for v in scene.values())
        assert all(torch.equal(scene[k], scene_again[k]) for k in _SCENE_SHAPES)
        assert not any(torch.equal(scene[k], scene_next[k]) for k in _SCENE_SHAPES)
        # One K for every view, P = K [R | t] per view, and points2d the pixels of points3d through each P, all
        # in front of every camera.
        K, R, t, P = scene["K"], scene["R"], scene["t"], scene["P"]
        self.assert_close(P, K @ torch.cat([R, t], -1))
        X = scene["points3d"].expand(3, -1, -1)
        x = torch.cat([X, torch.ones_like(X[..., :1])], -1) @ P.transpose(-2, -1)
        self.assert_close(x[..., :2] / x[..., 2:], scene["points2d"], low_tolerance=True)
        assert ((X @ R.transpose(-2, -1) + t.transpose(-2, -1))[..., 2] > 0).all()
