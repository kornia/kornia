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

import torch

from kornia.nerf.positional_encoder import PositionalEncoder


class TestPositionalEncoder:
    def test_dimensions(self, device, dtype):
        num_rays = 15
        num_ray_points = 11
        num_dims = 3
        x = torch.rand(num_rays, num_ray_points, num_dims, device=device, dtype=dtype)
        num_freqs = 10
        pos_encoder = PositionalEncoder(num_dims, num_freqs)
        x_encoded = pos_encoder(x)
        assert x_encoded.shape == (num_rays, num_ray_points, pos_encoder.num_encoded_dims)
