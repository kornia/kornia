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

from kornia.geometry.line import ParametrizedLine
from kornia.geometry.ray import Ray


def test_ray_is_parametrized_line():
    assert Ray is ParametrizedLine


def test_ray_smoke():
    origin = torch.zeros(3)
    direction = torch.tensor([0.0, 0.0, 1.0])
    ray = Ray(origin, direction)
    assert isinstance(ray, ParametrizedLine)


def test_ray_point_at():
    origin = torch.zeros(3)
    direction = torch.tensor([1.0, 0.0, 0.0])
    ray = Ray(origin, direction)
    pt = ray.point_at(2.0)
    expected = torch.tensor([2.0, 0.0, 0.0])
    assert torch.allclose(pt, expected)
