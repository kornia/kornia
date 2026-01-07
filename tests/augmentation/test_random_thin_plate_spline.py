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

from kornia.augmentation import RandomThinPlateSpline


def test_smoke():
    x = torch.randn(2, 3, 64, 64)
    aug = RandomThinPlateSpline()
    y = aug(x)
    assert y.shape == x.shape


def test_same_on_batch():
    torch.manual_seed(42)
    x = torch.randn(4, 3, 32, 32)

    aug = RandomThinPlateSpline(p=1.0, same_on_batch=True)
    y = aug(x)

    params = aug._params

    # src and dst contain TPS control points
    for key in ["src", "dst"]:
        for j in range(1, 4):
            assert torch.allclose(params[key][0], params[key][j])


def test_device_cpu():
    x = torch.randn(1, 3, 16, 16)
    aug = RandomThinPlateSpline(p=1.0)
    y = aug(x)
    assert y.device == x.device
