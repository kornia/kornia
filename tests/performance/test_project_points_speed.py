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

from time import time

import pytest
import torch

import kornia

points_shapes = [(64, 1024**2, 3), (8192, 8192, 3), (1024**2, 64, 3)]

# TODO: remove xfail once we have enough gpu bandwidth in the CI


@pytest.mark.xfail(reason="May cause memory issues.")
def test_performance_speed(device, dtype):
    if device.type != "cuda" or not torch.cuda.is_available():
        pytest.skip("Cuda not available in system,")

    print("Benchmarking project_points")
    for input_shape in points_shapes:
        data = torch.rand(input_shape).to(device)
        pose = torch.rand((1, 4, 4)).to(device)
        torch.cuda.synchronize(device)
        t = time()
        kornia.geometry.transform_points(pose, data)
        torch.cuda.synchronize(device)
        print(f"inp={input_shape}, dev={device}, {time() - t}, sec")
