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

shapes = [(512, 3, 256, 256), (256, 1, 64, 64)]
PSs = [224, 32]


@pytest.mark.xfail(reason="May cause memory issues.")
def test_performance_speed(device, dtype):
    if device.type != "cuda" or not torch.cuda.is_available():
        pytest.skip("Cuda not available in system,")

    print("Benchmarking warp_affine")
    for input_shape in shapes:
        for PS in PSs:
            BS = input_shape[0]
            data = torch.rand(input_shape).to(device)
            As = torch.eye(3).unsqueeze(0).repeat(BS, 1, 1)[:, :2, :].to(device)
            As += 0.1 * torch.rand(As.size()).to(device)
            torch.cuda.synchronize(device)
            t = time()
            _ = kornia.warp_affine(data, As, (PS, PS))
            print(f"inp={input_shape}, PS={PS}, dev={device}, {time() - t}, sec")
            torch.cuda.synchronize(device)
