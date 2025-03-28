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


def identity_matrix(batch_size, device, dtype):
    r"""Create a batched homogeneous identity matrix."""
    return torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # Nx4x4


def euler_angles_to_rotation_matrix(x, y, z):
    r"""Create a rotation matrix from x, y, z angles."""
    assert x.dim() == 1, x.shape
    assert x.shape == y.shape == z.shape
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    # the rotation matrix for the x-axis
    rx_tmp = [
        ones,
        zeros,
        zeros,
        zeros,
        zeros,
        torch.cos(x),
        -torch.sin(x),
        zeros,
        zeros,
        torch.sin(x),
        torch.cos(x),
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
    ]
    rx = torch.stack(rx_tmp, dim=-1).view(-1, 4, 4)
    # the rotation matrix for the y-axis
    ry_tmp = [
        torch.cos(y),
        zeros,
        torch.sin(y),
        zeros,
        zeros,
        ones,
        zeros,
        zeros,
        -torch.sin(y),
        zeros,
        torch.cos(y),
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
    ]
    ry = torch.stack(ry_tmp, dim=-1).view(-1, 4, 4)
    # the rotation matrix for the z-axis
    rz_tmp = [
        torch.cos(z),
        -torch.sin(z),
        zeros,
        zeros,
        torch.sin(z),
        torch.cos(z),
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
    ]
    rz = torch.stack(rz_tmp, dim=-1).view(-1, 4, 4)
    return torch.matmul(rz, torch.matmul(ry, rx))  # Bx4x4
