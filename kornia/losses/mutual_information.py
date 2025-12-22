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


def parzen_window_kernel(x: torch.Tensor, win_width: float = 1.0) -> torch.Tensor:
    """Implementation of the 2nd-order polynomial kernel (Xu et al., 2008).

    Range: [-1, 1]. Returns 0 outside this range.

    Ref: "Parzen-Window Based Normalized Mutual Information for Medical Image Registration", Eq. 22.
    """
    x = torch.abs(x) / win_width

    kernel_val = torch.zeros_like(x)

    mask1 = x < 0.5
    kernel_val[mask1] = -1.8 * (x[mask1] ** 2) - 0.1 * x[mask1] + 1.0
    mask2 = (x >= 0.5) & (x <= 1.0)
    kernel_val[mask2] = 1.8 * (x[mask2] ** 2) - 3.7 * x[mask2] + 1.9

    return kernel_val


def _normalize_signal(data: torch.Tensor, num_bins: int):
    min_val, _ = data.min(axis=-1)
    max_val, _ = data.max(axis=-1)
    return (data - min_val.unsqueeze(-1)) / (max_val - min_val).unsqueeze(-1) * num_bins


def compute_joint_histogram(
    img_1: torch.Tensor, img_2: torch.Tensor, num_bins: int = 64, sigma: float = 1.0
) -> torch.Tensor:
    """Computes the differentiable Joint Histogram using Parzen Window estimation.

    Input shapes: (B, C, H, W) or (N,)
    Output shape: (num_bins, num_bins)
    """
    i1_flat = img_1.reshape(-1)
    i2_flat = img_2.reshape(-1)

    min_val = min(i1_flat.min(), i2_flat.min()).detach()
    max_val = max(i1_flat.max(), i2_flat.max()).detach()

    # Add a small epsilon to max to ensure coverage
    bin_centers = torch.linspace(min_val, max_val, num_bins, device=img_1.device)
    bin_width = (max_val - min_val) / (num_bins - 1)

    d1 = i1_flat.unsqueeze(1) - bin_centers.unsqueeze(0)
    d2 = i2_flat.unsqueeze(1) - bin_centers.unsqueeze(0)

    window_size = bin_width * sigma

    w1 = parzen_window_kernel(d1, win_width=window_size)
    w2 = parzen_window_kernel(d2, win_width=window_size)

    H_joint = torch.einsum("ni,nj->ij", w1, w2)

    P_joint = H_joint / (H_joint.sum() + 1e-6)
    P_joint = H_joint / (H_joint.sum() + 1e-6)

    return P_joint


def mutual_information_loss(
    input: torch.Tensor, target: torch.Tensor, num_bins: int = 64, sigma: float = 2.0
) -> torch.Tensor:
    """Calculates the Negative Normalized Mutual Information Loss.

    loss = - (H(X) + H(Y)) / H(X,Y)
    """
    if input.shape != target.shape:
        raise ValueError(f"Shape mismatch: {input.shape} != {target.shape}")

    P_xy = compute_joint_histogram(input, target, num_bins, sigma)

    P_x = P_xy.sum(dim=1)
    P_y = P_xy.sum(dim=0)

    eps = 1e-8
    H_x = -torch.sum(P_x * torch.log(P_x + eps))
    H_y = -torch.sum(P_y * torch.log(P_y + eps))
    H_xy = -torch.sum(P_xy * torch.log(P_xy + eps))

    nmi = (H_x + H_y) / (H_xy + eps)

    return -nmi
